#!/usr/bin/env python3
"""
Example: CUDA-Accelerated Multi-Camera Voxel Processing

This script demonstrates how to use the CUDA voxel processing module
for real-time multi-camera voxel grid reconstruction.

Features demonstrated:
1. GPU device info and capability checking
2. Multi-camera setup with CUDA streams
3. Motion detection on GPU
4. Ray-casting with voxel accumulation
5. Post-processing (Gaussian blur)
6. Performance benchmarking

Usage:
    python example_cuda_usage.py [--num-cameras 10] [--8k]
"""

import numpy as np
import argparse
import time
from pathlib import Path

try:
    import voxel_cuda
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"CUDA module not available: {e}")
    print("Please compile the CUDA extension first:")
    print("  python setup.py build_ext --inplace")
    CUDA_AVAILABLE = False
    exit(1)


def check_gpu_requirements():
    """Check if GPU meets requirements for 8K processing"""
    print("=" * 70)
    print("GPU Requirements Check")
    print("=" * 70)

    voxel_cuda.print_device_info()

    # Check compute capability (RTX 3090: 8.6, RTX 4090: 8.9)
    if voxel_cuda.check_compute_capability(8, 6):
        print("✓ GPU supports RTX 3090/4090 features (Compute 8.6+)")
        return True
    elif voxel_cuda.check_compute_capability(7, 5):
        print("⚠ GPU is older generation (Compute 7.5+)")
        print("  Performance may be reduced, but should work")
        return True
    else:
        print("✗ GPU too old (Compute < 7.5)")
        print("  Please use RTX 2080 or newer")
        return False


def create_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """Create rotation matrix from Euler angles (matching C++ code)"""
    import math

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # Rz(yaw)
    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Ry(roll)
    Ry = np.array([
        [cr, 0, sr],
        [0, 1, 0],
        [-sr, 0, cr]
    ], dtype=np.float32)

    # Rx(pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp]
    ], dtype=np.float32)

    # Combined: Rz * Ry * Rx
    return (Rz @ Ry @ Rx).astype(np.float32)


def setup_circular_camera_array(num_cameras, radius=1000.0, height=0.0, fov_deg=60.0):
    """
    Setup cameras in a circular array pointing toward center

    Args:
        num_cameras: Number of cameras
        radius: Distance from center
        height: Height above ground plane
        fov_deg: Field of view in degrees

    Returns:
        List of (position, rotation_matrix, fov_rad) tuples
    """
    cameras = []

    for i in range(num_cameras):
        # Angle around circle
        angle = 2.0 * np.pi * i / num_cameras

        # Position on circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height

        position = np.array([x, y, z], dtype=np.float32)

        # Rotation to point toward center
        # Yaw points camera toward origin
        yaw_deg = np.degrees(angle) + 180.0  # Point inward
        pitch_deg = 0.0
        roll_deg = 0.0

        rotation = create_rotation_matrix(yaw_deg, pitch_deg, roll_deg)

        fov_rad = np.radians(fov_deg)

        cameras.append((position, rotation, fov_rad))

    return cameras


def generate_synthetic_frames(num_cameras, width, height, frame_idx, add_motion=True):
    """
    Generate synthetic test frames with optional motion

    Args:
        num_cameras: Number of cameras
        width: Frame width
        height: Frame height
        frame_idx: Current frame index
        add_motion: Whether to add synthetic motion

    Returns:
        frames: numpy array (num_cameras, height, width)
    """
    frames = np.zeros((num_cameras, height, width), dtype=np.float32)

    for cam_id in range(num_cameras):
        # Base pattern (gradient + noise)
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)

        # Radial gradient
        frame = (X + Y) / 2.0 * 255.0

        # Add noise
        frame += np.random.randn(height, width) * 5.0

        if add_motion:
            # Add moving bright spot
            spot_x = int(width * 0.3 + (frame_idx % 100) * width * 0.004)
            spot_y = int(height * 0.5)
            spot_radius = 50

            y_coords, x_coords = np.ogrid[:height, :width]
            dist = np.sqrt((x_coords - spot_x)**2 + (y_coords - spot_y)**2)
            spot = np.exp(-(dist**2) / (2 * spot_radius**2)) * 200.0
            frame += spot

        # Clamp to valid range
        frame = np.clip(frame, 0, 255).astype(np.float32)
        frames[cam_id] = frame

    return frames


def main():
    parser = argparse.ArgumentParser(description='CUDA Voxel Processing Example')
    parser.add_argument('--num-cameras', type=int, default=5,
                        help='Number of cameras (default: 5)')
    parser.add_argument('--8k', action='store_true',
                        help='Use 8K resolution (7680x4320), default is 1080p')
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of frames to process (default: 10)')
    parser.add_argument('--voxel-size', type=int, default=500,
                        help='Voxel grid size (NxNxN, default: 500)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    parser.add_argument('--save-output', action='store_true',
                        help='Save voxel grid to file')

    args = parser.parse_args()

    # Check GPU
    if not check_gpu_requirements():
        print("\nGPU requirements not met!")
        return

    print("\n" + "=" * 70)
    print("Configuration")
    print("=" * 70)

    # Resolution
    if args.__dict__['8k']:
        width, height = 7680, 4320
        print("Resolution: 8K (7680x4320)")
        voxel_cuda.optimize_for_8k()
    else:
        width, height = 1920, 1080
        print("Resolution: 1080p (1920x1080)")

    print(f"Number of cameras: {args.num_cameras}")
    print(f"Voxel grid size: {args.voxel_size}³")
    print(f"Frames to process: {args.frames}")

    # Create voxel grid on GPU
    print("\n" + "=" * 70)
    print("Initializing Voxel Grid")
    print("=" * 70)

    grid_center = np.array([0.0, 0.0, 500.0], dtype=np.float32)
    voxel_grid = voxel_cuda.VoxelGridGPU(
        N=args.voxel_size,
        voxel_size=6.0,
        grid_center=grid_center
    )

    print(f"Created: {voxel_grid}")
    print(f"Memory: ~{(args.voxel_size**3 * 4) / (1024**2):.1f} MB")

    # Setup camera manager
    print("\n" + "=" * 70)
    print("Initializing Camera Streams")
    print("=" * 70)

    camera_mgr = voxel_cuda.CameraStreamManager(num_cameras=args.num_cameras)
    print(f"Created: {camera_mgr}")

    # Configure cameras in circular array
    camera_configs = setup_circular_camera_array(
        num_cameras=args.num_cameras,
        radius=1000.0,
        height=0.0,
        fov_deg=60.0
    )

    for cam_id, (position, rotation, fov_rad) in enumerate(camera_configs):
        camera_mgr.set_camera(
            cam_id=cam_id,
            position=position,
            rotation_matrix=rotation.flatten(),
            fov_rad=fov_rad,
            width=width,
            height=height
        )
        print(f"Camera {cam_id}: pos={position}, fov={np.degrees(fov_rad):.1f}°")

    # Process frames
    print("\n" + "=" * 70)
    print("Processing Frames")
    print("=" * 70)

    prev_frames = None
    total_time = 0.0

    for frame_idx in range(args.frames):
        # Generate synthetic frames
        curr_frames = generate_synthetic_frames(
            args.num_cameras, width, height, frame_idx, add_motion=True
        )

        if prev_frames is None:
            prev_frames = curr_frames.copy()
            continue

        # Process on GPU
        start_time = time.time()

        camera_mgr.process_frames(
            prev_frames=prev_frames,
            curr_frames=curr_frames,
            voxel_grid=voxel_grid,
            motion_threshold=2.0
        )

        elapsed = time.time() - start_time
        total_time += elapsed

        # Calculate metrics
        fps = 1.0 / elapsed if elapsed > 0 else 0
        megapixels = (width * height * args.num_cameras) / 1e6
        throughput = megapixels / elapsed if elapsed > 0 else 0

        print(f"Frame {frame_idx:3d}: {elapsed*1000:6.2f} ms "
              f"({fps:5.1f} FPS, {throughput:6.1f} MP/s)")

        prev_frames = curr_frames.copy()

    # Statistics
    avg_time = total_time / (args.frames - 1) if args.frames > 1 else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    avg_throughput = (width * height * args.num_cameras) / 1e6 / avg_time if avg_time > 0 else 0

    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"Average time per frame: {avg_time*1000:.2f} ms")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average throughput: {avg_throughput:.1f} MP/s")
    print(f"Total processing time: {total_time:.2f} s")

    # Get results from GPU
    print("\n" + "=" * 70)
    print("Retrieving Results")
    print("=" * 70)

    start_time = time.time()
    voxel_data = voxel_grid.to_host()
    copy_time = time.time() - start_time

    print(f"Copy to host: {copy_time*1000:.2f} ms")
    print(f"Voxel grid shape: {voxel_data.shape}")
    print(f"Voxel grid dtype: {voxel_data.dtype}")
    print(f"Min value: {voxel_data.min():.2f}")
    print(f"Max value: {voxel_data.max():.2f}")
    print(f"Mean value: {voxel_data.mean():.2f}")
    print(f"Non-zero voxels: {np.count_nonzero(voxel_data)} "
          f"({100 * np.count_nonzero(voxel_data) / voxel_data.size:.3f}%)")

    # Optional: Apply Gaussian blur
    if voxel_data.max() > 0:
        print("\n" + "=" * 70)
        print("Applying 3D Gaussian Blur")
        print("=" * 70)

        start_time = time.time()
        blurred = voxel_cuda.apply_gaussian_blur(voxel_data, sigma=1.5)
        blur_time = time.time() - start_time

        print(f"Blur time: {blur_time*1000:.2f} ms")
        print(f"Blurred max value: {blurred.max():.2f}")

    # Save output
    if args.save_output:
        print("\n" + "=" * 70)
        print("Saving Output")
        print("=" * 70)

        output_dir = Path("output_cuda")
        output_dir.mkdir(exist_ok=True)

        # Save raw voxel grid
        raw_path = output_dir / "voxel_grid_raw.npy"
        np.save(raw_path, voxel_data)
        print(f"Saved raw voxel grid to: {raw_path}")

        # Save blurred voxel grid
        if voxel_data.max() > 0:
            blurred_path = output_dir / "voxel_grid_blurred.npy"
            np.save(blurred_path, blurred)
            print(f"Saved blurred voxel grid to: {blurred_path}")

        # Save as binary (matching C++ output format)
        bin_path = output_dir / "voxel_grid.bin"
        with open(bin_path, 'wb') as f:
            # Write metadata
            f.write(np.array([args.voxel_size], dtype=np.int32).tobytes())
            f.write(np.array([6.0], dtype=np.float32).tobytes())  # voxel_size
            # Write data
            f.write(voxel_data.astype(np.float32).tobytes())
        print(f"Saved binary voxel grid to: {bin_path}")

    # Run benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 70)
        print("Running Benchmark")
        print("=" * 70)

        voxel_cuda.benchmark(
            width=width,
            height=height,
            num_cameras=args.num_cameras,
            voxel_size=args.voxel_size,
            iterations=100
        )

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
