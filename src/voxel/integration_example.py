"""
integration_example.py

Complete integration example demonstrating the 5km+ range tracking system
using all three components:
1. C++ Sparse Voxel Grid (via ctypes)
2. CUDA Voxel Optimizer (via ctypes)
3. Python Grid Manager

This example simulates a realistic tracking scenario:
- 200 moving objects across 5km range
- 10 cameras tracking the objects
- Real-time updates at 30Hz
- Dynamic LOD based on distance
"""

import numpy as np
import time
import sys
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import grid manager
from grid_manager import VoxelGridManager, GridConfig, LODLevel, Camera, TrackedObject


class TrackingSimulator:
    """Simulates a multi-object tracking scenario"""

    def __init__(self, num_objects: int = 200, num_cameras: int = 10):
        self.num_objects = num_objects
        self.num_cameras = num_cameras

        # Initialize grid manager
        print("Initializing Voxel Grid Manager...")
        config = GridConfig(
            center=np.array([0.0, 0.0, 500.0]),
            size=np.array([5000.0, 5000.0, 2000.0]),
            base_resolution=1.0,
            max_memory_mb=500,
            enable_lod=True,
            enable_compression=True,
            prune_threshold=0.01,
            update_rate_hz=30.0
        )
        self.grid_manager = VoxelGridManager(config)

        # Generate objects
        self.objects = self._generate_objects()

        # Generate cameras
        self.cameras = self._generate_cameras()

        # Performance metrics
        self.frame_times = []
        self.memory_usage = []

    def _generate_objects(self) -> List[dict]:
        """Generate moving objects in the scene"""
        objects = []

        print(f"\nGenerating {self.num_objects} tracked objects...")

        for i in range(self.num_objects):
            # Distribute objects across different distance ranges
            if i < 50:
                # Close range: <500m
                distance = np.random.uniform(50, 500)
                height = np.random.uniform(0, 200)
            elif i < 150:
                # Medium range: 500-2500m
                distance = np.random.uniform(500, 2500)
                height = np.random.uniform(0, 800)
            else:
                # Long range: 2500-5000m
                distance = np.random.uniform(2500, 5000)
                height = np.random.uniform(0, 1500)

            angle = np.random.uniform(0, 2 * np.pi)

            position = np.array([
                np.cos(angle) * distance,
                np.sin(angle) * distance,
                height
            ])

            # Random velocity (0-50 m/s)
            velocity = np.random.uniform(-10, 10, size=3)
            velocity[2] *= 0.5  # Slower vertical motion

            # Object size varies with distance (perspective)
            size = np.random.uniform(1.0, 5.0)

            obj = {
                'id': i,
                'position': position,
                'velocity': velocity,
                'size': size,
                'type': 'aircraft' if height > 500 else 'ground'
            }

            objects.append(obj)

            # Add to grid manager
            self.grid_manager.add_tracked_object(
                object_id=i,
                position=position,
                size=size,
                velocity=velocity
            )

        print(f"  Objects by range:")
        print(f"    <500m:      {sum(1 for o in objects if np.linalg.norm(o['position'][:2]) < 500)}")
        print(f"    500-2500m:  {sum(1 for o in objects if 500 <= np.linalg.norm(o['position'][:2]) < 2500)}")
        print(f"    2500-5000m: {sum(1 for o in objects if np.linalg.norm(o['position'][:2]) >= 2500)}")

        return objects

    def _generate_cameras(self) -> List[dict]:
        """Generate camera positions and orientations"""
        cameras = []

        print(f"\nGenerating {self.num_cameras} cameras...")

        for i in range(self.num_cameras):
            angle = (i / self.num_cameras) * 2 * np.pi

            # Cameras arranged in a circle at 100m distance
            position = np.array([
                np.cos(angle) * 100.0,
                np.sin(angle) * 100.0,
                50.0
            ])

            # Point towards center with slight upward tilt
            forward = np.array([
                -np.cos(angle),
                -np.sin(angle),
                -0.3
            ])
            forward = forward / np.linalg.norm(forward)

            up = np.array([0.0, 0.0, 1.0])

            camera = {
                'id': i,
                'position': position,
                'forward': forward,
                'up': up,
                'fov': 60.0,
                'width': 1920,
                'height': 1080
            }

            cameras.append(camera)

            # Add to grid manager
            self.grid_manager.add_camera(
                camera_id=i,
                position=position,
                forward=forward,
                up=up,
                fov=60.0,
                width=1920,
                height=1080
            )

        return cameras

    def update_objects(self, dt: float):
        """Update object positions based on velocity"""
        for obj in self.objects:
            # Update position
            obj['position'] += obj['velocity'] * dt

            # Bounce off boundaries
            center = self.grid_manager.grid_center
            half_size = self.grid_manager.grid_size / 2

            for i in range(3):
                if obj['position'][i] < center[i] - half_size[i]:
                    obj['position'][i] = center[i] - half_size[i]
                    obj['velocity'][i] *= -1
                elif obj['position'][i] > center[i] + half_size[i]:
                    obj['position'][i] = center[i] + half_size[i]
                    obj['velocity'][i] *= -1

            # Update in grid manager
            self.grid_manager.update_object_position(
                object_id=obj['id'],
                position=obj['position']
            )

    def simulate_camera_observations(self):
        """Simulate camera observations and ray casting"""
        for camera in self.cameras:
            # For each camera, cast rays to visible objects
            # In a real system, this would be actual pixel data
            # Here we simulate by sampling some pixels

            num_rays = 100  # Sample 100 rays per camera
            pixel_coords = np.random.randint(
                0, [camera['width'], camera['height']],
                size=(num_rays, 2)
            )

            # Simulate intensity values (would come from actual images)
            values = np.random.uniform(0.5, 1.0, size=num_rays)

            # Cast rays (this updates voxels)
            self.grid_manager.cast_rays_from_camera(
                camera_id=camera['id'],
                pixel_coords=pixel_coords,
                values=values
            )

    def run_simulation(self, num_frames: int = 90, target_fps: float = 30.0):
        """Run the tracking simulation"""
        print(f"\n{'='*60}")
        print(f"Running Simulation: {num_frames} frames @ {target_fps} Hz")
        print(f"{'='*60}\n")

        # Start background processing
        self.grid_manager.start_background_processing()

        dt = 1.0 / target_fps

        try:
            for frame in range(num_frames):
                frame_start = time.time()

                # Update object positions
                self.update_objects(dt)

                # Simulate camera observations
                self.simulate_camera_observations()

                # Auto-adjust grid every 30 frames
                if frame % 30 == 0 and frame > 0:
                    self.grid_manager.auto_adjust_grid()

                # Record metrics
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                self.memory_usage.append(self.grid_manager.memory_usage_mb)

                # Print progress
                if frame % 10 == 0 or frame == num_frames - 1:
                    stats = self.grid_manager.get_statistics()
                    print(f"Frame {frame:3d}/{num_frames}: "
                          f"{frame_time*1000:5.1f}ms | "
                          f"{1.0/frame_time:5.1f} FPS | "
                          f"Memory: {stats['memory_usage_mb']:6.1f} MB | "
                          f"Objects: {stats['tracked_objects']}")

                # Sleep to maintain target frame rate (if ahead)
                elapsed = time.time() - frame_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")

        finally:
            # Stop background processing
            self.grid_manager.stop_background_processing()

        self._print_results()

    def _print_results(self):
        """Print simulation results and statistics"""
        print(f"\n{'='*60}")
        print("Simulation Results")
        print(f"{'='*60}\n")

        # Timing statistics
        frame_times_ms = np.array(self.frame_times) * 1000
        print("Performance Metrics:")
        print(f"  Average frame time: {np.mean(frame_times_ms):.2f} ms")
        print(f"  Min frame time:     {np.min(frame_times_ms):.2f} ms")
        print(f"  Max frame time:     {np.max(frame_times_ms):.2f} ms")
        print(f"  Std deviation:      {np.std(frame_times_ms):.2f} ms")
        print(f"  Average FPS:        {1000.0/np.mean(frame_times_ms):.1f}")

        # Memory statistics
        print(f"\nMemory Usage:")
        print(f"  Average: {np.mean(self.memory_usage):.2f} MB")
        print(f"  Peak:    {np.max(self.memory_usage):.2f} MB")
        print(f"  Min:     {np.min(self.memory_usage):.2f} MB")

        # Requirements check
        print(f"\nRequirements Check:")
        avg_fps = 1000.0 / np.mean(frame_times_ms)
        peak_mem = np.max(self.memory_usage)

        print(f"  {'✓' if avg_fps >= 30.0 else '✗'} Update rate: {avg_fps:.1f} Hz (target: ≥30 Hz)")
        print(f"  {'✓' if peak_mem < 500.0 else '✗'} Memory usage: {peak_mem:.1f} MB (target: <500 MB)")
        print(f"  {'✓' if self.num_objects >= 200 else '✗'} Object count: {self.num_objects} (target: ≥200)")

        # Grid statistics
        print(f"\nGrid Statistics:")
        self.grid_manager.print_statistics()

        # Resolution check at different distances
        print(f"\nResolution at different distances:")
        for distance in [100, 500, 1000, 2500, 5000]:
            lod = self.grid_manager._get_lod_for_distance(distance)
            print(f"  {distance:4d}m: {lod.resolution:.2f}m (LOD: {lod.name})")

    def visualize_scene(self):
        """Visualize the tracking scene"""
        print("\nGenerating visualization...")

        fig = plt.figure(figsize=(15, 10))

        # 3D scatter plot of objects
        ax1 = fig.add_subplot(221, projection='3d')
        positions = np.array([obj['position'] for obj in self.objects])

        # Color by distance
        distances = np.linalg.norm(positions[:, :2], axis=1)
        scatter = ax1.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=distances,
            cmap='viridis',
            s=20,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax1, label='Distance (m)')

        # Add cameras
        cam_positions = np.array([cam['position'] for cam in self.cameras])
        ax1.scatter(
            cam_positions[:, 0],
            cam_positions[:, 1],
            cam_positions[:, 2],
            c='red',
            marker='^',
            s=100,
            label='Cameras'
        )

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Scene Layout')
        ax1.legend()

        # Frame time plot
        ax2 = fig.add_subplot(222)
        ax2.plot(np.array(self.frame_times) * 1000, 'b-', alpha=0.7)
        ax2.axhline(y=33.33, color='r', linestyle='--', label='30 Hz target')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Frame Time (ms)')
        ax2.set_title('Frame Time Over Simulation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Memory usage plot
        ax3 = fig.add_subplot(223)
        ax3.plot(self.memory_usage, 'g-', alpha=0.7)
        ax3.axhline(y=500, color='r', linestyle='--', label='500 MB limit')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('Memory Usage Over Simulation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Distance distribution
        ax4 = fig.add_subplot(224)
        ax4.hist(distances, bins=50, alpha=0.7, color='purple')
        ax4.axvline(x=100, color='b', linestyle='--', alpha=0.5, label='LOD boundaries')
        ax4.axvline(x=500, color='b', linestyle='--', alpha=0.5)
        ax4.axvline(x=1000, color='b', linestyle='--', alpha=0.5)
        ax4.axvline(x=5000, color='r', linestyle='--', alpha=0.5, label='Max range')
        ax4.set_xlabel('Distance from Center (m)')
        ax4.set_ylabel('Object Count')
        ax4.set_title('Object Distance Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('tracking_simulation.png', dpi=150)
        print("Visualization saved to: tracking_simulation.png")
        plt.show()


def main():
    """Main entry point"""
    print("="*60)
    print("5km+ Range Tracking System - Integration Example")
    print("="*60)

    # Parse command line arguments
    num_objects = 200
    num_cameras = 10
    num_frames = 90

    if len(sys.argv) > 1:
        num_objects = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_cameras = int(sys.argv[2])
    if len(sys.argv) > 3:
        num_frames = int(sys.argv[3])

    print(f"\nConfiguration:")
    print(f"  Objects:  {num_objects}")
    print(f"  Cameras:  {num_cameras}")
    print(f"  Frames:   {num_frames}")
    print()

    # Create and run simulator
    simulator = TrackingSimulator(
        num_objects=num_objects,
        num_cameras=num_cameras
    )

    # Run simulation
    simulator.run_simulation(num_frames=num_frames, target_fps=30.0)

    # Save final state
    simulator.grid_manager.save_state("tracking_state.json")
    print("\nFinal state saved to: tracking_state.json")

    # Visualize results
    try:
        simulator.visualize_scene()
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("(This is normal if matplotlib is not fully configured)")

    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
