#!/usr/bin/env python3
"""
Demonstration and testing script for thermal + monochrome fusion system

Shows:
- Real-time fusion at 30+ FPS
- Image registration accuracy
- Target confirmation and false positive reduction
- Occlusion handling
- Performance metrics
"""

import numpy as np
import time
import logging
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse

# Import fusion system
try:
    from fusion_manager import FusionManager, FusionConfig
    from detection_fusion import (
        DetectionFusion, FusionWeights,
        MotionDetection, ThermalDetection
    )
    import thermal_mono_fusion as cpp_fusion
    FUSION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import fusion modules: {e}")
    FUSION_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionSimulator:
    """Simulates thermal + mono camera system for testing"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (480, 640),
        num_targets: int = 5
    ):
        self.image_size = image_size
        self.num_targets = num_targets
        self.frame_count = 0

        # Generate random targets with motion
        self.targets = []
        for i in range(num_targets):
            self.targets.append({
                'x': np.random.uniform(100, image_size[1] - 100),
                'y': np.random.uniform(100, image_size[0] - 100),
                'vx': np.random.uniform(-5, 5),
                'vy': np.random.uniform(-5, 5),
                'thermal_sig': np.random.uniform(0.5, 1.0),
                'size': np.random.uniform(20, 50),
                'occluded': False,
                'occlusion_countdown': 0
            })

    def generate_frame_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic thermal and monochrome images"""
        thermal = np.random.normal(0.1, 0.02, self.image_size)
        mono = np.random.normal(0.15, 0.03, self.image_size)

        # Update target positions
        for target in self.targets:
            # Update position
            target['x'] += target['vx']
            target['y'] += target['vy']

            # Bounce off edges
            if target['x'] < 0 or target['x'] >= self.image_size[1]:
                target['vx'] *= -1
            if target['y'] < 0 or target['y'] >= self.image_size[0]:
                target['vy'] *= -1

            # Random occlusion
            if target['occlusion_countdown'] > 0:
                target['occlusion_countdown'] -= 1
                target['occluded'] = True
            elif np.random.random() < 0.01:  # 1% chance per frame
                target['occluded'] = True
                target['occlusion_countdown'] = np.random.randint(10, 30)
            else:
                target['occluded'] = False

            if not target['occluded']:
                # Add target to images
                size = int(target['size'])
                x, y = int(target['x']), int(target['y'])

                # Thermal signature
                y1, y2 = max(0, y - size//2), min(self.image_size[0], y + size//2)
                x1, x2 = max(0, x - size//2), min(self.image_size[1], x + size//2)
                thermal[y1:y2, x1:x2] += target['thermal_sig']

                # Monochrome (slightly offset for registration challenge)
                offset = 2
                y1, y2 = max(0, y - size//2 + offset), min(self.image_size[0], y + size//2 + offset)
                x1, x2 = max(0, x - size//2 + offset), min(self.image_size[1], x + size//2 + offset)
                mono[y1:y2, x1:x2] += target['thermal_sig'] * 0.8

        # Add false positives (noise spikes)
        for _ in range(np.random.randint(2, 5)):
            x = np.random.randint(0, self.image_size[1])
            y = np.random.randint(0, self.image_size[0])
            # Only add to one modality (will be rejected by fusion)
            if np.random.random() < 0.5:
                thermal[y:y+10, x:x+10] += 0.4
            else:
                mono[y:y+10, x:x+10] += 0.4

        self.frame_count += 1

        return thermal, mono

    def get_ground_truth(self) -> List[dict]:
        """Get ground truth target positions"""
        return [
            {
                'x': t['x'],
                'y': t['y'],
                'vx': t['vx'],
                'vy': t['vy'],
                'thermal_sig': t['thermal_sig'],
                'occluded': t['occluded']
            }
            for t in self.targets
        ]


def test_cpp_fusion():
    """Test C++ fusion module directly"""
    logger.info("=" * 60)
    logger.info("Testing C++ Fusion Module")
    logger.info("=" * 60)

    if not FUSION_AVAILABLE:
        logger.error("Fusion modules not available")
        return

    # Create synthetic images
    thermal = np.random.rand(480, 640).astype(np.float64)
    mono = np.random.rand(480, 640).astype(np.float64)

    # Add some targets
    thermal[200:250, 300:350] += 0.5
    mono[202:252, 302:352] += 0.5

    logger.info("Running image registration...")
    start = time.time()
    reg_params = cpp_fusion.estimate_homography(thermal, mono)
    reg_time = time.time() - start

    logger.info(f"Registration completed in {reg_time*1000:.2f} ms")
    logger.info(f"  Valid: {reg_params.is_valid}")
    logger.info(f"  RMSE: {reg_params.rmse:.4f} pixels")
    logger.info(f"  Matches: {reg_params.num_matches}")

    logger.info("\nRunning fusion processing...")
    start = time.time()
    result = cpp_fusion.process_fusion(thermal, mono, camera_baseline_m=0.5, range_km=5.0)
    fusion_time = time.time() - start

    logger.info(f"Fusion completed in {fusion_time*1000:.2f} ms")
    logger.info(f"  FPS: {result.fps:.1f}")
    logger.info(f"  Detections: {len(result.detections)}")
    logger.info(f"  Registration error: {result.registration_error_mm:.2f} mm at 5km")
    logger.info(f"  False positives rejected: {result.false_positives_rejected}")
    logger.info(f"  Target confirmation rate: {result.target_confirmation_rate*100:.1f}%")

    # Test thermal palette
    logger.info("\nTesting thermal palettes...")
    for palette in ['iron', 'rainbow', 'whitehot', 'blackhot', 'grayscale']:
        colored = cpp_fusion.apply_thermal_palette(thermal, palette)
        logger.info(f"  {palette}: {colored.shape} RGB image")


def test_fusion_manager():
    """Test high-level fusion manager"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Fusion Manager")
    logger.info("=" * 60)

    if not FUSION_AVAILABLE:
        logger.error("Fusion modules not available")
        return

    # Create fusion manager
    config = FusionConfig(
        target_fps=30,
        thermal_threshold=0.3,
        mono_threshold=0.2,
        confidence_threshold=0.6,
        enable_cuda=True,
        thermal_palette='iron'
    )

    manager = FusionManager(config)

    # Add camera pairs
    pair_id_0 = manager.add_camera_pair(thermal_id=0, mono_id=0, baseline_m=0.5)
    pair_id_1 = manager.add_camera_pair(thermal_id=1, mono_id=1, baseline_m=0.5)

    logger.info(f"Added {len(manager.camera_pairs)} camera pairs")

    # Start processing
    manager.start(num_workers=4)

    # Simulate processing frames
    simulator = FusionSimulator(num_targets=5)

    num_frames = 100
    logger.info(f"\nProcessing {num_frames} frame pairs...")

    start_time = time.time()

    for frame_id in range(num_frames):
        thermal, mono = simulator.generate_frame_pair()
        timestamp = time.time()

        # Process frame pair
        detections = manager.process_frame_pair(
            pair_id=pair_id_0,
            thermal_image=thermal,
            mono_image=mono,
            timestamp=timestamp
        )

        if frame_id % 20 == 0 and detections:
            logger.info(f"Frame {frame_id}: {len(detections)} detections")

    total_time = time.time() - start_time

    # Get performance metrics
    metrics = manager.get_performance_metrics()

    logger.info("\nPerformance Metrics:")
    logger.info(f"  Average FPS: {metrics.get('avg_fps', 0):.1f}")
    logger.info(f"  Total frames: {metrics.get('frame_count', 0)}")
    logger.info(f"  Avg registration error: {metrics.get('avg_registration_error_mm', 0):.2f} mm")
    logger.info(f"  Target confirmation rate: {metrics.get('target_confirmation_rate', 0)*100:.1f}%")
    logger.info(f"  False positive reduction: {metrics.get('false_positive_reduction_rate', 0)*100:.1f}%")
    logger.info(f"  Active tracks: {metrics.get('active_tracks', 0)}")
    logger.info(f"  Total time: {total_time:.2f} s")

    manager.stop()


def test_detection_fusion():
    """Test detection-level fusion"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Detection Fusion")
    logger.info("=" * 60)

    # Create detection fusion system
    weights = FusionWeights(
        motion_weight=0.5,
        thermal_weight=0.5,
        signature_weight=0.3,
        spatial_weight=0.7
    )

    fusion = DetectionFusion(
        fusion_weights=weights,
        iou_threshold=0.3,
        confidence_threshold=0.6
    )

    # Simulate detections
    num_frames = 100
    logger.info(f"Processing {num_frames} frames with detection fusion...")

    for frame_id in range(num_frames):
        # Generate random motion detections
        motion_dets = []
        for _ in range(np.random.randint(3, 7)):
            motion_dets.append(MotionDetection(
                x=np.random.uniform(100, 540),
                y=np.random.uniform(100, 380),
                width=np.random.uniform(20, 50),
                height=np.random.uniform(20, 50),
                velocity_x=np.random.uniform(-5, 5),
                velocity_y=np.random.uniform(-5, 5),
                motion_confidence=np.random.uniform(0.5, 1.0),
                frame_id=frame_id,
                timestamp=time.time()
            ))

        # Generate random thermal detections (with some overlap)
        thermal_dets = []
        for motion_det in motion_dets[:3]:  # Match some motion detections
            thermal_dets.append(ThermalDetection(
                x=motion_det.x + np.random.uniform(-5, 5),  # Slight offset
                y=motion_det.y + np.random.uniform(-5, 5),
                width=motion_det.width + np.random.uniform(-5, 5),
                height=motion_det.height + np.random.uniform(-5, 5),
                temperature_kelvin=300 + np.random.uniform(0, 20),
                thermal_confidence=np.random.uniform(0.6, 1.0),
                signature_strength=np.random.uniform(0.5, 1.0),
                frame_id=frame_id,
                timestamp=time.time()
            ))

        # Add some unmatched thermal detections
        for _ in range(np.random.randint(1, 3)):
            thermal_dets.append(ThermalDetection(
                x=np.random.uniform(100, 540),
                y=np.random.uniform(100, 380),
                width=np.random.uniform(20, 50),
                height=np.random.uniform(20, 50),
                temperature_kelvin=300 + np.random.uniform(0, 20),
                thermal_confidence=np.random.uniform(0.6, 1.0),
                signature_strength=np.random.uniform(0.5, 1.0),
                frame_id=frame_id,
                timestamp=time.time()
            ))

        # Fuse detections
        fused_dets = fusion.fuse_detections(
            motion_dets, thermal_dets, frame_id, time.time()
        )

        if frame_id % 20 == 0:
            logger.info(f"Frame {frame_id}: {len(motion_dets)} motion, "
                       f"{len(thermal_dets)} thermal -> {len(fused_dets)} fused")

    # Get metrics
    metrics = fusion.get_performance_metrics()

    logger.info("\nDetection Fusion Metrics:")
    logger.info(f"  Total detections: {metrics['total_detections']}")
    logger.info(f"  Cross-validation rate: {metrics['cross_validation_rate']*100:.1f}%")
    logger.info(f"  False positive reduction: {metrics['false_positive_reduction_rate']*100:.1f}%")
    logger.info(f"  Active tracks: {metrics['active_tracks']}")
    logger.info(f"  Occluded tracks: {metrics['occluded_tracks']}")
    logger.info(f"  Occlusions handled: {metrics['occlusions_handled']}")


def benchmark_performance():
    """Benchmark fusion system performance"""
    logger.info("\n" + "=" * 60)
    logger.info("Performance Benchmark")
    logger.info("=" * 60)

    if not FUSION_AVAILABLE:
        logger.error("Fusion modules not available")
        return

    image_sizes = [(240, 320), (480, 640), (720, 1280), (1080, 1920)]

    for size in image_sizes:
        thermal = np.random.rand(*size).astype(np.float64)
        mono = np.random.rand(*size).astype(np.float64)

        # Add some targets
        thermal[size[0]//4:size[0]//4+50, size[1]//4:size[1]//4+50] += 0.5
        mono[size[0]//4+2:size[0]//4+52, size[1]//4+2:size[1]//4+52] += 0.5

        # Benchmark
        num_runs = 10
        times = []

        for _ in range(num_runs):
            start = time.time()
            result = cpp_fusion.process_fusion(thermal, mono)
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        avg_fps = 1000.0 / avg_time

        logger.info(f"\nImage size: {size[1]}x{size[0]}")
        logger.info(f"  Avg time: {avg_time:.2f} ms")
        logger.info(f"  Avg FPS: {avg_fps:.1f}")
        logger.info(f"  Detections: {len(result.detections)}")
        logger.info(f"  Registration error: {result.registration_error_mm:.2f} mm")


def main():
    parser = argparse.ArgumentParser(
        description='Thermal + Monochrome Fusion System Demo'
    )
    parser.add_argument(
        '--test',
        choices=['cpp', 'manager', 'detection', 'benchmark', 'all'],
        default='all',
        help='Which test to run'
    )

    args = parser.parse_args()

    logger.info("Thermal + Monochrome Camera Fusion System")
    logger.info("=" * 60)

    if args.test in ['cpp', 'all']:
        test_cpp_fusion()

    if args.test in ['manager', 'all']:
        test_fusion_manager()

    if args.test in ['detection', 'all']:
        test_detection_fusion()

    if args.test in ['benchmark', 'all']:
        benchmark_performance()

    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
