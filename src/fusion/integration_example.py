#!/usr/bin/env python3
"""
Complete integration example for thermal + monochrome fusion system

Demonstrates:
- Multi-camera pair setup
- Real-time video stream processing
- Detection tracking and visualization
- Performance monitoring
- Thermal palette switching
- Low-light enhancement
"""

import numpy as np
import time
import logging
from typing import Optional, List
from dataclasses import dataclass
import threading
import queue

# Import fusion system
from fusion_manager import FusionManager, FusionConfig
from detection_fusion import (
    DetectionFusion, FusionWeights,
    MotionDetection, ThermalDetection
)

try:
    import thermal_mono_fusion as cpp_fusion
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: C++ fusion module not available")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CameraStream:
    """Simulates a camera video stream"""
    camera_id: int
    is_thermal: bool
    resolution: tuple
    fps: int
    running: bool = False


class MultiCameraFusionSystem:
    """
    Complete multi-camera fusion system with real-time processing
    """

    def __init__(self):
        # Configure fusion system
        self.config = FusionConfig(
            target_fps=30,
            registration_update_interval_s=1.0,
            thermal_threshold=0.3,
            mono_threshold=0.2,
            confidence_threshold=0.6,
            max_range_km=5.0,
            enable_thermal_enhancement=True,
            enable_false_positive_reduction=True,
            enable_cuda=CUDA_AVAILABLE,
            thermal_palette='iron'
        )

        # Create fusion manager
        self.fusion_manager = FusionManager(self.config)

        # Create detection fusion
        self.detection_fusion = DetectionFusion(
            fusion_weights=FusionWeights(
                motion_weight=0.5,
                thermal_weight=0.5,
                signature_weight=0.3,
                spatial_weight=0.7,
                temporal_weight=0.4
            ),
            iou_threshold=0.3,
            confidence_threshold=0.6
        )

        # Camera streams
        self.thermal_streams = {}
        self.mono_streams = {}
        self.camera_pairs = []

        # Processing state
        self.running = False
        self.frame_count = 0
        self.start_time = None

        # Performance tracking
        self.performance_history = []
        self.detection_history = []

        logger.info("MultiCameraFusionSystem initialized")

    def add_camera_pair(
        self,
        thermal_camera_id: int,
        mono_camera_id: int,
        baseline_m: float = 0.5,
        resolution: tuple = (480, 640)
    ) -> int:
        """Add a camera pair to the system"""

        # Create camera streams
        self.thermal_streams[thermal_camera_id] = CameraStream(
            camera_id=thermal_camera_id,
            is_thermal=True,
            resolution=resolution,
            fps=30
        )

        self.mono_streams[mono_camera_id] = CameraStream(
            camera_id=mono_camera_id,
            is_thermal=False,
            resolution=resolution,
            fps=30
        )

        # Add to fusion manager
        pair_id = self.fusion_manager.add_camera_pair(
            thermal_id=thermal_camera_id,
            mono_id=mono_camera_id,
            baseline_m=baseline_m,
            calibration_params={
                'resolution': resolution,
                'fov': 45.0
            }
        )

        self.camera_pairs.append({
            'pair_id': pair_id,
            'thermal_id': thermal_camera_id,
            'mono_id': mono_camera_id,
            'baseline_m': baseline_m
        })

        logger.info(f"Added camera pair {pair_id}: "
                   f"thermal={thermal_camera_id}, mono={mono_camera_id}")

        return pair_id

    def start(self, duration_seconds: Optional[int] = None):
        """Start the fusion system"""

        if self.running:
            logger.warning("System already running")
            return

        self.running = True
        self.start_time = time.time()
        self.frame_count = 0

        # Start fusion manager
        self.fusion_manager.start(num_workers=len(self.camera_pairs))

        logger.info("Starting fusion processing...")

        # Process frames
        try:
            self._processing_loop(duration_seconds)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the fusion system"""

        if not self.running:
            return

        self.running = False

        # Stop fusion manager
        self.fusion_manager.stop()

        # Print final statistics
        self._print_statistics()

        logger.info("Fusion system stopped")

    def _processing_loop(self, duration_seconds: Optional[int]):
        """Main processing loop"""

        target_frame_time = 1.0 / self.config.target_fps

        while self.running:
            loop_start = time.time()

            # Check duration limit
            if duration_seconds and (time.time() - self.start_time) > duration_seconds:
                logger.info(f"Reached duration limit of {duration_seconds}s")
                break

            # Process each camera pair
            for pair in self.camera_pairs:
                self._process_camera_pair(pair)

            self.frame_count += 1

            # Log progress
            if self.frame_count % 30 == 0:  # Every second at 30 FPS
                self._log_progress()

            # Frame rate limiting
            elapsed = time.time() - loop_start
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

    def _process_camera_pair(self, pair: dict):
        """Process one camera pair"""

        # Generate synthetic frames (in real system, capture from cameras)
        thermal_image = self._capture_thermal_frame(pair['thermal_id'])
        mono_image = self._capture_mono_frame(pair['mono_id'])

        timestamp = time.time()

        # Process through fusion manager
        detections = self.fusion_manager.process_frame_pair(
            pair_id=pair['pair_id'],
            thermal_image=thermal_image,
            mono_image=mono_image,
            timestamp=timestamp
        )

        if detections:
            # Store detection history
            self.detection_history.append({
                'timestamp': timestamp,
                'pair_id': pair['pair_id'],
                'detections': detections,
                'count': len(detections)
            })

            # Process detections further if needed
            self._handle_detections(detections, pair['pair_id'])

    def _capture_thermal_frame(self, camera_id: int) -> np.ndarray:
        """Capture frame from thermal camera (simulated)"""

        stream = self.thermal_streams[camera_id]
        h, w = stream.resolution

        # Generate synthetic thermal image with moving targets
        thermal = np.random.normal(0.1, 0.02, (h, w))

        # Add some moving thermal signatures
        num_targets = np.random.randint(2, 6)
        for _ in range(num_targets):
            x = int(np.random.uniform(50, w - 50))
            y = int(np.random.uniform(50, h - 50))
            size = np.random.randint(15, 40)

            y1, y2 = max(0, y - size//2), min(h, y + size//2)
            x1, x2 = max(0, x - size//2), min(w, x + size//2)

            thermal[y1:y2, x1:x2] += np.random.uniform(0.4, 0.8)

        return thermal

    def _capture_mono_frame(self, camera_id: int) -> np.ndarray:
        """Capture frame from mono camera (simulated)"""

        stream = self.mono_streams[camera_id]
        h, w = stream.resolution

        # Generate synthetic mono image
        # Simulate lower brightness for low-light testing
        base_brightness = np.random.uniform(0.05, 0.2)
        mono = np.random.normal(base_brightness, 0.03, (h, w))

        # Add targets (slightly offset from thermal)
        num_targets = np.random.randint(2, 6)
        for _ in range(num_targets):
            x = int(np.random.uniform(50, w - 50))
            y = int(np.random.uniform(50, h - 50))
            size = np.random.randint(15, 40)

            # Slight offset to test registration
            x += np.random.randint(-3, 4)
            y += np.random.randint(-3, 4)

            y1, y2 = max(0, y - size//2), min(h, y + size//2)
            x1, x2 = max(0, x - size//2), min(w, x + size//2)

            if base_brightness > 0.15:  # Good visibility
                mono[y1:y2, x1:x2] += np.random.uniform(0.3, 0.6)
            else:  # Low light - weaker signal
                mono[y1:y2, x1:x2] += np.random.uniform(0.1, 0.3)

        return mono

    def _handle_detections(self, detections: List, pair_id: int):
        """Handle detected targets"""

        for det in detections:
            # In real system, you would:
            # - Update tracking database
            # - Send alerts for new targets
            # - Update display/visualization
            # - Log to file/database
            pass

    def _log_progress(self):
        """Log current processing status"""

        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed

        # Get metrics from fusion manager
        manager_metrics = self.fusion_manager.get_performance_metrics()

        # Get metrics from detection fusion
        detection_metrics = self.detection_fusion.get_performance_metrics()

        logger.info(
            f"Frame {self.frame_count} | "
            f"FPS: {fps:.1f} | "
            f"Tracks: {manager_metrics.get('active_tracks', 0)} | "
            f"Reg Error: {manager_metrics.get('avg_registration_error_mm', 0):.2f}mm | "
            f"Confirm: {manager_metrics.get('target_confirmation_rate', 0)*100:.1f}%"
        )

        # Store performance snapshot
        self.performance_history.append({
            'timestamp': time.time(),
            'frame': self.frame_count,
            'fps': fps,
            'manager_metrics': manager_metrics.copy(),
            'detection_metrics': detection_metrics.copy()
        })

    def _print_statistics(self):
        """Print final statistics"""

        if not self.performance_history:
            logger.warning("No performance data collected")
            return

        logger.info("\n" + "=" * 70)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 70)

        # Overall stats
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time

        logger.info(f"\nOverall Performance:")
        logger.info(f"  Total frames: {self.frame_count}")
        logger.info(f"  Total time: {total_time:.2f} s")
        logger.info(f"  Average FPS: {avg_fps:.2f}")
        logger.info(f"  Target FPS: {self.config.target_fps}")
        logger.info(f"  Performance: {avg_fps/self.config.target_fps*100:.1f}% of target")

        # Fusion manager stats
        manager_metrics = self.fusion_manager.get_performance_metrics()

        logger.info(f"\nFusion Manager Metrics:")
        logger.info(f"  Avg registration error: {manager_metrics.get('avg_registration_error_mm', 0):.2f} mm")
        logger.info(f"  Target confirmation rate: {manager_metrics.get('target_confirmation_rate', 0)*100:.1f}%")
        logger.info(f"  False positive reduction: {manager_metrics.get('false_positive_reduction_rate', 0)*100:.1f}%")
        logger.info(f"  Active tracks: {manager_metrics.get('active_tracks', 0)}")

        # Detection fusion stats
        detection_metrics = self.detection_fusion.get_performance_metrics()

        logger.info(f"\nDetection Fusion Metrics:")
        logger.info(f"  Total detections: {detection_metrics['total_detections']}")
        logger.info(f"  Cross-validation rate: {detection_metrics['cross_validation_rate']*100:.1f}%")
        logger.info(f"  False positive reduction: {detection_metrics['false_positive_reduction_rate']*100:.1f}%")
        logger.info(f"  Occlusions handled: {detection_metrics['occlusions_handled']}")

        # Detection history
        if self.detection_history:
            total_dets = sum(d['count'] for d in self.detection_history)
            avg_dets_per_frame = total_dets / len(self.detection_history)

            logger.info(f"\nDetection History:")
            logger.info(f"  Total detections: {total_dets}")
            logger.info(f"  Frames with detections: {len(self.detection_history)}")
            logger.info(f"  Avg detections/frame: {avg_dets_per_frame:.2f}")

        logger.info("\n" + "=" * 70)

        # Performance over time
        if len(self.performance_history) > 10:
            logger.info("\nPerformance Trends:")

            recent = self.performance_history[-30:]  # Last 30 samples
            recent_fps = [p['fps'] for p in recent]

            logger.info(f"  Recent FPS: {np.mean(recent_fps):.2f} ± {np.std(recent_fps):.2f}")
            logger.info(f"  Min FPS: {np.min(recent_fps):.2f}")
            logger.info(f"  Max FPS: {np.max(recent_fps):.2f}")

        # Requirements validation
        logger.info("\n" + "=" * 70)
        logger.info("REQUIREMENTS VALIDATION")
        logger.info("=" * 70)

        reqs = [
            ("Real-time FPS", avg_fps >= 30, f"{avg_fps:.1f} FPS", "30+ FPS"),
            ("Registration accuracy", manager_metrics.get('avg_registration_error_mm', 999) < 5,
             f"{manager_metrics.get('avg_registration_error_mm', 0):.2f} mm", "<5mm @ 5km"),
            ("Target confirmation", manager_metrics.get('target_confirmation_rate', 0) >= 0.99,
             f"{manager_metrics.get('target_confirmation_rate', 0)*100:.1f}%", "99%+"),
            ("False positive reduction", manager_metrics.get('false_positive_reduction_rate', 0) >= 0.90,
             f"{manager_metrics.get('false_positive_reduction_rate', 0)*100:.1f}%", "90%+"),
        ]

        for req_name, passed, actual, target in reqs:
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {status} | {req_name:25s} | Actual: {actual:15s} | Target: {target}")

        logger.info("=" * 70 + "\n")


def main():
    """Main demonstration"""

    logger.info("=" * 70)
    logger.info("Thermal + Monochrome Fusion System - Integration Example")
    logger.info("=" * 70)
    logger.info("")

    # Create system
    system = MultiCameraFusionSystem()

    # Add multiple camera pairs
    logger.info("Adding camera pairs...")
    system.add_camera_pair(
        thermal_camera_id=0,
        mono_camera_id=0,
        baseline_m=0.5,
        resolution=(480, 640)
    )

    system.add_camera_pair(
        thermal_camera_id=1,
        mono_camera_id=1,
        baseline_m=0.5,
        resolution=(480, 640)
    )

    logger.info("")

    # Run for 10 seconds
    logger.info("Starting fusion processing for 10 seconds...")
    logger.info("Press Ctrl+C to stop early")
    logger.info("")

    system.start(duration_seconds=10)

    logger.info("\nDemo completed!")


if __name__ == '__main__':
    main()
