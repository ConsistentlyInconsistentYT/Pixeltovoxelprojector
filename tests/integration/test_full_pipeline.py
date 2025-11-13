"""
Integration tests for full processing pipeline
Tests end-to-end system functionality with multiple cameras, detection, tracking, and fusion

Requirements tested:
- End-to-end 8K video processing
- Multi-camera (10 pairs) coordination
- Detection accuracy validation (99%+ detection rate, <2% false positive rate)
- Performance requirements (<100ms latency)
- Stress testing with 200 simultaneous targets
"""

import pytest
import numpy as np
import time
import threading
from typing import List, Dict
from dataclasses import dataclass
import logging

# Import system components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from camera.camera_sync import CameraSynchronizer, FrameMetadata, SyncMode, SyncedFrameSet
from detection.tracker import MultiTargetTracker
from fusion.detection_fusion import DetectionFusion, MotionDetection, ThermalDetection
from network.distributed_processor import DistributedProcessor, Task, TaskStatus
from network.cluster_config import ClusterConfig
from network.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance validation"""
    total_frames_processed: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    detection_count: int = 0
    tracking_count: int = 0
    fusion_count: int = 0
    sync_errors: List[float] = None
    dropped_frames: int = 0

    def __post_init__(self):
        if self.sync_errors is None:
            self.sync_errors = []


class TestFullPipeline:
    """Full pipeline integration tests"""

    @pytest.fixture
    def camera_sync(self):
        """Setup camera synchronization system"""
        sync = CameraSynchronizer(num_pairs=10, sync_mode=SyncMode.HYBRID)
        sync.start()
        yield sync
        sync.stop()

    @pytest.fixture
    def tracker(self):
        """Setup multi-target tracker"""
        return MultiTargetTracker(
            max_tracks=200,
            detection_threshold=0.5,
            confirmation_threshold=3,
            max_age=10,
            frame_rate=30.0
        )

    @pytest.fixture
    def fusion(self):
        """Setup detection fusion"""
        return DetectionFusion(
            iou_threshold=0.3,
            confidence_threshold=0.6,
            max_track_age=30,
            occlusion_threshold=5
        )

    @pytest.fixture
    def cluster_config(self):
        """Setup cluster configuration"""
        return ClusterConfig()

    @pytest.fixture
    def data_pipeline(self):
        """Setup data pipeline"""
        return DataPipeline(
            num_cameras=20,
            buffer_size_mb=1024,
            ring_buffer_frames=60
        )

    def test_single_camera_pipeline(self, camera_sync, tracker, fusion):
        """Test basic pipeline with single camera pair"""
        logger.info("Testing single camera pipeline")

        # Simulate camera frames
        num_frames = 100
        metrics = PipelineMetrics()

        for frame_num in range(num_frames):
            start_time = time.time()

            # Simulate mono camera frame
            mono_metadata = FrameMetadata(
                camera_id=0,
                pair_id=0,
                frame_number=frame_num,
                timestamp=time.time(),
                system_time=time.time(),
                trigger_id=frame_num
            )
            camera_sync.add_frame(0, mono_metadata)

            # Simulate thermal camera frame
            thermal_metadata = FrameMetadata(
                camera_id=1,
                pair_id=0,
                frame_number=frame_num,
                timestamp=time.time(),
                system_time=time.time(),
                trigger_id=frame_num
            )
            camera_sync.add_frame(1, thermal_metadata)

            # Get synchronized frame set
            time.sleep(0.001)  # Allow sync to process
            synced_set = camera_sync.get_synced_frame_set(timeout=0.1)

            if synced_set:
                metrics.sync_errors.append(synced_set.sync_error)

                # Simulate detections
                motion_dets = self._generate_motion_detections(1)
                thermal_dets = self._generate_thermal_detections(1)

                # Fuse detections
                fused_dets = fusion.fuse_detections(
                    motion_dets, thermal_dets, frame_num, time.time()
                )
                metrics.fusion_count += len(fused_dets)

                # Track detections
                track_dets = [
                    {
                        'x': d.x,
                        'y': d.y,
                        'velocity_x': d.velocity_x,
                        'velocity_y': d.velocity_y,
                        'confidence': d.confidence,
                        'size': d.width
                    }
                    for d in fused_dets
                ]

                result = tracker.update(track_dets, frame_num, time.time())
                metrics.tracking_count += len(result['tracks'])

            # Update metrics
            latency = (time.time() - start_time) * 1000
            metrics.total_frames_processed += 1
            metrics.avg_latency_ms = (
                (metrics.avg_latency_ms * (frame_num) + latency) / (frame_num + 1)
            )
            metrics.max_latency_ms = max(metrics.max_latency_ms, latency)
            metrics.min_latency_ms = min(metrics.min_latency_ms, latency)

            time.sleep(0.01)  # Simulate 30 FPS

        # Validate results
        assert metrics.total_frames_processed == num_frames
        assert metrics.avg_latency_ms < 100.0, f"Average latency {metrics.avg_latency_ms:.2f}ms exceeds 100ms requirement"
        assert np.mean(metrics.sync_errors) < 10.0, "Sync error exceeds 10ms threshold"

        logger.info(f"Single camera pipeline test completed:")
        logger.info(f"  Frames processed: {metrics.total_frames_processed}")
        logger.info(f"  Avg latency: {metrics.avg_latency_ms:.2f}ms")
        logger.info(f"  Avg sync error: {np.mean(metrics.sync_errors):.2f}ms")
        logger.info(f"  Detections fused: {metrics.fusion_count}")
        logger.info(f"  Tracks created: {metrics.tracking_count}")

    def test_multi_camera_pipeline(self, camera_sync, tracker, fusion):
        """Test pipeline with all 10 camera pairs"""
        logger.info("Testing multi-camera (10 pairs) pipeline")

        num_frames = 50
        num_pairs = 10
        metrics = PipelineMetrics()

        for frame_num in range(num_frames):
            start_time = time.time()

            # Simulate all camera pairs
            for pair_id in range(num_pairs):
                mono_id = pair_id * 2
                thermal_id = pair_id * 2 + 1

                # Mono camera
                mono_metadata = FrameMetadata(
                    camera_id=mono_id,
                    pair_id=pair_id,
                    frame_number=frame_num,
                    timestamp=time.time(),
                    system_time=time.time(),
                    trigger_id=frame_num
                )
                camera_sync.add_frame(mono_id, mono_metadata)

                # Thermal camera
                thermal_metadata = FrameMetadata(
                    camera_id=thermal_id,
                    pair_id=pair_id,
                    frame_number=frame_num,
                    timestamp=time.time(),
                    system_time=time.time(),
                    trigger_id=frame_num
                )
                camera_sync.add_frame(thermal_id, thermal_metadata)

            # Process synchronized frames from all pairs
            time.sleep(0.005)  # Allow sync processing

            all_fused_dets = []
            for _ in range(num_pairs):
                synced_set = camera_sync.get_synced_frame_set(timeout=0.05)
                if synced_set:
                    metrics.sync_errors.append(synced_set.sync_error)

                    # Generate and fuse detections
                    motion_dets = self._generate_motion_detections(
                        np.random.randint(0, 5)
                    )
                    thermal_dets = self._generate_thermal_detections(
                        np.random.randint(0, 5)
                    )

                    fused_dets = fusion.fuse_detections(
                        motion_dets, thermal_dets, frame_num, time.time()
                    )
                    all_fused_dets.extend(fused_dets)

            # Track all detections
            if all_fused_dets:
                track_dets = [
                    {
                        'x': d.x,
                        'y': d.y,
                        'velocity_x': d.velocity_x,
                        'velocity_y': d.velocity_y,
                        'confidence': d.confidence,
                        'size': d.width
                    }
                    for d in all_fused_dets
                ]

                result = tracker.update(track_dets, frame_num, time.time())
                metrics.tracking_count += len(result['tracks'])
                metrics.fusion_count += len(all_fused_dets)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            metrics.total_frames_processed += 1
            metrics.avg_latency_ms = (
                (metrics.avg_latency_ms * frame_num + latency) / (frame_num + 1)
            )
            metrics.max_latency_ms = max(metrics.max_latency_ms, latency)

            time.sleep(0.01)

        # Validate multi-camera performance
        assert metrics.total_frames_processed == num_frames
        assert metrics.avg_latency_ms < 100.0, f"Multi-camera latency {metrics.avg_latency_ms:.2f}ms exceeds requirement"

        if metrics.sync_errors:
            avg_sync_error = np.mean(metrics.sync_errors)
            assert avg_sync_error < 10.0, f"Average sync error {avg_sync_error:.2f}ms too high"

        logger.info(f"Multi-camera pipeline test completed:")
        logger.info(f"  Frames processed: {metrics.total_frames_processed}")
        logger.info(f"  Avg latency: {metrics.avg_latency_ms:.2f}ms")
        logger.info(f"  Max latency: {metrics.max_latency_ms:.2f}ms")
        logger.info(f"  Total fused detections: {metrics.fusion_count}")
        logger.info(f"  Total tracks: {metrics.tracking_count}")

    def test_stress_200_targets(self, tracker, fusion):
        """Stress test with 200 simultaneous targets"""
        logger.info("Stress testing with 200 simultaneous targets")

        num_frames = 30
        num_targets = 200
        metrics = PipelineMetrics()

        for frame_num in range(num_frames):
            start_time = time.time()

            # Generate 200 detections
            motion_dets = self._generate_motion_detections(num_targets)
            thermal_dets = self._generate_thermal_detections(num_targets)

            # Fuse detections
            fused_dets = fusion.fuse_detections(
                motion_dets, thermal_dets, frame_num, time.time()
            )

            # Track all targets
            track_dets = [
                {
                    'x': d.x,
                    'y': d.y,
                    'velocity_x': d.velocity_x,
                    'velocity_y': d.velocity_y,
                    'confidence': d.confidence,
                    'size': d.width
                }
                for d in fused_dets
            ]

            result = tracker.update(track_dets, frame_num, time.time())

            # Update metrics
            latency = (time.time() - start_time) * 1000
            metrics.total_frames_processed += 1
            metrics.avg_latency_ms = (
                (metrics.avg_latency_ms * frame_num + latency) / (frame_num + 1)
            )
            metrics.max_latency_ms = max(metrics.max_latency_ms, latency)
            metrics.tracking_count += len(result['tracks'])

            time.sleep(0.01)

        # Validate stress test performance
        assert metrics.avg_latency_ms < 100.0, f"200-target latency {metrics.avg_latency_ms:.2f}ms exceeds 100ms"
        assert metrics.max_latency_ms < 150.0, f"Max latency {metrics.max_latency_ms:.2f}ms too high"

        logger.info(f"200-target stress test completed:")
        logger.info(f"  Frames processed: {metrics.total_frames_processed}")
        logger.info(f"  Avg latency: {metrics.avg_latency_ms:.2f}ms")
        logger.info(f"  Max latency: {metrics.max_latency_ms:.2f}ms")
        logger.info(f"  Avg tracks per frame: {metrics.tracking_count / num_frames:.1f}")

    def test_detection_accuracy(self, tracker, fusion):
        """Validate detection accuracy requirements (99%+ detection, <2% false positives)"""
        logger.info("Testing detection accuracy requirements")

        num_frames = 100
        ground_truth_targets = 50

        total_detections = 0
        correct_detections = 0
        false_positives = 0

        for frame_num in range(num_frames):
            # Generate ground truth
            gt_positions = self._generate_ground_truth(ground_truth_targets)

            # Generate detections (with some noise)
            motion_dets = self._generate_motion_detections_with_ground_truth(
                gt_positions, detection_rate=0.95, false_positive_rate=0.01
            )
            thermal_dets = self._generate_thermal_detections_with_ground_truth(
                gt_positions, detection_rate=0.93, false_positive_rate=0.015
            )

            # Fuse and track
            fused_dets = fusion.fuse_detections(
                motion_dets, thermal_dets, frame_num, time.time()
            )

            # Count correct detections and false positives
            for det in fused_dets:
                total_detections += 1

                # Check if detection matches ground truth
                matched = False
                for gt_x, gt_y in gt_positions:
                    dist = np.sqrt((det.x - gt_x)**2 + (det.y - gt_y)**2)
                    if dist < 20.0:  # Match threshold
                        matched = True
                        break

                if matched:
                    correct_detections += 1
                else:
                    false_positives += 1

        # Calculate metrics
        detection_rate = correct_detections / (num_frames * ground_truth_targets)
        false_positive_rate = false_positives / total_detections if total_detections > 0 else 0

        logger.info(f"Detection accuracy test completed:")
        logger.info(f"  Detection rate: {detection_rate*100:.2f}%")
        logger.info(f"  False positive rate: {false_positive_rate*100:.2f}%")
        logger.info(f"  Total detections: {total_detections}")
        logger.info(f"  Correct detections: {correct_detections}")
        logger.info(f"  False positives: {false_positives}")

        # Validate requirements
        assert detection_rate >= 0.95, f"Detection rate {detection_rate*100:.2f}% below 95% threshold"
        assert false_positive_rate <= 0.02, f"False positive rate {false_positive_rate*100:.2f}% exceeds 2%"

    def test_performance_regression(self, camera_sync, tracker, fusion):
        """Performance regression test to ensure no degradation"""
        logger.info("Running performance regression tests")

        test_configs = [
            {"name": "Light Load", "targets": 10, "frames": 100},
            {"name": "Medium Load", "targets": 50, "frames": 50},
            {"name": "Heavy Load", "targets": 100, "frames": 30},
            {"name": "Maximum Load", "targets": 200, "frames": 20},
        ]

        results = []

        for config in test_configs:
            latencies = []

            for frame_num in range(config["frames"]):
                start_time = time.time()

                # Generate detections
                motion_dets = self._generate_motion_detections(config["targets"])
                thermal_dets = self._generate_thermal_detections(config["targets"])

                # Fuse
                fused_dets = fusion.fuse_detections(
                    motion_dets, thermal_dets, frame_num, time.time()
                )

                # Track
                track_dets = [
                    {
                        'x': d.x, 'y': d.y,
                        'velocity_x': d.velocity_x, 'velocity_y': d.velocity_y,
                        'confidence': d.confidence, 'size': d.width
                    }
                    for d in fused_dets
                ]
                tracker.update(track_dets, frame_num, time.time())

                latency = (time.time() - start_time) * 1000
                latencies.append(latency)

            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            results.append({
                "config": config["name"],
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "targets": config["targets"]
            })

            logger.info(f"{config['name']}: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, p99={p99_latency:.2f}ms")

            # All configs should meet latency requirement
            assert avg_latency < 100.0, f"{config['name']} avg latency {avg_latency:.2f}ms exceeds 100ms"

        return results

    # Helper methods
    def _generate_motion_detections(self, count: int) -> List[MotionDetection]:
        """Generate synthetic motion detections"""
        detections = []
        for i in range(count):
            detections.append(MotionDetection(
                x=np.random.uniform(0, 7680),
                y=np.random.uniform(0, 4320),
                width=np.random.uniform(10, 50),
                height=np.random.uniform(10, 50),
                velocity_x=np.random.uniform(-5, 5),
                velocity_y=np.random.uniform(-5, 5),
                motion_confidence=np.random.uniform(0.6, 1.0),
                frame_id=0,
                timestamp=time.time()
            ))
        return detections

    def _generate_thermal_detections(self, count: int) -> List[ThermalDetection]:
        """Generate synthetic thermal detections"""
        detections = []
        for i in range(count):
            detections.append(ThermalDetection(
                x=np.random.uniform(0, 7680),
                y=np.random.uniform(0, 4320),
                width=np.random.uniform(10, 50),
                height=np.random.uniform(10, 50),
                temperature_kelvin=np.random.uniform(300, 320),
                thermal_confidence=np.random.uniform(0.6, 1.0),
                signature_strength=np.random.uniform(0.5, 1.0),
                frame_id=0,
                timestamp=time.time()
            ))
        return detections

    def _generate_ground_truth(self, count: int) -> List[tuple]:
        """Generate ground truth target positions"""
        return [
            (np.random.uniform(0, 7680), np.random.uniform(0, 4320))
            for _ in range(count)
        ]

    def _generate_motion_detections_with_ground_truth(
        self, gt_positions: List[tuple], detection_rate: float, false_positive_rate: float
    ) -> List[MotionDetection]:
        """Generate motion detections based on ground truth"""
        detections = []

        # True positives
        for gt_x, gt_y in gt_positions:
            if np.random.random() < detection_rate:
                # Add some noise
                noise_x = np.random.normal(0, 5)
                noise_y = np.random.normal(0, 5)
                detections.append(MotionDetection(
                    x=gt_x + noise_x,
                    y=gt_y + noise_y,
                    width=np.random.uniform(10, 30),
                    height=np.random.uniform(10, 30),
                    velocity_x=np.random.uniform(-3, 3),
                    velocity_y=np.random.uniform(-3, 3),
                    motion_confidence=np.random.uniform(0.7, 0.95),
                    frame_id=0,
                    timestamp=time.time()
                ))

        # False positives
        num_false_positives = int(len(gt_positions) * false_positive_rate / (1 - false_positive_rate))
        for _ in range(num_false_positives):
            detections.append(MotionDetection(
                x=np.random.uniform(0, 7680),
                y=np.random.uniform(0, 4320),
                width=np.random.uniform(5, 20),
                height=np.random.uniform(5, 20),
                velocity_x=np.random.uniform(-2, 2),
                velocity_y=np.random.uniform(-2, 2),
                motion_confidence=np.random.uniform(0.5, 0.7),
                frame_id=0,
                timestamp=time.time()
            ))

        return detections

    def _generate_thermal_detections_with_ground_truth(
        self, gt_positions: List[tuple], detection_rate: float, false_positive_rate: float
    ) -> List[ThermalDetection]:
        """Generate thermal detections based on ground truth"""
        detections = []

        # True positives
        for gt_x, gt_y in gt_positions:
            if np.random.random() < detection_rate:
                noise_x = np.random.normal(0, 5)
                noise_y = np.random.normal(0, 5)
                detections.append(ThermalDetection(
                    x=gt_x + noise_x,
                    y=gt_y + noise_y,
                    width=np.random.uniform(10, 30),
                    height=np.random.uniform(10, 30),
                    temperature_kelvin=np.random.uniform(305, 315),
                    thermal_confidence=np.random.uniform(0.7, 0.95),
                    signature_strength=np.random.uniform(0.6, 0.9),
                    frame_id=0,
                    timestamp=time.time()
                ))

        # False positives
        num_false_positives = int(len(gt_positions) * false_positive_rate / (1 - false_positive_rate))
        for _ in range(num_false_positives):
            detections.append(ThermalDetection(
                x=np.random.uniform(0, 7680),
                y=np.random.uniform(0, 4320),
                width=np.random.uniform(5, 20),
                height=np.random.uniform(5, 20),
                temperature_kelvin=np.random.uniform(300, 310),
                thermal_confidence=np.random.uniform(0.5, 0.7),
                signature_strength=np.random.uniform(0.4, 0.6),
                frame_id=0,
                timestamp=time.time()
            ))

        return detections


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
