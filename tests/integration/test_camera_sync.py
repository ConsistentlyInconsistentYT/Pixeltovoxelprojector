"""
Camera Synchronization Integration Tests
Tests timestamp synchronization, frame alignment, dropped frame handling, and multi-pair coordination

Requirements tested:
- Sub-millisecond timestamp synchronization accuracy
- Frame alignment across 20 cameras (10 pairs)
- Dropped frame detection and recovery
- Hardware trigger coordination
- PTP synchronization quality
"""

import pytest
import numpy as np
import time
import threading
from typing import List, Dict
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from camera.camera_sync import (
    CameraSynchronizer, FrameMetadata, SyncedFrameSet,
    SyncMode, PTPManager, HardwareTriggerController,
    FrameBuffer, SyncStatistics
)

logger = logging.getLogger(__name__)


class TestCameraSynchronization:
    """Camera synchronization integration tests"""

    @pytest.fixture
    def sync_system(self):
        """Create camera synchronization system"""
        sync = CameraSynchronizer(num_pairs=10, sync_mode=SyncMode.HYBRID)
        sync.start()
        yield sync
        sync.stop()

    @pytest.fixture
    def ptp_manager(self):
        """Create PTP manager"""
        return PTPManager()

    @pytest.fixture
    def hw_trigger(self):
        """Create hardware trigger controller"""
        controller = HardwareTriggerController(num_cameras=20)
        controller.start_hardware_trigger(rate=30.0)
        yield controller
        controller.stop_hardware_trigger()

    def test_timestamp_synchronization_accuracy(self, sync_system):
        """Test sub-millisecond timestamp synchronization"""
        logger.info("Testing timestamp synchronization accuracy")

        num_frames = 100
        sync_errors = []

        for frame_num in range(num_frames):
            # Simulate synchronized camera pair
            timestamp = time.time()

            mono_metadata = FrameMetadata(
                camera_id=0,
                pair_id=0,
                frame_number=frame_num,
                timestamp=timestamp,
                system_time=timestamp,
                trigger_id=frame_num
            )

            # Thermal camera with slight offset
            thermal_timestamp = timestamp + np.random.normal(0, 0.0001)  # 0.1ms jitter

            thermal_metadata = FrameMetadata(
                camera_id=1,
                pair_id=0,
                frame_number=frame_num,
                timestamp=thermal_timestamp,
                system_time=thermal_timestamp,
                trigger_id=frame_num
            )

            sync_system.add_frame(0, mono_metadata)
            sync_system.add_frame(1, thermal_metadata)

            time.sleep(0.01)  # Allow processing

            # Get synced frame set
            synced_set = sync_system.get_synced_frame_set(timeout=0.1)

            if synced_set and synced_set.is_valid:
                sync_errors.append(synced_set.sync_error)

        # Validate synchronization accuracy
        assert len(sync_errors) > 0, "No synchronized frames produced"

        avg_sync_error = np.mean(sync_errors)
        max_sync_error = np.max(sync_errors)
        std_sync_error = np.std(sync_errors)

        logger.info(f"Timestamp sync results:")
        logger.info(f"  Average sync error: {avg_sync_error:.4f} ms")
        logger.info(f"  Max sync error: {max_sync_error:.4f} ms")
        logger.info(f"  Std sync error: {std_sync_error:.4f} ms")

        # Requirements: < 1ms average, < 10ms max
        assert avg_sync_error < 1.0, f"Average sync error {avg_sync_error:.4f}ms exceeds 1ms"
        assert max_sync_error < 10.0, f"Max sync error {max_sync_error:.4f}ms exceeds 10ms"

    def test_frame_alignment_all_pairs(self, sync_system):
        """Test frame alignment across all 10 camera pairs"""
        logger.info("Testing frame alignment for all 10 camera pairs")

        num_frames = 50
        aligned_frames = {pair_id: 0 for pair_id in range(10)}
        sync_errors_per_pair = {pair_id: [] for pair_id in range(10)}

        for frame_num in range(num_frames):
            base_timestamp = time.time()

            # Send frames for all camera pairs
            for pair_id in range(10):
                mono_id = pair_id * 2
                thermal_id = pair_id * 2 + 1

                # Add realistic jitter
                mono_jitter = np.random.normal(0, 0.0002)  # 0.2ms
                thermal_jitter = np.random.normal(0, 0.0002)

                mono_metadata = FrameMetadata(
                    camera_id=mono_id,
                    pair_id=pair_id,
                    frame_number=frame_num,
                    timestamp=base_timestamp + mono_jitter,
                    system_time=base_timestamp + mono_jitter,
                    trigger_id=frame_num
                )

                thermal_metadata = FrameMetadata(
                    camera_id=thermal_id,
                    pair_id=pair_id,
                    frame_number=frame_num,
                    timestamp=base_timestamp + thermal_jitter,
                    system_time=base_timestamp + thermal_jitter,
                    trigger_id=frame_num
                )

                sync_system.add_frame(mono_id, mono_metadata)
                sync_system.add_frame(thermal_id, thermal_metadata)

            # Collect synchronized frames
            time.sleep(0.02)  # Allow processing

            for _ in range(10):
                synced_set = sync_system.get_synced_frame_set(timeout=0.05)
                if synced_set and synced_set.is_valid:
                    aligned_frames[synced_set.pair_id] += 1
                    sync_errors_per_pair[synced_set.pair_id].append(synced_set.sync_error)

        # Validate all pairs are aligned
        logger.info("Frame alignment per pair:")
        for pair_id in range(10):
            count = aligned_frames[pair_id]
            avg_error = np.mean(sync_errors_per_pair[pair_id]) if sync_errors_per_pair[pair_id] else 0

            logger.info(f"  Pair {pair_id}: {count} frames aligned, avg error: {avg_error:.4f}ms")

            assert count >= num_frames * 0.9, f"Pair {pair_id} only aligned {count}/{num_frames} frames"
            if sync_errors_per_pair[pair_id]:
                assert avg_error < 2.0, f"Pair {pair_id} avg sync error {avg_error:.4f}ms too high"

    def test_dropped_frame_detection(self, sync_system):
        """Test detection and handling of dropped frames"""
        logger.info("Testing dropped frame detection")

        num_frames = 100
        drop_probability = 0.05  # 5% drop rate
        dropped_frames = []

        for frame_num in range(num_frames):
            # Randomly drop frames
            if np.random.random() > drop_probability:
                mono_metadata = FrameMetadata(
                    camera_id=0,
                    pair_id=0,
                    frame_number=frame_num,
                    timestamp=time.time(),
                    system_time=time.time(),
                    trigger_id=frame_num
                )
                sync_system.add_frame(0, mono_metadata)
            else:
                dropped_frames.append(frame_num)

            # Thermal camera (no drops)
            thermal_metadata = FrameMetadata(
                camera_id=1,
                pair_id=0,
                frame_number=frame_num,
                timestamp=time.time(),
                system_time=time.time(),
                trigger_id=frame_num
            )
            sync_system.add_frame(1, thermal_metadata)

            time.sleep(0.01)

        time.sleep(0.5)  # Allow processing

        # Check metrics for dropped frames
        metrics = sync_system.get_synchronization_metrics()

        logger.info(f"Dropped frame test results:")
        logger.info(f"  Frames intentionally dropped: {len(dropped_frames)}")
        logger.info(f"  System detected drops: {metrics['camera_stats'][0]['frames_dropped']}")
        logger.info(f"  Recovery attempts: {metrics['camera_stats'][0]['frames_recovered']}")

        # System should detect some dropped frames
        assert metrics['overall']['total_dropped'] > 0, "System did not detect any dropped frames"

    def test_dropped_frame_recovery(self, sync_system):
        """Test recovery mechanism for dropped frames"""
        logger.info("Testing dropped frame recovery")

        num_frames = 50
        recovered_frames = 0

        for frame_num in range(num_frames):
            timestamp = time.time()

            # Drop every 5th mono frame
            if frame_num % 5 != 0:
                mono_metadata = FrameMetadata(
                    camera_id=0,
                    pair_id=0,
                    frame_number=frame_num,
                    timestamp=timestamp,
                    system_time=timestamp,
                    trigger_id=frame_num
                )
                sync_system.add_frame(0, mono_metadata)

            # Always send thermal
            thermal_metadata = FrameMetadata(
                camera_id=1,
                pair_id=0,
                frame_number=frame_num,
                timestamp=timestamp,
                system_time=timestamp,
                trigger_id=frame_num
            )
            sync_system.add_frame(1, thermal_metadata)

            time.sleep(0.01)

            # Check for recovered frames
            synced_set = sync_system.get_synced_frame_set(timeout=0.05)
            if synced_set and synced_set.recovery_applied:
                recovered_frames += 1

        logger.info(f"Frame recovery results:")
        logger.info(f"  Recovered frames: {recovered_frames}")

        metrics = sync_system.get_synchronization_metrics()
        total_recovered = metrics['overall']['total_recovered']

        logger.info(f"  Total system recoveries: {total_recovered}")

        # Some frames should be recovered
        assert total_recovered > 0, "No frames were recovered"

    def test_hardware_trigger_coordination(self, hw_trigger):
        """Test hardware trigger coordination across cameras"""
        logger.info("Testing hardware trigger coordination")

        # Let trigger run for a bit
        time.sleep(2.0)

        # Simulate camera responses
        num_cameras = 20
        num_triggers = 50

        for trigger_id in range(1, num_triggers + 1):
            # Simulate most cameras responding
            for camera_id in range(num_cameras):
                if np.random.random() > 0.02:  # 98% response rate
                    hw_trigger.register_trigger_response(camera_id, trigger_id)

            time.sleep(0.033)  # 30 Hz

        # Check trigger statistics
        stats = hw_trigger.get_trigger_stats()

        logger.info(f"Hardware trigger statistics:")
        logger.info(f"  Total triggers: {stats['total_triggers']}")
        logger.info(f"  Avg response rate: {stats['avg_response_rate']*100:.2f}%")
        logger.info(f"  Trigger rate: {stats['trigger_rate']} Hz")

        # Validate trigger performance
        assert stats['total_triggers'] >= num_triggers, "Not enough triggers generated"
        assert stats['avg_response_rate'] > 0.95, f"Response rate {stats['avg_response_rate']*100:.2f}% too low"
        assert abs(stats['trigger_rate'] - 30.0) < 1.0, f"Trigger rate {stats['trigger_rate']} Hz not close to 30 Hz"

    def test_ptp_synchronization(self, ptp_manager):
        """Test PTP synchronization quality"""
        logger.info("Testing PTP synchronization")

        # Simulate PTP sync updates
        num_updates = 100
        master_time = time.time()

        for i in range(num_updates):
            local_time = time.time()
            # Simulate realistic offset and drift
            offset = 0.0005 + np.random.normal(0, 0.0001)  # 0.5ms ± 0.1ms
            ptp_manager.update_master_offset(local_time + offset, local_time)
            time.sleep(0.01)

        # Check sync quality
        quality = ptp_manager.get_sync_quality()

        logger.info(f"PTP synchronization quality:")
        logger.info(f"  Offset: {quality['offset']:.4f} ms")
        logger.info(f"  Jitter: {quality['jitter']:.2f} µs")
        logger.info(f"  Is synchronized: {quality['is_synced']}")
        logger.info(f"  Time since sync: {quality['time_since_sync']:.4f} s")

        # Validate PTP quality
        assert quality['is_synced'], "PTP failed to synchronize"
        assert abs(quality['offset']) < 2.0, f"PTP offset {quality['offset']:.4f}ms too large"
        assert quality['jitter'] < 1000.0, f"PTP jitter {quality['jitter']:.2f}µs too high"

    def test_multi_pair_coordination(self, sync_system):
        """Test coordination between multiple camera pairs"""
        logger.info("Testing multi-pair coordination")

        num_frames = 30
        num_pairs = 10

        pair_sync_errors = {pair_id: [] for pair_id in range(num_pairs)}
        pair_frame_counts = {pair_id: 0 for pair_id in range(num_pairs)}

        for frame_num in range(num_frames):
            base_time = time.time()

            # Simulate all pairs with different offsets
            for pair_id in range(num_pairs):
                mono_id = pair_id * 2
                thermal_id = pair_id * 2 + 1

                # Each pair has slightly different timing
                pair_offset = pair_id * 0.0001  # 0.1ms per pair
                timestamp = base_time + pair_offset

                mono_metadata = FrameMetadata(
                    camera_id=mono_id,
                    pair_id=pair_id,
                    frame_number=frame_num,
                    timestamp=timestamp,
                    system_time=timestamp,
                    trigger_id=frame_num
                )

                thermal_metadata = FrameMetadata(
                    camera_id=thermal_id,
                    pair_id=pair_id,
                    frame_number=frame_num,
                    timestamp=timestamp + np.random.normal(0, 0.0001),
                    system_time=timestamp,
                    trigger_id=frame_num
                )

                sync_system.add_frame(mono_id, mono_metadata)
                sync_system.add_frame(thermal_id, thermal_metadata)

            # Collect results
            time.sleep(0.02)

            for _ in range(num_pairs):
                synced_set = sync_system.get_synced_frame_set(timeout=0.05)
                if synced_set and synced_set.is_valid:
                    pair_sync_errors[synced_set.pair_id].append(synced_set.sync_error)
                    pair_frame_counts[synced_set.pair_id] += 1

        # Validate coordination
        logger.info("Multi-pair coordination results:")
        for pair_id in range(num_pairs):
            count = pair_frame_counts[pair_id]
            errors = pair_sync_errors[pair_id]
            avg_error = np.mean(errors) if errors else 0
            max_error = np.max(errors) if errors else 0

            logger.info(f"  Pair {pair_id}: {count} frames, avg error: {avg_error:.4f}ms, max: {max_error:.4f}ms")

            # Each pair should process most frames
            assert count >= num_frames * 0.8, f"Pair {pair_id} only processed {count}/{num_frames} frames"

    def test_sync_tolerance_adjustment(self, sync_system):
        """Test dynamic sync tolerance adjustment"""
        logger.info("Testing sync tolerance adjustment")

        # Test different tolerance levels
        tolerances = [0.5, 1.0, 2.0, 5.0]  # milliseconds

        for tolerance_ms in tolerances:
            sync_system.set_sync_tolerance(tolerance_ms)

            num_frames = 20
            accepted_frames = 0

            for frame_num in range(num_frames):
                timestamp = time.time()

                # Add frames with varying sync error
                mono_metadata = FrameMetadata(
                    camera_id=0,
                    pair_id=0,
                    frame_number=frame_num,
                    timestamp=timestamp,
                    system_time=timestamp,
                    trigger_id=frame_num
                )

                # Thermal with controlled offset
                offset = np.random.uniform(0, tolerance_ms / 1000.0)
                thermal_metadata = FrameMetadata(
                    camera_id=1,
                    pair_id=0,
                    frame_number=frame_num,
                    timestamp=timestamp + offset,
                    system_time=timestamp + offset,
                    trigger_id=frame_num
                )

                sync_system.add_frame(0, mono_metadata)
                sync_system.add_frame(1, thermal_metadata)

                time.sleep(0.01)

                synced_set = sync_system.get_synced_frame_set(timeout=0.05)
                if synced_set and synced_set.is_valid:
                    accepted_frames += 1

            logger.info(f"Tolerance {tolerance_ms}ms: {accepted_frames}/{num_frames} frames accepted")

            # More frames should be accepted with higher tolerance
            assert accepted_frames > 0, f"No frames accepted with {tolerance_ms}ms tolerance"

    def test_synchronization_performance_under_load(self, sync_system):
        """Test synchronization performance under high load"""
        logger.info("Testing synchronization under high load")

        num_frames = 100
        num_pairs = 10

        start_time = time.time()
        sync_latencies = []

        for frame_num in range(num_frames):
            frame_start = time.time()

            # Send frames for all pairs
            for pair_id in range(num_pairs):
                mono_id = pair_id * 2
                thermal_id = pair_id * 2 + 1

                timestamp = time.time()

                mono_metadata = FrameMetadata(
                    camera_id=mono_id,
                    pair_id=pair_id,
                    frame_number=frame_num,
                    timestamp=timestamp,
                    system_time=timestamp,
                    trigger_id=frame_num
                )

                thermal_metadata = FrameMetadata(
                    camera_id=thermal_id,
                    pair_id=pair_id,
                    frame_number=frame_num,
                    timestamp=timestamp + np.random.normal(0, 0.0002),
                    system_time=timestamp,
                    trigger_id=frame_num
                )

                sync_system.add_frame(mono_id, mono_metadata)
                sync_system.add_frame(thermal_id, thermal_metadata)

            sync_latency = (time.time() - frame_start) * 1000
            sync_latencies.append(sync_latency)

            time.sleep(0.005)  # High frame rate

        total_time = time.time() - start_time
        avg_latency = np.mean(sync_latencies)
        max_latency = np.max(sync_latencies)

        logger.info(f"Synchronization performance under load:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Avg sync latency: {avg_latency:.4f}ms")
        logger.info(f"  Max sync latency: {max_latency:.4f}ms")
        logger.info(f"  Effective frame rate: {num_frames/total_time:.2f} fps")

        # Validate performance
        assert avg_latency < 10.0, f"Average sync latency {avg_latency:.4f}ms too high"
        assert max_latency < 50.0, f"Max sync latency {max_latency:.4f}ms too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
