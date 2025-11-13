"""
Multi-Camera Synchronization Manager
Handles timestamp synchronization, frame alignment, and hardware triggering for 20 cameras (10 pairs)
"""

import time
import threading
import queue
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np


class SyncMode(Enum):
    """Synchronization modes for camera system"""
    SOFTWARE = 1  # Software-based timestamping
    PTP = 2       # Precision Time Protocol
    HARDWARE = 3  # Hardware trigger sync
    HYBRID = 4    # Combined PTP + Hardware


@dataclass
class FrameMetadata:
    """Metadata for a single camera frame"""
    camera_id: int
    pair_id: int
    frame_number: int
    timestamp: float  # Hardware timestamp in seconds
    system_time: float  # System clock time
    ptp_time: Optional[float] = None  # PTP synchronized time
    trigger_id: Optional[int] = None  # Hardware trigger ID
    exposure_time: float = 0.0
    gain: float = 1.0
    temperature: float = 0.0


@dataclass
class SyncedFrameSet:
    """Set of synchronized frames from a camera pair"""
    pair_id: int
    mono_frame: FrameMetadata
    thermal_frame: FrameMetadata
    sync_error: float  # Synchronization error in milliseconds
    is_valid: bool = True
    recovery_applied: bool = False


@dataclass
class SyncStatistics:
    """Real-time synchronization statistics"""
    camera_id: int
    avg_sync_error: float = 0.0
    max_sync_error: float = 0.0
    min_sync_error: float = float('inf')
    frames_processed: int = 0
    frames_dropped: int = 0
    frames_recovered: int = 0
    jitter: float = 0.0  # Timestamp jitter in microseconds
    last_update: float = field(default_factory=time.time)


class PTPManager:
    """Precision Time Protocol manager for sub-millisecond synchronization"""

    def __init__(self):
        self.master_offset = 0.0
        self.drift_rate = 0.0
        self.last_sync_time = time.time()
        self.sync_interval = 1.0  # Sync every second
        self.offset_history = deque(maxlen=100)
        self.is_synchronized = False
        self.lock = threading.Lock()

    def update_master_offset(self, master_timestamp: float, local_timestamp: float):
        """Update offset from PTP master clock"""
        with self.lock:
            offset = master_timestamp - local_timestamp
            self.offset_history.append(offset)

            # Calculate running average and drift
            if len(self.offset_history) > 10:
                self.master_offset = np.median(list(self.offset_history))
                self.drift_rate = np.std(list(self.offset_history))
                self.is_synchronized = self.drift_rate < 0.001  # < 1ms jitter

            self.last_sync_time = time.time()

    def get_ptp_time(self) -> float:
        """Get current PTP-synchronized time"""
        with self.lock:
            local_time = time.time()
            # Apply offset and drift compensation
            time_since_sync = local_time - self.last_sync_time
            drift_compensation = self.drift_rate * time_since_sync
            return local_time + self.master_offset + drift_compensation

    def get_sync_quality(self) -> Dict[str, float]:
        """Return synchronization quality metrics"""
        with self.lock:
            return {
                'offset': self.master_offset * 1000,  # ms
                'jitter': self.drift_rate * 1000000,  # microseconds
                'is_synced': self.is_synchronized,
                'time_since_sync': time.time() - self.last_sync_time
            }


class HardwareTriggerController:
    """Hardware trigger synchronization controller"""

    def __init__(self, num_cameras: int = 20):
        self.num_cameras = num_cameras
        self.trigger_rate = 30.0  # Hz
        self.trigger_counter = 0
        self.trigger_map = {}  # Maps trigger_id to expected camera responses
        self.missed_triggers = defaultdict(int)
        self.lock = threading.Lock()
        self.running = False
        self.trigger_thread = None

    def start_hardware_trigger(self, rate: float = 30.0):
        """Start hardware trigger generation"""
        self.trigger_rate = rate
        self.running = True
        self.trigger_thread = threading.Thread(target=self._trigger_loop, daemon=True)
        self.trigger_thread.start()
        logging.info(f"Hardware trigger started at {rate} Hz")

    def stop_hardware_trigger(self):
        """Stop hardware trigger generation"""
        self.running = False
        if self.trigger_thread:
            self.trigger_thread.join(timeout=2.0)
        logging.info("Hardware trigger stopped")

    def _trigger_loop(self):
        """Main trigger generation loop"""
        period = 1.0 / self.trigger_rate
        next_trigger_time = time.time()

        while self.running:
            current_time = time.time()

            if current_time >= next_trigger_time:
                with self.lock:
                    self.trigger_counter += 1
                    trigger_id = self.trigger_counter
                    self.trigger_map[trigger_id] = {
                        'timestamp': current_time,
                        'expected_cameras': set(range(self.num_cameras)),
                        'received_cameras': set()
                    }

                # In real implementation, send hardware trigger signal here
                # GPIO/DAQ card trigger pulse

                next_trigger_time += period

            # Sleep for a fraction of the period to maintain timing
            sleep_time = max(0, (next_trigger_time - time.time()) * 0.9)
            time.sleep(sleep_time)

    def register_trigger_response(self, camera_id: int, trigger_id: int) -> bool:
        """Register that a camera responded to a trigger"""
        with self.lock:
            if trigger_id in self.trigger_map:
                self.trigger_map[trigger_id]['received_cameras'].add(camera_id)
                return True
            else:
                self.missed_triggers[camera_id] += 1
                return False

    def get_trigger_stats(self) -> Dict:
        """Get trigger synchronization statistics"""
        with self.lock:
            total_triggers = len(self.trigger_map)
            if total_triggers == 0:
                return {'total_triggers': 0}

            # Calculate response rates
            response_rates = {}
            for trigger_id, data in list(self.trigger_map.items())[-100:]:  # Last 100 triggers
                response_rate = len(data['received_cameras']) / len(data['expected_cameras'])
                response_rates[trigger_id] = response_rate

            return {
                'total_triggers': self.trigger_counter,
                'avg_response_rate': np.mean(list(response_rates.values())) if response_rates else 0,
                'missed_by_camera': dict(self.missed_triggers),
                'trigger_rate': self.trigger_rate
            }


class FrameBuffer:
    """Thread-safe circular buffer for camera frames"""

    def __init__(self, camera_id: int, buffer_size: int = 100):
        self.camera_id = camera_id
        self.buffer_size = buffer_size
        self.frames = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.last_frame_number = -1

    def add_frame(self, metadata: FrameMetadata):
        """Add frame to buffer"""
        with self.lock:
            # Detect dropped frames
            if self.last_frame_number >= 0:
                expected = self.last_frame_number + 1
                if metadata.frame_number > expected:
                    logging.warning(
                        f"Camera {self.camera_id}: Dropped {metadata.frame_number - expected} frames"
                    )

            self.frames.append(metadata)
            self.last_frame_number = metadata.frame_number

    def get_frame_by_timestamp(self, target_time: float, tolerance: float = 0.01) -> Optional[FrameMetadata]:
        """Get frame closest to target timestamp within tolerance"""
        with self.lock:
            if not self.frames:
                return None

            # Binary search for closest timestamp
            timestamps = [f.timestamp for f in self.frames]
            idx = np.searchsorted(timestamps, target_time)

            # Check both neighbors
            candidates = []
            if idx > 0:
                candidates.append((idx - 1, abs(timestamps[idx - 1] - target_time)))
            if idx < len(timestamps):
                candidates.append((idx, abs(timestamps[idx] - target_time)))

            if not candidates:
                return None

            best_idx, best_diff = min(candidates, key=lambda x: x[1])

            if best_diff <= tolerance:
                return self.frames[best_idx]
            return None

    def get_frame_by_trigger(self, trigger_id: int) -> Optional[FrameMetadata]:
        """Get frame by trigger ID"""
        with self.lock:
            for frame in reversed(self.frames):
                if frame.trigger_id == trigger_id:
                    return frame
            return None

    def get_latest_frame(self) -> Optional[FrameMetadata]:
        """Get most recent frame"""
        with self.lock:
            return self.frames[-1] if self.frames else None


class CameraSynchronizer:
    """Main synchronization manager for 20 cameras (10 pairs)"""

    def __init__(self, num_pairs: int = 10, sync_mode: SyncMode = SyncMode.HYBRID):
        self.num_pairs = num_pairs
        self.num_cameras = num_pairs * 2
        self.sync_mode = sync_mode

        # Frame buffers for each camera
        self.frame_buffers: Dict[int, FrameBuffer] = {
            i: FrameBuffer(i, buffer_size=100) for i in range(self.num_cameras)
        }

        # Synchronization components
        self.ptp_manager = PTPManager()
        self.hw_trigger = HardwareTriggerController(self.num_cameras)

        # Statistics tracking
        self.sync_stats: Dict[int, SyncStatistics] = {
            i: SyncStatistics(camera_id=i) for i in range(self.num_cameras)
        }

        # Synchronized frame output queue
        self.synced_frames = queue.Queue(maxsize=1000)

        # Pair mapping (camera_id -> pair_id, is_thermal)
        self.pair_mapping = {}
        for pair_id in range(num_pairs):
            mono_id = pair_id * 2
            thermal_id = pair_id * 2 + 1
            self.pair_mapping[mono_id] = (pair_id, False)
            self.pair_mapping[thermal_id] = (pair_id, True)

        # Synchronization parameters
        self.sync_tolerance = 0.001  # 1ms tolerance
        self.max_frame_age = 0.1  # 100ms max age

        # Worker thread
        self.running = False
        self.sync_thread = None

        logging.info(f"CameraSynchronizer initialized: {num_pairs} pairs, mode={sync_mode.name}")

    def start(self):
        """Start synchronization system"""
        self.running = True

        # Start hardware trigger if needed
        if self.sync_mode in [SyncMode.HARDWARE, SyncMode.HYBRID]:
            self.hw_trigger.start_hardware_trigger(rate=30.0)

        # Start synchronization worker
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()

        logging.info("CameraSynchronizer started")

    def stop(self):
        """Stop synchronization system"""
        self.running = False

        if self.sync_mode in [SyncMode.HARDWARE, SyncMode.HYBRID]:
            self.hw_trigger.stop_hardware_trigger()

        if self.sync_thread:
            self.sync_thread.join(timeout=2.0)

        logging.info("CameraSynchronizer stopped")

    def add_frame(self, camera_id: int, metadata: FrameMetadata):
        """Add new frame from camera"""
        if camera_id >= self.num_cameras:
            logging.error(f"Invalid camera_id: {camera_id}")
            return

        # Apply PTP timestamp if available
        if self.sync_mode in [SyncMode.PTP, SyncMode.HYBRID] and self.ptp_manager.is_synchronized:
            metadata.ptp_time = self.ptp_manager.get_ptp_time()

        # Register trigger response
        if metadata.trigger_id is not None:
            self.hw_trigger.register_trigger_response(camera_id, metadata.trigger_id)

        # Add to buffer
        self.frame_buffers[camera_id].add_frame(metadata)

    def _sync_worker(self):
        """Worker thread for frame synchronization"""
        last_sync_times = {pair_id: 0.0 for pair_id in range(self.num_pairs)}

        while self.running:
            current_time = time.time()

            # Process each camera pair
            for pair_id in range(self.num_pairs):
                mono_id = pair_id * 2
                thermal_id = pair_id * 2 + 1

                # Try to synchronize pair
                synced_set = self._synchronize_pair(mono_id, thermal_id, pair_id)

                if synced_set and synced_set.is_valid:
                    # Check if this is a new frame set
                    if synced_set.mono_frame.timestamp > last_sync_times[pair_id]:
                        self.synced_frames.put(synced_set)
                        last_sync_times[pair_id] = synced_set.mono_frame.timestamp

                        # Update statistics
                        self._update_statistics(synced_set)

            # Sleep to avoid busy-waiting
            time.sleep(0.001)  # 1ms sleep

    def _synchronize_pair(self, mono_id: int, thermal_id: int, pair_id: int) -> Optional[SyncedFrameSet]:
        """Synchronize mono and thermal cameras in a pair"""
        mono_buffer = self.frame_buffers[mono_id]
        thermal_buffer = self.frame_buffers[thermal_id]

        # Get latest frames
        mono_frame = mono_buffer.get_latest_frame()
        thermal_frame = thermal_buffer.get_latest_frame()

        if not mono_frame or not thermal_frame:
            return None

        # Check frame age
        current_time = time.time()
        if (current_time - mono_frame.system_time > self.max_frame_age or
            current_time - thermal_frame.system_time > self.max_frame_age):
            return None

        # Synchronization strategy based on mode
        if self.sync_mode == SyncMode.HARDWARE or self.sync_mode == SyncMode.HYBRID:
            # Use hardware trigger IDs
            if mono_frame.trigger_id == thermal_frame.trigger_id:
                sync_error = abs(mono_frame.timestamp - thermal_frame.timestamp) * 1000  # ms
                return SyncedFrameSet(
                    pair_id=pair_id,
                    mono_frame=mono_frame,
                    thermal_frame=thermal_frame,
                    sync_error=sync_error,
                    is_valid=sync_error < 10.0  # < 10ms
                )

        # Use timestamp-based synchronization
        reference_time = mono_frame.ptp_time if mono_frame.ptp_time else mono_frame.timestamp

        # Find matching thermal frame
        matching_thermal = thermal_buffer.get_frame_by_timestamp(
            reference_time, tolerance=self.sync_tolerance
        )

        if matching_thermal:
            sync_error = abs(mono_frame.timestamp - matching_thermal.timestamp) * 1000  # ms
            return SyncedFrameSet(
                pair_id=pair_id,
                mono_frame=mono_frame,
                thermal_frame=matching_thermal,
                sync_error=sync_error,
                is_valid=sync_error < 10.0
            )

        # Try frame recovery
        recovered_set = self._recover_dropped_frame(mono_id, thermal_id, pair_id)
        if recovered_set:
            recovered_set.recovery_applied = True
            self.sync_stats[mono_id].frames_recovered += 1
            self.sync_stats[thermal_id].frames_recovered += 1

        return recovered_set

    def _recover_dropped_frame(self, mono_id: int, thermal_id: int, pair_id: int) -> Optional[SyncedFrameSet]:
        """Attempt to recover from dropped frames"""
        mono_buffer = self.frame_buffers[mono_id]
        thermal_buffer = self.frame_buffers[thermal_id]

        # Get recent frames from both cameras
        mono_frame = mono_buffer.get_latest_frame()
        thermal_frame = thermal_buffer.get_latest_frame()

        if not mono_frame or not thermal_frame:
            return None

        # Look for nearest frames within extended tolerance
        extended_tolerance = self.sync_tolerance * 5  # 5ms

        mono_time = mono_frame.ptp_time if mono_frame.ptp_time else mono_frame.timestamp
        matching_thermal = thermal_buffer.get_frame_by_timestamp(mono_time, extended_tolerance)

        if matching_thermal:
            sync_error = abs(mono_frame.timestamp - matching_thermal.timestamp) * 1000

            # Mark as valid if within extended tolerance
            is_valid = sync_error < 50.0  # 50ms for recovery

            if is_valid:
                logging.info(f"Pair {pair_id}: Recovered frame with {sync_error:.2f}ms sync error")
                return SyncedFrameSet(
                    pair_id=pair_id,
                    mono_frame=mono_frame,
                    thermal_frame=matching_thermal,
                    sync_error=sync_error,
                    is_valid=is_valid
                )

        return None

    def _update_statistics(self, synced_set: SyncedFrameSet):
        """Update synchronization statistics"""
        mono_id = synced_set.pair_id * 2
        thermal_id = synced_set.pair_id * 2 + 1

        for camera_id in [mono_id, thermal_id]:
            stats = self.sync_stats[camera_id]
            stats.frames_processed += 1

            # Update sync error statistics
            error = synced_set.sync_error
            stats.max_sync_error = max(stats.max_sync_error, error)
            stats.min_sync_error = min(stats.min_sync_error, error)

            # Running average
            alpha = 0.1  # Smoothing factor
            stats.avg_sync_error = alpha * error + (1 - alpha) * stats.avg_sync_error

            # Calculate jitter (timestamp variance)
            if camera_id == mono_id:
                frame = synced_set.mono_frame
            else:
                frame = synced_set.thermal_frame

            expected_time = stats.last_update + (1.0 / 30.0)  # Assuming 30 FPS
            jitter = abs(frame.timestamp - expected_time) * 1000000  # microseconds
            stats.jitter = alpha * jitter + (1 - alpha) * stats.jitter

            stats.last_update = frame.timestamp

    def get_synced_frame_set(self, timeout: float = 1.0) -> Optional[SyncedFrameSet]:
        """Get next synchronized frame set"""
        try:
            return self.synced_frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_synchronization_metrics(self) -> Dict:
        """Get comprehensive synchronization metrics"""
        metrics = {
            'sync_mode': self.sync_mode.name,
            'num_pairs': self.num_pairs,
            'ptp_quality': self.ptp_manager.get_sync_quality(),
            'trigger_stats': self.hw_trigger.get_trigger_stats(),
            'camera_stats': {}
        }

        for camera_id, stats in self.sync_stats.items():
            pair_id, is_thermal = self.pair_mapping[camera_id]
            cam_type = 'thermal' if is_thermal else 'mono'

            metrics['camera_stats'][camera_id] = {
                'pair_id': pair_id,
                'type': cam_type,
                'avg_sync_error_ms': stats.avg_sync_error,
                'max_sync_error_ms': stats.max_sync_error,
                'min_sync_error_ms': stats.min_sync_error,
                'jitter_us': stats.jitter,
                'frames_processed': stats.frames_processed,
                'frames_dropped': stats.frames_dropped,
                'frames_recovered': stats.frames_recovered,
                'drop_rate': stats.frames_dropped / max(stats.frames_processed, 1)
            }

        # Calculate overall statistics
        all_errors = [s.avg_sync_error for s in self.sync_stats.values() if s.frames_processed > 0]
        all_jitters = [s.jitter for s in self.sync_stats.values() if s.frames_processed > 0]

        metrics['overall'] = {
            'avg_sync_error_ms': np.mean(all_errors) if all_errors else 0.0,
            'max_sync_error_ms': max([s.max_sync_error for s in self.sync_stats.values()]),
            'avg_jitter_us': np.mean(all_jitters) if all_jitters else 0.0,
            'total_frames': sum(s.frames_processed for s in self.sync_stats.values()),
            'total_dropped': sum(s.frames_dropped for s in self.sync_stats.values()),
            'total_recovered': sum(s.frames_recovered for s in self.sync_stats.values())
        }

        return metrics

    def update_ptp_sync(self, master_timestamp: float, local_timestamp: float):
        """Update PTP synchronization from master clock"""
        self.ptp_manager.update_master_offset(master_timestamp, local_timestamp)

    def set_sync_tolerance(self, tolerance_ms: float):
        """Set synchronization tolerance in milliseconds"""
        self.sync_tolerance = tolerance_ms / 1000.0
        logging.info(f"Sync tolerance set to {tolerance_ms}ms")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
