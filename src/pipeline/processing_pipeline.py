"""
Processing Pipeline - Main data processing pipeline

This module orchestrates the entire processing flow:
1. Camera input management
2. Motion extraction
3. Voxel grid updates
4. Detection and tracking
5. Coordinate streaming

Features:
- Multi-threaded processing
- Frame synchronization across camera pairs
- Real-time motion extraction
- Voxel grid integration
- Network coordinate streaming
"""

import time
import threading
import queue
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

# Import system components
from camera.camera_manager import CameraManager, CameraConfiguration, CameraPair, CameraType, ConnectionType
from voxel.grid_manager import VoxelGridManager, GridConfig
from detection.tracker import MultiTargetTracker
from fusion.fusion_manager import FusionManager, FusionConfig, FusedDetection
from monitoring.system_monitor import SystemMonitor
from protocols.subscriber import MessageType

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for processing pipeline"""
    target_fps: float = 30.0
    enable_fusion: bool = True
    enable_tracking: bool = True
    enable_voxel_updates: bool = True
    enable_streaming: bool = True
    enable_monitoring: bool = True

    # Buffer settings
    frame_buffer_size: int = 30
    processing_queue_size: int = 100

    # Threading
    num_processing_threads: int = 8

    # Performance
    enable_prefetch: bool = True
    enable_parallel_processing: bool = True
    max_latency_ms: float = 100.0


@dataclass
class FrameData:
    """Container for frame data from camera pair"""
    pair_id: int
    mono_frame: np.ndarray
    thermal_frame: np.ndarray
    mono_timestamp: float
    thermal_timestamp: float
    frame_number: int

    @property
    def synchronized(self) -> bool:
        """Check if frames are synchronized"""
        return abs(self.mono_timestamp - self.thermal_timestamp) < 0.01  # 10ms threshold


@dataclass
class ProcessingResult:
    """Result from processing pipeline"""
    pair_id: int
    frame_number: int
    timestamp: float

    # Motion data
    motion_coordinates: List[Tuple[float, float]]
    motion_velocities: List[Tuple[float, float]]

    # Fusion results
    fused_detections: List[FusedDetection]

    # Tracking results
    confirmed_tracks: List[Dict[str, Any]]

    # Voxel updates
    voxel_updates: int

    # Performance metrics
    processing_time_ms: float
    latency_ms: float


class ProcessingPipeline:
    """
    Main processing pipeline for multi-camera motion tracking system

    Orchestrates:
    - Camera frame acquisition
    - Thermal-mono fusion
    - Motion detection
    - Multi-target tracking
    - Voxel grid updates
    - Coordinate streaming
    """

    def __init__(
        self,
        config: PipelineConfig,
        camera_manager: CameraManager,
        voxel_manager: VoxelGridManager,
        fusion_manager: FusionManager,
        tracker: MultiTargetTracker,
        monitor: Optional[SystemMonitor] = None
    ):
        """
        Initialize processing pipeline

        Args:
            config: Pipeline configuration
            camera_manager: Camera management system
            voxel_manager: Voxel grid manager
            fusion_manager: Fusion manager for thermal-mono pairs
            tracker: Multi-target tracker
            monitor: Optional system monitor
        """
        self.config = config
        self.camera_manager = camera_manager
        self.voxel_manager = voxel_manager
        self.fusion_manager = fusion_manager
        self.tracker = tracker
        self.monitor = monitor

        # Processing state
        self.running = False
        self.paused = False
        self.frame_count = 0

        # Queues for pipeline stages
        self.frame_queue = queue.Queue(maxsize=config.processing_queue_size)
        self.fusion_queue = queue.Queue(maxsize=config.processing_queue_size)
        self.tracking_queue = queue.Queue(maxsize=config.processing_queue_size)
        self.output_queue = queue.Queue(maxsize=config.processing_queue_size)

        # Worker threads
        self.processing_threads: List[threading.Thread] = []
        self.fusion_threads: List[threading.Thread] = []
        self.tracking_thread: Optional[threading.Thread] = None
        self.streaming_thread: Optional[threading.Thread] = None

        # Synchronization
        self.lock = threading.Lock()
        self.frame_sync_buffers: Dict[int, List[FrameData]] = defaultdict(list)

        # Performance metrics
        self.total_processing_time = 0.0
        self.total_latency = 0.0
        self.frames_processed = 0
        self.frames_dropped = 0

        # Streaming callbacks
        self.coordinate_callbacks: List[Any] = []

        logger.info("ProcessingPipeline initialized")

    def start(self):
        """Start the processing pipeline"""
        if self.running:
            logger.warning("Pipeline already running")
            return

        logger.info("Starting processing pipeline...")
        self.running = True

        # Start fusion manager
        if self.config.enable_fusion:
            self.fusion_manager.start(num_workers=4)

        # Start voxel grid background processing
        if self.config.enable_voxel_updates:
            self.voxel_manager.start_background_processing()

        # Start processing threads
        for i in range(self.config.num_processing_threads):
            thread = threading.Thread(
                target=self._processing_worker,
                args=(i,),
                daemon=True,
                name=f"ProcessingWorker-{i}"
            )
            thread.start()
            self.processing_threads.append(thread)

        # Start fusion processing threads
        if self.config.enable_fusion:
            for i in range(4):
                thread = threading.Thread(
                    target=self._fusion_worker,
                    args=(i,),
                    daemon=True,
                    name=f"FusionWorker-{i}"
                )
                thread.start()
                self.fusion_threads.append(thread)

        # Start tracking thread
        if self.config.enable_tracking:
            self.tracking_thread = threading.Thread(
                target=self._tracking_worker,
                daemon=True,
                name="TrackingWorker"
            )
            self.tracking_thread.start()

        # Start streaming thread
        if self.config.enable_streaming:
            self.streaming_thread = threading.Thread(
                target=self._streaming_worker,
                daemon=True,
                name="StreamingWorker"
            )
            self.streaming_thread.start()

        # Start camera acquisition
        self.camera_manager.start_all_acquisition()

        logger.info("Processing pipeline started successfully")

    def stop(self):
        """Stop the processing pipeline"""
        if not self.running:
            return

        logger.info("Stopping processing pipeline...")
        self.running = False

        # Stop camera acquisition
        self.camera_manager.stop_all_acquisition()

        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=2.0)

        for thread in self.fusion_threads:
            thread.join(timeout=2.0)

        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)

        if self.streaming_thread:
            self.streaming_thread.join(timeout=2.0)

        # Stop managers
        if self.config.enable_fusion:
            self.fusion_manager.stop()

        if self.config.enable_voxel_updates:
            self.voxel_manager.stop_background_processing()

        # Print final statistics
        self._print_statistics()

        logger.info("Processing pipeline stopped")

    def pause(self):
        """Pause processing"""
        self.paused = True
        logger.info("Pipeline paused")

    def resume(self):
        """Resume processing"""
        self.paused = False
        logger.info("Pipeline resumed")

    def process_camera_pair(self, pair_id: int) -> Optional[ProcessingResult]:
        """
        Process a single camera pair (main entry point)

        Args:
            pair_id: Camera pair ID

        Returns:
            ProcessingResult or None
        """
        start_time = time.time()

        # Get camera pair
        if pair_id not in self.camera_manager.pairs:
            logger.error(f"Camera pair {pair_id} not found")
            return None

        pair = self.camera_manager.pairs[pair_id]

        # Grab frames from both cameras
        mono_camera = self.camera_manager.cameras.get(pair.mono_camera_id)
        thermal_camera = self.camera_manager.cameras.get(pair.thermal_camera_id)

        if not mono_camera or not thermal_camera:
            logger.error(f"Cameras not found for pair {pair_id}")
            return None

        # Grab frames
        mono_result = mono_camera.grab_frame()
        thermal_result = thermal_camera.grab_frame()

        if not mono_result or not thermal_result:
            return None

        mono_frame, mono_metadata = mono_result
        thermal_frame, thermal_metadata = thermal_result

        # Create frame data
        frame_data = FrameData(
            pair_id=pair_id,
            mono_frame=mono_frame,
            thermal_frame=thermal_frame,
            mono_timestamp=mono_metadata['timestamp'],
            thermal_timestamp=thermal_metadata['timestamp'],
            frame_number=self.frame_count
        )

        # Check synchronization
        if not frame_data.synchronized:
            logger.warning(f"Frames not synchronized for pair {pair_id}: "
                         f"dt={abs(frame_data.mono_timestamp - frame_data.thermal_timestamp)*1000:.1f}ms")

        # Queue for processing
        try:
            self.frame_queue.put(frame_data, timeout=0.1)
        except queue.Full:
            logger.warning(f"Frame queue full, dropping frame from pair {pair_id}")
            self.frames_dropped += 1
            return None

        self.frame_count += 1
        return None

    def _processing_worker(self, worker_id: int):
        """Worker thread for initial frame processing"""
        logger.info(f"Processing worker {worker_id} started")

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                # Get frame data
                frame_data = self.frame_queue.get(timeout=0.1)

                # Extract motion from both frames
                mono_motion = self._extract_motion(frame_data.mono_frame)
                thermal_motion = self._extract_motion(frame_data.thermal_frame)

                # Package for fusion
                fusion_data = {
                    'frame_data': frame_data,
                    'mono_motion': mono_motion,
                    'thermal_motion': thermal_motion
                }

                # Queue for fusion
                if self.config.enable_fusion:
                    try:
                        self.fusion_queue.put(fusion_data, timeout=0.1)
                    except queue.Full:
                        logger.warning("Fusion queue full, dropping frame")
                        self.frames_dropped += 1

                self.frame_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker {worker_id} error: {e}", exc_info=True)

    def _fusion_worker(self, worker_id: int):
        """Worker thread for thermal-mono fusion"""
        logger.info(f"Fusion worker {worker_id} started")

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                # Get fusion data
                fusion_data = self.fusion_queue.get(timeout=0.1)
                frame_data = fusion_data['frame_data']

                # Perform fusion
                fused_detections = self.fusion_manager.process_frame_pair(
                    frame_data.pair_id,
                    frame_data.thermal_frame,
                    frame_data.mono_frame,
                    frame_data.mono_timestamp
                )

                if fused_detections is None:
                    fused_detections = []

                # Package for tracking
                tracking_data = {
                    'frame_data': frame_data,
                    'fused_detections': fused_detections,
                    'mono_motion': fusion_data['mono_motion'],
                    'thermal_motion': fusion_data['thermal_motion']
                }

                # Queue for tracking
                if self.config.enable_tracking:
                    try:
                        self.tracking_queue.put(tracking_data, timeout=0.1)
                    except queue.Full:
                        logger.warning("Tracking queue full, dropping frame")
                        self.frames_dropped += 1

                self.fusion_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Fusion worker {worker_id} error: {e}", exc_info=True)

    def _tracking_worker(self):
        """Worker thread for multi-target tracking"""
        logger.info("Tracking worker started")

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                # Get tracking data
                tracking_data = self.tracking_queue.get(timeout=0.1)
                frame_data = tracking_data['frame_data']
                fused_detections = tracking_data['fused_detections']

                # Convert fused detections to tracker format
                detections = []
                for det in fused_detections:
                    detections.append({
                        'x': det.x,
                        'y': det.y,
                        'velocity_x': 0.0,
                        'velocity_y': 0.0,
                        'confidence': det.confidence,
                        'size': det.width
                    })

                # Update tracker
                tracking_result = self.tracker.update(
                    detections,
                    frame_number=frame_data.frame_number,
                    timestamp=frame_data.mono_timestamp
                )

                # Update voxel grid with tracked objects
                if self.config.enable_voxel_updates:
                    self._update_voxel_grid(tracking_result['tracks'])

                # Create processing result
                result = ProcessingResult(
                    pair_id=frame_data.pair_id,
                    frame_number=frame_data.frame_number,
                    timestamp=frame_data.mono_timestamp,
                    motion_coordinates=tracking_data['mono_motion']['coordinates'],
                    motion_velocities=tracking_data['mono_motion']['velocities'],
                    fused_detections=fused_detections,
                    confirmed_tracks=tracking_result['tracks'],
                    voxel_updates=len(tracking_result['tracks']),
                    processing_time_ms=tracking_result['metrics']['latency_ms'],
                    latency_ms=(time.time() - frame_data.mono_timestamp) * 1000
                )

                # Queue for streaming
                try:
                    self.output_queue.put(result, timeout=0.1)
                except queue.Full:
                    logger.warning("Output queue full, dropping result")

                # Update metrics
                with self.lock:
                    self.frames_processed += 1
                    self.total_processing_time += result.processing_time_ms
                    self.total_latency += result.latency_ms

                self.tracking_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Tracking worker error: {e}", exc_info=True)

    def _streaming_worker(self):
        """Worker thread for coordinate streaming"""
        logger.info("Streaming worker started")

        while self.running:
            try:
                # Get result
                result = self.output_queue.get(timeout=0.1)

                # Stream coordinates
                self._stream_coordinates(result)

                self.output_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Streaming worker error: {e}", exc_info=True)

    def _extract_motion(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract motion from frame

        Args:
            frame: Input frame

        Returns:
            Dictionary with motion data
        """
        # Simplified motion extraction
        # In production, would use C++ motion extractor

        # Threshold-based detection
        threshold = 128
        motion_mask = frame > threshold

        # Find connected components (simplified)
        from scipy import ndimage
        labeled, num_features = ndimage.label(motion_mask)

        coordinates = []
        velocities = []
        confidence = []

        for i in range(1, num_features + 1):
            coords = np.where(labeled == i)
            if len(coords[0]) < 10:  # Minimum size
                continue

            y_center = np.mean(coords[0])
            x_center = np.mean(coords[1])

            coordinates.append((x_center, y_center))
            velocities.append((0.0, 0.0))  # Would be calculated from previous frame
            confidence.append(1.0)

        return {
            'coordinates': coordinates,
            'velocities': velocities,
            'confidence': confidence,
            'num_objects': len(coordinates)
        }

    def _update_voxel_grid(self, tracks: List[Dict[str, Any]]):
        """
        Update voxel grid with tracked objects

        Args:
            tracks: List of confirmed tracks
        """
        for track in tracks:
            # Convert 2D pixel coordinates to 3D world coordinates
            # This requires camera calibration and triangulation
            # Simplified version:
            position = np.array([
                track['x'],
                track['y'],
                500.0  # Assumed altitude
            ])

            # Update voxel grid
            self.voxel_manager.add_tracked_object(
                object_id=track['track_id'],
                position=position,
                size=2.0,
                velocity=np.array([track.get('vx', 0), track.get('vy', 0), 0])
            )

    def _stream_coordinates(self, result: ProcessingResult):
        """
        Stream coordinates to network

        Args:
            result: Processing result to stream
        """
        # Call registered callbacks
        for callback in self.coordinate_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Coordinate callback error: {e}")

    def register_coordinate_callback(self, callback):
        """Register callback for coordinate streaming"""
        self.coordinate_callbacks.append(callback)
        logger.info("Registered coordinate callback")

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        with self.lock:
            if self.frames_processed == 0:
                return {
                    'frames_processed': 0,
                    'frames_dropped': self.frames_dropped,
                    'avg_processing_time_ms': 0,
                    'avg_latency_ms': 0,
                    'throughput_fps': 0
                }

            return {
                'frames_processed': self.frames_processed,
                'frames_dropped': self.frames_dropped,
                'avg_processing_time_ms': self.total_processing_time / self.frames_processed,
                'avg_latency_ms': self.total_latency / self.frames_processed,
                'throughput_fps': self.frames_processed / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0,
                'queue_sizes': {
                    'frame_queue': self.frame_queue.qsize(),
                    'fusion_queue': self.fusion_queue.qsize(),
                    'tracking_queue': self.tracking_queue.qsize(),
                    'output_queue': self.output_queue.qsize()
                }
            }

    def _print_statistics(self):
        """Print pipeline statistics"""
        metrics = self.get_metrics()

        logger.info("="*60)
        logger.info("Processing Pipeline Statistics")
        logger.info("="*60)
        logger.info(f"Frames processed: {metrics['frames_processed']}")
        logger.info(f"Frames dropped: {metrics['frames_dropped']}")
        logger.info(f"Avg processing time: {metrics['avg_processing_time_ms']:.2f} ms")
        logger.info(f"Avg latency: {metrics['avg_latency_ms']:.2f} ms")
        logger.info(f"Throughput: {metrics['throughput_fps']:.2f} FPS")
        logger.info("="*60)
