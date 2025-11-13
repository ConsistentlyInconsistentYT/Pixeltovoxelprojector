"""
8K Video Processor with Hardware Acceleration and Real-time Motion Extraction
Supports dual camera streams (monochrome + thermal) with frame synchronization
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Event, Lock
import time
from collections import deque

from codec_handler import CodecHandler, VideoCodec, HardwareAccel


@dataclass
class VideoStream:
    """Configuration for a video stream"""
    path: str
    stream_type: str  # 'monochrome' or 'thermal'
    codec: Optional[VideoCodec] = None
    hw_accel: bool = True
    buffer_size: int = 30  # Frame buffer size


@dataclass
class FrameData:
    """Frame data with metadata"""
    frame: np.ndarray
    frame_number: int
    timestamp: float
    stream_type: str
    width: int
    height: int


@dataclass
class MotionData:
    """Motion detection results"""
    coordinates: List[Tuple[int, int]]  # (x, y) coordinates
    bounding_boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    velocities: List[Tuple[float, float]]  # (vx, vy) velocities
    confidence: List[float]
    timestamp: float
    frame_number: int


class FrameBuffer:
    """Thread-safe circular frame buffer for video streams"""

    def __init__(self, max_size: int = 30):
        """
        Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to buffer
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = Lock()
        self.not_empty = Event()

    def push(self, frame_data: FrameData):
        """Add frame to buffer"""
        with self.lock:
            self.buffer.append(frame_data)
            self.not_empty.set()

    def pop(self, timeout: float = 1.0) -> Optional[FrameData]:
        """Get frame from buffer (blocking with timeout)"""
        if not self.not_empty.wait(timeout):
            return None

        with self.lock:
            if len(self.buffer) == 0:
                self.not_empty.clear()
                return None

            frame_data = self.buffer.popleft()

            if len(self.buffer) == 0:
                self.not_empty.clear()

            return frame_data

    def peek(self) -> Optional[FrameData]:
        """Peek at next frame without removing it"""
        with self.lock:
            return self.buffer[0] if len(self.buffer) > 0 else None

    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

    def clear(self):
        """Clear all frames from buffer"""
        with self.lock:
            self.buffer.clear()
            self.not_empty.clear()


class VideoProcessor:
    """
    High-performance 8K video processor with hardware acceleration.
    Supports real-time motion extraction and dual camera streams.
    """

    def __init__(
        self,
        use_hardware_accel: bool = True,
        gpu_device_id: int = 0,
        target_fps: float = 30.0,
        enable_profiling: bool = True
    ):
        """
        Initialize video processor.

        Args:
            use_hardware_accel: Enable hardware acceleration (NVDEC)
            gpu_device_id: GPU device ID for multi-GPU systems
            target_fps: Target processing FPS
            enable_profiling: Enable performance profiling
        """
        self.use_hardware_accel = use_hardware_accel
        self.gpu_device_id = gpu_device_id
        self.target_fps = target_fps
        self.enable_profiling = enable_profiling

        # Initialize codec handlers
        self.codec_handler = CodecHandler(
            use_hardware_accel=use_hardware_accel,
            preferred_backend=HardwareAccel.NVDEC,
            gpu_device_id=gpu_device_id
        )

        # Video streams
        self.streams: Dict[str, Any] = {}
        self.frame_buffers: Dict[str, FrameBuffer] = {}

        # Processing state
        self.is_processing = False
        self.processing_threads: List[Thread] = []
        self.stop_event = Event()

        # Motion extraction
        self.motion_extractor = None
        self.motion_callbacks: List[Callable] = []

        # Performance metrics
        self.metrics = {
            'frames_processed': 0,
            'frames_dropped': 0,
            'avg_latency_ms': 0.0,
            'current_fps': 0.0,
            'decode_time_ms': 0.0,
            'motion_extract_time_ms': 0.0
        }
        self.latency_buffer = deque(maxlen=100)

        print("[VideoProcessor] Initialized with 8K support")
        print(f"[VideoProcessor] HW Acceleration: {use_hardware_accel}")
        print(f"[VideoProcessor] Target FPS: {target_fps}")

    def add_stream(
        self,
        stream_id: str,
        stream_config: VideoStream
    ) -> bool:
        """
        Add video stream for processing.

        Args:
            stream_id: Unique identifier for stream
            stream_config: Stream configuration

        Returns:
            Success status
        """
        success, cap = self.codec_handler.open_video(
            stream_config.path,
            stream_config.codec
        )

        if not success:
            print(f"[VideoProcessor] Failed to open stream: {stream_id}")
            return False

        # Verify 8K resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width != 7680 or height != 4320:
            print(f"[VideoProcessor] Warning: Non-8K resolution ({width}x{height})")

        self.streams[stream_id] = {
            'capture': cap,
            'config': stream_config,
            'width': width,
            'height': height,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': 0
        }

        self.frame_buffers[stream_id] = FrameBuffer(max_size=stream_config.buffer_size)

        print(f"[VideoProcessor] Added stream '{stream_id}': {width}x{height} @ {self.streams[stream_id]['fps']} FPS")

        return True

    def start_processing(self):
        """Start video processing threads"""
        if self.is_processing:
            print("[VideoProcessor] Already processing")
            return

        self.is_processing = True
        self.stop_event.clear()

        # Start decoding thread for each stream
        for stream_id in self.streams.keys():
            thread = Thread(
                target=self._decode_stream,
                args=(stream_id,),
                daemon=True,
                name=f"Decoder-{stream_id}"
            )
            thread.start()
            self.processing_threads.append(thread)

        # Start motion extraction thread
        motion_thread = Thread(
            target=self._motion_extraction_loop,
            daemon=True,
            name="MotionExtractor"
        )
        motion_thread.start()
        self.processing_threads.append(motion_thread)

        # Start metrics thread
        if self.enable_profiling:
            metrics_thread = Thread(
                target=self._metrics_loop,
                daemon=True,
                name="Metrics"
            )
            metrics_thread.start()
            self.processing_threads.append(metrics_thread)

        print(f"[VideoProcessor] Started processing with {len(self.processing_threads)} threads")

    def stop_processing(self):
        """Stop video processing"""
        print("[VideoProcessor] Stopping processing...")
        self.is_processing = False
        self.stop_event.set()

        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=2.0)

        self.processing_threads.clear()

        # Release video captures
        for stream_id, stream_data in self.streams.items():
            if 'capture' in stream_data:
                stream_data['capture'].release()

        print("[VideoProcessor] Processing stopped")

    def _decode_stream(self, stream_id: str):
        """
        Decode video stream in separate thread.

        Args:
            stream_id: Stream identifier
        """
        stream_data = self.streams[stream_id]
        cap = stream_data['capture']
        config = stream_data['config']
        frame_buffer = self.frame_buffers[stream_id]

        frame_num = 0
        target_frame_time = 1.0 / self.target_fps

        print(f"[Decoder-{stream_id}] Started decoding thread")

        while self.is_processing and not self.stop_event.is_set():
            start_time = time.perf_counter()

            # Decode frame
            ret, frame = self.codec_handler.decode_frame_gpu(cap)

            if not ret:
                # End of stream or error
                print(f"[Decoder-{stream_id}] End of stream")
                break

            decode_time = time.perf_counter() - start_time

            # Create frame data
            frame_data = FrameData(
                frame=frame,
                frame_number=frame_num,
                timestamp=time.perf_counter(),
                stream_type=config.stream_type,
                width=stream_data['width'],
                height=stream_data['height']
            )

            # Add to buffer
            frame_buffer.push(frame_data)

            # Update metrics
            self.metrics['decode_time_ms'] = decode_time * 1000
            stream_data['frame_count'] = frame_num

            frame_num += 1

            # Frame rate limiting
            elapsed = time.perf_counter() - start_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

        print(f"[Decoder-{stream_id}] Thread stopped ({frame_num} frames)")

    def _motion_extraction_loop(self):
        """Motion extraction processing loop"""
        print("[MotionExtractor] Started motion extraction thread")

        # Try to import C++ motion extractor
        try:
            from motion_extractor_cpp import MotionExtractor8K
            self.motion_extractor = MotionExtractor8K()
            print("[MotionExtractor] Using C++ accelerated motion extraction")
        except ImportError:
            print("[MotionExtractor] Warning: C++ module not found, using Python fallback")
            self.motion_extractor = None

        frame_count = 0

        while self.is_processing and not self.stop_event.is_set():
            # Get frames from all streams
            frames = {}
            for stream_id, buffer in self.frame_buffers.items():
                frame_data = buffer.pop(timeout=0.1)
                if frame_data:
                    frames[stream_id] = frame_data

            if not frames:
                continue

            start_time = time.perf_counter()

            # Extract motion from frames
            motion_data = self._extract_motion(frames)

            motion_time = (time.perf_counter() - start_time) * 1000
            self.metrics['motion_extract_time_ms'] = motion_time

            # Calculate total latency
            if frames:
                first_frame = next(iter(frames.values()))
                latency = (time.perf_counter() - first_frame.timestamp) * 1000
                self.latency_buffer.append(latency)
                self.metrics['avg_latency_ms'] = np.mean(self.latency_buffer)

            # Call motion callbacks
            for callback in self.motion_callbacks:
                try:
                    callback(motion_data)
                except Exception as e:
                    print(f"[MotionExtractor] Callback error: {e}")

            frame_count += 1
            self.metrics['frames_processed'] = frame_count

        print(f"[MotionExtractor] Thread stopped ({frame_count} frames)")

    def _extract_motion(self, frames: Dict[str, FrameData]) -> MotionData:
        """
        Extract motion from frame data.

        Args:
            frames: Dictionary of frame data from each stream

        Returns:
            Motion data
        """
        if self.motion_extractor:
            # Use C++ motion extractor
            frame_list = [fd.frame for fd in frames.values()]
            return self.motion_extractor.extract(frame_list)
        else:
            # Python fallback (basic motion detection)
            return self._extract_motion_python(frames)

    def _extract_motion_python(self, frames: Dict[str, FrameData]) -> MotionData:
        """
        Python fallback for motion extraction.

        Args:
            frames: Dictionary of frame data

        Returns:
            Motion data
        """
        coordinates = []
        bounding_boxes = []
        velocities = []
        confidence = []

        # Simple motion detection using frame differencing
        for stream_id, frame_data in frames.items():
            # For demonstration - actual implementation would be more sophisticated
            # This is a placeholder for the C++ implementation
            pass

        return MotionData(
            coordinates=coordinates,
            bounding_boxes=bounding_boxes,
            velocities=velocities,
            confidence=confidence,
            timestamp=time.perf_counter(),
            frame_number=self.metrics['frames_processed']
        )

    def _metrics_loop(self):
        """Performance metrics monitoring loop"""
        print("[Metrics] Started metrics thread")

        last_frame_count = 0
        last_time = time.perf_counter()

        while self.is_processing and not self.stop_event.is_set():
            time.sleep(1.0)  # Update metrics every second

            current_time = time.perf_counter()
            current_frames = self.metrics['frames_processed']

            # Calculate FPS
            elapsed = current_time - last_time
            frames_delta = current_frames - last_frame_count

            if elapsed > 0:
                self.metrics['current_fps'] = frames_delta / elapsed

            last_time = current_time
            last_frame_count = current_frames

            # Print metrics
            if self.enable_profiling:
                self._print_metrics()

        print("[Metrics] Thread stopped")

    def _print_metrics(self):
        """Print current performance metrics"""
        print("\n" + "=" * 80)
        print("8K VIDEO PROCESSOR - PERFORMANCE METRICS")
        print("=" * 80)
        print(f"Frames Processed:    {self.metrics['frames_processed']}")
        print(f"Frames Dropped:      {self.metrics['frames_dropped']}")
        print(f"Current FPS:         {self.metrics['current_fps']:.2f}")
        print(f"Avg Latency:         {self.metrics['avg_latency_ms']:.2f} ms")
        print(f"Decode Time:         {self.metrics['decode_time_ms']:.2f} ms")
        print(f"Motion Extract Time: {self.metrics['motion_extract_time_ms']:.2f} ms")

        # Stream-specific metrics
        for stream_id, stream_data in self.streams.items():
            buffer_size = self.frame_buffers[stream_id].size()
            print(f"Stream '{stream_id}': Frame {stream_data['frame_count']}, Buffer: {buffer_size}")

        print("=" * 80)

    def register_motion_callback(self, callback: Callable[[MotionData], None]):
        """
        Register callback for motion detection events.

        Args:
            callback: Function to call with motion data
        """
        self.motion_callbacks.append(callback)

    def get_frame(self, stream_id: str, timeout: float = 1.0) -> Optional[FrameData]:
        """
        Get frame from stream buffer.

        Args:
            stream_id: Stream identifier
            timeout: Timeout in seconds

        Returns:
            Frame data or None
        """
        if stream_id not in self.frame_buffers:
            return None

        return self.frame_buffers[stream_id].pop(timeout)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()

    def synchronize_streams(
        self,
        stream_ids: List[str],
        max_time_diff_ms: float = 10.0
    ) -> Optional[Dict[str, FrameData]]:
        """
        Get synchronized frames from multiple streams.

        Args:
            stream_ids: List of stream IDs to synchronize
            max_time_diff_ms: Maximum allowed time difference in ms

        Returns:
            Dictionary of synchronized frame data or None
        """
        frames = {}

        # Peek at next frame from each stream
        for stream_id in stream_ids:
            if stream_id not in self.frame_buffers:
                return None

            frame_data = self.frame_buffers[stream_id].peek()
            if frame_data is None:
                return None

            frames[stream_id] = frame_data

        # Check timestamp synchronization
        timestamps = [fd.timestamp for fd in frames.values()]
        time_diff = (max(timestamps) - min(timestamps)) * 1000

        if time_diff > max_time_diff_ms:
            # Frames not synchronized, drop oldest
            oldest_id = min(frames.keys(), key=lambda k: frames[k].timestamp)
            self.frame_buffers[oldest_id].pop(timeout=0)
            self.metrics['frames_dropped'] += 1
            return None

        # Frames are synchronized, pop all
        synced_frames = {}
        for stream_id in stream_ids:
            synced_frames[stream_id] = self.frame_buffers[stream_id].pop(timeout=0)

        return synced_frames


# Example usage and testing
if __name__ == '__main__':
    print("8K Video Processor - Test Suite")
    print("=" * 80)

    # Create processor
    processor = VideoProcessor(
        use_hardware_accel=True,
        target_fps=30.0,
        enable_profiling=True
    )

    # Example: Add dual camera streams
    # processor.add_stream(
    #     'mono_cam',
    #     VideoStream(
    #         path='path/to/monochrome_8k.mp4',
    #         stream_type='monochrome',
    #         codec=VideoCodec.HEVC
    #     )
    # )
    #
    # processor.add_stream(
    #     'thermal_cam',
    #     VideoStream(
    #         path='path/to/thermal_8k.raw',
    #         stream_type='thermal',
    #         codec=VideoCodec.RAW_THERMAL
    #     )
    # )

    # Register motion callback
    def motion_callback(motion_data: MotionData):
        print(f"Motion detected: {len(motion_data.coordinates)} points")

    processor.register_motion_callback(motion_callback)

    # Start processing
    # processor.start_processing()

    # Process for some time
    # time.sleep(10)

    # Stop processing
    # processor.stop_processing()

    print("\nTest complete")
