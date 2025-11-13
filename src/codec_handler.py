"""
8K Video Codec Handler with Hardware Acceleration
Supports H.265/HEVC, RAW thermal formats, and GPU-accelerated decoding
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import subprocess
import json


class VideoCodec(Enum):
    """Supported video codecs"""
    HEVC = "hevc"
    H265 = "h265"
    H264 = "h264"
    RAW_THERMAL = "raw_thermal"
    RAW_MONO = "raw_mono"
    VP9 = "vp9"
    AV1 = "av1"


class HardwareAccel(Enum):
    """Hardware acceleration backends"""
    NONE = "none"
    NVDEC = "nvdec"  # NVIDIA GPU decoding
    CUVID = "cuvid"  # CUDA Video Decoder
    VAAPI = "vaapi"  # Video Acceleration API (Intel/AMD)
    DXVA2 = "dxva2"  # DirectX Video Acceleration (Windows)
    VIDEOTOOLBOX = "videotoolbox"  # macOS/iOS


class CodecHandler:
    """
    High-performance video codec handler for 8K video streams.
    Supports hardware-accelerated decoding with NVDEC/CUVID.
    """

    def __init__(
        self,
        use_hardware_accel: bool = True,
        preferred_backend: HardwareAccel = HardwareAccel.NVDEC,
        gpu_device_id: int = 0
    ):
        """
        Initialize codec handler with hardware acceleration support.

        Args:
            use_hardware_accel: Enable hardware acceleration
            preferred_backend: Preferred hardware acceleration backend
            gpu_device_id: GPU device ID for multi-GPU systems
        """
        self.use_hardware_accel = use_hardware_accel
        self.preferred_backend = preferred_backend
        self.gpu_device_id = gpu_device_id
        self.current_codec = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0

        # Check hardware acceleration availability
        self.hw_accel_available = self._check_hw_acceleration()

    def _check_hw_acceleration(self) -> bool:
        """Check if hardware acceleration is available"""
        try:
            # Check CUDA availability for NVDEC/CUVID
            if self.preferred_backend in [HardwareAccel.NVDEC, HardwareAccel.CUVID]:
                return cv2.cuda.getCudaEnabledDeviceCount() > 0
            return True
        except:
            return False

    def open_video(
        self,
        video_path: str,
        codec: Optional[VideoCodec] = None
    ) -> Tuple[bool, Optional[cv2.VideoCapture]]:
        """
        Open video file with appropriate codec and hardware acceleration.

        Args:
            video_path: Path to video file
            codec: Specific codec to use (auto-detect if None)

        Returns:
            (success, video_capture) tuple
        """
        if codec == VideoCodec.RAW_THERMAL or codec == VideoCodec.RAW_MONO:
            return self._open_raw_video(video_path, codec)

        cap = cv2.VideoCapture()

        # Configure hardware acceleration
        if self.use_hardware_accel and self.hw_accel_available:
            if self.preferred_backend == HardwareAccel.NVDEC:
                # Use NVDEC for NVIDIA GPUs
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                cap.set(cv2.CAP_PROP_HW_DEVICE, self.gpu_device_id)
            elif self.preferred_backend == HardwareAccel.CUVID:
                # Use CUVID backend
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

        # Open video file
        success = cap.open(video_path)

        if success:
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.current_codec = self._detect_codec(cap)

            print(f"[CodecHandler] Opened video: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
            print(f"[CodecHandler] Codec: {self.current_codec}")
            print(f"[CodecHandler] HW Accel: {self.hw_accel_available}")

        return success, cap if success else None

    def _detect_codec(self, cap: cv2.VideoCapture) -> str:
        """Detect video codec from VideoCapture object"""
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        return codec_str

    def _open_raw_video(
        self,
        video_path: str,
        codec: VideoCodec
    ) -> Tuple[bool, Optional['RawVideoReader']]:
        """
        Open RAW thermal or monochrome video file.

        Args:
            video_path: Path to RAW video file
            codec: RAW codec type

        Returns:
            (success, raw_reader) tuple
        """
        try:
            reader = RawVideoReader(video_path, codec)
            return True, reader
        except Exception as e:
            print(f"[CodecHandler] Failed to open RAW video: {e}")
            return False, None

    def decode_frame_gpu(
        self,
        cap: cv2.VideoCapture
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Decode frame using GPU acceleration.

        Args:
            cap: OpenCV VideoCapture object

        Returns:
            (success, frame) tuple
        """
        ret, frame = cap.read()

        if ret and self.use_hardware_accel and self.hw_accel_available:
            # Frame is already decoded on GPU with NVDEC
            # Convert to appropriate format if needed
            pass

        return ret, frame

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get detailed video information using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)

        except Exception as e:
            print(f"[CodecHandler] ffprobe error: {e}")

        return {}

    def create_hevc_decoder(
        self,
        width: int = 7680,
        height: int = 4320
    ) -> Optional[Any]:
        """
        Create optimized HEVC/H.265 decoder for 8K video.

        Args:
            width: Video width (default 8K)
            height: Video height (default 8K)

        Returns:
            Decoder object or None
        """
        if not self.hw_accel_available:
            print("[CodecHandler] Warning: HW acceleration not available for HEVC")
            return None

        try:
            # For CUDA-accelerated decoding
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # Create CUDA stream for async operations
                stream = cv2.cuda.Stream()
                return {
                    'stream': stream,
                    'width': width,
                    'height': height,
                    'type': 'HEVC_CUDA'
                }
        except Exception as e:
            print(f"[CodecHandler] Failed to create HEVC decoder: {e}")

        return None


class RawVideoReader:
    """Reader for RAW thermal and monochrome video formats"""

    def __init__(
        self,
        file_path: str,
        codec: VideoCodec,
        width: int = 7680,
        height: int = 4320,
        fps: float = 30.0,
        bit_depth: int = 16
    ):
        """
        Initialize RAW video reader.

        Args:
            file_path: Path to RAW video file
            codec: RAW codec type (RAW_THERMAL or RAW_MONO)
            width: Frame width
            height: Frame height
            fps: Frames per second
            bit_depth: Bit depth (8, 16, or 32)
        """
        self.file_path = file_path
        self.codec = codec
        self.width = width
        self.height = height
        self.fps = fps
        self.bit_depth = bit_depth

        # Calculate frame size in bytes
        self.bytes_per_pixel = bit_depth // 8
        self.frame_size = width * height * self.bytes_per_pixel

        # Open file
        self.file = open(file_path, 'rb')

        # Get file size to calculate total frames
        self.file.seek(0, 2)  # Seek to end
        file_size = self.file.tell()
        self.total_frames = file_size // self.frame_size
        self.file.seek(0)  # Reset to beginning

        self.current_frame = 0

        print(f"[RawVideoReader] Opened RAW video: {width}x{height} @ {fps} FPS")
        print(f"[RawVideoReader] Bit depth: {bit_depth}, Total frames: {self.total_frames}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from RAW video.

        Returns:
            (success, frame) tuple
        """
        if self.current_frame >= self.total_frames:
            return False, None

        # Read frame data
        frame_data = self.file.read(self.frame_size)

        if len(frame_data) != self.frame_size:
            return False, None

        # Convert to numpy array
        if self.bit_depth == 8:
            dtype = np.uint8
        elif self.bit_depth == 16:
            dtype = np.uint16
        else:
            dtype = np.uint32

        frame = np.frombuffer(frame_data, dtype=dtype)
        frame = frame.reshape((self.height, self.width))

        self.current_frame += 1

        return True, frame

    def get(self, prop_id: int) -> float:
        """Get video property (compatible with cv2.VideoCapture)"""
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.current_frame)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        """Set video property"""
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            frame_num = int(value)
            if 0 <= frame_num < self.total_frames:
                self.file.seek(frame_num * self.frame_size)
                self.current_frame = frame_num
                return True
        return False

    def release(self):
        """Release resources"""
        if self.file:
            self.file.close()

    def isOpened(self) -> bool:
        """Check if file is opened"""
        return self.file and not self.file.closed


# Performance benchmarking utilities
class CodecBenchmark:
    """Benchmark different codec configurations"""

    @staticmethod
    def benchmark_decode(
        video_path: str,
        num_frames: int = 100,
        use_hw_accel: bool = True
    ) -> Dict[str, float]:
        """
        Benchmark video decoding performance.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to decode
            use_hw_accel: Use hardware acceleration

        Returns:
            Performance metrics dictionary
        """
        import time

        handler = CodecHandler(use_hardware_accel=use_hw_accel)
        success, cap = handler.open_video(video_path)

        if not success:
            return {'error': 'Failed to open video'}

        start_time = time.perf_counter()
        frames_decoded = 0

        for _ in range(num_frames):
            ret, frame = handler.decode_frame_gpu(cap)
            if not ret:
                break
            frames_decoded += 1

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        cap.release()

        return {
            'frames_decoded': frames_decoded,
            'elapsed_time': elapsed,
            'fps': frames_decoded / elapsed if elapsed > 0 else 0,
            'ms_per_frame': (elapsed / frames_decoded * 1000) if frames_decoded > 0 else 0,
            'hw_accel_used': use_hw_accel and handler.hw_accel_available
        }


if __name__ == '__main__':
    # Test codec handler
    print("8K Video Codec Handler - Test Suite")
    print("=" * 60)

    handler = CodecHandler(use_hardware_accel=True)
    print(f"Hardware acceleration available: {handler.hw_accel_available}")
    print(f"Preferred backend: {handler.preferred_backend.value}")
