# API Documentation

## Overview

This document provides comprehensive API reference for the 8K Motion Tracking and Voxel Processing System, including Python and C++ interfaces, protocol specifications, and integration examples.

---

## Table of Contents

1. [Python API](#python-api)
2. [C++ API](#cpp-api)
3. [Protocol Specifications](#protocol-specifications)
4. [Integration Examples](#integration-examples)

---

## Python API

### Core Modules

#### 1. Video Processing

##### VideoProcessor

Main class for video stream processing and motion extraction.

```python
class VideoProcessor:
    """
    8K Video processor with hardware acceleration and motion extraction.

    Args:
        use_hardware_accel (bool): Enable hardware-accelerated decoding
        target_fps (float): Target processing frame rate
        enable_profiling (bool): Enable performance profiling
    """

    def __init__(
        self,
        use_hardware_accel: bool = True,
        target_fps: float = 30.0,
        enable_profiling: bool = False
    ):
        pass

    def add_stream(
        self,
        stream_id: str,
        stream: VideoStream
    ) -> bool:
        """
        Add video stream for processing.

        Args:
            stream_id: Unique identifier for the stream
            stream: VideoStream configuration

        Returns:
            True if stream added successfully

        Example:
            >>> processor = VideoProcessor()
            >>> stream = VideoStream(
            ...     path='/path/to/video.mp4',
            ...     stream_type='monochrome',
            ...     codec=VideoCodec.HEVC
            ... )
            >>> processor.add_stream('camera1', stream)
            True
        """
        pass

    def remove_stream(self, stream_id: str) -> bool:
        """Remove video stream from processing."""
        pass

    def start_processing(self) -> bool:
        """
        Start video processing threads.

        Returns:
            True if started successfully
        """
        pass

    def stop_processing(self):
        """Stop all processing threads."""
        pass

    def register_motion_callback(
        self,
        callback: Callable[[MotionData], None]
    ):
        """
        Register callback for motion detection events.

        Args:
            callback: Function called when motion detected

        Example:
            >>> def on_motion(data: MotionData):
            ...     print(f"Detected {len(data.coordinates)} objects")
            >>> processor.register_motion_callback(on_motion)
        """
        pass

    def synchronize_streams(
        self,
        stream_ids: List[str],
        max_time_diff_ms: float = 10.0
    ) -> Optional[Dict[str, Frame]]:
        """
        Get synchronized frames from multiple streams.

        Args:
            stream_ids: List of stream IDs to synchronize
            max_time_diff_ms: Maximum timestamp difference (ms)

        Returns:
            Dict mapping stream_id to Frame, or None if not available

        Example:
            >>> frames = processor.synchronize_streams(
            ...     ['mono_cam', 'thermal_cam'],
            ...     max_time_diff_ms=10.0
            ... )
            >>> if frames:
            ...     mono_frame = frames['mono_cam']
            ...     thermal_frame = frames['thermal_cam']
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get processing metrics.

        Returns:
            Dict with metrics:
            - frames_processed: Total frames processed
            - current_fps: Current processing FPS
            - avg_latency_ms: Average latency (ms)
            - decode_time_ms: Average decode time (ms)
            - motion_extract_time_ms: Average motion extraction time (ms)
            - frames_dropped: Number of dropped frames
        """
        pass
```

##### VideoStream

Configuration for a video stream.

```python
@dataclass
class VideoStream:
    """
    Video stream configuration.

    Attributes:
        path: Path to video file or stream URL
        stream_type: 'monochrome' or 'thermal'
        codec: Video codec (HEVC, H264, etc.)
        buffer_size: Frame buffer size (default: 30)
        width: Frame width (auto-detected if not specified)
        height: Frame height (auto-detected if not specified)
        fps: Frame rate (auto-detected if not specified)
    """
    path: str
    stream_type: str
    codec: VideoCodec = VideoCodec.HEVC
    buffer_size: int = 30
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
```

##### MotionData

Motion detection results.

```python
@dataclass
class MotionData:
    """
    Motion detection results from a frame.

    Attributes:
        frame_number: Frame sequence number
        timestamp: Frame timestamp (seconds)
        coordinates: List of (x, y) centroids
        bounding_boxes: List of (x, y, width, height) boxes
        velocities: List of (vx, vy) velocity vectors
        confidence: List of confidence scores (0-1)
    """
    frame_number: int
    timestamp: float
    coordinates: List[Tuple[int, int]]
    bounding_boxes: List[Tuple[int, int, int, int]]
    velocities: List[Tuple[float, float]]
    confidence: List[float]
```

---

#### 2. Fusion System

##### FusionManager

Multi-modal sensor fusion for thermal and monochrome cameras.

```python
class FusionManager:
    """
    Manages thermal + monochrome camera fusion.

    Args:
        config: Fusion configuration parameters
    """

    def __init__(self, config: FusionConfig):
        pass

    def add_camera_pair(
        self,
        thermal_id: int,
        mono_id: int,
        baseline_m: float,
        calibration_params: Optional[Dict] = None
    ) -> int:
        """
        Add thermal-monochrome camera pair.

        Args:
            thermal_id: Thermal camera ID
            mono_id: Monochrome camera ID
            baseline_m: Physical baseline between cameras (meters)
            calibration_params: Optional calibration parameters

        Returns:
            Pair ID

        Example:
            >>> fusion = FusionManager(FusionConfig())
            >>> pair_id = fusion.add_camera_pair(
            ...     thermal_id=0,
            ...     mono_id=1,
            ...     baseline_m=0.5
            ... )
        """
        pass

    def start(self, num_workers: int = 4):
        """
        Start fusion processing threads.

        Args:
            num_workers: Number of worker threads
        """
        pass

    def stop(self):
        """Stop fusion processing."""
        pass

    def process_frame_pair(
        self,
        pair_id: int,
        thermal_image: np.ndarray,
        mono_image: np.ndarray,
        timestamp: float
    ) -> Optional[List[FusedDetection]]:
        """
        Process thermal-monochrome image pair.

        Args:
            pair_id: Camera pair ID
            thermal_image: Thermal image (numpy array)
            mono_image: Monochrome image (numpy array)
            timestamp: Frame timestamp

        Returns:
            List of fused detections, or None if processing queue full

        Example:
            >>> detections = fusion.process_frame_pair(
            ...     pair_id=0,
            ...     thermal_image=thermal_frame,
            ...     mono_image=mono_frame,
            ...     timestamp=time.time()
            ... )
            >>> for det in detections:
            ...     print(f"Detection at ({det.x}, {det.y})")
            ...     print(f"  Confidence: {det.confidence:.2f}")
            ...     print(f"  Thermal sig: {det.thermal_signature:.2f}")
        """
        pass

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get fusion performance metrics.

        Returns:
            Dict with metrics:
            - avg_fps: Average processing FPS
            - frame_count: Total frames processed
            - avg_registration_error_mm: Average registration error
            - target_confirmation_rate: Rate of confirmed targets
            - false_positive_reduction_rate: FP reduction rate
            - active_tracks: Number of active tracks
        """
        pass

    def apply_thermal_palette(
        self,
        thermal_image: np.ndarray,
        palette: str = "iron"
    ) -> np.ndarray:
        """
        Apply color palette to thermal image.

        Args:
            thermal_image: Thermal image (grayscale)
            palette: Palette name ('iron', 'rainbow', 'grayscale')

        Returns:
            Colorized thermal image
        """
        pass
```

##### FusionConfig

Configuration for fusion processing.

```python
@dataclass
class FusionConfig:
    """
    Fusion system configuration.

    Attributes:
        target_fps: Target processing frame rate
        registration_update_interval_s: Registration update interval
        thermal_threshold: Thermal detection threshold (0-1)
        mono_threshold: Monochrome detection threshold (0-1)
        confidence_threshold: Minimum confidence for output (0-1)
        max_range_km: Maximum detection range (km)
        enable_thermal_enhancement: Enable low-light enhancement
        enable_false_positive_reduction: Enable FP reduction
        enable_cuda: Use CUDA acceleration
        thermal_palette: Thermal colorization palette
        low_light_threshold: Brightness threshold for enhancement (0-1)
    """
    target_fps: int = 30
    registration_update_interval_s: float = 1.0
    thermal_threshold: float = 0.3
    mono_threshold: float = 0.2
    confidence_threshold: float = 0.6
    max_range_km: float = 5.0
    enable_thermal_enhancement: bool = True
    enable_false_positive_reduction: bool = True
    enable_cuda: bool = True
    thermal_palette: str = "iron"
    low_light_threshold: float = 0.1
```

##### FusedDetection

Result from multi-modal fusion.

```python
@dataclass
class FusedDetection:
    """
    Fused detection from thermal + monochrome.

    Attributes:
        x, y: Centroid coordinates
        width, height: Bounding box dimensions
        confidence: Overall confidence (0-1)
        thermal_signature: Thermal intensity (0-1)
        mono_signature: Monochrome intensity (0-1)
        track_id: Tracking ID
        camera_pair_id: Camera pair ID
        timestamp: Detection timestamp
        confirmed_by_thermal: Confirmed by thermal camera
        confirmed_by_mono: Confirmed by monochrome camera
        range_estimate_m: Estimated range (meters, optional)
    """
    x: float
    y: float
    width: float
    height: float
    confidence: float
    thermal_signature: float
    mono_signature: float
    track_id: int
    camera_pair_id: int
    timestamp: float
    confirmed_by_thermal: bool
    confirmed_by_mono: bool
    range_estimate_m: Optional[float] = None
```

---

#### 3. Distributed Processing

##### DistributedProcessor

Coordinates distributed processing across GPU cluster.

```python
class DistributedProcessor:
    """
    Distributed processing coordinator.

    Args:
        cluster_config: Cluster configuration
        data_pipeline: Data pipeline for frame management
        num_cameras: Number of camera pairs
        enable_fault_tolerance: Enable automatic failover
    """

    def __init__(
        self,
        cluster_config: ClusterConfig,
        data_pipeline: DataPipeline,
        num_cameras: int = 10,
        enable_fault_tolerance: bool = True
    ):
        pass

    def register_task_handler(
        self,
        task_type: str,
        handler: Callable[[Task], Any]
    ):
        """
        Register handler function for task type.

        Args:
            task_type: Type of task (e.g., 'process_frame')
            handler: Handler function that processes task

        Example:
            >>> def process_frame(task: Task) -> Result:
            ...     frame = task.input_data['frame']
            ...     # Process frame...
            ...     return result
            >>> processor.register_task_handler('process_frame', process_frame)
        """
        pass

    def start(self):
        """Start distributed processing."""
        pass

    def stop(self):
        """Stop distributed processing."""
        pass

    def submit_task(self, task: Task):
        """
        Submit task for execution.

        Args:
            task: Task to execute

        Example:
            >>> task = Task(
            ...     task_id=str(uuid.uuid4()),
            ...     task_type='process_frame',
            ...     camera_id=0,
            ...     frame_ids=[123],
            ...     input_data={'frame': frame_data},
            ...     priority=5
            ... )
            >>> processor.submit_task(task)
        """
        pass

    def submit_camera_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        metadata: FrameMetadata
    ) -> str:
        """
        Submit camera frame for processing.

        Args:
            camera_id: Camera ID
            frame: Frame data
            metadata: Frame metadata

        Returns:
            Task ID
        """
        pass

    def wait_for_task(
        self,
        task_id: str,
        timeout: float = 30.0
    ) -> Optional[Any]:
        """
        Wait for task completion.

        Args:
            task_id: Task ID to wait for
            timeout: Timeout in seconds

        Returns:
            Task result or None if timeout/failure
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dict with statistics:
            - tasks_submitted: Total tasks submitted
            - tasks_completed: Tasks completed successfully
            - tasks_failed: Failed tasks
            - total_workers: Total worker count
            - idle_workers: Idle worker count
            - busy_workers: Busy worker count
            - queue_size: Current queue size
            - avg_execution_time: Average task execution time
            - success_rate: Task success rate
        """
        pass

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health status.

        Returns:
            Dict with health metrics:
            - status: 'healthy', 'degraded', 'overloaded', or 'critical'
            - online_nodes: Number of online nodes
            - total_gpus: Total GPU count
            - active_workers: Active worker count
            - tasks_queued: Queued task count
            - success_rate: Task success rate
            - avg_latency_ms: Average latency (ms)
            - failover_count: Number of failovers
        """
        pass
```

---

#### 4. Camera Management

##### CameraManager

Manages camera system with 10 pairs (20 cameras).

```python
class CameraManager:
    """
    Camera management system.

    Args:
        num_pairs: Number of camera pairs
    """

    def __init__(self, num_pairs: int = 10):
        pass

    def add_camera(self, config: CameraConfiguration):
        """
        Add camera to management system.

        Args:
            config: Camera configuration

        Example:
            >>> config = CameraConfiguration(
            ...     camera_id=0,
            ...     pair_id=0,
            ...     camera_type=CameraType.MONO,
            ...     connection=ConnectionType.GIGE_VISION,
            ...     ip_address='192.168.1.10',
            ...     width=7680,
            ...     height=4320,
            ...     frame_rate=30.0
            ... )
            >>> manager.add_camera(config)
        """
        pass

    def add_pair(self, pair: CameraPair) -> bool:
        """
        Add camera pair.

        Args:
            pair: Camera pair configuration

        Returns:
            True if added successfully
        """
        pass

    def initialize_all_cameras(self) -> bool:
        """
        Initialize all cameras.

        Returns:
            True if all cameras initialized successfully
        """
        pass

    def start_all_acquisition(self) -> bool:
        """
        Start acquisition on all cameras.

        Returns:
            True if all cameras started successfully
        """
        pass

    def stop_all_acquisition(self):
        """Stop acquisition on all cameras."""
        pass

    def disconnect_all(self):
        """Disconnect all cameras."""
        pass

    def start_health_monitoring(self):
        """Start health monitoring thread."""
        pass

    def stop_health_monitoring(self):
        """Stop health monitoring thread."""
        pass

    def get_camera_health(
        self,
        camera_id: int
    ) -> Optional[CameraHealth]:
        """
        Get health status for specific camera.

        Args:
            camera_id: Camera ID

        Returns:
            CameraHealth object or None
        """
        pass

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health.

        Returns:
            Dict with system metrics:
            - total_cameras: Total camera count
            - streaming: Cameras currently streaming
            - ready: Cameras ready but not streaming
            - error: Cameras in error state
            - offline: Offline cameras
            - total_frames: Total frames received
            - total_dropped: Total dropped frames
            - drop_rate: Frame drop rate
            - avg_fps: Average FPS across cameras
            - avg_temperature: Average camera temperature
            - pairs_operational: Operational camera pairs
        """
        pass

    def save_configuration(self, filename: Optional[str] = None):
        """
        Save camera configurations to JSON file.

        Args:
            filename: Output filename (default: camera_config.json)
        """
        pass

    def load_configuration(self, filename: Optional[str] = None) -> bool:
        """
        Load camera configurations from JSON file.

        Args:
            filename: Input filename

        Returns:
            True if loaded successfully
        """
        pass
```

---

## C++ API

### Motion Extractor

High-performance motion extraction from 8K video frames.

```cpp
/**
 * @brief 8K Motion Extractor with OpenMP parallelization
 */
class MotionExtractor8K {
public:
    /**
     * @brief Constructor
     * @param width Frame width (default: 7680)
     * @param height Frame height (default: 4320)
     * @param threshold Motion threshold (0-255, default: 20)
     * @param min_area Minimum object area in pixels (default: 100)
     */
    MotionExtractor8K(
        int width = 7680,
        int height = 4320,
        int threshold = 20,
        int min_area = 100
    );

    /**
     * @brief Extract motion from frame
     * @param frame Input frame (numpy array)
     * @return Dict with results:
     *   - num_objects: Number of objects detected
     *   - coordinates: List of (x, y) centroids
     *   - bounding_boxes: List of (x, y, w, h) boxes
     *   - velocities: List of (vx, vy) vectors
     *   - processing_time_ms: Processing time in milliseconds
     */
    py::dict extract(py::array_t<uint8_t> frame);

    /**
     * @brief Get performance statistics
     * @return Dict with stats:
     *   - frames_processed: Total frames processed
     *   - avg_processing_time_ms: Average processing time
     *   - num_tracked_objects: Current tracked objects
     */
    py::dict getStats() const;

    /**
     * @brief Reset extractor state
     */
    void reset();

private:
    // Internal implementation
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Python binding example
PYBIND11_MODULE(motion_extractor_cpp, m) {
    py::class_<MotionExtractor8K>(m, "MotionExtractor8K")
        .def(py::init<int, int, int, int>(),
             py::arg("width") = 7680,
             py::arg("height") = 4320,
             py::arg("threshold") = 20,
             py::arg("min_area") = 100)
        .def("extract", &MotionExtractor8K::extract)
        .def("getStats", &MotionExtractor8K::getStats)
        .def("reset", &MotionExtractor8K::reset);
}
```

**Usage Example**:

```cpp
// C++ usage
MotionExtractor8K extractor(7680, 4320, 20, 100);

// Process frame
cv::Mat frame = cv::imread("frame.png", cv::IMREAD_GRAYSCALE);
auto result = extractor.extract(frame);

std::cout << "Objects detected: " << result["num_objects"] << std::endl;
std::cout << "Processing time: " << result["processing_time_ms"] << " ms" << std::endl;
```

```python
# Python usage
from motion_extractor_cpp import MotionExtractor8K

extractor = MotionExtractor8K(width=7680, height=4320)

result = extractor.extract(frame)
print(f"Objects detected: {result['num_objects']}")
print(f"Processing time: {result['processing_time_ms']:.2f} ms")
```

---

### Sparse Voxel Grid

CUDA-accelerated sparse voxel grid for 3D reconstruction.

```cpp
/**
 * @brief Sparse voxel grid with CUDA acceleration
 */
class SparseVoxelGrid {
public:
    /**
     * @brief Constructor
     * @param resolution Voxel grid resolution (e.g., 512)
     * @param voxel_size Size of each voxel in meters
     * @param use_cuda Enable CUDA acceleration
     */
    SparseVoxelGrid(
        int resolution,
        float voxel_size,
        bool use_cuda = true
    );

    /**
     * @brief Project 2D points to 3D voxel grid
     * @param coords 2D coordinates (N x 2 array)
     * @param camera_pose Camera position and orientation
     * @param confidence Confidence values for each point
     * @return Number of voxels updated
     */
    int project(
        py::array_t<float> coords,
        py::dict camera_pose,
        py::array_t<float> confidence
    );

    /**
     * @brief Get occupied voxels
     * @param threshold Occupancy threshold
     * @return Array of occupied voxel coordinates
     */
    py::array_t<int> get_occupied_voxels(float threshold = 0.5);

    /**
     * @brief Get voxel data
     * @return Dict with voxel grid data
     */
    py::dict get_voxel_data() const;

    /**
     * @brief Clear voxel grid
     */
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

---

### Thermal-Mono Fusion

C++ fusion engine for multi-spectral processing.

```cpp
/**
 * @brief Thermal-Monochrome fusion engine
 */
class ThermalMonoFusion {
public:
    /**
     * @brief Process fused frame pair
     * @param thermal_image Thermal image
     * @param mono_image Monochrome image
     * @param baseline Stereo baseline (meters)
     * @param max_range Maximum detection range (km)
     * @param use_cuda Enable CUDA acceleration
     * @return Fusion result with detections and metrics
     */
    static FusionResult process_fusion(
        py::array_t<uint8_t> thermal_image,
        py::array_t<uint8_t> mono_image,
        float baseline,
        float max_range,
        bool use_cuda
    );

    /**
     * @brief Estimate homography for image registration
     * @param thermal_image Source thermal image
     * @param mono_image Target monochrome image
     * @param ransac_threshold RANSAC threshold
     * @param max_iterations Maximum iterations
     * @return Registration parameters
     */
    static RegistrationParams estimate_homography(
        py::array_t<double> thermal_image,
        py::array_t<double> mono_image,
        double ransac_threshold,
        int max_iterations
    );

    /**
     * @brief Warp thermal image to align with mono
     * @param thermal_image Thermal image
     * @param homography Homography matrix (9 elements)
     * @param width Output width
     * @param height Output height
     * @return Warped thermal image
     */
    static py::array_t<uint8_t> warp_thermal_image(
        py::array_t<uint8_t> thermal_image,
        std::vector<double> homography,
        int width,
        int height
    );
};
```

---

## Protocol Specifications

### Frame Metadata Protocol

Protocol buffer definition for frame metadata:

```protobuf
syntax = "proto3";

package motion_tracking;

message FrameMetadata {
    uint64 frame_id = 1;
    uint64 camera_id = 2;
    double timestamp = 3;
    uint32 width = 4;
    uint32 height = 5;
    uint32 channels = 6;
    string pixel_format = 7;
    double exposure_time = 8;
    double gain = 9;
    CameraPose camera_pose = 10;
}

message CameraPose {
    Vector3 position = 1;
    Quaternion orientation = 2;
}

message Vector3 {
    double x = 1;
    double y = 2;
    double z = 3;
}

message Quaternion {
    double w = 1;
    double x = 2;
    double y = 3;
    double z = 4;
}
```

### Motion Data Protocol

```protobuf
message MotionData {
    uint64 frame_id = 1;
    double timestamp = 2;
    repeated Detection detections = 3;
}

message Detection {
    Vector2 centroid = 1;
    BoundingBox bbox = 2;
    Vector2 velocity = 3;
    float confidence = 4;
    uint32 track_id = 5;
}

message Vector2 {
    float x = 1;
    float y = 2;
}

message BoundingBox {
    float x = 1;
    float y = 2;
    float width = 3;
    float height = 4;
}
```

### Network Protocol

ZeroMQ-based protocol for inter-node communication:

**Message Format**:
```
[Frame ID (8 bytes)] [Timestamp (8 bytes)] [Data Length (4 bytes)] [Data]
```

**Message Types**:
- `FRAME_DATA`: Camera frame data
- `MOTION_DATA`: Motion detection results
- `TASK_ASSIGN`: Task assignment
- `TASK_RESULT`: Task completion result
- `HEARTBEAT`: Node health check
- `STATUS_UPDATE`: System status update

---

## Integration Examples

### Example 1: Complete Pipeline Integration

```python
"""
Complete integration example: Camera → Processing → Fusion → Voxel
"""

import numpy as np
import time
from src.camera import CameraManager, CameraConfiguration, CameraType
from src.video_processor import VideoProcessor, VideoStream
from src.fusion import FusionManager, FusionConfig
from src.voxel import SparseVoxelGrid
from src.network import DistributedProcessor, ClusterConfig, DataPipeline

def main():
    # 1. Initialize camera system
    camera_mgr = CameraManager(num_pairs=2)

    # Add camera pair 0
    camera_mgr.add_camera(CameraConfiguration(
        camera_id=0,
        pair_id=0,
        camera_type=CameraType.MONO,
        ip_address='192.168.1.10'
    ))
    camera_mgr.add_camera(CameraConfiguration(
        camera_id=1,
        pair_id=0,
        camera_type=CameraType.THERMAL,
        ip_address='192.168.1.11'
    ))

    # Initialize cameras
    camera_mgr.initialize_all_cameras()
    camera_mgr.start_all_acquisition()
    camera_mgr.start_health_monitoring()

    # 2. Initialize video processor
    video_processor = VideoProcessor(use_hardware_accel=True, target_fps=30.0)

    # 3. Initialize fusion system
    fusion_config = FusionConfig(
        enable_cuda=True,
        enable_false_positive_reduction=True
    )
    fusion_mgr = FusionManager(fusion_config)
    fusion_mgr.add_camera_pair(thermal_id=1, mono_id=0, baseline_m=0.5)
    fusion_mgr.start(num_workers=4)

    # 4. Initialize voxel grid
    voxel_grid = SparseVoxelGrid(resolution=512, voxel_size=0.1, use_cuda=True)

    # 5. Initialize distributed processor
    cluster = ClusterConfig()
    pipeline = DataPipeline(num_cameras=2)
    dist_processor = DistributedProcessor(cluster, pipeline, num_cameras=2)

    # Register processing handler
    def process_frame(task):
        camera_id = task.camera_id
        frame = task.input_data['frame']

        # Extract motion
        motion_data = video_processor.extract_motion(frame)

        return motion_data

    dist_processor.register_task_handler('process_frame', process_frame)
    dist_processor.start()

    # 6. Main processing loop
    try:
        while True:
            # Get synchronized frames from camera pair
            mono_frame, mono_metadata = camera_mgr.cameras[0].grab_frame()
            thermal_frame, thermal_metadata = camera_mgr.cameras[1].grab_frame()

            if mono_frame is None or thermal_frame is None:
                time.sleep(0.001)
                continue

            # Submit for distributed processing
            mono_task_id = dist_processor.submit_camera_frame(
                camera_id=0, frame=mono_frame, metadata=mono_metadata
            )
            thermal_task_id = dist_processor.submit_camera_frame(
                camera_id=1, frame=thermal_frame, metadata=thermal_metadata
            )

            # Get motion data
            mono_motion = dist_processor.wait_for_task(mono_task_id, timeout=0.1)
            thermal_motion = dist_processor.wait_for_task(thermal_task_id, timeout=0.1)

            # Fusion processing
            detections = fusion_mgr.process_frame_pair(
                pair_id=0,
                thermal_image=thermal_frame,
                mono_image=mono_frame,
                timestamp=time.time()
            )

            if detections:
                # Project to voxel grid
                coords = np.array([[d.x, d.y] for d in detections])
                confidence = np.array([d.confidence for d in detections])

                camera_pose = {
                    'position': mono_metadata['camera_pose']['position'],
                    'orientation': mono_metadata['camera_pose']['orientation']
                }

                voxel_grid.project(coords, camera_pose, confidence)

            # Print metrics
            metrics = dist_processor.get_statistics()
            print(f"FPS: {metrics['tasks_completed'] / 10:.2f}")
            print(f"Queue size: {metrics['queue_size']}")

            time.sleep(1.0 / 30.0)  # 30 FPS

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        # Cleanup
        dist_processor.stop()
        fusion_mgr.stop()
        camera_mgr.stop_all_acquisition()
        camera_mgr.disconnect_all()

if __name__ == '__main__':
    main()
```

### Example 2: Performance Monitoring Integration

```python
"""
Integration with monitoring and metrics collection
"""

import time
import json
from src.monitoring import SystemMonitor

def monitoring_example():
    # Initialize system monitor
    monitor = SystemMonitor(
        collect_interval=1.0,
        log_file='/var/log/motion_tracking/metrics.log'
    )

    # Initialize all components
    video_processor = VideoProcessor()
    fusion_mgr = FusionManager(FusionConfig())
    dist_processor = DistributedProcessor(...)

    # Register components with monitor
    monitor.register_component('video_processor', video_processor)
    monitor.register_component('fusion', fusion_mgr)
    monitor.register_component('distributed', dist_processor)

    # Start monitoring
    monitor.start()

    try:
        # Main processing loop
        while True:
            # Process frames...
            time.sleep(0.01)

            # Get system-wide metrics
            metrics = monitor.get_metrics()

            # Log to file every 10 seconds
            if metrics['uptime'] % 10 == 0:
                with open('/var/log/motion_tracking/metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)

            # Alert on issues
            if metrics['avg_fps'] < 25:
                print("WARNING: FPS below target!")

            if metrics['gpu_utilization'] > 95:
                print("WARNING: GPU overutilized!")

    finally:
        monitor.stop()
```

### Example 3: Custom Task Handler Integration

```python
"""
Custom task handler for specialized processing
"""

from src.network import DistributedProcessor, Task
import numpy as np

class CustomDetector:
    """Custom detection algorithm"""

    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        # Load your custom ML model
        pass

    def detect(self, frame):
        # Run custom detection
        detections = self.model.predict(frame)
        return detections

def custom_task_handler_example():
    # Initialize distributed processor
    dist_processor = DistributedProcessor(...)

    # Create custom detector
    detector = CustomDetector()

    # Register custom handler
    def custom_detect_handler(task: Task):
        frame = task.input_data['frame']

        # Run custom detection
        detections = detector.detect(frame)

        # Convert to standard format
        result = {
            'num_objects': len(detections),
            'coordinates': [(d.x, d.y) for d in detections],
            'confidence': [d.confidence for d in detections]
        }

        return result

    dist_processor.register_task_handler('custom_detect', custom_detect_handler)

    # Submit custom tasks
    task = Task(
        task_id='custom_001',
        task_type='custom_detect',
        camera_id=0,
        frame_ids=[123],
        input_data={'frame': frame_data},
        priority=10  # High priority
    )

    dist_processor.submit_task(task)
    result = dist_processor.wait_for_task('custom_001')
```

---

## Error Handling

### Common Error Codes

| Code | Name | Description |
|------|------|-------------|
| 1000 | CAMERA_CONNECTION_FAILED | Camera connection failed |
| 1001 | CAMERA_TIMEOUT | Camera frame timeout |
| 1002 | CAMERA_CALIBRATION_INVALID | Invalid calibration data |
| 2000 | VIDEO_DECODE_FAILED | Video decode error |
| 2001 | MOTION_EXTRACT_FAILED | Motion extraction error |
| 3000 | FUSION_REGISTRATION_FAILED | Image registration failed |
| 3001 | FUSION_QUEUE_FULL | Fusion queue overflow |
| 4000 | DISTRIBUTED_NODE_OFFLINE | Worker node offline |
| 4001 | DISTRIBUTED_TASK_TIMEOUT | Task execution timeout |
| 5000 | VOXEL_CUDA_ERROR | CUDA error in voxel processing |

### Exception Hierarchy

```python
MotionTrackingException (base)
├── CameraException
│   ├── ConnectionException
│   ├── TimeoutException
│   └── CalibrationException
├── ProcessingException
│   ├── DecodeException
│   ├── MotionExtractionException
│   └── FusionException
├── DistributedException
│   ├── NodeException
│   ├── TaskException
│   └── NetworkException
└── VoxelException
    └── CUDAException
```

---

## Best Practices

1. **Resource Management**: Always use context managers or try/finally for cleanup
2. **Error Handling**: Catch specific exceptions, log errors, implement retry logic
3. **Performance**: Enable hardware acceleration, tune buffer sizes, monitor metrics
4. **Threading**: Use thread-safe data structures, avoid global state
5. **GPU Memory**: Pre-allocate buffers, use pinned memory, monitor usage
6. **Network**: Use RDMA when available, implement exponential backoff
7. **Calibration**: Validate calibration data, update periodically, handle drift

---

## Version History

- **v0.1.0** (2024-01): Initial API release
- **v0.2.0** (2024-03): Added fusion system
- **v0.3.0** (2024-05): Added distributed processing
- **v0.4.0** (2024-07): Added CUDA acceleration

---

## Support

For API support, please contact the development team or file an issue in the project repository.
