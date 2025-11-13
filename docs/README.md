# 8K Motion Tracking and Voxel Processing System

## System Overview

A high-performance distributed system for real-time motion tracking, target detection, and 3D voxel reconstruction from multiple 8K camera pairs. The system combines thermal and monochrome imaging with GPU-accelerated processing to deliver sub-33ms latency at 30 FPS.

### Key Capabilities

- **Real-time 8K Processing**: Processes 7680x4320 video streams at 30 FPS
- **Multi-Modal Fusion**: Combines thermal and monochrome imaging for enhanced detection
- **Distributed Architecture**: Scales across multiple GPU nodes with automatic load balancing
- **CUDA Acceleration**: GPU-optimized kernels for motion extraction and voxel processing
- **Fault Tolerance**: Automatic failover and recovery from node/camera failures
- **Low Latency**: Sub-33ms end-to-end pipeline latency

### System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Camera Layer (10 Pairs)                       │
│  ┌─────────────┐  ┌─────────────┐         ┌─────────────┐      │
│  │ Mono + Therm│  │ Mono + Therm│   ...   │ Mono + Therm│      │
│  │   Pair 0    │  │   Pair 1    │         │   Pair 9    │      │
│  └─────────────┘  └─────────────┘         └─────────────┘      │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Video Processing Layer                         │
│  • Hardware-accelerated decode (HEVC, H.264)                     │
│  • Motion extraction (C++ with OpenMP)                           │
│  • Frame synchronization (<10ms time diff)                       │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Fusion Layer                                   │
│  • Image registration and alignment                              │
│  • Multi-spectral detection                                      │
│  • False positive reduction                                      │
│  • Target tracking                                               │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Distributed Processing Layer                   │
│  • Task scheduling and load balancing                            │
│  • Multi-GPU coordination                                        │
│  • Fault tolerance and failover                                  │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Voxel Reconstruction Layer                     │
│  • Sparse voxel grid (CUDA accelerated)                          │
│  • 3D motion projection                                          │
│  • Spatial tracking                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Guide

### Prerequisites

- **Hardware**:
  - NVIDIA GPU with CUDA 11.0+ (RTX 3090/4090 recommended)
  - 32GB+ RAM
  - 10GbE network adapter (for distributed deployment)

- **Software**:
  - Ubuntu 20.04+ or compatible Linux distribution
  - CUDA Toolkit 11.0+
  - Python 3.8+
  - GCC 9.0+ with C++17 support

### Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd Pixeltovoxelprojector
```

#### 2. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    python3-dev \
    python3-pip

# CUDA Toolkit (if not installed)
# Download from https://developer.nvidia.com/cuda-downloads
```

#### 3. Install Python Dependencies

```bash
pip install -r requirements.txt

# Camera system dependencies
pip install -r requirements_camera.txt
```

#### 4. Build C++ Extensions

```bash
python setup.py build_ext --inplace
```

This will compile:
- Motion extractor (C++ with OpenMP)
- Sparse voxel grid
- Stream manager
- Fusion engine
- CUDA extensions (if CUDA available)

#### 5. Verify Installation

```bash
python verify_tracking_system.py
```

Expected output:
```
================================================================================
Camera Tracking System - Verification
================================================================================

1. Checking implementation files...
✓ Pose Tracker: /home/user/Pixeltovoxelprojector/src/camera/pose_tracker.py
✓ Orientation Manager: /home/user/Pixeltovoxelprojector/src/camera/orientation_manager.cpp
...
✓ ALL CHECKS PASSED!
```

### Running Your First Example

#### Example 1: Single Stream Motion Extraction

```bash
cd src
python example_8k_pipeline.py --example 1
```

#### Example 2: Dual Camera Fusion

```python
from src.fusion import FusionManager, FusionConfig

# Create fusion manager
config = FusionConfig(
    target_fps=30,
    enable_cuda=True,
    enable_false_positive_reduction=True
)
fusion_mgr = FusionManager(config)

# Add camera pair
pair_id = fusion_mgr.add_camera_pair(
    thermal_id=0,
    mono_id=1,
    baseline_m=0.5
)

# Start processing
fusion_mgr.start(num_workers=4)

# Process frame pairs
detections = fusion_mgr.process_frame_pair(
    pair_id, thermal_frame, mono_frame, timestamp
)
```

#### Example 3: Distributed Processing

```bash
# On master node
python examples/distributed_processing_example.py --master

# On worker nodes
python examples/distributed_processing_example.py --worker --master-ip 192.168.1.100
```

---

## Installation Instructions

### Detailed Installation

#### Option 1: Standard Installation

```bash
# Install package and dependencies
pip install -e .

# Run tests
python -m pytest tests/
```

#### Option 2: Development Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Option 3: Docker Installation

```bash
# Build Docker image
docker build -t 8k-motion-tracking .

# Run container
docker run --gpus all -it 8k-motion-tracking
```

### Building C++ Extensions Manually

```bash
# Motion extractor
g++ -O3 -shared -std=c++17 -fopenmp -fPIC \
    $(python -m pybind11 --includes) \
    src/motion_extractor.cpp \
    -o motion_extractor_cpp$(python3-config --extension-suffix)

# CUDA extensions (requires CUDA toolkit)
nvcc -O3 --std=c++17 -Xcompiler -fPIC \
    -gencode arch=compute_86,code=sm_86 \
    cuda/voxel_cuda.cu -c -o voxel_cuda.o

g++ -shared voxel_cuda.o cuda/voxel_cuda_wrapper.cpp \
    -L/usr/local/cuda/lib64 -lcudart \
    -o voxel_cuda$(python3-config --extension-suffix)
```

---

## Configuration Guide

### Camera Configuration

Configure camera pairs in `camera_config.json`:

```json
{
  "num_pairs": 10,
  "cameras": {
    "0": {
      "camera_id": 0,
      "pair_id": 0,
      "camera_type": "MONO",
      "connection": "GIGE_VISION",
      "ip_address": "192.168.1.10",
      "width": 7680,
      "height": 4320,
      "frame_rate": 30.0,
      "exposure_time": 10000.0,
      "gain": 1.0,
      "trigger_mode": "Hardware",
      "position": [0.0, 0.0, 0.0],
      "orientation": [[1,0,0], [0,1,0], [0,0,1]]
    },
    "1": {
      "camera_id": 1,
      "pair_id": 0,
      "camera_type": "THERMAL",
      "connection": "GIGE_VISION",
      "ip_address": "192.168.1.11",
      "width": 7680,
      "height": 4320,
      "frame_rate": 30.0
    }
  },
  "pairs": {
    "0": {
      "pair_id": 0,
      "mono_camera_id": 0,
      "thermal_camera_id": 1,
      "stereo_baseline": 0.5
    }
  }
}
```

### Network Configuration

Configure cluster nodes in your application:

```python
from src.network import ClusterConfig

cluster = ClusterConfig(
    node_id="node1",
    discovery_port=9999,
    data_port_range=(10000, 11000),
    heartbeat_interval=1.0,
    heartbeat_timeout=5.0,
    enable_rdma=True,
    rdma_device="mlx5_0"
)

# Start cluster
cluster.start(is_master=True)
```

### Fusion Configuration

```python
from src.fusion import FusionConfig

config = FusionConfig(
    target_fps=30,
    registration_update_interval_s=1.0,
    thermal_threshold=0.3,
    mono_threshold=0.2,
    confidence_threshold=0.6,
    max_range_km=5.0,
    enable_thermal_enhancement=True,
    enable_false_positive_reduction=True,
    enable_cuda=True,
    thermal_palette="iron"
)
```

### Performance Tuning

Edit configuration parameters for your hardware:

```python
# For RTX 4090
CUDA_CONFIG = {
    'compute_capability': '89',
    'max_threads_per_block': 1024,
    'shared_memory_per_block': 49152,
    'registers_per_thread': 128
}

# For high-throughput processing
PIPELINE_CONFIG = {
    'buffer_size': 60,  # frames
    'num_decoder_threads': 4,
    'num_processing_threads': 8,
    'enable_hardware_accel': True,
    'codec': 'hevc_cuvid'  # NVIDIA hardware decoder
}
```

---

## API Reference

### Core Classes

#### VideoProcessor

Main class for 8K video processing:

```python
from src.video_processor import VideoProcessor, VideoStream

processor = VideoProcessor(
    use_hardware_accel=True,
    target_fps=30.0,
    enable_profiling=True
)

# Add video stream
processor.add_stream('camera1', VideoStream(
    path='/path/to/video.mp4',
    stream_type='monochrome',
    codec=VideoCodec.HEVC
))

# Register callback
def on_motion(motion_data):
    print(f"Detected {len(motion_data.coordinates)} objects")

processor.register_motion_callback(on_motion)

# Start processing
processor.start_processing()
```

#### FusionManager

Multi-modal sensor fusion:

```python
from src.fusion import FusionManager, FusionConfig

fusion_mgr = FusionManager(FusionConfig())
fusion_mgr.start(num_workers=4)

# Process frames
detections = fusion_mgr.process_frame_pair(
    pair_id=0,
    thermal_image=thermal_frame,
    mono_image=mono_frame,
    timestamp=time.time()
)

# Get metrics
metrics = fusion_mgr.get_performance_metrics()
print(f"FPS: {metrics['avg_fps']:.2f}")
print(f"False positive reduction: {metrics['false_positive_reduction_rate']:.2%}")
```

#### DistributedProcessor

Cluster processing coordination:

```python
from src.network import DistributedProcessor, ClusterConfig, DataPipeline

cluster = ClusterConfig()
pipeline = DataPipeline(num_cameras=10)
processor = DistributedProcessor(cluster, pipeline, num_cameras=10)

# Register task handler
def process_frame(task):
    # Process frame
    return result

processor.register_task_handler('process_frame', process_frame)

# Start distributed processing
processor.start()

# Submit tasks
task_id = processor.submit_camera_frame(camera_id, frame, metadata)
result = processor.wait_for_task(task_id, timeout=1.0)
```

#### CameraManager

Camera system management:

```python
from src.camera import CameraManager, CameraConfiguration, CameraType

mgr = CameraManager(num_pairs=10)

# Add cameras
for i in range(20):
    config = CameraConfiguration(
        camera_id=i,
        pair_id=i // 2,
        camera_type=CameraType.MONO if i % 2 == 0 else CameraType.THERMAL,
        connection=ConnectionType.GIGE_VISION,
        ip_address=f"192.168.1.{10+i}"
    )
    mgr.add_camera(config)

# Initialize and start
mgr.initialize_all_cameras()
mgr.start_all_acquisition()
mgr.start_health_monitoring()

# Check health
health = mgr.get_system_health()
print(f"Cameras streaming: {health['streaming']}/{health['total_cameras']}")
```

---

## Performance Benchmarks

See [docs/PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks.

### Quick Reference

| Component | Throughput | Latency | GPU Usage |
|-----------|------------|---------|-----------|
| 8K HEVC Decode (HW) | 60+ FPS | 5-8 ms | 15% |
| Motion Extraction | 35+ FPS | 12-18 ms | 40% |
| Fusion Processing | 30+ FPS | 8-12 ms | 25% |
| Voxel Reconstruction | 30+ FPS | 5-8 ms | 30% |
| End-to-End Pipeline | 30 FPS | <33 ms | 65% |

Tested on: NVIDIA RTX 4090, Intel i9-13900K, 64GB RAM

---

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found

```bash
# Check CUDA installation
nvcc --version

# Set CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 2. Low FPS / High Latency

- Enable hardware acceleration: `use_hardware_accel=True`
- Check GPU utilization: `nvidia-smi`
- Reduce buffer size to decrease latency
- Increase number of processing threads

#### 3. Camera Connection Failures

```bash
# Check network connectivity
ping <camera-ip>

# Check GigE Vision devices
arv-tool-0.8 -l

# Increase network buffer size
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.rmem_default=26214400
```

#### 4. Build Failures

```bash
# Install missing dependencies
pip install pybind11 numpy

# Clean and rebuild
python setup.py clean --all
python setup.py build_ext --inplace
```

### Getting Help

- Check [docs/DEPLOYMENT.md](DEPLOYMENT.md) for deployment issues
- Check [docs/API.md](API.md) for API usage examples
- Review logs in `/var/log/motion_tracking/`
- Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

---

## System Requirements

### Minimum Requirements

- CPU: Intel Core i7 or AMD Ryzen 7
- RAM: 32GB
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- Storage: 500GB SSD
- Network: 1GbE

### Recommended Requirements

- CPU: Intel Core i9-13900K or AMD Ryzen 9 7950X
- RAM: 64GB DDR5
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Storage: 2TB NVMe SSD
- Network: 10GbE or InfiniBand

### Multi-Node Cluster

- 4+ GPU nodes
- 10GbE or InfiniBand interconnect
- Sub-5ms inter-node latency
- Shared storage (NFS/Lustre) for calibration data

---

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Authors

See AUTHORS file for contributor list.

## References

- [Architecture Documentation](ARCHITECTURE.md)
- [API Reference](API.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Performance Guide](PERFORMANCE.md)
