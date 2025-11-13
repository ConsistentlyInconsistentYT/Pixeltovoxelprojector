# Dependencies Documentation

## Overview

This document provides a comprehensive list of all dependencies required for the Pixel to Voxel Projector system, including system libraries, Python packages, and optional components.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [System Libraries](#system-libraries)
3. [Python Packages](#python-packages)
4. [GPU Dependencies](#gpu-dependencies)
5. [Optional Dependencies](#optional-dependencies)
6. [Hardware-Specific](#hardware-specific)
7. [Dependency Graphs](#dependency-graphs)

---

## System Requirements

### Operating System

| OS | Version | Status |
|----|---------|--------|
| Ubuntu | 20.04+ | **Recommended** |
| Debian | 11+ | Supported |
| CentOS/RHEL | 8+ | Supported |
| Windows | 10/11 with WSL2 | Supported |
| macOS | 11+ | Limited (CPU only) |

### Hardware

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| CPU | Intel i5 / Ryzen 5 | Intel i9 / Ryzen 9 | 8+ cores preferred |
| RAM | 16 GB | 32+ GB | 64 GB for 8K processing |
| GPU | GTX 1660 | RTX 3090/4090 | NVIDIA only |
| VRAM | 6 GB | 24 GB | For GPU acceleration |
| Storage | 50 GB | 500 GB+ | SSD recommended |

---

## System Libraries

### Build Tools

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config
```

| Package | Version | Purpose |
|---------|---------|---------|
| gcc/g++ | 9.0+ | C++ compiler |
| cmake | 3.18+ | Build system |
| ninja | 1.10+ | Fast build tool |
| git | 2.25+ | Version control |
| pkg-config | 0.29+ | Library configuration |

### Core Libraries

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libomp-dev \
    libgomp1 \
    libpthread-stubs0-dev
```

| Library | Version | Purpose |
|---------|---------|---------|
| OpenMP | 4.5+ | Parallel computing |
| pthread | 2.31+ | Threading |

### Video Processing

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev
```

| Library | Version | Purpose |
|---------|---------|---------|
| FFmpeg | 4.3+ | Video encoding/decoding |
| x264 | 0.160+ | H.264 encoding |
| x265 | 3.4+ | H.265/HEVC encoding |
| libvpx | 1.9+ | VP8/VP9 encoding |

### Computer Vision

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libopencv-dev \
    libopencv-contrib-dev
```

| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | 4.5+ | Computer vision algorithms |

### Graphics & Visualization

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libglfw3-dev \
    libglew-dev
```

| Library | Version | Purpose |
|---------|---------|---------|
| OpenGL | 4.5+ | 3D graphics |
| GLFW | 3.3+ | Window/context management |
| GLEW | 2.1+ | OpenGL extension loading |

### Networking & Communication

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libzmq3-dev \
    libzmqpp-dev
```

| Library | Version | Purpose |
|---------|---------|---------|
| ZeroMQ | 4.3+ | High-performance messaging |

### Compression

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    liblz4-dev \
    libzstd-dev \
    libsnappy-dev
```

| Library | Version | Purpose |
|---------|---------|---------|
| LZ4 | 1.9+ | Fast compression |
| Zstandard | 1.4+ | High-ratio compression |
| Snappy | 1.1+ | Fast compression |

### Protocol Buffers

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev
```

| Library | Version | Purpose |
|---------|---------|---------|
| Protocol Buffers | 3.19+ | Data serialization |

---

## Python Packages

### Core Dependencies

| Package | Version | Purpose | Required |
|---------|---------|---------|----------|
| numpy | ≥1.21.0, <2.0.0 | Numerical computing | ✓ |
| pybind11 | ≥2.9.0, <3.0.0 | C++ bindings | ✓ |
| opencv-python | ≥4.5.5, <5.0.0 | Computer vision | ✓ |
| ffmpeg-python | ≥0.2.0 | Video processing | ✓ |
| scipy | ≥1.7.0, <2.0.0 | Scientific computing | ✓ |

### GPU Acceleration

| Package | Version | CUDA Version | Notes |
|---------|---------|--------------|-------|
| cupy-cuda11x | ≥10.0.0 | 11.x | For CUDA 11 |
| cupy-cuda12x | ≥12.0.0 | 12.x | For CUDA 12 |
| pycuda | ≥2021.1 | All | GPU detection |

### Networking

| Package | Version | Purpose |
|---------|---------|---------|
| pyzmq | ≥22.3.0, <26.0.0 | ZeroMQ bindings |
| websockets | ≥10.1, <12.0 | WebSocket support |
| msgpack | ≥1.0.3, <2.0.0 | Message packing |

### Protocol Buffers & gRPC

| Package | Version | Purpose |
|---------|---------|---------|
| protobuf | ≥3.19.0, <5.0.0 | Protocol buffers |
| grpcio | ≥1.43.0, <2.0.0 | gRPC framework |
| grpcio-tools | ≥1.43.0, <2.0.0 | gRPC code generation |

### Compression

| Package | Version | Purpose |
|---------|---------|---------|
| lz4 | ≥4.0.0, <5.0.0 | LZ4 compression |
| zstandard | ≥0.17.0, <1.0.0 | Zstandard compression |
| python-snappy | ≥0.6.1, <1.0.0 | Snappy compression |

### Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| open3d | ≥0.15.0, <1.0.0 | 3D visualization |
| vtk | ≥9.1.0, <10.0.0 | Visualization Toolkit |
| PyOpenGL | ≥3.1.5, <4.0.0 | OpenGL bindings |
| plotly | ≥5.5.0, <6.0.0 | Interactive plots |

### System Monitoring

| Package | Version | Purpose |
|---------|---------|---------|
| psutil | ≥5.9.0, <6.0.0 | System metrics |
| py-cpuinfo | ≥8.0.0, <10.0.0 | CPU information |
| pynvml | ≥11.4.1 | NVIDIA GPU monitoring |

### Development Tools

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | ≥7.0.0, <8.0.0 | Testing framework |
| pytest-benchmark | ≥3.4.1 | Performance testing |
| black | ≥22.3.0, <24.0.0 | Code formatting |
| mypy | ≥0.942, <2.0.0 | Type checking |

---

## GPU Dependencies

### NVIDIA CUDA Toolkit

| Component | Version | Purpose |
|-----------|---------|---------|
| CUDA Runtime | 11.0+ / 12.0+ | GPU computing |
| cuBLAS | Included | Linear algebra |
| cuFFT | Included | Fast Fourier Transform |
| cuSPARSE | Included | Sparse matrix operations |
| NVRTC | Included | Runtime compilation |

### cuDNN

| Version | CUDA Compatibility | Purpose |
|---------|-------------------|---------|
| 8.x | CUDA 11.x / 12.x | Deep learning primitives |

### NVIDIA Driver

| Driver Version | CUDA Support | GPU Support |
|---------------|--------------|-------------|
| 470+ | CUDA 11.x | GTX 10xx+ |
| 525+ | CUDA 12.x | RTX 30xx/40xx |

### Installation

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify compute capability
python3 << EOF
import pycuda.driver as cuda
import pycuda.autoinit
for i in range(cuda.Device.count()):
    dev = cuda.Device(i)
    print(f"GPU {i}: {dev.name()}")
    print(f"  Compute Capability: {dev.compute_capability()}")
EOF
```

---

## Optional Dependencies

### Camera Hardware

| Package | Hardware | Installation |
|---------|----------|--------------|
| pypylon | Basler GigE cameras | `pip install pypylon` |
| simple-pyspin | FLIR cameras | `pip install simple-pyspin` |
| picamera | Raspberry Pi cameras | `pip install picamera` |

**Note**: These packages require hardware-specific drivers:

```bash
# Basler Pylon SDK
wget https://www.baslerweb.com/media/downloads/software/pylon_software/pylon_7.2.0_linux-x86_64_setup.tar.gz
tar -xzf pylon_*.tar.gz
cd pylon_*/
sudo ./setup-usb.sh

# FLIR Spinnaker SDK
# Download from: https://www.flir.com/products/spinnaker-sdk/
```

### Machine Learning

| Package | Purpose | Installation |
|---------|---------|--------------|
| torch | PyTorch deep learning | `pip install torch torchvision` |
| onnxruntime | ONNX inference | `pip install onnxruntime-gpu` |
| tensorflow | TensorFlow (optional) | `pip install tensorflow-gpu` |

### Distributed Computing

| Package | Purpose | Installation |
|---------|---------|--------------|
| dask | Parallel computing | `pip install dask[complete]` |
| ray | Distributed computing | `pip install ray[default]` |

### Interactive Development

| Package | Purpose | Installation |
|---------|---------|--------------|
| jupyter | Jupyter notebooks | `pip install jupyter` |
| jupyterlab | JupyterLab IDE | `pip install jupyterlab` |
| matplotlib | Plotting | `pip install matplotlib` |

---

## Hardware-Specific

### Camera Requirements

#### Basler GigE Cameras
- **Driver**: Pylon SDK 7.2+
- **Network**: Gigabit Ethernet adapter
- **Bandwidth**: 1000 Mbps minimum
- **MTU**: 9000 (Jumbo frames recommended)

#### FLIR Cameras
- **Driver**: Spinnaker SDK 3.0+
- **Interface**: USB 3.0 / GigE
- **Power**: External power for high-speed operation

### Network Requirements

| Feature | Requirement | Purpose |
|---------|-------------|---------|
| Bandwidth | 10 Gbps+ | Multi-camera streaming |
| Latency | <1ms | Real-time processing |
| Protocol | UDP multicast | Efficient distribution |

---

## Dependency Graphs

### Build Dependencies

```
Build System
├── cmake (3.18+)
├── ninja-build
├── gcc/g++ (9.0+)
└── pkg-config

Python Build
├── setuptools
├── wheel
├── pybind11
└── grpcio-tools (for protobuf)

CUDA Build (Optional)
├── CUDA Toolkit (11.x/12.x)
├── cuDNN (8.x)
└── NVIDIA Driver (470+/525+)
```

### Runtime Dependencies

```
Core Runtime
├── numpy
├── opencv-python
├── scipy
└── ffmpeg-python

GPU Runtime (Optional)
├── cupy-cuda11x/12x
├── pycuda
└── CUDA Runtime

Networking
├── pyzmq
├── websockets
└── msgpack

Visualization
├── open3d
├── vtk
└── PyOpenGL
```

### Development Dependencies

```
Development
├── Testing
│   ├── pytest
│   ├── pytest-benchmark
│   └── pytest-cov
├── Code Quality
│   ├── black
│   ├── mypy
│   └── flake8
└── Documentation
    ├── sphinx
    └── sphinx-rtd-theme
```

---

## Installation Order

### Recommended Installation Sequence

1. **System Libraries** (requires sudo)
   ```bash
   sudo apt-get install build-essential cmake ninja-build
   sudo apt-get install ffmpeg libopencv-dev libgl1-mesa-dev
   ```

2. **CUDA Toolkit** (if using GPU)
   ```bash
   # Download from NVIDIA website
   wget https://developer.download.nvidia.com/compute/cuda/...
   sudo sh cuda_*.run
   ```

3. **Python Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   ```

4. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **GPU Packages** (optional)
   ```bash
   pip install cupy-cuda12x pycuda
   ```

6. **Build Project**
   ```bash
   pip install -e ".[full,dev]"
   ```

---

## Version Compatibility Matrix

### CUDA Version Compatibility

| CUDA | Python | GCC | CuPy | PyTorch |
|------|--------|-----|------|---------|
| 11.8 | 3.8-3.11 | 9-11 | 11.x | 2.0+ |
| 12.0 | 3.8-3.11 | 9-12 | 12.x | 2.1+ |
| 12.2 | 3.8-3.12 | 9-12 | 12.x | 2.2+ |

### Operating System Compatibility

| OS | Python | CUDA | Notes |
|----|--------|------|-------|
| Ubuntu 20.04 | 3.8+ | 11.x/12.x | Recommended |
| Ubuntu 22.04 | 3.10+ | 12.x | Latest |
| Debian 11 | 3.9+ | 11.x/12.x | Stable |
| CentOS 8 | 3.8+ | 11.x/12.x | Supported |
| Windows WSL2 | 3.8+ | 11.x/12.x | GPU passthrough required |

---

## Troubleshooting

### Common Issues

#### Missing System Libraries
```bash
# Error: cannot find -lGL
sudo apt-get install libgl1-mesa-dev

# Error: cannot find -lX11
sudo apt-get install libx11-dev
```

#### Python Package Conflicts
```bash
# Clear pip cache
pip cache purge

# Reinstall with no cache
pip install --no-cache-dir -r requirements.txt
```

#### CUDA Version Mismatch
```bash
# Check CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Install matching cupy
pip install cupy-cuda11x  # or cupy-cuda12x
```

---

## Dependency Updates

### Keeping Dependencies Current

```bash
# List outdated packages
pip list --outdated

# Update specific package
pip install --upgrade numpy

# Update all packages (careful!)
pip install --upgrade -r requirements.txt
```

### Security Updates

```bash
# Check for security vulnerabilities
pip-audit

# Update to secure versions
pip install --upgrade package-name
```

---

## Resources

- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- [cuDNN Downloads](https://developer.nvidia.com/cudnn)
- [OpenCV Documentation](https://docs.opencv.org/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Protocol Buffers Guide](https://developers.google.com/protocol-buffers)

---

**Last Updated**: 2025-01-13
**Version**: 1.0.0
