# Build Instructions

## Overview

This document provides comprehensive instructions for building and installing the Pixel to Voxel Projector system. The system supports multiple build methods and configurations.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Build Methods](#detailed-build-methods)
4. [Docker Installation](#docker-installation)
5. [Troubleshooting](#troubleshooting)
6. [Development Setup](#development-setup)

---

## System Requirements

### Minimum Requirements

- **OS**: Ubuntu 20.04+ (Linux primary), Windows 10/11 with WSL2, macOS 11+
- **CPU**: Intel Core i7 or AMD Ryzen 7 (8 cores recommended)
- **RAM**: 16 GB (32 GB recommended for 8K video processing)
- **Storage**: 50 GB free space
- **Python**: 3.8 or higher

### GPU Requirements (Recommended)

- **GPU**: NVIDIA RTX 3060 or better
  - RTX 3090/4090 recommended for optimal performance
  - Compute Capability 7.5+ required
- **VRAM**: 8 GB minimum (24 GB recommended)
- **CUDA**: 11.x or 12.x
- **cuDNN**: 8.x
- **NVIDIA Driver**: 470+ for CUDA 11, 525+ for CUDA 12

### Software Dependencies

#### System Packages (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libomp-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libopencv-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    protobuf-compiler \
    libprotobuf-dev \
    libzmq3-dev \
    liblz4-dev \
    libzstd-dev
```

---

## Quick Start

### Option 1: One-Command Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/Pixeltovoxelprojector.git
cd Pixeltovoxelprojector

# Install with all dependencies
pip install -e ".[full,dev]"
```

### Option 2: Docker (Easiest)

```bash
# Build and run with GPU support
docker build -t pixeltovoxel:latest -f docker/Dockerfile .
docker run --gpus all -it --rm -v $(pwd):/app pixeltovoxel:latest
```

### Option 3: Basic Install (CPU only)

```bash
pip install -e .
```

---

## Detailed Build Methods

### Method 1: Python setuptools (setup.py)

This is the primary build method using Python's setuptools with custom CUDA compilation.

#### Step 1: Install Python Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt

# For GPU support (choose based on your CUDA version)
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x
```

#### Step 2: Build Extensions

```bash
# Development install (editable)
pip install -e .

# Production install
pip install .

# With optional dependencies
pip install -e ".[full,dev,cuda]"
```

#### Step 3: Verify Installation

```bash
# Test CUDA extensions
python -c "import voxel_cuda; print('CUDA extensions loaded successfully')"

# Run quick test
python tests/benchmarks/test_installation.py
```

#### Build Configuration

The setup.py build supports several environment variables:

```bash
# Specify CUDA path
export CUDA_HOME=/usr/local/cuda-12.0

# Set compute architectures
export TORCH_CUDA_ARCH_LIST="8.6;8.9"  # RTX 3090/4090

# Parallel build
export MAX_JOBS=8

# Build with custom flags
CFLAGS="-march=native" pip install -e .
```

### Method 2: CMake Build System

For more control over the build process, use CMake.

#### Step 1: Configure Build

```bash
mkdir build
cd build

# Configure with CMake
cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CUDA=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DUSE_OPENMP=ON \
    -DENABLE_FAST_MATH=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86;89"
```

#### Step 2: Build

```bash
# Build with Ninja (fast)
ninja

# Or with make
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### Step 3: Install

```bash
# System-wide install (requires sudo)
sudo ninja install

# Or install to custom prefix
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
ninja install
```

#### CMake Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_CUDA` | ON | Build CUDA extensions |
| `BUILD_TESTS` | ON | Build test suite |
| `BUILD_BENCHMARKS` | ON | Build benchmarks |
| `BUILD_PYTHON_BINDINGS` | ON | Build Python bindings |
| `USE_OPENMP` | ON | Enable OpenMP |
| `ENABLE_FAST_MATH` | ON | Enable fast math |
| `CMAKE_CUDA_ARCHITECTURES` | "86;89" | Target GPU architectures |

Example configurations:

```bash
# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_CUDA=ON

# CPU-only build
cmake .. -DBUILD_CUDA=OFF -DUSE_OPENMP=ON

# Minimal build
cmake .. -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF

# Custom CUDA architectures
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"
```

### Method 3: Protocol Buffer Compilation

Protocol buffers are automatically compiled during setup, but you can compile them manually:

```bash
cd src/protocols

# Compile all .proto files
for proto in *.proto; do
    protoc --python_out=. --grpc_python_out=. -I. "$proto"
done

# Or use the Python grpcio-tools
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    motion_protocol.proto
```

---

## Docker Installation

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit

#### Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Build Docker Image

```bash
# Build image
docker build -t pixeltovoxel:latest -f docker/Dockerfile .

# Build with specific CUDA version
docker build --build-arg CUDA_VERSION=11.8.0 \
    -t pixeltovoxel:cuda11 \
    -f docker/Dockerfile .
```

### Run Container

```bash
# Basic run with GPU
docker run --gpus all -it --rm \
    -v $(pwd):/app \
    pixeltovoxel:latest

# With GUI support (X11)
xhost +local:docker
docker run --gpus all -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/app \
    pixeltovoxel:latest

# With Jupyter Lab
docker run --gpus all -p 8888:8888 -it --rm \
    -v $(pwd):/app \
    pixeltovoxel:latest \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Specific GPU selection
docker run --gpus '"device=0"' -it --rm \
    -v $(pwd):/app \
    pixeltovoxel:latest
```

### Docker Compose

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Start specific service
docker-compose -f docker/docker-compose.yml up pixeltovoxel

# With custom directories
DATA_DIR=/path/to/data OUTPUT_DIR=/path/to/output \
    docker-compose -f docker/docker-compose.yml up

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop all services
docker-compose -f docker/docker-compose.yml down
```

---

## Troubleshooting

### CUDA Not Found

**Problem**: `CUDA not found` during build

**Solutions**:
```bash
# Set CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### GPU Not Detected

**Problem**: GPU not accessible in Docker

**Solutions**:
```bash
# Install nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Compilation Errors

**Problem**: C++ compilation fails

**Solutions**:
```bash
# Update compiler
sudo apt-get install -y build-essential g++-10

# Use specific compiler
export CC=gcc-10
export CXX=g++-10

# Clean and rebuild
rm -rf build/
pip install -e . --force-reinstall --no-cache-dir
```

### Memory Errors

**Problem**: Out of memory during compilation

**Solutions**:
```bash
# Limit parallel jobs
export MAX_JOBS=4
pip install -e .

# Use swap space
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Protocol Buffer Errors

**Problem**: `protoc` command not found or protobuf import errors

**Solutions**:
```bash
# Install protocol buffers
sudo apt-get install -y protobuf-compiler libprotobuf-dev
pip install protobuf grpcio grpcio-tools

# Recompile protocol buffers
cd src/protocols
protoc --python_out=. *.proto
```

### Python Import Errors

**Problem**: Cannot import compiled extensions

**Solutions**:
```bash
# Check installation
pip show pixeltovoxelprojector

# Set PYTHONPATH
export PYTHONPATH=/path/to/Pixeltovoxelprojector:$PYTHONPATH

# Reinstall in development mode
pip install -e . --force-reinstall
```

---

## Development Setup

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/Pixeltovoxelprojector.git
cd Pixeltovoxelprojector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,full]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/benchmarks/test_installation.py

# Run with coverage
pytest --cov=src tests/

# Run benchmarks
python tests/benchmarks/run_all_benchmarks.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/

# Run all checks
./scripts/check_code_quality.sh  # If available
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs
make html

# View documentation
python -m http.server -d _build/html
```

---

## Performance Optimization

### CPU Optimization

```bash
# Build with native architecture optimization
CFLAGS="-march=native -O3" pip install -e .

# Enable OpenMP
export OMP_NUM_THREADS=16  # Set to your CPU core count
```

### GPU Optimization

```bash
# Set compute architecture for your GPU
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3090
# or
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090

# Enable TensorFloat-32
export NVIDIA_TF32_OVERRIDE=1

# Optimize CUDA compilation
export CUDA_LAUNCH_BLOCKING=0
```

### Memory Optimization

```bash
# Reduce memory usage during build
export MAX_JOBS=4

# Use shared memory for data transfer
export USE_SHM=1

# Set CUDA memory fraction
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## Next Steps

After successful installation:

1. **Verify Installation**: Run `python tests/benchmarks/test_installation.py`
2. **Run Examples**: Try examples in the `examples/` directory
3. **Read Documentation**: Check `src/README.md` for usage instructions
4. **Run Benchmarks**: Execute `python tests/benchmarks/run_all_benchmarks.py`

---

## Additional Resources

- [Project README](README.md)
- [API Documentation](docs/)
- [Examples](examples/)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Issue Tracker](https://github.com/yourusername/Pixeltovoxelprojector/issues)

---

## Support

For build issues:
1. Check this document first
2. Search existing issues on GitHub
3. Create a new issue with:
   - Your system configuration
   - Complete error messages
   - Build commands used
   - Output of `nvidia-smi`, `nvcc --version`, `python --version`

---

**Last Updated**: 2025-01-13
**Version**: 1.0.0
