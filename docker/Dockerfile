# ============================================================================
# Pixel to Voxel Projector - Docker Image
# CUDA-enabled development and production environment
# ============================================================================

# Base image: NVIDIA CUDA with cuDNN
# Change to cuda:11.8.0-cudnn8-devel-ubuntu22.04 for CUDA 11.x
FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04

# Metadata
LABEL maintainer="Voxel Processing Team"
LABEL description="High-performance 8K video motion tracking and voxel processing system"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# ============================================================================
# System Dependencies
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    ca-certificates \
    \
    # Python
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    \
    # OpenMP and parallel computing
    libomp-dev \
    \
    # FFmpeg and video processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    \
    # OpenCV dependencies
    libopencv-dev \
    libgtk-3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    \
    # OpenGL and visualization
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-utils \
    libglfw3 \
    libglfw3-dev \
    \
    # X11 for GUI applications
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    libxcursor-dev \
    x11-apps \
    \
    # Protocol Buffers
    protobuf-compiler \
    libprotobuf-dev \
    \
    # Compression libraries
    liblz4-dev \
    libzstd-dev \
    libsnappy-dev \
    \
    # Networking
    libzmq3-dev \
    \
    # HDF5
    libhdf5-dev \
    \
    # System utilities
    htop \
    iotop \
    vim \
    nano \
    tmux \
    screen \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Python Environment
# ============================================================================

# Update pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python build tools
RUN pip install --no-cache-dir \
    cmake \
    ninja \
    pybind11

# ============================================================================
# Application Setup
# ============================================================================

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA-accelerated packages
# Adjust cupy version based on CUDA version in base image
RUN pip install --no-cache-dir cupy-cuda12x || \
    echo "Warning: cupy installation failed, continuing without GPU Python acceleration"

# Install optional dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipykernel \
    matplotlib \
    seaborn || \
    echo "Warning: Some optional packages failed to install"

# ============================================================================
# Copy Application Code
# ============================================================================

# Copy source code
COPY setup.py /app/
COPY CMakeLists.txt /app/
COPY src/ /app/src/
COPY cuda/ /app/cuda/
COPY tests/ /app/tests/
COPY examples/ /app/examples/
COPY *.cpp /app/
COPY *.py /app/

# ============================================================================
# Build C++ and CUDA Extensions
# ============================================================================

# Build with setup.py (includes CUDA extensions)
RUN pip install -e . || \
    echo "Warning: Some extensions failed to build"

# Alternative: Build with CMake for better control
RUN mkdir -p build && cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CUDA=ON \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DUSE_OPENMP=ON \
        -DENABLE_FAST_MATH=ON && \
    ninja && \
    ninja install || \
    echo "Warning: CMake build failed"

# ============================================================================
# Protocol Buffers Compilation
# ============================================================================

RUN if [ -d "src/protocols" ]; then \
        cd src/protocols && \
        for proto in *.proto; do \
            protoc --python_out=. --grpc_python_out=. -I. "$proto" || true; \
        done; \
    fi

# ============================================================================
# Runtime Configuration
# ============================================================================

# Create directories for data and output
RUN mkdir -p /data /output /logs

# Set Python path
ENV PYTHONPATH=/app:/app/src:${PYTHONPATH}

# Expose ports for visualization and networking
EXPOSE 8888 5555 5556 6006

# ============================================================================
# GPU Configuration Check
# ============================================================================

# Add a script to verify GPU access
RUN echo '#!/bin/bash\n\
echo "================================================="\n\
echo "GPU Configuration Check"\n\
echo "================================================="\n\
nvidia-smi || echo "Warning: nvidia-smi not available"\n\
python3 -c "import torch; print(f\"PyTorch CUDA available: {torch.cuda.is_available()}\")" 2>/dev/null || echo "PyTorch not installed"\n\
python3 -c "import cupy as cp; print(f\"CuPy CUDA available: {cp.cuda.is_available()}\"); print(f\"Device: {cp.cuda.Device()}\")" || echo "CuPy not available"\n\
echo "================================================="\n\
' > /usr/local/bin/check_gpu.sh && chmod +x /usr/local/bin/check_gpu.sh

# ============================================================================
# Entrypoint and Command
# ============================================================================

# Set entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh || echo "#!/bin/bash\nexec \"\$@\"" > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]

# ============================================================================
# Build Instructions
# ============================================================================
#
# Build the image:
#   docker build -t pixeltovoxel:latest -f docker/Dockerfile .
#
# Run with GPU support:
#   docker run --gpus all -it --rm \
#       -v $(pwd):/app \
#       -v /data:/data \
#       pixeltovoxel:latest
#
# Run with specific GPU:
#   docker run --gpus '"device=0"' -it --rm \
#       -v $(pwd):/app \
#       pixeltovoxel:latest
#
# Run Jupyter Lab:
#   docker run --gpus all -p 8888:8888 -it --rm \
#       -v $(pwd):/app \
#       pixeltovoxel:latest \
#       jupyter lab --ip=0.0.0.0 --allow-root --no-browser
#
# Run with X11 for GUI:
#   docker run --gpus all -it --rm \
#       -e DISPLAY=$DISPLAY \
#       -v /tmp/.X11-unix:/tmp/.X11-unix \
#       -v $(pwd):/app \
#       pixeltovoxel:latest
#
# ============================================================================
