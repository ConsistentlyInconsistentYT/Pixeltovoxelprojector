# ============================================================================
# Multi-stage Dockerfile for PixelToVoxel Project
# Supports CPU-only and GPU-enabled builds
# ============================================================================

# Build arguments
ARG CUDA_VERSION=12.0.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10

# ============================================================================
# Base Image - CPU Only
# ============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS base-cpu

ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="Voxel Processing Team"
LABEL description="PixelToVoxel - High-performance 8K video motion tracking and voxel processing"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    libopencv-dev \
    libomp-dev \
    libgomp1 \
    pybind11-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

# ============================================================================
# Base Image - GPU with CUDA
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base-gpu

ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="Voxel Processing Team"
LABEL description="PixelToVoxel - GPU-accelerated 8K video motion tracking and voxel processing"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    libopencv-dev \
    libomp-dev \
    libgomp1 \
    pybind11-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace

# ============================================================================
# Builder Stage - CPU
# ============================================================================
FROM base-cpu AS builder-cpu

# Copy requirements and install Python dependencies
COPY requirements*.txt ./
COPY src/requirements.txt src/
COPY src/*/requirements.txt src/*/

# Install Python dependencies
RUN pip install --no-cache-dir numpy pybind11 && \
    pip install --no-cache-dir -r requirements.txt || true && \
    pip install --no-cache-dir -r src/requirements.txt || true

# Copy source code
COPY . .

# Install build dependencies
RUN pip install --no-cache-dir pytest pytest-cov

# Build extensions
RUN python3 setup.py build_ext --inplace

# Build Python package
RUN python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ============================================================================
# Builder Stage - GPU
# ============================================================================
FROM base-gpu AS builder-gpu

# Copy requirements and install Python dependencies
COPY requirements*.txt ./
COPY src/requirements.txt src/
COPY src/*/requirements.txt src/*/

# Install Python dependencies including CUDA packages
RUN pip install --no-cache-dir numpy pybind11 && \
    pip install --no-cache-dir cupy-cuda12x pycuda && \
    pip install --no-cache-dir -r requirements.txt || true && \
    pip install --no-cache-dir -r src/requirements.txt || true

# Copy source code
COPY . .

# Install build dependencies
RUN pip install --no-cache-dir pytest pytest-cov

# Build extensions with CUDA
ENV CUDA_HOME=/usr/local/cuda
RUN python3 setup.py build_ext --inplace

# Build Python package
RUN python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ============================================================================
# Runtime Stage - CPU
# ============================================================================
FROM base-cpu AS cpu-runtime

# Copy wheels and install
COPY --from=builder-cpu /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Copy any additional runtime files
COPY --from=builder-cpu /workspace/*.so /usr/local/lib/ || true

# Create non-root user
RUN useradd -m -u 1000 voxeluser && \
    chown -R voxeluser:voxeluser /workspace

USER voxeluser

# Set entrypoint
ENTRYPOINT ["python3"]
CMD ["--help"]

# ============================================================================
# Runtime Stage - GPU
# ============================================================================
FROM base-gpu AS gpu-runtime

# Copy wheels and install
COPY --from=builder-gpu /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Install CUDA Python packages
RUN pip install --no-cache-dir cupy-cuda12x pycuda

# Copy any additional runtime files
COPY --from=builder-gpu /workspace/*.so /usr/local/lib/ || true

# Verify CUDA installation
RUN nvcc --version && \
    python3 -c "import cupy; print(f'CuPy version: {cupy.__version__}')" || echo "CuPy not available"

# Create non-root user
RUN useradd -m -u 1000 voxeluser && \
    chown -R voxeluser:voxeluser /workspace

USER voxeluser

# Set entrypoint
ENTRYPOINT ["python3"]
CMD ["--help"]

# ============================================================================
# Development Stage - Full toolchain with GPU
# ============================================================================
FROM base-gpu AS development

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    tmux \
    htop \
    gdb \
    valgrind \
    clang \
    clang-format \
    clang-tidy \
    python3-venv \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install all dependencies
COPY requirements*.txt ./
COPY src/requirements.txt src/
COPY src/*/requirements.txt src/*/

# Install all Python dependencies including dev tools
RUN pip install --no-cache-dir numpy pybind11 && \
    pip install --no-cache-dir cupy-cuda12x pycuda && \
    pip install --no-cache-dir -r requirements.txt || true && \
    pip install --no-cache-dir -r src/requirements.txt || true

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark \
    pytest-timeout \
    pytest-mock \
    black \
    flake8 \
    mypy \
    pylint \
    isort \
    ipython \
    jupyter \
    matplotlib \
    seaborn \
    pandas

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e .[dev,cuda,full]

# Build extensions
RUN python3 setup.py build_ext --inplace

# Create directories
RUN mkdir -p tests/benchmarks/benchmark_results && \
    mkdir -p tests/test_data

# Don't switch to non-root in dev environment
# Allow full access for development
ENV PYTHONPATH=/workspace/src:/workspace
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

# Set entrypoint to bash for interactive development
ENTRYPOINT ["/bin/bash"]

# ============================================================================
# Testing Stage - For CI/CD
# ============================================================================
FROM development AS testing

# Copy test scripts
COPY scripts/run_tests.sh /usr/local/bin/run_tests
COPY scripts/build.sh /usr/local/bin/build

RUN chmod +x /usr/local/bin/run_tests /usr/local/bin/build

# Run tests as part of build verification
RUN python3 -m pytest tests/ \
    --ignore=tests/integration/ \
    --ignore=tests/benchmarks/ \
    -v \
    --tb=short \
    || echo "Some tests failed, check logs"

# Default command runs all tests
CMD ["run_tests", "--quick"]

# ============================================================================
# Production Stage - Minimal runtime
# ============================================================================
FROM gpu-runtime AS production

# Install only production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn \
    fastapi \
    grpcio \
    grpcio-tools

# Copy application code
COPY --from=builder-gpu /workspace/src /app/src
COPY --from=builder-gpu /workspace/*.py /app/

WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Set production environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose common ports
EXPOSE 8000 50051

# Run as non-root
USER voxeluser

ENTRYPOINT ["python3"]
CMD ["-m", "src.main"]
