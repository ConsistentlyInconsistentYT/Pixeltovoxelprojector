#!/bin/bash
# build.sh - Build script for drone detection system

set -e  # Exit on error

echo "========================================"
echo "Drone Detection System - Build Script"
echo "========================================"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA (nvcc) not found. Please install CUDA Toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "Found CUDA version: $CUDA_VERSION"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake not found. Please install CMake 3.18+."
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
echo "Found CMake version: $CMAKE_VERSION"

# Detect GPU architecture
echo ""
echo "Detecting GPU architecture..."

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "Detected GPU: $GPU_NAME"

    # Set architecture based on GPU
    if [[ "$GPU_NAME" == *"RTX 40"* ]] || [[ "$GPU_NAME" == *"4090"* ]] || [[ "$GPU_NAME" == *"4080"* ]]; then
        CUDA_ARCH="89"
    elif [[ "$GPU_NAME" == *"RTX 30"* ]] || [[ "$GPU_NAME" == *"3090"* ]] || [[ "$GPU_NAME" == *"3080"* ]]; then
        CUDA_ARCH="86"
    elif [[ "$GPU_NAME" == *"A100"* ]]; then
        CUDA_ARCH="80"
    elif [[ "$GPU_NAME" == *"RTX 20"* ]] || [[ "$GPU_NAME" == *"2080"* ]]; then
        CUDA_ARCH="75"
    else
        CUDA_ARCH="75;80;86;89"  # Build for all common architectures
        echo "Warning: Unknown GPU, building for multiple architectures (slower build)"
    fi
else
    CUDA_ARCH="75;80;86;89"
    echo "Warning: nvidia-smi not found, building for multiple architectures"
fi

echo "Target CUDA architecture: $CUDA_ARCH"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo ""
    read -p "Build directory exists. Clean rebuild? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo ""
echo "Configuring project..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math"

# Build
echo ""
echo "Building (using $(nproc) parallel jobs)..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "BUILD SUCCESSFUL!"
    echo "========================================"
    echo ""
    echo "Library built: $(pwd)/drone_detector*.so"
    echo ""
    echo "To install system-wide:"
    echo "  sudo make install"
    echo ""
    echo "To test:"
    echo "  cd .."
    echo "  python test_detection.py"
    echo ""
else
    echo ""
    echo "========================================"
    echo "BUILD FAILED!"
    echo "========================================"
    exit 1
fi
