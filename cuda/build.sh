#!/bin/bash
#
# Build script for CUDA voxel processing module
#
# Usage:
#   ./build.sh [--clean] [--verbose]
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "CUDA Voxel Module Build Script"
echo "=================================="
echo ""

# Parse arguments
CLEAN=0
VERBOSE=0

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean     Clean build artifacts before building"
            echo "  --verbose   Show detailed compilation output"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

# Check for CUDA
echo "Checking for CUDA installation..."

if [ -z "$CUDA_HOME" ]; then
    # Try to find CUDA
    for cuda_path in /usr/local/cuda /usr/local/cuda-12.0 /usr/local/cuda-11.8 /opt/cuda; do
        if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
            export CUDA_HOME="$cuda_path"
            echo "Found CUDA at: $CUDA_HOME"
            break
        fi
    done
fi

if [ -z "$CUDA_HOME" ] || [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: CUDA not found!"
    echo "Please set CUDA_HOME environment variable or install CUDA toolkit"
    echo ""
    echo "Example:"
    echo "  export CUDA_HOME=/usr/local/cuda-12.0"
    exit 1
fi

# Add CUDA to PATH
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Check nvcc version
echo ""
echo "CUDA Compiler Information:"
nvcc --version | grep "release"

# Check GPU
echo ""
echo "Detecting GPUs..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | \
        nl -w2 -s'. ' || echo "No NVIDIA GPUs detected"
else
    echo "nvidia-smi not found, skipping GPU detection"
fi

# Clean if requested
if [ $CLEAN -eq 1 ]; then
    echo ""
    echo "Cleaning build artifacts..."
    cd "$PROJECT_DIR"
    rm -rf build/
    rm -rf *.egg-info/
    rm -f *.so
    rm -f cuda/*.o
    echo "Clean complete"
fi

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."

python3 -c "import numpy" 2>/dev/null || {
    echo "ERROR: NumPy not installed"
    echo "Install with: pip install numpy"
    exit 1
}

python3 -c "import pybind11" 2>/dev/null || {
    echo "ERROR: pybind11 not installed"
    echo "Install with: pip install pybind11"
    exit 1
}

echo "All dependencies found"

# Build
echo ""
echo "=================================="
echo "Building CUDA Extension"
echo "=================================="
echo ""

cd "$PROJECT_DIR"

if [ $VERBOSE -eq 1 ]; then
    python3 setup.py build_ext --inplace
else
    python3 setup.py build_ext --inplace 2>&1 | grep -E "(Compiling|Linking|CUDA|error|warning)" || true
fi

# Verify build
echo ""
echo "=================================="
echo "Verifying Build"
echo "=================================="
echo ""

if python3 -c "import voxel_cuda; print('✓ CUDA module loaded successfully')" 2>/dev/null; then
    echo ""
    python3 -c "import voxel_cuda; voxel_cuda.print_device_info()"
    echo ""
    echo "=================================="
    echo "Build Complete!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run example: python3 cuda/example_cuda_usage.py"
    echo "  2. Run benchmark: python3 cuda/example_cuda_usage.py --benchmark"
    echo "  3. Process 8K: python3 cuda/example_cuda_usage.py --8k --num-cameras 10"
    echo ""
else
    echo "ERROR: Build completed but module failed to load"
    echo ""
    echo "Try running with --verbose for more details:"
    echo "  ./build.sh --verbose"
    exit 1
fi
