#!/bin/bash
# Build script for thermal_mono_fusion C++ module

set -e  # Exit on error

echo "======================================"
echo "Thermal + Mono Fusion Build Script"
echo "======================================"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

# Check for required packages
echo "Checking dependencies..."
python3 -c "import pybind11" 2>/dev/null || {
    echo "Installing pybind11..."
    pip install pybind11
}

python3 -c "import numpy" 2>/dev/null || {
    echo "Installing numpy..."
    pip install numpy
}

python3 -c "import scipy" 2>/dev/null || {
    echo "Installing scipy..."
    pip install scipy
}

echo "Dependencies OK"
echo ""

# Check for CUDA
echo "Checking for CUDA..."
if command -v nvcc &> /dev/null; then
    echo "CUDA found: $(nvcc --version | grep release)"
    CUDA_AVAILABLE=1
else
    echo "CUDA not found - building CPU-only version"
    CUDA_AVAILABLE=0
fi
echo ""

# Build
echo "Building thermal_mono_fusion module..."
cd "$(dirname "$0")"

if [ -f thermal_mono_fusion*.so ]; then
    echo "Cleaning old builds..."
    rm -f thermal_mono_fusion*.so
    rm -rf build/
fi

python3 setup_fusion.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Build successful!"
    echo "======================================"
    echo ""

    # Test the module
    echo "Testing module import..."
    python3 -c "import thermal_mono_fusion; print('Module imported successfully')"

    if [ $? -eq 0 ]; then
        echo ""
        echo "Module is ready to use!"
        echo ""
        echo "Run tests with:"
        echo "  python3 fusion_demo.py --test all"
        echo ""
        echo "Or import in Python:"
        echo "  import thermal_mono_fusion"
        echo ""
    else
        echo "Warning: Module built but import failed"
        exit 1
    fi
else
    echo ""
    echo "======================================"
    echo "Build failed!"
    echo "======================================"
    exit 1
fi
