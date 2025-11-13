#!/bin/bash
# Build script for Camera Tracking System

set -e  # Exit on error

echo "=================================="
echo "Camera Tracking System - Build"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
echo "Checking required tools..."

# Check for g++ or clang++
if command -v g++ &> /dev/null; then
    CXX=g++
    echo -e "${GREEN}✓${NC} Found g++"
elif command -v clang++ &> /dev/null; then
    CXX=clang++
    echo -e "${GREEN}✓${NC} Found clang++"
else
    echo -e "${RED}✗${NC} No C++ compiler found (g++ or clang++)"
    exit 1
fi

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
    echo -e "${GREEN}✓${NC} Found Python 3"
else
    echo -e "${RED}✗${NC} Python 3 not found"
    exit 1
fi

# Check for CMake (optional)
if command -v cmake &> /dev/null; then
    HAS_CMAKE=1
    echo -e "${GREEN}✓${NC} Found CMake"
else
    HAS_CMAKE=0
    echo -e "${YELLOW}⚠${NC} CMake not found (will use direct compilation)"
fi

echo ""

# Build C++ Orientation Manager
echo "Building orientation_manager..."

if [ $HAS_CMAKE -eq 1 ]; then
    # Use CMake
    echo "Using CMake build system"
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc)
    cd ..
    echo -e "${GREEN}✓${NC} Built with CMake"
else
    # Direct compilation
    echo "Using direct compilation"
    $CXX -O3 -std=c++17 -fopenmp orientation_manager.cpp -o orientation_manager -lpthread
    echo -e "${GREEN}✓${NC} Built with direct compilation"
fi

echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
$PYTHON -m pip install -r requirements.txt --quiet
echo -e "${GREEN}✓${NC} Python dependencies installed"

echo ""

# Optional: Build Python bindings
read -p "Build Python bindings for orientation_manager? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v pybind11-config &> /dev/null; then
        echo "Building Python bindings..."
        PYBIND11_INCLUDES=$(pybind11-config --includes)
        PYTHON_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
        $CXX -O3 -Wall -shared -std=c++17 -fPIC $PYBIND11_INCLUDES \
            orientation_manager.cpp \
            -o orientation_manager_cpp$PYTHON_SUFFIX
        echo -e "${GREEN}✓${NC} Python bindings built"
    else
        echo -e "${YELLOW}⚠${NC} pybind11 not found - skipping Python bindings"
        echo "Install with: pip install pybind11"
    fi
fi

echo ""
echo "=================================="
echo "Build Complete!"
echo "=================================="
echo ""
echo "Quick test:"
echo "  1. C++ executable: ./orientation_manager"
if [ $HAS_CMAKE -eq 1 ]; then
    echo "     (or ./build/orientation_manager)"
fi
echo "  2. Python example: python ../../examples/camera_tracking_example.py"
echo ""
