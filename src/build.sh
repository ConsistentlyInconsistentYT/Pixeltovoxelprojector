#!/bin/bash
# Build script for 8K Video Processing Pipeline

set -e  # Exit on error

echo "========================================"
echo "8K Video Processing Pipeline - Build"
echo "========================================"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "Dependencies installed!"

# Install pybind11
echo ""
echo "[3/5] Installing pybind11..."
pip install pybind11 --quiet
echo "pybind11 installed!"

# Build C++ motion extractor module
echo ""
echo "[4/5] Building C++ motion extractor module..."
python3 setup_motion_extractor.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "C++ module built successfully!"
else
    echo "Warning: C++ module build failed. Python fallback will be used."
fi

# Check system capabilities
echo ""
echo "[5/5] Checking system capabilities..."
echo ""

# Check OpenCV
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>/dev/null || echo "OpenCV: Not installed"

# Check NumPy
python3 -c "import numpy; print('NumPy version:', numpy.__version__)" 2>/dev/null || echo "NumPy: Not installed"

# Check CUDA support
python3 -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null || echo "CUDA: Not available"

# Check if C++ module loaded
python3 -c "from motion_extractor_cpp import MotionExtractor8K; print('C++ Motion Extractor: Available')" 2>/dev/null || echo "C++ Motion Extractor: Not available (using Python fallback)"

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run tests: make test"
echo "  2. Run benchmarks: make benchmark"
echo "  3. Run examples: python example_8k_pipeline.py"
echo ""
