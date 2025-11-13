#!/bin/bash
#
# Build script for Calibration System
#
# This script builds the C++ and CUDA components of the calibration system
#

set -e  # Exit on error

echo "========================================"
echo "Building Calibration System"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA (nvcc) not found${NC}"
    echo "Please install CUDA toolkit"
    exit 1
fi
echo -e "${GREEN}✓ CUDA found:${NC} $(nvcc --version | grep release)"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake not found${NC}"
    echo "Please install CMake 3.18+"
    exit 1
fi
echo -e "${GREEN}✓ CMake found:${NC} $(cmake --version | head -n1)"

# Check Eigen3
if [ ! -d "/usr/include/eigen3" ] && [ ! -d "/usr/local/include/eigen3" ]; then
    echo -e "${YELLOW}Warning: Eigen3 headers not found in standard locations${NC}"
    echo "Install with: sudo apt-get install libeigen3-dev"
fi

# Check OpenCV (optional for C++ components)
if pkg-config --exists opencv4; then
    echo -e "${GREEN}✓ OpenCV found:${NC} $(pkg-config --modversion opencv4)"
else
    echo -e "${YELLOW}Warning: OpenCV not found (optional for C++)${NC}"
fi

# Create build directory
echo -e "\n${YELLOW}Creating build directory...${NC}"
BUILD_DIR="build"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
echo -e "\n${YELLOW}Configuring with CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo -e "\n${YELLOW}Building...${NC}"
make -j$(nproc)

# Check build success
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================"
    echo "Build successful!"
    echo "========================================${NC}"

    echo -e "\nBuilt libraries:"
    ls -lh *.so 2>/dev/null || ls -lh *.dylib 2>/dev/null || echo "  (check build directory for libraries)"

    echo -e "\nBuilt executables:"
    ls -lh *_example 2>/dev/null || echo "  (no example executables built)"

    echo -e "\n${YELLOW}Running tests...${NC}"
    if [ -f "./coordinate_transformer_example" ]; then
        ./coordinate_transformer_example
    fi

    if [ -f "./transform_optimizer_example" ]; then
        ./transform_optimizer_example
    fi

    echo -e "\n${GREEN}To install libraries:${NC}"
    echo "  sudo make install"

else
    echo -e "\n${RED}========================================"
    echo "Build failed!"
    echo "========================================${NC}"
    exit 1
fi

cd ..

# Build Python components
echo -e "\n${YELLOW}Setting up Python environment...${NC}"
echo "Installing Python dependencies..."

if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt --user
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}Warning: pip3 not found, skipping Python dependencies${NC}"
fi

# Create symlink for easy access
if [ -f "build/libcoordinate_transformer.so" ]; then
    ln -sf build/libcoordinate_transformer.so . 2>/dev/null || true
fi

if [ -f "build/libtransform_optimizer.so" ]; then
    ln -sf build/libtransform_optimizer.so . 2>/dev/null || true
fi

echo -e "\n${GREEN}========================================"
echo "Setup complete!"
echo "========================================${NC}"

echo -e "\nQuick start:"
echo "  1. Python calibration:"
echo "     python3 camera_calibrator.py"
echo ""
echo "  2. C++ coordinate transformer:"
echo "     ./build/coordinate_transformer_example"
echo ""
echo "  3. CUDA transform optimizer:"
echo "     ./build/transform_optimizer_example"
echo ""
echo "For detailed instructions, see CALIBRATION_PROCEDURES.md"
