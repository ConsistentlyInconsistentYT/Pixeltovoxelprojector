#!/bin/bash
# Build script for optimized voxel grid system

set -e  # Exit on error

echo "=========================================="
echo "Building Voxel Grid System"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 not found${NC}"
        echo "Please install $1 first"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Found $1"
}

echo -e "\n${YELLOW}Checking prerequisites...${NC}"
check_tool g++
check_tool nvcc

# Create output directories
mkdir -p lib
mkdir -p bin

# Build C++ Sparse Voxel Grid
echo -e "\n${YELLOW}Building C++ Sparse Voxel Grid...${NC}"
g++ -std=c++17 -O3 -march=native -fPIC -shared \
    sparse_voxel_grid.cpp \
    -o lib/libsparsevoxel.so

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Built lib/libsparsevoxel.so"
else
    echo -e "${RED}✗${NC} Failed to build sparse voxel grid"
    exit 1
fi

# Build C++ example
echo -e "\n${YELLOW}Building C++ example...${NC}"
g++ -std=c++17 -O3 -march=native \
    -DBUILD_EXAMPLE \
    sparse_voxel_grid.cpp \
    -o bin/sparse_voxel_example

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Built bin/sparse_voxel_example"
else
    echo -e "${RED}✗${NC} Failed to build C++ example"
    exit 1
fi

# Detect GPU architecture
echo -e "\n${YELLOW}Detecting GPU architecture...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "Detected GPU: $GPU_NAME"

    # Determine compute capability
    if [[ $GPU_NAME == *"RTX 40"* ]]; then
        SM_ARCH="sm_89"
    elif [[ $GPU_NAME == *"RTX 30"* ]]; then
        SM_ARCH="sm_86"
    elif [[ $GPU_NAME == *"RTX 20"* ]] || [[ $GPU_NAME == *"GTX 16"* ]]; then
        SM_ARCH="sm_75"
    elif [[ $GPU_NAME == *"V100"* ]]; then
        SM_ARCH="sm_70"
    else
        SM_ARCH="sm_75"  # Default
        echo -e "${YELLOW}Warning: Unknown GPU, using default architecture sm_75${NC}"
    fi
    echo "Using compute capability: $SM_ARCH"
else
    echo -e "${YELLOW}Warning: nvidia-smi not found, using default sm_75${NC}"
    SM_ARCH="sm_75"
fi

# Build CUDA Voxel Optimizer
echo -e "\n${YELLOW}Building CUDA Voxel Optimizer...${NC}"
nvcc -std=c++14 -O3 -arch=$SM_ARCH -Xcompiler -fPIC --shared \
    voxel_optimizer.cu \
    -o lib/libvoxeloptimizer.so

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Built lib/libvoxeloptimizer.so"
else
    echo -e "${RED}✗${NC} Failed to build CUDA optimizer"
    echo "Note: This requires NVIDIA GPU and CUDA toolkit"
    echo "You can still use the C++ sparse grid without CUDA"
fi

# Build CUDA example
echo -e "\n${YELLOW}Building CUDA example...${NC}"
nvcc -std=c++14 -O3 -arch=$SM_ARCH \
    -DBUILD_EXAMPLE \
    voxel_optimizer.cu \
    -o bin/voxel_optimizer_example

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Built bin/voxel_optimizer_example"
else
    echo -e "${RED}✗${NC} Failed to build CUDA example"
fi

# Check Python dependencies
echo -e "\n${YELLOW}Checking Python dependencies...${NC}"
python3 -c "import numpy" 2>/dev/null && echo -e "${GREEN}✓${NC} numpy installed" || echo -e "${YELLOW}⚠${NC} numpy not found (pip install numpy)"
python3 -c "import threading" 2>/dev/null && echo -e "${GREEN}✓${NC} threading available"
python3 -c "import dataclasses" 2>/dev/null && echo -e "${GREEN}✓${NC} dataclasses available"

# Create symlinks for easy access
echo -e "\n${YELLOW}Creating convenience symlinks...${NC}"
ln -sf lib/libsparsevoxel.so libsparsevoxel.so 2>/dev/null || true
ln -sf lib/libvoxeloptimizer.so libvoxeloptimizer.so 2>/dev/null || true

echo -e "\n${GREEN}=========================================="
echo "Build Complete!"
echo "==========================================${NC}"
echo ""
echo "Libraries:"
echo "  lib/libsparsevoxel.so      - C++ sparse voxel grid"
echo "  lib/libvoxeloptimizer.so   - CUDA optimizer"
echo ""
echo "Examples:"
echo "  bin/sparse_voxel_example   - C++ example"
echo "  bin/voxel_optimizer_example - CUDA example"
echo "  grid_manager.py            - Python manager"
echo ""
echo "To run examples:"
echo "  ./bin/sparse_voxel_example"
echo "  ./bin/voxel_optimizer_example"
echo "  python3 grid_manager.py"
echo ""
