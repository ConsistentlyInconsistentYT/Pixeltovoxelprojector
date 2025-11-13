#!/bin/bash
# ============================================================================
# Docker Entrypoint Script
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pixel to Voxel Projector Container${NC}"
echo -e "${GREEN}========================================${NC}"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n${GREEN}✓ GPU Status:${NC}"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || echo -e "${YELLOW}⚠ GPU information unavailable${NC}"
else
    echo -e "\n${YELLOW}⚠ nvidia-smi not found - GPU may not be available${NC}"
fi

# Check CUDA
if [ -f "/usr/local/cuda/version.txt" ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt)
    echo -e "\n${GREEN}✓ CUDA Version:${NC} $CUDA_VERSION"
elif [ -f "/usr/local/cuda/version.json" ]; then
    CUDA_VERSION=$(grep -oP '(?<="version": ")[^"]*' /usr/local/cuda/version.json | head -1)
    echo -e "\n${GREEN}✓ CUDA Version:${NC} $CUDA_VERSION"
else
    echo -e "\n${YELLOW}⚠ CUDA version information not found${NC}"
fi

# Check Python packages
echo -e "\n${GREEN}Python Environment:${NC}"
python3 --version
echo "Installed packages:"
pip list | grep -E "(numpy|opencv|cupy|pybind11|protobuf|torch)" || echo "Core packages not found"

# Run GPU check script if available
if [ -f "/usr/local/bin/check_gpu.sh" ]; then
    echo ""
    /usr/local/bin/check_gpu.sh
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Container Ready${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Execute the main command
exec "$@"
