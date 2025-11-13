#!/bin/bash
#
# MTRTP Build Script
# Automated build and installation for Motion Tracking Real-Time Protocol
#
# Usage:
#   ./build.sh [options]
#
# Options:
#   --clean          Clean build directory
#   --install        Install after building
#   --tests          Build and run tests
#   --release        Release build (default is Debug)
#   --python         Setup Python environment
#   --deps           Install dependencies
#   --all            Do everything (deps, build, tests, install)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Default options
DO_CLEAN=0
DO_INSTALL=0
DO_TESTS=0
BUILD_TYPE="Debug"
DO_PYTHON=0
DO_DEPS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            DO_CLEAN=1
            shift
            ;;
        --install)
            DO_INSTALL=1
            shift
            ;;
        --tests)
            DO_TESTS=1
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --python)
            DO_PYTHON=1
            shift
            ;;
        --deps)
            DO_DEPS=1
            shift
            ;;
        --all)
            DO_DEPS=1
            DO_CLEAN=1
            DO_TESTS=1
            DO_INSTALL=1
            DO_PYTHON=1
            BUILD_TYPE="Release"
            shift
            ;;
        --help|-h)
            head -n 15 "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}"
echo "=========================================="
echo "  MTRTP Build Script"
echo "=========================================="
echo -e "${NC}"

# Install dependencies
if [ $DO_DEPS -eq 1 ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"

    if [ -x "$(command -v apt-get)" ]; then
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            protobuf-compiler \
            libprotobuf-dev \
            liblz4-dev \
            libgtest-dev \
            python3 \
            python3-pip \
            python3-dev
    elif [ -x "$(command -v yum)" ]; then
        sudo yum install -y \
            gcc-c++ \
            cmake \
            protobuf-devel \
            lz4-devel \
            gtest-devel \
            python3 \
            python3-pip \
            python3-devel
    else
        echo -e "${RED}Unsupported package manager. Please install dependencies manually.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Dependencies installed successfully${NC}"
fi

# Setup Python environment
if [ $DO_PYTHON -eq 1 ]; then
    echo -e "${YELLOW}Setting up Python environment...${NC}"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    # Activate and install requirements
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    echo -e "${GREEN}Python environment setup complete${NC}"
fi

# Clean build directory
if [ $DO_CLEAN -eq 1 ] && [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}Clean complete${NC}"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Generate protocol buffers first
echo -e "${YELLOW}Generating protocol buffers...${NC}"
mkdir -p generated

protoc \
    --cpp_out=generated \
    --python_out=generated \
    --proto_path="${SCRIPT_DIR}" \
    "${SCRIPT_DIR}/motion_protocol.proto"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Protocol buffers generated successfully${NC}"
else
    echo -e "${RED}Failed to generate protocol buffers${NC}"
    exit 1
fi

# Configure CMake
echo -e "${YELLOW}Configuring CMake...${NC}"
cmake \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DBUILD_TESTS=$([ $DO_TESTS -eq 1 ] && echo "ON" || echo "OFF") \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    ..

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
fi

# Build
echo -e "${YELLOW}Building MTRTP...${NC}"
NPROC=$(nproc 2>/dev/null || echo 4)
make -j${NPROC}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful${NC}"
else
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

# Run tests
if [ $DO_TESTS -eq 1 ]; then
    echo -e "${YELLOW}Running tests...${NC}"

    if [ -f "protocol_tests" ]; then
        ./protocol_tests
    else
        echo -e "${YELLOW}No C++ tests found${NC}"
    fi

    # Run Python tests if available
    cd "${SCRIPT_DIR}"
    if [ -f "tests/test_subscriber.py" ]; then
        if [ $DO_PYTHON -eq 1 ]; then
            source venv/bin/activate
        fi
        python3 -m pytest tests/
    fi

    echo -e "${GREEN}Tests complete${NC}"
fi

# Install
if [ $DO_INSTALL -eq 1 ]; then
    echo -e "${YELLOW}Installing MTRTP...${NC}"
    cd "$BUILD_DIR"
    sudo make install

    # Install Python package
    cd "${SCRIPT_DIR}"
    if [ $DO_PYTHON -eq 1 ]; then
        source venv/bin/activate
    fi
    pip install -e .

    echo -e "${GREEN}Installation complete${NC}"
fi

# Print summary
echo -e "${BLUE}"
echo "=========================================="
echo "  Build Summary"
echo "=========================================="
echo -e "${NC}"
echo -e "Build Type:    ${BUILD_TYPE}"
echo -e "Build Dir:     ${BUILD_DIR}"
echo -e "Install:       $([ $DO_INSTALL -eq 1 ] && echo 'Yes' || echo 'No')"
echo -e "Tests:         $([ $DO_TESTS -eq 1 ] && echo 'Yes' || echo 'No')"
echo -e ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e ""
echo -e "Next steps:"
echo -e "  1. Run examples: cd examples && python motion_subscriber_example.py"
echo -e "  2. Run benchmarks: cd examples && python benchmark.py"
echo -e "  3. Read documentation: cat README.md"
echo -e ""
