#!/bin/bash
#
# Comprehensive Build Script for PixelToVoxel Project
#
# Usage:
#   ./build.sh [options]
#
# Options:
#   --clean             Clean all build artifacts before building
#   --cuda              Build with CUDA support (default if CUDA available)
#   --no-cuda           Build without CUDA support
#   --release           Build with release optimizations
#   --debug             Build with debug symbols
#   --verbose           Show detailed compilation output
#   --install           Install after building
#   --dev               Install in development mode
#   --deps              Install dependencies first
#   --proto             Compile protocol buffers
#   --test              Run tests after building
#   --parallel N        Use N parallel jobs (default: number of cores)
#   --help              Show this help message

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
CLEAN_BUILD=false
BUILD_CUDA=true
RELEASE_BUILD=false
DEBUG_BUILD=false
VERBOSE=false
INSTALL=false
DEV_INSTALL=false
INSTALL_DEPS=false
COMPILE_PROTO=false
RUN_TESTS=false
PARALLEL_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --cuda)
            BUILD_CUDA=true
            shift
            ;;
        --no-cuda)
            BUILD_CUDA=false
            shift
            ;;
        --release)
            RELEASE_BUILD=true
            shift
            ;;
        --debug)
            DEBUG_BUILD=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --dev)
            DEV_INSTALL=true
            shift
            ;;
        --deps)
            INSTALL_DEPS=true
            shift
            ;;
        --proto)
            COMPILE_PROTO=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --help|-h)
            head -n 30 "$0" | tail -n +3
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PixelToVoxel Build System${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# ============================================================================
# Environment Check
# ============================================================================
echo -e "${YELLOW}Checking build environment...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check for required build tools
for tool in gcc g++ cmake; do
    if command -v $tool &> /dev/null; then
        VERSION=$($tool --version | head -n1)
        echo -e "${GREEN}✓ $VERSION${NC}"
    else
        echo -e "${RED}✗ $tool not found${NC}"
        exit 1
    fi
done

# Check for CUDA
CUDA_AVAILABLE=false
if [ "$BUILD_CUDA" = true ]; then
    if [ -z "$CUDA_HOME" ]; then
        # Try to find CUDA
        for cuda_path in /usr/local/cuda /usr/local/cuda-12.0 /usr/local/cuda-11.8 /usr/local/cuda-11.0 /opt/cuda; do
            if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
                export CUDA_HOME="$cuda_path"
                break
            fi
        done
    fi

    if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
        CUDA_AVAILABLE=true
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
        NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        echo -e "${GREEN}✓ CUDA $NVCC_VERSION at $CUDA_HOME${NC}"

        # Check GPU
        if command -v nvidia-smi &> /dev/null; then
            echo -e "${GREEN}✓ GPU detected:${NC}"
            nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | nl -w2 -s'. '
        fi
    else
        echo -e "${YELLOW}! CUDA not found${NC}"
        BUILD_CUDA=false
    fi
fi

echo ""

# ============================================================================
# Install Dependencies
# ============================================================================
if [ "$INSTALL_DEPS" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Installing Dependencies${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Upgrade pip
    echo "Upgrading pip..."
    python3 -m pip install --upgrade pip setuptools wheel

    # Install build dependencies
    echo "Installing build dependencies..."
    python3 -m pip install pybind11 numpy

    # Install project dependencies
    if [ -f "requirements.txt" ]; then
        echo "Installing project dependencies..."
        python3 -m pip install -r requirements.txt
    fi

    # Install development dependencies
    if [ "$DEV_INSTALL" = true ]; then
        echo "Installing development dependencies..."
        python3 -m pip install pytest pytest-cov pytest-xdist pytest-benchmark \
            black flake8 mypy pylint isort
    fi

    # Install CUDA dependencies
    if [ "$BUILD_CUDA" = true ]; then
        echo "Installing CUDA Python dependencies..."
        python3 -m pip install cupy-cuda12x pycuda || true
    fi

    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo ""
fi

# ============================================================================
# Clean Build
# ============================================================================
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Cleaning Build Artifacts${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    echo "Removing build directories..."
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf src/*.egg-info/

    echo "Removing compiled files..."
    find . -type f -name "*.so" -delete
    find . -type f -name "*.o" -delete
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    echo "Removing protobuf generated files..."
    find src/protocols -type f -name "*_pb2.py" -delete
    find src/protocols -type f -name "*_pb2_grpc.py" -delete

    echo -e "${GREEN}✓ Clean complete${NC}"
    echo ""
fi

# ============================================================================
# Compile Protocol Buffers
# ============================================================================
if [ "$COMPILE_PROTO" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Compiling Protocol Buffers${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    PROTO_DIR="src/protocols"
    if [ -d "$PROTO_DIR" ]; then
        PROTO_FILES=$(find "$PROTO_DIR" -name "*.proto" 2>/dev/null || true)
        if [ -n "$PROTO_FILES" ]; then
            # Check if protoc is available
            if command -v protoc &> /dev/null; then
                for proto_file in $PROTO_FILES; do
                    echo "Compiling $(basename $proto_file)..."
                    protoc -I="$PROTO_DIR" \
                        --python_out="$PROTO_DIR" \
                        --grpc_python_out="$PROTO_DIR" \
                        "$proto_file"
                done
                echo -e "${GREEN}✓ Protocol buffers compiled${NC}"
            else
                echo -e "${YELLOW}! protoc not found, skipping protocol buffer compilation${NC}"
                echo "Install with: apt-get install protobuf-compiler"
            fi
        else
            echo "No .proto files found"
        fi
    fi
    echo ""
fi

# ============================================================================
# Build Configuration
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Build Configuration${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

BUILD_TYPE="normal"
if [ "$RELEASE_BUILD" = true ]; then
    BUILD_TYPE="release"
    export CFLAGS="-O3 -march=native -ffast-math"
    export CXXFLAGS="-O3 -march=native -ffast-math"
    echo "Build Type: Release (optimized)"
elif [ "$DEBUG_BUILD" = true ]; then
    BUILD_TYPE="debug"
    export CFLAGS="-O0 -g -DDEBUG"
    export CXXFLAGS="-O0 -g -DDEBUG"
    echo "Build Type: Debug (with symbols)"
else
    echo "Build Type: Normal"
fi

echo "CUDA Support: $([ "$BUILD_CUDA" = true ] && echo 'Enabled' || echo 'Disabled')"
echo "Parallel Jobs: $PARALLEL_JOBS"
echo "Python: $PYTHON_VERSION"
echo ""

# ============================================================================
# Build Extensions
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Building Extensions${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

BUILD_CMD="python3 setup.py build_ext --inplace"

# Add parallel build
if [ "$PARALLEL_JOBS" -gt 1 ]; then
    BUILD_CMD="$BUILD_CMD -j $PARALLEL_JOBS"
fi

# Execute build
START_TIME=$(date +%s)

if [ "$VERBOSE" = true ]; then
    echo "Running: $BUILD_CMD"
    echo ""
    $BUILD_CMD
else
    echo "Building extensions (this may take a few minutes)..."
    $BUILD_CMD 2>&1 | grep -E "(Compiling|Linking|Building|error|warning|Error|Warning)" || true
fi

END_TIME=$(date +%s)
BUILD_DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}✓ Build completed in ${BUILD_DURATION}s${NC}"
echo ""

# ============================================================================
# Verify Build
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Verifying Build${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check for built extensions
BUILT_EXTENSIONS=$(find . -maxdepth 1 -name "*.so" 2>/dev/null || true)
if [ -n "$BUILT_EXTENSIONS" ]; then
    echo "Built extensions:"
    for ext in $BUILT_EXTENSIONS; do
        echo -e "  ${GREEN}✓${NC} $(basename $ext)"
    done
else
    echo -e "${YELLOW}! No .so files found in project root${NC}"
fi

# Test imports
echo ""
echo "Testing Python imports..."

# Test core modules
TEST_IMPORTS=(
    "numpy"
    "cv2"
)

for module in "${TEST_IMPORTS[@]}"; do
    if python3 -c "import $module" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $module"
    else
        echo -e "  ${YELLOW}!${NC} $module (optional)"
    fi
done

# Test CUDA module if built
if [ "$BUILD_CUDA" = true ]; then
    echo ""
    if python3 -c "import voxel_cuda; voxel_cuda.print_device_info()" 2>/dev/null; then
        echo -e "${GREEN}✓ CUDA module loaded successfully${NC}"
        python3 -c "import voxel_cuda; voxel_cuda.print_device_info()"
    else
        echo -e "${YELLOW}! CUDA module failed to load${NC}"
        echo "  This may be normal if CUDA extensions were not built"
    fi
fi

echo ""

# ============================================================================
# Install Package
# ============================================================================
if [ "$INSTALL" = true ] || [ "$DEV_INSTALL" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Installing Package${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    if [ "$DEV_INSTALL" = true ]; then
        echo "Installing in development mode..."
        python3 -m pip install -e .
        if [ "$BUILD_CUDA" = true ]; then
            python3 -m pip install -e .[cuda,dev,full]
        else
            python3 -m pip install -e .[dev]
        fi
    else
        echo "Installing package..."
        python3 -m pip install .
        if [ "$BUILD_CUDA" = true ]; then
            python3 -m pip install .[cuda,full]
        fi
    fi

    echo -e "${GREEN}✓ Package installed${NC}"
    echo ""
fi

# ============================================================================
# Run Tests
# ============================================================================
if [ "$RUN_TESTS" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Running Tests${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    if [ -f "$SCRIPT_DIR/run_tests.sh" ]; then
        "$SCRIPT_DIR/run_tests.sh" --quick
    else
        echo "Running basic tests..."
        python3 -m pytest tests/ -v --ignore=tests/benchmarks/ --ignore=tests/integration/ -x
    fi

    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo "Next steps:"
if [ "$DEV_INSTALL" = false ] && [ "$INSTALL" = false ]; then
    echo "  1. Install package: pip install -e .[dev,cuda,full]"
else
    echo "  1. Package is installed and ready to use"
fi

echo "  2. Run tests: ./scripts/run_tests.sh"
echo "  3. Run benchmarks: ./scripts/run_tests.sh --benchmark"

if [ "$BUILD_CUDA" = true ]; then
    echo "  4. Test CUDA: python3 cuda/example_cuda_usage.py"
fi

echo ""
echo "Build artifacts:"
echo "  Extensions: $(find . -maxdepth 1 -name "*.so" | wc -l) .so files"
echo "  Build time: ${BUILD_DURATION}s"
echo ""

exit 0
