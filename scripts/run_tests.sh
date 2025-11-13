#!/bin/bash
#
# Comprehensive Test Execution Script
#
# Usage:
#   ./run_tests.sh [options]
#
# Options:
#   --unit              Run unit tests only
#   --integration       Run integration tests only
#   --benchmark         Run performance benchmarks
#   --all               Run all tests (default)
#   --coverage          Generate coverage report
#   --html              Generate HTML coverage report
#   --gpu               Include GPU tests
#   --cpu-only          Skip GPU tests
#   --quick             Run quick test suite only
#   --verbose           Verbose output
#   --parallel N        Run tests with N parallel workers (default: auto)
#   --timeout SECONDS   Test timeout in seconds (default: 300)
#   --marker MARKER     Run tests with specific pytest marker
#   --regression        Check for performance regressions
#   --help              Show this help message

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$PROJECT_DIR/tests"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_BENCHMARK=false
RUN_ALL=true
GENERATE_COVERAGE=true
GENERATE_HTML=false
INCLUDE_GPU=true
VERBOSE=false
PARALLEL="auto"
TIMEOUT=300
QUICK_MODE=false
PYTEST_MARKER=""
CHECK_REGRESSION=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT=true
            RUN_ALL=false
            shift
            ;;
        --integration)
            RUN_INTEGRATION=true
            RUN_ALL=false
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            RUN_ALL=false
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --coverage)
            GENERATE_COVERAGE=true
            shift
            ;;
        --html)
            GENERATE_HTML=true
            shift
            ;;
        --gpu)
            INCLUDE_GPU=true
            shift
            ;;
        --cpu-only)
            INCLUDE_GPU=false
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --marker|-m)
            PYTEST_MARKER="$2"
            shift 2
            ;;
        --regression)
            CHECK_REGRESSION=true
            shift
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

# If no specific test type selected, run all
if [ "$RUN_ALL" = true ]; then
    RUN_UNIT=true
    RUN_INTEGRATION=true
    RUN_BENCHMARK=false  # Benchmarks are opt-in
fi

# Banner
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PixelToVoxel Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check Python
echo -e "${YELLOW}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"

# Check pytest
if ! python3 -c "import pytest" 2>/dev/null; then
    echo -e "${RED}Error: pytest not installed${NC}"
    echo "Install with: pip install pytest pytest-cov pytest-xdist"
    exit 1
fi

# Check for GPU
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        echo -e "${GREEN}✓ GPU available${NC}"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | nl -w2 -s'. '
    fi
else
    echo -e "${YELLOW}! GPU not available or CUDA not installed${NC}"
fi

if [ "$INCLUDE_GPU" = true ] && [ "$GPU_AVAILABLE" = false ]; then
    echo -e "${YELLOW}Warning: GPU tests requested but GPU not available${NC}"
    INCLUDE_GPU=false
fi

echo ""

# Build pytest command
PYTEST_CMD="python3 -m pytest"
PYTEST_ARGS="-v"

# Add coverage options
if [ "$GENERATE_COVERAGE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src --cov-report=term-missing --cov-report=xml"
    if [ "$GENERATE_HTML" = true ]; then
        PYTEST_ARGS="$PYTEST_ARGS --cov-report=html"
    fi
fi

# Add parallel execution
if [ "$PARALLEL" != "1" ]; then
    PYTEST_ARGS="$PYTEST_ARGS -n $PARALLEL"
fi

# Add timeout
PYTEST_ARGS="$PYTEST_ARGS --timeout=$TIMEOUT"

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -vv -s"
fi

# Add marker filter
if [ -n "$PYTEST_MARKER" ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m $PYTEST_MARKER"
fi

# GPU marker handling
if [ "$INCLUDE_GPU" = false ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m 'not (gpu or cuda)'"
fi

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_DURATION=0

# Function to run tests
run_test_suite() {
    local suite_name=$1
    local test_path=$2
    local extra_args=$3

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Running $suite_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    START_TIME=$(date +%s)

    if $PYTEST_CMD $test_path $PYTEST_ARGS $extra_args; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        TOTAL_DURATION=$((TOTAL_DURATION + DURATION))
        echo ""
        echo -e "${GREEN}✓ $suite_name PASSED${NC} (${DURATION}s)"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        TOTAL_DURATION=$((TOTAL_DURATION + DURATION))
        echo ""
        echo -e "${RED}✗ $suite_name FAILED${NC} (${DURATION}s)"
        return 1
    fi
    echo ""
}

# ============================================================================
# Run Unit Tests
# ============================================================================
if [ "$RUN_UNIT" = true ]; then
    if [ "$QUICK_MODE" = true ]; then
        run_test_suite "Unit Tests (Quick)" \
            "$TESTS_DIR" \
            "--ignore=$TESTS_DIR/integration/ --ignore=$TESTS_DIR/benchmarks/ -m 'not slow'" || true
    else
        run_test_suite "Unit Tests" \
            "$TESTS_DIR" \
            "--ignore=$TESTS_DIR/integration/ --ignore=$TESTS_DIR/benchmarks/" || true
    fi
fi

# ============================================================================
# Run Integration Tests
# ============================================================================
if [ "$RUN_INTEGRATION" = true ]; then
    # Check if integration test data exists
    if [ ! -d "$TESTS_DIR/test_data" ]; then
        echo -e "${YELLOW}Warning: Test data directory not found${NC}"
        echo "Creating test data directory..."
        mkdir -p "$TESTS_DIR/test_data"
    fi

    run_test_suite "Integration Tests" \
        "$TESTS_DIR/integration/" \
        "" || true
fi

# ============================================================================
# Run Performance Benchmarks
# ============================================================================
if [ "$RUN_BENCHMARK" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Running Performance Benchmarks${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Create benchmark results directory
    BENCHMARK_DIR="$TESTS_DIR/benchmarks/benchmark_results"
    mkdir -p "$BENCHMARK_DIR"

    # Run comprehensive benchmark suite
    if [ -f "$TESTS_DIR/benchmarks/run_all_benchmarks.py" ]; then
        echo "Running comprehensive benchmark suite..."
        if python3 "$TESTS_DIR/benchmarks/run_all_benchmarks.py"; then
            echo -e "${GREEN}✓ Benchmarks completed${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}✗ Benchmarks failed${NC}"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    else
        echo -e "${YELLOW}Warning: Benchmark runner not found${NC}"
    fi

    # Performance regression check
    if [ "$CHECK_REGRESSION" = true ]; then
        echo ""
        echo -e "${YELLOW}Checking for performance regressions...${NC}"

        BASELINE_FILE="$BENCHMARK_DIR/baseline.json"
        LATEST_FILE="$BENCHMARK_DIR/latest.json"

        if [ -f "$BASELINE_FILE" ] && [ -f "$LATEST_FILE" ]; then
            # Compare benchmarks
            if python3 -m tests.benchmarks.compare_benchmarks \
                --baseline "$BASELINE_FILE" \
                --current "$LATEST_FILE" \
                --threshold 10.0 \
                --fail-on-regression; then
                echo -e "${GREEN}✓ No performance regressions detected${NC}"
            else
                echo -e "${RED}✗ Performance regressions detected${NC}"
                TESTS_FAILED=$((TESTS_FAILED + 1))
            fi
        else
            echo -e "${YELLOW}! Baseline or latest results not found, skipping regression check${NC}"
        fi
    fi
    echo ""
fi

# ============================================================================
# Coverage Report
# ============================================================================
if [ "$GENERATE_COVERAGE" = true ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Coverage Report${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    if [ -f "coverage.xml" ]; then
        # Check coverage percentage
        COVERAGE_PERCENT=$(python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('coverage.xml')
root = tree.getroot()
coverage = root.attrib.get('line-rate', '0')
print(f'{float(coverage) * 100:.2f}')
" 2>/dev/null || echo "0")

        echo -e "Total Coverage: ${GREEN}${COVERAGE_PERCENT}%${NC}"

        # Check against threshold
        MIN_COVERAGE=80.0
        if python3 -c "exit(0 if float('$COVERAGE_PERCENT') >= $MIN_COVERAGE else 1)"; then
            echo -e "${GREEN}✓ Coverage meets minimum threshold (${MIN_COVERAGE}%)${NC}"
        else
            echo -e "${RED}✗ Coverage below minimum threshold (${MIN_COVERAGE}%)${NC}"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi

        if [ "$GENERATE_HTML" = true ] && [ -d "htmlcov" ]; then
            echo ""
            echo -e "HTML coverage report generated: ${BLUE}htmlcov/index.html${NC}"
        fi
    else
        echo -e "${YELLOW}No coverage data found${NC}"
    fi
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))

echo "Total Test Suites: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo "Total Duration: ${TOTAL_DURATION}s"
echo ""

# Exit with appropriate code
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  All Tests Passed! ✓${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Some Tests Failed ✗${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
