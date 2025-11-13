# Makefile for 8K Video Processing Pipeline

.PHONY: all build install clean test benchmark help

# Default target
all: build

# Build C++ motion extractor module
build:
	@echo "Building C++ motion extractor module..."
	python setup_motion_extractor.py build_ext --inplace
	@echo "Build complete!"

# Install all dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install pybind11
	@echo "Dependencies installed!"

# Build and install
install-all: install build
	@echo "Installation complete!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -f *.so
	rm -f *.pyd
	rm -f motion_extractor_cpp*.so
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

# Run tests
test:
	@echo "Running tests..."
	pytest test_*.py -v
	@echo "Tests complete!"

# Run benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	python benchmark_8k.py --duration 10
	@echo "Benchmark complete!"

# Run example
example:
	@echo "Running example 4 (C++ motion extractor)..."
	python example_8k_pipeline.py --example 4

# Check requirements
check:
	@echo "Checking system requirements..."
	@python -c "import cv2; print('OpenCV version:', cv2.__version__)"
	@python -c "import numpy; print('NumPy version:', numpy.__version__)"
	@python -c "import pybind11; print('pybind11 version:', pybind11.__version__)"
	@python -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())" || echo "CUDA not available"
	@echo "System check complete!"

# Help
help:
	@echo "8K Video Processing Pipeline - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make all         - Build C++ modules (default)"
	@echo "  make build       - Build C++ motion extractor"
	@echo "  make install     - Install Python dependencies"
	@echo "  make install-all - Install dependencies and build"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make test        - Run test suite"
	@echo "  make benchmark   - Run performance benchmarks"
	@echo "  make example     - Run example code"
	@echo "  make check       - Check system requirements"
	@echo "  make help        - Show this help message"
