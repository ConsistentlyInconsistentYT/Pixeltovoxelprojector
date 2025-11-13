# ============================================================================
# Pixel to Voxel Projector - Makefile
# Convenient build commands for common operations
# ============================================================================

.PHONY: help install install-dev install-full build clean test benchmark docker docker-run lint format check docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Python executable
PYTHON := python3
PIP := $(PYTHON) -m pip

# Docker configuration
DOCKER_IMAGE := pixeltovoxel:latest
DOCKER_FILE := docker/Dockerfile

# ============================================================================
# Help
# ============================================================================

help: ## Show this help message
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)  Pixel to Voxel Projector - Build System$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# ============================================================================
# Installation
# ============================================================================

install: ## Install package in production mode
	@echo "$(GREEN)Installing Pixel to Voxel Projector...$(NC)"
	$(PIP) install .
	@echo "$(GREEN)✓ Installation complete!$(NC)"

install-dev: ## Install package in development mode with dev dependencies
	@echo "$(GREEN)Installing in development mode...$(NC)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Development installation complete!$(NC)"

install-full: ## Install with all optional dependencies
	@echo "$(GREEN)Installing with all dependencies...$(NC)"
	$(PIP) install -e ".[full,dev,cuda]"
	@echo "$(GREEN)✓ Full installation complete!$(NC)"

install-cuda: ## Install CUDA dependencies (auto-detect CUDA version)
	@echo "$(GREEN)Installing CUDA dependencies...$(NC)"
	@if nvcc --version | grep -q "11"; then \
		echo "$(YELLOW)Detected CUDA 11.x$(NC)"; \
		$(PIP) install cupy-cuda11x; \
	elif nvcc --version | grep -q "12"; then \
		echo "$(YELLOW)Detected CUDA 12.x$(NC)"; \
		$(PIP) install cupy-cuda12x; \
	else \
		echo "$(RED)CUDA not detected or unsupported version$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ CUDA dependencies installed!$(NC)"

requirements: ## Install Python dependencies from requirements.txt
	@echo "$(GREEN)Installing requirements...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Requirements installed!$(NC)"

# ============================================================================
# Build
# ============================================================================

build: ## Build C++ and CUDA extensions
	@echo "$(GREEN)Building extensions...$(NC)"
	$(PYTHON) setup.py build_ext --inplace
	@echo "$(GREEN)✓ Build complete!$(NC)"

build-cmake: ## Build using CMake
	@echo "$(GREEN)Building with CMake...$(NC)"
	mkdir -p build
	cd build && cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
	cd build && ninja
	@echo "$(GREEN)✓ CMake build complete!$(NC)"

protobuf: ## Compile protocol buffers
	@echo "$(GREEN)Compiling protocol buffers...$(NC)"
	cd src/protocols && \
	for proto in *.proto; do \
		protoc --python_out=. --grpc_python_out=. -I. "$$proto"; \
	done
	@echo "$(GREEN)✓ Protocol buffers compiled!$(NC)"

# ============================================================================
# Cleaning
# ============================================================================

clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.so' -delete
	find . -type f -name '*.o' -delete
	@echo "$(GREEN)✓ Cleaned!$(NC)"

clean-all: clean ## Deep clean including cache and temporary files
	@echo "$(YELLOW)Deep cleaning...$(NC)"
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf logs/
	rm -rf tmp/
	@echo "$(GREEN)✓ Deep clean complete!$(NC)"

# ============================================================================
# Testing
# ============================================================================

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v

test-fast: ## Run tests without slow markers
	@echo "$(GREEN)Running fast tests...$(NC)"
	pytest tests/ -v -m "not slow"

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(NC)"

test-installation: ## Test installation and GPU availability
	@echo "$(GREEN)Testing installation...$(NC)"
	$(PYTHON) tests/benchmarks/test_installation.py

# ============================================================================
# Benchmarking
# ============================================================================

benchmark: ## Run all benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(PYTHON) tests/benchmarks/run_all_benchmarks.py

benchmark-quick: ## Run quick benchmark
	@echo "$(GREEN)Running quick benchmark...$(NC)"
	$(PYTHON) tests/benchmarks/quick_benchmark.py

benchmark-gpu: ## Run GPU-specific benchmarks
	@echo "$(GREEN)Running GPU benchmarks...$(NC)"
	$(PYTHON) tests/benchmarks/voxel_benchmark.py

benchmark-network: ## Run network benchmarks
	@echo "$(GREEN)Running network benchmarks...$(NC)"
	$(PYTHON) tests/benchmarks/network_benchmark.py

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linting checks
	@echo "$(GREEN)Running linters...$(NC)"
	flake8 src/ tests/ --max-line-length=100 || true
	pylint src/ tests/ --max-line-length=100 || true

format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	black src/ tests/ examples/ --line-length=100
	isort src/ tests/ examples/
	@echo "$(GREEN)✓ Code formatted!$(NC)"

format-check: ## Check code formatting without modifying files
	@echo "$(GREEN)Checking code format...$(NC)"
	black src/ tests/ examples/ --check --line-length=100
	isort src/ tests/ examples/ --check-only

typecheck: ## Run type checking with mypy
	@echo "$(GREEN)Running type checker...$(NC)"
	mypy src/ --ignore-missing-imports

check: format-check lint typecheck ## Run all code quality checks
	@echo "$(GREEN)✓ All checks passed!$(NC)"

# ============================================================================
# Docker
# ============================================================================

docker: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) -f $(DOCKER_FILE) .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE)$(NC)"

docker-run: ## Run Docker container with GPU support
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run --gpus all -it --rm \
		-v $(PWD):/app \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=$(DISPLAY) \
		$(DOCKER_IMAGE)

docker-jupyter: ## Run Jupyter Lab in Docker
	@echo "$(GREEN)Starting Jupyter Lab...$(NC)"
	docker run --gpus all -it --rm \
		-p 8888:8888 \
		-v $(PWD):/app \
		$(DOCKER_IMAGE) \
		jupyter lab --ip=0.0.0.0 --allow-root --no-browser

docker-compose-up: ## Start all services with docker-compose
	@echo "$(GREEN)Starting services with docker-compose...$(NC)"
	docker-compose -f docker/docker-compose.yml up -d

docker-compose-down: ## Stop all docker-compose services
	@echo "$(GREEN)Stopping services...$(NC)"
	docker-compose -f docker/docker-compose.yml down

# ============================================================================
# Documentation
# ============================================================================

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✓ Documentation built: docs/_build/html/index.html$(NC)"

docs-serve: docs ## Build and serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"
	$(PYTHON) -m http.server -d docs/_build/html 8000

# ============================================================================
# Utilities
# ============================================================================

info: ## Display system information
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)  System Information$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(YELLOW)Python:$(NC)"
	@$(PYTHON) --version
	@echo ""
	@echo "$(YELLOW)CUDA:$(NC)"
	@nvcc --version 2>/dev/null || echo "CUDA not found"
	@echo ""
	@echo "$(YELLOW)GPU:$(NC)"
	@nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "GPU not found"
	@echo ""
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

check-deps: ## Check if all dependencies are installed
	@echo "$(GREEN)Checking dependencies...$(NC)"
	@echo ""
	@echo "$(YELLOW)System packages:$(NC)"
	@dpkg -l | grep -E "(cmake|ninja-build|ffmpeg|libopencv)" || echo "Some system packages missing"
	@echo ""
	@echo "$(YELLOW)Python packages:$(NC)"
	@$(PIP) list | grep -E "(numpy|opencv|cupy|pybind11)" || echo "Some Python packages missing"

upgrade-deps: ## Upgrade Python dependencies
	@echo "$(GREEN)Upgrading dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)✓ Dependencies upgraded!$(NC)"

# ============================================================================
# Examples
# ============================================================================

example-8k: ## Run 8K processing example
	@echo "$(GREEN)Running 8K processing example...$(NC)"
	$(PYTHON) src/example_8k_pipeline.py

example-camera: ## Run camera tracking example
	@echo "$(GREEN)Running camera tracking example...$(NC)"
	$(PYTHON) examples/camera_tracking_example.py

example-distributed: ## Run distributed processing example
	@echo "$(GREEN)Running distributed processing example...$(NC)"
	$(PYTHON) examples/distributed_processing_example.py

# ============================================================================
# Quick Commands
# ============================================================================

dev: install-dev ## Quick setup for development

all: clean install-full test ## Complete build and test

quick: install test-fast ## Quick install and test

# ============================================================================
# Version Information
# ============================================================================

version: ## Show version information
	@echo "$(GREEN)Pixel to Voxel Projector v1.0.0$(NC)"
	@$(PYTHON) -c "import sys; print(f'Python: {sys.version}')"
	@nvcc --version 2>/dev/null | grep "release" || echo "CUDA: Not installed"

# ============================================================================
# End of Makefile
# ============================================================================
