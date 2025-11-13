# CI/CD Pipeline Documentation

## Overview

This document describes the comprehensive CI/CD pipeline for the PixelToVoxel project, including automated testing, building, deployment, and Docker image management.

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [GitHub Actions Workflows](#github-actions-workflows)
- [Docker Build System](#docker-build-system)
- [Testing Framework](#testing-framework)
- [Build Scripts](#build-scripts)
- [Performance Benchmarking](#performance-benchmarking)
- [Code Coverage](#code-coverage)
- [Deployment](#deployment)
- [Self-Hosted GPU Runners](#self-hosted-gpu-runners)
- [Troubleshooting](#troubleshooting)

## Pipeline Architecture

The CI/CD pipeline consists of several integrated components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Code Push / PR                          │
└───────────────┬─────────────────────────────────────────────┘
                │
                ├──► Code Quality (lint, format)
                │
                ├──► Unit Tests (CPU)
                │    └─ Python 3.8, 3.9, 3.10, 3.11
                │
                ├──► Unit Tests (GPU)
                │    └─ CUDA 12.0 with GPU runners
                │
                ├──► Integration Tests
                │    └─ Full pipeline validation
                │
                ├──► Performance Benchmarks
                │    ├─ Regression detection
                │    └─ Performance comparison
                │
                ├──► Build Verification
                │    └─ Package building
                │
                ├──► Security Scanning
                │    ├─ Trivy vulnerability scan
                │    └─ Bandit security linter
                │
                └──► Docker Build & Publish
                     ├─ CPU image
                     ├─ GPU image (CUDA 12.0, 11.8)
                     ├─ Development image
                     └─ Production image
```

## GitHub Actions Workflows

### 1. Main CI Workflow (`.github/workflows/ci.yml`)

**Trigger Events:**
- Push to `main`, `develop`, or `claude/**` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs:**

#### Code Quality Checks (`lint`)
- Black code formatting
- isort import sorting
- Flake8 linting
- Pylint analysis
- Runs on: `ubuntu-latest`

#### CPU Unit Tests (`test-cpu`)
- Matrix testing across Python 3.8, 3.9, 3.10, 3.11
- Parallel test execution with pytest-xdist
- Code coverage reporting
- Runs on: `ubuntu-latest`

#### GPU Unit Tests (`test-gpu`)
- CUDA 12.0 with CuDNN 8
- GPU-accelerated tests
- CUDA extension building
- Runs on: `self-hosted, linux, gpu`

#### Integration Tests (`integration-tests`)
- Full pipeline validation
- Multi-camera simulations
- Network synchronization tests
- Runs on: `self-hosted, linux, gpu`

#### Performance Benchmarks (`benchmarks`)
- Comprehensive performance testing
- Regression detection
- PR comments with results
- Baseline comparison
- Runs on: `self-hosted, linux, gpu`

#### Build Verification (`build`)
- C++ extension building
- Python package creation
- Artifact uploading
- Runs on: `ubuntu-latest`

#### Security Scanning (`security`)
- Trivy vulnerability scanning
- Bandit security linting
- SARIF report upload
- Runs on: `ubuntu-latest`

### 2. Docker Workflow (`.github/workflows/docker.yml`)

**Trigger Events:**
- Push to `main` or `develop`
- Version tags (`v*.*.*`)
- Manual workflow dispatch

**Jobs:**

#### CPU Image Build (`build-cpu`)
- Multi-stage Dockerfile
- CPU-optimized image
- Tag: `{version}-cpu`

#### GPU Image Build (`build-gpu`)
- Matrix: CUDA 12.0 and 11.8
- GPU-enabled with CUDA runtime
- Tag: `{version}-gpu-cuda{version}`

#### Development Image (`build-dev`)
- Full development toolchain
- Debug symbols included
- All development dependencies
- Tag: `{version}-dev`

#### Image Testing (`test-images`)
- Basic functionality tests
- Import verification
- Runs after all builds

#### Security Scanning (`scan-images`)
- Trivy container scanning
- Vulnerability reporting
- SARIF upload to GitHub Security

#### Release Publishing (`publish-release`)
- Triggered on version tags
- Multi-registry push (GitHub + Docker Hub)
- Automated release notes

## Docker Build System

### Image Targets

The Dockerfile supports multiple build targets:

#### 1. `cpu-runtime` - CPU-Only Production
```bash
docker build --target cpu-runtime -t pixeltovoxel:cpu .
```

**Use Cases:**
- Testing without GPU
- CPU-only deployments
- CI/CD runners without GPU

**Size:** ~2.5 GB

#### 2. `gpu-runtime` - GPU Production
```bash
docker build --target gpu-runtime -t pixeltovoxel:gpu \
  --build-arg CUDA_VERSION=12.0.0 .
```

**Use Cases:**
- Production GPU deployments
- High-performance processing
- CUDA-accelerated workloads

**Size:** ~8 GB (includes CUDA runtime)

#### 3. `development` - Full Development Environment
```bash
docker build --target development -t pixeltovoxel:dev .
```

**Use Cases:**
- Interactive development
- Debugging
- Testing new features

**Size:** ~12 GB (includes all tools)

**Includes:**
- All development dependencies
- Debugging tools (gdb, valgrind)
- Code quality tools
- Jupyter notebooks
- Full documentation

#### 4. `testing` - CI/CD Testing
```bash
docker build --target testing -t pixeltovoxel:test .
```

**Use Cases:**
- CI/CD pipelines
- Automated testing
- Pre-deployment validation

#### 5. `production` - Minimal Production
```bash
docker build --target production -t pixeltovoxel:prod .
```

**Use Cases:**
- Production deployments
- Minimal attack surface
- Optimized performance

**Size:** ~5 GB

### Running Containers

#### CPU Container
```bash
docker run --rm -it pixeltovoxel:cpu python3 -m src.example_8k_pipeline
```

#### GPU Container
```bash
docker run --rm -it --gpus all pixeltovoxel:gpu \
  python3 -m src.example_8k_pipeline --use-gpu
```

#### Development Container
```bash
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  pixeltovoxel:dev
```

## Testing Framework

### Test Execution Script (`scripts/run_tests.sh`)

Comprehensive test runner with multiple modes:

```bash
# Run all tests
./scripts/run_tests.sh --all

# Unit tests only
./scripts/run_tests.sh --unit

# Integration tests
./scripts/run_tests.sh --integration

# Performance benchmarks
./scripts/run_tests.sh --benchmark

# Quick test suite
./scripts/run_tests.sh --quick

# With coverage
./scripts/run_tests.sh --coverage --html

# GPU tests only
./scripts/run_tests.sh --gpu

# CPU tests only
./scripts/run_tests.sh --cpu-only

# Performance regression check
./scripts/run_tests.sh --benchmark --regression
```

### Test Organization

```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_voxel_grid.py
│   ├── test_motion_detection.py
│   └── test_cuda_kernels.py
├── integration/             # Integration tests (slower, multi-component)
│   ├── test_full_pipeline.py
│   └── test_camera_sync.py
├── benchmarks/              # Performance benchmarks
│   ├── run_all_benchmarks.py
│   ├── benchmark_suite.py
│   ├── camera_benchmark.py
│   ├── network_benchmark.py
│   └── compare_benchmarks.py
└── test_data/               # Test data files
```

## Build Scripts

### Build Script (`scripts/build.sh`)

Comprehensive build automation:

```bash
# Clean build
./scripts/build.sh --clean

# Build with CUDA
./scripts/build.sh --cuda

# Release build (optimized)
./scripts/build.sh --release

# Debug build
./scripts/build.sh --debug

# Install after building
./scripts/build.sh --install

# Development install
./scripts/build.sh --dev

# Install dependencies first
./scripts/build.sh --deps --dev

# Full workflow
./scripts/build.sh --clean --deps --dev --cuda --test
```

### Build Workflow

1. **Environment Check**
   - Python version
   - Compiler availability
   - CUDA detection
   - GPU detection

2. **Dependency Installation**
   - System packages
   - Python packages
   - CUDA packages (if enabled)

3. **Protocol Buffer Compilation**
   - Compile .proto files
   - Generate Python bindings

4. **Extension Building**
   - C++ extensions
   - CUDA kernels
   - Python bindings

5. **Verification**
   - Import tests
   - Module loading
   - GPU functionality

## Performance Benchmarking

### Benchmark Suite

The benchmark suite measures performance across multiple dimensions:

1. **Voxel Operations**
   - Ray casting
   - Grid updates
   - Spatial queries

2. **Motion Detection**
   - 8K frame processing
   - Optical flow
   - Feature tracking

3. **Camera Synchronization**
   - Multi-camera timing
   - Frame alignment
   - Latency measurement

4. **Network Performance**
   - Data throughput
   - Message latency
   - Compression efficiency

### Regression Detection

The `compare_benchmarks.py` tool detects performance regressions:

```bash
python3 tests/benchmarks/compare_benchmarks.py \
  --baseline tests/benchmarks/benchmark_results/baseline.json \
  --current tests/benchmarks/benchmark_results/latest.json \
  --threshold 10.0 \
  --fail-on-regression
```

**Features:**
- Configurable regression threshold (default: 10%)
- Detailed comparison reports
- Visual indicators for regressions/improvements
- JSON export for tracking

**CI Integration:**
- Automatic comparison on push/PR
- PR comments with results
- Baseline caching
- Trend tracking

## Code Coverage

### Coverage Requirements

- **Minimum Coverage:** 80%
- **Enforcement:** CI fails below threshold
- **Reports:** XML + HTML + Terminal

### Generating Coverage Reports

```bash
# With test script
./scripts/run_tests.sh --coverage --html

# Direct pytest
pytest tests/ --cov=src --cov-report=html --cov-report=xml --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

### Coverage CI Integration

- Automatic upload to Codecov
- Coverage badges
- PR comments with coverage diff
- Per-suite coverage tracking (CPU, GPU, integration)

## Deployment

### Automated Deployment

On version tag push (`v*.*.*`):

1. Build all image variants
2. Run security scans
3. Test images
4. Push to registries:
   - GitHub Container Registry
   - Docker Hub (if configured)
5. Create release notes
6. Generate deployment manifests

### Manual Deployment

```bash
# Tag version
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Or trigger workflow manually
gh workflow run docker.yml
```

## Self-Hosted GPU Runners

### Requirements

For GPU testing and benchmarking, set up self-hosted runners with:

**Hardware:**
- NVIDIA GPU (Compute Capability ≥ 7.0)
- 16GB+ RAM
- 100GB+ disk space

**Software:**
- Ubuntu 22.04
- NVIDIA drivers (≥ 525)
- CUDA 12.0
- Docker with nvidia-container-toolkit

### Runner Setup

```bash
# Install NVIDIA drivers
sudo apt-get install nvidia-driver-525

# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda

# Install Docker with GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Install GitHub Actions runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf actions-runner-linux-x64-2.311.0.tar.gz
./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO --token YOUR_TOKEN --labels self-hosted,linux,gpu
./run.sh
```

### Runner Labels

- `self-hosted`: Self-hosted runner
- `linux`: Linux OS
- `gpu`: GPU available

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found

**Symptom:** Build fails with "CUDA not found"

**Solution:**
```bash
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 2. Tests Timeout

**Symptom:** Tests hang or timeout

**Solution:**
```bash
# Increase timeout
./scripts/run_tests.sh --timeout 600

# Or in CI, adjust .github/workflows/ci.yml
```

#### 3. Coverage Below Threshold

**Symptom:** CI fails due to low coverage

**Solution:**
- Add more tests
- Or adjust threshold in `.github/workflows/ci.yml`:
  ```yaml
  env:
    MIN_COVERAGE: '75'  # Adjust as needed
  ```

#### 4. Docker Build Fails

**Symptom:** Out of memory or disk space

**Solution:**
```bash
# Clean up Docker
docker system prune -a -f

# Increase Docker memory in Docker Desktop settings
# Or use multi-stage build with --target
docker build --target cpu-runtime -t pixeltovoxel:cpu .
```

#### 5. Benchmark Regression False Positives

**Symptom:** Spurious regression warnings

**Solution:**
```bash
# Increase threshold
python3 tests/benchmarks/compare_benchmarks.py \
  --threshold 15.0 \
  --baseline baseline.json \
  --current latest.json
```

## Best Practices

1. **Always run tests locally before pushing:**
   ```bash
   ./scripts/run_tests.sh --quick
   ```

2. **Check code quality:**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

3. **Update baseline benchmarks after intentional changes:**
   ```bash
   cp tests/benchmarks/benchmark_results/latest.json \
      tests/benchmarks/benchmark_results/baseline.json
   ```

4. **Test Docker builds locally:**
   ```bash
   docker build --target cpu-runtime -t test:cpu .
   docker run --rm test:cpu python3 -c "import src; print('OK')"
   ```

5. **Monitor CI resource usage:**
   - Check runner performance
   - Monitor GPU utilization
   - Track build times

## Configuration Files

- `.github/workflows/ci.yml` - Main CI workflow
- `.github/workflows/docker.yml` - Docker build workflow
- `scripts/run_tests.sh` - Test execution script
- `scripts/build.sh` - Build script
- `Dockerfile` - Multi-stage Docker build
- `.dockerignore` - Docker build exclusions
- `tests/benchmarks/compare_benchmarks.py` - Regression detection

## Metrics and Monitoring

### CI Metrics

Track these metrics over time:
- Build success rate
- Test pass rate
- Code coverage percentage
- Build duration
- Test duration
- Docker image sizes

### Performance Metrics

Monitor benchmark results:
- Voxel throughput (FPS)
- Motion detection latency
- Network bandwidth
- GPU utilization
- Memory usage

## Security

### Automated Scanning

- **Trivy:** Container vulnerability scanning
- **Bandit:** Python security linting
- **Dependency scanning:** GitHub Dependabot

### Best Practices

1. Keep dependencies updated
2. Use minimal base images
3. Run containers as non-root
4. Scan images before deployment
5. Use secrets for sensitive data

## Support

For issues or questions:
- GitHub Issues: Report bugs or feature requests
- CI/CD Issues: Check runner logs and workflow runs
- Docker Issues: Check Docker build logs

---

**Last Updated:** 2025-11-13
**Version:** 1.0.0
