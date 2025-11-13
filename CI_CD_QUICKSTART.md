# CI/CD Pipeline - Quick Start Guide

## Overview

This project includes a comprehensive CI/CD pipeline for automated testing, building, and deployment of the PixelToVoxel 8K motion tracking system.

## Quick Setup

### 1. Enable GitHub Actions

The workflows are already configured in `.github/workflows/`. They will automatically run on:
- Push to `main`, `develop`, or `claude/**` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

### 2. Configure Self-Hosted GPU Runners (Optional)

For GPU tests and benchmarks, you'll need self-hosted runners:

```bash
# On your GPU machine
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf actions-runner-linux-x64-2.311.0.tar.gz

# Configure runner (get token from GitHub repo settings)
./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO \
  --token YOUR_TOKEN \
  --labels self-hosted,linux,gpu

# Run as service
sudo ./svc.sh install
sudo ./svc.sh start
```

### 3. Configure Secrets (Optional)

For Docker Hub publishing, add these secrets in GitHub repo settings:
- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token

## Local Development

### Build Project

```bash
# Clean build with dependencies
./scripts/build.sh --clean --deps --dev

# With CUDA support
./scripts/build.sh --cuda

# Release build
./scripts/build.sh --release --install
```

### Run Tests

```bash
# Quick test suite
./scripts/run_tests.sh --quick

# All unit tests
./scripts/run_tests.sh --unit --coverage

# Integration tests
./scripts/run_tests.sh --integration

# Performance benchmarks
./scripts/run_tests.sh --benchmark

# With GPU tests
./scripts/run_tests.sh --all --gpu

# CPU only
./scripts/run_tests.sh --all --cpu-only
```

### Docker Build

```bash
# CPU image
docker build --target cpu-runtime -t pixeltovoxel:cpu .

# GPU image
docker build --target gpu-runtime -t pixeltovoxel:gpu \
  --build-arg CUDA_VERSION=12.0.0 .

# Development image
docker build --target development -t pixeltovoxel:dev .

# Run with GPU
docker run --rm -it --gpus all pixeltovoxel:gpu
```

## CI/CD Workflows

### Main CI Pipeline (`.github/workflows/ci.yml`)

**Jobs:**
1. **lint** - Code quality checks (Black, Flake8, Pylint)
2. **test-cpu** - Unit tests on multiple Python versions
3. **test-gpu** - GPU-accelerated tests with CUDA
4. **integration-tests** - Full pipeline validation
5. **benchmarks** - Performance testing with regression detection
6. **build** - Build verification and package creation
7. **security** - Vulnerability scanning (Trivy, Bandit)

**Coverage Threshold:** 80% (configurable in workflow)

### Docker Build Pipeline (`.github/workflows/docker.yml`)

**Jobs:**
1. **build-cpu** - CPU-only production image
2. **build-gpu** - GPU-enabled images (CUDA 12.0, 11.8)
3. **build-dev** - Development environment
4. **test-images** - Container functionality tests
5. **scan-images** - Security vulnerability scanning
6. **publish-release** - Automated release publishing

**Triggered On:**
- Push to main/develop
- Version tags (`v*.*.*`)
- Manual dispatch

## Common Tasks

### Run Tests Before Committing

```bash
# Quick validation
./scripts/run_tests.sh --quick --cpu-only

# Full test suite (if you have GPU)
./scripts/run_tests.sh --all --coverage
```

### Check Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check for issues
flake8 src/ tests/
pylint src/
```

### Update Performance Baseline

After intentional performance changes:

```bash
# Run benchmarks
./scripts/run_tests.sh --benchmark

# Update baseline
cp tests/benchmarks/benchmark_results/latest.json \
   tests/benchmarks/benchmark_results/baseline.json

# Commit updated baseline
git add tests/benchmarks/benchmark_results/baseline.json
git commit -m "Update performance baseline"
```

### Create a Release

```bash
# Tag version
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# CI automatically builds and publishes Docker images
```

## Monitoring CI/CD

### Check Workflow Status

```bash
# Using GitHub CLI
gh run list
gh run view <run-id>
gh run watch

# Or visit GitHub Actions tab in repository
```

### View Test Results

Test results and artifacts are available in the GitHub Actions UI:
- Code coverage reports
- Benchmark results
- Build artifacts
- Security scan results

### Docker Images

Published images are available at:
- GitHub Container Registry: `ghcr.io/YOUR_ORG/YOUR_REPO`
- Docker Hub: `YOUR_ORG/pixeltovoxelprojector` (if configured)

Pull images:
```bash
# Latest CPU
docker pull ghcr.io/YOUR_ORG/YOUR_REPO:latest-cpu

# Latest GPU
docker pull ghcr.io/YOUR_ORG/YOUR_REPO:latest-gpu

# Specific version
docker pull ghcr.io/YOUR_ORG/YOUR_REPO:v1.0.0-gpu-cuda12.0
```

## Troubleshooting

### Tests Failing Locally

```bash
# Clean environment
./scripts/build.sh --clean

# Reinstall dependencies
pip install -r requirements.txt
pip install -e .[dev,cuda,full]

# Rebuild extensions
./scripts/build.sh --cuda

# Run tests with verbose output
./scripts/run_tests.sh --unit --verbose
```

### Docker Build Issues

```bash
# Clean Docker cache
docker system prune -a -f

# Build with no cache
docker build --no-cache --target cpu-runtime -t test .

# Check build logs
docker build --progress=plain --target cpu-runtime -t test .
```

### CI Failing on GPU Tests

If you don't have GPU runners:
1. Comment out or skip GPU tests in `.github/workflows/ci.yml`
2. Or set up self-hosted GPU runners (see setup guide above)
3. Or use GitHub-hosted runners with GPU support (if available)

## Performance Requirements

### Minimum Hardware for GPU Tests

- **GPU:** NVIDIA GPU with Compute Capability ≥ 7.0
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 100GB+ for Docker images and test data
- **CUDA:** Version 11.8 or 12.0

### Expected CI Times

- **Lint:** ~2 minutes
- **CPU Tests:** ~5-10 minutes per Python version
- **GPU Tests:** ~10-15 minutes
- **Integration Tests:** ~15-20 minutes
- **Benchmarks:** ~20-30 minutes
- **Docker Builds:** ~10-15 minutes per image

## Advanced Configuration

### Customize Coverage Threshold

Edit `.github/workflows/ci.yml`:
```yaml
env:
  MIN_COVERAGE: '80'  # Change to desired percentage
```

### Adjust Benchmark Regression Threshold

Edit test execution or workflow:
```bash
python3 tests/benchmarks/compare_benchmarks.py \
  --threshold 15.0  # Change from default 10.0
```

### Add More Python Versions

Edit `.github/workflows/ci.yml`:
```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

## Resources

- **Full Documentation:** See `docs/CI_CD_GUIDE.md`
- **Test Scripts:** See `scripts/run_tests.sh`
- **Build Scripts:** See `scripts/build.sh`
- **Dockerfile:** See `Dockerfile`
- **Workflows:** See `.github/workflows/`

## Support

For issues:
1. Check workflow logs in GitHub Actions
2. Review `docs/CI_CD_GUIDE.md` for detailed troubleshooting
3. Check runner logs for self-hosted runners
4. Open an issue on GitHub

---

**Quick Commands Reference:**

```bash
# Development
./scripts/build.sh --clean --dev --cuda
./scripts/run_tests.sh --quick

# Before commit
black src/ tests/
./scripts/run_tests.sh --unit --coverage

# Docker
docker build --target development -t pixeltovoxel:dev .
docker run --rm -it --gpus all -v $(pwd):/workspace pixeltovoxel:dev

# Release
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0
```

---

**Last Updated:** 2025-11-13
