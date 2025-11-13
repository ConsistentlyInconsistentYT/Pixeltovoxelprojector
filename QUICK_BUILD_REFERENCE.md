# Quick Build Reference Card

## Single-Command Builds

### Most Common Use Cases

```bash
# 1. Development setup (recommended for developers)
make dev

# 2. Full installation with all features
make install-full

# 3. Production installation
make install

# 4. Docker deployment
make docker && make docker-run
```

---

## Build Methods Cheat Sheet

### Method 1: Makefile (Easiest)

```bash
make help              # Show all commands
make install-dev       # Install for development
make test             # Run tests
make benchmark        # Run benchmarks
```

### Method 2: Python pip

```bash
# Basic
pip install -e .

# Development
pip install -e ".[dev]"

# Full with GPU
pip install -e ".[full,dev,cuda]"
```

### Method 3: CMake

```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
sudo ninja install
```

### Method 4: Docker

```bash
# Build
docker build -t pixeltovoxel:latest -f docker/Dockerfile .

# Run
docker run --gpus all -it --rm -v $(pwd):/app pixeltovoxel:latest

# Docker Compose
docker-compose -f docker/docker-compose.yml up -d
```

---

## Common Tasks

### Installation

| Task | Command |
|------|---------|
| Install deps only | `make requirements` |
| Dev install | `make dev` |
| Full install | `make install-full` |
| GPU support | `make install-cuda` |

### Building

| Task | Command |
|------|---------|
| Build extensions | `make build` |
| CMake build | `make build-cmake` |
| Protocol buffers | `make protobuf` |
| Clean build | `make clean` |

### Testing

| Task | Command |
|------|---------|
| All tests | `make test` |
| Fast tests | `make test-fast` |
| With coverage | `make test-coverage` |
| Benchmarks | `make benchmark` |

### Docker

| Task | Command |
|------|---------|
| Build image | `make docker` |
| Run container | `make docker-run` |
| Jupyter Lab | `make docker-jupyter` |
| Start services | `make docker-compose-up` |

### Code Quality

| Task | Command |
|------|---------|
| Format code | `make format` |
| Lint code | `make lint` |
| Type check | `make typecheck` |
| All checks | `make check` |

---

## Environment Variables

```bash
# CUDA configuration
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build configuration
export MAX_JOBS=8                    # Parallel build jobs
export TORCH_CUDA_ARCH_LIST="86;89" # GPU architectures

# Runtime configuration
export OMP_NUM_THREADS=16            # OpenMP threads
export CUDA_VISIBLE_DEVICES=0        # GPU selection
```

---

## Troubleshooting Quick Fixes

```bash
# CUDA not found
export CUDA_HOME=/usr/local/cuda
nvidia-smi  # Verify GPU

# Build fails
make clean-all
make install-full

# Import errors
export PYTHONPATH=$(pwd):$PYTHONPATH
python -c "import sys; print(sys.path)"

# GPU not available
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Dependencies missing
make check-deps
make requirements
```

---

## System Requirements Check

```bash
# Check all prerequisites
make info
make check-deps

# Individual checks
python --version        # Python 3.8+
nvcc --version         # CUDA 11.x/12.x
nvidia-smi             # GPU and driver
cmake --version        # CMake 3.18+
gcc --version          # GCC 9+
```

---

## Files Created

| File | Purpose |
|------|---------|
| `setup.py` | Enhanced build script with CUDA support |
| `CMakeLists.txt` | CMake build configuration |
| `requirements.txt` | Python dependencies |
| `Makefile` | Convenient build commands |
| `docker/Dockerfile` | CUDA-enabled container |
| `docker/docker-compose.yml` | Multi-service orchestration |
| `.dockerignore` | Docker build optimization |
| `BUILD.md` | Detailed build instructions |
| `DEPENDENCIES.md` | Dependency documentation |
| `BUILD_SYSTEM_SUMMARY.md` | Build system overview |

---

## Quick Start Workflows

### Workflow 1: First Time Setup

```bash
git clone <repository>
cd Pixeltovoxelprojector
make dev
make test-installation
```

### Workflow 2: Docker Deployment

```bash
cd Pixeltovoxelprojector
make docker
make docker-compose-up
# Access at http://localhost:8888 for Jupyter
```

### Workflow 3: Development Cycle

```bash
# Make changes to code
make format           # Format code
make test-fast        # Quick tests
git commit -m "..."   # Commit changes
```

### Workflow 4: Production Build

```bash
make clean
make install
make test
make benchmark
```

---

## Documentation Links

- **Detailed Instructions**: [BUILD.md](BUILD.md)
- **Dependencies**: [DEPENDENCIES.md](DEPENDENCIES.md)
- **Build System Overview**: [BUILD_SYSTEM_SUMMARY.md](BUILD_SYSTEM_SUMMARY.md)
- **Main README**: [README.md](README.md)

---

## Support

**For build issues:**
1. Run `make info` to check system configuration
2. Check [BUILD.md](BUILD.md) troubleshooting section
3. Verify CUDA: `nvidia-smi && nvcc --version`
4. Clean rebuild: `make clean-all && make install-full`

**For usage questions:**
- Check examples in `examples/` directory
- Run demos: `make example-8k`
- Read API documentation in `docs/`

---

**Quick Reference Version**: 1.0.0
**Last Updated**: 2025-01-13

Print this card for quick access to common commands!
