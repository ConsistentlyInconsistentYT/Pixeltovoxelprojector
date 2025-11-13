# Makefile for CUDA Voxel Benchmarks

# Compiler
NVCC = nvcc

# Flags
NVCC_FLAGS = -O3 -arch=sm_60 -std=c++11
LDFLAGS = -lrt

# Targets
TARGET = voxel_benchmark

# Source files
CUDA_SRC = voxel_benchmark.cu

# Default target
all: $(TARGET)

# Compile CUDA benchmark
$(TARGET): $(CUDA_SRC)
	@echo "Compiling CUDA voxel benchmark..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_SRC) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Run benchmarks
run: $(TARGET)
	@echo "Running CUDA voxel benchmarks..."
	./$(TARGET)

# Clean
clean:
	rm -f $(TARGET)
	rm -f *.o
	@echo "Cleaned build artifacts"

# Check CUDA
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc || echo "ERROR: nvcc not found. Please install CUDA Toolkit."
	@nvcc --version || echo "ERROR: Cannot run nvcc"
	@nvidia-smi || echo "WARNING: nvidia-smi not available"

# Help
help:
	@echo "PixelToVoxel CUDA Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build voxel_benchmark (default)"
	@echo "  run          - Build and run benchmarks"
	@echo "  clean        - Remove build artifacts"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Build"
	@echo "  make run          # Build and run"
	@echo "  make clean        # Clean"

.PHONY: all run clean check-cuda help
