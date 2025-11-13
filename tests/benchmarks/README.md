# PixelToVoxelProjector Benchmark Suite

Comprehensive performance benchmarking suite for the PixelToVoxelProjector system.

## Overview

This benchmark suite provides detailed performance analysis across all major components:

- **Main Benchmark Suite** (`benchmark_suite.py`) - End-to-end pipeline benchmarking
- **Camera Benchmarks** (`camera_benchmark.py`) - 8K video processing performance
- **Voxel Benchmarks** (`voxel_benchmark.cu`) - CUDA kernel performance
- **Network Benchmarks** (`network_benchmark.py`) - Streaming performance

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

### CUDA Requirements (for voxel_benchmark.cu)

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or later
- nvcc compiler

## Installation

1. Install Python dependencies:
```bash
cd /home/user/Pixeltovoxelprojector/tests/benchmarks
pip install -r requirements.txt
```

2. Compile CUDA benchmarks:
```bash
make voxel_benchmark
```

## Usage

### Run All Benchmarks

```bash
python run_all_benchmarks.py
```

### Run Individual Benchmarks

**Main Benchmark Suite:**
```bash
python benchmark_suite.py
```

**Camera Pipeline Benchmarks:**
```bash
python camera_benchmark.py
```

**CUDA Voxel Benchmarks:**
```bash
./voxel_benchmark
```

**Network Benchmarks:**
```bash
python network_benchmark.py
```

## Benchmark Details

### 1. Main Benchmark Suite

**Tests:**
- Voxel ray casting performance
- Motion detection (8K frames)
- Voxel grid update throughput
- End-to-end pipeline latency

**Metrics:**
- Throughput (FPS)
- Latency percentiles (p50, p95, p99)
- CPU/GPU utilization
- Memory usage

**Output:**
- JSON results file
- CSV summary
- HTML report with graphs
- Performance baseline for regression detection

### 2. Camera Benchmarks

**Tests:**
- 8K video decode performance
- Motion extraction at multiple resolutions
- Multi-camera synchronization (8 cameras)
- Frame drop detection and analysis
- End-to-end camera pipeline

**Metrics:**
- Decode FPS and latency
- Motion detection throughput
- Synchronization accuracy
- Packet loss rates

**Output:**
- JSON results in `benchmark_results/camera/`

### 3. CUDA Voxel Benchmarks

**Tests:**
- Ray casting with DDA algorithm
- Atomic voxel updates
- Memory bandwidth (coalesced access)
- Voxel reduction operations

**Metrics:**
- Kernel execution time
- Throughput (GOPS)
- Memory bandwidth (GB/s)
- Grid size scalability

**Output:**
- Console output with detailed metrics
- Kernel configuration (blocks, threads)

### 4. Network Benchmarks

**Tests:**
- TCP throughput
- UDP throughput with packet loss tracking
- Latency measurement (ping-pong)
- Multi-client scalability
- Streaming latency (simulating voxel data)

**Metrics:**
- Throughput (Mbps)
- Latency (avg, p95, p99)
- Packet loss percentage
- Jitter
- Multi-client aggregate throughput

**Output:**
- JSON results in `benchmark_results/network/`

## Performance Baselines

The benchmark suite supports performance regression detection:

1. Run initial benchmarks to establish baseline:
```bash
python benchmark_suite.py
# When prompted, save as baseline: y
```

2. Future runs will compare against baseline and report regressions

3. Baselines are stored in: `benchmark_results/baselines.json`

## Interpreting Results

### Throughput
- Higher is better
- Target: >30 FPS for real-time processing
- 8K decode: 30-60 FPS typical
- Motion detection: 50-100 FPS typical

### Latency
- Lower is better
- Target p99 latency: <33ms (for 30 FPS)
- p50 should be <10ms for interactive performance

### GPU Utilization
- 70-95% indicates good GPU usage
- <50% may indicate CPU bottleneck
- >98% may indicate over-saturation

### Memory Bandwidth
- Modern GPUs: 300-900 GB/s theoretical
- Actual: 60-80% of theoretical is good
- <50% indicates inefficient memory access patterns

### Packet Loss
- TCP: Should be 0%
- UDP: <1% acceptable for real-time
- >5% indicates network issues

## Example Output

```
========================================
Benchmark: Voxel Ray Casting (500^3)
========================================
Duration:          2450.32 ms
Throughput:        40.81 FPS
Latency (p50):     23.12 ms
Latency (p95):     28.45 ms
Latency (p99):     31.67 ms
CPU Util:          45.2%
Memory:            1234.56 MB
GPU Util:          87.3%
GPU Memory:        2345.67 MB

No performance regressions detected.
```

## Troubleshooting

### GPU Not Detected

If CUDA benchmarks fail to find GPU:
```bash
nvidia-smi  # Check GPU is visible
nvcc --version  # Check CUDA toolkit installed
```

### Python Benchmarks Slow

1. Ensure OpenCV is using optimized build:
```bash
python -c "import cv2; print(cv2.getBuildInformation())"
```

2. Check for CPU-only operations (should use GPU when available)

### Network Benchmarks Show High Latency

When testing on localhost (127.0.0.1):
- Latency will be very low (< 1ms typical)
- For realistic results, test between separate machines
- Firewall rules may affect results

## Customization

### Adjust Test Parameters

Edit the benchmark scripts to modify:
- Grid sizes
- Number of iterations
- Test duration
- Resolution settings

Example:
```python
suite.run_benchmark(
    "Custom Test",
    benchmark_function,
    iterations=200,  # Increase for more accuracy
    warmup=20,       # More warmup iterations
    grid_size=1000   # Larger grid
)
```

### Add Custom Benchmarks

1. Create benchmark function:
```python
def my_custom_benchmark(param1, param2):
    # Your code here
    pass
```

2. Add to suite:
```python
suite.run_benchmark(
    "My Custom Test",
    my_custom_benchmark,
    iterations=100,
    param1=value1,
    param2=value2
)
```

## CI/CD Integration

For automated performance testing:

```bash
# Run benchmarks and exit with error on regression
python benchmark_suite.py --check-regression --exit-on-failure
```

## Performance Optimization Tips

Based on benchmark results:

1. **Low GPU Utilization**: Increase batch size or parallelize more work
2. **High CPU Utilization**: Move more work to GPU
3. **High Memory Usage**: Optimize data structures or streaming
4. **High Latency**: Check for synchronization points or blocking operations
5. **Low Throughput**: Profile to find bottlenecks

## Contributing

When adding new benchmarks:
1. Follow existing structure
2. Include warmup iterations
3. Report multiple metrics (throughput, latency, utilization)
4. Add documentation
5. Include baseline values

## License

Same as parent project.
