# Benchmark Suite Quick Start Guide

## Installation

### 1. Install Dependencies

```bash
cd /home/user/Pixeltovoxelprojector/tests/benchmarks
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python test_installation.py
```

This will check all dependencies and show what's available.

### 3. (Optional) Compile CUDA Benchmarks

If you have an NVIDIA GPU and CUDA toolkit installed:

```bash
make
```

This will compile `voxel_benchmark` for GPU performance testing.

## Running Benchmarks

### Quick Test (Fast)

For a quick performance check during development:

```bash
python quick_benchmark.py
```

**Duration:** ~30 seconds
**Tests:** 2 core benchmarks with reduced iterations

### Individual Suites

Run specific benchmark suites:

```bash
# Main benchmark suite
python benchmark_suite.py

# Camera pipeline benchmarks
python camera_benchmark.py

# Network benchmarks
python network_benchmark.py

# CUDA voxel benchmarks (if compiled)
./voxel_benchmark
```

### Complete Suite (Recommended)

Run all benchmarks and generate comprehensive reports:

```bash
python run_all_benchmarks.py
```

**Duration:** 5-15 minutes
**Output:** Complete performance analysis with graphs and reports

## Understanding Results

### Output Files

All results are saved to `benchmark_results/`:

```
benchmark_results/
├── results_YYYYMMDD_HHMMSS.json     # Raw benchmark data
├── results_YYYYMMDD_HHMMSS.csv      # CSV format
├── report_YYYYMMDD_HHMMSS.html      # HTML report with graphs
├── throughput_comparison.png         # Throughput bar chart
├── latency_distribution.png          # Latency percentiles
├── resource_utilization.png          # CPU/GPU/Memory usage
├── camera/                           # Camera benchmark results
│   └── camera_benchmark_*.json
├── network/                          # Network benchmark results
│   └── network_benchmark_*.json
├── combined_results_*.json           # All suites combined
└── summary_*.txt                     # Text summary
```

### Key Metrics

**Throughput (FPS)**
- Higher is better
- Target: >30 FPS for real-time
- Indicates how many frames/operations per second

**Latency (ms)**
- Lower is better
- p50: Median latency
- p95: 95th percentile
- p99: 99th percentile (worst case)
- Target p99: <33ms for 30 FPS

**GPU Utilization (%)**
- 70-95% is optimal
- <50% may indicate CPU bottleneck
- >98% may indicate saturation

**Memory Bandwidth (GB/s)**
- Modern GPUs: 300-900 GB/s theoretical
- 60-80% of theoretical is good

## Performance Baselines

### Create Your Baseline

After first run:

```bash
python benchmark_suite.py
# When prompted: "Save these results as performance baseline? (y/n):"
# Type: y
```

This creates `baselines.json` for regression detection.

### Check for Regressions

Future runs will automatically compare against baseline:

```
WARNING: Performance regressions detected:
  - Throughput regression: 28.50 < 35.00 FPS
  - Latency regression: 38.20 > 35.00 ms
```

### Reset Baselines

To update baselines after optimization:

```bash
# Delete old baselines
rm benchmark_results/baselines.json

# Run benchmarks and save new baseline
python benchmark_suite.py
# Type: y when prompted
```

## Common Workflows

### Development Testing

Quick check after code changes:

```bash
python quick_benchmark.py
```

### Pre-Commit Testing

Before committing performance-critical changes:

```bash
python benchmark_suite.py
# Check for regressions in output
```

### Release Testing

Full performance validation before release:

```bash
python run_all_benchmarks.py
# Review HTML report
# Compare against previous releases
```

### Hardware Comparison

Testing on different hardware:

```bash
# Machine A
python run_all_benchmarks.py
cp benchmark_results/combined_results_*.json results_machine_a.json

# Machine B
python run_all_benchmarks.py
cp benchmark_results/combined_results_*.json results_machine_b.json

# Compare results
python compare_results.py results_machine_a.json results_machine_b.json
```

## Troubleshooting

### ImportError: No module named 'X'

```bash
pip install -r requirements.txt
```

### CUDA benchmarks fail

1. Check GPU is visible:
   ```bash
   nvidia-smi
   ```

2. Check CUDA toolkit:
   ```bash
   nvcc --version
   ```

3. Recompile:
   ```bash
   make clean
   make
   ```

### Benchmarks are slow

This is normal! Benchmarks run many iterations for accuracy:
- Quick benchmark: ~30 seconds
- Full suite: 5-15 minutes

For faster testing, reduce iterations in the code:
```python
suite.run_benchmark(..., iterations=10, warmup=2)  # Instead of 100/10
```

### Memory errors with large grids

Reduce grid size in benchmark calls:
```python
benchmark_voxel_ray_casting(grid_size=256)  # Instead of 500
```

### Network benchmarks show errors

Network benchmarks use localhost (127.0.0.1) by default.
For realistic results, test between separate machines:
```python
benchmark.benchmark_tcp_throughput(host="192.168.1.100")
```

## Advanced Usage

### Custom Benchmarks

Add your own benchmark function:

```python
def my_benchmark(param1, param2):
    # Your code to benchmark
    pass

# Run it
from benchmark_suite import BenchmarkSuite
suite = BenchmarkSuite()
suite.run_benchmark(
    "My Custom Test",
    my_benchmark,
    iterations=100,
    param1=value1,
    param2=value2
)
```

### Adjust Test Parameters

Edit the benchmark scripts:

```python
# In benchmark_suite.py
suite.run_benchmark(
    "Voxel Ray Casting",
    benchmark_voxel_ray_casting,
    iterations=200,      # More iterations = more accurate
    warmup=20,           # More warmup = more stable
    grid_size=1000,      # Larger grid = more realistic
    num_rays=5000        # More rays = stress test
)
```

### Export for CI/CD

```bash
# Run and save results
python benchmark_suite.py

# Extract key metrics for CI
python -c "
import json
with open('benchmark_results/results_*.json') as f:
    data = json.load(f)
    for result in data:
        if result['throughput_fps'] < 30:
            exit(1)  # Fail CI
"
```

## Best Practices

1. **Consistent Environment**
   - Close other applications
   - Ensure adequate cooling
   - Run on AC power (laptops)
   - Disable CPU throttling

2. **Baseline Management**
   - Create baseline on clean system
   - Update after major optimizations
   - Document hardware configuration
   - Version control baselines

3. **Interpretation**
   - Look at trends, not absolute values
   - Compare similar hardware only
   - Account for thermal throttling
   - Check multiple runs for consistency

4. **Optimization Workflow**
   - Baseline before changes
   - Make incremental changes
   - Benchmark after each change
   - Document what worked

## Getting Help

- Check `README.md` for detailed documentation
- Run `python test_installation.py` to verify setup
- Review example output in documentation
- Check hardware compatibility

## Next Steps

After running benchmarks:

1. Review HTML report for visual analysis
2. Identify bottlenecks (CPU, GPU, memory, network)
3. Optimize critical paths
4. Re-run and compare
5. Save new baseline if improvements validated

Happy benchmarking!
