# Example Benchmark Output

This document shows example output from the benchmark suite.

## Quick Benchmark Output

```
============================================================
 Quick Performance Benchmark
============================================================

Running subset of benchmarks for quick verification...
For full benchmarks, use: python run_all_benchmarks.py

============================================================
Running benchmark: Quick Voxel Ray Casting
============================================================
Warmup (3 iterations)...
Running (20 iterations)...
  Progress: 10/20
  Progress: 20/20

============================================================
Results for: Quick Voxel Ray Casting
============================================================
Duration:          458.23 ms
Throughput:        43.65 FPS
Latency (p50):     21.84 ms
Latency (p95):     25.32 ms
Latency (p99):     27.15 ms
CPU Util:          42.3%
Memory:            987.45 MB
GPU Util:          0.0%
GPU Memory:        0.00 MB

No performance regressions detected.

============================================================
Running benchmark: Quick Voxel Updates
============================================================
Warmup (3 iterations)...
Running (30 iterations)...
  Progress: 10/30
  Progress: 20/30
  Progress: 30/30

============================================================
Results for: Quick Voxel Updates
============================================================
Duration:          234.56 ms
Throughput:        127.94 FPS
Latency (p50):     7.45 ms
Latency (p95):     8.92 ms
Latency (p99):     9.34 ms
CPU Util:          38.7%
Memory:            856.23 MB
GPU Util:          0.0%
GPU Memory:        0.00 MB

No performance regressions detected.

Saved results to benchmark_results/results_20251113_143022.json
Saved CSV to benchmark_results/results_20251113_143022.csv

============================================================
 Quick Benchmark Complete
============================================================

Results saved to: benchmark_results/
```

## Main Benchmark Suite Output

```
============================================================
PixelToVoxel Performance Benchmark Suite
============================================================

============================================================
Running benchmark: Voxel Ray Casting (500^3)
============================================================
Warmup (5 iterations)...
Running (50 iterations)...
  Progress: 10/50
  Progress: 20/50
  Progress: 30/50
  Progress: 40/50
  Progress: 50/50

============================================================
Results for: Voxel Ray Casting (500^3)
============================================================
Duration:          1234.56 ms
Throughput:        40.51 FPS
Latency (p50):     23.45 ms
Latency (p95):     27.89 ms
Latency (p99):     30.12 ms
CPU Util:          45.2%
Memory:            2134.56 MB
GPU Util:          0.0%
GPU Memory:        0.00 MB

No performance regressions detected.

============================================================
Running benchmark: Motion Detection (8K)
============================================================
Warmup (5 iterations)...
Running (50 iterations)...
  Progress: 10/50
  Progress: 20/50
  Progress: 30/50
  Progress: 40/50
  Progress: 50/50

============================================================
Results for: Motion Detection (8K)
============================================================
Duration:          2345.67 ms
Throughput:        21.32 FPS
Latency (p50):     45.23 ms
Latency (p95):     48.76 ms
Latency (p99):     51.34 ms
CPU Util:          67.8%
Memory:            2567.89 MB
GPU Util:          0.0%
GPU Memory:        0.00 MB

No performance regressions detected.

Saved results to benchmark_results/results_20251113_143530.json
Saved CSV to benchmark_results/results_20251113_143530.csv

Generated report: benchmark_results/report_20251113_143530.html

============================================================
Save these results as performance baseline? (y/n): y
Saved 3 baselines to benchmark_results/baselines.json

============================================================
Benchmark suite completed!
============================================================
```

## Camera Benchmark Output

```
============================================================
Benchmarking 8K Video Decode Performance
============================================================
Generating synthetic 8K frames (7680x4320)...
  Processed 50/300 frames
  Processed 100/300 frames
  Processed 150/300 frames
  Processed 200/300 frames
  Processed 250/300 frames
  Processed 300/300 frames

Results:
  Avg Decode Time: 42.35 ms
  Decode FPS:      23.61
  Max FPS:         28.45
  p95 Latency:     45.67 ms
  p99 Latency:     48.92 ms

============================================================
Benchmarking Motion Extraction Throughput
============================================================

Testing 8K (7680x4320)...
  Processed 50/300 frames
  Processed 100/300 frames
  Processed 150/300 frames
  Processed 200/300 frames
  Processed 250/300 frames
  Processed 300/300 frames
  Avg Motion Time: 38.45 ms
  Motion FPS:      26.01
  p99 Latency:     42.34 ms

Testing 4K (3840x2160)...
  Processed 50/300 frames
  Processed 100/300 frames
  Processed 150/300 frames
  Processed 200/300 frames
  Processed 250/300 frames
  Processed 300/300 frames
  Avg Motion Time: 9.23 ms
  Motion FPS:      108.35
  p99 Latency:     11.45 ms

Testing 1080p (1920x1080)...
  Processed 50/300 frames
  Processed 100/300 frames
  Processed 150/300 frames
  Processed 200/300 frames
  Processed 250/300 frames
  Processed 300/300 frames
  Avg Motion Time: 2.34 ms
  Motion FPS:      427.35
  p99 Latency:     3.12 ms

============================================================
Results saved to: benchmark_results/camera/camera_benchmark_20251113_144022.json
============================================================
```

## CUDA Benchmark Output

```
========================================
CUDA Voxel Benchmark Suite
========================================

GPU: NVIDIA GeForce RTX 3080
Compute Capability: 8.6
Global Memory: 10.00 GB
Multiprocessors: 68
Max Threads/Block: 1024

Benchmarking Ray Casting (500^3 grid, 100000 rays)...

========================================
Benchmark: Voxel Ray Casting (DDA)
========================================
Duration:         8.45 ms
Throughput:       68.34 GOPS
Memory BW:        234.56 GB/s
Kernel Time:      8.23 ms
Blocks:           391
Threads/Block:    256
========================================

Benchmarking Voxel Updates (500^3 grid, 1000000 updates)...

========================================
Benchmark: Voxel Updates (Atomic)
========================================
Duration:         4.23 ms
Throughput:       236.41 GOPS
Memory BW:        12.34 GB/s
Kernel Time:      4.12 ms
Blocks:           3907
Threads/Block:    256
========================================

Benchmarking Memory Bandwidth (125000000 elements)...

========================================
Benchmark: Memory Bandwidth (Coalesced)
========================================
Duration:         1.23 ms
Throughput:       101.63 GOPS
Memory BW:        406.50 GB/s
Kernel Time:      1.23 ms
Blocks:           488282
Threads/Block:    256
========================================

Benchmark suite completed!
```

## Network Benchmark Output

```
============================================================
Benchmarking TCP Throughput (10s)
============================================================

Results:
  Bytes Sent:    8,456,789,012
  Duration:      10.02 s
  Throughput:    6,749.23 Mbps

============================================================
Benchmarking UDP Throughput (10s)
============================================================

Results:
  Packets Sent:     7,142,857
  Packets Received: 7,135,324
  Packet Loss:      0.11%
  Throughput:       7,999.45 Mbps

============================================================
Benchmarking TCP Latency (1000 pings)
============================================================
  Progress: 100/1000
  Progress: 200/1000
  Progress: 300/1000
  Progress: 400/1000
  Progress: 500/1000
  Progress: 600/1000
  Progress: 700/1000
  Progress: 800/1000
  Progress: 900/1000
  Progress: 1000/1000

Results:
  Avg Latency:  0.23 ms
  p50 Latency:  0.21 ms
  p95 Latency:  0.34 ms
  p99 Latency:  0.45 ms

============================================================
Benchmarking Multi-Client Scalability (10 clients)
============================================================

Results:
  Clients Completed:    10/10
  Total Bytes:          12,345,678,901
  Aggregate Throughput: 9,876.54 Mbps
  Per-Client Avg:       987.65 Mbps

============================================================
Results saved to: benchmark_results/network/network_benchmark_20251113_144530.json
============================================================
```

## Full Suite Output

```
======================================================================
 PixelToVoxel Comprehensive Benchmark Suite
======================================================================

Started: 2025-11-13 14:45:30

Checking environment...
  Python: 3.11.14
  numpy: OK
  cv2: OK
  matplotlib: OK
  psutil: OK
  CUDA: OK

======================================================================
 Running Main Benchmark Suite
======================================================================
[... main suite output ...]

✓ Main benchmark suite completed

======================================================================
 Running Camera Benchmark Suite
======================================================================
[... camera suite output ...]

✓ Camera benchmark suite completed

======================================================================
 Running CUDA Voxel Benchmarks
======================================================================
[... CUDA output ...]

✓ CUDA benchmark suite completed

======================================================================
 Running Network Benchmark Suite
======================================================================
[... network output ...]

✓ Network benchmark suite completed

Combined results saved to: benchmark_results/combined_results_20251113_144530.json
Summary saved to: benchmark_results/summary_20251113_144530.txt

======================================================================
 Benchmark Suite Completed
======================================================================

Total Duration: 487.3 seconds
Results saved to: /home/user/Pixeltovoxelprojector/tests/benchmarks/benchmark_results
```

## HTML Report Example

The HTML report includes:

### Summary Section
```
Avg Throughput: 45.2 FPS
Avg Latency: 24.3 ms
Avg CPU Usage: 52%
Avg GPU Usage: 78%
```

### Performance Charts
- Throughput Comparison (bar chart)
- Latency Distribution (grouped bar chart with p50/p95/p99)
- Resource Utilization (CPU/GPU utilization and memory)

### Detailed Results Table
| Benchmark | Throughput (FPS) | p50 (ms) | p95 (ms) | p99 (ms) | CPU % | GPU % | Memory (MB) | Status |
|-----------|------------------|----------|----------|----------|-------|-------|-------------|--------|
| Voxel Ray Casting | 40.51 | 23.45 | 27.89 | 30.12 | 45.2 | 0.0 | 2134.6 | PASS |
| Motion Detection | 21.32 | 45.23 | 48.76 | 51.34 | 67.8 | 0.0 | 2567.9 | PASS |
| Voxel Updates | 127.94 | 7.45 | 8.92 | 9.34 | 38.7 | 0.0 | 856.2 | PASS |

## CSV Output Example

```csv
name,duration_ms,throughput_fps,latency_p50_ms,latency_p95_ms,latency_p99_ms,cpu_percent,memory_mb,gpu_percent,gpu_memory_mb,timestamp
Voxel Ray Casting (500^3),1234.56,40.51,23.45,27.89,30.12,45.2,2134.56,0.0,0.00,2025-11-13T14:35:30.123456
Motion Detection (8K),2345.67,21.32,45.23,48.76,51.34,67.8,2567.89,0.0,0.00,2025-11-13T14:37:15.654321
Voxel Grid Updates,187.23,127.94,7.45,8.92,9.34,38.7,856.23,0.0,0.00,2025-11-13T14:37:22.987654
```

## Performance Regression Example

```
============================================================
Results for: Voxel Ray Casting (500^3)
============================================================
Duration:          1456.78 ms
Throughput:        34.32 FPS
Latency (p50):     27.89 ms
Latency (p95):     32.45 ms
Latency (p99):     35.67 ms
CPU Util:          48.9%
Memory:            2345.67 MB
GPU Util:          0.0%
GPU Memory:        0.00 MB

WARNING: Performance regressions detected:
  - Throughput regression: 34.32 < 36.45 FPS
  - Latency regression: 35.67 > 30.12 ms
```

## Summary Text File Example

```
======================================================================
 PixelToVoxel Benchmark Summary
======================================================================

Started:  2025-11-13 14:45:30
Finished: 2025-11-13 14:53:37
Duration: 487.3 seconds

======================================================================
 Suite Status
======================================================================

main_suite           ✓ PASS
camera_suite         ✓ PASS
cuda_suite           ✓ PASS
network_suite        ✓ PASS

======================================================================
 Main Suite Results
======================================================================

Voxel Ray Casting (500^3)
  Throughput: 40.51 FPS
  p99 Latency: 30.12 ms

Motion Detection (8K)
  Throughput: 21.32 FPS
  p99 Latency: 51.34 ms

Voxel Grid Updates
  Throughput: 127.94 FPS
  p99 Latency: 9.34 ms

======================================================================
 End of Summary
======================================================================
```
