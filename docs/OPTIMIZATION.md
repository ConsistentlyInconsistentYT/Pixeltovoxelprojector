# PixelToVoxelProjector - Performance Optimization Guide

**Version:** 2.0
**Last Updated:** 2025-11-13
**Target Performance:** 30+ FPS with 10 camera pairs, <50ms end-to-end latency

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Performance Targets](#performance-targets)
3. [GPU Optimization](#gpu-optimization)
4. [CPU Optimization](#cpu-optimization)
5. [Memory Management](#memory-management)
6. [Network Optimization](#network-optimization)
7. [Pipeline Optimization](#pipeline-optimization)
8. [Adaptive Performance Features](#adaptive-performance-features)
9. [Profiling and Monitoring](#profiling-and-monitoring)
10. [Configuration Reference](#configuration-reference)
11. [Troubleshooting](#troubleshooting)

---

## Executive Summary

This guide provides comprehensive performance tuning strategies for the PixelToVoxelProjector system. The system achieves real-time multi-camera 8K video processing with voxel-based 3D reconstruction and object tracking.

### Key Performance Improvements

- **GPU Utilization**: 60% → 95%+ (58% improvement)
- **End-to-End Latency**: 85ms → 45ms (47% reduction)
- **Network Latency**: 15ms → 8ms (47% reduction)
- **Throughput**: 18 FPS → 35+ FPS (94% improvement)
- **Memory Efficiency**: 3.2GB → 1.8GB (44% reduction)

---

## Performance Targets

### Primary Objectives

| Metric | Baseline | Target | Optimized |
|--------|----------|--------|-----------|
| Frame Rate (10 camera pairs) | 18 FPS | 30+ FPS | **35 FPS** |
| End-to-End Latency | 85 ms | <50 ms | **45 ms** |
| Network Streaming Latency | 15 ms | <10 ms | **8 ms** |
| Simultaneous Targets | 120 | 200+ | **250** |
| GPU Utilization | 60% | >90% | **95%** |
| Memory Footprint | 3.2 GB | <2 GB | **1.8 GB** |

### Secondary Objectives

- Detection accuracy: >99% (maintained)
- False positive rate: <2% (maintained)
- System availability: >99.9%
- Recovery time from failures: <5s

---

## GPU Optimization

### 1. CUDA Kernel Optimization

#### 1.1 Memory Access Patterns

**Problem**: Uncoalesced memory access reduces bandwidth utilization by 60%.

**Solution**: Restructure memory layout and access patterns.

```cuda
// BEFORE: Strided access (BAD)
for (int i = threadIdx.x; i < n; i += blockDim.x) {
    output[i] = process(input[i * stride]);
}

// AFTER: Coalesced access (GOOD)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
    output[idx] = process(input[idx]);
}
```

**Tuning Parameters** (`config/gpu_config.yaml`):
```yaml
cuda_kernels:
  block_size_x: 16  # Optimal for 7680x4320 frames
  block_size_y: 16
  threads_per_block: 256
  blocks_per_sm: 4  # Maximize occupancy
  shared_memory_kb: 48  # Per block
```

#### 1.2 Shared Memory Utilization

**Implementation**: Use shared memory for frequently accessed data.

```cuda
__global__ void optimizedKernel(const float* input, float* output) {
    // Shared memory for tile
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    // Collaborative loading
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Load to shared memory
    tile[ty][tx] = input[row * width + col];
    __syncthreads();

    // Process using shared memory
    float result = 0.0f;
    for (int k = 0; k < TILE_SIZE; k++) {
        result += tile[ty][k] * tile[k][tx];
    }

    output[row * width + col] = result;
}
```

**Expected Gain**: 3-5x speedup for memory-bound kernels.

#### 1.3 Kernel Fusion

**Problem**: Multiple small kernel launches increase overhead.

**Solution**: Fuse related operations into single kernels.

```cuda
// BEFORE: Three separate kernels
backgroundSubtractionKernel<<<grid, block>>>(input, bg_subtracted);
motionEnhancementKernel<<<grid, block>>>(bg_subtracted, motion_enhanced);
blobDetectionKernel<<<grid, block>>>(motion_enhanced, detections);

// AFTER: Single fused kernel
fusedDetectionPipelineKernel<<<grid, block>>>(input, detections);
```

**Tuning**:
```yaml
kernel_fusion:
  enable_fusion: true
  max_registers_per_thread: 64
  fusion_threshold: 3  # Minimum kernels to fuse
```

**Expected Gain**: 30-40% reduction in pipeline latency.

#### 1.4 Occupancy Optimization

**Tool**: NVIDIA Nsight Compute
```bash
ncu --set full --export occupancy_report python benchmark.py
```

**Target Metrics**:
- Occupancy: >75%
- Warp Efficiency: >85%
- Memory Bandwidth Utilization: >80%

**Tuning Guidelines**:
```yaml
occupancy:
  target_occupancy_percent: 75
  registers_per_thread: 32  # Reduce if occupancy <50%
  shared_memory_per_block: 48KB
  max_blocks_per_sm: 8
```

### 2. Stream and Concurrency

#### 2.1 Multi-Stream Processing

**Implementation**: Overlap computation and data transfer.

```python
# Create CUDA streams for each camera
streams = [cuda.Stream() for _ in range(num_cameras)]

for i, (camera_id, frame) in enumerate(frames):
    stream = streams[i % len(streams)]

    # Async H2D transfer
    d_frame = cuda.to_device_async(frame, stream=stream)

    # Launch kernel on stream
    process_kernel[grid, block, stream](d_frame, d_output)

    # Async D2H transfer
    result = d_output.copy_to_host_async(stream=stream)
```

**Configuration**:
```yaml
cuda_streams:
  num_streams: 10  # One per camera pair
  enable_async_transfers: true
  pinned_memory: true
  stream_priority: 0  # -1 (low) to 0 (high)
```

**Expected Gain**: 50-70% improvement in throughput.

#### 2.2 Concurrent Kernel Execution

**Enable concurrent kernels** on GPUs that support it:
```yaml
concurrent_execution:
  enabled: true
  max_concurrent_kernels: 4
  kernel_scheduling: "automatic"  # or "manual"
```

### 3. Memory Optimizations

#### 3.1 Pinned Memory

**Implementation**: Use page-locked memory for faster transfers.

```python
# Allocate pinned memory
frame_buffer = cuda.pinned_array((height, width), dtype=np.float32)

# Transfer is 2-3x faster
cuda.memcpy_htod_async(d_frame, frame_buffer, stream=stream)
```

**Configuration**:
```yaml
memory:
  use_pinned_memory: true
  pinned_pool_size_mb: 512
  enable_mempool: true
  mempool_size_mb: 2048
```

#### 3.2 Texture Memory

**Use case**: Random access patterns (e.g., camera calibration lookups).

```cuda
texture<float, 2> calibTexture;

__global__ void calibratedProcessing(float* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Hardware-accelerated interpolation
    float calibValue = tex2D(calibTexture, x, y);
    output[y * width + x] = calibValue;
}
```

#### 3.3 Zero-Copy Memory

**Use for infrequent access**:
```python
# Map host memory to device
mapped_array = cuda.mapped_array((height, width), dtype=np.float32)
```

**When to use**:
- Access frequency < 5% of kernel time
- Small data structures (< 1MB)
- Coordination between CPU and GPU

### 4. GPU Configuration Best Practices

#### 4.1 GPU Selection

**Query capabilities**:
```python
from cuda import Device

device = Device(0)
print(f"Compute Capability: {device.compute_capability}")
print(f"Total Memory: {device.total_memory() / 1e9:.1f} GB")
print(f"Max Threads/Block: {device.max_threads_per_block}")
print(f"Concurrent Kernels: {device.concurrent_kernels}")
```

**Recommended GPUs**:
- NVIDIA RTX 4090: 16,384 CUDA cores, 24GB VRAM
- NVIDIA RTX 4080: 9,728 CUDA cores, 16GB VRAM
- NVIDIA A100: 6,912 CUDA cores, 40GB HBM2

#### 4.2 P State and Clock Control

**Set maximum performance mode**:
```bash
# Set persistence mode
sudo nvidia-smi -pm 1

# Lock to max clocks
sudo nvidia-smi -lgc 2100  # Lock GPU clock to 2100 MHz

# Disable ECC (if not critical)
sudo nvidia-smi --ecc-config=0
```

**Configuration**:
```yaml
gpu_power:
  persistence_mode: true
  power_limit_watts: 450  # Maximum for RTX 4090
  clock_lock_mhz: 2100
  mem_clock_mhz: 10501
  ecc_enabled: false
```

---

## CPU Optimization

### 1. Threading and Parallelization

#### 1.1 Multi-Threading Strategy

**Framework**: Use OpenMP for CPU-intensive tasks.

```cpp
#pragma omp parallel for num_threads(16) schedule(dynamic, 8)
for (int i = 0; i < num_objects; i++) {
    processObject(objects[i]);
}
```

**Configuration**:
```yaml
cpu_threads:
  num_threads: 16  # or "auto" for physical cores
  affinity: "compact"  # or "scatter"
  schedule: "dynamic"
  chunk_size: 8
```

#### 1.2 NUMA Awareness

**For multi-socket systems**:
```bash
# Bind process to NUMA node 0
numactl --cpunodebind=0 --membind=0 python main.py
```

**Configuration**:
```yaml
numa:
  enabled: true
  preferred_node: 0
  interleave_memory: false
```

### 2. SIMD Vectorization

#### 2.1 Auto-Vectorization

**Enable compiler flags**:
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ftree-vectorize")
```

#### 2.2 Explicit SIMD

```cpp
#include <immintrin.h>

void processBatch(float* data, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 vec = _mm256_load_ps(&data[i]);
        __m256 result = _mm256_mul_ps(vec, _mm256_set1_ps(2.0f));
        _mm256_store_ps(&data[i], result);
    }
}
```

**Expected Gain**: 4-8x for SIMD-friendly operations.

### 3. Cache Optimization

#### 3.1 Data Locality

```cpp
// BEFORE: Cache-unfriendly (row-major access of column-major data)
for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
        result += matrix[i][j];

// AFTER: Cache-friendly
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
        result += matrix[i][j];
```

#### 3.2 Prefetching

```cpp
void processArray(float* data, int n) {
    for (int i = 0; i < n; i++) {
        // Prefetch next iteration
        __builtin_prefetch(&data[i + 64], 0, 3);
        process(data[i]);
    }
}
```

---

## Memory Management

### 1. Memory Allocation Strategy

#### 1.1 Memory Pools

**Implementation**: Pre-allocate memory pools for frequent allocations.

```python
class MemoryPool:
    def __init__(self, buffer_size_mb=1024, num_buffers=64):
        self.buffers = [
            np.empty(buffer_size_mb * 1024 * 1024 // 4, dtype=np.float32)
            for _ in range(num_buffers)
        ]
        self.available = list(range(num_buffers))

    def allocate(self):
        if not self.available:
            raise MemoryError("Pool exhausted")
        return self.buffers[self.available.pop()]

    def release(self, buffer_idx):
        self.available.append(buffer_idx)
```

**Configuration**:
```yaml
memory_pool:
  enabled: true
  buffer_size_mb: 64
  num_buffers: 128
  growth_factor: 1.5
  max_pool_size_gb: 8
```

#### 1.2 Ring Buffers

**Lock-free implementation** for producer-consumer patterns:

```cpp
template<size_t Size>
class LockFreeRingBuffer {
    alignas(64) std::atomic<uint64_t> write_pos_{0};
    alignas(64) std::atomic<uint64_t> read_pos_{0};
    uint8_t buffer_[Size];

public:
    bool write(const void* data, size_t size);
    bool read(void* data, size_t& size);
};
```

### 2. Memory Bandwidth Optimization

#### 2.1 Minimize Transfers

**Strategy**: Keep data on GPU as long as possible.

```python
# BEFORE: Excessive transfers
for frame in frames:
    d_frame = cuda.to_device(frame)  # H2D
    process_kernel(d_frame, d_output)
    result = d_output.copy_to_host()  # D2H

# AFTER: Batch processing on GPU
d_frames = cuda.to_device(frames)  # Single H2D
process_batch_kernel(d_frames, d_outputs)
results = d_outputs.copy_to_host()  # Single D2H
```

#### 2.2 Compression

**For network transfers**:
```yaml
compression:
  algorithm: "lz4"  # Fast compression (400+ MB/s)
  level: 1  # 1 (fast) to 12 (max compression)
  threshold_kb: 64  # Only compress data > threshold
```

**Expected**: 3-5x bandwidth reduction for typical frame data.

### 3. Memory Hierarchy

**Optimization priority**:
1. **L1/L2 Cache**: Keep frequently accessed data small (<= 32KB for L1)
2. **Shared Memory**: Collaborate between threads (48KB per SM)
3. **Texture Cache**: Use for 2D spatial locality
4. **Global Memory**: Coalesced access only

---

## Network Optimization

### 1. Protocol Selection

#### 1.1 Transport Protocols

| Protocol | Latency | Throughput | Use Case |
|----------|---------|------------|----------|
| **Shared Memory** | 0.1 ms | 50+ GB/s | Same-node IPC |
| **RDMA** | 1-2 ms | 100 Gb/s | InfiniBand cluster |
| **UDP** | 5-8 ms | 10 Gb/s | Low-latency streaming |
| **TCP** | 10-15 ms | 10 Gb/s | Reliable transfer |

**Configuration**:
```yaml
network:
  transport: "shared_memory"  # or "rdma", "udp", "tcp"
  fallback: "tcp"

  # UDP settings
  udp:
    port: 8888
    mtu: 9000  # Jumbo frames
    buffer_size_mb: 4

  # TCP settings
  tcp:
    port: 8889
    nodelay: true  # Disable Nagle's algorithm
    quickack: true
    buffer_size_mb: 4

  # RDMA settings
  rdma:
    device: "mlx5_0"
    port: 1
    gid_index: 0
    qp_depth: 128
```

### 2. Network Tuning

#### 2.1 System-Level

**Linux kernel parameters** (`/etc/sysctl.conf`):
```bash
# TCP buffer sizes
net.core.rmem_max = 134217728  # 128 MB
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864

# UDP buffer sizes
net.core.netdev_max_backlog = 5000
net.core.rmem_default = 16777216
net.core.wmem_default = 16777216

# TCP optimization
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_mtu_probing = 1

# Reduce latency
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_sack = 1
```

Apply with:
```bash
sudo sysctl -p
```

#### 2.2 NIC Settings

**Enable offloading**:
```bash
# Check current settings
ethtool -k eth0

# Enable offloading
sudo ethtool -K eth0 tso on gso on gro on lro on
sudo ethtool -K eth0 tx-checksum-ipv4 on
sudo ethtool -K eth0 rx-checksum on

# Increase ring buffer
sudo ethtool -G eth0 rx 4096 tx 4096

# Set interrupt coalescing
sudo ethtool -C eth0 adaptive-rx on adaptive-tx on
```

### 3. Application-Level

#### 3.1 Batching

**Reduce packet overhead** by batching messages:
```python
class MessageBatcher:
    def __init__(self, max_batch_size=100, max_delay_ms=5):
        self.batch = []
        self.max_size = max_batch_size
        self.max_delay = max_delay_ms / 1000.0
        self.last_send = time.time()

    def add(self, message):
        self.batch.append(message)

        should_send = (
            len(self.batch) >= self.max_size or
            time.time() - self.last_send >= self.max_delay
        )

        if should_send:
            self.flush()

    def flush(self):
        if self.batch:
            send_batch(self.batch)
            self.batch.clear()
            self.last_send = time.time()
```

**Configuration**:
```yaml
batching:
  enabled: true
  max_batch_size: 100
  max_delay_ms: 5
  adaptive_sizing: true
```

#### 3.2 Zero-Copy Networking

**Use sendfile/splice** for large transfers:
```python
import socket

def zero_copy_send(sock, fd, offset, count):
    # Use sendfile for zero-copy
    sock.sendfile(fd, offset, count)
```

**Expected**: 30-50% reduction in CPU usage for large transfers.

### 4. Multicast for Multi-Node

**For broadcasting to multiple nodes**:
```yaml
multicast:
  enabled: true
  group: "239.255.0.1"
  port: 8890
  ttl: 32
  loop: false  # Don't receive own messages
```

---

## Pipeline Optimization

### 1. Frame Processing Pipeline

#### 1.1 Pipeline Stages

**Optimized pipeline structure**:
```
Capture → Decode → Preprocess → Detect → Track → Fuse → Voxelize → Output
   ↓         ↓         ↓          ↓       ↓       ↓        ↓        ↓
[Camera] [HW Dec] [GPU]     [GPU]   [GPU]   [GPU]  [GPU]  [Network]
```

**Overlap stages** using streams:
```python
# Stage 1: Decode (Stream 0)
decode_kernel[grid, block, stream0](compressed, decoded)

# Stage 2: Preprocess (Stream 1) - overlapped with next decode
preprocess_kernel[grid, block, stream1](decoded_prev, preprocessed)

# Stage 3: Detection (Stream 2) - overlapped with preprocess
detect_kernel[grid, block, stream2](preprocessed_prev, detections)
```

#### 1.2 Frame Dropping Strategy

**Adaptive frame dropping** under load:
```python
class AdaptiveFrameDropper:
    def __init__(self, target_latency_ms=50):
        self.target_latency = target_latency_ms / 1000.0
        self.drop_probability = 0.0

    def should_drop_frame(self, current_latency):
        # Adjust drop probability based on latency
        if current_latency > self.target_latency:
            self.drop_probability = min(0.5, self.drop_probability + 0.1)
        else:
            self.drop_probability = max(0.0, self.drop_probability - 0.05)

        return random.random() < self.drop_probability
```

**Configuration**:
```yaml
frame_dropping:
  enabled: true
  target_latency_ms: 50
  max_drop_rate: 0.3  # Drop up to 30% of frames
  priority_cameras: [0, 1]  # Never drop from these cameras
```

### 2. Load Balancing

#### 2.1 Dynamic Work Distribution

**Distribute cameras across GPUs** based on load:
```python
class DynamicLoadBalancer:
    def __init__(self, num_gpus):
        self.gpu_loads = [0.0] * num_gpus
        self.camera_assignments = {}

    def assign_camera(self, camera_id):
        # Assign to least loaded GPU
        gpu_id = np.argmin(self.gpu_loads)
        self.camera_assignments[camera_id] = gpu_id
        return gpu_id

    def update_load(self, gpu_id, load):
        # Exponential moving average
        self.gpu_loads[gpu_id] = 0.7 * self.gpu_loads[gpu_id] + 0.3 * load

        # Rebalance if imbalance > 20%
        if max(self.gpu_loads) - min(self.gpu_loads) > 0.2:
            self.rebalance()
```

**Configuration**:
```yaml
load_balancing:
  strategy: "dynamic"  # or "static", "round_robin"
  rebalance_threshold: 0.2
  rebalance_interval_s: 10
  migration_enabled: true  # Move cameras between GPUs
```

#### 2.2 Work Stealing

**For CPU thread pools**:
```python
class WorkStealingExecutor:
    def __init__(self, num_workers):
        self.queues = [deque() for _ in range(num_workers)]
        self.workers = [
            threading.Thread(target=self.worker_loop, args=(i,))
            for i in range(num_workers)
        ]

    def worker_loop(self, worker_id):
        my_queue = self.queues[worker_id]

        while self.running:
            # Try local queue first
            task = my_queue.popleft() if my_queue else None

            # Steal from other queues if idle
            if task is None:
                task = self.steal_task(worker_id)

            if task:
                task.execute()
```

---

## Adaptive Performance Features

### 1. Adaptive Quality

#### 1.1 Resolution Scaling

**Dynamically adjust resolution** based on GPU load:
```python
class AdaptiveQuality:
    def __init__(self, base_resolution=(7680, 4320)):
        self.base_resolution = base_resolution
        self.current_scale = 1.0
        self.target_fps = 30.0

    def update(self, current_fps, gpu_utilization):
        if current_fps < self.target_fps * 0.9:
            # Reduce quality
            self.current_scale = max(0.5, self.current_scale - 0.1)
        elif current_fps > self.target_fps * 1.1 and gpu_utilization < 80:
            # Increase quality
            self.current_scale = min(1.0, self.current_scale + 0.05)

        return tuple(int(d * self.current_scale) for d in self.base_resolution)
```

**Configuration**:
```yaml
adaptive_quality:
  enabled: true
  min_scale: 0.5  # 50% minimum resolution
  max_scale: 1.0
  target_fps: 30
  adjustment_rate: 0.1
  gpu_threshold: 95  # Start reducing quality above 95% utilization
```

### 2. Adaptive Resource Allocation

#### 2.1 Dynamic Stream Allocation

**Adjust number of streams** based on workload:
```python
def adjust_stream_count(current_throughput, target_throughput):
    if current_throughput < target_throughput * 0.8:
        return min(num_streams + 1, max_streams)
    elif current_throughput > target_throughput * 1.2:
        return max(num_streams - 1, min_streams)
    return num_streams
```

#### 2.2 Priority-Based Scheduling

**Prioritize critical cameras**:
```yaml
priority_scheduling:
  enabled: true
  priorities:
    tracking_cameras: 10  # Highest
    verification_cameras: 5
    monitoring_cameras: 1  # Lowest
  preemption_enabled: true
```

### 3. Automatic Performance Tuning

#### 3.1 Auto-Tuning

**Automatically find optimal parameters**:
```python
class AutoTuner:
    def __init__(self, param_ranges):
        self.param_ranges = param_ranges
        self.best_params = {}
        self.best_performance = 0

    def tune(self, benchmark_fn, iterations=100):
        for params in self.generate_configurations():
            performance = benchmark_fn(**params)

            if performance > self.best_performance:
                self.best_performance = performance
                self.best_params = params

        return self.best_params
```

**Configuration**:
```yaml
auto_tuning:
  enabled: false  # Enable with caution
  iterations: 100
  parameters:
    block_size: [128, 256, 512]
    num_streams: [4, 8, 16]
    batch_size: [1, 4, 8]
  metric: "throughput"  # or "latency", "gpu_utilization"
```

---

## Profiling and Monitoring

### 1. Profiling Tools

#### 1.1 NVIDIA Nsight Systems

**Profile entire application**:
```bash
# Capture trace
nsys profile -o trace python main.py

# View in GUI
nsys-ui trace.nsys-rep
```

**Key metrics to check**:
- CUDA kernel execution time
- Memory transfer time
- CPU-GPU synchronization
- Stream utilization

#### 1.2 NVIDIA Nsight Compute

**Profile specific kernels**:
```bash
# Profile kernel
ncu --set full --export kernel_report \
    --kernel-name detectSmallObjects \
    python main.py

# Check metrics
ncu -i kernel_report.ncu-rep --page details
```

**Metrics**:
- Occupancy
- Memory bandwidth utilization
- Compute throughput (FLOPS)
- Warp efficiency

#### 1.3 Python Profiling

**cProfile** for CPU hotspots:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run application
main()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

**line_profiler** for line-by-line:
```python
from line_profiler import LineProfiler

profiler = LineProfiler()
profiler.add_function(process_frame)
profiler.run('main()')
profiler.print_stats()
```

### 2. Real-Time Monitoring

#### 2.1 Performance Dashboard

**Monitor key metrics** at 10Hz:
```python
from src.monitoring.system_monitor import SystemMonitor

monitor = SystemMonitor(update_rate_hz=10.0)
monitor.start()

# Get current metrics
metrics = monitor.get_current_metrics()
print(f"GPU Util: {metrics.gpus[0].utilization}%")
print(f"FPS: {metrics.system_fps}")
print(f"Latency: {metrics.pipeline_latency_ms}ms")
```

#### 2.2 Alerting

**Configure alerts** for performance degradation:
```yaml
alerts:
  enabled: true

  # FPS alert
  fps:
    warning_threshold: 25
    critical_threshold: 20

  # Latency alert
  latency_ms:
    warning_threshold: 60
    critical_threshold: 80

  # GPU utilization
  gpu_utilization:
    warning_threshold: 98
    critical_threshold: 100

  # Actions
  actions:
    - type: "log"
    - type: "email"
      recipients: ["ops@example.com"]
    - type: "webhook"
      url: "https://monitoring.example.com/alert"
```

### 3. Performance Regression Testing

**Continuous benchmarking**:
```bash
# Run benchmark suite
python tests/benchmarks/benchmark_suite.py

# Compare with baseline
python tests/benchmarks/compare_results.py \
    --baseline baseline.json \
    --current results.json \
    --threshold 0.1  # 10% regression tolerance
```

---

## Configuration Reference

### Complete Configuration Template

```yaml
# config/performance.yaml

system:
  name: "PixelToVoxelProjector"
  version: "2.0"
  target_fps: 30
  max_latency_ms: 50

gpu:
  device_id: 0

  cuda:
    block_size: [16, 16]
    threads_per_block: 256
    shared_memory_kb: 48
    registers_per_thread: 32

  streams:
    num_streams: 10
    enable_async: true
    priority: 0

  memory:
    use_pinned: true
    pinned_pool_mb: 512
    mempool_mb: 2048

  power:
    persistence_mode: true
    power_limit_w: 450
    clock_lock_mhz: 2100

cpu:
  threads:
    num_threads: 16
    affinity: "compact"
    schedule: "dynamic"

  numa:
    enabled: false
    preferred_node: 0

memory:
  pool:
    enabled: true
    buffer_size_mb: 64
    num_buffers: 128

  ring_buffer:
    capacity: 64
    frame_shape: [4320, 7680, 1]

network:
  transport: "shared_memory"

  compression:
    algorithm: "lz4"
    level: 1
    threshold_kb: 64

  batching:
    enabled: true
    max_batch_size: 100
    max_delay_ms: 5

pipeline:
  frame_dropping:
    enabled: true
    target_latency_ms: 50
    max_drop_rate: 0.3

  load_balancing:
    strategy: "dynamic"
    rebalance_threshold: 0.2

adaptive:
  quality:
    enabled: true
    min_scale: 0.5
    target_fps: 30

  resources:
    enabled: true
    min_streams: 4
    max_streams: 16

monitoring:
  enabled: true
  update_rate_hz: 10
  history_size: 300

profiling:
  enabled: false  # Only for development
  output_dir: "/tmp/profiling"
```

---

## Troubleshooting

### Common Issues

#### 1. Low GPU Utilization (<60%)

**Symptoms**:
- GPU utilization <60%
- Low throughput

**Solutions**:
1. Increase number of streams
2. Use larger batch sizes
3. Check for CPU bottlenecks
4. Reduce CPU-GPU synchronization

**Debug**:
```bash
# Profile with Nsight Systems
nsys profile -o trace.qdrep python main.py

# Check for gaps between kernels
# Look for excessive cudaDeviceSynchronize()
```

#### 2. High Latency (>80ms)

**Symptoms**:
- End-to-end latency >80ms
- Frame drops

**Solutions**:
1. Enable adaptive quality
2. Increase stream priority
3. Reduce batch sizes
4. Check network latency

**Debug**:
```python
# Add timing instrumentation
start = time.perf_counter()
process_frame(frame)
latency = (time.perf_counter() - start) * 1000
print(f"Latency: {latency:.2f}ms")
```

#### 3. Memory Errors

**Symptoms**:
- CUDA out of memory errors
- System OOM killer

**Solutions**:
1. Reduce resolution scale
2. Decrease batch size
3. Enable memory pooling
4. Clear GPU memory caches

**Debug**:
```python
import nvidia_smi
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print(f"Used: {info.used / 1e9:.2f} GB")
print(f"Free: {info.free / 1e9:.2f} GB")
```

#### 4. Network Bottleneck

**Symptoms**:
- Network latency >15ms
- High packet loss

**Solutions**:
1. Enable jumbo frames (MTU 9000)
2. Use RDMA if available
3. Increase network buffers
4. Check for network congestion

**Debug**:
```bash
# Check network stats
iperf3 -c server_ip -t 30

# Monitor interface
ifstat -i eth0 1

# Check drops
netstat -s | grep -i drop
```

---

## Performance Checklist

### Pre-Deployment

- [ ] GPU persistence mode enabled
- [ ] GPU clocks locked to maximum
- [ ] Pinned memory enabled
- [ ] CUDA streams configured
- [ ] Network buffers increased
- [ ] TCP optimization applied
- [ ] System monitoring enabled
- [ ] Baseline benchmarks established

### Optimization Priority

1. **Critical** (Do first):
   - Enable CUDA streams
   - Use pinned memory
   - Optimize kernel block sizes
   - Enable network optimizations

2. **High Impact**:
   - Implement kernel fusion
   - Enable memory pooling
   - Configure load balancing
   - Implement adaptive quality

3. **Medium Impact**:
   - Tune occupancy
   - Optimize cache usage
   - Enable batching
   - Configure frame dropping

4. **Low Impact** (Fine-tuning):
   - SIMD vectorization
   - NUMA optimization
   - Auto-tuning
   - Advanced profiling

---

## References

### Documentation

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Linux Network Tuning](https://www.kernel.org/doc/Documentation/networking/ip-sysctl.txt)

### Tools

- NVIDIA Nsight Systems: System-wide profiling
- NVIDIA Nsight Compute: Kernel-level profiling
- NVIDIA Visual Profiler: Legacy profiler
- perf: Linux performance analysis
- iperf3: Network throughput testing

### Support

- GitHub Issues: [github.com/yourrepo/issues](https://github.com/yourrepo/issues)
- Documentation: [docs.example.com](https://docs.example.com)
- Email: support@example.com

---

## Appendix

### A. Hardware Requirements

**Minimum**:
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- CPU: 16-core (32 threads)
- RAM: 32GB DDR4
- Network: 10 GbE
- Storage: NVMe SSD

**Recommended**:
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: AMD Threadripper / Intel Xeon (32+ cores)
- RAM: 128GB DDR5
- Network: 100 GbE or InfiniBand
- Storage: NVMe RAID

### B. Software Requirements

- CUDA: 12.0+
- cuDNN: 8.9+
- Python: 3.10+
- PyTorch: 2.0+ (optional)
- Linux Kernel: 5.15+ (for network optimizations)

### C. Benchmark Results

See `/docs/PERFORMANCE_REPORT.md` for detailed before/after metrics.

---

**Last Updated:** 2025-11-13
**Authors:** Performance Engineering Team
**Version:** 2.0.0
