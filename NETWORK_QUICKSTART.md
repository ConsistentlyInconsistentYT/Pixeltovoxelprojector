# Network Infrastructure Quick Start Guide

## Installation

### 1. Install Dependencies

```bash
# Navigate to project directory
cd /home/user/Pixeltovoxelprojector

# Install core dependencies
pip install -r src/network/requirements.txt

# Optional: Install RDMA support (for InfiniBand)
# pip install pyverbs

# Optional: Install advanced shared memory
# pip install posix_ipc
```

### 2. Verify Installation

```bash
# Run simple test
python3 -c "from src.network import ClusterConfig, DataPipeline, DistributedProcessor; print('OK')"
```

---

## Quick Start: Single Node

### Basic Example

```python
from src.network import ClusterConfig, DataPipeline, DistributedProcessor
import numpy as np
import time

# 1. Initialize cluster (single node)
cluster = ClusterConfig()
cluster.start(is_master=True)
time.sleep(1)

# 2. Create data pipeline
pipeline = DataPipeline(
    buffer_capacity=32,
    frame_shape=(1080, 1920, 3),  # HD resolution
    enable_rdma=False,
    enable_shared_memory=True
)

# 3. Initialize processor
processor = DistributedProcessor(
    cluster_config=cluster,
    data_pipeline=pipeline,
    num_cameras=2
)

# 4. Register task handler
def my_task_handler(task):
    frame = task.input_data['frame']
    # Process frame here
    result = np.mean(frame)
    return {'average': result}

processor.register_task_handler('process_frame', my_task_handler)

# 5. Start processing
processor.start()
time.sleep(1)

# 6. Submit a frame
frame = np.random.rand(1080, 1920, 3).astype(np.float32)
from src.network import FrameMetadata

metadata = FrameMetadata(
    frame_id=0,
    camera_id=0,
    timestamp=time.time(),
    width=1920,
    height=1080,
    channels=3,
    dtype='float32',
    compressed=False,
    checksum='',
    sequence_number=0
)

task_id = processor.submit_camera_frame(0, frame, metadata)

# 7. Wait for result
result = processor.wait_for_task(task_id, timeout=5.0)
print(f"Result: {result}")

# 8. Cleanup
processor.stop()
cluster.stop()
pipeline.cleanup()
```

---

## Quick Start: Multi-Node Cluster

### On Each Node

**Master Node** (run first):
```python
from src.network import ClusterConfig
import time

cluster = ClusterConfig(
    discovery_port=9999,
    enable_rdma=True  # Set False if no InfiniBand
)

cluster.start(is_master=True)

# Keep running
try:
    while True:
        time.sleep(1)
        status = cluster.get_cluster_status()
        print(f"Nodes: {status['online_nodes']}, GPUs: {status['total_gpus']}")
except KeyboardInterrupt:
    cluster.stop()
```

**Worker Nodes** (run on other machines):
```python
from src.network import ClusterConfig
import time

cluster = ClusterConfig(
    discovery_port=9999,
    enable_rdma=True
)

cluster.start(is_master=False)

# Keep running
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    cluster.stop()
```

### Run Distributed Processing

On master node:
```python
from src.network import ClusterConfig, DataPipeline, DistributedProcessor
import time

# Initialize (master node)
cluster = ClusterConfig(enable_rdma=True)
cluster.start(is_master=True)
time.sleep(3)  # Wait for node discovery

# Create pipeline
pipeline = DataPipeline(
    buffer_capacity=64,
    frame_shape=(2160, 3840, 3),  # 8K
    enable_rdma=True,
    enable_shared_memory=True,
    shm_size_mb=2048
)

# Create processor
processor = DistributedProcessor(
    cluster_config=cluster,
    data_pipeline=pipeline,
    num_cameras=10,
    enable_fault_tolerance=True
)

# Register handler and start
def process_voxel_frame(task):
    # Your processing logic here
    return {'status': 'ok'}

processor.register_task_handler('process_frame', process_voxel_frame)
processor.start()
time.sleep(2)

# Allocate cameras
allocation = cluster.allocate_cameras(10)
print(f"Camera allocation: {allocation}")

# Get system health
health = processor.get_system_health()
print(f"System health: {health['status']}")
print(f"Active workers: {health['active_workers']}")

# Submit frames...
# (see full example in examples/distributed_processing_example.py)
```

---

## Running Examples

### Full Distributed Processing Demo

```bash
python3 examples/distributed_processing_example.py
```

**Output**:
- Cluster initialization
- Node discovery
- Camera allocation
- Task processing
- Performance statistics

### Network Benchmark

```bash
python3 examples/benchmark_network.py
```

**Tests**:
- Ring buffer latency
- Data pipeline throughput
- Task scheduling overhead
- End-to-end latency

---

## Configuration Options

### ClusterConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `discovery_port` | 9999 | UDP port for node discovery |
| `heartbeat_interval` | 1.0 | Seconds between heartbeats |
| `heartbeat_timeout` | 5.0 | Timeout before node offline |
| `enable_rdma` | True | Enable InfiniBand RDMA |

### DataPipeline

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_capacity` | 64 | Frames per ring buffer |
| `frame_shape` | (1080,1920,3) | Frame dimensions |
| `enable_rdma` | True | Use RDMA for transfers |
| `enable_shared_memory` | True | Use shared memory IPC |
| `shm_size_mb` | 1024 | Shared memory size (MB) |

### DistributedProcessor

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_cameras` | 10 | Number of camera pairs |
| `enable_fault_tolerance` | True | Auto failover on failure |

---

## Monitoring

### Get Real-Time Statistics

```python
# Cluster status
cluster_status = cluster.get_cluster_status()
print(f"Online nodes: {cluster_status['online_nodes']}")
print(f"Total GPUs: {cluster_status['total_gpus']}")

# Processing statistics
stats = processor.get_statistics()
print(f"Tasks completed: {stats['tasks_completed']}")
print(f"Success rate: {stats['success_rate']*100:.1f}%")
print(f"Avg execution time: {stats['avg_execution_time']*1000:.2f}ms")

# Pipeline statistics
pipeline_stats = stats['pipeline']
print(f"Frames processed: {pipeline_stats['frames_processed']}")
print(f"Throughput: {pipeline_stats['bytes_transferred']/1e9:.2f} GB")

# System health
health = processor.get_system_health()
print(f"Status: {health['status']}")
print(f"Avg latency: {health['avg_latency_ms']:.2f}ms")
```

---

## Network Configuration

### InfiniBand Setup

1. **Verify InfiniBand devices**:
```bash
ibstat
ibv_devices
```

2. **Check connectivity**:
```bash
# On node 1
ib_send_lat

# On node 2
ib_send_lat <node1_ip>
```

3. **Expected latency**: <1 μs

### 10GbE Setup

1. **Enable jumbo frames**:
```bash
sudo ip link set eth0 mtu 9000
```

2. **Verify**:
```bash
ip link show eth0 | grep mtu
```

3. **Test bandwidth**:
```bash
# On receiver
iperf3 -s

# On sender
iperf3 -c <receiver_ip> -t 10
```

4. **Expected throughput**: 9+ Gbps

---

## Troubleshooting

### Issue: Nodes not discovering each other

**Solution**:
```bash
# Check firewall
sudo ufw allow 9999/udp

# Check network connectivity
ping <other_node_ip>

# Verify broadcast is enabled
sudo sysctl net.ipv4.icmp_echo_ignore_broadcasts=0
```

### Issue: RDMA not available

**Solution**:
```python
# Disable RDMA
cluster = ClusterConfig(enable_rdma=False)
pipeline = DataPipeline(enable_rdma=False)
```

### Issue: GPU not detected

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Install pynvml
pip install pynvml

# Verify CUDA
python3 -c "import pynvml; pynvml.nvmlInit(); print('OK')"
```

### Issue: High latency (>5ms)

**Solutions**:
- Enable jumbo frames (MTU 9000)
- Check network utilization: `iftop -i eth0`
- Optimize topology: `cluster.optimize_network_topology()`
- Reduce CPU usage on nodes

### Issue: Tasks failing

**Solutions**:
```python
# Check error logs
stats = processor.get_statistics()
print(f"Failed tasks: {stats['tasks_failed']}")

# Review specific task
task = processor.task_registry.get(task_id)
if task:
    print(f"Error: {task.error}")

# Increase timeout
result = processor.wait_for_task(task_id, timeout=60.0)
```

---

## Performance Tuning

### For Maximum Throughput

```python
# Larger buffers
pipeline = DataPipeline(
    buffer_capacity=128,  # Increased from 64
    frame_shape=(2160, 3840, 3)
)

# More workers per GPU
# (automatically scales with available GPUs)
```

### For Minimum Latency

```python
# Smaller buffers (reduces queueing delay)
pipeline = DataPipeline(
    buffer_capacity=16,
    frame_shape=(2160, 3840, 3)
)

# Enable RDMA
cluster = ClusterConfig(enable_rdma=True)
pipeline = DataPipeline(enable_rdma=True)

# High priority tasks
task.priority = 10  # Higher = processed first
```

### For Reliability

```python
# Enable all fault tolerance features
processor = DistributedProcessor(
    cluster_config=cluster,
    data_pipeline=pipeline,
    num_cameras=10,
    enable_fault_tolerance=True  # Must be True
)

# Increase retries
task.max_retries = 5  # Default is 3

# Shorter heartbeat interval
cluster = ClusterConfig(
    heartbeat_interval=0.5,  # More frequent checks
    heartbeat_timeout=3.0     # Faster failure detection
)
```

---

## Best Practices

1. **Always start master node first**, wait 2-3 seconds before starting workers
2. **Enable RDMA for 10+ cameras** to achieve target latency
3. **Monitor system health** using `get_system_health()` every few seconds
4. **Set appropriate timeouts** based on expected task duration
5. **Test failover** before production deployment
6. **Log all events** for debugging and analysis
7. **Profile regularly** using built-in statistics
8. **Reserve compute headroom** (20-30%) for load spikes

---

## Next Steps

1. Read full architecture documentation: `DISTRIBUTED_ARCHITECTURE.md`
2. Review example code: `examples/distributed_processing_example.py`
3. Run benchmarks: `examples/benchmark_network.py`
4. Customize task handlers for your workload
5. Deploy to production cluster
6. Set up monitoring and alerting

---

## Additional Resources

- **Architecture Details**: `/home/user/Pixeltovoxelprojector/DISTRIBUTED_ARCHITECTURE.md`
- **Example Code**: `/home/user/Pixeltovoxelprojector/examples/`
- **API Documentation**: Inline code comments in `/home/user/Pixeltovoxelprojector/src/network/`

---

**Need Help?**
- Check inline code documentation
- Review examples directory
- See troubleshooting section above
- Examine debug logs (set `logging.level=DEBUG`)
