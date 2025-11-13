# MTRTP Quick Reference Guide

## Installation

### One-Command Setup
```bash
./build.sh --all  # Install deps, build, test, and install
```

### Manual Setup
```bash
# C++ only
./build.sh --deps --release --install

# Python only
pip install -r requirements.txt

# Both
./build.sh --deps --python --release --install
```

## Common Usage Patterns

### 1. Shared Memory (Local, Ultra-Fast)

#### C++ Publisher
```cpp
#include "stream_manager.cpp"
using namespace mtrtp;

// Setup
StreamManager::Config cfg;
cfg.transport = StreamManager::TransportType::SHARED_MEMORY;
cfg.shared_mem_name = "/mtrtp_motion";

StreamManager mgr(cfg);
mgr.initialize(true);

// Send
MotionCoordinateMessage msg;
msg.set_message_id(seq++);
msg.mutable_single_point()->set_point_id(1);
msg.mutable_single_point()->mutable_position()->set_x(1.0f);
mgr.send_message(msg);
```

#### Python Subscriber
```python
from subscriber import *

sub = Subscriber(SubscriberConfig(
    transport=TransportType.SHARED_MEMORY,
    shared_mem_name="/mtrtp_motion"
))

@sub.on_message(MessageType.MOTION_COORDINATE)
def on_motion(msg, ts):
    print(f"Motion at {ts}")

sub.start()
```

### 2. UDP (Network, Low Latency)

#### C++ Publisher
```cpp
StreamManager::Config cfg;
cfg.transport = StreamManager::TransportType::UDP;
cfg.remote_ip = "192.168.1.100";
cfg.remote_port = 8888;

StreamManager mgr(cfg);
mgr.initialize(true);
```

#### Python Subscriber
```python
sub = Subscriber(SubscriberConfig(
    transport=TransportType.UDP,
    udp_port=8888
))
```

### 3. TCP (Network, Reliable)

#### C++ Publisher
```cpp
StreamManager::Config cfg;
cfg.transport = StreamManager::TransportType::TCP;
cfg.remote_ip = "192.168.1.100";
cfg.remote_port = 8889;

StreamManager mgr(cfg);
mgr.initialize(true);
```

#### Python Subscriber
```python
sub = Subscriber(SubscriberConfig(
    transport=TransportType.TCP,
    host="0.0.0.0",
    tcp_port=8889
))
```

## Message Types Quick Reference

### Motion Coordinate
```cpp
// C++
MotionCoordinateMessage msg;
msg.set_message_id(1);
msg.set_protocol_version(1);

auto* pt = msg.mutable_single_point();
pt->set_point_id(0);
pt->mutable_position()->set_x(1.0f);
pt->mutable_position()->set_y(2.0f);
pt->mutable_position()->set_z(3.0f);
pt->mutable_velocity()->set_x(0.1f);
pt->set_confidence(0.95f);
```

```python
# Python callback
@sub.on_message(MessageType.MOTION_COORDINATE)
def handle(msg, ts):
    x = msg['position'][0]
    y = msg['position'][1]
    z = msg['position'][2]
```

### Camera Pose
```cpp
// C++
CameraPoseMessage msg;
msg.set_camera_id(0);

auto* pos = msg.mutable_extrinsics()->mutable_position();
pos->set_x(0.0f);
pos->set_y(0.0f);
pos->set_z(1.0f);

auto* quat = msg.mutable_extrinsics()->mutable_orientation();
quat->set_w(1.0f);
quat->set_x(0.0f);
quat->set_y(0.0f);
quat->set_z(0.0f);
```

### Target Tracking
```cpp
// C++
TargetTrackingMessage msg;
auto* target = msg.add_targets();

target->set_target_id(1);
target->set_target_name("object_1");
target->set_state(TrackingState::TRACKING_ACTIVE);
target->mutable_position()->set_x(1.0f);
target->set_tracking_confidence(0.98f);
```

### System Status
```cpp
// C++
SystemStatusMessage msg;
msg.set_system_id("tracker_01");
msg.set_overall_health(HealthStatus::HEALTH_OK);

auto* comp = msg.add_components();
comp->set_component_id(1);
comp->set_component_name("camera");
comp->set_health(HealthStatus::HEALTH_OK);
comp->set_fps(30.0f);
```

## Performance Tuning

### Maximize Throughput
```cpp
// Use shared memory
cfg.transport = TransportType::SHARED_MEMORY;

// Disable compression
cfg.compression = MessageCompressor::Type::NONE;

// Large buffers
cfg.buffer_size = 1024 * 1024;  // 1MB
```

### Minimize Latency
```cpp
// Shared memory
cfg.transport = TransportType::SHARED_MEMORY;

// No compression
cfg.compression = MessageCompressor::Type::NONE;

// Small messages
// Keep message size < 1KB
```

### Network Optimization
```cpp
// Enable compression for >1KB messages
cfg.compression = MessageCompressor::Type::LZ4;

// Use UDP for low latency
cfg.transport = TransportType::UDP;

// Or TCP for reliability
cfg.transport = TransportType::TCP;
```

## Debugging

### Enable Verbose Logging (C++)
```cpp
#define MTRTP_DEBUG 1
// Rebuild
```

### Enable Verbose Logging (Python)
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Shared Memory
```bash
# List shared memory objects
ls -la /dev/shm/

# Remove stale objects
sudo rm /dev/shm/mtrtp_*
```

### Monitor Performance
```cpp
// C++
mgr.print_statistics();
```

```python
# Python
sub.get_statistics().print_stats()
```

### Network Debugging
```bash
# Check UDP packets
sudo tcpdump -i any -n udp port 8888

# Check TCP connections
sudo netstat -anp | grep 8889

# Increase buffer sizes
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

## Common Callbacks

### Motion Processing
```python
@sub.on_message(MessageType.MOTION_COORDINATE)
def process_motion(msg, ts):
    # Extract data
    point_id = msg.get('point_id')
    pos = msg.get('position', [0, 0, 0])
    vel = msg.get('velocity', [0, 0, 0])
    conf = msg.get('confidence', 1.0)

    # Filter by confidence
    if conf < 0.8:
        return

    # Process
    print(f"Point {point_id}: pos={pos}, vel={vel}")
```

### Camera Tracking
```python
@sub.on_message(MessageType.CAMERA_POSE)
def process_camera(msg, ts):
    cam_id = msg.get('camera_id')
    pos = msg.get('position')
    orient = msg.get('orientation')

    # Update camera model
    update_camera(cam_id, pos, orient)
```

### All Messages
```python
@sub.on_message(MessageType.ALL)
def process_all(msg, ts):
    # Handle any message type
    msg_type = msg.get('type')
    print(f"Received {msg_type} at {ts}")
```

## Configuration Options

### Stream Manager (C++)
```cpp
StreamManager::Config cfg;

// Transport
cfg.transport = TransportType::SHARED_MEMORY;  // or UDP, TCP
cfg.shared_mem_name = "/mtrtp_default";
cfg.udp_port = 8888;
cfg.tcp_port = 8889;
cfg.remote_ip = "127.0.0.1";
cfg.remote_port = 8888;

// Performance
cfg.buffer_size = 65536;  // 64KB
cfg.compression = MessageCompressor::Type::LZ4;

// Monitoring
cfg.enable_stats = true;
```

### Subscriber (Python)
```python
cfg = SubscriberConfig(
    # Transport
    transport=TransportType.SHARED_MEMORY,
    host="127.0.0.1",
    udp_port=8888,
    tcp_port=8889,
    shared_mem_name="/mtrtp_default",

    # Connection
    auto_reconnect=True,
    reconnect_delay=1.0,
    max_reconnect_delay=30.0,
    connection_timeout=5.0,

    # Buffers
    buffer_size=65536,
    message_queue_size=1000,

    # Compression
    compression=CompressionType.LZ4,

    # Performance
    enable_stats=True,
    stats_interval=10.0,
    num_worker_threads=2,
)
```

## Testing & Benchmarking

### Run Unit Tests
```bash
# C++
./build.sh --tests

# Python
pytest tests/
```

### Run Benchmarks
```bash
cd examples

# All benchmarks
python benchmark.py --test all

# Specific tests
python benchmark.py --test latency --duration 30
python benchmark.py --test throughput --transport udp
python benchmark.py --test compression
python benchmark.py --test scalability --max-subscribers 1000

# Save results
python benchmark.py --test all --output results.json
```

### Example Programs
```bash
cd examples

# Publisher
python motion_publisher_example.py \
    --transport shared_memory \
    --rate 1000 \
    --num-points 10

# Subscriber
python motion_subscriber_example.py \
    --transport shared_memory \
    --filter "confidence > 0.9"
```

## Protocol Version Compatibility

### Check Version
```cpp
// C++
msg.set_protocol_version(PROTOCOL_VERSION);  // Current: 1
```

### Handle Multiple Versions
```cpp
if (msg.protocol_version() == 1) {
    // Handle v1
} else {
    // Handle other versions or error
}
```

## Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| -1 | Connection failed | Check network/shared mem |
| -2 | Invalid message | Check protocol version |
| -3 | Checksum mismatch | Data corruption, retry |
| -4 | Buffer full | Increase buffer size |
| -5 | Timeout | Increase timeout or check connection |

## Performance Expectations

| Transport | Latency | Throughput | Use Case |
|-----------|---------|------------|----------|
| Shared Mem | <1 μs | 100k+ msg/s | Local, ultra-fast |
| UDP | 50-200 μs | 50k+ msg/s | Network, low latency |
| TCP | 100-500 μs | 25k+ msg/s | Network, reliable |

| Message Size | Compression | Time | Ratio |
|--------------|-------------|------|-------|
| 100 bytes | LZ4 | 2 μs | 1.2x |
| 1 KB | LZ4 | 8 μs | 2.5x |
| 10 KB | LZ4 | 25 μs | 4x |
| 64 KB | LZ4 | 80 μs | 5x |

## Troubleshooting Checklist

- [ ] Dependencies installed? (`./build.sh --deps`)
- [ ] Built successfully? (`./build.sh`)
- [ ] Shared memory cleaned? (`rm /dev/shm/mtrtp_*`)
- [ ] Correct permissions? (`ls -la /dev/shm/`)
- [ ] Firewall allows UDP/TCP? (`sudo ufw status`)
- [ ] Buffer sizes adequate? (Check sysctl settings)
- [ ] Protocol versions match? (Check message headers)
- [ ] Network reachable? (`ping`, `telnet`)

## Useful Commands

```bash
# Build and install
./build.sh --all

# Clean and rebuild
./build.sh --clean --release

# Run tests
./build.sh --tests

# Check shared memory
ls -la /dev/shm/

# Monitor network
sudo tcpdump -i any -n port 8888

# System tuning
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728

# Check processes
ps aux | grep mtrtp
```

---

**For detailed information, see:** `README.md` and `PROTOCOL_SPEC.md`
