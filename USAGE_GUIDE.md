# 8K Motion Tracking System - Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Running the System](#running-the-system)
4. [Application Architecture](#application-architecture)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Pixeltovoxelprojector.git
cd Pixeltovoxelprojector

# Install dependencies
pip install -r src/requirements.txt

# Install system requirements (Ubuntu/Debian)
sudo apt-get install libyaml-dev

# Build C++ extensions (optional, for best performance)
cd src
python setup_motion_extractor.py build_ext --inplace
```

### Basic Usage

```bash
# Run with default configuration
cd src
python main.py

# Run with custom configuration
python main.py --config path/to/config.yaml

# Run in verbose mode
python main.py --verbose

# Run in simulation mode (no hardware required)
python main.py --simulate

# Validate configuration
python main.py --validate-config
```

## Configuration

The system is configured via a YAML file (`config/system_config.yaml`).

### Key Configuration Sections

#### 1. System Settings
```yaml
system:
  name: "8K Motion Tracking System"
  version: "1.0.0"
  environment: "production"  # production, development, testing
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
```

#### 2. Camera Configuration
```yaml
cameras:
  num_pairs: 10
  pairs:
    - pair_id: 0
      mono:
        camera_id: 0
        ip_address: "192.168.1.100"
        width: 7680
        height: 4320
        frame_rate: 30.0
      thermal:
        camera_id: 1
        ip_address: "192.168.1.101"
        width: 7680
        height: 4320
        frame_rate: 30.0
      baseline_m: 0.5
      position: [0.0, 0.0, 10.0]
```

#### 3. Voxel Grid Configuration
```yaml
voxel_grid:
  center: [0.0, 0.0, 500.0]  # meters
  size: [5000.0, 5000.0, 2000.0]  # meters
  base_resolution: 1.0  # meters per voxel
  enable_lod: true
  max_memory_mb: 500
```

#### 4. Detection and Tracking
```yaml
detection:
  motion_threshold: 0.2
  max_tracks: 200
  detection_confidence: 0.5
  enable_kalman_filter: true
```

#### 5. Performance Settings
```yaml
performance:
  num_processing_threads: 8
  enable_gpu: true
  enable_memory_pooling: true
  process_priority: "high"
```

## Running the System

### Standard Operation

```bash
# 1. Ensure cameras are connected and powered
# 2. Verify network connectivity (ping camera IPs)
# 3. Run the system
cd src
python main.py --config config/system_config.yaml
```

### Simulation Mode

For development and testing without hardware:

```bash
python main.py --simulate
```

This mode:
- Generates synthetic camera data
- Simulates all processing stages
- Useful for testing configurations
- No camera hardware required

### With Monitoring

```bash
# Terminal 1: Run main system
python main.py --verbose

# Terminal 2: Monitor system metrics
python monitoring/system_monitor.py

# Terminal 3: View logs
tail -f logs/motion_tracking.log
```

## Application Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Application                        │
│                        (main.py)                             │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴──────────────┐
         │                               │
┌────────▼──────────┐         ┌─────────▼──────────┐
│  Configuration    │         │  Pipeline          │
│  Loader           │         │  Coordinator       │
└────────┬──────────┘         └─────────┬──────────┘
         │                               │
         │        ┌──────────────────────┴─────────────────┐
         │        │                                         │
         │  ┌─────▼──────┐  ┌──────────┐  ┌─────────────┐ │
         └─►│  Camera    │  │  Fusion  │  │  Voxel      │ │
            │  Manager   │  │  Manager │  │  Manager    │ │
            └─────┬──────┘  └─────┬────┘  └──────┬──────┘ │
                  │               │               │        │
            ┌─────▼───────────────▼───────────────▼──────┐ │
            │        Processing Pipeline                  │ │
            │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
            │  │ Motion   │→ │ Tracking │→ │ Voxel    │ │ │
            │  │ Extract  │  │          │  │ Update   │ │ │
            │  └──────────┘  └──────────┘  └──────────┘ │ │
            └────────────────────┬────────────────────────┘ │
                                 │                           │
                        ┌────────▼──────────┐               │
                        │  System Monitor   │◄──────────────┘
                        └───────────────────┘
```

### Data Flow

1. **Camera Acquisition**
   - Cameras capture 8K frames at 30 FPS
   - Hardware-triggered synchronization
   - Frame buffering and management

2. **Motion Extraction**
   - C++ accelerated motion detection
   - Thermal-mono fusion
   - Coordinate extraction

3. **Tracking**
   - Multi-target Kalman filter tracking
   - Track management (200+ simultaneous)
   - Occlusion handling

4. **Voxel Updates**
   - 3D position mapping
   - Multi-resolution LOD grid
   - Memory-efficient updates

5. **Streaming**
   - Real-time coordinate streaming
   - Network protocol (UDP/TCP/Shared Memory)
   - Low-latency delivery

## Examples

### Example 1: Basic System Startup

```python
#!/usr/bin/env python3
from main import MotionTrackingSystem

# Create system
system = MotionTrackingSystem(
    config_file='config/system_config.yaml',
    verbose=True,
    simulate=False
)

# Load configuration
system.load_configuration()

# Initialize components
system.initialize_components()

# Start system
system.start()

# Run for 60 seconds
import time
time.sleep(60)

# Stop system
system.stop()
```

### Example 2: Custom Coordinate Callback

```python
from main import MotionTrackingSystem

def coordinate_callback(result):
    """Custom callback for coordinate streaming"""
    print(f"Frame {result.frame_number}: {len(result.confirmed_tracks)} tracks")
    for track in result.confirmed_tracks:
        print(f"  Track {track['track_id']}: ({track['x']:.1f}, {track['y']:.1f})")

# Initialize system
system = MotionTrackingSystem('config/system_config.yaml')
system.load_configuration()
system.initialize_components()

# Register callback
system.pipeline.register_coordinate_callback(coordinate_callback)

# Start and run
system.start()
system.run()
```

### Example 3: Monitor System Status

```python
import time
from main import MotionTrackingSystem

system = MotionTrackingSystem('config/system_config.yaml')
system.load_configuration()
system.initialize_components()
system.start()

try:
    while True:
        # Get system status
        if system.coordinator:
            status = system.coordinator.get_system_status()
            print(f"Health: {status['overall_health']}")
            print(f"Uptime: {status['uptime_seconds']:.1f}s")

        # Get pipeline metrics
        if system.pipeline:
            metrics = system.pipeline.get_metrics()
            print(f"FPS: {metrics['throughput_fps']:.1f}")
            print(f"Latency: {metrics['avg_latency_ms']:.1f}ms")

        time.sleep(5)

except KeyboardInterrupt:
    system.stop()
```

### Example 4: Configuration Validation

```python
from main import MotionTrackingSystem

# Validate configuration without running
system = MotionTrackingSystem('config/system_config.yaml')

if system.load_configuration():
    print("✓ Configuration is valid")

    # Print summary
    print(f"Cameras: {system.config['cameras']['num_pairs']} pairs")
    print(f"Voxel grid: {system.config['voxel_grid']['size']}")
    print(f"Max tracks: {system.config['detection']['max_tracks']}")
else:
    print("✗ Configuration is invalid")
```

## Command-Line Interface

### Available Options

```bash
python main.py [OPTIONS]

Options:
  --config PATH         Path to configuration file
                        (default: config/system_config.yaml)

  -v, --verbose         Enable verbose logging

  --simulate            Run in simulation mode (no hardware)

  --validate-config     Validate configuration and exit

  --version             Show version and exit

  -h, --help            Show help message and exit
```

### Usage Examples

```bash
# Basic run with default config
python main.py

# Run with custom config
python main.py --config my_config.yaml

# Verbose logging
python main.py --verbose

# Simulation mode for testing
python main.py --simulate

# Just validate the configuration
python main.py --validate-config
```

## Troubleshooting

### Common Issues

#### 1. Configuration File Not Found
```
Error: Configuration file not found: config/system_config.yaml
```
**Solution:** Ensure the config file exists or specify the correct path:
```bash
python main.py --config /path/to/config.yaml
```

#### 2. Camera Connection Failed
```
Error: Failed to connect camera 0 at 192.168.1.100
```
**Solution:**
- Verify camera is powered on
- Check network connectivity: `ping 192.168.1.100`
- Ensure IP address is correct in config
- Check firewall settings

#### 3. Insufficient Memory
```
Error: Failed to allocate voxel grid: out of memory
```
**Solution:**
- Reduce `voxel_grid.max_memory_mb` in config
- Enable compression: `voxel_grid.enable_compression: true`
- Use higher resolution: `voxel_grid.base_resolution: 2.0`

#### 4. Low FPS Performance
```
Warning: FPS below target: 15.2 / 30.0
```
**Solution:**
- Enable GPU: `performance.enable_gpu: true`
- Increase threads: `performance.num_processing_threads: 16`
- Reduce resolution in cameras section
- Enable hardware acceleration

#### 5. High CPU Usage
```
Warning: CPU utilization critical: 95%
```
**Solution:**
- Reduce number of processing threads
- Enable GPU acceleration
- Optimize detection thresholds
- Check for runaway processes

### Debug Mode

Enable debug logging for detailed information:

```yaml
# In config file
logging:
  levels:
    root: "DEBUG"
    camera: "DEBUG"
    pipeline: "DEBUG"
```

Or use verbose mode:
```bash
python main.py --verbose
```

### Performance Profiling

```bash
# Enable profiling
python -m cProfile -o profile.stats main.py

# View results
python -m pstats profile.stats
```

### Log Files

Logs are written to:
- `logs/motion_tracking.log` - Main application log
- `logs/system_metrics.json` - Performance metrics
- `logs/profile.txt` - Profiling data (if enabled)

### Getting Help

1. Check logs: `tail -f logs/motion_tracking.log`
2. View system status via monitoring interface
3. Run configuration validation: `python main.py --validate-config`
4. Open an issue on GitHub with logs and configuration

## Advanced Usage

### Custom Pipeline Configuration

You can customize the processing pipeline:

```python
from pipeline import ProcessingPipeline, PipelineConfig

custom_config = PipelineConfig(
    target_fps=60.0,
    enable_fusion=True,
    enable_tracking=True,
    num_processing_threads=16
)

pipeline = ProcessingPipeline(
    config=custom_config,
    camera_manager=camera_mgr,
    voxel_manager=voxel_mgr,
    fusion_manager=fusion_mgr,
    tracker=tracker
)
```

### Network Streaming

Configure network streaming in the config file:

```yaml
network:
  protocol: "mtrtp"
  transport: "udp"  # udp, tcp, shared_memory
  udp:
    host: "0.0.0.0"
    port: 8888
  compression: "lz4"
  stream_coordinates: true
```

### Recording Sessions

Enable data recording:

```yaml
recording:
  enabled: true
  output_dir: "recordings"
  record_motion_data: true
  record_tracking_data: true
```

## Performance Tips

1. **Enable GPU Acceleration**
   ```yaml
   performance:
     enable_gpu: true
   ```

2. **Optimize Threading**
   - Set threads to 2x CPU cores for I/O bound
   - Set threads to 1x CPU cores for CPU bound

3. **Use Hardware Triggers**
   ```yaml
   cameras:
     trigger:
       enabled: true
       source: "external"
   ```

4. **Enable Memory Pooling**
   ```yaml
   performance:
     enable_memory_pooling: true
     preallocate_buffers: true
   ```

5. **Tune Network Settings**
   ```yaml
   cameras:
     network:
       packet_size: 9000  # Jumbo frames
       packet_delay: 1000
   ```

---

For more information, see:
- [README.md](README.md) - Project overview
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) - Performance analysis
