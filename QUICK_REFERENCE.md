# 8K Motion Tracking System - Quick Reference

## Quick Start

```bash
# 1. Install dependencies
pip install -r src/requirements.txt

# 2. Run simulation demo (no hardware needed)
python quick_start.py

# 3. Run full system
cd src
python main.py --config config/system_config.yaml
```

## Command Line

```bash
python main.py                      # Run with defaults
python main.py --verbose            # Verbose logging
python main.py --simulate           # Simulation mode
python main.py --validate-config    # Validate only
python main.py --config my.yaml     # Custom config
```

## Configuration Sections

| Section | Purpose |
|---------|---------|
| `system` | Basic settings, log levels |
| `cameras` | 10 pairs (20 cameras) config |
| `voxel_grid` | 3D space (5km³) settings |
| `detection` | Motion thresholds, tracking |
| `fusion` | Thermal-mono fusion |
| `network` | Streaming configuration |
| `performance` | Threading, GPU settings |
| `monitoring` | Health checks, metrics |

## Key Parameters

```yaml
# Essential settings to tune
cameras.num_pairs: 10              # Number of camera pairs
cameras.pairs[0].mono.frame_rate: 30.0  # Target FPS
voxel_grid.base_resolution: 1.0    # Meters per voxel
detection.max_tracks: 200          # Max simultaneous tracks
performance.num_processing_threads: 8  # Processing threads
monitoring.update_rate_hz: 10.0    # Monitor rate
```

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Total Latency | <100ms | ~28ms |
| FPS | 30 | 30 |
| Cameras | 20 | 20 |
| Max Tracks | 200 | 200+ |
| Memory | <4GB | ~3GB |
| CPU Usage | <90% | 70-85% |

## Architecture

```
main.py
  ├─ Configuration (YAML)
  ├─ Pipeline Coordinator
  │   ├─ Camera Manager (20 cameras)
  │   ├─ Fusion Manager (10 pairs)
  │   ├─ Voxel Manager (5km³)
  │   ├─ Tracker (200 tracks)
  │   └─ System Monitor (10Hz)
  └─ Processing Pipeline
      ├─ Frame Acquisition
      ├─ Motion Extraction
      ├─ Fusion Processing
      ├─ Multi-target Tracking
      └─ Coordinate Streaming
```

## Python API

```python
from main import MotionTrackingSystem

# Create system
system = MotionTrackingSystem(
    config_file='config/system_config.yaml',
    verbose=True,
    simulate=False
)

# Initialize and start
system.load_configuration()
system.initialize_components()
system.start()

# Register callback
def callback(result):
    print(f"Tracks: {len(result.confirmed_tracks)}")

system.pipeline.register_coordinate_callback(callback)

# Run
system.run()

# Stop
system.stop()
```

## Component States

```
UNINITIALIZED → INITIALIZING → READY → RUNNING
                                   ↓
                              STOPPING → STOPPED
                                   ↓
                                ERROR → RECOVERING
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Config not found | `--config /path/to/config.yaml` |
| Camera connection failed | Check IP, ping camera |
| Out of memory | Reduce `voxel_grid.max_memory_mb` |
| Low FPS | Enable GPU, increase threads |
| High CPU | Reduce threads, enable GPU |

## Log Files

```
logs/motion_tracking.log       # Main log
logs/system_metrics.json       # Performance metrics
logs/profile.txt              # Profiling (if enabled)
```

## Monitoring

```python
# Get system status
status = system.coordinator.get_system_status()
print(status['overall_health'])

# Get pipeline metrics
metrics = system.pipeline.get_metrics()
print(f"FPS: {metrics['throughput_fps']}")
print(f"Latency: {metrics['avg_latency_ms']}ms")

# Get monitor summary
summary = system.monitor.get_summary()
print(f"CPU: {summary['cpu_utilization']:.1f}%")
print(f"Cameras: {summary['cameras_online']}/{summary['cameras_total']}")
```

## File Locations

```
src/
├── main.py                          # Main application
├── config/system_config.yaml        # Configuration
├── pipeline/
│   ├── processing_pipeline.py       # Pipeline
│   └── pipeline_coordinator.py      # Coordinator
├── camera/camera_manager.py         # Cameras
├── voxel/grid_manager.py            # Voxel grid
├── detection/tracker.py             # Tracking
├── fusion/fusion_manager.py         # Fusion
└── monitoring/system_monitor.py     # Monitoring
```

## Documentation

- `USAGE_GUIDE.md` - Complete usage guide
- `APPLICATION_ARCHITECTURE.md` - Technical architecture
- `FRAMEWORK_SUMMARY.md` - Implementation summary
- `QUICK_REFERENCE.md` - This document

## Support

Check logs: `tail -f logs/motion_tracking.log`
Validate config: `python main.py --validate-config`
Run tests: `pytest src/`
Simulation: `python main.py --simulate`

---
**Version:** 1.0.0 | **Updated:** 2025-11-13
