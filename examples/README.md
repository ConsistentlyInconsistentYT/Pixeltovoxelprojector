# 8K Motion Tracking System - Example Applications

This directory contains comprehensive example applications demonstrating the capabilities of the 8K motion tracking system.

## Overview

The examples progress from simple to complex, showcasing different aspects of the system:

1. **basic_tracking.py** - Simple single-pair tracking (beginner)
2. **multi_camera_demo.py** - Full 10-camera system (advanced)
3. **drone_tracking_sim.py** - 200 drone simulation (testing)
4. **streaming_client.py** - Real-time data streaming (integration)
5. **calibration_tool.py** - Camera calibration (setup)

## Requirements

### Common Requirements
- Python 3.8 or higher
- NumPy
- System modules from `../src`

### Optional Requirements
- Matplotlib (for visualization)
- OpenCV (cv2) - for actual calibration with real images
- LZ4 - for compressed streaming

### Installation
```bash
# Install from project root
pip install -r requirements.txt

# Or install minimal requirements
pip install numpy
```

---

## 1. Basic Tracking Example

**File:** `basic_tracking.py`

Simple demonstration of single camera pair tracking with minimal configuration.

### Features
- Single mono-thermal camera pair
- Basic object detection and tracking
- Voxel visualization
- Real-time motion coordinate display
- Minimal setup required

### Usage

```bash
# Basic usage (30 seconds, camera pair 0)
python basic_tracking.py

# Specify duration
python basic_tracking.py --duration 60

# Different camera pair
python basic_tracking.py --camera-id 1 --duration 120

# With visualization
python basic_tracking.py --visualize

# Save tracking output
python basic_tracking.py --save-output tracking_data.json
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--camera-id ID` | Camera pair ID | 0 |
| `--duration SEC` | Run duration in seconds | 30 |
| `--visualize` | Enable 3D visualization | False |
| `--save-output FILE` | Save tracking data to file | None |

### Output Example

```
==============================================================
BASIC TRACKING SYSTEM - Frame 450
==============================================================
Uptime:           15.0s
FPS:              30.1
Active Tracks:    8
Confirmed Tracks: 6
Detections:       4
Latency:          12.45 ms
--------------------------------------------------------------
Active Tracks:
  Track   0: pos=(  125.3,  -45.2) vel=( -2.1,   3.5) conf=0.89
  Track   1: pos=( -234.1,   89.7) vel=(  4.2,  -1.8) conf=0.92
  ...
--------------------------------------------------------------
Camera Status:    2/2 streaming
Avg Temperature:  47.3°C
==============================================================
```

---

## 2. Multi-Camera Demo

**File:** `multi_camera_demo.py`

Demonstrates the full-scale system with 10 camera pairs providing 360° coverage.

### Features
- 10 mono-thermal camera pairs (20 cameras total)
- Circular array for 360° coverage
- Synchronized acquisition
- Multi-camera fusion
- Real-time performance monitoring
- System health dashboard
- Stress testing capability

### Usage

```bash
# Normal operation (60 seconds)
python multi_camera_demo.py

# Extended run
python multi_camera_demo.py --duration 300

# Show detailed health metrics
python multi_camera_demo.py --show-health

# Save performance metrics
python multi_camera_demo.py --save-metrics metrics.json

# Stress test with 200 targets
python multi_camera_demo.py --stress-test --duration 120

# Custom number of pairs
python multi_camera_demo.py --num-pairs 5
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--duration SEC` | Run duration in seconds | 60 |
| `--show-health` | Display detailed health metrics | False |
| `--save-metrics FILE` | Save metrics to JSON file | None |
| `--stress-test` | Run with 200 targets | False |
| `--num-pairs N` | Number of camera pairs | 10 |

### Output Example

```
================================================================================
MULTI-CAMERA TRACKING SYSTEM - Frame 1800
================================================================================
Uptime:             60.0s
FPS:                30.0
Frame Time:         32.1 ms

Tracking:
  Active Tracks:    156
  Confirmed:        149
  Detections:       168
  Latency:          45.23 ms

Multi-Camera Fusion:
  Avg Observations: 3.2 cameras/target
  Coverage:         360°

Camera System:
  Streaming:        20/20
  Ready:            0
  Errors:           0
  Avg FPS:          29.8
  Avg Temp:         48.5°C

System Resources:
  Memory:           287.3 MB
  Bandwidth:        3547.2 MB/s

Requirements Check:
  FPS ≥ 30:         ✓ (30.0)
  Latency < 100ms:  ✓ (45.23 ms)
  Tracks ≤ 200:     ✓ (149)
================================================================================
```

### Performance Metrics File

When using `--save-metrics`, the output JSON contains:
- System configuration
- Per-frame metrics
- Summary statistics
- Camera health data
- Tracking performance

---

## 3. Drone Tracking Simulation

**File:** `drone_tracking_sim.py`

Realistic simulation of tracking 200 drones with various flight patterns.

### Features
- Simulates up to 200 drones
- Realistic drone physics and trajectories
- Multiple trajectory types (linear, circular, hover, zigzag, spiral, evasive)
- Detection accuracy modeling
- Ground truth comparison
- Accuracy analysis (position/velocity RMSE)
- Configurable trajectory mix

### Trajectory Types

1. **Linear** - Straight-line flight
2. **Circular** - Circular patterns
3. **Hover** - Hovering with small drift
4. **Zigzag** - Zigzag patterns
5. **Spiral** - Spiral ascent/descent
6. **Evasive** - Evasive maneuvers

### Usage

```bash
# Standard 200 drone simulation
python drone_tracking_sim.py

# Custom drone count
python drone_tracking_sim.py --num-drones 100 --duration 60

# Different trajectory mixes
python drone_tracking_sim.py --trajectory-mix aggressive
python drone_tracking_sim.py --trajectory-mix calm

# With visualization
python drone_tracking_sim.py --visualize

# Save accuracy analysis
python drone_tracking_sim.py --save-analysis accuracy_report.json
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num-drones N` | Number of drones | 200 |
| `--duration SEC` | Simulation duration | 120 |
| `--visualize` | Enable visualization | False |
| `--save-analysis FILE` | Save analysis to file | None |
| `--trajectory-mix TYPE` | Trajectory distribution | balanced |

### Trajectory Mix Options

- **balanced** - Mix of all trajectory types
- **aggressive** - More evasive and zigzag patterns
- **calm** - More linear and hovering patterns

### Output Example

```
======================================================================
DRONE TRACKING SIMULATION - Frame 3600
======================================================================
Sim Time:         120.0s
Drones:           200
Detections:       192
Active Tracks:    197
Confirmed:        189
Latency:          67.89 ms

Accuracy (last 100 frames):
  Position RMSE:  1.23 m
  Velocity RMSE:  0.45 m/s
  Detection Rate: 96.8%
======================================================================

FINAL ACCURACY ANALYSIS
======================================================================
Simulation Duration:  120.0s
Total Frames:         3600
Total Drones:         200

Tracking Accuracy:
  Position RMSE:      1.18 m
  Velocity RMSE:      0.42 m/s
  Detection Rate:     97.2%

Tracker Performance:
  Tracks Created:     245
  Tracks Confirmed:   198
  Avg Latency:        68.45 ms

Requirements:
  Detection Rate >99%:  ✗ (97.2%)
  Position RMSE <5m:    ✓ (1.18m)
  Latency <100ms:       ✓
======================================================================
```

---

## 4. Streaming Client

**File:** `streaming_client.py`

Real-time client for subscribing to and visualizing motion tracking data streams.

### Features
- Multiple transport options (UDP, TCP, Shared Memory)
- Real-time target display
- 3D visualization (optional)
- Target statistics and analysis
- Low-latency streaming
- Data recording capability
- Automatic reconnection

### Transport Types

1. **Shared Memory** - Lowest latency, local only
2. **UDP** - Low latency, unreliable
3. **TCP** - Reliable, slightly higher latency

### Usage

```bash
# Connect via shared memory (lowest latency)
python streaming_client.py

# Connect via UDP
python streaming_client.py --transport udp --host 192.168.1.100 --port 8888

# Connect via TCP
python streaming_client.py --transport tcp --host 192.168.1.100 --port 8889

# Run for specific duration
python streaming_client.py --duration 300

# With visualization
python streaming_client.py --visualize

# Save received data
python streaming_client.py --save-data stream_data.json

# Custom statistics interval
python streaming_client.py --stats-interval 10
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--transport TYPE` | Transport: udp, tcp, shared_memory | shared_memory |
| `--host HOST` | Server host address | 127.0.0.1 |
| `--port PORT` | Server port | 8888 |
| `--duration SEC` | Run duration (None = indefinite) | None |
| `--visualize` | Enable 3D visualization | False |
| `--save-data FILE` | Save received data | None |
| `--stats-interval SEC` | Statistics interval | 5 |

### Output Example

```
======================================================================
STREAMING CLIENT STATUS
======================================================================
Connection:       shared_memory
Uptime:           45.2s
Messages:         1356
Message Rate:     30.0 msg/s
Avg Latency:      2.34 ms

Targets:
  Total Seen:     87
  Active (5s):    23
  Update Rate:    156 updates/s

Top Active Targets:
  Target   5: updates= 245 age=42.3s conf=0.95 max_speed=18.2m/s
  Target  12: updates= 198 age=38.7s conf=0.91 max_speed=14.5m/s
  Target  23: updates= 176 age=35.2s conf=0.88 max_speed=16.8m/s
  ...

Subscriber Stats:
  Received:       1356
  Dropped:        3
  Errors:         0
  Throughput:     30.0 msg/s
  Bandwidth:      1.2 MB/s
======================================================================
```

---

## 5. Calibration Tool

**File:** `calibration_tool.py`

Interactive tool for camera calibration and validation.

### Features
- Intrinsic calibration (individual cameras)
- Stereo calibration (camera pairs)
- Mono-thermal registration
- Checkerboard/circle pattern detection
- Calibration validation
- Quality assessment
- Parameter export to JSON
- Batch calibration for all pairs

### Calibration Workflow

1. **Intrinsic Calibration** - Calibrate individual camera parameters
2. **Stereo Calibration** - Calibrate relative geometry between pair
3. **Registration** - Register mono and thermal images
4. **Validation** - Verify calibration quality
5. **Export** - Save parameters to files

### Usage

```bash
# Calibrate single pair (complete workflow)
python calibration_tool.py --pair-id 0

# Specify calibration target
python calibration_tool.py --target checkerboard --pair-id 0
python calibration_tool.py --target circles --pair-id 1

# Custom number of calibration images
python calibration_tool.py --num-images 30 --pair-id 0

# With validation
python calibration_tool.py --validate --pair-id 0

# Export to custom directory
python calibration_tool.py --export-dir ./my_calibration --pair-id 0

# Calibrate all 10 pairs
python calibration_tool.py --all-pairs --num-images 25

# Show calibration report
python calibration_tool.py --report
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pair-id ID` | Camera pair to calibrate | 0 |
| `--target TYPE` | Target: checkerboard, circles | checkerboard |
| `--num-images N` | Number of calibration images | 20 |
| `--validate` | Run validation after calibration | False |
| `--export-dir DIR` | Export directory | calibration |
| `--all-pairs` | Calibrate all 10 pairs | False |
| `--report` | Show calibration report | False |

### Output Example

```
======================================================================
COMPLETE CALIBRATION WORKFLOW - Pair 0
======================================================================

This will perform:
  1. Mono camera intrinsic calibration
  2. Thermal camera intrinsic calibration
  3. Stereo pair calibration
  4. Mono-thermal registration
  5. Validation
======================================================================

======================================================================
INTRINSIC CALIBRATION - Pair 0 MONO Camera
======================================================================

1. Capturing 20 calibration images...
2. Detecting calibration pattern...
Pattern detection: 20/20 successful (100.0%)
3. Computing camera calibration...

✓ Calibration successful!

Intrinsic Parameters:
  Focal Length:     fx=9216.0, fy=9216.0
  Principal Point:  cx=3840.0, cy=2160.0
  Distortion:       [0.05 -0.02 0.0 0.0 0.01]
  Reprojection Error: 0.387 pixels
======================================================================

... (continues for thermal, stereo, registration) ...

======================================================================
CALIBRATION VALIDATION - Pair 0
======================================================================

1. Intrinsic Calibration:
   Mono:    ✓ (error=0.387px)
   Thermal: ✓ (error=0.421px)

2. Stereo Calibration:
   Status:  ✓
   Reproj:  0.523px
   Epipolar: 0.287px

3. Mono-Thermal Registration:
   Status:  ✓
   Error:   1.834px
   MI:      0.892

----------------------------------------------------------------------
Overall: ✓ VALID
======================================================================

✓ Calibration exported successfully
```

### Calibration Files

Exported calibration files (JSON format):
- `pair_0_calibration.json` - Complete calibration data
- Contains:
  - Camera intrinsics (focal length, distortion)
  - Stereo parameters (baseline, rotation, translation)
  - Registration parameters (homography, mapping)
  - Quality metrics

---

## Integration Examples

### Example 1: Run Basic Tracking and Stream Data

```bash
# Terminal 1: Start basic tracking (generates motion data)
python basic_tracking.py --duration 300

# Terminal 2: Connect streaming client
python streaming_client.py --save-data motion_log.json
```

### Example 2: Calibrate and Validate System

```bash
# Calibrate all camera pairs
python calibration_tool.py --all-pairs --num-images 25

# Validate with drone simulation
python drone_tracking_sim.py --num-drones 200 --save-analysis validation.json

# View calibration report
python calibration_tool.py --report
```

### Example 3: Performance Testing

```bash
# Stress test multi-camera system
python multi_camera_demo.py --stress-test --duration 300 --save-metrics stress_test.json

# Analyze results
python -c "import json; data=json.load(open('stress_test.json')); print('Avg FPS:', data['summary']['avg_fps'])"
```

---

## Performance Expectations

### Basic Tracking (1 pair, 50 targets)
- **FPS:** 30+
- **Latency:** 10-20 ms
- **Memory:** < 100 MB
- **CPU:** Single core

### Multi-Camera (10 pairs, 100 targets)
- **FPS:** 30+
- **Latency:** 40-60 ms
- **Memory:** 200-400 MB
- **Bandwidth:** 3-4 GB/s (compressed)

### Stress Test (10 pairs, 200 targets)
- **FPS:** 28-30
- **Latency:** 60-80 ms
- **Memory:** 400-500 MB
- **CPU:** Multi-core utilized

### Drone Simulation (200 drones)
- **Position RMSE:** < 2 m
- **Velocity RMSE:** < 0.5 m/s
- **Detection Rate:** 95-98%
- **Latency:** 60-80 ms

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you run from examples directory
cd /path/to/Pixeltovoxelprojector/examples
python basic_tracking.py
```

**2. Shared Memory Connection Failed**
```bash
# Check if shared memory exists
ls -la /dev/shm/

# Try UDP instead
python streaming_client.py --transport udp
```

**3. Low FPS**
```bash
# Reduce number of targets
python multi_camera_demo.py --num-pairs 5

# Check system resources
htop
```

**4. Calibration Pattern Not Detected**
```bash
# Ensure proper lighting
# Use higher contrast checkerboard
# Increase number of images
python calibration_tool.py --num-images 30
```

---

## Development Notes

### Simulated vs Real Implementation

These examples use **simulated data** for demonstration purposes:
- Camera frames are simulated (random data)
- Detections are generated algorithmically
- No actual hardware required

### For Real Hardware

To use with real cameras:

1. Replace `simulate_detections()` with actual frame grabbing
2. Implement real detection algorithms (YOLO, etc.)
3. Add actual calibration pattern detection (OpenCV)
4. Connect to real GigE Vision cameras
5. Implement real coordinate streaming protocol

### Key Files to Modify

- `basic_tracking.py`: Line ~150 (`simulate_detections`)
- `multi_camera_demo.py`: Line ~300 (`simulate_multi_camera_detections`)
- `calibration_tool.py`: Line ~100 (`simulate_calibration_images`)

---

## Additional Resources

- [System Architecture](../docs/architecture.md)
- [API Documentation](../docs/api.md)
- [Performance Tuning](../docs/performance.md)
- [Hardware Setup](../docs/hardware_setup.md)

---

## License

See [LICENSE](../LICENSE) file for details.

## Support

For issues or questions:
- Open an issue on GitHub
- Contact: support@motiontracking.com
- Documentation: https://docs.motiontracking.com
