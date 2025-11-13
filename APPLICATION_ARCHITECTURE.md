# Application Architecture Documentation

## System Overview

The 8K Motion Tracking System is a comprehensive multi-camera tracking solution designed for real-time detection and tracking of up to 200 simultaneous targets at ranges exceeding 5 kilometers. The system processes 20 cameras (10 thermal-monochrome pairs) at 8K resolution and 30 FPS.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Main Application                                │
│                              (src/main.py)                                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    MotionTrackingSystem Class                       │    │
│  │                                                                     │    │
│  │  • Configuration Loading & Validation                              │    │
│  │  • Component Initialization                                        │    │
│  │  • Lifecycle Management                                            │    │
│  │  • Graceful Shutdown                                               │    │
│  └─────────────────────────┬──────────────────────────────────────────┘    │
└────────────────────────────┼───────────────────────────────────────────────┘
                             │
         ┌───────────────────┴──────────────────┐
         │                                       │
┌────────▼──────────────┐           ┌───────────▼────────────┐
│  Configuration Layer  │           │  Coordination Layer     │
│  (YAML Config)        │           │  (Pipeline Coordinator) │
│                       │           │                         │
│  • System settings    │           │  • Component registry   │
│  • Camera config      │           │  • Dependency mgmt      │
│  • Voxel grid params  │           │  • Health monitoring    │
│  • Detection settings │           │  • Error recovery       │
│  • Network settings   │           │  • Resource allocation  │
└───────────────────────┘           └───────────┬─────────────┘
                                                 │
                     ┌───────────────────────────┴────────────────────────┐
                     │                                                     │
        ┌────────────▼─────────────┐                   ┌─────────────────▼──────────┐
        │   Component Layer        │                   │   Processing Layer          │
        │                          │                   │   (Processing Pipeline)     │
        │  ┌──────────────────┐   │                   │                             │
        │  │ Camera Manager   │   │                   │  ┌──────────────────────┐  │
        │  │  • 20 cameras    │◄──┼───────────────────┼──┤  Frame Acquisition   │  │
        │  │  • Pair sync     │   │                   │  └──────────┬───────────┘  │
        │  │  • Health mon.   │   │                   │             │              │
        │  └──────────────────┘   │                   │  ┌──────────▼───────────┐  │
        │                          │                   │  │  Motion Extraction   │  │
        │  ┌──────────────────┐   │                   │  │  • C++ accelerated   │  │
        │  │ Fusion Manager   │◄──┼───────────────────┼──┤  • 8K processing     │  │
        │  │  • Thermal-mono  │   │                   │  └──────────┬───────────┘  │
        │  │  • Registration  │   │                   │             │              │
        │  │  • Enhancement   │   │                   │  ┌──────────▼───────────┐  │
        │  └──────────────────┘   │                   │  │  Fusion Processing   │  │
        │                          │                   │  │  • Cross-validation  │  │
        │  ┌──────────────────┐   │                   │  │  • FP reduction      │  │
        │  │ Voxel Manager    │◄──┼───────────────────┼──┤  • Confidence boost  │  │
        │  │  • 5km³ grid     │   │                   │  └──────────┬───────────┘  │
        │  │  • Multi-LOD     │   │                   │             │              │
        │  │  • 500MB limit   │   │                   │  ┌──────────▼───────────┐  │
        │  └──────────────────┘   │                   │  │  Tracking            │  │
        │                          │                   │  │  • Kalman filter     │  │
        │  ┌──────────────────┐   │                   │  │  • 200 tracks        │  │
        │  │ Tracker          │◄──┼───────────────────┼──┤  • Occlusion handle  │  │
        │  │  • Multi-target  │   │                   │  └──────────┬───────────┘  │
        │  │  • 200+ objects  │   │                   │             │              │
        │  │  • Kalman filter │   │                   │  ┌──────────▼───────────┐  │
        │  └──────────────────┘   │                   │  │  Voxel Updates       │  │
        │                          │                   │  │  • 3D mapping        │  │
        │  ┌──────────────────┐   │                   │  │  • LOD selection     │  │
        │  │ System Monitor   │   │                   │  └──────────┬───────────┘  │
        │  │  • 10Hz updates  │   │                   │             │              │
        │  │  • CPU/GPU/Mem   │   │                   │  ┌──────────▼───────────┐  │
        │  │  • <1% overhead  │   │                   │  │  Coordinate Stream   │  │
        │  └──────────────────┘   │                   │  │  • Network output    │  │
        └──────────────────────────┘                   │  └──────────────────────┘  │
                                                       └────────────────────────────┘
```

## Component Details

### 1. Main Application (`src/main.py`)

**Purpose:** Entry point and top-level orchestration

**Key Features:**
- Command-line interface with argparse
- YAML configuration loading and validation
- Component initialization in dependency order
- Signal handling for graceful shutdown
- Simulation mode for testing

**Key Classes:**
- `MotionTrackingSystem`: Main application class

**Usage:**
```bash
python main.py --config config/system_config.yaml
```

### 2. Configuration Layer (`src/config/system_config.yaml`)

**Purpose:** Centralized system configuration

**Sections:**
- **System:** Basic system settings
- **Cameras:** 10 camera pairs (20 cameras total)
- **Voxel Grid:** 3D space representation
- **Detection:** Motion detection parameters
- **Fusion:** Thermal-mono fusion settings
- **Network:** Streaming configuration
- **Performance:** Threading and GPU settings
- **Monitoring:** Health monitoring configuration
- **Logging:** Log levels and outputs

### 3. Pipeline Coordinator (`src/pipeline/pipeline_coordinator.py`)

**Purpose:** Component lifecycle and health management

**Responsibilities:**
- Component registration and dependency management
- Initialization order calculation (topological sort)
- Health monitoring and watchdog
- Automatic error recovery
- Graceful shutdown coordination

**Key Classes:**
- `PipelineCoordinator`: Main coordinator
- `ComponentStatus`: Component state tracking
- `ComponentState`: Lifecycle states enum

**Component States:**
```
UNINITIALIZED → INITIALIZING → READY → RUNNING
                                   ↓
                              STOPPING → STOPPED
                                   ↓
                                ERROR → RECOVERING
```

### 4. Processing Pipeline (`src/pipeline/processing_pipeline.py`)

**Purpose:** Main data processing flow

**Pipeline Stages:**

1. **Frame Acquisition**
   - Grab frames from camera pairs
   - Synchronization verification
   - Frame buffering

2. **Motion Extraction**
   - C++ accelerated processing
   - 8K frame analysis
   - Coordinate extraction

3. **Fusion Processing**
   - Thermal-mono alignment
   - Multi-spectral fusion
   - False positive reduction

4. **Tracking**
   - Multi-target association
   - Kalman filter updates
   - Track management

5. **Voxel Updates**
   - 3D position mapping
   - LOD selection
   - Grid updates

6. **Coordinate Streaming**
   - Network output
   - Callback execution

**Threading Model:**
- N processing workers (configurable, default 8)
- M fusion workers (default 4)
- 1 tracking worker
- 1 streaming worker

**Key Classes:**
- `ProcessingPipeline`: Main pipeline orchestrator
- `PipelineConfig`: Pipeline configuration
- `FrameData`: Frame data container
- `ProcessingResult`: Result container

### 5. Camera Manager (`src/camera/camera_manager.py`)

**Purpose:** Camera hardware management

**Features:**
- GigE Vision camera support
- Hardware trigger synchronization
- Health monitoring per camera
- Automatic reconnection
- Configuration persistence

**Key Classes:**
- `CameraManager`: Main camera coordinator
- `CameraInterface`: Individual camera control
- `CameraConfiguration`: Camera parameters
- `CameraPair`: Pair association
- `CameraHealth`: Health metrics

### 6. Voxel Grid Manager (`src/voxel/grid_manager.py`)

**Purpose:** 3D space representation

**Features:**
- Multi-resolution LOD hierarchy
- Dynamic grid sizing
- Memory management (<500MB)
- Background pruning
- Object tracking

**LOD Levels:**
- Ultra High: 0.1m @ <100m
- High: 0.25m @ <500m
- Medium: 0.5m @ <1km
- Low: 1.0m @ <5km
- Ultra Low: 2.0m @ >5km

**Key Classes:**
- `VoxelGridManager`: Main grid manager
- `GridConfig`: Grid configuration
- `TrackedObject`: Object representation
- `LODLevel`: Level of detail enum

### 7. Fusion Manager (`src/fusion/fusion_manager.py`)

**Purpose:** Thermal-monochrome fusion

**Features:**
- Image registration (homography)
- Cross-modal validation
- Thermal enhancement in low light
- False positive reduction
- Multi-threaded processing

**Key Classes:**
- `FusionManager`: Main fusion orchestrator
- `FusionConfig`: Fusion parameters
- `FusedDetection`: Fused detection result
- `CameraPair`: Pair configuration

### 8. Multi-Target Tracker (`src/detection/tracker.py`)

**Purpose:** Track management

**Features:**
- Kalman filter tracking
- Hungarian algorithm association
- Occlusion handling
- Track lifecycle management
- 200+ simultaneous tracks

**Key Classes:**
- `MultiTargetTracker`: Main tracker
- `Track`: Individual track
- `TrackMetrics`: Performance metrics

### 9. System Monitor (`src/monitoring/system_monitor.py`)

**Purpose:** Performance and health monitoring

**Metrics:**
- CPU, Memory, GPU utilization
- Network bandwidth
- Camera health (20 cameras)
- Detection accuracy
- <1% overhead

**Update Rate:** 10 Hz

**Key Classes:**
- `SystemMonitor`: Main monitor
- `SystemMetrics`: Metrics container
- `GPUMetrics`, `CPUMetrics`, etc.

## Data Flow

### Frame Processing Flow

```
Camera Frames (8K @ 30fps)
         ↓
   Frame Queue
         ↓
   Processing Workers (parallel)
    ├─ Motion Extraction
    └─ Feature Detection
         ↓
   Fusion Queue
         ↓
   Fusion Workers (parallel)
    ├─ Registration
    ├─ Thermal-Mono Fusion
    └─ False Positive Reduction
         ↓
   Tracking Queue
         ↓
   Tracking Worker (single)
    ├─ Prediction
    ├─ Association
    └─ Update
         ↓
   Output Queue
         ↓
   ├─ Voxel Grid Updates
   ├─ Network Streaming
   └─ Callbacks
```

### Queue Management

**Queues:**
1. `frame_queue`: Raw camera frames
2. `fusion_queue`: Extracted motion data
3. `tracking_queue`: Fused detections
4. `output_queue`: Final results

**Queue Sizes:** Configurable (default 100)

**Overflow Handling:** Drop oldest frames, increment drop counter

## Performance Characteristics

### Processing Latency Budget

| Stage | Target | Typical | Max |
|-------|--------|---------|-----|
| Frame Acquisition | 2ms | 1ms | 5ms |
| Motion Extraction | 20ms | 15ms | 30ms |
| Fusion | 10ms | 8ms | 15ms |
| Tracking | 5ms | 3ms | 10ms |
| Voxel Update | 2ms | 1ms | 5ms |
| Streaming | 1ms | 0.5ms | 2ms |
| **Total** | **40ms** | **28.5ms** | **67ms** |

### Resource Requirements

**CPU:**
- Minimum: 8 cores
- Recommended: 16 cores
- Usage: 70-85% (balanced)

**Memory:**
- System: 16GB minimum
- Voxel Grid: <500MB
- Frame Buffers: ~2GB
- Total: ~4GB active

**GPU:**
- NVIDIA GPU with CUDA support
- 8GB VRAM minimum
- Usage: 60-80%

**Network:**
- 10 GigE recommended
- ~8 Gbps bandwidth per camera pair
- Total: ~80 Gbps for 10 pairs

**Storage:**
- Logs: ~100 MB/hour
- Recordings: ~1 TB/hour (if enabled)

## Error Handling

### Error Recovery Hierarchy

1. **Component-Level**
   - Automatic reconnection
   - Buffer resets
   - Thread restarts

2. **Pipeline-Level**
   - Frame drops
   - Queue overflow handling
   - Processing timeout recovery

3. **System-Level**
   - Component restart (up to 3 attempts)
   - Graceful degradation
   - Emergency shutdown

### Health Monitoring

**Health Check Interval:** 5 seconds

**Health States:**
- `healthy`: Normal operation
- `warning`: Minor issues, still functional
- `critical`: Major issues, may need intervention
- `offline`: Component not responding

**Watchdog:** 1 second interval, detects hangs

## Configuration Validation

### Validation Stages

1. **Syntax Validation**
   - YAML syntax check
   - Required sections present
   - Data types correct

2. **Semantic Validation**
   - Value ranges
   - Logical consistency
   - Resource availability

3. **Runtime Validation**
   - Camera connectivity
   - GPU availability
   - Network interfaces

### Validation Errors

Configuration errors are detected at startup and logged:
- Missing sections
- Invalid parameters
- Resource conflicts
- Hardware unavailability

## Extensibility

### Adding New Components

1. Implement component with required methods:
   - `initialize()` or `init()`
   - `start()`
   - `stop()`
   - `get_health()` (optional)
   - `get_metrics()` (optional)

2. Register with coordinator:
```python
coordinator.register_component(
    name='my_component',
    component=instance,
    dependencies=['camera_manager']
)
```

3. Add to configuration file

### Adding Custom Callbacks

**Pipeline Results:**
```python
def my_callback(result):
    # Process result
    pass

pipeline.register_coordinate_callback(my_callback)
```

**State Changes:**
```python
def on_state_change(component, old_state, new_state):
    # Handle state change
    pass

coordinator.register_state_change_callback(on_state_change)
```

## Deployment Considerations

### Single Machine Deployment

- All components on one system
- Shared memory for low latency
- Requires high-end workstation

### Distributed Deployment

- Cameras on separate acquisition nodes
- Processing on GPU cluster
- Network streaming between nodes
- Requires distributed coordinator (future work)

### Container Deployment

- Docker container with dependencies
- Volume mounts for configuration
- Host network mode for cameras
- GPU passthrough required

## Testing

### Unit Tests

Each component has unit tests:
```bash
pytest src/camera/test_camera_system.py
pytest src/detection/test_detection.py
pytest src/voxel/test_requirements.py
```

### Integration Tests

Full system tests:
```bash
python verify_tracking_system.py
```

### Simulation Mode

Test without hardware:
```bash
python main.py --simulate
```

## Future Enhancements

### Planned Features

1. **Distributed Processing**
   - Multi-node deployment
   - Load balancing
   - Fault tolerance

2. **Machine Learning Integration**
   - Deep learning detection
   - Improved classification
   - Behavior analysis

3. **Advanced Visualization**
   - 3D visualization
   - Real-time dashboards
   - Replay capability

4. **Cloud Integration**
   - Cloud storage
   - Remote monitoring
   - API access

---

**Last Updated:** 2025-11-13
**Version:** 1.0.0
