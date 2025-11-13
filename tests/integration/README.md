# Integration Testing Framework

Comprehensive integration tests for the pixel-to-voxel projection system with multi-camera processing, detection, and tracking.

## Test Structure

```
tests/integration/
├── test_full_pipeline.py      # End-to-end system tests
├── test_camera_sync.py         # Camera synchronization tests
├── test_streaming.py           # Network streaming tests
├── test_detection.py           # Detection accuracy tests
└── README.md                   # This file

tests/test_data/
├── synthetic_video_generator.py   # 8K video generation
├── trajectory_generator.py        # Drone trajectory simulation
├── ground_truth_generator.py      # Ground truth annotations
└── __init__.py
```

## Running Tests

### Run all integration tests
```bash
pytest tests/integration/ -v
```

### Run specific test file
```bash
pytest tests/integration/test_full_pipeline.py -v
```

### Run with coverage
```bash
pytest tests/integration/ --cov=src --cov-report=html
```

### Run specific test categories
```bash
# Camera synchronization tests
pytest tests/integration/ -m camera

# Detection tests
pytest tests/integration/ -m detection

# Stress tests (200 targets)
pytest tests/integration/ -m stress

# Slow tests
pytest tests/integration/ -m slow
```

### Run in parallel (faster)
```bash
pytest tests/integration/ -n auto
```

## Test Requirements

### System Requirements
- Python 3.8+
- NumPy, SciPy, OpenCV
- 8GB+ RAM (16GB recommended for stress tests)
- Network access (for streaming tests)

### Performance Requirements Tested
- **Latency**: < 100ms end-to-end processing
- **Detection Rate**: > 99%
- **False Positive Rate**: < 2%
- **Synchronization**: < 1ms average, < 10ms max
- **Target Capacity**: 200 simultaneous tracks
- **Range**: 5km detection capability

## Test Categories

### 1. Full Pipeline Tests (`test_full_pipeline.py`)
- **Single camera pipeline**: Basic end-to-end processing
- **Multi-camera pipeline**: All 10 camera pairs
- **200 target stress test**: Maximum capacity validation
- **Detection accuracy**: 99%+ detection, <2% false positives
- **Performance regression**: Latency validation across loads

**Key Metrics:**
- Average latency < 100ms
- Sync error < 10ms
- Detection rate > 95%

### 2. Camera Synchronization Tests (`test_camera_sync.py`)
- **Timestamp sync accuracy**: Sub-millisecond synchronization
- **Frame alignment**: All 10 pairs synchronized
- **Dropped frame detection**: Detection and recovery
- **Hardware trigger coordination**: 20-camera trigger sync
- **PTP synchronization**: Precision Time Protocol quality
- **Multi-pair coordination**: Cross-pair synchronization

**Key Metrics:**
- Average sync error < 1ms
- Max sync error < 10ms
- PTP jitter < 1000µs
- Hardware trigger response > 95%

### 3. Network Streaming Tests (`test_streaming.py`)
- **Network reliability**: Packet delivery validation
- **Latency measurements**: End-to-end timing
- **Multi-client streaming**: Concurrent client support
- **Failover scenarios**: Automatic node recovery
- **Bandwidth utilization**: 8K streaming capacity
- **Load balancing**: Worker distribution efficiency

**Key Metrics:**
- Network latency < 50ms
- Failover completion < 5s
- Load balance std < 0.3
- Multi-client success > 90%

### 4. Detection System Tests (`test_detection.py`)
- **5km range detection**: Distance-dependent accuracy
- **200 simultaneous targets**: Maximum tracking capacity
- **Detection accuracy validation**: Precision/recall metrics
- **Occlusion handling**: Track recovery from occlusion
- **False positive rejection**: Multi-modal filtering
- **Track continuity**: ID consistency across frames
- **Velocity estimation**: Motion prediction accuracy

**Key Metrics:**
- Detection rate > 99% (up to 4km)
- Detection rate > 70% (at 5km)
- False positive rate < 2%
- Track ID stability > 80%
- Velocity error < 2 pixels/frame

## Test Data Generation

### Synthetic Video Generation
```python
from tests.test_data.synthetic_video_generator import SyntheticVideoGenerator, DroneTarget

generator = SyntheticVideoGenerator(width=7680, height=4320)
generator.generate_background("clear")

drones = [
    DroneTarget(0, position=(100, 50, 2000), velocity=(5, 0, 0), size=0.2, temperature=310.0)
]

frame, detections = generator.generate_frame(drones, "monochrome")
```

### Trajectory Generation
```python
from tests.test_data.trajectory_generator import TrajectoryGenerator

generator = TrajectoryGenerator(duration_seconds=60.0, sample_rate_hz=30.0)

linear = generator.generate_linear(0, (0, 0, 1000), (500, 300, 2000))
circular = generator.generate_circular(1, (0, 0, 1500), radius=200)
evasive = generator.generate_evasive(2, (100, 100, 2000), (1, 0, 0.5))
```

### Ground Truth Generation
```python
from tests.test_data.ground_truth_generator import GroundTruthGenerator

gt_gen = GroundTruthGenerator(frame_width=7680, frame_height=4320)
ground_truth = gt_gen.generate_from_trajectories(trajectories, projection_func, num_frames=100)
gt_gen.save_ground_truth(ground_truth, "ground_truth.json")
```

## Coverage Requirements

Target coverage: **80%+**

Generate coverage reports:
```bash
# HTML report
pytest tests/integration/ --cov=src --cov-report=html
open coverage_html/index.html

# Terminal report
pytest tests/integration/ --cov=src --cov-report=term-missing

# XML report (for CI/CD)
pytest tests/integration/ --cov=src --cov-report=xml
```

## CI/CD Integration

The integration tests run automatically on:
- Every push to `main` or `develop`
- Every pull request
- Nightly at 2 AM UTC
- Manual trigger with `[stress-test]` in commit message

See `.github/workflows/integration-tests.yml` for configuration.

### CI Pipeline Stages
1. **Integration Tests**: Core functionality validation
2. **Benchmark Tests**: Performance regression detection
3. **Stress Tests**: Maximum load validation (nightly)

## Performance Benchmarking

Run performance benchmarks:
```bash
pytest tests/benchmarks/ --benchmark-only
```

Compare results:
```bash
pytest tests/benchmarks/ --benchmark-compare
```

## Troubleshooting

### Tests timing out
- Increase timeout: `pytest --timeout=600`
- Run fewer tests: `pytest tests/integration/test_detection.py`

### Out of memory
- Reduce test data size
- Run tests sequentially: `pytest -n 1`

### GPU tests failing
- Tests requiring GPU are automatically skipped if no GPU available
- Force skip: `pytest -m "not requires_gpu"`

### Network tests failing
- Check network connectivity
- Skip network tests: `pytest -m "not requires_network"`

## Contributing

When adding new tests:
1. Follow existing test structure
2. Add appropriate markers (`@pytest.mark.integration`, etc.)
3. Update this README with new test categories
4. Ensure tests are deterministic (use fixtures/seeds)
5. Add docstrings describing test purpose and requirements
6. Validate performance requirements are met

## Test Markers

Available markers:
- `@pytest.mark.integration` - Integration test
- `@pytest.mark.slow` - Slow running test (> 30s)
- `@pytest.mark.stress` - Stress test with high load
- `@pytest.mark.requires_gpu` - Requires GPU hardware
- `@pytest.mark.requires_network` - Requires network access
- `@pytest.mark.camera` - Camera system test
- `@pytest.mark.detection` - Detection/tracking test
- `@pytest.mark.streaming` - Network streaming test
- `@pytest.mark.fusion` - Multi-modal fusion test
