"""
Pytest configuration and shared fixtures for integration tests
"""

import pytest
import numpy as np
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory"""
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def output_dir():
    """Get output directory for test results"""
    out_dir = Path(__file__).parent.parent / "test-results"
    out_dir.mkdir(exist_ok=True)
    return out_dir


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility"""
    np.random.seed(42)
    return 42


@pytest.fixture
def mock_8k_frame():
    """Generate mock 8K frame for testing"""
    # Create smaller frame for faster tests
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def mock_detection():
    """Generate mock detection"""
    return {
        'x': np.random.uniform(0, 7680),
        'y': np.random.uniform(0, 4320),
        'width': np.random.uniform(5, 20),
        'height': np.random.uniform(5, 20),
        'confidence': np.random.uniform(0.7, 0.95),
        'velocity_x': np.random.uniform(-5, 5),
        'velocity_y': np.random.uniform(-5, 5)
    }


# Performance markers
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as stress test"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )


# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GPU requirements"""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    if not has_gpu:
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "requires_gpu" in item.keywords:
                item.add_marker(skip_gpu)


# Performance tracking
@pytest.fixture(autouse=True)
def track_performance(request):
    """Track test performance"""
    import time
    start_time = time.time()

    yield

    duration = time.time() - start_time

    if duration > 60:
        logging.warning(
            f"Test {request.node.name} took {duration:.2f} seconds (> 60s threshold)"
        )
