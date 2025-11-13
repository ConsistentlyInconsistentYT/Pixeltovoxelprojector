"""
Performance Optimization Package

Provides adaptive performance management, profiling, and optimization tools
for the PixelToVoxelProjector system.

Modules:
- adaptive_manager: Dynamic parameter adjustment based on real-time metrics
- profiler: Comprehensive performance profiling with minimal overhead

Usage:
    from src.performance import AdaptivePerformanceManager, PerformanceProfiler

    # Create profiler
    profiler = PerformanceProfiler()
    profiler.start()

    # Create adaptive manager
    manager = AdaptivePerformanceManager()
    manager.start()

    # Process frames with profiling
    with profiler.section("process_frame"):
        process_frame()

    # Update adaptive manager
    manager.update_metrics(fps, latency, gpu_util)
"""

from .adaptive_manager import (
    AdaptivePerformanceManager,
    PerformanceMode,
    PerformanceTarget,
    AdaptiveParameters,
    PerformancePredictor
)

from .profiler import (
    PerformanceProfiler,
    ProfileSection,
    SectionTimer,
    GPUProfiler,
    MetricType
)

__all__ = [
    # Adaptive Management
    'AdaptivePerformanceManager',
    'PerformanceMode',
    'PerformanceTarget',
    'AdaptiveParameters',
    'PerformancePredictor',

    # Profiling
    'PerformanceProfiler',
    'ProfileSection',
    'SectionTimer',
    'GPUProfiler',
    'MetricType',
]

__version__ = '2.0.0'
