"""
Thermal + Monochrome Camera Fusion System

High-performance multi-spectral fusion for long-range target detection and tracking.
Combines thermal signature detection with monochrome motion analysis for robust
target confirmation and false positive reduction.

Components:
- thermal_mono_fusion (C++): CUDA-accelerated image registration and fusion
- fusion_manager: Multi-camera pair management and coordination
- detection_fusion: Cross-modal detection validation and tracking
"""

__version__ = "1.0.0"
__author__ = "Pixeltovoxelprojector Team"

from .fusion_manager import (
    FusionManager,
    FusionConfig,
    CameraPair,
    FusedDetection as FusedDetectionManager
)

from .detection_fusion import (
    DetectionFusion,
    FusionWeights,
    MotionDetection,
    ThermalDetection,
    FusedDetection,
    DetectionSource,
    OcclusionType
)

# Try to import C++ module
try:
    import thermal_mono_fusion as cpp_fusion
    CUDA_AVAILABLE = True
except ImportError:
    cpp_fusion = None
    CUDA_AVAILABLE = False

__all__ = [
    # Manager classes
    'FusionManager',
    'FusionConfig',
    'CameraPair',
    'FusedDetectionManager',

    # Detection fusion
    'DetectionFusion',
    'FusionWeights',
    'MotionDetection',
    'ThermalDetection',
    'FusedDetection',
    'DetectionSource',
    'OcclusionType',

    # C++ module (if available)
    'cpp_fusion',
    'CUDA_AVAILABLE',
]
