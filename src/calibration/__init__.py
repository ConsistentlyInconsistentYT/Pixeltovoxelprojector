"""
Calibration System for Multi-Camera Tracking

This module provides comprehensive camera calibration and coordinate transformation
capabilities for multi-camera tracking systems.

Components:
- camera_calibrator: Camera calibration (intrinsic, extrinsic, stereo, multi-camera)
- coordinate_transformer: Coordinate system transformations (C++ library)
- transform_optimizer: CUDA-accelerated batch transformations

Usage:
    from calibration import CameraCalibrator, MultiCameraCalibrator

    # Create calibrator
    calibrator = CameraCalibrator(camera_id=0, name="cam0")

    # Add calibration images
    for img in calibration_images:
        calibrator.add_calibration_image(img)

    # Calibrate
    result = calibrator.calibrate_intrinsics()
    print(f"RMS Error: {result.rms_error} pixels")

    # Save calibration
    calibrator.save_calibration("calibration.json")
"""

from .camera_calibrator import (
    CameraCalibrator,
    MultiCameraCalibrator,
    CameraIntrinsics,
    CameraExtrinsics,
    CalibrationPattern,
    CalibrationResult
)

__all__ = [
    'CameraCalibrator',
    'MultiCameraCalibrator',
    'CameraIntrinsics',
    'CameraExtrinsics',
    'CalibrationPattern',
    'CalibrationResult'
]

__version__ = '1.0.0'
