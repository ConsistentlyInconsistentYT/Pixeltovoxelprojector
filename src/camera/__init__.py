"""
Camera Tracking System

High-precision camera pose tracking system for multi-camera pixel-to-voxel projection

Components:
- pose_tracker: Real-time 6DOF pose estimation with sensor fusion
- orientation_manager: High-frequency quaternion-based orientation tracking
- position_broadcast: Real-time pose broadcasting and coordinate transformations

Author: Pixeltovoxelprojector Team
"""

from .pose_tracker import (
    CameraPoseTracker,
    CameraPose,
    IMUMeasurement,
    GPSMeasurement,
    VIOMeasurement,
    ExtendedKalmanFilter
)

from .position_broadcast import (
    PositionBroadcaster,
    PositionReceiver,
    BroadcastPose,
    CoordinateTransformer,
    CoordinateFrame,
    CameraCalibration
)

__all__ = [
    # Pose Tracker
    'CameraPoseTracker',
    'CameraPose',
    'IMUMeasurement',
    'GPSMeasurement',
    'VIOMeasurement',
    'ExtendedKalmanFilter',

    # Position Broadcast
    'PositionBroadcaster',
    'PositionReceiver',
    'BroadcastPose',
    'CoordinateTransformer',
    'CoordinateFrame',
    'CameraCalibration',
]

__version__ = '1.0.0'
