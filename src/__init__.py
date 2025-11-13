"""
8K Video Analysis and Motion Extraction Pipeline

This package provides high-performance video processing for 8K video streams
with hardware-accelerated decoding and real-time motion extraction.

Camera Tracking System:
- Real-time 6DOF pose estimation with sensor fusion
- High-frequency quaternion-based orientation tracking
- Position broadcasting and coordinate transformations
"""

from .video_processor import VideoProcessor, VideoStream, FrameData, MotionData
from .codec_handler import CodecHandler, VideoCodec, HardwareAccel

# Camera tracking imports
from .camera import (
    CameraPoseTracker,
    CameraPose,
    PositionBroadcaster,
    PositionReceiver,
    CoordinateTransformer,
)

__version__ = '1.0.0'
__all__ = [
    'VideoProcessor',
    'VideoStream',
    'FrameData',
    'MotionData',
    'CodecHandler',
    'VideoCodec',
    'HardwareAccel',
    # Camera tracking
    'CameraPoseTracker',
    'CameraPose',
    'PositionBroadcaster',
    'PositionReceiver',
    'CoordinateTransformer',
]
