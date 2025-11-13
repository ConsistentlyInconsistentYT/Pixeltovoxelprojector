#!/usr/bin/env python3
"""
Basic Tracking Example - Single Camera Pair Motion Tracking
============================================================

This example demonstrates the simplest setup for the 8K motion tracking system:
- Single mono-thermal camera pair
- Basic object detection and tracking
- Voxel visualization
- Real-time motion coordinate display

Requirements:
- Python 3.8+
- NumPy
- Matplotlib (for visualization)

Usage:
    python basic_tracking.py

    Optional arguments:
        --camera-id PAIR_ID     Camera pair ID (default: 0)
        --duration SECONDS      Run duration in seconds (default: 30)
        --visualize             Enable 3D visualization
        --save-output FILE      Save tracking data to file

Author: Motion Tracking System
Date: 2025-11-13
"""

import sys
import time
import argparse
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import system components
from camera.camera_manager import (
    CameraManager, CameraConfiguration, CameraPair,
    CameraType, ConnectionType
)
from detection.tracker import MultiTargetTracker
from voxel.grid_manager import VoxelGridManager, GridConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BasicTrackingSystem:
    """
    Simple single-pair tracking system for demonstration
    """

    def __init__(self, camera_pair_id: int = 0):
        """
        Initialize basic tracking system

        Args:
            camera_pair_id: ID of the camera pair to use
        """
        self.pair_id = camera_pair_id

        # Initialize camera manager with just 1 pair
        logger.info("Initializing camera manager...")
        self.camera_manager = CameraManager(num_pairs=1)

        # Initialize tracker
        logger.info("Initializing multi-target tracker...")
        self.tracker = MultiTargetTracker(
            max_tracks=50,  # Track up to 50 objects
            detection_threshold=0.6,
            confirmation_threshold=3,
            max_age=10,
            frame_rate=30.0
        )

        # Initialize voxel grid manager
        logger.info("Initializing voxel grid...")
        grid_config = GridConfig(
            center=np.array([0.0, 0.0, 500.0]),  # 500m altitude
            size=np.array([1000.0, 1000.0, 500.0]),  # 1km x 1km x 500m
            base_resolution=1.0,  # 1 meter resolution
            max_memory_mb=100,  # Limited memory for basic example
            enable_lod=True
        )
        self.voxel_grid = VoxelGridManager(grid_config)

        # Tracking state
        self.frame_count = 0
        self.running = False
        self.start_time = 0.0

        # Statistics
        self.total_detections = 0
        self.total_tracks_created = 0

    def setup_cameras(self):
        """Configure and initialize the camera pair"""
        logger.info(f"Setting up camera pair {self.pair_id}...")

        # Configure mono camera (Camera 0)
        mono_config = CameraConfiguration(
            camera_id=0,
            pair_id=self.pair_id,
            camera_type=CameraType.MONO,
            connection=ConnectionType.GIGE_VISION,
            ip_address="192.168.1.10",
            width=7680,  # 8K resolution
            height=4320,
            frame_rate=30.0,
            exposure_time=10000.0,
            position=np.array([0.0, 0.0, 50.0]),  # 50m height
            orientation=np.eye(3)
        )

        # Configure thermal camera (Camera 1)
        thermal_config = CameraConfiguration(
            camera_id=1,
            pair_id=self.pair_id,
            camera_type=CameraType.THERMAL,
            connection=ConnectionType.GIGE_VISION,
            ip_address="192.168.1.11",
            width=640,
            height=512,
            frame_rate=30.0,
            exposure_time=5000.0,
            position=np.array([0.15, 0.0, 50.0]),  # 15cm offset from mono
            orientation=np.eye(3)
        )

        # Add cameras to manager
        self.camera_manager.add_camera(mono_config)
        self.camera_manager.add_camera(thermal_config)

        # Create camera pair
        pair = CameraPair(
            pair_id=self.pair_id,
            mono_camera_id=0,
            thermal_camera_id=1,
            stereo_baseline=0.15  # 15cm baseline
        )
        self.camera_manager.add_pair(pair)

        # Initialize cameras
        logger.info("Initializing cameras...")
        if not self.camera_manager.initialize_all_cameras():
            logger.error("Failed to initialize cameras")
            return False

        logger.info("Cameras initialized successfully!")
        return True

    def start_acquisition(self):
        """Start camera acquisition"""
        logger.info("Starting camera acquisition...")

        if not self.camera_manager.start_all_acquisition():
            logger.error("Failed to start acquisition")
            return False

        # Start health monitoring
        self.camera_manager.start_health_monitoring()

        # Start voxel background processing
        self.voxel_grid.start_background_processing()

        self.running = True
        self.start_time = time.time()

        logger.info("Acquisition started successfully!")
        return True

    def stop_acquisition(self):
        """Stop camera acquisition"""
        logger.info("Stopping acquisition...")

        self.running = False

        # Stop cameras
        self.camera_manager.stop_all_acquisition()
        self.camera_manager.stop_health_monitoring()

        # Stop voxel processing
        self.voxel_grid.stop_background_processing()

        logger.info("Acquisition stopped")

    def simulate_detections(self) -> List[Dict]:
        """
        Simulate object detections from cameras

        In a real system, this would:
        1. Grab frames from cameras
        2. Run detection algorithms
        3. Match mono and thermal detections
        4. Triangulate 3D positions

        Returns:
            List of detection dictionaries
        """
        # Simulate 2-5 random detections per frame
        num_detections = np.random.randint(2, 6)
        detections = []

        for _ in range(num_detections):
            # Random position in field of view
            x = np.random.uniform(-200, 200)  # meters
            y = np.random.uniform(-200, 200)
            z = np.random.uniform(100, 800)  # altitude

            # Random velocity
            vx = np.random.uniform(-5, 5)  # m/s
            vy = np.random.uniform(-5, 5)

            detection = {
                'x': x,
                'y': y,
                'z': z,
                'velocity_x': vx,
                'velocity_y': vy,
                'confidence': 0.7 + np.random.rand() * 0.3,
                'size': 0.5 + np.random.rand() * 2.0,  # 0.5-2.5m
                'thermal_signature': 30.0 + np.random.rand() * 20.0  # Temperature
            }

            detections.append(detection)

        return detections

    def process_frame(self):
        """Process a single frame"""
        # Simulate detections
        detections = self.simulate_detections()
        self.total_detections += len(detections)

        # Update tracker
        timestamp = time.time()
        tracking_result = self.tracker.update(
            detections,
            frame_number=self.frame_count,
            timestamp=timestamp
        )

        # Update voxel grid with tracked objects
        for track in tracking_result['tracks']:
            position = np.array([track['x'], track['y'],
                               detections[0].get('z', 500.0)])
            self.voxel_grid.add_tracked_object(
                track['track_id'],
                position,
                size=2.0
            )

        self.frame_count += 1

        return tracking_result

    def print_status(self, tracking_result: Dict):
        """Print current tracking status"""
        metrics = tracking_result['metrics']
        tracks = tracking_result['tracks']

        # Calculate uptime
        uptime = time.time() - self.start_time
        fps = self.frame_count / uptime if uptime > 0 else 0

        # Print status
        print("\n" + "="*60)
        print(f"BASIC TRACKING SYSTEM - Frame {self.frame_count}")
        print("="*60)
        print(f"Uptime:           {uptime:.1f}s")
        print(f"FPS:              {fps:.1f}")
        print(f"Active Tracks:    {metrics['num_active_tracks']}")
        print(f"Confirmed Tracks: {metrics['num_confirmed_tracks']}")
        print(f"Detections:       {metrics['num_detections']}")
        print(f"Latency:          {metrics['latency_ms']:.2f} ms")
        print("-"*60)

        # Print first few tracks
        if tracks:
            print("Active Tracks:")
            for track in tracks[:5]:  # Show up to 5 tracks
                print(f"  Track {track['track_id']:3d}: "
                      f"pos=({track['x']:7.1f}, {track['y']:7.1f}) "
                      f"vel=({track['vx']:5.1f}, {track['vy']:5.1f}) "
                      f"conf={track['confidence']:.2f}")
            if len(tracks) > 5:
                print(f"  ... and {len(tracks)-5} more tracks")

        # Camera health
        health = self.camera_manager.get_system_health()
        print("-"*60)
        print(f"Camera Status:    {health['streaming']}/{health['total_cameras']} streaming")
        print(f"Avg Temperature:  {health['avg_temperature']:.1f}°C")
        print("="*60)

    def run(self, duration: float = 30.0, visualize: bool = False):
        """
        Run the tracking system

        Args:
            duration: Run duration in seconds
            visualize: Enable visualization (not implemented in basic example)
        """
        logger.info(f"Running tracking for {duration} seconds...")

        if not self.setup_cameras():
            logger.error("Camera setup failed")
            return

        if not self.start_acquisition():
            logger.error("Failed to start acquisition")
            return

        try:
            # Main tracking loop
            last_print = time.time()
            print_interval = 2.0  # Print status every 2 seconds

            while self.running and (time.time() - self.start_time) < duration:
                # Process frame
                tracking_result = self.process_frame()

                # Print status periodically
                current_time = time.time()
                if current_time - last_print >= print_interval:
                    self.print_status(tracking_result)
                    last_print = current_time

                # Simulate frame rate (30 FPS)
                time.sleep(1.0 / 30.0)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.stop_acquisition()
            self.print_final_statistics()

    def print_final_statistics(self):
        """Print final statistics"""
        uptime = time.time() - self.start_time
        avg_fps = self.frame_count / uptime if uptime > 0 else 0

        tracker_stats = self.tracker.get_performance_summary()
        voxel_stats = self.voxel_grid.get_statistics()

        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Total Runtime:        {uptime:.1f}s")
        print(f"Total Frames:         {self.frame_count}")
        print(f"Average FPS:          {avg_fps:.1f}")
        print(f"Total Detections:     {self.total_detections}")
        print("\nTracking Performance:")
        print(f"  Tracks Created:     {tracker_stats['total_tracks_created']}")
        print(f"  Tracks Confirmed:   {tracker_stats['total_tracks_confirmed']}")
        print(f"  Tracks Lost:        {tracker_stats['total_tracks_lost']}")
        print(f"  Avg Track Length:   {tracker_stats['avg_track_length']:.1f} frames")
        print(f"  Avg Latency:        {tracker_stats['avg_latency_ms']:.2f} ms")
        print("\nVoxel Grid:")
        print(f"  Tracked Objects:    {voxel_stats['tracked_objects']}")
        print(f"  Memory Usage:       {voxel_stats['memory_usage_mb']:.1f} MB")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Basic single-pair motion tracking example'
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Camera pair ID (default: 0)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='Run duration in seconds (default: 30)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable 3D visualization (requires matplotlib)'
    )
    parser.add_argument(
        '--save-output',
        type=str,
        help='Save tracking data to file'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*60)
    print("8K MOTION TRACKING SYSTEM - Basic Tracking Example")
    print("="*60)
    print(f"Camera Pair ID: {args.camera_id}")
    print(f"Duration:       {args.duration}s")
    print(f"Visualization:  {'Enabled' if args.visualize else 'Disabled'}")
    print("="*60 + "\n")

    # Create and run tracking system
    system = BasicTrackingSystem(camera_pair_id=args.camera_id)
    system.run(duration=args.duration, visualize=args.visualize)

    logger.info("Basic tracking example complete!")


if __name__ == "__main__":
    main()
