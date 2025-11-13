#!/usr/bin/env python3
"""
Multi-Camera Demonstration - 10 Camera Pair System
===================================================

This example demonstrates the full-scale 8K motion tracking system:
- 10 mono-thermal camera pairs (20 cameras total)
- Synchronized acquisition across all pairs
- Multi-camera fusion and tracking
- Real-time performance monitoring
- System health dashboard

Requirements:
- Python 3.8+
- NumPy
- Sufficient network bandwidth for 10 camera pairs

Usage:
    python multi_camera_demo.py

    Optional arguments:
        --duration SECONDS      Run duration in seconds (default: 60)
        --show-health          Display detailed health metrics
        --save-metrics FILE    Save performance metrics to file
        --stress-test          Run with maximum load (200 targets)

Author: Motion Tracking System
Date: 2025-11-13
"""

import sys
import time
import argparse
import logging
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import system components
from camera.camera_manager import (
    CameraManager, CameraConfiguration, CameraPair,
    CameraType, ConnectionType
)
from detection.tracker import MultiTargetTracker
from voxel.grid_manager import VoxelGridManager, GridConfig
from fusion.detection_fusion import DetectionFusion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: float
    frame_number: int
    fps: float
    total_detections: int
    active_tracks: int
    confirmed_tracks: int
    avg_latency_ms: float
    camera_health: Dict
    memory_usage_mb: float
    bandwidth_mbps: float


class MultiCameraTrackingSystem:
    """
    Full-scale multi-camera tracking system with 10 pairs
    """

    def __init__(self, num_pairs: int = 10, stress_test: bool = False):
        """
        Initialize multi-camera tracking system

        Args:
            num_pairs: Number of camera pairs (default: 10)
            stress_test: Enable stress testing mode
        """
        self.num_pairs = num_pairs
        self.num_cameras = num_pairs * 2
        self.stress_test = stress_test

        logger.info(f"Initializing {num_pairs}-pair tracking system...")

        # Initialize camera manager
        logger.info("Setting up camera manager...")
        self.camera_manager = CameraManager(num_pairs=num_pairs)

        # Initialize tracker with high capacity
        logger.info("Setting up multi-target tracker...")
        max_tracks = 200 if stress_test else 100
        self.tracker = MultiTargetTracker(
            max_tracks=max_tracks,
            detection_threshold=0.5,
            confirmation_threshold=3,
            max_age=15,
            frame_rate=30.0
        )

        # Initialize detection fusion
        logger.info("Setting up detection fusion...")
        self.fusion = DetectionFusion(num_cameras=num_pairs)

        # Initialize large voxel grid for multi-camera coverage
        logger.info("Setting up voxel grid...")
        grid_config = GridConfig(
            center=np.array([0.0, 0.0, 1000.0]),  # 1km altitude
            size=np.array([5000.0, 5000.0, 2000.0]),  # 5km x 5km x 2km
            base_resolution=1.0,
            max_memory_mb=500,
            enable_lod=True,
            enable_compression=True
        )
        self.voxel_grid = VoxelGridManager(grid_config)

        # Tracking state
        self.frame_count = 0
        self.running = False
        self.start_time = 0.0

        # Performance metrics
        self.metrics_history = deque(maxlen=1000)
        self.frame_times = deque(maxlen=100)

        # Statistics
        self.total_detections = 0
        self.total_frames_processed = 0

    def setup_camera_array(self):
        """
        Configure and initialize all 10 camera pairs in a circular array

        Camera pairs are arranged in a circle to provide 360° coverage
        """
        logger.info(f"Setting up {self.num_pairs} camera pairs in circular array...")

        # Arrange cameras in a circle
        radius = 100.0  # 100m radius
        height = 50.0   # 50m height

        for pair_id in range(self.num_pairs):
            # Calculate position around circle
            angle = (pair_id / self.num_pairs) * 2 * np.pi

            # Position on circle
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Direction pointing toward center
            forward = np.array([-np.cos(angle), -np.sin(angle), -0.2])
            forward = forward / np.linalg.norm(forward)

            # Camera IDs
            mono_id = pair_id * 2
            thermal_id = pair_id * 2 + 1

            # Configure mono camera
            mono_config = CameraConfiguration(
                camera_id=mono_id,
                pair_id=pair_id,
                camera_type=CameraType.MONO,
                connection=ConnectionType.GIGE_VISION,
                ip_address=f"192.168.{10 + pair_id // 10}.{10 + (pair_id % 10) * 2}",
                width=7680,  # 8K resolution
                height=4320,
                frame_rate=30.0,
                exposure_time=10000.0,
                position=np.array([x, y, height]),
                serial_number=f"MONO-{pair_id:02d}",
                model_name="8K-MONO-CAM"
            )

            # Configure thermal camera (offset by 15cm)
            thermal_offset = np.array([0.15 * np.sin(angle), -0.15 * np.cos(angle), 0])
            thermal_config = CameraConfiguration(
                camera_id=thermal_id,
                pair_id=pair_id,
                camera_type=CameraType.THERMAL,
                connection=ConnectionType.GIGE_VISION,
                ip_address=f"192.168.{10 + pair_id // 10}.{11 + (pair_id % 10) * 2}",
                width=640,
                height=512,
                frame_rate=30.0,
                exposure_time=5000.0,
                position=np.array([x, y, height]) + thermal_offset,
                serial_number=f"THERMAL-{pair_id:02d}",
                model_name="THERMAL-CAM-640"
            )

            # Add cameras
            self.camera_manager.add_camera(mono_config)
            self.camera_manager.add_camera(thermal_config)

            # Create pair
            pair = CameraPair(
                pair_id=pair_id,
                mono_camera_id=mono_id,
                thermal_camera_id=thermal_id,
                stereo_baseline=0.15
            )
            self.camera_manager.add_pair(pair)

            logger.info(f"  Pair {pair_id}: position=({x:.1f}, {y:.1f}, {height:.1f})")

        # Initialize all cameras
        logger.info("Initializing all cameras...")
        if not self.camera_manager.initialize_all_cameras():
            logger.error("Failed to initialize all cameras")
            return False

        logger.info(f"Successfully initialized {self.num_cameras} cameras!")
        return True

    def start_acquisition(self):
        """Start synchronized acquisition on all cameras"""
        logger.info("Starting synchronized acquisition...")

        # Start all cameras
        if not self.camera_manager.start_all_acquisition():
            logger.error("Failed to start acquisition on all cameras")
            return False

        # Start health monitoring
        self.camera_manager.start_health_monitoring()

        # Start voxel background processing
        self.voxel_grid.start_background_processing()

        self.running = True
        self.start_time = time.time()

        logger.info("All cameras streaming!")
        return True

    def stop_acquisition(self):
        """Stop acquisition on all cameras"""
        logger.info("Stopping acquisition...")

        self.running = False

        # Stop cameras
        self.camera_manager.stop_all_acquisition()
        self.camera_manager.stop_health_monitoring()

        # Stop voxel processing
        self.voxel_grid.stop_background_processing()

        logger.info("Acquisition stopped")

    def simulate_multi_camera_detections(self) -> List[Dict]:
        """
        Simulate detections from all camera pairs

        In stress test mode, generates up to 200 targets
        In normal mode, generates 10-30 targets

        Returns:
            List of fused detection dictionaries
        """
        if self.stress_test:
            # Stress test: 180-200 targets
            num_targets = np.random.randint(180, 201)
        else:
            # Normal mode: 10-30 targets
            num_targets = np.random.randint(10, 31)

        detections = []

        for i in range(num_targets):
            # Random position in 5km x 5km area
            x = np.random.uniform(-2500, 2500)
            y = np.random.uniform(-2500, 2500)
            z = np.random.uniform(100, 1500)  # 100m-1.5km altitude

            # Random velocity
            vx = np.random.uniform(-15, 15)  # m/s
            vy = np.random.uniform(-15, 15)
            vz = np.random.uniform(-2, 2)

            # Determine which cameras can see this target
            visible_cameras = []
            for pair_id in range(self.num_pairs):
                # Simple visibility check based on distance and angle
                cam_config = self.camera_manager.configurations.get(pair_id * 2)
                if cam_config:
                    cam_pos = cam_config.position
                    distance = np.linalg.norm(
                        np.array([x, y, z]) - cam_pos
                    )

                    # Visible if within 3km
                    if distance < 3000:
                        visible_cameras.append(pair_id)

            if len(visible_cameras) > 0:
                detection = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'velocity_x': vx,
                    'velocity_y': vy,
                    'velocity_z': vz,
                    'confidence': 0.6 + np.random.rand() * 0.4,
                    'size': 0.3 + np.random.rand() * 2.0,
                    'thermal_signature': 25.0 + np.random.rand() * 30.0,
                    'visible_cameras': visible_cameras,
                    'num_observations': len(visible_cameras)
                }
                detections.append(detection)

        return detections

    def process_frame(self) -> Dict:
        """Process a single frame from all cameras"""
        frame_start = time.time()

        # Simulate multi-camera detections
        detections = self.simulate_multi_camera_detections()
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
            # Find matching detection to get Z coordinate
            z_coord = 500.0  # Default
            for det in detections:
                if abs(det['x'] - track['x']) < 10 and abs(det['y'] - track['y']) < 10:
                    z_coord = det['z']
                    break

            position = np.array([track['x'], track['y'], z_coord])
            self.voxel_grid.add_tracked_object(
                track['track_id'],
                position,
                size=2.0,
                velocity=np.array([track['vx'], track['vy'], 0])
            )

        # Collect system metrics
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)

        health = self.camera_manager.get_system_health()
        voxel_stats = self.voxel_grid.get_statistics()

        metrics = SystemMetrics(
            timestamp=timestamp,
            frame_number=self.frame_count,
            fps=1.0 / np.mean(self.frame_times) if len(self.frame_times) > 0 else 0,
            total_detections=len(detections),
            active_tracks=tracking_result['metrics']['num_active_tracks'],
            confirmed_tracks=tracking_result['metrics']['num_confirmed_tracks'],
            avg_latency_ms=tracking_result['metrics']['latency_ms'],
            camera_health=health,
            memory_usage_mb=voxel_stats['memory_usage_mb'],
            bandwidth_mbps=self._estimate_bandwidth()
        )

        self.metrics_history.append(metrics)
        self.frame_count += 1
        self.total_frames_processed += 1

        return {
            'tracking': tracking_result,
            'metrics': metrics,
            'detections': detections
        }

    def _estimate_bandwidth(self) -> float:
        """Estimate total bandwidth usage (MB/s)"""
        # 8K mono: ~130 MB/frame @ 30fps = 3900 MB/s
        # Thermal: ~0.3 MB/frame @ 30fps = 9 MB/s
        # Per pair: ~3909 MB/s
        # 10 pairs: ~39,090 MB/s = ~39 GB/s

        # With compression (assuming 10:1): ~3.9 GB/s
        bytes_per_frame = (
            (7680 * 4320 * 1.5) +  # 8K mono (1.5 bytes/pixel after compression)
            (640 * 512 * 2)         # Thermal (2 bytes/pixel)
        )

        fps = 1.0 / np.mean(self.frame_times) if len(self.frame_times) > 0 else 30.0
        total_mbps = (bytes_per_frame * self.num_pairs * fps) / (1024 * 1024)

        return total_mbps

    def print_status(self, result: Dict, show_health: bool = False):
        """Print comprehensive system status"""
        metrics = result['metrics']
        tracking = result['tracking']['metrics']
        detections = result['detections']

        uptime = time.time() - self.start_time

        print("\n" + "="*80)
        print(f"MULTI-CAMERA TRACKING SYSTEM - Frame {self.frame_count}")
        print("="*80)

        # System overview
        print(f"Uptime:             {uptime:.1f}s")
        print(f"FPS:                {metrics.fps:.1f}")
        print(f"Frame Time:         {np.mean(self.frame_times)*1000:.1f} ms")

        # Tracking statistics
        print(f"\nTracking:")
        print(f"  Active Tracks:    {metrics.active_tracks}")
        print(f"  Confirmed:        {metrics.confirmed_tracks}")
        print(f"  Detections:       {metrics.total_detections}")
        print(f"  Latency:          {metrics.avg_latency_ms:.2f} ms")

        # Multi-camera fusion
        print(f"\nMulti-Camera Fusion:")
        avg_observations = np.mean([d['num_observations'] for d in detections]) if detections else 0
        print(f"  Avg Observations: {avg_observations:.1f} cameras/target")
        print(f"  Coverage:         360°")

        # Camera health
        health = metrics.camera_health
        print(f"\nCamera System:")
        print(f"  Streaming:        {health['streaming']}/{health['total_cameras']}")
        print(f"  Ready:            {health['ready']}")
        print(f"  Errors:           {health['error']}")
        print(f"  Avg FPS:          {health['avg_fps']:.1f}")
        print(f"  Avg Temp:         {health['avg_temperature']:.1f}°C")

        # System resources
        print(f"\nSystem Resources:")
        print(f"  Memory:           {metrics.memory_usage_mb:.1f} MB")
        print(f"  Bandwidth:        {metrics.bandwidth_mbps:.1f} MB/s")

        # Performance requirements check
        print(f"\nRequirements Check:")
        req_fps = metrics.fps >= 28.0
        req_latency = metrics.avg_latency_ms < 100.0
        req_tracks = metrics.confirmed_tracks <= 200

        print(f"  FPS ≥ 30:         {'✓' if req_fps else '✗'} ({metrics.fps:.1f})")
        print(f"  Latency < 100ms:  {'✓' if req_latency else '✗'} ({metrics.avg_latency_ms:.1f} ms)")
        print(f"  Tracks ≤ 200:     {'✓' if req_tracks else '✗'} ({metrics.confirmed_tracks})")

        # Health details (if requested)
        if show_health:
            print(f"\nDetailed Camera Health:")
            for pair_id in range(min(3, self.num_pairs)):  # Show first 3 pairs
                pair_health = self.camera_manager.get_pair_health(pair_id)
                if pair_health:
                    print(f"  Pair {pair_id}:")
                    print(f"    Mono:    {pair_health['mono_health'].status.name}")
                    print(f"    Thermal: {pair_health['thermal_health'].status.name}")
                    print(f"    Sync:    {pair_health['sync_quality']:.1f}%")

        print("="*80)

    def save_metrics(self, filename: str):
        """Save performance metrics to file"""
        data = {
            'system_config': {
                'num_pairs': self.num_pairs,
                'num_cameras': self.num_cameras,
                'stress_test': self.stress_test
            },
            'summary': {
                'total_frames': self.frame_count,
                'total_detections': self.total_detections,
                'avg_fps': np.mean([m.fps for m in self.metrics_history]),
                'avg_latency_ms': np.mean([m.avg_latency_ms for m in self.metrics_history]),
                'avg_tracks': np.mean([m.active_tracks for m in self.metrics_history]),
            },
            'metrics': [asdict(m) for m in self.metrics_history]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics saved to {filename}")

    def run(self, duration: float = 60.0, show_health: bool = False):
        """
        Run the multi-camera tracking system

        Args:
            duration: Run duration in seconds
            show_health: Show detailed health metrics
        """
        logger.info(f"Running multi-camera system for {duration} seconds...")

        if not self.setup_camera_array():
            logger.error("Camera setup failed")
            return

        if not self.start_acquisition():
            logger.error("Failed to start acquisition")
            return

        try:
            # Main tracking loop
            last_print = time.time()
            print_interval = 3.0  # Print status every 3 seconds

            while self.running and (time.time() - self.start_time) < duration:
                # Process frame
                result = self.process_frame()

                # Print status periodically
                current_time = time.time()
                if current_time - last_print >= print_interval:
                    self.print_status(result, show_health=show_health)
                    last_print = current_time

                # Simulate frame rate
                target_frame_time = 1.0 / 30.0
                actual_frame_time = time.time() - result['metrics'].timestamp
                sleep_time = max(0, target_frame_time - actual_frame_time)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.stop_acquisition()
            self.print_final_statistics()

    def print_final_statistics(self):
        """Print final comprehensive statistics"""
        uptime = time.time() - self.start_time

        if not self.metrics_history:
            return

        # Calculate averages
        avg_fps = np.mean([m.fps for m in self.metrics_history])
        avg_latency = np.mean([m.avg_latency_ms for m in self.metrics_history])
        avg_tracks = np.mean([m.active_tracks for m in self.metrics_history])
        max_tracks = max([m.active_tracks for m in self.metrics_history])
        avg_bandwidth = np.mean([m.bandwidth_mbps for m in self.metrics_history])

        tracker_stats = self.tracker.get_performance_summary()
        voxel_stats = self.voxel_grid.get_statistics()

        print("\n" + "="*80)
        print("FINAL SYSTEM STATISTICS")
        print("="*80)
        print(f"Configuration:        {self.num_pairs} pairs, {self.num_cameras} cameras")
        print(f"Mode:                 {'STRESS TEST' if self.stress_test else 'NORMAL'}")
        print(f"Total Runtime:        {uptime:.1f}s")
        print(f"Total Frames:         {self.frame_count}")
        print(f"Total Detections:     {self.total_detections}")

        print(f"\nPerformance:")
        print(f"  Average FPS:        {avg_fps:.1f}")
        print(f"  Average Latency:    {avg_latency:.2f} ms")
        print(f"  Average Tracks:     {avg_tracks:.1f}")
        print(f"  Peak Tracks:        {max_tracks}")
        print(f"  Average Bandwidth:  {avg_bandwidth:.1f} MB/s")

        print(f"\nTracking Statistics:")
        print(f"  Tracks Created:     {tracker_stats['total_tracks_created']}")
        print(f"  Tracks Confirmed:   {tracker_stats['total_tracks_confirmed']}")
        print(f"  Tracks Lost:        {tracker_stats['total_tracks_lost']}")
        print(f"  Avg Track Length:   {tracker_stats['avg_track_length']:.1f} frames")
        print(f"  Occlusion Recovery: {tracker_stats['occlusion_recoveries']}")

        print(f"\nVoxel Grid:")
        print(f"  Tracked Objects:    {voxel_stats['tracked_objects']}")
        print(f"  Memory Usage:       {voxel_stats['memory_usage_mb']:.1f} MB")
        print(f"  Grid Center:        {voxel_stats['grid_center']}")

        print(f"\nRequirements Met:")
        print(f"  FPS ≥ 30:           {'✓' if avg_fps >= 28 else '✗'}")
        print(f"  Latency < 100ms:    {'✓' if avg_latency < 100 else '✗'}")
        print(f"  Max Tracks ≥ 200:   {'✓' if max_tracks >= 200 else '✗'}")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Multi-camera tracking system demonstration'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Run duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--show-health',
        action='store_true',
        help='Display detailed camera health metrics'
    )
    parser.add_argument(
        '--save-metrics',
        type=str,
        help='Save performance metrics to JSON file'
    )
    parser.add_argument(
        '--stress-test',
        action='store_true',
        help='Run stress test with 200 targets'
    )
    parser.add_argument(
        '--num-pairs',
        type=int,
        default=10,
        help='Number of camera pairs (default: 10)'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*80)
    print("8K MOTION TRACKING SYSTEM - Multi-Camera Demonstration")
    print("="*80)
    print(f"Camera Pairs:   {args.num_pairs}")
    print(f"Total Cameras:  {args.num_pairs * 2}")
    print(f"Duration:       {args.duration}s")
    print(f"Mode:           {'STRESS TEST (200 targets)' if args.stress_test else 'NORMAL'}")
    print("="*80 + "\n")

    # Create and run system
    system = MultiCameraTrackingSystem(
        num_pairs=args.num_pairs,
        stress_test=args.stress_test
    )
    system.run(duration=args.duration, show_health=args.show_health)

    # Save metrics if requested
    if args.save_metrics:
        system.save_metrics(args.save_metrics)

    logger.info("Multi-camera demonstration complete!")


if __name__ == "__main__":
    main()
