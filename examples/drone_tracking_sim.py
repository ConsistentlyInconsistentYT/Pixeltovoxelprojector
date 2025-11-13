#!/usr/bin/env python3
"""
Drone Tracking Simulation - 200 Drone Scenario
==============================================

This example simulates a complex scenario with 200 drones:
- Realistic drone trajectories (linear, circular, hovering)
- Physics-based motion simulation
- Detection accuracy modeling
- Tracking performance analysis
- Visualization of detection and tracking

Perfect for testing system capacity and accuracy analysis.

Requirements:
- Python 3.8+
- NumPy
- Matplotlib (optional, for visualization)

Usage:
    python drone_tracking_sim.py

    Optional arguments:
        --num-drones NUM        Number of drones (default: 200)
        --duration SECONDS      Simulation duration (default: 120)
        --visualize            Enable 2D/3D visualization
        --save-analysis FILE    Save accuracy analysis to file
        --trajectory-mix STR    Trajectory mix: balanced, aggressive, calm

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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import system components
from camera.camera_manager import CameraManager, CameraConfiguration, CameraPair, CameraType, ConnectionType
from detection.tracker import MultiTargetTracker
from voxel.grid_manager import VoxelGridManager, GridConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrajectoryType(Enum):
    """Drone trajectory types"""
    LINEAR = "linear"          # Straight line flight
    CIRCULAR = "circular"      # Circular pattern
    HOVER = "hover"           # Hovering with small movements
    ZIGZAG = "zigzag"         # Zigzag pattern
    SPIRAL = "spiral"         # Spiral ascent/descent
    EVASIVE = "evasive"       # Evasive maneuvers


@dataclass
class DroneState:
    """Physical state of a simulated drone"""
    drone_id: int
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Physical properties
    size: float = 0.5  # meters (typical quadcopter)
    max_speed: float = 20.0  # m/s
    max_acceleration: float = 5.0  # m/s^2

    # Trajectory
    trajectory_type: TrajectoryType = TrajectoryType.LINEAR
    trajectory_params: Dict = field(default_factory=dict)

    # Track state
    is_tracked: bool = False
    track_id: Optional[int] = None
    detection_probability: float = 0.95


class DroneSimulator:
    """
    Simulates realistic drone physics and trajectories
    """

    def __init__(self, num_drones: int = 200, trajectory_mix: str = "balanced"):
        """
        Initialize drone simulator

        Args:
            num_drones: Number of drones to simulate
            trajectory_mix: Trajectory distribution (balanced, aggressive, calm)
        """
        self.num_drones = num_drones
        self.drones: Dict[int, DroneState] = {}
        self.dt = 1.0 / 30.0  # 30 Hz simulation

        # Initialize drones with different trajectories
        self._initialize_drones(trajectory_mix)

        logger.info(f"Initialized {num_drones} drones with {trajectory_mix} trajectory mix")

    def _initialize_drones(self, mix: str):
        """Initialize drones with various trajectories"""
        # Determine trajectory distribution based on mix
        if mix == "aggressive":
            traj_dist = {
                TrajectoryType.LINEAR: 0.2,
                TrajectoryType.CIRCULAR: 0.15,
                TrajectoryType.HOVER: 0.05,
                TrajectoryType.ZIGZAG: 0.25,
                TrajectoryType.SPIRAL: 0.15,
                TrajectoryType.EVASIVE: 0.2
            }
        elif mix == "calm":
            traj_dist = {
                TrajectoryType.LINEAR: 0.4,
                TrajectoryType.CIRCULAR: 0.2,
                TrajectoryType.HOVER: 0.3,
                TrajectoryType.ZIGZAG: 0.05,
                TrajectoryType.SPIRAL: 0.05,
                TrajectoryType.EVASIVE: 0.0
            }
        else:  # balanced
            traj_dist = {
                TrajectoryType.LINEAR: 0.3,
                TrajectoryType.CIRCULAR: 0.25,
                TrajectoryType.HOVER: 0.15,
                TrajectoryType.ZIGZAG: 0.15,
                TrajectoryType.SPIRAL: 0.1,
                TrajectoryType.EVASIVE: 0.05
            }

        # Create drones
        for i in range(self.num_drones):
            # Random initial position in 5km x 5km area
            x = np.random.uniform(-2500, 2500)
            y = np.random.uniform(-2500, 2500)
            z = np.random.uniform(50, 1500)  # 50m to 1.5km altitude

            position = np.array([x, y, z])

            # Select trajectory type based on distribution
            traj_type = np.random.choice(
                list(traj_dist.keys()),
                p=list(traj_dist.values())
            )

            # Generate trajectory parameters
            traj_params = self._generate_trajectory_params(traj_type, position)

            # Initial velocity based on trajectory
            velocity = self._compute_initial_velocity(traj_type, traj_params)

            # Create drone
            drone = DroneState(
                drone_id=i,
                position=position,
                velocity=velocity,
                size=0.3 + np.random.rand() * 0.4,  # 0.3-0.7m
                max_speed=15.0 + np.random.rand() * 10.0,  # 15-25 m/s
                trajectory_type=traj_type,
                trajectory_params=traj_params,
                detection_probability=0.90 + np.random.rand() * 0.09  # 0.90-0.99
            )

            self.drones[i] = drone

    def _generate_trajectory_params(self, traj_type: TrajectoryType,
                                    position: np.ndarray) -> Dict:
        """Generate parameters for trajectory type"""
        params = {}

        if traj_type == TrajectoryType.LINEAR:
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            params['direction'] = np.array([np.cos(angle), np.sin(angle), 0])
            params['speed'] = 10.0 + np.random.rand() * 10.0

        elif traj_type == TrajectoryType.CIRCULAR:
            # Circle center and radius
            params['center'] = position.copy()
            params['center'][2] = position[2]  # Keep altitude
            params['radius'] = 50.0 + np.random.rand() * 150.0
            params['angular_velocity'] = 0.1 + np.random.rand() * 0.3  # rad/s
            params['phase'] = np.random.uniform(0, 2 * np.pi)

        elif traj_type == TrajectoryType.HOVER:
            # Hover position with small drift
            params['hover_center'] = position.copy()
            params['drift_range'] = 5.0 + np.random.rand() * 10.0  # meters

        elif traj_type == TrajectoryType.ZIGZAG:
            angle = np.random.uniform(0, 2 * np.pi)
            params['base_direction'] = np.array([np.cos(angle), np.sin(angle), 0])
            params['zigzag_amplitude'] = 20.0 + np.random.rand() * 30.0
            params['zigzag_frequency'] = 0.1 + np.random.rand() * 0.2

        elif traj_type == TrajectoryType.SPIRAL:
            params['center'] = position.copy()
            params['radius'] = 30.0 + np.random.rand() * 70.0
            params['angular_velocity'] = 0.2 + np.random.rand() * 0.3
            params['vertical_speed'] = 2.0 + np.random.rand() * 3.0
            params['phase'] = np.random.uniform(0, 2 * np.pi)
            params['ascending'] = np.random.choice([True, False])

        elif traj_type == TrajectoryType.EVASIVE:
            angle = np.random.uniform(0, 2 * np.pi)
            params['base_direction'] = np.array([np.cos(angle), np.sin(angle), 0])
            params['maneuver_frequency'] = 0.3 + np.random.rand() * 0.5
            params['maneuver_magnitude'] = 5.0 + np.random.rand() * 10.0

        return params

    def _compute_initial_velocity(self, traj_type: TrajectoryType,
                                  params: Dict) -> np.ndarray:
        """Compute initial velocity for trajectory"""
        if traj_type == TrajectoryType.LINEAR:
            return params['direction'] * params['speed']

        elif traj_type == TrajectoryType.CIRCULAR:
            # Tangential velocity
            speed = params['angular_velocity'] * params['radius']
            angle = params['phase']
            return np.array([-np.sin(angle), np.cos(angle), 0]) * speed

        elif traj_type == TrajectoryType.HOVER:
            return np.random.randn(3) * 0.5  # Small random drift

        elif traj_type == TrajectoryType.ZIGZAG:
            return params['base_direction'] * 15.0

        elif traj_type == TrajectoryType.SPIRAL:
            speed = params['angular_velocity'] * params['radius']
            angle = params['phase']
            vz = params['vertical_speed'] if params['ascending'] else -params['vertical_speed']
            return np.array([-np.sin(angle), np.cos(angle), 0]) * speed + np.array([0, 0, vz])

        elif traj_type == TrajectoryType.EVASIVE:
            return params['base_direction'] * 18.0

        return np.zeros(3)

    def update(self, current_time: float):
        """Update all drone positions"""
        for drone in self.drones.values():
            # Compute desired velocity based on trajectory
            desired_velocity = self._compute_trajectory_velocity(drone, current_time)

            # Compute acceleration to reach desired velocity
            velocity_error = desired_velocity - drone.velocity
            drone.acceleration = np.clip(
                velocity_error / self.dt,
                -drone.max_acceleration,
                drone.max_acceleration
            )

            # Update velocity
            drone.velocity += drone.acceleration * self.dt

            # Limit velocity
            speed = np.linalg.norm(drone.velocity)
            if speed > drone.max_speed:
                drone.velocity = (drone.velocity / speed) * drone.max_speed

            # Update position
            drone.position += drone.velocity * self.dt

            # Keep drones within bounds
            self._apply_boundary_constraints(drone)

    def _compute_trajectory_velocity(self, drone: DroneState,
                                     current_time: float) -> np.ndarray:
        """Compute desired velocity for drone trajectory"""
        params = drone.trajectory_params
        traj_type = drone.trajectory_type

        if traj_type == TrajectoryType.LINEAR:
            return params['direction'] * params['speed']

        elif traj_type == TrajectoryType.CIRCULAR:
            # Current phase
            phase = params['phase'] + params['angular_velocity'] * current_time

            # Tangential velocity
            speed = params['angular_velocity'] * params['radius']
            return np.array([-np.sin(phase), np.cos(phase), 0]) * speed

        elif traj_type == TrajectoryType.HOVER:
            # Drift toward hover center
            to_center = params['hover_center'] - drone.position
            distance = np.linalg.norm(to_center)

            if distance > params['drift_range']:
                return to_center / distance * 2.0
            else:
                # Small random drift
                return np.random.randn(3) * 0.5

        elif traj_type == TrajectoryType.ZIGZAG:
            # Base velocity with perpendicular oscillation
            base_vel = params['base_direction'] * 15.0
            perpendicular = np.array([-params['base_direction'][1],
                                     params['base_direction'][0], 0])

            oscillation = perpendicular * params['zigzag_amplitude'] * \
                         np.sin(params['zigzag_frequency'] * current_time)

            return base_vel + oscillation * 0.1

        elif traj_type == TrajectoryType.SPIRAL:
            phase = params['phase'] + params['angular_velocity'] * current_time

            # Tangential + vertical
            speed = params['angular_velocity'] * params['radius']
            vz = params['vertical_speed'] if params['ascending'] else -params['vertical_speed']

            return np.array([-np.sin(phase), np.cos(phase), 0]) * speed + \
                   np.array([0, 0, vz])

        elif traj_type == TrajectoryType.EVASIVE:
            # Base velocity with random maneuvers
            base_vel = params['base_direction'] * 18.0

            # Random maneuvers
            maneuver = np.random.randn(3) * params['maneuver_magnitude'] * \
                      np.sin(params['maneuver_frequency'] * current_time * 2 * np.pi)

            return base_vel + maneuver

        return np.zeros(3)

    def _apply_boundary_constraints(self, drone: DroneState):
        """Keep drones within simulation bounds"""
        # X, Y bounds: ±3000m
        if abs(drone.position[0]) > 3000:
            drone.position[0] = np.sign(drone.position[0]) * 2900
            drone.velocity[0] *= -0.5  # Bounce back

        if abs(drone.position[1]) > 3000:
            drone.position[1] = np.sign(drone.position[1]) * 2900
            drone.velocity[1] *= -0.5

        # Z bounds: 50m to 2000m
        if drone.position[2] < 50:
            drone.position[2] = 50
            drone.velocity[2] = abs(drone.velocity[2])

        if drone.position[2] > 2000:
            drone.position[2] = 2000
            drone.velocity[2] = -abs(drone.velocity[2])

    def generate_detections(self) -> List[Dict]:
        """
        Generate detections with realistic detection probability and noise
        """
        detections = []

        for drone in self.drones.values():
            # Check if detected (based on probability)
            if np.random.rand() > drone.detection_probability:
                continue  # Missed detection

            # Add measurement noise
            position_noise = np.random.randn(3) * 0.5  # 0.5m standard deviation
            velocity_noise = np.random.randn(3) * 0.2  # 0.2 m/s

            noisy_position = drone.position + position_noise
            noisy_velocity = drone.velocity + velocity_noise

            detection = {
                'x': noisy_position[0],
                'y': noisy_position[1],
                'z': noisy_position[2],
                'velocity_x': noisy_velocity[0],
                'velocity_y': noisy_velocity[1],
                'velocity_z': noisy_velocity[2],
                'confidence': drone.detection_probability + np.random.randn() * 0.02,
                'size': drone.size,
                'true_id': drone.drone_id  # Ground truth for analysis
            }

            detections.append(detection)

        return detections

    def get_ground_truth(self) -> Dict[int, np.ndarray]:
        """Get ground truth positions for accuracy analysis"""
        return {
            drone_id: drone.position.copy()
            for drone_id, drone in self.drones.items()
        }


class DroneTrackingSimulation:
    """
    Complete simulation system for drone tracking
    """

    def __init__(self, num_drones: int = 200, trajectory_mix: str = "balanced"):
        """Initialize simulation"""
        logger.info(f"Initializing drone tracking simulation ({num_drones} drones)...")

        # Drone simulator
        self.simulator = DroneSimulator(num_drones, trajectory_mix)

        # Tracker
        self.tracker = MultiTargetTracker(
            max_tracks=250,  # Extra capacity
            detection_threshold=0.5,
            confirmation_threshold=3,
            max_age=15,
            frame_rate=30.0
        )

        # Voxel grid for visualization
        grid_config = GridConfig(
            center=np.array([0.0, 0.0, 1000.0]),
            size=np.array([6000.0, 6000.0, 2000.0]),
            base_resolution=2.0,  # 2m resolution for large area
            max_memory_mb=300,
            enable_lod=True
        )
        self.voxel_grid = VoxelGridManager(grid_config)

        # Simulation state
        self.frame_count = 0
        self.sim_time = 0.0
        self.start_time = 0.0
        self.running = False

        # Accuracy analysis
        self.position_errors = []
        self.velocity_errors = []
        self.detection_rates = []
        self.false_positive_counts = []

    def run(self, duration: float = 120.0, visualize: bool = False):
        """
        Run the drone tracking simulation

        Args:
            duration: Simulation duration in seconds
            visualize: Enable visualization (requires matplotlib)
        """
        logger.info(f"Running simulation for {duration} seconds...")

        self.running = True
        self.start_time = time.time()
        self.voxel_grid.start_background_processing()

        try:
            last_print = time.time()
            print_interval = 5.0

            while self.running and self.sim_time < duration:
                # Update simulation
                self.simulator.update(self.sim_time)

                # Generate detections
                detections = self.simulator.generate_detections()

                # Update tracker
                tracking_result = self.tracker.update(
                    detections,
                    frame_number=self.frame_count,
                    timestamp=time.time()
                )

                # Analyze accuracy
                self._analyze_accuracy(
                    tracking_result['tracks'],
                    self.simulator.get_ground_truth(),
                    detections
                )

                # Update voxel grid
                for track in tracking_result['tracks']:
                    position = np.array([track['x'], track['y'],
                                       detections[0].get('z', 500) if detections else 500])
                    self.voxel_grid.add_tracked_object(track['track_id'], position)

                # Print status
                if time.time() - last_print >= print_interval:
                    self._print_status(tracking_result, detections)
                    last_print = time.time()

                # Advance simulation
                self.sim_time += 1.0 / 30.0
                self.frame_count += 1

                # Real-time execution
                time.sleep(1.0 / 30.0)

        except KeyboardInterrupt:
            logger.info("Simulation interrupted")

        finally:
            self.running = False
            self.voxel_grid.stop_background_processing()
            self._print_final_analysis()

    def _analyze_accuracy(self, tracks: List[Dict], ground_truth: Dict,
                         detections: List[Dict]):
        """Analyze tracking accuracy against ground truth"""
        if not tracks:
            return

        # Match tracks to ground truth
        position_errors = []
        velocity_errors = []

        for track in tracks:
            track_pos = np.array([track['x'], track['y'], 0])  # 2D for now

            # Find closest ground truth
            min_distance = float('inf')
            closest_gt = None

            for gt_id, gt_pos in ground_truth.items():
                distance = np.linalg.norm(track_pos[:2] - gt_pos[:2])
                if distance < min_distance:
                    min_distance = distance
                    closest_gt = gt_id

            if min_distance < 50.0:  # Within 50m considered a match
                position_errors.append(min_distance)

                # Velocity error
                drone = self.simulator.drones[closest_gt]
                track_vel = np.array([track['vx'], track['vy']])
                true_vel = drone.velocity[:2]
                vel_error = np.linalg.norm(track_vel - true_vel)
                velocity_errors.append(vel_error)

        if position_errors:
            self.position_errors.append(np.mean(position_errors))
            self.velocity_errors.append(np.mean(velocity_errors))

        # Detection rate
        detection_rate = len(detections) / len(ground_truth)
        self.detection_rates.append(detection_rate)

    def _print_status(self, tracking_result: Dict, detections: List[Dict]):
        """Print simulation status"""
        metrics = tracking_result['metrics']

        print("\n" + "="*70)
        print(f"DRONE TRACKING SIMULATION - Frame {self.frame_count}")
        print("="*70)
        print(f"Sim Time:         {self.sim_time:.1f}s")
        print(f"Drones:           {len(self.simulator.drones)}")
        print(f"Detections:       {len(detections)}")
        print(f"Active Tracks:    {metrics['num_active_tracks']}")
        print(f"Confirmed:        {metrics['num_confirmed_tracks']}")
        print(f"Latency:          {metrics['latency_ms']:.2f} ms")

        if self.position_errors:
            print(f"\nAccuracy (last 100 frames):")
            recent_pos = self.position_errors[-100:]
            recent_vel = self.velocity_errors[-100:] if self.velocity_errors else [0]
            recent_det = self.detection_rates[-100:]
            print(f"  Position RMSE:  {np.sqrt(np.mean(np.array(recent_pos)**2)):.2f} m")
            print(f"  Velocity RMSE:  {np.sqrt(np.mean(np.array(recent_vel)**2)):.2f} m/s")
            print(f"  Detection Rate: {np.mean(recent_det)*100:.1f}%")

        print("="*70)

    def _print_final_analysis(self):
        """Print final accuracy analysis"""
        if not self.position_errors:
            return

        print("\n" + "="*70)
        print("FINAL ACCURACY ANALYSIS")
        print("="*70)

        # Overall statistics
        pos_rmse = np.sqrt(np.mean(np.array(self.position_errors)**2))
        vel_rmse = np.sqrt(np.mean(np.array(self.velocity_errors)**2)) if self.velocity_errors else 0
        avg_detection_rate = np.mean(self.detection_rates) * 100

        print(f"Simulation Duration:  {self.sim_time:.1f}s")
        print(f"Total Frames:         {self.frame_count}")
        print(f"Total Drones:         {len(self.simulator.drones)}")

        print(f"\nTracking Accuracy:")
        print(f"  Position RMSE:      {pos_rmse:.2f} m")
        print(f"  Velocity RMSE:      {vel_rmse:.2f} m/s")
        print(f"  Detection Rate:     {avg_detection_rate:.1f}%")

        tracker_stats = self.tracker.get_performance_summary()
        print(f"\nTracker Performance:")
        print(f"  Tracks Created:     {tracker_stats['total_tracks_created']}")
        print(f"  Tracks Confirmed:   {tracker_stats['total_tracks_confirmed']}")
        print(f"  Avg Latency:        {tracker_stats['avg_latency_ms']:.2f} ms")

        # Requirements check
        print(f"\nRequirements:")
        print(f"  Detection Rate >99%:  {'✓' if avg_detection_rate > 99 else '✗'} ({avg_detection_rate:.1f}%)")
        print(f"  Position RMSE <5m:    {'✓' if pos_rmse < 5 else '✗'} ({pos_rmse:.2f}m)")
        print(f"  Latency <100ms:       {'✓' if tracker_stats['avg_latency_ms'] < 100 else '✗'}")

        print("="*70 + "\n")

    def save_analysis(self, filename: str):
        """Save accuracy analysis to file"""
        data = {
            'simulation_config': {
                'num_drones': len(self.simulator.drones),
                'duration': self.sim_time,
                'frames': self.frame_count
            },
            'accuracy': {
                'position_rmse': float(np.sqrt(np.mean(np.array(self.position_errors)**2))),
                'velocity_rmse': float(np.sqrt(np.mean(np.array(self.velocity_errors)**2))) if self.velocity_errors else 0,
                'avg_detection_rate': float(np.mean(self.detection_rates)),
                'position_errors': [float(e) for e in self.position_errors],
                'velocity_errors': [float(e) for e in self.velocity_errors],
                'detection_rates': [float(r) for r in self.detection_rates]
            },
            'tracker_stats': self.tracker.get_performance_summary()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Analysis saved to {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Drone tracking simulation with 200 drones'
    )
    parser.add_argument('--num-drones', type=int, default=200,
                       help='Number of drones (default: 200)')
    parser.add_argument('--duration', type=float, default=120.0,
                       help='Simulation duration in seconds (default: 120)')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    parser.add_argument('--save-analysis', type=str,
                       help='Save accuracy analysis to file')
    parser.add_argument('--trajectory-mix', type=str, default='balanced',
                       choices=['balanced', 'aggressive', 'calm'],
                       help='Trajectory mix type')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("DRONE TRACKING SIMULATION")
    print("="*70)
    print(f"Drones:         {args.num_drones}")
    print(f"Duration:       {args.duration}s")
    print(f"Trajectory Mix: {args.trajectory_mix}")
    print("="*70 + "\n")

    sim = DroneTrackingSimulation(args.num_drones, args.trajectory_mix)
    sim.run(duration=args.duration, visualize=args.visualize)

    if args.save_analysis:
        sim.save_analysis(args.save_analysis)

    logger.info("Simulation complete!")


if __name__ == "__main__":
    main()
