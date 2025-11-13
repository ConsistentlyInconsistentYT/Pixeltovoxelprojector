"""
Drone Trajectory Generator
Generates realistic drone flight paths for testing including linear, circular, evasive patterns
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class TrajectoryType(Enum):
    """Types of flight trajectories"""
    LINEAR = "linear"
    CIRCULAR = "circular"
    FIGURE_EIGHT = "figure_eight"
    EVASIVE = "evasive"
    SPIRAL = "spiral"
    RANDOM_WALK = "random_walk"
    FORMATION = "formation"


@dataclass
class TrajectoryPoint:
    """Single point in trajectory"""
    timestamp: float
    position: Tuple[float, float, float]  # x, y, z
    velocity: Tuple[float, float, float]  # vx, vy, vz
    acceleration: Optional[Tuple[float, float, float]] = None


@dataclass
class Trajectory:
    """Complete trajectory for a drone"""
    drone_id: int
    trajectory_type: TrajectoryType
    points: List[TrajectoryPoint] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class TrajectoryGenerator:
    """Generate realistic drone trajectories"""

    def __init__(
        self,
        duration_seconds: float = 60.0,
        sample_rate_hz: float = 30.0,
        max_velocity: float = 20.0,  # m/s
        max_acceleration: float = 5.0  # m/s^2
    ):
        """
        Initialize trajectory generator

        Args:
            duration_seconds: Total duration of trajectory
            sample_rate_hz: Sampling rate for trajectory points
            max_velocity: Maximum drone velocity
            max_acceleration: Maximum drone acceleration
        """
        self.duration = duration_seconds
        self.sample_rate = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.num_points = int(duration_seconds * sample_rate_hz)
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def generate_linear(
        self,
        drone_id: int,
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        constant_velocity: bool = True
    ) -> Trajectory:
        """Generate linear trajectory from start to end"""
        trajectory = Trajectory(drone_id=drone_id, trajectory_type=TrajectoryType.LINEAR)

        # Calculate velocity
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dz = end_pos[2] - start_pos[2]

        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        duration = self.duration

        vx = dx / duration
        vy = dy / duration
        vz = dz / duration

        # Clamp to max velocity
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_mag > self.max_velocity:
            scale = self.max_velocity / v_mag
            vx *= scale
            vy *= scale
            vz *= scale

        # Generate points
        for i in range(self.num_points):
            t = i * self.dt
            progress = t / self.duration

            x = start_pos[0] + dx * progress
            y = start_pos[1] + dy * progress
            z = start_pos[2] + dz * progress

            point = TrajectoryPoint(
                timestamp=t,
                position=(x, y, z),
                velocity=(vx, vy, vz),
                acceleration=(0, 0, 0)
            )
            trajectory.points.append(point)

        trajectory.metadata = {
            'start_position': start_pos,
            'end_position': end_pos,
            'distance': distance,
            'avg_velocity': v_mag
        }

        return trajectory

    def generate_circular(
        self,
        drone_id: int,
        center: Tuple[float, float, float],
        radius: float,
        angular_velocity: float = 0.1,  # rad/s
        clockwise: bool = True
    ) -> Trajectory:
        """Generate circular trajectory"""
        trajectory = Trajectory(drone_id=drone_id, trajectory_type=TrajectoryType.CIRCULAR)

        direction = -1 if clockwise else 1

        for i in range(self.num_points):
            t = i * self.dt
            angle = direction * angular_velocity * t

            # Position on circle
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]

            # Velocity (tangent to circle)
            vx = -direction * radius * angular_velocity * np.sin(angle)
            vy = direction * radius * angular_velocity * np.cos(angle)
            vz = 0

            # Centripetal acceleration
            ax = -angular_velocity**2 * radius * np.cos(angle)
            ay = -angular_velocity**2 * radius * np.sin(angle)
            az = 0

            point = TrajectoryPoint(
                timestamp=t,
                position=(x, y, z),
                velocity=(vx, vy, vz),
                acceleration=(ax, ay, az)
            )
            trajectory.points.append(point)

        trajectory.metadata = {
            'center': center,
            'radius': radius,
            'angular_velocity': angular_velocity,
            'period': 2 * np.pi / angular_velocity
        }

        return trajectory

    def generate_figure_eight(
        self,
        drone_id: int,
        center: Tuple[float, float, float],
        width: float,
        height: float
    ) -> Trajectory:
        """Generate figure-eight trajectory"""
        trajectory = Trajectory(drone_id=drone_id, trajectory_type=TrajectoryType.FIGURE_EIGHT)

        omega = 2 * np.pi / self.duration  # Complete one figure-eight

        for i in range(self.num_points):
            t = i * self.dt

            # Lissajous curve (figure-eight)
            x = center[0] + width * np.sin(omega * t)
            y = center[1] + height * np.sin(2 * omega * t)
            z = center[2]

            # Velocity
            vx = width * omega * np.cos(omega * t)
            vy = 2 * height * omega * np.cos(2 * omega * t)
            vz = 0

            # Acceleration
            ax = -width * omega**2 * np.sin(omega * t)
            ay = -4 * height * omega**2 * np.sin(2 * omega * t)
            az = 0

            point = TrajectoryPoint(
                timestamp=t,
                position=(x, y, z),
                velocity=(vx, vy, vz),
                acceleration=(ax, ay, az)
            )
            trajectory.points.append(point)

        trajectory.metadata = {
            'center': center,
            'width': width,
            'height': height
        }

        return trajectory

    def generate_evasive(
        self,
        drone_id: int,
        start_pos: Tuple[float, float, float],
        general_direction: Tuple[float, float, float],
        evasion_amplitude: float = 50.0
    ) -> Trajectory:
        """Generate evasive maneuver trajectory"""
        trajectory = Trajectory(drone_id=drone_id, trajectory_type=TrajectoryType.EVASIVE)

        # Normalize direction
        dx, dy, dz = general_direction
        magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
            dz /= magnitude

        base_velocity = self.max_velocity * 0.7

        # Generate random evasion pattern
        evasion_frequency = np.random.uniform(0.5, 2.0)  # Hz

        x, y, z = start_pos
        vx, vy, vz = dx * base_velocity, dy * base_velocity, dz * base_velocity

        for i in range(self.num_points):
            t = i * self.dt

            # Add evasive maneuvers
            evasion_x = evasion_amplitude * np.sin(2 * np.pi * evasion_frequency * t)
            evasion_y = evasion_amplitude * np.cos(2 * np.pi * evasion_frequency * t * 1.3)

            # Random direction changes
            if i % int(self.sample_rate * 2) == 0:  # Every 2 seconds
                vx += np.random.uniform(-2, 2)
                vy += np.random.uniform(-2, 2)

                # Clamp velocity
                v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
                if v_mag > self.max_velocity:
                    scale = self.max_velocity / v_mag
                    vx *= scale
                    vy *= scale
                    vz *= scale

            x += vx * self.dt + evasion_x * self.dt
            y += vy * self.dt + evasion_y * self.dt
            z += vz * self.dt

            point = TrajectoryPoint(
                timestamp=t,
                position=(x, y, z),
                velocity=(vx, vy, vz)
            )
            trajectory.points.append(point)

        trajectory.metadata = {
            'start_position': start_pos,
            'evasion_amplitude': evasion_amplitude,
            'general_direction': general_direction
        }

        return trajectory

    def generate_spiral(
        self,
        drone_id: int,
        center: Tuple[float, float, float],
        initial_radius: float,
        final_radius: float,
        height_gain: float
    ) -> Trajectory:
        """Generate spiral trajectory"""
        trajectory = Trajectory(drone_id=drone_id, trajectory_type=TrajectoryType.SPIRAL)

        angular_velocity = 2 * np.pi / (self.duration / 2)  # 2 full rotations

        for i in range(self.num_points):
            t = i * self.dt
            progress = t / self.duration

            # Radius changes over time
            radius = initial_radius + (final_radius - initial_radius) * progress
            angle = angular_velocity * t

            # Position
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2] + height_gain * progress

            # Velocity
            dr_dt = (final_radius - initial_radius) / self.duration
            vx = dr_dt * np.cos(angle) - radius * angular_velocity * np.sin(angle)
            vy = dr_dt * np.sin(angle) + radius * angular_velocity * np.cos(angle)
            vz = height_gain / self.duration

            point = TrajectoryPoint(
                timestamp=t,
                position=(x, y, z),
                velocity=(vx, vy, vz)
            )
            trajectory.points.append(point)

        trajectory.metadata = {
            'center': center,
            'initial_radius': initial_radius,
            'final_radius': final_radius,
            'height_gain': height_gain
        }

        return trajectory

    def generate_random_walk(
        self,
        drone_id: int,
        start_pos: Tuple[float, float, float],
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ) -> Trajectory:
        """Generate random walk trajectory within bounds"""
        trajectory = Trajectory(drone_id=drone_id, trajectory_type=TrajectoryType.RANDOM_WALK)

        x, y, z = start_pos
        vx, vy, vz = 0.0, 0.0, 0.0

        for i in range(self.num_points):
            t = i * self.dt

            # Random acceleration changes
            if i % int(self.sample_rate) == 0:  # Every second
                ax = np.random.uniform(-self.max_acceleration, self.max_acceleration)
                ay = np.random.uniform(-self.max_acceleration, self.max_acceleration)
                az = np.random.uniform(-self.max_acceleration / 2, self.max_acceleration / 2)

                vx += ax * self.dt
                vy += ay * self.dt
                vz += az * self.dt

                # Clamp velocity
                v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
                if v_mag > self.max_velocity:
                    scale = self.max_velocity / v_mag
                    vx *= scale
                    vy *= scale
                    vz *= scale

            # Update position
            x += vx * self.dt
            y += vy * self.dt
            z += vz * self.dt

            # Bounce off bounds
            if x < bounds[0][0] or x > bounds[0][1]:
                vx = -vx
                x = np.clip(x, bounds[0][0], bounds[0][1])

            if y < bounds[1][0] or y > bounds[1][1]:
                vy = -vy
                y = np.clip(y, bounds[1][0], bounds[1][1])

            if z < bounds[2][0] or z > bounds[2][1]:
                vz = -vz
                z = np.clip(z, bounds[2][0], bounds[2][1])

            point = TrajectoryPoint(
                timestamp=t,
                position=(x, y, z),
                velocity=(vx, vy, vz)
            )
            trajectory.points.append(point)

        trajectory.metadata = {
            'start_position': start_pos,
            'bounds': bounds
        }

        return trajectory

    def generate_formation(
        self,
        num_drones: int,
        formation_center: Tuple[float, float, float],
        formation_spacing: float,
        base_velocity: Tuple[float, float, float]
    ) -> List[Trajectory]:
        """Generate formation flight for multiple drones"""
        trajectories = []

        # Arrange drones in a grid formation
        grid_size = int(np.ceil(np.sqrt(num_drones)))

        for i in range(num_drones):
            row = i // grid_size
            col = i % grid_size

            # Offset from formation center
            offset_x = (col - grid_size / 2) * formation_spacing
            offset_y = (row - grid_size / 2) * formation_spacing

            start_x = formation_center[0] + offset_x
            start_y = formation_center[1] + offset_y
            start_z = formation_center[2]

            trajectory = Trajectory(
                drone_id=i,
                trajectory_type=TrajectoryType.FORMATION,
                metadata={'formation_position': (row, col)}
            )

            vx, vy, vz = base_velocity

            for j in range(self.num_points):
                t = j * self.dt

                x = start_x + vx * t
                y = start_y + vy * t
                z = start_z + vz * t

                point = TrajectoryPoint(
                    timestamp=t,
                    position=(x, y, z),
                    velocity=base_velocity
                )
                trajectory.points.append(point)

            trajectories.append(trajectory)

        return trajectories

    def save_trajectory(self, trajectory: Trajectory, filename: str):
        """Save trajectory to JSON file"""
        data = {
            'drone_id': trajectory.drone_id,
            'trajectory_type': trajectory.trajectory_type.value,
            'metadata': trajectory.metadata,
            'points': [
                {
                    'timestamp': p.timestamp,
                    'position': p.position,
                    'velocity': p.velocity,
                    'acceleration': p.acceleration
                }
                for p in trajectory.points
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_trajectory(self, filename: str) -> Trajectory:
        """Load trajectory from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)

        trajectory = Trajectory(
            drone_id=data['drone_id'],
            trajectory_type=TrajectoryType(data['trajectory_type']),
            metadata=data['metadata']
        )

        for p_data in data['points']:
            point = TrajectoryPoint(
                timestamp=p_data['timestamp'],
                position=tuple(p_data['position']),
                velocity=tuple(p_data['velocity']),
                acceleration=tuple(p_data['acceleration']) if p_data['acceleration'] else None
            )
            trajectory.points.append(point)

        return trajectory


if __name__ == "__main__":
    # Example usage
    generator = TrajectoryGenerator(duration_seconds=30.0, sample_rate_hz=30.0)

    # Generate different trajectory types
    linear = generator.generate_linear(0, (0, 0, 1000), (500, 300, 2000))
    circular = generator.generate_circular(1, (0, 0, 1500), radius=200)
    evasive = generator.generate_evasive(2, (100, 100, 2000), (1, 0, 0.5))

    print(f"Generated trajectories:")
    print(f"  Linear: {len(linear.points)} points")
    print(f"  Circular: {len(circular.points)} points")
    print(f"  Evasive: {len(evasive.points)} points")

    # Save example
    generator.save_trajectory(linear, "/tmp/linear_trajectory.json")
    print(f"Saved linear trajectory to /tmp/linear_trajectory.json")
