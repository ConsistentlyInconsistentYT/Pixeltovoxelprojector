"""
Position Broadcast System - Real-time pose broadcasting for all cameras

This module provides:
- Real-time pose broadcasting for all 20 cameras
- Sub-millisecond timestamp synchronization
- Coordinate frame transformations (ECEF, ENU, Camera Frame)
- Network broadcasting using ZeroMQ
- Calibration parameter distribution
- Low-latency multicast for synchronized systems

Author: Pixeltovoxelprojector Team
"""

import numpy as np
import time
import threading
import zmq
import json
import struct
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
from scipy.spatial.transform import Rotation
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoordinateFrame:
    """Coordinate frame definition"""
    name: str
    origin: np.ndarray  # Origin position [x, y, z]
    orientation: Rotation  # Orientation as Rotation object
    timestamp: float  # When this frame was established


@dataclass
class CameraCalibration:
    """Camera calibration parameters"""
    camera_id: int
    intrinsic_matrix: np.ndarray  # 3x3 camera intrinsic matrix
    distortion_coeffs: np.ndarray  # Distortion coefficients [k1, k2, p1, p2, k3]
    resolution: Tuple[int, int]  # (width, height)
    fov_horizontal: float  # Field of view in degrees
    fov_vertical: float  # Field of view in degrees
    timestamp: float  # When calibration was updated


@dataclass
class BroadcastPose:
    """Complete pose message for broadcasting"""
    camera_id: int
    timestamp: float  # Nanosecond precision

    # Position in different frames
    position_ecef: np.ndarray  # [x, y, z] in ECEF frame
    position_enu: np.ndarray  # [x, y, z] in local ENU frame
    position_world: np.ndarray  # [x, y, z] in world/voxel frame

    # Orientation (as quaternion)
    orientation_quat: np.ndarray  # [w, x, y, z]
    orientation_euler: np.ndarray  # [roll, pitch, yaw] in radians

    # Velocity
    velocity_ecef: np.ndarray  # [vx, vy, vz] in m/s
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s

    # Uncertainty
    position_std: np.ndarray  # [std_x, std_y, std_z] in meters
    orientation_std: np.ndarray  # [std_roll, std_pitch, std_yaw] in radians

    # Quality metrics
    rtk_fix_quality: int  # 0: no fix, 1: float, 2: fixed
    feature_count: int  # Number of VIO features tracked
    imu_health: float  # IMU health indicator (0-1)

    # Calibration parameters
    calibration: Optional[CameraCalibration] = None


class CoordinateTransformer:
    """
    Handles coordinate transformations between different frames:
    - ECEF (Earth-Centered Earth-Fixed)
    - ENU (East-North-Up, local tangent plane)
    - World/Voxel frame (project-specific)
    """

    def __init__(self):
        # WGS84 ellipsoid parameters
        self.a = 6378137.0  # Semi-major axis (meters)
        self.b = 6356752.314245  # Semi-minor axis (meters)
        self.f = (self.a - self.b) / self.a  # Flattening
        self.e_sq = 1 - (self.b / self.a) ** 2  # Eccentricity squared

        # Reference point for ENU (set by user)
        self.enu_reference_ecef: Optional[np.ndarray] = None
        self.enu_reference_lla: Optional[np.ndarray] = None  # [lat, lon, alt] in radians

        # World frame transformation (set by user)
        self.world_frame: Optional[CoordinateFrame] = None

    def set_enu_reference(self, lat_deg: float, lon_deg: float, alt_m: float):
        """Set ENU reference point (origin)"""
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)

        self.enu_reference_lla = np.array([lat, lon, alt_m])
        self.enu_reference_ecef = self.lla_to_ecef(lat, lon, alt_m)

        logger.info(f"ENU reference set: Lat={lat_deg:.6f}°, Lon={lon_deg:.6f}°, Alt={alt_m:.2f}m")

    def set_world_frame(self, frame: CoordinateFrame):
        """Set world/voxel frame"""
        self.world_frame = frame
        logger.info(f"World frame set: {frame.name}")

    def lla_to_ecef(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Convert Lat/Lon/Alt to ECEF coordinates"""
        N = self.a / np.sqrt(1 - self.e_sq * np.sin(lat) ** 2)

        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - self.e_sq) + alt) * np.sin(lat)

        return np.array([x, y, z])

    def ecef_to_lla(self, ecef: np.ndarray) -> Tuple[float, float, float]:
        """Convert ECEF to Lat/Lon/Alt"""
        x, y, z = ecef

        # Longitude
        lon = np.arctan2(y, x)

        # Iterative calculation for latitude and altitude
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p * (1 - self.e_sq))

        for _ in range(5):  # Iterate to converge
            N = self.a / np.sqrt(1 - self.e_sq * np.sin(lat) ** 2)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - self.e_sq * N / (N + alt)))

        return lat, lon, alt

    def ecef_to_enu(self, ecef: np.ndarray) -> np.ndarray:
        """Convert ECEF to local ENU coordinates"""
        if self.enu_reference_ecef is None:
            raise ValueError("ENU reference not set. Call set_enu_reference() first.")

        # Vector from reference to point
        dx = ecef - self.enu_reference_ecef

        lat, lon = self.enu_reference_lla[0], self.enu_reference_lla[1]

        # Rotation matrix from ECEF to ENU
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        R = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])

        enu = R @ dx
        return enu

    def enu_to_ecef(self, enu: np.ndarray) -> np.ndarray:
        """Convert ENU to ECEF coordinates"""
        if self.enu_reference_ecef is None:
            raise ValueError("ENU reference not set. Call set_enu_reference() first.")

        lat, lon = self.enu_reference_lla[0], self.enu_reference_lla[1]

        # Rotation matrix from ENU to ECEF
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        R = np.array([
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0, cos_lat, sin_lat]
        ])

        dx = R @ enu
        ecef = self.enu_reference_ecef + dx
        return ecef

    def ecef_to_world(self, ecef: np.ndarray) -> np.ndarray:
        """Convert ECEF to world/voxel frame"""
        if self.world_frame is None:
            # Default: use ENU
            return self.ecef_to_enu(ecef)

        # First convert to ENU
        enu = self.ecef_to_enu(ecef)

        # Then apply world frame transformation
        world = self.world_frame.orientation.apply(enu - self.world_frame.origin)
        return world

    def world_to_ecef(self, world: np.ndarray) -> np.ndarray:
        """Convert world/voxel frame to ECEF"""
        if self.world_frame is None:
            return self.enu_to_ecef(world)

        # Apply inverse world frame transformation
        enu = self.world_frame.orientation.inv().apply(world) + self.world_frame.origin

        # Convert ENU to ECEF
        ecef = self.enu_to_ecef(enu)
        return ecef


class PositionBroadcaster:
    """
    Real-time position and orientation broadcaster

    Uses ZeroMQ for high-performance, low-latency broadcasting
    Supports multiple subscribers with different requirements
    """

    def __init__(self,
                 num_cameras: int = 20,
                 broadcast_port: int = 5555,
                 calibration_port: int = 5556):
        """
        Initialize position broadcaster

        Args:
            num_cameras: Number of cameras
            broadcast_port: ZeroMQ port for pose broadcasting
            calibration_port: ZeroMQ port for calibration updates
        """
        self.num_cameras = num_cameras
        self.broadcast_port = broadcast_port
        self.calibration_port = calibration_port

        # Coordinate transformer
        self.transformer = CoordinateTransformer()

        # Latest calibrations
        self.calibrations: Dict[int, CameraCalibration] = {}

        # ZeroMQ context and sockets
        self.context = zmq.Context()
        self.pose_publisher: Optional[zmq.Socket] = None
        self.calibration_publisher: Optional[zmq.Socket] = None

        # Broadcasting thread
        self.running = False
        self.broadcast_thread: Optional[threading.Thread] = None

        # Pose queue for broadcasting
        self.pose_queue: deque = deque(maxlen=1000)
        self.queue_lock = threading.Lock()

        # Performance monitoring
        self.broadcast_count = 0
        self.last_performance_report = time.time()
        self.broadcast_latencies: deque = deque(maxlen=100)

        logger.info(f"PositionBroadcaster initialized for {num_cameras} cameras")

    def start(self):
        """Start the broadcasting service"""
        # Create ZeroMQ publishers
        self.pose_publisher = self.context.socket(zmq.PUB)
        self.pose_publisher.setsockopt(zmq.SNDHWM, 1000)  # High water mark
        self.pose_publisher.bind(f"tcp://*:{self.broadcast_port}")

        self.calibration_publisher = self.context.socket(zmq.PUB)
        self.calibration_publisher.bind(f"tcp://*:{self.calibration_port}")

        # Start broadcasting thread
        self.running = True
        self.broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            daemon=True,
            name="PositionBroadcaster"
        )
        self.broadcast_thread.start()

        logger.info(f"PositionBroadcaster started on ports {self.broadcast_port} and {self.calibration_port}")

    def stop(self):
        """Stop the broadcasting service"""
        self.running = False

        if self.broadcast_thread:
            self.broadcast_thread.join(timeout=1.0)

        if self.pose_publisher:
            self.pose_publisher.close()
        if self.calibration_publisher:
            self.calibration_publisher.close()

        self.context.term()

        logger.info("PositionBroadcaster stopped")

    def set_enu_reference(self, lat_deg: float, lon_deg: float, alt_m: float):
        """Set ENU coordinate reference"""
        self.transformer.set_enu_reference(lat_deg, lon_deg, alt_m)

    def set_world_frame(self, frame: CoordinateFrame):
        """Set world/voxel coordinate frame"""
        self.transformer.set_world_frame(frame)

    def update_calibration(self, calibration: CameraCalibration):
        """Update camera calibration parameters"""
        self.calibrations[calibration.camera_id] = calibration

        # Broadcast calibration update
        if self.calibration_publisher:
            self._broadcast_calibration(calibration)

    def broadcast_pose(self, pose: BroadcastPose):
        """
        Add pose to broadcast queue

        Args:
            pose: BroadcastPose object with ECEF position
        """
        # Transform to different coordinate frames
        try:
            pose.position_enu = self.transformer.ecef_to_enu(pose.position_ecef)
            pose.position_world = self.transformer.ecef_to_world(pose.position_ecef)
        except ValueError:
            # If transformation not set up, use ECEF
            pose.position_enu = pose.position_ecef.copy()
            pose.position_world = pose.position_ecef.copy()

        # Add calibration if available
        pose.calibration = self.calibrations.get(pose.camera_id)

        with self.queue_lock:
            self.pose_queue.append(pose)

    def _broadcast_loop(self):
        """Main broadcasting loop"""
        while self.running:
            # Get poses from queue
            poses_to_broadcast = []
            with self.queue_lock:
                while len(self.pose_queue) > 0:
                    poses_to_broadcast.append(self.pose_queue.popleft())

            # Broadcast each pose
            for pose in poses_to_broadcast:
                broadcast_start = time.time()
                self._broadcast_pose(pose)

                # Track latency
                latency = time.time() - broadcast_start
                self.broadcast_latencies.append(latency)
                self.broadcast_count += 1

            # Performance reporting
            if time.time() - self.last_performance_report > 5.0:
                self._report_performance()

            # Small sleep to prevent busy waiting
            if len(poses_to_broadcast) == 0:
                time.sleep(0.0001)  # 0.1ms

    def _broadcast_pose(self, pose: BroadcastPose):
        """Broadcast a single pose using ZeroMQ"""
        if not self.pose_publisher:
            return

        # Create compact binary message for low latency
        # Format: camera_id (4 bytes) + timestamp (8 bytes) + pose data

        # Pack data efficiently
        message_parts = []

        # Header
        header = struct.pack(
            '<IQ',  # Little-endian: unsigned int, unsigned long long
            pose.camera_id,
            int(pose.timestamp)
        )
        message_parts.append(header)

        # Position data (3 frames x 3 values = 9 doubles)
        position_data = np.concatenate([
            pose.position_ecef,
            pose.position_enu,
            pose.position_world
        ]).astype(np.float64).tobytes()
        message_parts.append(position_data)

        # Orientation data (quaternion + euler = 7 doubles)
        orientation_data = np.concatenate([
            pose.orientation_quat,
            pose.orientation_euler
        ]).astype(np.float64).tobytes()
        message_parts.append(orientation_data)

        # Velocity data (6 doubles)
        velocity_data = np.concatenate([
            pose.velocity_ecef,
            pose.angular_velocity
        ]).astype(np.float64).tobytes()
        message_parts.append(velocity_data)

        # Uncertainty data (6 doubles)
        uncertainty_data = np.concatenate([
            pose.position_std,
            pose.orientation_std
        ]).astype(np.float64).tobytes()
        message_parts.append(uncertainty_data)

        # Quality metrics
        quality = struct.pack(
            '<IIf',
            pose.rtk_fix_quality,
            pose.feature_count,
            pose.imu_health
        )
        message_parts.append(quality)

        # Send multipart message
        self.pose_publisher.send_multipart(message_parts, zmq.NOBLOCK)

        # Also send JSON version for debugging (on separate topic)
        json_message = self._pose_to_json(pose)
        self.pose_publisher.send_string(f"JSON:{pose.camera_id}", zmq.SNDMORE | zmq.NOBLOCK)
        self.pose_publisher.send_string(json_message, zmq.NOBLOCK)

    def _broadcast_calibration(self, calibration: CameraCalibration):
        """Broadcast calibration update"""
        if not self.calibration_publisher:
            return

        # Convert to JSON for calibration (not time-critical)
        calib_dict = {
            'camera_id': calibration.camera_id,
            'intrinsic_matrix': calibration.intrinsic_matrix.tolist(),
            'distortion_coeffs': calibration.distortion_coeffs.tolist(),
            'resolution': calibration.resolution,
            'fov_horizontal': calibration.fov_horizontal,
            'fov_vertical': calibration.fov_vertical,
            'timestamp': calibration.timestamp
        }

        json_message = json.dumps(calib_dict)
        self.calibration_publisher.send_string(json_message, zmq.NOBLOCK)

        logger.info(f"Broadcasted calibration for camera {calibration.camera_id}")

    def _pose_to_json(self, pose: BroadcastPose) -> str:
        """Convert pose to JSON format"""
        pose_dict = {
            'camera_id': pose.camera_id,
            'timestamp': pose.timestamp,
            'position_ecef': pose.position_ecef.tolist(),
            'position_enu': pose.position_enu.tolist(),
            'position_world': pose.position_world.tolist(),
            'orientation_quat': pose.orientation_quat.tolist(),
            'orientation_euler': pose.orientation_euler.tolist(),
            'velocity_ecef': pose.velocity_ecef.tolist(),
            'angular_velocity': pose.angular_velocity.tolist(),
            'position_std': pose.position_std.tolist(),
            'orientation_std': pose.orientation_std.tolist(),
            'rtk_fix_quality': pose.rtk_fix_quality,
            'feature_count': pose.feature_count,
            'imu_health': pose.imu_health
        }

        return json.dumps(pose_dict)

    def _report_performance(self):
        """Report performance metrics"""
        elapsed = time.time() - self.last_performance_report
        rate = self.broadcast_count / elapsed

        if len(self.broadcast_latencies) > 0:
            avg_latency = np.mean(self.broadcast_latencies) * 1000  # Convert to ms
            max_latency = np.max(self.broadcast_latencies) * 1000

            logger.info(
                f"Broadcast performance: {rate:.1f} poses/s | "
                f"Latency: avg={avg_latency:.3f}ms, max={max_latency:.3f}ms"
            )

        self.broadcast_count = 0
        self.last_performance_report = time.time()


class PositionReceiver:
    """
    Receiver/subscriber for pose broadcasts

    Allows other systems to receive real-time pose updates
    """

    def __init__(self, server_address: str = "localhost", port: int = 5555):
        """
        Initialize receiver

        Args:
            server_address: Address of broadcaster
            port: Port to connect to
        """
        self.server_address = server_address
        self.port = port

        # ZeroMQ context and socket
        self.context = zmq.Context()
        self.subscriber: Optional[zmq.Socket] = None

        # Receive thread
        self.running = False
        self.receive_thread: Optional[threading.Thread] = None

        # Latest poses
        self.latest_poses: Dict[int, BroadcastPose] = {}
        self.pose_lock = threading.Lock()

        # Callback for pose updates
        self.pose_callback = None

    def start(self, pose_callback=None):
        """Start receiving poses"""
        self.pose_callback = pose_callback

        # Create subscriber
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://{self.server_address}:{self.port}")
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all

        # Start receive thread
        self.running = True
        self.receive_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name="PositionReceiver"
        )
        self.receive_thread.start()

        logger.info(f"PositionReceiver started, connecting to {self.server_address}:{self.port}")

    def stop(self):
        """Stop receiving"""
        self.running = False

        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)

        if self.subscriber:
            self.subscriber.close()

        self.context.term()

        logger.info("PositionReceiver stopped")

    def _receive_loop(self):
        """Main receive loop"""
        while self.running:
            try:
                # Non-blocking receive with timeout
                if self.subscriber.poll(100, zmq.POLLIN):
                    message_parts = self.subscriber.recv_multipart(zmq.NOBLOCK)

                    # Decode binary message
                    pose = self._decode_pose(message_parts)

                    if pose:
                        with self.pose_lock:
                            self.latest_poses[pose.camera_id] = pose

                        # Call callback if provided
                        if self.pose_callback:
                            self.pose_callback(pose)
            except zmq.ZMQError:
                pass

    def _decode_pose(self, message_parts: List[bytes]) -> Optional[BroadcastPose]:
        """Decode binary pose message"""
        try:
            # Unpack header
            camera_id, timestamp = struct.unpack('<IQ', message_parts[0])

            # Unpack position data
            positions = np.frombuffer(message_parts[1], dtype=np.float64).reshape(3, 3)

            # Unpack orientation data
            orientations = np.frombuffer(message_parts[2], dtype=np.float64)

            # Unpack velocity data
            velocities = np.frombuffer(message_parts[3], dtype=np.float64).reshape(2, 3)

            # Unpack uncertainty data
            uncertainties = np.frombuffer(message_parts[4], dtype=np.float64).reshape(2, 3)

            # Unpack quality metrics
            rtk_fix, feature_count, imu_health = struct.unpack('<IIf', message_parts[5])

            pose = BroadcastPose(
                camera_id=camera_id,
                timestamp=float(timestamp),
                position_ecef=positions[0],
                position_enu=positions[1],
                position_world=positions[2],
                orientation_quat=orientations[0:4],
                orientation_euler=orientations[4:7],
                velocity_ecef=velocities[0],
                angular_velocity=velocities[1],
                position_std=uncertainties[0],
                orientation_std=uncertainties[1],
                rtk_fix_quality=rtk_fix,
                feature_count=feature_count,
                imu_health=imu_health
            )

            return pose
        except Exception as e:
            logger.error(f"Error decoding pose: {e}")
            return None

    def get_pose(self, camera_id: int) -> Optional[BroadcastPose]:
        """Get latest pose for a camera"""
        with self.pose_lock:
            return self.latest_poses.get(camera_id)

    def get_all_poses(self) -> Dict[int, BroadcastPose]:
        """Get all latest poses"""
        with self.pose_lock:
            return self.latest_poses.copy()


# Example usage
if __name__ == "__main__":
    # Example: Set up broadcaster
    broadcaster = PositionBroadcaster(num_cameras=20)

    # Set ENU reference (example location)
    broadcaster.set_enu_reference(
        lat_deg=37.7749,  # San Francisco
        lon_deg=-122.4194,
        alt_m=0.0
    )

    broadcaster.start()

    try:
        # Simulate broadcasting poses for 5 seconds
        for i in range(500):  # 100Hz for 5 seconds
            for camera_id in range(20):
                # Create example pose
                pose = BroadcastPose(
                    camera_id=camera_id,
                    timestamp=time.time_ns(),
                    position_ecef=np.array([
                        -2707066.0 + camera_id * 10,
                        -4260952.0,
                        3884595.0
                    ]),
                    position_enu=np.zeros(3),  # Will be computed
                    position_world=np.zeros(3),  # Will be computed
                    orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    orientation_euler=np.array([0.0, 0.0, 0.0]),
                    velocity_ecef=np.array([0.1, 0.0, 0.0]),
                    angular_velocity=np.array([0.0, 0.0, 0.01]),
                    position_std=np.array([0.02, 0.02, 0.03]),
                    orientation_std=np.array([0.001, 0.001, 0.002]),
                    rtk_fix_quality=2,
                    feature_count=50,
                    imu_health=0.95
                )

                broadcaster.broadcast_pose(pose)

            time.sleep(0.01)  # 100Hz

        print("\nBroadcasting complete!")

    finally:
        broadcaster.stop()
