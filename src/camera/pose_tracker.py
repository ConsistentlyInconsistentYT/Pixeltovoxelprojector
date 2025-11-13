"""
Camera Pose Tracker - Real-time 6DOF Pose Estimation System

This module provides high-precision camera pose tracking using sensor fusion from:
- RTK GPS for position (<5cm accuracy)
- IMU for orientation
- Visual-Inertial Odometry (VIO)
- Multi-sensor fusion with Extended Kalman Filter

Supports 20 cameras with synchronized pose estimation at 1000Hz update rate.

Author: Pixeltovoxelprojector Team
"""

import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IMUMeasurement:
    """IMU measurement data structure"""
    timestamp: float  # Nanosecond precision timestamp
    angular_velocity: np.ndarray  # rad/s [3]
    linear_acceleration: np.ndarray  # m/s^2 [3]
    camera_id: int


@dataclass
class GPSMeasurement:
    """RTK GPS measurement data structure"""
    timestamp: float  # Nanosecond precision timestamp
    position: np.ndarray  # ECEF coordinates [x, y, z] in meters
    position_covariance: np.ndarray  # 3x3 covariance matrix
    rtk_fix_quality: int  # 0: no fix, 1: float, 2: fixed (RTK)
    camera_id: int


@dataclass
class VIOMeasurement:
    """Visual-Inertial Odometry measurement"""
    timestamp: float
    relative_pose: np.ndarray  # 4x4 transformation matrix
    covariance: np.ndarray  # 6x6 covariance (position + orientation)
    camera_id: int
    feature_count: int  # Number of tracked features


@dataclass
class CameraPose:
    """Complete camera pose with uncertainty"""
    timestamp: float
    position: np.ndarray  # [x, y, z] in meters (ECEF or local frame)
    orientation: Rotation  # Scipy Rotation object (quaternion internally)
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    position_covariance: np.ndarray  # 3x3 position uncertainty
    orientation_covariance: np.ndarray  # 3x3 orientation uncertainty
    camera_id: int
    frame: str = "ECEF"  # Coordinate frame identifier


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for sensor fusion

    State vector [15x1]:
    - position [3]: x, y, z
    - velocity [3]: vx, vy, vz
    - orientation [3]: roll, pitch, yaw (euler angles)
    - gyro bias [3]: bias_wx, bias_wy, bias_wz
    - accel bias [3]: bias_ax, bias_ay, bias_az
    """

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.state_dim = 15

        # State vector and covariance
        self.x = np.zeros(self.state_dim)  # State
        self.P = np.eye(self.state_dim) * 1.0  # Covariance

        # Process noise covariance
        self.Q = block_diag(
            np.eye(3) * 0.001,  # position process noise
            np.eye(3) * 0.01,   # velocity process noise
            np.eye(3) * 0.0001, # orientation process noise
            np.eye(3) * 1e-6,   # gyro bias process noise
            np.eye(3) * 1e-5    # accel bias process noise
        )

        # Measurement noise covariance for GPS
        self.R_gps = np.eye(3) * 0.0025  # (5cm)^2

        # Measurement noise covariance for IMU
        self.R_imu_gyro = np.eye(3) * (0.001)**2  # 0.001 rad/s noise
        self.R_imu_accel = np.eye(3) * (0.01)**2  # 0.01 m/s^2 noise

        # Last update time
        self.last_update_time = None

    def predict(self, dt: float, angular_velocity: np.ndarray, linear_acceleration: np.ndarray):
        """
        Prediction step using IMU measurements

        Args:
            dt: Time step in seconds
            angular_velocity: Angular velocity from gyro [3]
            linear_acceleration: Linear acceleration from accelerometer [3]
        """
        if dt <= 0 or dt > 1.0:  # Sanity check
            logger.warning(f"Invalid dt={dt} for camera {self.camera_id}")
            return

        # Extract current state
        pos = self.x[0:3]
        vel = self.x[3:6]
        euler = self.x[6:9]
        gyro_bias = self.x[9:12]
        accel_bias = self.x[12:15]

        # Corrected measurements
        omega_corrected = angular_velocity - gyro_bias
        accel_corrected = linear_acceleration - accel_bias

        # Rotation matrix from body to world frame
        R = Rotation.from_euler('xyz', euler).as_matrix()

        # Gravity vector in world frame
        g = np.array([0, 0, -9.81])

        # State prediction
        # Position: p = p + v*dt + 0.5*a*dt^2
        accel_world = R @ accel_corrected + g
        pos_new = pos + vel * dt + 0.5 * accel_world * dt**2

        # Velocity: v = v + a*dt
        vel_new = vel + accel_world * dt

        # Orientation: integrate angular velocity
        delta_euler = omega_corrected * dt
        R_delta = Rotation.from_euler('xyz', delta_euler).as_matrix()
        R_new = R @ R_delta
        euler_new = Rotation.from_matrix(R_new).as_euler('xyz')

        # Biases remain constant (random walk)
        gyro_bias_new = gyro_bias
        accel_bias_new = accel_bias

        # Update state
        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6:9] = euler_new
        self.x[9:12] = gyro_bias_new
        self.x[12:15] = accel_bias_new

        # Jacobian of state transition (simplified - full version would be more complex)
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * dt  # position depends on velocity

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_gps(self, gps_position: np.ndarray, gps_covariance: np.ndarray):
        """
        Update step using GPS measurement

        Args:
            gps_position: GPS position measurement [3]
            gps_covariance: GPS measurement covariance [3x3]
        """
        # Measurement matrix (GPS measures position directly)
        H = np.zeros((3, self.state_dim))
        H[0:3, 0:3] = np.eye(3)

        # Innovation
        z = gps_position
        z_pred = self.x[0:3]
        y = z - z_pred

        # Innovation covariance
        R = gps_covariance if gps_covariance is not None else self.R_gps
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

    def update_vio(self, vio_pose: np.ndarray, vio_covariance: np.ndarray):
        """
        Update step using Visual-Inertial Odometry

        Args:
            vio_pose: 4x4 transformation matrix from VIO
            vio_covariance: 6x6 covariance (position + orientation)
        """
        # Extract position and orientation from VIO
        vio_position = vio_pose[0:3, 3]
        vio_rotation = Rotation.from_matrix(vio_pose[0:3, 0:3])
        vio_euler = vio_rotation.as_euler('xyz')

        # Measurement vector
        z = np.concatenate([vio_position, vio_euler])

        # Measurement matrix (VIO measures position and orientation)
        H = np.zeros((6, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # position
        H[3:6, 6:9] = np.eye(3)  # orientation

        # Innovation
        z_pred = np.concatenate([self.x[0:3], self.x[6:9]])
        y = z - z_pred

        # Handle angle wrapping for orientation
        y[3:6] = np.arctan2(np.sin(y[3:6]), np.cos(y[3:6]))

        # Innovation covariance
        R = vio_covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

    def get_pose(self, timestamp: float) -> CameraPose:
        """Extract current pose estimate"""
        position = self.x[0:3]
        velocity = self.x[3:6]
        euler = self.x[6:9]
        angular_velocity = np.zeros(3)  # Not directly stored in state

        orientation = Rotation.from_euler('xyz', euler)

        return CameraPose(
            timestamp=timestamp,
            position=position.copy(),
            orientation=orientation,
            velocity=velocity.copy(),
            angular_velocity=angular_velocity,
            position_covariance=self.P[0:3, 0:3].copy(),
            orientation_covariance=self.P[6:9, 6:9].copy(),
            camera_id=self.camera_id
        )


class CameraPoseTracker:
    """
    Main camera pose tracking system

    Manages real-time 6DOF pose estimation for up to 20 cameras using:
    - RTK GPS for position accuracy <5cm
    - High-frequency IMU (1000Hz) for orientation <0.1°
    - Visual-Inertial Odometry for drift correction
    - Extended Kalman Filter for sensor fusion
    """

    def __init__(self, num_cameras: int = 20, update_rate_hz: float = 1000.0):
        """
        Initialize pose tracker

        Args:
            num_cameras: Number of cameras to track
            update_rate_hz: Target pose update rate in Hz
        """
        self.num_cameras = num_cameras
        self.update_rate_hz = update_rate_hz
        self.update_period = 1.0 / update_rate_hz

        # Extended Kalman Filters for each camera
        self.ekf_filters: Dict[int, ExtendedKalmanFilter] = {}
        for camera_id in range(num_cameras):
            self.ekf_filters[camera_id] = ExtendedKalmanFilter(camera_id)

        # Measurement queues
        self.imu_queues: Dict[int, deque] = {i: deque(maxlen=1000) for i in range(num_cameras)}
        self.gps_queues: Dict[int, deque] = {i: deque(maxlen=100) for i in range(num_cameras)}
        self.vio_queues: Dict[int, deque] = {i: deque(maxlen=100) for i in range(num_cameras)}

        # Latest pose estimates
        self.latest_poses: Dict[int, CameraPose] = {}

        # Pose history for interpolation
        self.pose_history: Dict[int, deque] = {i: deque(maxlen=100) for i in range(num_cameras)}

        # Thread control
        self.running = False
        self.processing_threads: List[threading.Thread] = []
        self.lock = threading.Lock()

        # Performance metrics
        self.pose_update_count = {i: 0 for i in range(num_cameras)}
        self.last_performance_report = time.time()

        logger.info(f"CameraPoseTracker initialized for {num_cameras} cameras at {update_rate_hz}Hz")

    def start(self):
        """Start the pose tracking system"""
        self.running = True

        # Start processing thread for each camera
        for camera_id in range(self.num_cameras):
            thread = threading.Thread(
                target=self._processing_loop,
                args=(camera_id,),
                daemon=True,
                name=f"PoseTracker-Camera{camera_id}"
            )
            thread.start()
            self.processing_threads.append(thread)

        # Start performance monitoring thread
        monitor_thread = threading.Thread(
            target=self._performance_monitor,
            daemon=True,
            name="PoseTracker-Monitor"
        )
        monitor_thread.start()
        self.processing_threads.append(monitor_thread)

        logger.info("CameraPoseTracker started")

    def stop(self):
        """Stop the pose tracking system"""
        self.running = False
        for thread in self.processing_threads:
            thread.join(timeout=1.0)
        logger.info("CameraPoseTracker stopped")

    def add_imu_measurement(self, measurement: IMUMeasurement):
        """Add IMU measurement to processing queue"""
        camera_id = measurement.camera_id
        if camera_id < self.num_cameras:
            self.imu_queues[camera_id].append(measurement)

    def add_gps_measurement(self, measurement: GPSMeasurement):
        """Add GPS measurement to processing queue"""
        camera_id = measurement.camera_id
        if camera_id < self.num_cameras:
            self.gps_queues[camera_id].append(measurement)

    def add_vio_measurement(self, measurement: VIOMeasurement):
        """Add VIO measurement to processing queue"""
        camera_id = measurement.camera_id
        if camera_id < self.num_cameras:
            self.vio_queues[camera_id].append(measurement)

    def _processing_loop(self, camera_id: int):
        """Main processing loop for each camera"""
        last_imu_timestamp = None

        while self.running:
            loop_start = time.time()

            # Process IMU measurements (highest frequency)
            if len(self.imu_queues[camera_id]) > 0:
                imu_meas = self.imu_queues[camera_id].popleft()

                # Calculate dt
                if last_imu_timestamp is not None:
                    dt = (imu_meas.timestamp - last_imu_timestamp) * 1e-9  # Convert ns to seconds

                    # Prediction step
                    self.ekf_filters[camera_id].predict(
                        dt,
                        imu_meas.angular_velocity,
                        imu_meas.linear_acceleration
                    )

                last_imu_timestamp = imu_meas.timestamp

            # Process GPS measurements
            if len(self.gps_queues[camera_id]) > 0:
                gps_meas = self.gps_queues[camera_id].popleft()

                # Only use high-quality RTK fixes
                if gps_meas.rtk_fix_quality >= 2:
                    self.ekf_filters[camera_id].update_gps(
                        gps_meas.position,
                        gps_meas.position_covariance
                    )

            # Process VIO measurements
            if len(self.vio_queues[camera_id]) > 0:
                vio_meas = self.vio_queues[camera_id].popleft()

                # VIO covariance - scale based on feature count
                vio_cov = vio_meas.covariance.copy()
                if vio_meas.feature_count < 10:
                    vio_cov *= 10.0  # Increase uncertainty for low feature count

                self.ekf_filters[camera_id].update_vio(
                    vio_meas.relative_pose,
                    vio_cov
                )

            # Update latest pose
            current_timestamp = time.time_ns()
            pose = self.ekf_filters[camera_id].get_pose(current_timestamp)

            with self.lock:
                self.latest_poses[camera_id] = pose
                self.pose_history[camera_id].append(pose)
                self.pose_update_count[camera_id] += 1

            # Maintain update rate
            loop_duration = time.time() - loop_start
            sleep_time = self.update_period - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_pose(self, camera_id: int, timestamp: Optional[float] = None) -> Optional[CameraPose]:
        """
        Get pose for a specific camera at a specific timestamp

        Args:
            camera_id: Camera identifier
            timestamp: Desired timestamp (nanoseconds). If None, returns latest pose

        Returns:
            CameraPose object or None if not available
        """
        with self.lock:
            if timestamp is None:
                return self.latest_poses.get(camera_id)
            else:
                # Interpolate from pose history
                return self._interpolate_pose(camera_id, timestamp)

    def _interpolate_pose(self, camera_id: int, timestamp: float) -> Optional[CameraPose]:
        """Interpolate pose at specific timestamp from pose history"""
        history = self.pose_history[camera_id]

        if len(history) < 2:
            return self.latest_poses.get(camera_id)

        # Find bounding poses
        poses = list(history)

        # Find poses before and after target timestamp
        before_pose = None
        after_pose = None

        for i, pose in enumerate(poses):
            if pose.timestamp <= timestamp:
                before_pose = pose
            if pose.timestamp >= timestamp and after_pose is None:
                after_pose = pose
                break

        if before_pose is None or after_pose is None:
            return self.latest_poses.get(camera_id)

        # Linear interpolation
        t_total = after_pose.timestamp - before_pose.timestamp
        if t_total == 0:
            return before_pose

        t = (timestamp - before_pose.timestamp) / t_total

        # Interpolate position
        position = before_pose.position + t * (after_pose.position - before_pose.position)

        # SLERP for orientation
        orientation = Rotation.from_quat(
            self._slerp_quaternions(
                before_pose.orientation.as_quat(),
                after_pose.orientation.as_quat(),
                t
            )
        )

        # Interpolate velocity
        velocity = before_pose.velocity + t * (after_pose.velocity - before_pose.velocity)
        angular_velocity = before_pose.angular_velocity + t * (after_pose.angular_velocity - before_pose.angular_velocity)

        return CameraPose(
            timestamp=timestamp,
            position=position,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            position_covariance=before_pose.position_covariance,
            orientation_covariance=before_pose.orientation_covariance,
            camera_id=camera_id
        )

    def _slerp_quaternions(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions"""
        dot = np.dot(q0, q1)

        # If quaternions are close, use linear interpolation
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)

        # Ensure shortest path
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        q2 = q1 - q0 * dot
        q2 = q2 / np.linalg.norm(q2)

        return q0 * np.cos(theta) + q2 * np.sin(theta)

    def get_all_poses(self, timestamp: Optional[float] = None) -> Dict[int, CameraPose]:
        """Get poses for all cameras"""
        poses = {}
        for camera_id in range(self.num_cameras):
            pose = self.get_pose(camera_id, timestamp)
            if pose is not None:
                poses[camera_id] = pose
        return poses

    def _performance_monitor(self):
        """Monitor and report performance metrics"""
        while self.running:
            time.sleep(5.0)  # Report every 5 seconds

            current_time = time.time()
            elapsed = current_time - self.last_performance_report

            logger.info("=== Pose Tracker Performance ===")
            for camera_id in range(self.num_cameras):
                count = self.pose_update_count[camera_id]
                rate = count / elapsed

                pose = self.latest_poses.get(camera_id)
                if pose is not None:
                    pos_std = np.sqrt(np.diag(pose.position_covariance))
                    ori_std = np.sqrt(np.diag(pose.orientation_covariance))

                    logger.info(
                        f"Camera {camera_id:2d}: {rate:6.1f} Hz | "
                        f"Pos STD: [{pos_std[0]*100:.2f}, {pos_std[1]*100:.2f}, {pos_std[2]*100:.2f}] cm | "
                        f"Ori STD: [{np.rad2deg(ori_std[0]):.3f}, {np.rad2deg(ori_std[1]):.3f}, {np.rad2deg(ori_std[2]):.3f}]°"
                    )

                self.pose_update_count[camera_id] = 0

            self.last_performance_report = current_time

    def get_accuracy_statistics(self, camera_id: int) -> Dict[str, float]:
        """
        Get accuracy statistics for a camera

        Returns:
            Dictionary with position and orientation accuracy metrics
        """
        with self.lock:
            pose = self.latest_poses.get(camera_id)
            if pose is None:
                return {}

            pos_cov = pose.position_covariance
            ori_cov = pose.orientation_covariance

            # Standard deviations (1-sigma)
            pos_std = np.sqrt(np.diag(pos_cov))
            ori_std = np.sqrt(np.diag(ori_cov))

            return {
                'position_std_x_cm': pos_std[0] * 100,
                'position_std_y_cm': pos_std[1] * 100,
                'position_std_z_cm': pos_std[2] * 100,
                'position_3d_std_cm': np.linalg.norm(pos_std) * 100,
                'orientation_std_roll_deg': np.rad2deg(ori_std[0]),
                'orientation_std_pitch_deg': np.rad2deg(ori_std[1]),
                'orientation_std_yaw_deg': np.rad2deg(ori_std[2]),
                'orientation_3d_std_deg': np.rad2deg(np.linalg.norm(ori_std))
            }


# Example usage and testing
if __name__ == "__main__":
    # Create pose tracker for 20 cameras at 1000Hz
    tracker = CameraPoseTracker(num_cameras=20, update_rate_hz=1000.0)
    tracker.start()

    # Simulate sensor data for camera 0
    camera_id = 0
    start_time = time.time_ns()

    try:
        # Simulate 5 seconds of operation
        for i in range(5000):  # 1000Hz for 5 seconds
            current_time = time.time_ns()

            # Simulate IMU measurement (1000Hz)
            imu = IMUMeasurement(
                timestamp=current_time,
                angular_velocity=np.array([0.01, 0.02, 0.001]),  # Small rotation
                linear_acceleration=np.array([0.0, 0.0, 9.81]),  # Gravity
                camera_id=camera_id
            )
            tracker.add_imu_measurement(imu)

            # Simulate GPS measurement (10Hz)
            if i % 100 == 0:
                gps = GPSMeasurement(
                    timestamp=current_time,
                    position=np.array([1000.0, 2000.0, 100.0]) + np.random.randn(3) * 0.02,
                    position_covariance=np.eye(3) * 0.0025,  # 5cm standard deviation
                    rtk_fix_quality=2,  # RTK fixed
                    camera_id=camera_id
                )
                tracker.add_gps_measurement(gps)

            # Simulate VIO measurement (30Hz)
            if i % 33 == 0:
                vio_pose = np.eye(4)
                vio_pose[0:3, 3] = np.array([1000.0, 2000.0, 100.0])
                vio = VIOMeasurement(
                    timestamp=current_time,
                    relative_pose=vio_pose,
                    covariance=np.eye(6) * 0.01,
                    camera_id=camera_id,
                    feature_count=50
                )
                tracker.add_vio_measurement(vio)

            time.sleep(0.001)  # 1ms sleep for 1000Hz

        # Get final statistics
        stats = tracker.get_accuracy_statistics(camera_id)
        print("\n=== Final Accuracy Statistics ===")
        print(f"Position accuracy (3D): {stats['position_3d_std_cm']:.2f} cm")
        print(f"Orientation accuracy (3D): {stats['orientation_3d_std_deg']:.4f}°")

    finally:
        tracker.stop()
