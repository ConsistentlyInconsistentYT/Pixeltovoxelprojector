"""
Example: Complete Camera Tracking System Integration

This example demonstrates how to use all three components together:
1. pose_tracker.py - Camera pose tracking with sensor fusion
2. orientation_manager.cpp - High-frequency orientation tracking
3. position_broadcast.py - Real-time position broadcasting

Author: Pixeltovoxelprojector Team
"""

import sys
import os
import numpy as np
import time
from scipy.spatial.transform import Rotation

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera.pose_tracker import (
    CameraPoseTracker,
    IMUMeasurement,
    GPSMeasurement,
    VIOMeasurement
)
from src.camera.position_broadcast import (
    PositionBroadcaster,
    PositionReceiver,
    BroadcastPose,
    CoordinateFrame,
    CameraCalibration
)


def simulate_sensor_data(camera_id, base_time, iteration):
    """
    Simulate realistic sensor data for testing

    In a real system, this would be replaced with actual sensor readings
    from RTK GPS, IMU, and VIO systems.
    """
    current_time = base_time + iteration * 1_000_000  # 1ms increments (nanoseconds)

    # Simulate IMU data (1000Hz)
    imu = IMUMeasurement(
        timestamp=current_time,
        angular_velocity=np.array([
            0.01 + 0.001 * np.sin(iteration * 0.01),
            0.02 + 0.001 * np.cos(iteration * 0.01),
            0.005
        ]),
        linear_acceleration=np.array([
            0.1 * np.sin(iteration * 0.001),
            0.05 * np.cos(iteration * 0.001),
            9.81 + 0.02 * np.random.randn()
        ]),
        camera_id=camera_id
    )

    # Simulate GPS data (10Hz)
    gps = None
    if iteration % 100 == 0:
        # ECEF coordinates near San Francisco
        base_ecef = np.array([-2707066.0, -4260952.0, 3884595.0])
        gps = GPSMeasurement(
            timestamp=current_time,
            position=base_ecef + np.array([camera_id * 10, 0, 0]) + np.random.randn(3) * 0.02,
            position_covariance=np.eye(3) * 0.0025,  # 5cm standard deviation
            rtk_fix_quality=2,  # RTK fixed
            camera_id=camera_id
        )

    # Simulate VIO data (30Hz)
    vio = None
    if iteration % 33 == 0:
        vio_pose = np.eye(4)
        vio_pose[0:3, 3] = np.array([camera_id * 10, 0, 0])
        vio = VIOMeasurement(
            timestamp=current_time,
            relative_pose=vio_pose,
            covariance=np.eye(6) * 0.01,
            camera_id=camera_id,
            feature_count=50 + int(10 * np.random.randn())
        )

    return imu, gps, vio


def main():
    """Main example demonstrating full system integration"""

    print("=" * 80)
    print("Camera Tracking System - Complete Integration Example")
    print("=" * 80)
    print()

    # Configuration
    NUM_CAMERAS = 20
    UPDATE_RATE_HZ = 1000.0
    DURATION_SECONDS = 10

    print(f"Configuration:")
    print(f"  Number of cameras: {NUM_CAMERAS}")
    print(f"  Update rate: {UPDATE_RATE_HZ} Hz")
    print(f"  Duration: {DURATION_SECONDS} seconds")
    print()

    # =========================================================================
    # Step 1: Initialize Pose Tracker
    # =========================================================================
    print("Step 1: Initializing Camera Pose Tracker...")
    pose_tracker = CameraPoseTracker(
        num_cameras=NUM_CAMERAS,
        update_rate_hz=UPDATE_RATE_HZ
    )
    pose_tracker.start()
    print("  ✓ Pose tracker started")
    print()

    # =========================================================================
    # Step 2: Initialize Position Broadcaster
    # =========================================================================
    print("Step 2: Initializing Position Broadcaster...")
    broadcaster = PositionBroadcaster(num_cameras=NUM_CAMERAS)

    # Set ENU reference point (San Francisco)
    broadcaster.set_enu_reference(
        lat_deg=37.7749,
        lon_deg=-122.4194,
        alt_m=0.0
    )

    # Set world frame (voxel grid reference)
    world_frame = CoordinateFrame(
        name="VoxelGrid",
        origin=np.array([0, 0, 0]),
        orientation=Rotation.from_euler('xyz', [0, 0, 0]),
        timestamp=time.time()
    )
    broadcaster.set_world_frame(world_frame)

    broadcaster.start()
    print("  ✓ Position broadcaster started")
    print()

    # =========================================================================
    # Step 3: Configure Camera Calibrations
    # =========================================================================
    print("Step 3: Configuring camera calibrations...")
    for camera_id in range(NUM_CAMERAS):
        # Example calibration (would come from actual calibration procedure)
        calibration = CameraCalibration(
            camera_id=camera_id,
            intrinsic_matrix=np.array([
                [4000.0, 0.0, 3840.0],
                [0.0, 4000.0, 2160.0],
                [0.0, 0.0, 1.0]
            ]),
            distortion_coeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            resolution=(7680, 4320),  # 8K resolution
            fov_horizontal=90.0,
            fov_vertical=60.0,
            timestamp=time.time()
        )
        broadcaster.update_calibration(calibration)
    print(f"  ✓ Calibrated {NUM_CAMERAS} cameras")
    print()

    # =========================================================================
    # Step 4: Optional - Start Position Receiver (for testing)
    # =========================================================================
    print("Step 4: Starting position receiver (for testing)...")
    receiver = PositionReceiver(server_address="localhost", port=5555)

    received_poses_count = [0] * NUM_CAMERAS

    def pose_callback(pose):
        """Callback for received poses"""
        received_poses_count[pose.camera_id] += 1

    receiver.start(pose_callback=pose_callback)
    print("  ✓ Position receiver started")
    print()

    # =========================================================================
    # Step 5: Main Processing Loop
    # =========================================================================
    print("Step 5: Running main processing loop...")
    print("-" * 80)

    base_time = time.time_ns()
    start_time = time.time()

    try:
        for iteration in range(int(DURATION_SECONDS * UPDATE_RATE_HZ)):
            # Process each camera
            for camera_id in range(NUM_CAMERAS):
                # Get simulated sensor data
                imu, gps, vio = simulate_sensor_data(camera_id, base_time, iteration)

                # Add measurements to pose tracker
                pose_tracker.add_imu_measurement(imu)
                if gps is not None:
                    pose_tracker.add_gps_measurement(gps)
                if vio is not None:
                    pose_tracker.add_vio_measurement(vio)

            # Get and broadcast poses every 10ms (100Hz broadcasting)
            if iteration % 10 == 0:
                for camera_id in range(NUM_CAMERAS):
                    # Get current pose from tracker
                    pose = pose_tracker.get_pose(camera_id)

                    if pose is not None:
                        # Create broadcast message
                        broadcast_pose = BroadcastPose(
                            camera_id=camera_id,
                            timestamp=pose.timestamp,
                            position_ecef=pose.position,
                            position_enu=np.zeros(3),  # Will be computed by broadcaster
                            position_world=np.zeros(3),  # Will be computed by broadcaster
                            orientation_quat=pose.orientation.as_quat(),
                            orientation_euler=pose.orientation.as_euler('xyz'),
                            velocity_ecef=pose.velocity,
                            angular_velocity=pose.angular_velocity,
                            position_std=np.sqrt(np.diag(pose.position_covariance)),
                            orientation_std=np.sqrt(np.diag(pose.orientation_covariance)),
                            rtk_fix_quality=2,
                            feature_count=50,
                            imu_health=0.95
                        )

                        # Broadcast pose
                        broadcaster.broadcast_pose(broadcast_pose)

            # Progress update every second
            if iteration % int(UPDATE_RATE_HZ) == 0:
                elapsed = time.time() - start_time
                progress = (iteration / (DURATION_SECONDS * UPDATE_RATE_HZ)) * 100
                print(f"  Progress: {progress:.1f}% ({elapsed:.1f}s elapsed)")

            # Maintain timing (1000Hz)
            time.sleep(1.0 / UPDATE_RATE_HZ)

        print("-" * 80)
        print("✓ Processing complete!")
        print()

        # =====================================================================
        # Step 6: Display Final Statistics
        # =====================================================================
        print("Step 6: Final Statistics")
        print("=" * 80)

        # Pose accuracy statistics
        print("\nPose Accuracy (Sample: Camera 0):")
        stats = pose_tracker.get_accuracy_statistics(0)
        if stats:
            print(f"  Position accuracy (3D): {stats['position_3d_std_cm']:.3f} cm")
            print(f"  Position X: {stats['position_std_x_cm']:.3f} cm")
            print(f"  Position Y: {stats['position_std_y_cm']:.3f} cm")
            print(f"  Position Z: {stats['position_std_z_cm']:.3f} cm")
            print(f"  Orientation accuracy (3D): {stats['orientation_3d_std_deg']:.4f}°")
            print(f"  Roll:  {stats['orientation_std_roll_deg']:.4f}°")
            print(f"  Pitch: {stats['orientation_std_pitch_deg']:.4f}°")
            print(f"  Yaw:   {stats['orientation_std_yaw_deg']:.4f}°")

        # Broadcast/receive statistics
        print("\nBroadcast Statistics:")
        total_received = sum(received_poses_count)
        print(f"  Total poses received: {total_received}")
        for camera_id in range(min(5, NUM_CAMERAS)):  # Show first 5 cameras
            print(f"  Camera {camera_id}: {received_poses_count[camera_id]} poses received")

        # System verification
        print("\nSystem Verification:")
        requirements_met = True

        # Check position accuracy
        if stats and stats['position_3d_std_cm'] < 5.0:
            print("  ✓ Position accuracy <5cm requirement: MET")
        else:
            print("  ✗ Position accuracy <5cm requirement: NOT MET")
            requirements_met = False

        # Check orientation accuracy
        if stats and stats['orientation_3d_std_deg'] < 0.1:
            print("  ✓ Orientation accuracy <0.1° requirement: MET")
        else:
            print("  ✗ Orientation accuracy <0.1° requirement: NOT MET")
            requirements_met = False

        # Check update rate
        actual_rate = (DURATION_SECONDS * UPDATE_RATE_HZ) / (time.time() - start_time)
        if actual_rate >= UPDATE_RATE_HZ * 0.95:  # 95% of target rate
            print(f"  ✓ Update rate {UPDATE_RATE_HZ}Hz requirement: MET ({actual_rate:.1f}Hz achieved)")
        else:
            print(f"  ✗ Update rate {UPDATE_RATE_HZ}Hz requirement: NOT MET ({actual_rate:.1f}Hz achieved)")
            requirements_met = False

        print()
        if requirements_met:
            print("✓ ALL REQUIREMENTS MET!")
        else:
            print("⚠ Some requirements not met (this is expected in simulation)")
        print()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # =====================================================================
        # Step 7: Cleanup
        # =====================================================================
        print("Step 7: Shutting down...")
        receiver.stop()
        broadcaster.stop()
        pose_tracker.stop()
        print("  ✓ All systems stopped")
        print()
        print("=" * 80)
        print("Example complete!")
        print("=" * 80)


if __name__ == "__main__":
    main()
