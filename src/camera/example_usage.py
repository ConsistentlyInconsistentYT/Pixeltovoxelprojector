"""
Example usage of the Multi-Camera Synchronization System

This script demonstrates:
1. Setting up 10 camera pairs (20 cameras)
2. Configuring synchronization (PTP + Hardware)
3. Calibrating camera pairs
4. Monitoring system health
5. Processing synchronized frames
"""

import time
import logging
import numpy as np
from pathlib import Path

# Import camera system modules
from camera_sync import CameraSynchronizer, SyncMode, FrameMetadata
from camera_manager import (
    CameraManager, CameraConfiguration, CameraType,
    ConnectionType, CameraPair
)
from pair_calibration import PairCalibrationManager, CalibrationTarget


def setup_camera_system(num_pairs: int = 10):
    """
    Setup complete camera system with 10 pairs

    Returns:
        Tuple of (camera_manager, synchronizer, calibration_manager)
    """
    logging.info(f"Setting up camera system with {num_pairs} pairs...")

    # Initialize managers
    camera_manager = CameraManager(num_pairs=num_pairs)
    synchronizer = CameraSynchronizer(num_pairs=num_pairs, sync_mode=SyncMode.HYBRID)
    calibration_manager = PairCalibrationManager(num_pairs=num_pairs)

    # Configure each camera pair
    for pair_id in range(num_pairs):
        # Mono camera (even ID)
        mono_id = pair_id * 2
        mono_config = CameraConfiguration(
            camera_id=mono_id,
            pair_id=pair_id,
            camera_type=CameraType.MONO,
            connection=ConnectionType.GIGE_VISION,
            ip_address=f"192.168.1.{10 + mono_id}",
            mac_address=f"00:11:22:33:44:{mono_id:02X}",
            width=1920,
            height=1080,
            frame_rate=30.0,
            exposure_time=10000.0,  # 10ms
            gain=1.0,
            trigger_mode="Hardware",
            trigger_source="Line1",
            packet_size=9000,
            serial_number=f"MONO_{pair_id:02d}",
            model_name="Basler acA1920-40gm"
        )

        # Set physical position (cameras arranged in a grid)
        row = pair_id // 5
        col = pair_id % 5
        mono_config.position = np.array([col * 0.5, row * 0.5, 2.0])  # meters
        mono_config.orientation = np.eye(3)  # Looking straight down

        camera_manager.add_camera(mono_config)

        # Thermal camera (odd ID)
        thermal_id = pair_id * 2 + 1
        thermal_config = CameraConfiguration(
            camera_id=thermal_id,
            pair_id=pair_id,
            camera_type=CameraType.THERMAL,
            connection=ConnectionType.GIGE_VISION,
            ip_address=f"192.168.1.{10 + thermal_id}",
            mac_address=f"00:11:22:33:44:{thermal_id:02X}",
            width=640,
            height=480,
            frame_rate=30.0,
            exposure_time=8000.0,  # 8ms
            gain=1.2,
            trigger_mode="Hardware",
            trigger_source="Line1",
            packet_size=9000,
            serial_number=f"THERMAL_{pair_id:02d}",
            model_name="FLIR A65"
        )

        # Thermal camera slightly offset from mono
        thermal_config.position = mono_config.position + np.array([0.12, 0, 0])  # 12cm offset
        thermal_config.orientation = np.eye(3)

        camera_manager.add_camera(thermal_config)

        # Create camera pair
        pair = CameraPair(
            pair_id=pair_id,
            mono_camera_id=mono_id,
            thermal_camera_id=thermal_id,
            stereo_baseline=0.12  # 12cm
        )
        camera_manager.add_pair(pair)

    logging.info("Camera system configured successfully")
    return camera_manager, synchronizer, calibration_manager


def calibrate_camera_pairs(camera_manager, calibration_manager):
    """
    Calibrate all camera pairs

    This would typically be done once during system setup or when
    cameras are repositioned.
    """
    logging.info("Starting calibration process...")

    num_pairs = calibration_manager.num_pairs
    calibration_target = CalibrationTarget(
        target_type="checkerboard",
        rows=9,
        cols=6,
        square_size=0.025  # 25mm squares
    )

    for pair_id in range(num_pairs):
        mono_id = pair_id * 2
        thermal_id = pair_id * 2 + 1

        logging.info(f"Calibrating pair {pair_id}...")

        # 1. Calibrate intrinsics for each camera
        # In real implementation, capture multiple images of calibration target
        num_calib_images = 20

        # Simulate calibration image capture
        mono_images = [np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
                      for _ in range(num_calib_images)]
        thermal_images = [np.random.randint(0, 255, (480, 640), dtype=np.uint8)
                         for _ in range(num_calib_images)]

        # Simulate detected points (in real implementation, use cv2.findChessboardCorners)
        mono_points = [np.random.rand(54, 2) * [1920, 1080]
                      for _ in range(num_calib_images)]
        thermal_points = [np.random.rand(54, 2) * [640, 480]
                         for _ in range(num_calib_images)]

        # Calibrate mono camera
        calibration_manager.calibrate_camera_intrinsic(
            mono_id, mono_images, mono_points
        )

        # Calibrate thermal camera
        calibration_manager.calibrate_camera_intrinsic(
            thermal_id, thermal_images, thermal_points
        )

        # 2. Stereo calibration
        calibration_manager.calibrate_stereo_pair(
            pair_id, mono_images, thermal_images, mono_points, thermal_points
        )

        # 3. Mono-thermal registration
        # Use synchronized image pair
        calibration_manager.register_mono_thermal_pair(
            pair_id, mono_images[0], thermal_images[0]
        )

        # Save calibration data
        calibration_manager.save_calibration(pair_id, directory="calibration")

        logging.info(f"Pair {pair_id} calibration complete")

    # Generate calibration report
    report = calibration_manager.get_calibration_report()
    logging.info(f"Calibration Summary: {report['summary']}")

    return report


def simulate_camera_frames(synchronizer, num_pairs: int, duration: float = 10.0):
    """
    Simulate camera frame acquisition and synchronization

    In real implementation, this would be replaced by actual camera callbacks
    """
    logging.info(f"Simulating frame acquisition for {duration}s...")

    start_time = time.time()
    frame_number = 0
    fps = 30.0
    frame_period = 1.0 / fps

    while time.time() - start_time < duration:
        frame_time = time.time()

        # Simulate hardware trigger
        trigger_id = frame_number

        # Generate frames for all cameras
        for pair_id in range(num_pairs):
            mono_id = pair_id * 2
            thermal_id = pair_id * 2 + 1

            # Simulate small timestamp offset between mono and thermal
            mono_offset = np.random.normal(0, 0.0001)  # 0.1ms jitter
            thermal_offset = np.random.normal(0, 0.0001)

            # Create mono frame metadata
            mono_metadata = FrameMetadata(
                camera_id=mono_id,
                pair_id=pair_id,
                frame_number=frame_number,
                timestamp=frame_time + mono_offset,
                system_time=frame_time,
                trigger_id=trigger_id,
                exposure_time=10000.0,
                gain=1.0,
                temperature=45.0 + np.random.randn()
            )

            # Create thermal frame metadata
            thermal_metadata = FrameMetadata(
                camera_id=thermal_id,
                pair_id=pair_id,
                frame_number=frame_number,
                timestamp=frame_time + thermal_offset,
                system_time=frame_time,
                trigger_id=trigger_id,
                exposure_time=8000.0,
                gain=1.2,
                temperature=48.0 + np.random.randn()
            )

            # Add frames to synchronizer
            synchronizer.add_frame(mono_id, mono_metadata)
            synchronizer.add_frame(thermal_id, thermal_metadata)

        frame_number += 1

        # Sleep to maintain frame rate
        elapsed = time.time() - frame_time
        sleep_time = max(0, frame_period - elapsed)
        time.sleep(sleep_time)

    logging.info(f"Simulation complete: {frame_number} frames generated")


def process_synchronized_frames(synchronizer, duration: float = 10.0):
    """
    Process synchronized frame sets from all pairs

    This demonstrates how to retrieve and use synchronized frames
    """
    logging.info("Processing synchronized frames...")

    start_time = time.time()
    processed_count = 0

    while time.time() - start_time < duration:
        # Get next synchronized frame set
        synced_set = synchronizer.get_synced_frame_set(timeout=0.1)

        if synced_set and synced_set.is_valid:
            # Process the synchronized pair
            pair_id = synced_set.pair_id
            mono_frame = synced_set.mono_frame
            thermal_frame = synced_set.thermal_frame
            sync_error = synced_set.sync_error

            # In real implementation:
            # - Load actual image data
            # - Apply calibration corrections
            # - Perform stereo processing
            # - Register thermal overlay
            # - Run computer vision algorithms

            if processed_count % 100 == 0:
                logging.info(
                    f"Pair {pair_id}: Frame {mono_frame.frame_number}, "
                    f"Sync error: {sync_error:.3f}ms, "
                    f"Recovery: {synced_set.recovery_applied}"
                )

            processed_count += 1

    logging.info(f"Processed {processed_count} synchronized frame sets")
    return processed_count


def monitor_system_health(camera_manager, synchronizer, interval: float = 5.0):
    """
    Monitor and display system health metrics
    """
    logging.info("=" * 80)
    logging.info("SYSTEM HEALTH REPORT")
    logging.info("=" * 80)

    # Camera manager health
    system_health = camera_manager.get_system_health()
    logging.info(f"\nCamera System:")
    logging.info(f"  Total Cameras: {system_health['total_cameras']}")
    logging.info(f"  Streaming: {system_health['streaming']}")
    logging.info(f"  Ready: {system_health['ready']}")
    logging.info(f"  Error: {system_health['error']}")
    logging.info(f"  Avg FPS: {system_health['avg_fps']:.1f}")
    logging.info(f"  Avg Temperature: {system_health['avg_temperature']:.1f}°C")
    logging.info(f"  Frame Drop Rate: {system_health['drop_rate']*100:.3f}%")

    # Synchronization metrics
    sync_metrics = synchronizer.get_synchronization_metrics()
    logging.info(f"\nSynchronization:")
    logging.info(f"  Mode: {sync_metrics['sync_mode']}")
    logging.info(f"  Avg Sync Error: {sync_metrics['overall']['avg_sync_error_ms']:.4f}ms")
    logging.info(f"  Max Sync Error: {sync_metrics['overall']['max_sync_error_ms']:.4f}ms")
    logging.info(f"  Avg Jitter: {sync_metrics['overall']['avg_jitter_us']:.2f}μs")
    logging.info(f"  Total Frames: {sync_metrics['overall']['total_frames']}")
    logging.info(f"  Dropped: {sync_metrics['overall']['total_dropped']}")
    logging.info(f"  Recovered: {sync_metrics['overall']['total_recovered']}")

    # PTP Quality
    ptp_quality = sync_metrics['ptp_quality']
    logging.info(f"\nPTP Synchronization:")
    logging.info(f"  Synchronized: {ptp_quality['is_synced']}")
    logging.info(f"  Master Offset: {ptp_quality['offset']:.4f}ms")
    logging.info(f"  Jitter: {ptp_quality['jitter']:.2f}μs")

    # Hardware Trigger
    trigger_stats = sync_metrics['trigger_stats']
    logging.info(f"\nHardware Trigger:")
    logging.info(f"  Total Triggers: {trigger_stats['total_triggers']}")
    logging.info(f"  Avg Response Rate: {trigger_stats.get('avg_response_rate', 0)*100:.1f}%")

    # Per-pair health
    logging.info(f"\nPer-Pair Status:")
    for pair_id in range(camera_manager.num_pairs):
        pair_health = camera_manager.get_pair_health(pair_id)
        if pair_health:
            mono_status = pair_health['mono_health'].status.name
            thermal_status = pair_health['thermal_health'].status.name
            sync_quality = pair_health['sync_quality']
            logging.info(
                f"  Pair {pair_id}: Mono={mono_status}, Thermal={thermal_status}, "
                f"Sync={sync_quality:.1f}%"
            )

    logging.info("=" * 80)


def main():
    """Main demonstration of the camera synchronization system"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Multi-Camera Synchronization System Demo")
    logging.info("=" * 80)

    # 1. Setup camera system
    num_pairs = 10
    camera_manager, synchronizer, calibration_manager = setup_camera_system(num_pairs)

    # 2. Initialize cameras
    logging.info("\nInitializing cameras...")
    if not camera_manager.initialize_all_cameras():
        logging.error("Failed to initialize all cameras")
        return

    # 3. Load or perform calibration
    calibration_dir = Path("calibration")
    if calibration_dir.exists():
        logging.info("\nLoading existing calibrations...")
        for pair_id in range(num_pairs):
            calibration_manager.load_calibration(pair_id)
    else:
        logging.info("\nPerforming initial calibration...")
        calibrate_camera_pairs(camera_manager, calibration_manager)

    # 4. Start calibration monitoring
    calibration_manager.start_realtime_updates(update_interval=60.0)

    # 5. Start synchronization system
    logging.info("\nStarting synchronization system...")
    synchronizer.start()
    synchronizer.set_sync_tolerance(1.0)  # 1ms tolerance

    # Simulate PTP synchronization
    for i in range(10):
        master_time = time.time()
        local_time = time.time() - 0.0005  # 0.5ms offset
        synchronizer.update_ptp_sync(master_time, local_time)
        time.sleep(0.1)

    # 6. Start camera acquisition
    logging.info("\nStarting camera acquisition...")
    camera_manager.start_all_acquisition()
    camera_manager.start_health_monitoring()

    # 7. Simulate frame acquisition (in separate thread in real implementation)
    import threading
    sim_duration = 10.0
    sim_thread = threading.Thread(
        target=simulate_camera_frames,
        args=(synchronizer, num_pairs, sim_duration),
        daemon=True
    )
    sim_thread.start()

    # 8. Process synchronized frames
    process_synchronized_frames(synchronizer, duration=sim_duration)

    # Wait for simulation to complete
    sim_thread.join()

    # 9. Monitor system health
    time.sleep(1.0)  # Let stats accumulate
    monitor_system_health(camera_manager, synchronizer)

    # 10. Cleanup
    logging.info("\nShutting down system...")
    camera_manager.stop_health_monitoring()
    camera_manager.stop_all_acquisition()
    synchronizer.stop()
    calibration_manager.stop_realtime_updates()
    camera_manager.disconnect_all()

    # Save final configuration
    camera_manager.save_configuration("camera_config.json")

    # Generate final calibration report
    calib_report = calibration_manager.get_calibration_report()
    logging.info(f"\nFinal Calibration Status:")
    logging.info(f"  Calibrated Pairs: {calib_report['summary']['calibrated_pairs']}/{num_pairs}")
    logging.info(f"  Avg Stereo Error: {calib_report['summary']['avg_stereo_error']:.3f}px")
    logging.info(f"  Avg Registration Error: {calib_report['summary']['avg_registration_error']:.3f}px")

    logging.info("\nDemo complete!")


if __name__ == "__main__":
    main()
