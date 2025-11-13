"""
Example Usage of Monitoring System

This script demonstrates how to integrate the complete monitoring system
with the Pixel-to-Voxel projection pipeline.
"""

import time
import logging
import numpy as np
from typing import List, Dict

# Import monitoring components
from src.monitoring import (
    SystemMonitor,
    DataValidator,
    AlertManager,
    WebDashboard,
    CoordinateBounds,
    AlertLevel,
    AlertCategory,
    ValidationLevel,
    create_default_rules
)

# Import existing system components (if available)
try:
    from src.camera.camera_manager import CameraManager
    from src.detection.tracker import MultiTargetTracker
    HAVE_CAMERA = True
except ImportError:
    HAVE_CAMERA = False
    print("Warning: Camera and tracker modules not found, using mock data")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_monitoring_system():
    """
    Setup complete monitoring system

    Returns:
        Tuple of (monitor, validator, alert_mgr, dashboard)
    """
    logger.info("Initializing monitoring system...")

    # 1. System Monitor - Real-time performance tracking
    monitor = SystemMonitor(
        update_rate_hz=10.0,      # 10Hz update rate
        num_cameras=20,            # 20 cameras (10 pairs)
        history_size=300           # 30 seconds at 10Hz
    )

    # 2. Data Validator - Comprehensive validation
    validator = DataValidator(
        bounds=CoordinateBounds(
            x_min=-5000.0, x_max=5000.0,
            y_min=-5000.0, y_max=5000.0,
            z_min=0.0, z_max=2000.0
        ),
        min_confidence=0.5,        # 50% minimum confidence
        max_velocity=100.0,        # 100 m/s max velocity
        max_acceleration=50.0      # 50 m/s² max acceleration
    )

    # 3. Alert Manager - Intelligent alerting
    alert_mgr = AlertManager(
        max_alerts_per_minute=100,
        alert_history_size=1000,
        enable_auto_diagnostics=True
    )

    # Configure email notifications (optional)
    # Uncomment and configure for production use
    """
    alert_mgr.configure_email(
        smtp_host='smtp.gmail.com',
        smtp_port=587,
        username='your_email@gmail.com',
        password='your_app_password',
        from_addr='alerts@yourcompany.com',
        to_addrs=['admin@yourcompany.com', 'ops@yourcompany.com']
    )
    """

    # Configure webhook (optional)
    # Uncomment for Slack/Discord/Teams integration
    """
    alert_mgr.configure_webhook(
        webhook_url='https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
    )
    """

    # Add default alert rules
    for rule in create_default_rules():
        alert_mgr.add_rule(rule)

    logger.info(f"Added {len(alert_mgr.rules)} default alert rules")

    # 4. Web Dashboard - Real-time visualization
    dashboard = WebDashboard(
        host='0.0.0.0',          # Listen on all interfaces
        port=5000,               # Default port
        update_rate_hz=2.0,      # 2Hz dashboard updates
        debug=False
    )

    # Link components
    alert_mgr.set_system_monitor(monitor)
    dashboard.set_system_monitor(monitor)
    dashboard.set_alert_manager(alert_mgr)
    dashboard.set_validator(validator)

    logger.info("Monitoring system initialized successfully")

    return monitor, validator, alert_mgr, dashboard


def integrate_with_camera_system(monitor, camera_manager):
    """
    Integrate monitoring with camera system

    Args:
        monitor: SystemMonitor instance
        camera_manager: CameraManager instance
    """
    # Set camera manager reference for monitoring
    monitor.set_camera_manager(camera_manager)

    # Register callback for camera events
    def camera_callback(metrics):
        """Callback for camera metric updates"""
        # Check for camera failures
        for cam_id, cam in metrics.cameras.items():
            if cam.health_status.value == 'critical':
                logger.warning(f"Camera {cam_id} in critical state: {cam.status}")

    monitor.register_callback(camera_callback)

    logger.info("Camera system integration complete")


def integrate_with_tracker(monitor, tracker):
    """
    Integrate monitoring with multi-target tracker

    Args:
        monitor: SystemMonitor instance
        tracker: MultiTargetTracker instance
    """
    # Set tracker reference for monitoring
    monitor.set_tracker(tracker)

    logger.info("Tracker integration complete")


def process_frame_with_monitoring(frame_data: Dict,
                                  validator: DataValidator,
                                  alert_mgr: AlertManager) -> Dict:
    """
    Process a single frame with validation and monitoring

    Args:
        frame_data: Frame data including detections
        validator: DataValidator instance
        alert_mgr: AlertManager instance

    Returns:
        Processed frame data with validation results
    """
    # Extract detections
    detections = frame_data.get('detections', [])

    # Validate each detection
    validation_results = []
    for i, detection in enumerate(detections):
        result = validator.validate_detection(detection, camera_id=frame_data.get('camera_id'))
        validation_results.append(result)

        # Create alerts for validation failures
        if not result.passed:
            for issue in result.issues:
                if issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
                    alert_mgr.create_alert(
                        level=AlertLevel.ERROR if issue.level == ValidationLevel.ERROR else AlertLevel.CRITICAL,
                        category=AlertCategory.DATA_QUALITY,
                        title=f"Detection validation failed",
                        message=issue.message,
                        camera_id=issue.camera_id,
                        detection_id=issue.detection_id,
                        current_value=issue.value,
                        expected_value=issue.expected,
                        suggestions=[issue.suggestion] if issue.suggestion else []
                    )

    # Add validation results to frame data
    frame_data['validation_results'] = validation_results
    frame_data['validation_passed'] = all(r.passed for r in validation_results)

    return frame_data


def simulate_processing_loop(monitor, validator, alert_mgr):
    """
    Simulate the main processing loop with monitoring

    Args:
        monitor: SystemMonitor instance
        validator: DataValidator instance
        alert_mgr: AlertManager instance
    """
    logger.info("Starting simulated processing loop...")

    # Simulation parameters
    num_cameras = 20
    frame_count = 0

    try:
        for frame_num in range(300):  # Simulate 10 seconds at 30 FPS
            frame_count += 1

            # Simulate detections for each camera
            for camera_id in range(num_cameras):
                # Generate mock detections
                num_detections = np.random.randint(0, 10)
                detections = []

                for det_id in range(num_detections):
                    # Random position within bounds
                    detection = {
                        'x': np.random.uniform(-1000, 1000),
                        'y': np.random.uniform(-1000, 1000),
                        'z': np.random.uniform(10, 500),
                        'confidence': np.random.uniform(0.6, 0.99),
                        'detection_id': det_id
                    }
                    detections.append(detection)

                # Process frame with monitoring
                frame_data = {
                    'camera_id': camera_id,
                    'frame_number': frame_count,
                    'detections': detections,
                    'timestamp': time.time()
                }

                frame_data = process_frame_with_monitoring(
                    frame_data, validator, alert_mgr
                )

            # Check alert rules periodically
            if frame_num % 30 == 0:  # Every second
                summary = monitor.get_summary()
                alert_mgr.check_rules(summary)

                # Print status
                logger.info(
                    f"Frame {frame_count}: "
                    f"CPU={summary['cpu_utilization']:.1f}%, "
                    f"Mem={summary['memory_percent']:.1f}%, "
                    f"Cameras={summary['cameras_online']}/{summary['cameras_total']}, "
                    f"Tracks={summary['confirmed_tracks']}, "
                    f"Alerts={len(alert_mgr.active_alerts)}"
                )

            # Simulate processing time
            time.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")


def main():
    """
    Main entry point demonstrating complete monitoring system usage
    """
    print("=" * 80)
    print("Pixel-to-Voxel Monitoring System - Example Usage")
    print("=" * 80)
    print()

    # Setup monitoring system
    monitor, validator, alert_mgr, dashboard = setup_monitoring_system()

    # Setup camera system if available
    if HAVE_CAMERA:
        camera_manager = CameraManager(num_pairs=10)
        tracker = MultiTargetTracker(max_tracks=200)

        integrate_with_camera_system(monitor, camera_manager)
        integrate_with_tracker(monitor, tracker)

    # Start monitoring
    logger.info("Starting monitoring services...")
    monitor.start()
    dashboard.start(blocking=False)

    print()
    print("Monitoring system is now active:")
    print(f"  - System Monitor: Running at 10Hz")
    print(f"  - Data Validator: Ready")
    print(f"  - Alert Manager: Active with {len(alert_mgr.rules)} rules")
    print(f"  - Web Dashboard: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop")
    print()

    try:
        # Run simulation or actual processing
        if not HAVE_CAMERA:
            simulate_processing_loop(monitor, validator, alert_mgr)
        else:
            # Real processing loop would go here
            while True:
                time.sleep(1)

                # Print periodic status
                summary = monitor.get_summary()
                logger.info(
                    f"Status: CPU={summary['cpu_utilization']:.1f}%, "
                    f"Mem={summary['memory_percent']:.1f}%, "
                    f"Alerts={len(alert_mgr.active_alerts)}"
                )

    except KeyboardInterrupt:
        print("\n\nShutting down monitoring system...")

    finally:
        # Stop monitoring
        monitor.stop()
        dashboard.stop()

        # Print final statistics
        print("\n" + "=" * 80)
        print("Final Statistics:")
        print("=" * 80)

        monitor_summary = monitor.get_summary()
        print("\nSystem Monitor:")
        print(f"  Samples collected: {monitor_summary['samples_collected']}")
        print(f"  Monitoring overhead: {monitor.get_performance_overhead():.3f}%")

        validator_stats = validator.get_statistics()
        print("\nData Validator:")
        print(f"  Total validations: {validator_stats['total_validations']}")
        print(f"  Total issues: {validator_stats['total_issues']}")
        print(f"  Avg validation time: {validator_stats['avg_validation_time_ms']:.2f} ms")

        alert_stats = alert_mgr.get_statistics()
        print("\nAlert Manager:")
        print(f"  Total alerts: {alert_stats['total_alerts_generated']}")
        print(f"  Active alerts: {alert_stats['active_alerts']}")
        print(f"  Notifications sent: {alert_stats['notifications_sent']}")

        print("\nMonitoring system stopped successfully")


if __name__ == "__main__":
    main()
