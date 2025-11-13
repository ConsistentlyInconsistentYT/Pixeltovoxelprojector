#!/usr/bin/env python3
"""
Quick Start Demo - 8K Motion Tracking System

This script demonstrates basic usage of the motion tracking system
in simulation mode (no hardware required).

Usage:
    python quick_start.py
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import MotionTrackingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*20 + "8K MOTION TRACKING SYSTEM - QUICK START DEMO")
    print("="*80)
    print()
    print("This demo runs the system in SIMULATION MODE")
    print("No camera hardware is required.")
    print()
    print("The system will:")
    print("  1. Load configuration")
    print("  2. Initialize all components")
    print("  3. Start processing pipeline")
    print("  4. Display system metrics")
    print("  5. Run for 30 seconds")
    print()
    print("="*80 + "\n")


def custom_coordinate_callback(result):
    """Custom callback for displaying coordinate data"""
    print(f"\n[Frame {result.frame_number}] Pair {result.pair_id}")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  Latency: {result.latency_ms:.2f}ms")
    print(f"  Motion coordinates: {len(result.motion_coordinates)}")
    print(f"  Fused detections: {len(result.fused_detections)}")
    print(f"  Confirmed tracks: {len(result.confirmed_tracks)}")

    if result.confirmed_tracks:
        print(f"  Active tracks:")
        for track in result.confirmed_tracks[:5]:  # Show first 5
            print(f"    Track {track['track_id']}: "
                  f"({track['x']:.1f}, {track['y']:.1f}) "
                  f"confidence={track['confidence']:.2f}")


def main():
    """Main demo function"""
    print_banner()

    # Create system in simulation mode
    logger.info("Creating motion tracking system...")
    system = MotionTrackingSystem(
        config_file='src/config/system_config.yaml',
        verbose=False,  # Set to True for detailed logs
        simulate=True   # Simulation mode - no hardware required
    )

    # Load configuration
    logger.info("Loading configuration...")
    if not system.load_configuration():
        logger.error("Failed to load configuration")
        return 1

    logger.info("Configuration loaded successfully")
    logger.info(f"  Camera pairs: {system.config['cameras']['num_pairs']}")
    logger.info(f"  Target FPS: {system.config['cameras']['pairs'][0]['mono']['frame_rate']}")
    logger.info(f"  Max tracks: {system.config['detection']['max_tracks']}")

    # Initialize components
    logger.info("\nInitializing components...")
    if not system.initialize_components():
        logger.error("Failed to initialize components")
        return 1

    logger.info("Components initialized successfully")

    # Register coordinate callback
    logger.info("\nRegistering coordinate callback...")
    system.pipeline.register_coordinate_callback(custom_coordinate_callback)

    # Start system
    logger.info("\nStarting motion tracking system...")
    if not system.start():
        logger.error("Failed to start system")
        return 1

    logger.info("System started successfully!\n")

    try:
        # Run for 30 seconds
        duration = 30.0
        start_time = time.time()

        logger.info(f"Running for {duration} seconds...")
        logger.info("Press Ctrl+C to stop early\n")

        while (time.time() - start_time) < duration:
            # Display system status every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                if system.monitor:
                    summary = system.monitor.get_summary()
                    print("\n" + "="*60)
                    print("SYSTEM STATUS")
                    print("="*60)
                    print(f"Status: {summary['overall_status']}")
                    print(f"Uptime: {summary.get('uptime_seconds', 0):.1f}s")
                    print(f"CPU: {summary['cpu_utilization']:.1f}%")
                    print(f"Memory: {summary['memory_percent']:.1f}%")
                    print(f"Cameras online: {summary['cameras_online']}/{summary['cameras_total']}")
                    print(f"Confirmed tracks: {summary['confirmed_tracks']}")
                    print(f"Monitoring overhead: {summary['monitoring_overhead_percent']:.3f}%")
                    print("="*60 + "\n")

                # Display pipeline metrics
                if system.pipeline:
                    metrics = system.pipeline.get_metrics()
                    print("PIPELINE METRICS")
                    print(f"  Frames processed: {metrics['frames_processed']}")
                    print(f"  Frames dropped: {metrics['frames_dropped']}")
                    print(f"  Avg processing time: {metrics['avg_processing_time_ms']:.2f}ms")
                    print(f"  Avg latency: {metrics['avg_latency_ms']:.2f}ms")
                    print(f"  Throughput: {metrics['throughput_fps']:.1f} FPS")
                    print()

                time.sleep(1)
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt")

    finally:
        # Stop system
        logger.info("\nStopping system...")
        system.stop()

        # Print final statistics
        print("\n" + "="*80)
        print("FINAL STATISTICS")
        print("="*80)

        if system.pipeline:
            metrics = system.pipeline.get_metrics()
            print(f"Total frames processed: {metrics['frames_processed']}")
            print(f"Total frames dropped: {metrics['frames_dropped']}")
            print(f"Average processing time: {metrics['avg_processing_time_ms']:.2f}ms")
            print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
            print(f"Overall throughput: {metrics['throughput_fps']:.1f} FPS")

        if system.tracker:
            tracker_stats = system.tracker.get_performance_summary()
            print(f"\nTracking statistics:")
            print(f"  Total tracks created: {tracker_stats['total_tracks_created']}")
            print(f"  Total tracks confirmed: {tracker_stats['total_tracks_confirmed']}")
            print(f"  Total tracks lost: {tracker_stats['total_tracks_lost']}")
            print(f"  Average track length: {tracker_stats['avg_track_length']:.1f}")
            print(f"  Frames processed: {tracker_stats['frames_processed']}")

        if system.coordinator:
            status = system.coordinator.get_system_status()
            print(f"\nSystem statistics:")
            print(f"  Total errors: {status['statistics']['total_errors']}")
            print(f"  Total recoveries: {status['statistics']['total_recoveries']}")
            print(f"  Final health: {status['overall_health']}")

        print("="*80 + "\n")

        logger.info("Demo completed successfully!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
