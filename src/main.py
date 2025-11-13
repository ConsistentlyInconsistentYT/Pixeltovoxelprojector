#!/usr/bin/env python3
"""
Main Application - 8K Motion Tracking System

This is the main entry point for the multi-camera motion tracking system.

Features:
- Command-line interface
- Configuration loading and validation
- Component initialization
- Multi-camera orchestration
- Graceful shutdown
- Performance monitoring
- Error handling

Usage:
    python main.py --config config/system_config.yaml
    python main.py --config config/system_config.yaml --verbose
    python main.py --simulate  # Simulation mode for testing
"""

import sys
import os
import argparse
import logging
import time
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import system components
from camera.camera_manager import (
    CameraManager, CameraConfiguration, CameraPair,
    CameraType, ConnectionType
)
from voxel.grid_manager import VoxelGridManager, GridConfig
from detection.tracker import MultiTargetTracker
from fusion.fusion_manager import FusionManager, FusionConfig
from monitoring.system_monitor import SystemMonitor
from pipeline.processing_pipeline import ProcessingPipeline, PipelineConfig
from pipeline.pipeline_coordinator import PipelineCoordinator, CoordinatorConfig

# Version information
VERSION = "1.0.0"
APP_NAME = "8K Motion Tracking System"


class MotionTrackingSystem:
    """
    Main application class for 8K motion tracking system

    Coordinates:
    - Configuration loading
    - Component initialization
    - Pipeline execution
    - Shutdown and cleanup
    """

    def __init__(self, config_file: str, verbose: bool = False, simulate: bool = False):
        """
        Initialize motion tracking system

        Args:
            config_file: Path to YAML configuration file
            verbose: Enable verbose logging
            simulate: Run in simulation mode
        """
        self.config_file = config_file
        self.verbose = verbose
        self.simulate = simulate

        # Configuration
        self.config: Optional[Dict[str, Any]] = None

        # Components
        self.camera_manager: Optional[CameraManager] = None
        self.voxel_manager: Optional[VoxelGridManager] = None
        self.tracker: Optional[MultiTargetTracker] = None
        self.fusion_manager: Optional[FusionManager] = None
        self.monitor: Optional[SystemMonitor] = None
        self.pipeline: Optional[ProcessingPipeline] = None
        self.coordinator: Optional[PipelineCoordinator] = None

        # State
        self.initialized = False
        self.running = False

        # Setup logging
        self._setup_logging()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"{APP_NAME} v{VERSION}")

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configure root logger
        log_level = logging.DEBUG if self.verbose else logging.INFO

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "motion_tracking.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def load_configuration(self) -> bool:
        """
        Load and validate configuration from YAML file

        Returns:
            True if configuration loaded successfully
        """
        self.logger.info(f"Loading configuration from: {self.config_file}")

        config_path = Path(self.config_file)
        if not config_path.exists():
            self.logger.error(f"Configuration file not found: {self.config_file}")
            return False

        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            # Validate configuration
            if not self._validate_configuration():
                return False

            self.logger.info("Configuration loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            return False

    def _validate_configuration(self) -> bool:
        """Validate configuration"""
        required_sections = ['system', 'cameras', 'voxel_grid', 'detection', 'fusion', 'network', 'performance']

        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Missing required configuration section: {section}")
                return False

        self.logger.info("Configuration validation passed")
        return True

    def initialize_components(self) -> bool:
        """
        Initialize all system components

        Returns:
            True if all components initialized successfully
        """
        self.logger.info("Initializing system components...")

        try:
            # 1. Initialize Camera Manager
            if not self._initialize_camera_manager():
                return False

            # 2. Initialize Voxel Grid Manager
            if not self._initialize_voxel_manager():
                return False

            # 3. Initialize Fusion Manager
            if not self._initialize_fusion_manager():
                return False

            # 4. Initialize Tracker
            if not self._initialize_tracker():
                return False

            # 5. Initialize System Monitor
            if not self._initialize_monitor():
                return False

            # 6. Initialize Processing Pipeline
            if not self._initialize_pipeline():
                return False

            # 7. Initialize Pipeline Coordinator
            if not self._initialize_coordinator():
                return False

            self.initialized = True
            self.logger.info("All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}", exc_info=True)
            return False

    def _initialize_camera_manager(self) -> bool:
        """Initialize camera management system"""
        self.logger.info("Initializing camera manager...")

        camera_config = self.config['cameras']
        num_pairs = camera_config['num_pairs']

        self.camera_manager = CameraManager(num_pairs=num_pairs)

        # Configure camera pairs
        for pair_config in camera_config['pairs']:
            pair_id = pair_config['pair_id']

            # Mono camera
            mono_config = CameraConfiguration(
                camera_id=pair_config['mono']['camera_id'],
                pair_id=pair_id,
                camera_type=CameraType.MONO,
                connection=ConnectionType.GIGE_VISION,
                ip_address=pair_config['mono']['ip_address'],
                mac_address=pair_config['mono']['mac_address'],
                width=pair_config['mono']['width'],
                height=pair_config['mono']['height'],
                pixel_format=pair_config['mono']['pixel_format'],
                frame_rate=pair_config['mono']['frame_rate'],
                exposure_time=pair_config['mono']['exposure_time'],
                gain=pair_config['mono']['gain'],
                serial_number=pair_config['mono']['serial_number'],
                model_name=pair_config['mono']['model_name'],
                position=np.array(pair_config['position']),
                orientation=np.array(pair_config['orientation'])
            )
            self.camera_manager.add_camera(mono_config)

            # Thermal camera
            thermal_config = CameraConfiguration(
                camera_id=pair_config['thermal']['camera_id'],
                pair_id=pair_id,
                camera_type=CameraType.THERMAL,
                connection=ConnectionType.GIGE_VISION,
                ip_address=pair_config['thermal']['ip_address'],
                mac_address=pair_config['thermal']['mac_address'],
                width=pair_config['thermal']['width'],
                height=pair_config['thermal']['height'],
                pixel_format=pair_config['thermal']['pixel_format'],
                frame_rate=pair_config['thermal']['frame_rate'],
                exposure_time=pair_config['thermal']['exposure_time'],
                gain=pair_config['thermal']['gain'],
                serial_number=pair_config['thermal']['serial_number'],
                model_name=pair_config['thermal']['model_name'],
                position=np.array(pair_config['position']),
                orientation=np.array(pair_config['orientation'])
            )
            self.camera_manager.add_camera(thermal_config)

            # Create camera pair
            pair = CameraPair(
                pair_id=pair_id,
                mono_camera_id=pair_config['mono']['camera_id'],
                thermal_camera_id=pair_config['thermal']['camera_id'],
                stereo_baseline=pair_config['baseline_m']
            )
            self.camera_manager.add_pair(pair)

        # Initialize cameras (in simulation mode, this will succeed without hardware)
        if not self.simulate:
            if not self.camera_manager.initialize_all_cameras():
                self.logger.error("Failed to initialize cameras")
                return False

        self.logger.info(f"Camera manager initialized: {num_pairs} pairs, {num_pairs * 2} cameras")
        return True

    def _initialize_voxel_manager(self) -> bool:
        """Initialize voxel grid management system"""
        self.logger.info("Initializing voxel grid manager...")

        voxel_config = self.config['voxel_grid']

        grid_config = GridConfig(
            center=np.array(voxel_config['center']),
            size=np.array(voxel_config['size']),
            base_resolution=voxel_config['base_resolution'],
            max_memory_mb=voxel_config['max_memory_mb'],
            enable_lod=voxel_config['enable_lod'],
            enable_compression=voxel_config['enable_compression'],
            prune_threshold=voxel_config['prune_threshold'],
            update_rate_hz=voxel_config['update_rate_hz']
        )

        self.voxel_manager = VoxelGridManager(config=grid_config)

        self.logger.info("Voxel grid manager initialized")
        return True

    def _initialize_fusion_manager(self) -> bool:
        """Initialize fusion management system"""
        self.logger.info("Initializing fusion manager...")

        fusion_config_data = self.config['fusion']

        fusion_config = FusionConfig(
            target_fps=self.config['cameras']['pairs'][0]['mono']['frame_rate'],
            thermal_threshold=fusion_config_data['thermal_threshold'],
            mono_threshold=fusion_config_data['mono_threshold'],
            confidence_threshold=fusion_config_data['confidence_threshold'],
            max_range_km=5.0,
            enable_thermal_enhancement=fusion_config_data['enable_thermal_enhancement'],
            enable_false_positive_reduction=fusion_config_data['enable_false_positive_reduction'],
            enable_cuda=fusion_config_data['enable_cuda'],
            thermal_palette=fusion_config_data['thermal_palette'],
            low_light_threshold=fusion_config_data['low_light_threshold']
        )

        self.fusion_manager = FusionManager(config=fusion_config)

        # Add camera pairs to fusion manager
        for pair_config in self.config['cameras']['pairs']:
            self.fusion_manager.add_camera_pair(
                thermal_id=pair_config['thermal']['camera_id'],
                mono_id=pair_config['mono']['camera_id'],
                baseline_m=pair_config['baseline_m']
            )

        self.logger.info("Fusion manager initialized")
        return True

    def _initialize_tracker(self) -> bool:
        """Initialize multi-target tracker"""
        self.logger.info("Initializing tracker...")

        detection_config = self.config['detection']

        self.tracker = MultiTargetTracker(
            max_tracks=detection_config['max_tracks'],
            detection_threshold=detection_config['detection_confidence'],
            confirmation_threshold=detection_config['confirmation_threshold'],
            max_age=detection_config['max_age'],
            iou_threshold=detection_config['iou_threshold'],
            max_velocity=detection_config['max_velocity'],
            frame_rate=self.config['cameras']['pairs'][0]['mono']['frame_rate']
        )

        self.logger.info("Tracker initialized")
        return True

    def _initialize_monitor(self) -> bool:
        """Initialize system monitor"""
        self.logger.info("Initializing system monitor...")

        monitoring_config = self.config['monitoring']

        if not monitoring_config['enabled']:
            self.logger.info("System monitoring disabled")
            return True

        self.monitor = SystemMonitor(
            update_rate_hz=monitoring_config['update_rate_hz'],
            num_cameras=self.config['cameras']['num_pairs'] * 2,
            history_size=monitoring_config['history_size']
        )

        # Set references to other components
        self.monitor.set_camera_manager(self.camera_manager)
        self.monitor.set_tracker(self.tracker)

        self.logger.info("System monitor initialized")
        return True

    def _initialize_pipeline(self) -> bool:
        """Initialize processing pipeline"""
        self.logger.info("Initializing processing pipeline...")

        performance_config = self.config['performance']

        pipeline_config = PipelineConfig(
            target_fps=self.config['cameras']['pairs'][0]['mono']['frame_rate'],
            enable_fusion=True,
            enable_tracking=True,
            enable_voxel_updates=True,
            enable_streaming=self.config['network']['stream_coordinates'],
            enable_monitoring=self.config['monitoring']['enabled'],
            num_processing_threads=performance_config['num_processing_threads']
        )

        self.pipeline = ProcessingPipeline(
            config=pipeline_config,
            camera_manager=self.camera_manager,
            voxel_manager=self.voxel_manager,
            fusion_manager=self.fusion_manager,
            tracker=self.tracker,
            monitor=self.monitor
        )

        self.logger.info("Processing pipeline initialized")
        return True

    def _initialize_coordinator(self) -> bool:
        """Initialize pipeline coordinator"""
        self.logger.info("Initializing pipeline coordinator...")

        coordinator_config = CoordinatorConfig(
            health_check_interval_s=self.config['monitoring'].get('health_check_interval_s', 5.0),
            enable_auto_recovery=self.config['monitoring'].get('auto_recovery', True),
            max_recovery_attempts=self.config['monitoring'].get('max_recovery_attempts', 3)
        )

        self.coordinator = PipelineCoordinator(config=coordinator_config)

        # Register components
        self.coordinator.register_component('camera_manager', self.camera_manager, dependencies=[])
        self.coordinator.register_component('voxel_manager', self.voxel_manager, dependencies=[])
        self.coordinator.register_component('fusion_manager', self.fusion_manager, dependencies=['camera_manager'])
        self.coordinator.register_component('tracker', self.tracker, dependencies=[])
        if self.monitor:
            self.coordinator.register_component('monitor', self.monitor, dependencies=['camera_manager', 'tracker'])
        self.coordinator.register_component('pipeline', self.pipeline,
                                           dependencies=['camera_manager', 'voxel_manager', 'fusion_manager', 'tracker'])

        self.logger.info("Pipeline coordinator initialized")
        return True

    def start(self) -> bool:
        """
        Start the motion tracking system

        Returns:
            True if system started successfully
        """
        if not self.initialized:
            self.logger.error("System not initialized")
            return False

        self.logger.info("Starting motion tracking system...")

        try:
            # Start monitoring first
            if self.monitor:
                self.monitor.start()

            # Start processing pipeline
            self.pipeline.start()

            # Start coordinator (which monitors everything)
            if not self.coordinator.start_all():
                self.logger.error("Failed to start coordinator")
                return False

            self.running = True
            self.logger.info("Motion tracking system started successfully")

            # Print system status
            self._print_startup_status()

            return True

        except Exception as e:
            self.logger.error(f"Failed to start system: {e}", exc_info=True)
            return False

    def stop(self):
        """Stop the motion tracking system"""
        if not self.running:
            return

        self.logger.info("Stopping motion tracking system...")

        try:
            # Stop coordinator (gracefully stops all components)
            if self.coordinator:
                self.coordinator.stop_all()

            # Stop pipeline
            if self.pipeline:
                self.pipeline.stop()

            # Stop monitoring
            if self.monitor:
                self.monitor.stop()

            self.running = False
            self.logger.info("Motion tracking system stopped successfully")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)

    def run(self):
        """Main run loop"""
        if not self.running:
            self.logger.error("System not started")
            return

        self.logger.info("Entering main run loop...")

        try:
            # Main processing loop
            while self.running:
                # Process each camera pair
                for pair_id in range(self.config['cameras']['num_pairs']):
                    if self.simulate:
                        # In simulation mode, just sleep
                        time.sleep(1.0 / 30.0)
                    else:
                        # Process camera pair
                        self.pipeline.process_camera_pair(pair_id)

                # Check system status periodically
                if self.monitor and self.monitor.samples_collected % 100 == 0:
                    self._print_system_status()

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.stop()

    def _print_startup_status(self):
        """Print system status at startup"""
        self.logger.info("="*80)
        self.logger.info(f"{APP_NAME} v{VERSION} - STARTUP STATUS")
        self.logger.info("="*80)
        self.logger.info(f"Configuration: {self.config_file}")
        self.logger.info(f"Camera pairs: {self.config['cameras']['num_pairs']}")
        self.logger.info(f"Target FPS: {self.config['cameras']['pairs'][0]['mono']['frame_rate']}")
        self.logger.info(f"Voxel grid: {self.config['voxel_grid']['size']} @ {self.config['voxel_grid']['base_resolution']}m")
        self.logger.info(f"Max tracks: {self.config['detection']['max_tracks']}")
        self.logger.info(f"Simulation mode: {self.simulate}")
        self.logger.info("="*80)

    def _print_system_status(self):
        """Print current system status"""
        if self.monitor:
            summary = self.monitor.get_summary()
            self.logger.info(f"Status: {summary['overall_status']} | "
                           f"CPU: {summary['cpu_utilization']:.1f}% | "
                           f"Mem: {summary['memory_percent']:.1f}% | "
                           f"Cameras: {summary['cameras_online']}/{summary['cameras_total']} | "
                           f"Tracks: {summary['confirmed_tracks']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python main.py

  # Run with custom configuration
  python main.py --config my_config.yaml

  # Run in verbose mode
  python main.py --verbose

  # Run in simulation mode (no hardware required)
  python main.py --simulate

  # Validate configuration only
  python main.py --validate-config

For more information, visit: https://github.com/yourusername/motion-tracking
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/system_config.yaml',
        help='Path to configuration file (default: config/system_config.yaml)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run in simulation mode (no hardware required)'
    )

    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration and exit'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'{APP_NAME} v{VERSION}'
    )

    args = parser.parse_args()

    # Create system
    system = MotionTrackingSystem(
        config_file=args.config,
        verbose=args.verbose,
        simulate=args.simulate
    )

    # Load configuration
    if not system.load_configuration():
        print("Failed to load configuration", file=sys.stderr)
        return 1

    # Validate configuration only
    if args.validate_config:
        print("Configuration is valid")
        return 0

    # Initialize components
    if not system.initialize_components():
        print("Failed to initialize components", file=sys.stderr)
        return 1

    # Start system
    if not system.start():
        print("Failed to start system", file=sys.stderr)
        return 1

    # Run main loop
    system.run()

    return 0


if __name__ == '__main__':
    sys.exit(main())
