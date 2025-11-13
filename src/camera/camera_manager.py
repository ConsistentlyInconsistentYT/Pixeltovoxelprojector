"""
Camera Management System
Manages camera pair configuration, calibration, positioning, and health monitoring
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import json
from pathlib import Path


class CameraType(Enum):
    """Camera types in the system"""
    MONO = 1
    THERMAL = 2


class CameraStatus(Enum):
    """Camera operational status"""
    OFFLINE = 0
    INITIALIZING = 1
    READY = 2
    STREAMING = 3
    ERROR = 4
    MAINTENANCE = 5


class ConnectionType(Enum):
    """Camera connection interface"""
    GIGE_VISION = 1
    USB3 = 2
    CAMERA_LINK = 3


@dataclass
class CameraConfiguration:
    """Configuration parameters for a single camera"""
    camera_id: int
    pair_id: int
    camera_type: CameraType
    connection: ConnectionType

    # Network configuration (for GigE)
    ip_address: str = ""
    mac_address: str = ""
    subnet_mask: str = "255.255.255.0"
    gateway: str = ""

    # Camera settings
    width: int = 1920
    height: int = 1080
    pixel_format: str = "Mono8"
    frame_rate: float = 30.0
    exposure_time: float = 10000.0  # microseconds
    gain: float = 1.0

    # Trigger settings
    trigger_mode: str = "Hardware"  # Hardware, Software, Freerun
    trigger_source: str = "Line1"
    trigger_delay: float = 0.0  # microseconds

    # Advanced settings
    packet_size: int = 9000  # MTU for GigE
    packet_delay: int = 1000  # Inter-packet delay
    bandwidth_reserve: float = 10.0  # Percent

    # Physical location
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z] in meters
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))  # Rotation matrix

    # Calibration file paths
    intrinsic_file: str = ""
    extrinsic_file: str = ""

    # Serial number and model
    serial_number: str = ""
    model_name: str = ""
    firmware_version: str = ""


@dataclass
class CameraHealth:
    """Health monitoring data for a camera"""
    camera_id: int
    status: CameraStatus = CameraStatus.OFFLINE
    last_update: float = field(default_factory=time.time)

    # Connection health
    connection_stable: bool = False
    reconnect_count: int = 0
    last_reconnect: float = 0.0

    # Frame statistics
    frames_received: int = 0
    frames_dropped: int = 0
    frames_corrupted: int = 0
    current_fps: float = 0.0
    target_fps: float = 30.0

    # Network health (for GigE)
    packet_loss_rate: float = 0.0
    resend_requests: int = 0
    bandwidth_usage: float = 0.0  # MB/s

    # Hardware health
    temperature: float = 0.0  # Celsius
    max_temperature: float = 60.0
    power_voltage: float = 12.0  # Volts
    sensor_errors: int = 0

    # Performance metrics
    frame_latency: float = 0.0  # ms
    processing_time: float = 0.0  # ms
    buffer_usage: float = 0.0  # Percent

    # Error tracking
    error_count: int = 0
    last_error: str = ""
    last_error_time: float = 0.0


@dataclass
class CameraPair:
    """Camera pair configuration and state"""
    pair_id: int
    mono_camera_id: int
    thermal_camera_id: int

    # Pair calibration
    stereo_baseline: float = 0.0  # Distance between cameras in meters
    relative_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    relative_translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Registration parameters (mono-thermal)
    registration_matrix: Optional[np.ndarray] = None
    registration_quality: float = 0.0

    # Pair health
    sync_quality: float = 100.0  # Percentage
    calibration_age: float = 0.0  # Days since last calibration
    needs_recalibration: bool = False


class CameraInterface:
    """Abstract interface for camera control (GigE Vision implementation)"""

    def __init__(self, config: CameraConfiguration):
        self.config = config
        self.is_connected = False
        self.is_streaming = False
        self.frame_callback = None
        self.lock = threading.Lock()

        # Simulated camera state
        self.frame_counter = 0
        self.last_frame_time = 0.0

    def connect(self) -> bool:
        """Connect to camera"""
        logging.info(f"Connecting to camera {self.config.camera_id} at {self.config.ip_address}")

        try:
            # In real implementation:
            # - Initialize GigE transport layer
            # - Open camera by IP/MAC
            # - Configure network parameters
            # - Load camera features

            # Simulated connection
            time.sleep(0.1)
            self.is_connected = True
            logging.info(f"Camera {self.config.camera_id} connected successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to connect camera {self.config.camera_id}: {e}")
            return False

    def disconnect(self):
        """Disconnect from camera"""
        if self.is_streaming:
            self.stop_acquisition()

        self.is_connected = False
        logging.info(f"Camera {self.config.camera_id} disconnected")

    def configure(self) -> bool:
        """Configure camera parameters"""
        if not self.is_connected:
            logging.error(f"Camera {self.config.camera_id} not connected")
            return False

        try:
            # In real implementation, set camera features:
            # - Width, Height, PixelFormat
            # - ExposureTime, Gain
            # - TriggerMode, TriggerSource
            # - PacketSize, InterPacketDelay
            # - AcquisitionFrameRate

            logging.info(f"Camera {self.config.camera_id} configured: "
                        f"{self.config.width}x{self.config.height} @ {self.config.frame_rate}fps")
            return True

        except Exception as e:
            logging.error(f"Failed to configure camera {self.config.camera_id}: {e}")
            return False

    def start_acquisition(self, callback=None) -> bool:
        """Start frame acquisition"""
        if not self.is_connected:
            return False

        self.frame_callback = callback
        self.is_streaming = True
        self.frame_counter = 0
        self.last_frame_time = time.time()

        # In real implementation: camera.StartGrabbing()
        logging.info(f"Camera {self.config.camera_id} started acquisition")
        return True

    def stop_acquisition(self):
        """Stop frame acquisition"""
        self.is_streaming = False
        # In real implementation: camera.StopGrabbing()
        logging.info(f"Camera {self.config.camera_id} stopped acquisition")

    def grab_frame(self) -> Optional[Tuple[np.ndarray, Dict]]:
        """Grab single frame (blocking)"""
        if not self.is_streaming:
            return None

        # In real implementation:
        # - Wait for frame from camera
        # - Extract image data and metadata
        # - Handle timeouts and errors

        # Simulated frame grab
        frame = np.random.randint(0, 255, (self.config.height, self.config.width), dtype=np.uint8)
        metadata = {
            'frame_number': self.frame_counter,
            'timestamp': time.time(),
            'exposure_time': self.config.exposure_time,
            'gain': self.config.gain
        }

        self.frame_counter += 1
        self.last_frame_time = time.time()

        return frame, metadata

    def set_exposure(self, exposure_us: float) -> bool:
        """Set exposure time in microseconds"""
        if not self.is_connected:
            return False

        self.config.exposure_time = exposure_us
        # In real implementation: camera.ExposureTime.SetValue(exposure_us)
        return True

    def set_gain(self, gain: float) -> bool:
        """Set camera gain"""
        if not self.is_connected:
            return False

        self.config.gain = gain
        # In real implementation: camera.Gain.SetValue(gain)
        return True

    def get_temperature(self) -> float:
        """Get camera temperature"""
        # In real implementation: return camera.DeviceTemperature.GetValue()
        return 45.0 + np.random.randn() * 2.0  # Simulated


class CameraManager:
    """Main camera management system for 10 pairs (20 cameras)"""

    def __init__(self, num_pairs: int = 10):
        self.num_pairs = num_pairs
        self.num_cameras = num_pairs * 2

        # Camera components
        self.cameras: Dict[int, CameraInterface] = {}
        self.configurations: Dict[int, CameraConfiguration] = {}
        self.health: Dict[int, CameraHealth] = {}
        self.pairs: Dict[int, CameraPair] = {}

        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds

        # Configuration storage
        self.config_file = Path("camera_config.json")

        logging.info(f"CameraManager initialized for {num_pairs} pairs ({self.num_cameras} cameras)")

    def add_camera(self, config: CameraConfiguration):
        """Add camera to management system"""
        camera_id = config.camera_id

        if camera_id in self.cameras:
            logging.warning(f"Camera {camera_id} already exists, updating configuration")

        self.configurations[camera_id] = config
        self.cameras[camera_id] = CameraInterface(config)
        self.health[camera_id] = CameraHealth(camera_id=camera_id)

        logging.info(f"Added camera {camera_id} (Pair {config.pair_id}, Type: {config.camera_type.name})")

    def add_pair(self, pair: CameraPair):
        """Add camera pair"""
        if pair.mono_camera_id not in self.cameras or pair.thermal_camera_id not in self.cameras:
            logging.error(f"Cannot add pair {pair.pair_id}: cameras not found")
            return False

        self.pairs[pair.pair_id] = pair
        logging.info(f"Added camera pair {pair.pair_id}: "
                    f"Mono={pair.mono_camera_id}, Thermal={pair.thermal_camera_id}")
        return True

    def initialize_all_cameras(self) -> bool:
        """Initialize all cameras"""
        logging.info("Initializing all cameras...")
        success_count = 0

        for camera_id, camera in self.cameras.items():
            self.health[camera_id].status = CameraStatus.INITIALIZING

            if camera.connect():
                if camera.configure():
                    self.health[camera_id].status = CameraStatus.READY
                    self.health[camera_id].connection_stable = True
                    success_count += 1
                else:
                    self.health[camera_id].status = CameraStatus.ERROR
            else:
                self.health[camera_id].status = CameraStatus.OFFLINE

        logging.info(f"Initialized {success_count}/{self.num_cameras} cameras")
        return success_count == self.num_cameras

    def start_all_acquisition(self) -> bool:
        """Start acquisition on all cameras"""
        logging.info("Starting acquisition on all cameras...")
        success_count = 0

        for camera_id, camera in self.cameras.items():
            if camera.start_acquisition():
                self.health[camera_id].status = CameraStatus.STREAMING
                success_count += 1

        logging.info(f"Started acquisition on {success_count}/{self.num_cameras} cameras")
        return success_count == self.num_cameras

    def stop_all_acquisition(self):
        """Stop acquisition on all cameras"""
        logging.info("Stopping acquisition on all cameras...")

        for camera in self.cameras.values():
            camera.stop_acquisition()

        for health in self.health.values():
            if health.status == CameraStatus.STREAMING:
                health.status = CameraStatus.READY

    def disconnect_all(self):
        """Disconnect all cameras"""
        self.stop_all_acquisition()

        for camera in self.cameras.values():
            camera.disconnect()

        for health in self.health.values():
            health.status = CameraStatus.OFFLINE
            health.connection_stable = False

    def start_health_monitoring(self):
        """Start health monitoring thread"""
        if self.monitoring_active:
            logging.warning("Health monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Health monitoring started")

    def stop_health_monitoring(self):
        """Stop health monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logging.info("Health monitoring stopped")

    def _monitor_loop(self):
        """Health monitoring loop"""
        while self.monitoring_active:
            for camera_id, camera in self.cameras.items():
                health = self.health[camera_id]

                if camera.is_streaming:
                    # Update FPS
                    current_time = time.time()
                    if health.last_update > 0:
                        dt = current_time - health.last_update
                        if dt > 0:
                            health.current_fps = 1.0 / dt

                    health.last_update = current_time

                    # Get temperature
                    health.temperature = camera.get_temperature()

                    # Check for overheating
                    if health.temperature > health.max_temperature:
                        logging.warning(f"Camera {camera_id} overheating: {health.temperature:.1f}C")
                        health.error_count += 1

                    # Check FPS deviation
                    fps_error = abs(health.current_fps - health.target_fps) / health.target_fps
                    if fps_error > 0.1:  # >10% deviation
                        logging.debug(f"Camera {camera_id} FPS deviation: "
                                    f"{health.current_fps:.1f}/{health.target_fps:.1f}")

                # Update connection status
                health.connection_stable = camera.is_connected

            time.sleep(self.monitor_interval)

    def get_camera_health(self, camera_id: int) -> Optional[CameraHealth]:
        """Get health status for specific camera"""
        return self.health.get(camera_id)

    def get_pair_health(self, pair_id: int) -> Optional[Dict]:
        """Get health status for camera pair"""
        if pair_id not in self.pairs:
            return None

        pair = self.pairs[pair_id]
        mono_health = self.health.get(pair.mono_camera_id)
        thermal_health = self.health.get(pair.thermal_camera_id)

        return {
            'pair_id': pair_id,
            'mono_health': mono_health,
            'thermal_health': thermal_health,
            'sync_quality': pair.sync_quality,
            'needs_recalibration': pair.needs_recalibration,
            'both_streaming': (mono_health.status == CameraStatus.STREAMING and
                             thermal_health.status == CameraStatus.STREAMING)
        }

    def get_system_health(self) -> Dict:
        """Get overall system health metrics"""
        total_cameras = len(self.cameras)
        streaming_count = sum(1 for h in self.health.values() if h.status == CameraStatus.STREAMING)
        ready_count = sum(1 for h in self.health.values() if h.status == CameraStatus.READY)
        error_count = sum(1 for h in self.health.values() if h.status == CameraStatus.ERROR)

        total_frames = sum(h.frames_received for h in self.health.values())
        total_drops = sum(h.frames_dropped for h in self.health.values())

        avg_fps = np.mean([h.current_fps for h in self.health.values() if h.current_fps > 0])
        avg_temp = np.mean([h.temperature for h in self.health.values() if h.temperature > 0])

        return {
            'total_cameras': total_cameras,
            'streaming': streaming_count,
            'ready': ready_count,
            'error': error_count,
            'offline': total_cameras - streaming_count - ready_count - error_count,
            'total_frames': total_frames,
            'total_dropped': total_drops,
            'drop_rate': total_drops / max(total_frames, 1),
            'avg_fps': avg_fps,
            'avg_temperature': avg_temp,
            'pairs_operational': sum(1 for p in self.pairs.values() if not p.needs_recalibration)
        }

    def update_camera_position(self, camera_id: int, position: np.ndarray, orientation: np.ndarray):
        """Update camera position and orientation"""
        if camera_id not in self.configurations:
            logging.error(f"Camera {camera_id} not found")
            return False

        self.configurations[camera_id].position = position
        self.configurations[camera_id].orientation = orientation

        logging.info(f"Updated camera {camera_id} position: {position}")
        return True

    def load_calibration(self, camera_id: int, intrinsic_file: str, extrinsic_file: str = "") -> bool:
        """Load calibration parameters for camera"""
        if camera_id not in self.configurations:
            return False

        config = self.configurations[camera_id]
        config.intrinsic_file = intrinsic_file
        config.extrinsic_file = extrinsic_file

        logging.info(f"Loaded calibration for camera {camera_id}")
        return True

    def save_configuration(self, filename: Optional[str] = None):
        """Save camera configurations to JSON file"""
        if filename:
            filepath = Path(filename)
        else:
            filepath = self.config_file

        config_data = {
            'num_pairs': self.num_pairs,
            'cameras': {},
            'pairs': {}
        }

        # Save camera configurations
        for camera_id, config in self.configurations.items():
            config_data['cameras'][camera_id] = {
                'camera_id': config.camera_id,
                'pair_id': config.pair_id,
                'camera_type': config.camera_type.name,
                'connection': config.connection.name,
                'ip_address': config.ip_address,
                'mac_address': config.mac_address,
                'width': config.width,
                'height': config.height,
                'frame_rate': config.frame_rate,
                'exposure_time': config.exposure_time,
                'gain': config.gain,
                'position': config.position.tolist(),
                'orientation': config.orientation.tolist(),
                'serial_number': config.serial_number,
                'model_name': config.model_name
            }

        # Save pair configurations
        for pair_id, pair in self.pairs.items():
            config_data['pairs'][pair_id] = {
                'pair_id': pair.pair_id,
                'mono_camera_id': pair.mono_camera_id,
                'thermal_camera_id': pair.thermal_camera_id,
                'stereo_baseline': pair.stereo_baseline
            }

        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)

        logging.info(f"Configuration saved to {filepath}")

    def load_configuration(self, filename: Optional[str] = None) -> bool:
        """Load camera configurations from JSON file"""
        if filename:
            filepath = Path(filename)
        else:
            filepath = self.config_file

        if not filepath.exists():
            logging.error(f"Configuration file not found: {filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)

            # Load camera configurations
            for cam_id_str, cam_data in config_data['cameras'].items():
                config = CameraConfiguration(
                    camera_id=cam_data['camera_id'],
                    pair_id=cam_data['pair_id'],
                    camera_type=CameraType[cam_data['camera_type']],
                    connection=ConnectionType[cam_data['connection']],
                    ip_address=cam_data['ip_address'],
                    mac_address=cam_data['mac_address'],
                    width=cam_data['width'],
                    height=cam_data['height'],
                    frame_rate=cam_data['frame_rate'],
                    exposure_time=cam_data['exposure_time'],
                    gain=cam_data['gain'],
                    position=np.array(cam_data['position']),
                    orientation=np.array(cam_data['orientation']),
                    serial_number=cam_data['serial_number'],
                    model_name=cam_data['model_name']
                )
                self.add_camera(config)

            # Load pair configurations
            for pair_id_str, pair_data in config_data['pairs'].items():
                pair = CameraPair(
                    pair_id=pair_data['pair_id'],
                    mono_camera_id=pair_data['mono_camera_id'],
                    thermal_camera_id=pair_data['thermal_camera_id'],
                    stereo_baseline=pair_data['stereo_baseline']
                )
                self.add_pair(pair)

            logging.info(f"Configuration loaded from {filepath}")
            return True

        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            return False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
