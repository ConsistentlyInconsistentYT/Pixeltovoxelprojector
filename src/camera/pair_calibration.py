"""
Camera Pair Calibration System
Handles mono-thermal registration, stereo calibration, and real-time calibration updates
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pathlib import Path
import numpy as np
import json


class CalibrationStatus(Enum):
    """Calibration status states"""
    NOT_CALIBRATED = 0
    CALIBRATING = 1
    CALIBRATED = 2
    NEEDS_UPDATE = 3
    FAILED = 4


class CalibrationType(Enum):
    """Types of calibration"""
    INTRINSIC = 1
    EXTRINSIC = 2
    STEREO = 3
    MONO_THERMAL = 4


@dataclass
class IntrinsicParameters:
    """Camera intrinsic parameters"""
    camera_id: int

    # Camera matrix (3x3)
    fx: float = 0.0  # Focal length x
    fy: float = 0.0  # Focal length y
    cx: float = 0.0  # Principal point x
    cy: float = 0.0  # Principal point y

    # Distortion coefficients (k1, k2, p1, p2, k3)
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # Image dimensions
    width: int = 1920
    height: int = 1080

    # Calibration quality
    reprojection_error: float = 0.0
    calibration_date: float = field(default_factory=time.time)
    num_images: int = 0

    def get_camera_matrix(self) -> np.ndarray:
        """Get 3x3 camera matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def get_distortion_coeffs(self) -> np.ndarray:
        """Get distortion coefficients"""
        return self.distortion


@dataclass
class ExtrinsicParameters:
    """Camera extrinsic parameters (position and orientation in world frame)"""
    camera_id: int

    # Rotation matrix (3x3) - camera to world
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Translation vector (3,) - camera position in world
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Rodrigues rotation vector
    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Calibration quality
    calibration_error: float = 0.0
    calibration_date: float = field(default_factory=time.time)

    def get_transformation_matrix(self) -> np.ndarray:
        """Get 4x4 homogeneous transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T


@dataclass
class StereoCalibration:
    """Stereo calibration between camera pair"""
    pair_id: int
    camera1_id: int
    camera2_id: int

    # Relative pose (camera2 relative to camera1)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Essential and Fundamental matrices
    essential_matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    fundamental_matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    # Rectification parameters
    rectification1: np.ndarray = field(default_factory=lambda: np.eye(3))
    rectification2: np.ndarray = field(default_factory=lambda: np.eye(3))
    projection1: np.ndarray = field(default_factory=lambda: np.zeros((3, 4)))
    projection2: np.ndarray = field(default_factory=lambda: np.zeros((3, 4)))

    # Stereo geometry
    baseline: float = 0.0  # Distance between cameras in meters

    # Quality metrics
    reprojection_error: float = 0.0
    epipolar_error: float = 0.0
    calibration_date: float = field(default_factory=time.time)
    num_point_pairs: int = 0


@dataclass
class MonoThermalRegistration:
    """Registration parameters between mono and thermal cameras"""
    pair_id: int
    mono_camera_id: int
    thermal_camera_id: int

    # Transformation from thermal to mono coordinate frame
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 2D image registration (homography)
    homography: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Pixel mapping (thermal -> mono)
    pixel_map_x: Optional[np.ndarray] = None  # X coordinates
    pixel_map_y: Optional[np.ndarray] = None  # Y coordinates

    # Temperature-to-intensity calibration
    temp_min: float = -20.0  # Celsius
    temp_max: float = 120.0
    intensity_scale: float = 1.0
    intensity_offset: float = 0.0

    # Quality metrics
    registration_error: float = 0.0
    mutual_information: float = 0.0  # Image similarity metric
    feature_matches: int = 0
    calibration_date: float = field(default_factory=time.time)


@dataclass
class CalibrationTarget:
    """Calibration target specification"""
    target_type: str = "checkerboard"  # checkerboard, circles, aruco

    # Checkerboard parameters
    rows: int = 9
    cols: int = 6
    square_size: float = 0.025  # meters

    # Circle grid parameters
    circle_diameter: float = 0.02  # meters

    # 3D world coordinates of target points
    object_points: Optional[np.ndarray] = None

    def get_object_points(self) -> np.ndarray:
        """Get 3D coordinates of calibration pattern"""
        if self.object_points is not None:
            return self.object_points

        if self.target_type == "checkerboard":
            # Create checkerboard pattern points
            objp = np.zeros((self.rows * self.cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
            objp *= self.square_size
            return objp

        return np.array([])


class CalibrationEngine:
    """Core calibration computation engine"""

    @staticmethod
    def calibrate_intrinsic(image_points: List[np.ndarray],
                           object_points: List[np.ndarray],
                           image_size: Tuple[int, int]) -> IntrinsicParameters:
        """
        Calibrate camera intrinsic parameters

        Args:
            image_points: List of 2D image point arrays for each calibration image
            object_points: List of 3D object point arrays
            image_size: (width, height) of images

        Returns:
            Calibrated intrinsic parameters
        """
        # In real implementation, use cv2.calibrateCamera()
        # For now, simulate calibration

        intrinsic = IntrinsicParameters(camera_id=0, width=image_size[0], height=image_size[1])

        # Typical values for a 1920x1080 camera with ~50° FOV
        intrinsic.fx = image_size[0] * 1.2
        intrinsic.fy = image_size[0] * 1.2
        intrinsic.cx = image_size[0] / 2.0
        intrinsic.cy = image_size[1] / 2.0

        # Simulate small distortion
        intrinsic.distortion = np.array([0.05, -0.02, 0.0, 0.0, 0.01])

        # Simulate calibration quality
        intrinsic.reprojection_error = 0.3 + np.random.rand() * 0.2  # pixels
        intrinsic.num_images = len(image_points)

        logging.info(f"Intrinsic calibration complete: error={intrinsic.reprojection_error:.3f}px")

        return intrinsic

    @staticmethod
    def calibrate_stereo(intrinsic1: IntrinsicParameters,
                        intrinsic2: IntrinsicParameters,
                        image_points1: List[np.ndarray],
                        image_points2: List[np.ndarray],
                        object_points: List[np.ndarray]) -> StereoCalibration:
        """
        Calibrate stereo camera pair

        Args:
            intrinsic1: Intrinsics of first camera
            intrinsic2: Intrinsics of second camera
            image_points1: Image points from first camera
            image_points2: Image points from second camera
            object_points: 3D object points

        Returns:
            Stereo calibration parameters
        """
        # In real implementation, use cv2.stereoCalibrate()

        stereo = StereoCalibration(
            pair_id=0,
            camera1_id=intrinsic1.camera_id,
            camera2_id=intrinsic2.camera_id
        )

        # Simulate stereo parameters (horizontal stereo rig)
        stereo.baseline = 0.12  # 12cm baseline
        stereo.translation = np.array([stereo.baseline, 0, 0])

        # Small rotation (cameras not perfectly aligned)
        angle = np.deg2rad(0.5)  # 0.5 degree misalignment
        stereo.rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # Quality metrics
        stereo.reprojection_error = 0.4 + np.random.rand() * 0.3
        stereo.epipolar_error = 0.2 + np.random.rand() * 0.1
        stereo.num_point_pairs = len(image_points1) * image_points1[0].shape[0]

        logging.info(f"Stereo calibration complete: error={stereo.reprojection_error:.3f}px")

        return stereo

    @staticmethod
    def register_mono_thermal(mono_image: np.ndarray,
                             thermal_image: np.ndarray,
                             mono_intrinsic: IntrinsicParameters,
                             thermal_intrinsic: IntrinsicParameters) -> MonoThermalRegistration:
        """
        Register thermal camera to mono camera

        Args:
            mono_image: Mono camera image
            thermal_image: Thermal camera image
            mono_intrinsic: Mono camera intrinsics
            thermal_intrinsic: Thermal camera intrinsics

        Returns:
            Registration parameters
        """
        # In real implementation:
        # 1. Extract features from both images (SIFT, ORB, etc.)
        # 2. Match features
        # 3. Compute homography or fundamental matrix
        # 4. Refine with bundle adjustment
        # 5. Generate pixel mapping

        registration = MonoThermalRegistration(
            pair_id=0,
            mono_camera_id=mono_intrinsic.camera_id,
            thermal_camera_id=thermal_intrinsic.camera_id
        )

        # Simulate small offset between cameras
        registration.translation = np.array([0.05, 0.02, 0.0])  # 5cm horizontal, 2cm vertical

        # Small rotation
        angle = np.deg2rad(2.0)
        registration.rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # Simulate homography (slight perspective difference)
        registration.homography = np.array([
            [0.98, -0.02, 50],
            [0.01, 0.99, 30],
            [0.0001, -0.00005, 1]
        ])

        # Quality metrics
        registration.registration_error = 1.5 + np.random.rand() * 0.5  # pixels
        registration.mutual_information = 0.85 + np.random.rand() * 0.1
        registration.feature_matches = 150 + int(np.random.rand() * 100)

        logging.info(f"Mono-thermal registration complete: "
                    f"error={registration.registration_error:.3f}px, "
                    f"matches={registration.feature_matches}")

        return registration


class PairCalibrationManager:
    """Manages calibration for all camera pairs"""

    def __init__(self, num_pairs: int = 10):
        self.num_pairs = num_pairs
        self.num_cameras = num_pairs * 2

        # Calibration storage
        self.intrinsics: Dict[int, IntrinsicParameters] = {}
        self.extrinsics: Dict[int, ExtrinsicParameters] = {}
        self.stereo_calibrations: Dict[int, StereoCalibration] = {}
        self.registrations: Dict[int, MonoThermalRegistration] = {}

        # Calibration status
        self.calibration_status: Dict[int, CalibrationStatus] = {
            i: CalibrationStatus.NOT_CALIBRATED for i in range(num_pairs)
        }

        # Calibration target
        self.target = CalibrationTarget()

        # Real-time update
        self.update_enabled = False
        self.update_thread = None
        self.update_interval = 60.0  # seconds

        # Thread safety
        self.lock = threading.Lock()

        logging.info(f"PairCalibrationManager initialized for {num_pairs} pairs")

    def calibrate_camera_intrinsic(self, camera_id: int,
                                   calibration_images: List[np.ndarray],
                                   detected_points: List[np.ndarray]) -> bool:
        """
        Calibrate intrinsic parameters for a single camera

        Args:
            camera_id: Camera to calibrate
            calibration_images: List of calibration images
            detected_points: Detected calibration pattern points in each image

        Returns:
            Success status
        """
        try:
            # Get object points for each image
            object_points = [self.target.get_object_points()] * len(detected_points)

            # Get image size
            image_size = (calibration_images[0].shape[1], calibration_images[0].shape[0])

            # Calibrate
            intrinsic = CalibrationEngine.calibrate_intrinsic(
                detected_points, object_points, image_size
            )
            intrinsic.camera_id = camera_id

            with self.lock:
                self.intrinsics[camera_id] = intrinsic

            logging.info(f"Camera {camera_id} intrinsic calibration successful")
            return True

        except Exception as e:
            logging.error(f"Camera {camera_id} intrinsic calibration failed: {e}")
            return False

    def calibrate_stereo_pair(self, pair_id: int,
                             images1: List[np.ndarray],
                             images2: List[np.ndarray],
                             points1: List[np.ndarray],
                             points2: List[np.ndarray]) -> bool:
        """
        Calibrate stereo geometry for a camera pair

        Args:
            pair_id: Pair to calibrate
            images1: Images from first camera
            images2: Images from second camera (synchronized)
            points1: Detected points in first camera
            points2: Detected points in second camera

        Returns:
            Success status
        """
        camera1_id = pair_id * 2
        camera2_id = pair_id * 2 + 1

        # Check if intrinsics are available
        if camera1_id not in self.intrinsics or camera2_id not in self.intrinsics:
            logging.error(f"Pair {pair_id}: Intrinsic calibration required first")
            return False

        try:
            object_points = [self.target.get_object_points()] * len(points1)

            stereo = CalibrationEngine.calibrate_stereo(
                self.intrinsics[camera1_id],
                self.intrinsics[camera2_id],
                points1, points2, object_points
            )
            stereo.pair_id = pair_id

            with self.lock:
                self.stereo_calibrations[pair_id] = stereo
                self.calibration_status[pair_id] = CalibrationStatus.CALIBRATED

            logging.info(f"Pair {pair_id} stereo calibration successful")
            return True

        except Exception as e:
            logging.error(f"Pair {pair_id} stereo calibration failed: {e}")
            self.calibration_status[pair_id] = CalibrationStatus.FAILED
            return False

    def register_mono_thermal_pair(self, pair_id: int,
                                  mono_image: np.ndarray,
                                  thermal_image: np.ndarray) -> bool:
        """
        Register mono and thermal cameras in a pair

        Args:
            pair_id: Pair to register
            mono_image: Image from mono camera
            thermal_image: Image from thermal camera (synchronized)

        Returns:
            Success status
        """
        mono_id = pair_id * 2
        thermal_id = pair_id * 2 + 1

        # Check intrinsics
        if mono_id not in self.intrinsics or thermal_id not in self.intrinsics:
            logging.error(f"Pair {pair_id}: Intrinsic calibration required for registration")
            return False

        try:
            registration = CalibrationEngine.register_mono_thermal(
                mono_image, thermal_image,
                self.intrinsics[mono_id],
                self.intrinsics[thermal_id]
            )
            registration.pair_id = pair_id

            with self.lock:
                self.registrations[pair_id] = registration

            logging.info(f"Pair {pair_id} mono-thermal registration successful")
            return True

        except Exception as e:
            logging.error(f"Pair {pair_id} mono-thermal registration failed: {e}")
            return False

    def get_intrinsics(self, camera_id: int) -> Optional[IntrinsicParameters]:
        """Get intrinsic parameters for camera"""
        with self.lock:
            return self.intrinsics.get(camera_id)

    def get_stereo_calibration(self, pair_id: int) -> Optional[StereoCalibration]:
        """Get stereo calibration for pair"""
        with self.lock:
            return self.stereo_calibrations.get(pair_id)

    def get_registration(self, pair_id: int) -> Optional[MonoThermalRegistration]:
        """Get mono-thermal registration for pair"""
        with self.lock:
            return self.registrations.get(pair_id)

    def check_calibration_validity(self, pair_id: int,
                                  max_age_days: float = 30.0) -> Tuple[bool, str]:
        """
        Check if calibration is still valid

        Args:
            pair_id: Pair to check
            max_age_days: Maximum calibration age in days

        Returns:
            (is_valid, reason) tuple
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 86400

        # Check stereo calibration
        if pair_id not in self.stereo_calibrations:
            return False, "No stereo calibration"

        stereo = self.stereo_calibrations[pair_id]
        age = current_time - stereo.calibration_date

        if age > max_age_seconds:
            return False, f"Calibration too old ({age/86400:.1f} days)"

        if stereo.reprojection_error > 1.0:  # pixels
            return False, f"High reprojection error ({stereo.reprojection_error:.2f}px)"

        # Check registration
        if pair_id not in self.registrations:
            return False, "No mono-thermal registration"

        registration = self.registrations[pair_id]

        if registration.registration_error > 3.0:  # pixels
            return False, f"High registration error ({registration.registration_error:.2f}px)"

        if registration.mutual_information < 0.5:
            return False, "Low image similarity"

        return True, "Valid"

    def start_realtime_updates(self, update_interval: float = 60.0):
        """
        Start real-time calibration quality monitoring and updates

        Args:
            update_interval: Update interval in seconds
        """
        if self.update_enabled:
            logging.warning("Real-time updates already enabled")
            return

        self.update_interval = update_interval
        self.update_enabled = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        logging.info(f"Real-time calibration updates started (interval={update_interval}s)")

    def stop_realtime_updates(self):
        """Stop real-time calibration updates"""
        self.update_enabled = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        logging.info("Real-time calibration updates stopped")

    def _update_loop(self):
        """Real-time calibration monitoring loop"""
        while self.update_enabled:
            # Check all pair calibrations
            for pair_id in range(self.num_pairs):
                is_valid, reason = self.check_calibration_validity(pair_id)

                if not is_valid:
                    logging.warning(f"Pair {pair_id} calibration invalid: {reason}")
                    self.calibration_status[pair_id] = CalibrationStatus.NEEDS_UPDATE
                elif self.calibration_status[pair_id] == CalibrationStatus.CALIBRATED:
                    # Still valid
                    pass

            time.sleep(self.update_interval)

    def save_calibration(self, pair_id: int, directory: str = "calibration"):
        """Save calibration data for a pair to disk"""
        calib_dir = Path(directory)
        calib_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'pair_id': pair_id,
            'calibration_date': time.time(),
            'cameras': {},
            'stereo': None,
            'registration': None
        }

        # Save intrinsics
        mono_id = pair_id * 2
        thermal_id = pair_id * 2 + 1

        for camera_id in [mono_id, thermal_id]:
            if camera_id in self.intrinsics:
                intrinsic = self.intrinsics[camera_id]
                data['cameras'][camera_id] = {
                    'camera_matrix': intrinsic.get_camera_matrix().tolist(),
                    'distortion': intrinsic.distortion.tolist(),
                    'image_size': [intrinsic.width, intrinsic.height],
                    'reprojection_error': intrinsic.reprojection_error,
                    'num_images': intrinsic.num_images
                }

        # Save stereo calibration
        if pair_id in self.stereo_calibrations:
            stereo = self.stereo_calibrations[pair_id]
            data['stereo'] = {
                'rotation': stereo.rotation.tolist(),
                'translation': stereo.translation.tolist(),
                'baseline': stereo.baseline,
                'reprojection_error': stereo.reprojection_error,
                'epipolar_error': stereo.epipolar_error
            }

        # Save registration
        if pair_id in self.registrations:
            reg = self.registrations[pair_id]
            data['registration'] = {
                'rotation': reg.rotation.tolist(),
                'translation': reg.translation.tolist(),
                'homography': reg.homography.tolist(),
                'registration_error': reg.registration_error,
                'mutual_information': reg.mutual_information
            }

        # Write to file
        filepath = calib_dir / f"pair_{pair_id}_calibration.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logging.info(f"Calibration for pair {pair_id} saved to {filepath}")

    def load_calibration(self, pair_id: int, directory: str = "calibration") -> bool:
        """Load calibration data for a pair from disk"""
        calib_dir = Path(directory)
        filepath = calib_dir / f"pair_{pair_id}_calibration.json"

        if not filepath.exists():
            logging.error(f"Calibration file not found: {filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Load intrinsics
            for camera_id_str, cam_data in data['cameras'].items():
                camera_id = int(camera_id_str)
                intrinsic = IntrinsicParameters(camera_id=camera_id)

                cam_matrix = np.array(cam_data['camera_matrix'])
                intrinsic.fx = cam_matrix[0, 0]
                intrinsic.fy = cam_matrix[1, 1]
                intrinsic.cx = cam_matrix[0, 2]
                intrinsic.cy = cam_matrix[1, 2]
                intrinsic.distortion = np.array(cam_data['distortion'])
                intrinsic.width, intrinsic.height = cam_data['image_size']
                intrinsic.reprojection_error = cam_data['reprojection_error']
                intrinsic.num_images = cam_data['num_images']

                self.intrinsics[camera_id] = intrinsic

            # Load stereo calibration
            if data['stereo']:
                stereo_data = data['stereo']
                stereo = StereoCalibration(
                    pair_id=pair_id,
                    camera1_id=pair_id * 2,
                    camera2_id=pair_id * 2 + 1
                )
                stereo.rotation = np.array(stereo_data['rotation'])
                stereo.translation = np.array(stereo_data['translation'])
                stereo.baseline = stereo_data['baseline']
                stereo.reprojection_error = stereo_data['reprojection_error']
                stereo.epipolar_error = stereo_data['epipolar_error']

                self.stereo_calibrations[pair_id] = stereo

            # Load registration
            if data['registration']:
                reg_data = data['registration']
                reg = MonoThermalRegistration(
                    pair_id=pair_id,
                    mono_camera_id=pair_id * 2,
                    thermal_camera_id=pair_id * 2 + 1
                )
                reg.rotation = np.array(reg_data['rotation'])
                reg.translation = np.array(reg_data['translation'])
                reg.homography = np.array(reg_data['homography'])
                reg.registration_error = reg_data['registration_error']
                reg.mutual_information = reg_data['mutual_information']

                self.registrations[pair_id] = reg

            self.calibration_status[pair_id] = CalibrationStatus.CALIBRATED
            logging.info(f"Calibration for pair {pair_id} loaded from {filepath}")
            return True

        except Exception as e:
            logging.error(f"Failed to load calibration: {e}")
            return False

    def get_calibration_report(self) -> Dict:
        """Generate comprehensive calibration report"""
        report = {
            'num_pairs': self.num_pairs,
            'timestamp': time.time(),
            'pairs': {}
        }

        for pair_id in range(self.num_pairs):
            pair_report = {
                'pair_id': pair_id,
                'status': self.calibration_status[pair_id].name,
                'intrinsics_available': {
                    'mono': pair_id * 2 in self.intrinsics,
                    'thermal': pair_id * 2 + 1 in self.intrinsics
                },
                'stereo_available': pair_id in self.stereo_calibrations,
                'registration_available': pair_id in self.registrations
            }

            # Add quality metrics if available
            if pair_id in self.stereo_calibrations:
                stereo = self.stereo_calibrations[pair_id]
                age_days = (time.time() - stereo.calibration_date) / 86400

                pair_report['stereo_quality'] = {
                    'reprojection_error': stereo.reprojection_error,
                    'epipolar_error': stereo.epipolar_error,
                    'baseline': stereo.baseline,
                    'age_days': age_days
                }

            if pair_id in self.registrations:
                reg = self.registrations[pair_id]
                age_days = (time.time() - reg.calibration_date) / 86400

                pair_report['registration_quality'] = {
                    'registration_error': reg.registration_error,
                    'mutual_information': reg.mutual_information,
                    'feature_matches': reg.feature_matches,
                    'age_days': age_days
                }

            # Validity check
            is_valid, reason = self.check_calibration_validity(pair_id)
            pair_report['is_valid'] = is_valid
            pair_report['validation_message'] = reason

            report['pairs'][pair_id] = pair_report

        # Overall statistics
        calibrated_pairs = sum(1 for s in self.calibration_status.values()
                              if s == CalibrationStatus.CALIBRATED)
        needs_update = sum(1 for s in self.calibration_status.values()
                          if s == CalibrationStatus.NEEDS_UPDATE)

        report['summary'] = {
            'calibrated_pairs': calibrated_pairs,
            'needs_update': needs_update,
            'not_calibrated': self.num_pairs - calibrated_pairs - needs_update,
            'avg_stereo_error': np.mean([s.reprojection_error
                                        for s in self.stereo_calibrations.values()])
                                if self.stereo_calibrations else 0.0,
            'avg_registration_error': np.mean([r.registration_error
                                              for r in self.registrations.values()])
                                      if self.registrations else 0.0
        }

        return report


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
