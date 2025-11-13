"""
Data Validator - Comprehensive Data Validation and Sanity Checking

Features:
- Coordinate sanity checking (bounds, range validation)
- Detection confidence validation
- Cross-camera consistency checks
- Temporal coherence validation
- Outlier detection with statistical methods
- Multi-level validation (warning, error, critical)

Requirements:
- Real-time validation at 30Hz
- <1ms validation overhead per frame
- Comprehensive error reporting
- Automatic correction suggestions
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from collections import deque
import time

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validation checks"""
    COORDINATE_BOUNDS = "coordinate_bounds"
    COORDINATE_RANGE = "coordinate_range"
    CONFIDENCE = "confidence"
    TEMPORAL_COHERENCE = "temporal_coherence"
    CROSS_CAMERA = "cross_camera"
    OUTLIER = "outlier"
    DETECTION_RATE = "detection_rate"
    TRACKING_CONSISTENCY = "tracking_consistency"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    level: ValidationLevel
    type: ValidationType
    message: str
    timestamp: float = field(default_factory=time.time)
    camera_id: Optional[int] = None
    track_id: Optional[int] = None
    detection_id: Optional[int] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation check"""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    validation_time_ms: float = 0.0

    def add_issue(self, issue: ValidationIssue):
        """Add validation issue"""
        self.issues.append(issue)
        if issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
            self.passed = False

    def get_issues_by_level(self, level: ValidationLevel) -> List[ValidationIssue]:
        """Get issues of specific level"""
        return [issue for issue in self.issues if issue.level == level]

    def has_critical_issues(self) -> bool:
        """Check if there are critical issues"""
        return any(issue.level == ValidationLevel.CRITICAL for issue in self.issues)


@dataclass
class CoordinateBounds:
    """3D coordinate bounds"""
    x_min: float = -5000.0
    x_max: float = 5000.0
    y_min: float = -5000.0
    y_max: float = 5000.0
    z_min: float = 0.0
    z_max: float = 2000.0


class CoordinateValidator:
    """
    Validates 3D coordinates for physical plausibility
    """

    def __init__(self, bounds: CoordinateBounds = None):
        """
        Initialize coordinate validator

        Args:
            bounds: Coordinate bounds to validate against
        """
        self.bounds = bounds or CoordinateBounds()

    def validate_coordinate(self, coord: np.ndarray, coord_name: str = "coordinate") -> ValidationResult:
        """
        Validate a 3D coordinate

        Args:
            coord: 3D coordinate [x, y, z]
            coord_name: Name for error messages

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        # Check dimensions
        if len(coord) != 3:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.ERROR,
                type=ValidationType.COORDINATE_BOUNDS,
                message=f"{coord_name} has invalid dimensions: {len(coord)} (expected 3)",
                value=len(coord),
                expected=3
            ))
            return result

        x, y, z = coord

        # Check for NaN/Inf
        if np.any(~np.isfinite(coord)):
            result.add_issue(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                type=ValidationType.COORDINATE_BOUNDS,
                message=f"{coord_name} contains NaN or Inf values",
                value=coord.tolist(),
                suggestion="Check sensor data and coordinate transformation"
            ))
            return result

        # Check X bounds
        if not (self.bounds.x_min <= x <= self.bounds.x_max):
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING if abs(x) < 10000 else ValidationLevel.ERROR,
                type=ValidationType.COORDINATE_BOUNDS,
                message=f"{coord_name} X out of bounds: {x:.2f}",
                value=x,
                expected=f"[{self.bounds.x_min}, {self.bounds.x_max}]"
            ))

        # Check Y bounds
        if not (self.bounds.y_min <= y <= self.bounds.y_max):
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING if abs(y) < 10000 else ValidationLevel.ERROR,
                type=ValidationType.COORDINATE_BOUNDS,
                message=f"{coord_name} Y out of bounds: {y:.2f}",
                value=y,
                expected=f"[{self.bounds.y_min}, {self.bounds.y_max}]"
            ))

        # Check Z bounds
        if not (self.bounds.z_min <= z <= self.bounds.z_max):
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING if z < 5000 else ValidationLevel.ERROR,
                type=ValidationType.COORDINATE_BOUNDS,
                message=f"{coord_name} Z out of bounds: {z:.2f}",
                value=z,
                expected=f"[{self.bounds.z_min}, {self.bounds.z_max}]",
                suggestion="Check altitude sensor or ground plane calibration"
            ))

        return result

    def validate_coordinate_range(self, coord1: np.ndarray, coord2: np.ndarray,
                                 max_distance: float = 5000.0) -> ValidationResult:
        """
        Validate distance between two coordinates

        Args:
            coord1: First coordinate
            coord2: Second coordinate
            max_distance: Maximum allowed distance

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        distance = np.linalg.norm(coord2 - coord1)

        if distance > max_distance:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING,
                type=ValidationType.COORDINATE_RANGE,
                message=f"Distance between coordinates exceeds maximum: {distance:.2f}m",
                value=distance,
                expected=f"<= {max_distance}m"
            ))

        return result


class ConfidenceValidator:
    """
    Validates detection confidence scores
    """

    def __init__(self, min_confidence: float = 0.5, warn_threshold: float = 0.7):
        """
        Initialize confidence validator

        Args:
            min_confidence: Minimum acceptable confidence
            warn_threshold: Threshold for warnings
        """
        self.min_confidence = min_confidence
        self.warn_threshold = warn_threshold

    def validate_confidence(self, confidence: float,
                          detection_id: Optional[int] = None) -> ValidationResult:
        """
        Validate detection confidence score

        Args:
            confidence: Confidence score [0, 1]
            detection_id: Optional detection ID

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        # Check range
        if not (0.0 <= confidence <= 1.0):
            result.add_issue(ValidationIssue(
                level=ValidationLevel.ERROR,
                type=ValidationType.CONFIDENCE,
                message=f"Confidence out of valid range: {confidence:.3f}",
                detection_id=detection_id,
                value=confidence,
                expected="[0.0, 1.0]",
                suggestion="Normalize confidence scores"
            ))
            return result

        # Check minimum threshold
        if confidence < self.min_confidence:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING,
                type=ValidationType.CONFIDENCE,
                message=f"Confidence below minimum: {confidence:.3f}",
                detection_id=detection_id,
                value=confidence,
                expected=f">= {self.min_confidence}",
                suggestion="Consider rejecting or flagging low-confidence detections"
            ))

        # Check warning threshold
        elif confidence < self.warn_threshold:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.INFO,
                type=ValidationType.CONFIDENCE,
                message=f"Low confidence detection: {confidence:.3f}",
                detection_id=detection_id,
                value=confidence
            ))

        return result


class TemporalValidator:
    """
    Validates temporal coherence of tracks
    """

    def __init__(self, max_velocity: float = 100.0,  # m/s
                 max_acceleration: float = 50.0,  # m/s^2
                 max_position_jump: float = 10.0):  # m
        """
        Initialize temporal validator

        Args:
            max_velocity: Maximum plausible velocity (m/s)
            max_acceleration: Maximum plausible acceleration (m/s^2)
            max_position_jump: Maximum position jump between frames (m)
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_position_jump = max_position_jump

    def validate_velocity(self, velocity: np.ndarray,
                         track_id: Optional[int] = None) -> ValidationResult:
        """
        Validate velocity vector

        Args:
            velocity: Velocity vector [vx, vy, vz]
            track_id: Optional track ID

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        speed = np.linalg.norm(velocity)

        if not np.isfinite(speed):
            result.add_issue(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                type=ValidationType.TEMPORAL_COHERENCE,
                message="Invalid velocity (NaN or Inf)",
                track_id=track_id,
                value=velocity.tolist(),
                suggestion="Check Kalman filter state update"
            ))
            return result

        if speed > self.max_velocity:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.ERROR,
                type=ValidationType.TEMPORAL_COHERENCE,
                message=f"Velocity exceeds maximum: {speed:.2f} m/s",
                track_id=track_id,
                value=speed,
                expected=f"<= {self.max_velocity} m/s",
                suggestion="Check for track ID swap or measurement error"
            ))

        return result

    def validate_position_jump(self, pos1: np.ndarray, pos2: np.ndarray,
                              dt: float, track_id: Optional[int] = None) -> ValidationResult:
        """
        Validate position change between consecutive frames

        Args:
            pos1: Previous position
            pos2: Current position
            dt: Time delta (seconds)
            track_id: Optional track ID

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        displacement = np.linalg.norm(pos2 - pos1)

        if displacement > self.max_position_jump:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING,
                type=ValidationType.TEMPORAL_COHERENCE,
                message=f"Large position jump: {displacement:.2f}m in {dt:.3f}s",
                track_id=track_id,
                value=displacement,
                expected=f"<= {self.max_position_jump}m",
                suggestion="Possible occlusion or track ID swap"
            ))

        # Check implied velocity
        if dt > 0:
            implied_velocity = displacement / dt
            if implied_velocity > self.max_velocity:
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    type=ValidationType.TEMPORAL_COHERENCE,
                    message=f"Implied velocity too high: {implied_velocity:.2f} m/s",
                    track_id=track_id,
                    value=implied_velocity,
                    expected=f"<= {self.max_velocity} m/s"
                ))

        return result

    def validate_acceleration(self, accel: np.ndarray,
                            track_id: Optional[int] = None) -> ValidationResult:
        """
        Validate acceleration vector

        Args:
            accel: Acceleration vector [ax, ay, az]
            track_id: Optional track ID

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        magnitude = np.linalg.norm(accel)

        if not np.isfinite(magnitude):
            result.add_issue(ValidationIssue(
                level=ValidationLevel.ERROR,
                type=ValidationType.TEMPORAL_COHERENCE,
                message="Invalid acceleration (NaN or Inf)",
                track_id=track_id,
                value=accel.tolist()
            ))
            return result

        if magnitude > self.max_acceleration:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING,
                type=ValidationType.TEMPORAL_COHERENCE,
                message=f"High acceleration: {magnitude:.2f} m/s^2",
                track_id=track_id,
                value=magnitude,
                expected=f"<= {self.max_acceleration} m/s^2"
            ))

        return result


class CrossCameraValidator:
    """
    Validates consistency across multiple cameras
    """

    def __init__(self, max_position_error: float = 2.0,  # meters
                 min_overlap_ratio: float = 0.5):
        """
        Initialize cross-camera validator

        Args:
            max_position_error: Maximum allowed position difference between cameras
            min_overlap_ratio: Minimum required detection overlap ratio
        """
        self.max_position_error = max_position_error
        self.min_overlap_ratio = min_overlap_ratio

    def validate_multi_camera_detection(self,
                                       detections: Dict[int, np.ndarray],
                                       detection_id: Optional[int] = None) -> ValidationResult:
        """
        Validate detection consistency across multiple cameras

        Args:
            detections: Dictionary mapping camera_id to 3D position
            detection_id: Optional detection ID

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        if len(detections) < 2:
            return result

        # Calculate pairwise position differences
        positions = list(detections.values())
        camera_ids = list(detections.keys())

        max_error = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                error = np.linalg.norm(positions[i] - positions[j])
                max_error = max(max_error, error)

                if error > self.max_position_error:
                    result.add_issue(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        type=ValidationType.CROSS_CAMERA,
                        message=f"Position mismatch between cameras {camera_ids[i]} and {camera_ids[j]}: {error:.2f}m",
                        detection_id=detection_id,
                        value=error,
                        expected=f"<= {self.max_position_error}m",
                        suggestion="Check camera calibration and synchronization"
                    ))

        return result

    def validate_detection_overlap(self, camera1_detections: List[Dict],
                                  camera2_detections: List[Dict],
                                  camera1_id: int, camera2_id: int) -> ValidationResult:
        """
        Validate detection overlap between two cameras

        Args:
            camera1_detections: Detections from camera 1
            camera2_detections: Detections from camera 2
            camera1_id: Camera 1 ID
            camera2_id: Camera 2 ID

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        n1 = len(camera1_detections)
        n2 = len(camera2_detections)

        if n1 == 0 or n2 == 0:
            overlap_ratio = 0.0
        else:
            # Simple overlap: count matching detections (within threshold)
            matches = 0
            for det1 in camera1_detections:
                pos1 = np.array([det1['x'], det1['y'], det1.get('z', 0)])
                for det2 in camera2_detections:
                    pos2 = np.array([det2['x'], det2['y'], det2.get('z', 0)])
                    if np.linalg.norm(pos1 - pos2) < self.max_position_error:
                        matches += 1
                        break

            overlap_ratio = matches / max(n1, n2)

        if overlap_ratio < self.min_overlap_ratio:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING,
                type=ValidationType.CROSS_CAMERA,
                message=f"Low detection overlap between cameras {camera1_id} and {camera2_id}: {overlap_ratio:.1%}",
                camera_id=camera1_id,
                value=overlap_ratio,
                expected=f">= {self.min_overlap_ratio:.1%}",
                suggestion="Check camera FOV overlap and detection sensitivity"
            ))

        return result


class OutlierDetector:
    """
    Statistical outlier detection for tracks and detections
    """

    def __init__(self, history_size: int = 30, z_score_threshold: float = 3.0):
        """
        Initialize outlier detector

        Args:
            history_size: Number of historical samples to keep
            z_score_threshold: Z-score threshold for outlier detection
        """
        self.history_size = history_size
        self.z_score_threshold = z_score_threshold

        # Historical data per track
        self.position_history: Dict[int, deque] = {}
        self.velocity_history: Dict[int, deque] = {}

    def detect_position_outlier(self, track_id: int, position: np.ndarray) -> ValidationResult:
        """
        Detect if position is an outlier based on history

        Args:
            track_id: Track ID
            position: Current position

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        # Initialize history if needed
        if track_id not in self.position_history:
            self.position_history[track_id] = deque(maxlen=self.history_size)

        history = self.position_history[track_id]
        history.append(position.copy())

        # Need sufficient history for outlier detection
        if len(history) < 10:
            return result

        # Calculate statistics
        positions = np.array(list(history))
        mean = np.mean(positions, axis=0)
        std = np.std(positions, axis=0)

        # Avoid division by zero
        std = np.where(std < 1e-6, 1e-6, std)

        # Calculate z-scores
        z_scores = np.abs((position - mean) / std)
        max_z_score = np.max(z_scores)

        if max_z_score > self.z_score_threshold:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING,
                type=ValidationType.OUTLIER,
                message=f"Position outlier detected (z-score: {max_z_score:.2f})",
                track_id=track_id,
                value=max_z_score,
                expected=f"< {self.z_score_threshold}",
                suggestion="Possible measurement error or track swap"
            ))

        return result

    def detect_velocity_outlier(self, track_id: int, velocity: np.ndarray) -> ValidationResult:
        """
        Detect if velocity is an outlier based on history

        Args:
            track_id: Track ID
            velocity: Current velocity

        Returns:
            ValidationResult
        """
        result = ValidationResult(passed=True)

        # Initialize history if needed
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = deque(maxlen=self.history_size)

        history = self.velocity_history[track_id]
        history.append(velocity.copy())

        # Need sufficient history
        if len(history) < 10:
            return result

        # Calculate statistics
        velocities = np.array(list(history))
        speeds = np.linalg.norm(velocities, axis=1)
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)

        if std_speed < 1e-6:
            std_speed = 1e-6

        current_speed = np.linalg.norm(velocity)
        z_score = abs((current_speed - mean_speed) / std_speed)

        if z_score > self.z_score_threshold:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.WARNING,
                type=ValidationType.OUTLIER,
                message=f"Velocity outlier detected (z-score: {z_score:.2f})",
                track_id=track_id,
                value=z_score,
                expected=f"< {self.z_score_threshold}",
                suggestion="Check for sudden maneuver or tracking error"
            ))

        return result


class DataValidator:
    """
    Main data validator coordinating all validation checks
    """

    def __init__(self,
                 bounds: CoordinateBounds = None,
                 min_confidence: float = 0.5,
                 max_velocity: float = 100.0,
                 max_acceleration: float = 50.0):
        """
        Initialize data validator

        Args:
            bounds: Coordinate bounds
            min_confidence: Minimum detection confidence
            max_velocity: Maximum plausible velocity (m/s)
            max_acceleration: Maximum plausible acceleration (m/s^2)
        """
        self.coord_validator = CoordinateValidator(bounds)
        self.conf_validator = ConfidenceValidator(min_confidence)
        self.temporal_validator = TemporalValidator(max_velocity, max_acceleration)
        self.cross_camera_validator = CrossCameraValidator()
        self.outlier_detector = OutlierDetector()

        # Statistics
        self.total_validations = 0
        self.total_issues = 0
        self.issues_by_type: Dict[ValidationType, int] = {}
        self.validation_times: deque = deque(maxlen=100)

        logger.info("DataValidator initialized")

    def validate_detection(self, detection: Dict,
                          camera_id: Optional[int] = None) -> ValidationResult:
        """
        Validate a single detection

        Args:
            detection: Detection dictionary with keys: x, y, z, confidence, etc.
            camera_id: Optional camera ID

        Returns:
            ValidationResult
        """
        start_time = time.time()
        result = ValidationResult(passed=True)

        # Validate position
        position = np.array([detection['x'], detection['y'], detection.get('z', 0)])
        coord_result = self.coord_validator.validate_coordinate(position, "Detection position")
        result.issues.extend(coord_result.issues)

        # Validate confidence
        if 'confidence' in detection:
            conf_result = self.conf_validator.validate_confidence(
                detection['confidence'],
                detection.get('detection_id')
            )
            result.issues.extend(conf_result.issues)

        # Update camera ID in issues
        for issue in result.issues:
            if issue.camera_id is None:
                issue.camera_id = camera_id

        # Update statistics
        self._update_statistics(result, time.time() - start_time)

        return result

    def validate_track(self, track: Dict, previous_track: Optional[Dict] = None,
                      dt: float = 0.033) -> ValidationResult:
        """
        Validate a track with temporal checks

        Args:
            track: Track dictionary with position, velocity, acceleration
            previous_track: Previous track state for temporal validation
            dt: Time delta between tracks

        Returns:
            ValidationResult
        """
        start_time = time.time()
        result = ValidationResult(passed=True)

        track_id = track.get('track_id')

        # Validate position
        position = np.array([track['x'], track['y'], track.get('z', 0)])
        coord_result = self.coord_validator.validate_coordinate(position, "Track position")
        result.issues.extend(coord_result.issues)

        # Validate velocity
        if 'vx' in track and 'vy' in track:
            velocity = np.array([track['vx'], track['vy'], track.get('vz', 0)])
            vel_result = self.temporal_validator.validate_velocity(velocity, track_id)
            result.issues.extend(vel_result.issues)

            # Outlier detection
            outlier_result = self.outlier_detector.detect_velocity_outlier(track_id, velocity)
            result.issues.extend(outlier_result.issues)

        # Validate acceleration
        if 'ax' in track and 'ay' in track:
            accel = np.array([track['ax'], track['ay'], track.get('az', 0)])
            accel_result = self.temporal_validator.validate_acceleration(accel, track_id)
            result.issues.extend(accel_result.issues)

        # Temporal coherence check
        if previous_track:
            prev_position = np.array([previous_track['x'], previous_track['y'],
                                    previous_track.get('z', 0)])
            jump_result = self.temporal_validator.validate_position_jump(
                prev_position, position, dt, track_id
            )
            result.issues.extend(jump_result.issues)

        # Position outlier detection
        outlier_result = self.outlier_detector.detect_position_outlier(track_id, position)
        result.issues.extend(outlier_result.issues)

        # Validate confidence
        if 'confidence' in track:
            conf_result = self.conf_validator.validate_confidence(track['confidence'])
            result.issues.extend(conf_result.issues)

        # Update statistics
        self._update_statistics(result, time.time() - start_time)

        return result

    def validate_multi_camera_detection(self,
                                       detections: Dict[int, np.ndarray],
                                       detection_id: Optional[int] = None) -> ValidationResult:
        """
        Validate detection consistency across cameras

        Args:
            detections: Dictionary mapping camera_id to 3D position
            detection_id: Optional detection ID

        Returns:
            ValidationResult
        """
        start_time = time.time()

        result = self.cross_camera_validator.validate_multi_camera_detection(
            detections, detection_id
        )

        self._update_statistics(result, time.time() - start_time)
        return result

    def _update_statistics(self, result: ValidationResult, validation_time: float):
        """Update validation statistics"""
        self.total_validations += 1
        self.total_issues += len(result.issues)
        self.validation_times.append(validation_time * 1000.0)  # Convert to ms

        result.validation_time_ms = validation_time * 1000.0

        for issue in result.issues:
            self.issues_by_type[issue.type] = self.issues_by_type.get(issue.type, 0) + 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        avg_time = np.mean(self.validation_times) if self.validation_times else 0.0

        return {
            'total_validations': self.total_validations,
            'total_issues': self.total_issues,
            'avg_validation_time_ms': avg_time,
            'issues_by_type': {k.value: v for k, v in self.issues_by_type.items()},
            'issue_rate': self.total_issues / max(self.total_validations, 1)
        }

    def reset_statistics(self):
        """Reset validation statistics"""
        self.total_validations = 0
        self.total_issues = 0
        self.issues_by_type.clear()
        self.validation_times.clear()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Data Validator - Comprehensive Validation System")
    print("=" * 60)

    # Create validator
    validator = DataValidator(
        bounds=CoordinateBounds(
            x_min=-5000, x_max=5000,
            y_min=-5000, y_max=5000,
            z_min=0, z_max=2000
        ),
        min_confidence=0.5,
        max_velocity=100.0
    )

    # Test detection validation
    print("\nValidating detections:")
    detections = [
        {'x': 100.0, 'y': 200.0, 'z': 50.0, 'confidence': 0.95},
        {'x': 10000.0, 'y': 200.0, 'z': 50.0, 'confidence': 0.85},  # Out of bounds
        {'x': 100.0, 'y': 200.0, 'z': 50.0, 'confidence': 0.3},  # Low confidence
    ]

    for i, det in enumerate(detections):
        result = validator.validate_detection(det, camera_id=i)
        print(f"\nDetection {i}: {'PASS' if result.passed else 'FAIL'}")
        for issue in result.issues:
            print(f"  [{issue.level.value.upper()}] {issue.message}")

    # Test track validation
    print("\n\nValidating tracks:")
    track = {
        'track_id': 1,
        'x': 100.0, 'y': 200.0, 'z': 50.0,
        'vx': 10.0, 'vy': 5.0, 'vz': 0.0,
        'ax': 0.5, 'ay': 0.2, 'az': 0.0,
        'confidence': 0.92
    }

    result = validator.validate_track(track)
    print(f"\nTrack validation: {'PASS' if result.passed else 'FAIL'}")
    for issue in result.issues:
        print(f"  [{issue.level.value.upper()}] {issue.message}")

    # Print statistics
    print("\n" + "=" * 60)
    stats = validator.get_statistics()
    print("Validation Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
