"""
Detection Fusion - Combine monochrome motion detection with thermal signatures
Provides cross-validation, confidence weighting, and occlusion handling
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict


class DetectionSource(Enum):
    """Source of detection"""
    THERMAL = "thermal"
    MONO = "mono"
    MOTION = "motion"
    FUSED = "fused"


class OcclusionType(Enum):
    """Type of occlusion"""
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"
    INTERMITTENT = "intermittent"


@dataclass
class MotionDetection:
    """Motion detection from monochrome camera"""
    x: float
    y: float
    width: float
    height: float
    velocity_x: float
    velocity_y: float
    motion_confidence: float
    frame_id: int
    timestamp: float


@dataclass
class ThermalDetection:
    """Thermal signature detection"""
    x: float
    y: float
    width: float
    height: float
    temperature_kelvin: float
    thermal_confidence: float
    signature_strength: float
    frame_id: int
    timestamp: float


@dataclass
class FusedDetection:
    """Fused detection combining motion and thermal"""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    source: DetectionSource
    track_id: int = -1

    # Multi-modal attributes
    motion_confidence: float = 0.0
    thermal_confidence: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    temperature_kelvin: Optional[float] = None
    thermal_signature: float = 0.0

    # Validation flags
    confirmed_by_motion: bool = False
    confirmed_by_thermal: bool = False
    cross_validated: bool = False

    # Occlusion handling
    occlusion_state: OcclusionType = OcclusionType.NONE
    frames_occluded: int = 0
    predicted_position: Optional[Tuple[float, float]] = None

    # Timing
    timestamp: float = 0.0
    frame_id: int = 0

    # Additional metadata
    metadata: Dict = field(default_factory=dict)


@dataclass
class FusionWeights:
    """Weights for multi-modal fusion"""
    motion_weight: float = 0.5
    thermal_weight: float = 0.5
    signature_weight: float = 0.3
    spatial_weight: float = 0.7
    temporal_weight: float = 0.4


class DetectionFusion:
    """
    Fuses detections from monochrome motion detection and thermal signatures
    Provides cross-validation, confidence weighting, and occlusion handling
    """

    def __init__(
        self,
        fusion_weights: Optional[FusionWeights] = None,
        iou_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
        max_track_age: int = 30,
        occlusion_threshold: int = 5
    ):
        """
        Initialize detection fusion system

        Args:
            fusion_weights: Weights for combining different detection modalities
            iou_threshold: IoU threshold for matching detections
            confidence_threshold: Minimum confidence for accepting detections
            max_track_age: Maximum frames to keep track alive without detection
            occlusion_threshold: Frames before declaring occlusion
        """
        self.fusion_weights = fusion_weights or FusionWeights()
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.max_track_age = max_track_age
        self.occlusion_threshold = occlusion_threshold

        # Track management
        self.active_tracks: Dict[int, List[FusedDetection]] = defaultdict(list)
        self.next_track_id = 0
        self.occluded_tracks: Dict[int, List[FusedDetection]] = defaultdict(list)

        # Performance metrics
        self.total_detections = 0
        self.cross_validated_detections = 0
        self.false_positives_rejected = 0
        self.occlusions_handled = 0

        logging.info("DetectionFusion initialized")

    def fuse_detections(
        self,
        motion_detections: List[MotionDetection],
        thermal_detections: List[ThermalDetection],
        frame_id: int,
        timestamp: float
    ) -> List[FusedDetection]:
        """
        Fuse motion and thermal detections

        Args:
            motion_detections: List of motion detections from mono camera
            thermal_detections: List of thermal signature detections
            frame_id: Current frame ID
            timestamp: Current timestamp

        Returns:
            List of fused detections with confidence weighting
        """
        fused_detections = []

        # Step 1: Match motion and thermal detections
        matched_pairs, unmatched_motion, unmatched_thermal = self._match_detections(
            motion_detections, thermal_detections
        )

        # Step 2: Create fused detections from matched pairs (highest confidence)
        for motion_det, thermal_det in matched_pairs:
            fused = self._create_fused_detection(
                motion_det, thermal_det, frame_id, timestamp
            )
            fused.cross_validated = True
            fused.confirmed_by_motion = True
            fused.confirmed_by_thermal = True
            fused_detections.append(fused)
            self.cross_validated_detections += 1

        # Step 3: Add high-confidence unmatched motion detections
        for motion_det in unmatched_motion:
            if motion_det.motion_confidence > 0.7:
                fused = self._motion_to_fused(motion_det, frame_id, timestamp)
                fused.confirmed_by_motion = True
                fused_detections.append(fused)

        # Step 4: Add high-confidence unmatched thermal detections
        for thermal_det in unmatched_thermal:
            if thermal_det.thermal_confidence > 0.7:
                fused = self._thermal_to_fused(thermal_det, frame_id, timestamp)
                fused.confirmed_by_thermal = True
                fused_detections.append(fused)

        # Step 5: Cross-validate detections to reduce false positives
        validated_detections = self._cross_validate(fused_detections)

        # Step 6: Update tracks and handle occlusions
        tracked_detections = self._update_tracks(
            validated_detections, frame_id, timestamp
        )

        # Step 7: Handle occluded tracks
        recovered_detections = self._handle_occlusions(frame_id, timestamp)
        tracked_detections.extend(recovered_detections)

        self.total_detections += len(tracked_detections)

        return tracked_detections

    def _match_detections(
        self,
        motion_detections: List[MotionDetection],
        thermal_detections: List[ThermalDetection]
    ) -> Tuple[List[Tuple[MotionDetection, ThermalDetection]],
               List[MotionDetection],
               List[ThermalDetection]]:
        """
        Match motion and thermal detections using IoU and spatial proximity

        Returns:
            Tuple of (matched_pairs, unmatched_motion, unmatched_thermal)
        """
        matched_pairs = []
        matched_motion_idx: Set[int] = set()
        matched_thermal_idx: Set[int] = set()

        # Compute IoU matrix
        for i, motion_det in enumerate(motion_detections):
            best_iou = 0.0
            best_thermal_idx = -1

            for j, thermal_det in enumerate(thermal_detections):
                if j in matched_thermal_idx:
                    continue

                iou = self._compute_iou(
                    (motion_det.x, motion_det.y, motion_det.width, motion_det.height),
                    (thermal_det.x, thermal_det.y, thermal_det.width, thermal_det.height)
                )

                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_thermal_idx = j

            if best_thermal_idx >= 0:
                matched_pairs.append((motion_det, thermal_detections[best_thermal_idx]))
                matched_motion_idx.add(i)
                matched_thermal_idx.add(best_thermal_idx)

        # Collect unmatched detections
        unmatched_motion = [
            det for i, det in enumerate(motion_detections)
            if i not in matched_motion_idx
        ]
        unmatched_thermal = [
            det for i, det in enumerate(thermal_detections)
            if i not in matched_thermal_idx
        ]

        return matched_pairs, unmatched_motion, unmatched_thermal

    def _compute_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Compute Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to corner coordinates
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2

        # Compute intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0

        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / (union + 1e-8)

    def _create_fused_detection(
        self,
        motion_det: MotionDetection,
        thermal_det: ThermalDetection,
        frame_id: int,
        timestamp: float
    ) -> FusedDetection:
        """Create fused detection from motion and thermal detections"""
        # Weighted fusion of position and size
        w = self.fusion_weights

        x = w.spatial_weight * motion_det.x + (1 - w.spatial_weight) * thermal_det.x
        y = w.spatial_weight * motion_det.y + (1 - w.spatial_weight) * thermal_det.y
        width = (motion_det.width + thermal_det.width) / 2
        height = (motion_det.height + thermal_det.height) / 2

        # Confidence fusion
        motion_conf = motion_det.motion_confidence * w.motion_weight
        thermal_conf = thermal_det.thermal_confidence * w.thermal_weight
        signature_conf = thermal_det.signature_strength * w.signature_weight

        confidence = motion_conf + thermal_conf + signature_conf
        confidence = min(1.0, confidence)  # Clamp to [0, 1]

        fused = FusedDetection(
            x=x,
            y=y,
            width=width,
            height=height,
            confidence=confidence,
            source=DetectionSource.FUSED,
            motion_confidence=motion_det.motion_confidence,
            thermal_confidence=thermal_det.thermal_confidence,
            velocity_x=motion_det.velocity_x,
            velocity_y=motion_det.velocity_y,
            temperature_kelvin=thermal_det.temperature_kelvin,
            thermal_signature=thermal_det.signature_strength,
            timestamp=timestamp,
            frame_id=frame_id
        )

        return fused

    def _motion_to_fused(
        self,
        motion_det: MotionDetection,
        frame_id: int,
        timestamp: float
    ) -> FusedDetection:
        """Convert motion detection to fused detection"""
        confidence = motion_det.motion_confidence * self.fusion_weights.motion_weight

        return FusedDetection(
            x=motion_det.x,
            y=motion_det.y,
            width=motion_det.width,
            height=motion_det.height,
            confidence=confidence,
            source=DetectionSource.MOTION,
            motion_confidence=motion_det.motion_confidence,
            velocity_x=motion_det.velocity_x,
            velocity_y=motion_det.velocity_y,
            timestamp=timestamp,
            frame_id=frame_id
        )

    def _thermal_to_fused(
        self,
        thermal_det: ThermalDetection,
        frame_id: int,
        timestamp: float
    ) -> FusedDetection:
        """Convert thermal detection to fused detection"""
        confidence = (
            thermal_det.thermal_confidence * self.fusion_weights.thermal_weight +
            thermal_det.signature_strength * self.fusion_weights.signature_weight
        )

        return FusedDetection(
            x=thermal_det.x,
            y=thermal_det.y,
            width=thermal_det.width,
            height=thermal_det.height,
            confidence=confidence,
            source=DetectionSource.THERMAL,
            thermal_confidence=thermal_det.thermal_confidence,
            temperature_kelvin=thermal_det.temperature_kelvin,
            thermal_signature=thermal_det.signature_strength,
            timestamp=timestamp,
            frame_id=frame_id
        )

    def _cross_validate(
        self,
        detections: List[FusedDetection]
    ) -> List[FusedDetection]:
        """
        Cross-validate detections to reduce false positives
        Require confirmation from multiple modalities or high confidence
        """
        validated = []

        for det in detections:
            # High confidence cross-validated detections always pass
            if det.cross_validated and det.confidence > 0.8:
                validated.append(det)
                continue

            # For single-modality detections, require higher confidence
            if det.source == DetectionSource.MOTION:
                if det.motion_confidence > 0.8:
                    validated.append(det)
                else:
                    self.false_positives_rejected += 1
            elif det.source == DetectionSource.THERMAL:
                if det.thermal_confidence > 0.8 and det.thermal_signature > 0.5:
                    validated.append(det)
                else:
                    self.false_positives_rejected += 1
            elif det.source == DetectionSource.FUSED:
                if det.confidence > self.confidence_threshold:
                    validated.append(det)
                else:
                    self.false_positives_rejected += 1

        return validated

    def _update_tracks(
        self,
        detections: List[FusedDetection],
        frame_id: int,
        timestamp: float
    ) -> List[FusedDetection]:
        """Update tracking IDs and track history"""
        # Get active track predictions
        track_predictions = self._predict_track_positions()

        # Match detections to tracks
        for det in detections:
            best_track_id = -1
            best_match_score = 0.0

            for track_id, (pred_x, pred_y, history) in track_predictions.items():
                if not history:
                    continue

                prev_det = history[-1]

                # Spatial distance
                dx = det.x - pred_x
                dy = det.y - pred_y
                spatial_dist = np.sqrt(dx * dx + dy * dy)

                if spatial_dist < 100:  # Max tracking distance
                    spatial_score = 1.0 - (spatial_dist / 100.0)

                    # Signature similarity
                    sig_score = 1.0
                    if det.thermal_signature > 0 and prev_det.thermal_signature > 0:
                        sig_diff = abs(det.thermal_signature - prev_det.thermal_signature)
                        sig_score = np.exp(-sig_diff * 3.0)

                    # Velocity consistency
                    vel_score = 1.0
                    if prev_det.velocity_x != 0 or prev_det.velocity_y != 0:
                        vel_diff = np.sqrt(
                            (det.velocity_x - prev_det.velocity_x)**2 +
                            (det.velocity_y - prev_det.velocity_y)**2
                        )
                        vel_score = np.exp(-vel_diff * 0.1)

                    # Combined matching score
                    match_score = (
                        0.5 * spatial_score +
                        0.3 * sig_score +
                        0.2 * vel_score
                    )

                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_track_id = track_id

            # Assign track ID
            if best_track_id >= 0 and best_match_score > 0.5:
                det.track_id = best_track_id
            else:
                det.track_id = self.next_track_id
                self.next_track_id += 1

            # Update track history
            self.active_tracks[det.track_id].append(det)

            # Keep limited history
            if len(self.active_tracks[det.track_id]) > self.max_track_age:
                self.active_tracks[det.track_id].pop(0)

        # Clean up old tracks
        self._cleanup_old_tracks(frame_id)

        return detections

    def _predict_track_positions(self) -> Dict[int, Tuple[float, float, List[FusedDetection]]]:
        """Predict next position for each active track"""
        predictions = {}

        for track_id, history in self.active_tracks.items():
            if not history:
                continue

            latest = history[-1]

            # Simple constant velocity prediction
            pred_x = latest.x + latest.velocity_x
            pred_y = latest.y + latest.velocity_y

            # If we have multiple frames, use better prediction
            if len(history) >= 2:
                prev = history[-2]
                actual_vx = latest.x - prev.x
                actual_vy = latest.y - prev.y
                pred_x = latest.x + actual_vx
                pred_y = latest.y + actual_vy

            predictions[track_id] = (pred_x, pred_y, history)

        return predictions

    def _handle_occlusions(
        self,
        frame_id: int,
        timestamp: float
    ) -> List[FusedDetection]:
        """Handle occluded tracks and attempt recovery"""
        recovered = []

        # Check active tracks for occlusion
        for track_id, history in list(self.active_tracks.items()):
            if not history:
                continue

            latest = history[-1]
            frames_since_detection = frame_id - latest.frame_id

            if frames_since_detection > self.occlusion_threshold:
                # Track is occluded
                latest.occlusion_state = OcclusionType.FULL
                latest.frames_occluded = frames_since_detection

                # Predict position during occlusion
                pred_x = latest.x + latest.velocity_x * frames_since_detection
                pred_y = latest.y + latest.velocity_y * frames_since_detection
                latest.predicted_position = (pred_x, pred_y)

                # Move to occluded tracks
                self.occluded_tracks[track_id] = self.active_tracks[track_id]
                del self.active_tracks[track_id]

                self.occlusions_handled += 1

        # Try to recover occluded tracks
        for track_id, history in list(self.occluded_tracks.items()):
            latest = history[-1]
            frames_occluded = frame_id - latest.frame_id

            if frames_occluded > self.max_track_age:
                # Track lost
                del self.occluded_tracks[track_id]
            else:
                # Create predicted detection for occluded track
                pred_x, pred_y = latest.predicted_position or (latest.x, latest.y)

                predicted_det = FusedDetection(
                    x=pred_x,
                    y=pred_y,
                    width=latest.width,
                    height=latest.height,
                    confidence=latest.confidence * 0.5,  # Lower confidence for predicted
                    source=latest.source,
                    track_id=track_id,
                    motion_confidence=latest.motion_confidence * 0.5,
                    thermal_confidence=latest.thermal_confidence * 0.5,
                    velocity_x=latest.velocity_x,
                    velocity_y=latest.velocity_y,
                    occlusion_state=OcclusionType.FULL,
                    frames_occluded=frames_occluded,
                    timestamp=timestamp,
                    frame_id=frame_id
                )

                recovered.append(predicted_det)

        return recovered

    def _cleanup_old_tracks(self, current_frame_id: int):
        """Remove tracks that haven't been updated recently"""
        tracks_to_remove = []

        for track_id, history in self.active_tracks.items():
            if not history:
                tracks_to_remove.append(track_id)
                continue

            latest = history[-1]
            age = current_frame_id - latest.frame_id

            if age > self.max_track_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get fusion performance metrics"""
        if self.total_detections == 0:
            return {
                'total_detections': 0,
                'cross_validation_rate': 0.0,
                'false_positive_reduction_rate': 0.0,
                'active_tracks': 0,
                'occluded_tracks': 0,
                'occlusions_handled': 0
            }

        cross_val_rate = self.cross_validated_detections / self.total_detections
        fp_rate = self.false_positives_rejected / (
            self.total_detections + self.false_positives_rejected
        )

        return {
            'total_detections': self.total_detections,
            'cross_validation_rate': cross_val_rate,
            'false_positive_reduction_rate': fp_rate,
            'active_tracks': len(self.active_tracks),
            'occluded_tracks': len(self.occluded_tracks),
            'occlusions_handled': self.occlusions_handled
        }

    def reset_metrics(self):
        """Reset performance metrics"""
        self.total_detections = 0
        self.cross_validated_detections = 0
        self.false_positives_rejected = 0
        self.occlusions_handled = 0
