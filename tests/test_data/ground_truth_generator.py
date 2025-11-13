"""
Ground Truth Data Generator
Generates ground truth annotations for validation and accuracy testing
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthDetection:
    """Ground truth detection annotation"""
    frame_id: int
    target_id: int
    position_2d: Tuple[float, float]  # pixel coordinates
    position_3d: Tuple[float, float, float]  # world coordinates
    bounding_box: Tuple[float, float, float, float]  # x, y, width, height
    velocity_2d: Tuple[float, float]  # pixel velocity
    velocity_3d: Tuple[float, float, float]  # world velocity
    distance_m: float
    visible: bool
    occluded: bool
    truncated: bool
    confidence: float = 1.0  # Ground truth confidence


@dataclass
class GroundTruthFrame:
    """Ground truth annotations for a single frame"""
    frame_id: int
    timestamp: float
    camera_id: int
    detections: List[GroundTruthDetection]
    metadata: Dict


class GroundTruthGenerator:
    """Generate ground truth annotations from simulated data"""

    def __init__(self, frame_width: int = 7680, frame_height: int = 4320):
        """
        Initialize ground truth generator

        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

    def generate_from_trajectories(
        self,
        trajectories: Dict,
        projection_func,
        num_frames: int,
        camera_id: int = 0
    ) -> List[GroundTruthFrame]:
        """
        Generate ground truth from trajectory data

        Args:
            trajectories: Dictionary of drone trajectories
            projection_func: Function to project 3D to 2D coordinates
            num_frames: Number of frames to generate
            camera_id: Camera ID for annotations

        Returns:
            List of ground truth frames
        """
        ground_truth_frames = []

        for frame_id in range(num_frames):
            detections = []

            for drone_id, trajectory in trajectories.items():
                if frame_id < len(trajectory.points):
                    point = trajectory.points[frame_id]

                    # Project 3D position to 2D
                    pixel_x, pixel_y, distance = projection_func(point.position)

                    if pixel_x is not None and pixel_y is not None:
                        # Check visibility
                        visible = (
                            0 <= pixel_x < self.frame_width and
                            0 <= pixel_y < self.frame_height and
                            distance > 0
                        )

                        # Determine occlusion (simplified)
                        occluded = False

                        # Determine truncation
                        truncated = (
                            pixel_x < 10 or pixel_x > self.frame_width - 10 or
                            pixel_y < 10 or pixel_y > self.frame_height - 10
                        )

                        # Calculate bounding box (simplified)
                        # In reality, this would depend on drone size and distance
                        pixel_size = max(5, 100 / max(distance / 100, 1))
                        bbox = (
                            pixel_x - pixel_size / 2,
                            pixel_y - pixel_size / 2,
                            pixel_size,
                            pixel_size
                        )

                        detection = GroundTruthDetection(
                            frame_id=frame_id,
                            target_id=drone_id,
                            position_2d=(pixel_x, pixel_y),
                            position_3d=point.position,
                            bounding_box=bbox,
                            velocity_2d=(0, 0),  # Would need to calculate from consecutive frames
                            velocity_3d=point.velocity,
                            distance_m=distance,
                            visible=visible,
                            occluded=occluded,
                            truncated=truncated
                        )

                        detections.append(detection)

            gt_frame = GroundTruthFrame(
                frame_id=frame_id,
                timestamp=frame_id / 30.0,  # Assuming 30 FPS
                camera_id=camera_id,
                detections=detections,
                metadata={'num_targets': len(detections)}
            )

            ground_truth_frames.append(gt_frame)

        logger.info(f"Generated ground truth for {num_frames} frames")
        return ground_truth_frames

    def calculate_detection_metrics(
        self,
        ground_truth: GroundTruthFrame,
        predicted_detections: List[Dict],
        iou_threshold: float = 0.5,
        distance_threshold: float = 50.0
    ) -> Dict:
        """
        Calculate detection metrics by comparing predictions to ground truth

        Args:
            ground_truth: Ground truth frame
            predicted_detections: Predicted detections
            iou_threshold: IoU threshold for matching
            distance_threshold: Distance threshold for matching (pixels)

        Returns:
            Metrics dictionary
        """
        gt_visible = [d for d in ground_truth.detections if d.visible and not d.occluded]

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        matched_distances = []

        matched_gt = set()

        # Match predictions to ground truth
        for pred in predicted_detections:
            pred_x = pred.get('x', pred.get('pixel_x'))
            pred_y = pred.get('y', pred.get('pixel_y'))

            best_match = None
            best_distance = float('inf')

            for i, gt_det in enumerate(gt_visible):
                if i in matched_gt:
                    continue

                gt_x, gt_y = gt_det.position_2d
                distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

                if distance < distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = i

            if best_match is not None:
                true_positives += 1
                matched_gt.add(best_match)
                matched_distances.append(best_distance)
            else:
                false_positives += 1

        false_negatives = len(gt_visible) - true_positives

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        avg_distance_error = np.mean(matched_distances) if matched_distances else 0

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_distance_error': avg_distance_error,
            'num_ground_truth': len(gt_visible),
            'num_predictions': len(predicted_detections)
        }

    def save_ground_truth(self, frames: List[GroundTruthFrame], filename: str):
        """Save ground truth to JSON file"""
        data = {
            'metadata': {
                'num_frames': len(frames),
                'frame_width': self.frame_width,
                'frame_height': self.frame_height
            },
            'frames': []
        }

        for frame in frames:
            frame_data = {
                'frame_id': frame.frame_id,
                'timestamp': frame.timestamp,
                'camera_id': frame.camera_id,
                'metadata': frame.metadata,
                'detections': [asdict(det) for det in frame.detections]
            }
            data['frames'].append(frame_data)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved ground truth to {filename}")

    def load_ground_truth(self, filename: str) -> List[GroundTruthFrame]:
        """Load ground truth from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)

        frames = []
        for frame_data in data['frames']:
            detections = [
                GroundTruthDetection(
                    frame_id=det['frame_id'],
                    target_id=det['target_id'],
                    position_2d=tuple(det['position_2d']),
                    position_3d=tuple(det['position_3d']),
                    bounding_box=tuple(det['bounding_box']),
                    velocity_2d=tuple(det['velocity_2d']),
                    velocity_3d=tuple(det['velocity_3d']),
                    distance_m=det['distance_m'],
                    visible=det['visible'],
                    occluded=det['occluded'],
                    truncated=det['truncated'],
                    confidence=det.get('confidence', 1.0)
                )
                for det in frame_data['detections']
            ]

            frame = GroundTruthFrame(
                frame_id=frame_data['frame_id'],
                timestamp=frame_data['timestamp'],
                camera_id=frame_data['camera_id'],
                detections=detections,
                metadata=frame_data['metadata']
            )
            frames.append(frame)

        logger.info(f"Loaded {len(frames)} ground truth frames from {filename}")
        return frames

    def generate_validation_report(
        self,
        ground_truth_frames: List[GroundTruthFrame],
        all_predictions: List[List[Dict]]
    ) -> Dict:
        """
        Generate comprehensive validation report

        Args:
            ground_truth_frames: List of ground truth frames
            all_predictions: List of predictions for each frame

        Returns:
            Validation report dictionary
        """
        all_metrics = []

        for gt_frame, predictions in zip(ground_truth_frames, all_predictions):
            metrics = self.calculate_detection_metrics(gt_frame, predictions)
            all_metrics.append(metrics)

        # Aggregate metrics
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in all_metrics])

        total_tp = sum(m['true_positives'] for m in all_metrics)
        total_fp = sum(m['false_positives'] for m in all_metrics)
        total_fn = sum(m['false_negatives'] for m in all_metrics)

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        report = {
            'summary': {
                'num_frames': len(ground_truth_frames),
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1_score': avg_f1,
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1_score': overall_f1,
                'total_true_positives': total_tp,
                'total_false_positives': total_fp,
                'total_false_negatives': total_fn
            },
            'per_frame_metrics': all_metrics
        }

        logger.info(f"Validation report generated:")
        logger.info(f"  Overall Precision: {overall_precision:.3f}")
        logger.info(f"  Overall Recall: {overall_recall:.3f}")
        logger.info(f"  Overall F1: {overall_f1:.3f}")

        return report


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    generator = GroundTruthGenerator(width=1920, height=1080)

    # Create example ground truth
    example_detection = GroundTruthDetection(
        frame_id=0,
        target_id=0,
        position_2d=(960, 540),
        position_3d=(0, 0, 1000),
        bounding_box=(950, 530, 20, 20),
        velocity_2d=(5, 0),
        velocity_3d=(10, 0, 0),
        distance_m=1000.0,
        visible=True,
        occluded=False,
        truncated=False
    )

    gt_frame = GroundTruthFrame(
        frame_id=0,
        timestamp=0.0,
        camera_id=0,
        detections=[example_detection],
        metadata={'num_targets': 1}
    )

    # Save example
    generator.save_ground_truth([gt_frame], "/tmp/ground_truth.json")
    print("Saved example ground truth")
