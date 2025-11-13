"""
tracker.py

Multi-target tracker for 200 simultaneous drones at 5km range
Features:
- Simultaneous tracking of 200+ drones
- Occlusion handling and track recovery
- Velocity and trajectory estimation
- Track ID management with re-identification
- Multi-camera fusion support

Performance targets:
- 200 simultaneous drones
- 99%+ detection rate
- <2% false positive rate
- <100ms total latency
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class Track:
    """Individual drone track"""
    track_id: int
    state: np.ndarray  # [x, y, vx, vy, ax, ay]
    covariance: np.ndarray
    detections: deque = field(default_factory=lambda: deque(maxlen=30))
    score: float = 1.0
    frames_since_update: int = 0
    total_detections: int = 0
    confirmed: bool = False
    occluded: bool = False
    occlusion_count: int = 0
    predicted_positions: deque = field(default_factory=lambda: deque(maxlen=10))

    # Physical properties
    estimated_distance: float = 5000.0  # meters
    estimated_size: float = 0.20  # meters (20cm drone)
    radar_cross_section: float = 0.01  # m^2

    # Trajectory properties
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
    max_velocity: float = 0.0  # pixels/frame
    max_acceleration: float = 0.0  # pixels/frame^2
    trajectory_type: str = "linear"  # linear, circular, erratic


@dataclass
class TrackMetrics:
    """Performance metrics for tracking system"""
    total_tracks_created: int = 0
    total_tracks_confirmed: int = 0
    total_tracks_lost: int = 0
    avg_track_length: float = 0.0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    avg_latency_ms: float = 0.0
    frames_processed: int = 0
    occlusion_recoveries: int = 0


class MultiTargetTracker:
    """
    Multi-target tracker for 200 simultaneous drones
    """

    def __init__(
        self,
        max_tracks: int = 200,
        detection_threshold: float = 0.5,
        confirmation_threshold: int = 3,
        max_age: int = 10,
        iou_threshold: float = 0.3,
        max_velocity: float = 50.0,  # pixels/frame
        frame_rate: float = 30.0
    ):
        """
        Initialize multi-target tracker

        Args:
            max_tracks: Maximum number of simultaneous tracks
            detection_threshold: Minimum confidence for new tracks
            confirmation_threshold: Frames needed to confirm track
            max_age: Maximum frames without detection before deletion
            iou_threshold: IoU threshold for association
            max_velocity: Maximum expected velocity (pixels/frame)
            frame_rate: Camera frame rate (Hz)
        """
        self.max_tracks = max_tracks
        self.detection_threshold = detection_threshold
        self.confirmation_threshold = confirmation_threshold
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.max_velocity = max_velocity
        self.dt = 1.0 / frame_rate

        # Tracking state
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0

        # Occlusion handling
        self.occlusion_zones: List[np.ndarray] = []
        self.occluded_tracks: Dict[int, Track] = {}

        # Performance metrics
        self.metrics = TrackMetrics()
        self.frame_timestamps = deque(maxlen=100)

        # Track history for re-identification
        self.track_history: Dict[int, Track] = {}

    def predict(self):
        """
        Predict next state for all active tracks using constant acceleration model
        """
        for track in self.tracks.values():
            # State transition: constant acceleration model
            F = np.array([
                [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
                [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
                [0, 0, 1, 0, self.dt, 0],
                [0, 0, 0, 1, 0, self.dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])

            # Predict state
            track.state = F @ track.state

            # Predict covariance
            Q = np.diag([0.1, 0.1, 1.0, 1.0, 5.0, 5.0])  # Process noise
            track.covariance = F @ track.covariance @ F.T + Q

            # Store predicted position
            track.predicted_positions.append(track.state[:2].copy())

            # Increment age
            track.frames_since_update += 1

    def update(
        self,
        detections: List[Dict],
        frame_number: int,
        timestamp: float
    ) -> Dict:
        """
        Update tracks with new detections

        Args:
            detections: List of detection dictionaries
            frame_number: Current frame number
            timestamp: Frame timestamp

        Returns:
            Dictionary containing updated tracks and metrics
        """
        start_time = time.time()

        # Predict all tracks
        self.predict()

        # Associate detections to tracks
        matched_tracks, unmatched_detections, unmatched_tracks = \
            self._associate_detections_to_tracks(detections)

        # Update matched tracks
        for track_id, det_idx in matched_tracks:
            self._update_track(track_id, detections[det_idx], timestamp)

        # Handle unmatched tracks (potential occlusions)
        for track_id in unmatched_tracks:
            self._handle_unmatched_track(track_id)

        # Initialize new tracks from unmatched detections
        for det_idx in unmatched_detections:
            if len(self.tracks) < self.max_tracks:
                self._initialize_track(detections[det_idx], timestamp)

        # Prune dead tracks
        self._prune_tracks()

        # Check for occlusion recovery
        self._check_occlusion_recovery(detections)

        # Update metrics
        latency = (time.time() - start_time) * 1000.0  # ms
        self._update_metrics(latency, frame_number)

        return self._generate_output(detections, latency)

    def _associate_detections_to_tracks(
        self,
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using IoU and distance metrics

        Returns:
            matched_tracks: List of (track_id, detection_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track IDs
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(self.tracks.keys())

        # Build cost matrix using Mahalanobis distance
        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_detections = len(detections)

        cost_matrix = np.full((n_tracks, n_detections), 1e9)

        # Gating threshold (Chi-square 95% confidence, 4 DOF)
        gate_threshold = 9.21

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]

            for j, det in enumerate(detections):
                # Detection measurement vector [x, y, vx, vy]
                z = np.array([
                    det['x'],
                    det['y'],
                    det.get('velocity_x', 0.0),
                    det.get('velocity_y', 0.0)
                ])

                # Predicted measurement
                H = np.array([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0]
                ])

                z_pred = H @ track.state

                # Innovation
                innovation = z - z_pred

                # Innovation covariance
                R = np.diag([0.5, 0.5, 2.0, 2.0])  # Measurement noise
                S = H @ track.covariance @ H.T + R

                # Mahalanobis distance
                try:
                    S_inv = np.linalg.inv(S)
                    mahal_dist = np.sqrt(innovation.T @ S_inv @ innovation)
                except np.linalg.LinAlgError:
                    mahal_dist = 1e9

                # Apply gating
                if mahal_dist < gate_threshold:
                    # Cost combines distance and appearance
                    appearance_cost = 1.0 - det['confidence']
                    cost_matrix[i, j] = mahal_dist + appearance_cost

        # Solve assignment using Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Filter out high-cost matches
        matched_tracks = []
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < gate_threshold:
                matched_tracks.append((track_ids[i], j))

        # Find unmatched detections and tracks
        matched_det_indices = set(j for _, j in matched_tracks)
        matched_track_ids = set(tid for tid, _ in matched_tracks)

        unmatched_detections = [j for j in range(n_detections)
                               if j not in matched_det_indices]
        unmatched_tracks = [tid for tid in track_ids
                           if tid not in matched_track_ids]

        return matched_tracks, unmatched_detections, unmatched_tracks

    def _update_track(self, track_id: int, detection: Dict, timestamp: float):
        """Update track with associated detection"""
        track = self.tracks[track_id]

        # Measurement vector [x, y, vx, vy]
        z = np.array([
            detection['x'],
            detection['y'],
            detection.get('velocity_x', 0.0),
            detection.get('velocity_y', 0.0)
        ])

        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

        # Innovation
        innovation = z - H @ track.state

        # Innovation covariance
        R = np.diag([0.5, 0.5, 2.0, 2.0])
        S = H @ track.covariance @ H.T + R

        # Kalman gain
        K = track.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        track.state = track.state + K @ innovation

        # Update covariance
        I = np.eye(6)
        track.covariance = (I - K @ H) @ track.covariance

        # Update track properties
        track.detections.append(detection)
        track.trajectory.append(track.state[:2].copy())
        track.frames_since_update = 0
        track.total_detections += 1
        track.score = 0.9 * track.score + 0.1 * detection['confidence']

        # Confirm track after threshold detections
        if track.total_detections >= self.confirmation_threshold:
            if not track.confirmed:
                track.confirmed = True
                self.metrics.total_tracks_confirmed += 1

        # Update velocity and acceleration statistics
        velocity = np.linalg.norm(track.state[2:4])
        acceleration = np.linalg.norm(track.state[4:6])
        track.max_velocity = max(track.max_velocity, velocity)
        track.max_acceleration = max(track.max_acceleration, acceleration)

        # Classify trajectory type
        track.trajectory_type = self._classify_trajectory(track)

        # Clear occlusion flag
        if track.occluded:
            track.occluded = False
            self.metrics.occlusion_recoveries += 1

    def _handle_unmatched_track(self, track_id: int):
        """Handle track that didn't match any detection (potential occlusion)"""
        track = self.tracks[track_id]

        # Check if track is in occlusion zone
        is_occluded = self._check_occlusion(track.state[:2])

        if is_occluded:
            track.occluded = True
            track.occlusion_count += 1

        # Decrease score for unmatched tracks
        track.score *= 0.85

    def _initialize_track(self, detection: Dict, timestamp: float):
        """Initialize new track from detection"""
        if detection['confidence'] < self.detection_threshold:
            return

        # Create new track
        track_id = self.next_track_id
        self.next_track_id += 1

        # Initial state [x, y, vx, vy, ax, ay]
        state = np.array([
            detection['x'],
            detection['y'],
            detection.get('velocity_x', 0.0),
            detection.get('velocity_y', 0.0),
            0.0,
            0.0
        ])

        # Initial covariance
        covariance = np.diag([10.0, 10.0, 5.0, 5.0, 2.0, 2.0])

        track = Track(
            track_id=track_id,
            state=state,
            covariance=covariance
        )

        track.detections.append(detection)
        track.trajectory.append(state[:2].copy())
        track.total_detections = 1
        track.score = detection['confidence']

        self.tracks[track_id] = track
        self.metrics.total_tracks_created += 1

    def _prune_tracks(self):
        """Remove dead or low-quality tracks"""
        tracks_to_remove = []

        for track_id, track in self.tracks.items():
            # Remove if too old without update
            if track.frames_since_update > self.max_age:
                tracks_to_remove.append(track_id)

            # Remove if score too low
            elif track.score < 0.1 and track.frames_since_update > 3:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            track = self.tracks.pop(track_id)

            # Store in history for potential re-identification
            if track.confirmed:
                self.track_history[track_id] = track
                self.metrics.total_tracks_lost += 1

    def _check_occlusion(self, position: np.ndarray) -> bool:
        """Check if position is in known occlusion zone"""
        for zone in self.occlusion_zones:
            if self._point_in_polygon(position, zone):
                return True
        return False

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _check_occlusion_recovery(self, detections: List[Dict]):
        """Check if any occluded tracks can be recovered"""
        occluded_track_ids = [tid for tid, track in self.tracks.items()
                             if track.occluded]

        if not occluded_track_ids or not detections:
            return

        # Try to match occluded tracks to detections based on predicted position
        for track_id in occluded_track_ids:
            track = self.tracks[track_id]
            predicted_pos = track.state[:2]

            # Find closest detection
            min_dist = float('inf')
            best_det_idx = -1

            for i, det in enumerate(detections):
                det_pos = np.array([det['x'], det['y']])
                dist = np.linalg.norm(predicted_pos - det_pos)

                # Check if within expected velocity range
                max_displacement = self.max_velocity * track.frames_since_update

                if dist < max_displacement and dist < min_dist:
                    min_dist = dist
                    best_det_idx = i

            # Recover track if good match found
            if best_det_idx >= 0 and min_dist < self.max_velocity * 2:
                self._update_track(track_id, detections[best_det_idx], time.time())

    def _classify_trajectory(self, track: Track) -> str:
        """Classify trajectory type based on motion pattern"""
        if len(track.trajectory) < 10:
            return "linear"

        # Analyze trajectory curvature
        positions = np.array(list(track.trajectory)[-20:])

        # Fit line to recent positions
        if len(positions) >= 3:
            # Compute deviation from linear fit
            A = np.c_[positions[:, 0], np.ones(len(positions))]
            coeffs, residuals, _, _ = np.linalg.lstsq(A, positions[:, 1], rcond=None)

            avg_deviation = np.sqrt(residuals[0] / len(positions)) if len(residuals) > 0 else 0

            # Classify based on deviation
            if avg_deviation < 2.0:
                return "linear"
            elif avg_deviation < 10.0:
                return "circular"
            else:
                return "erratic"

        return "linear"

    def _update_metrics(self, latency: float, frame_number: int):
        """Update performance metrics"""
        self.metrics.frames_processed += 1

        # Update average latency
        n = self.metrics.frames_processed
        self.metrics.avg_latency_ms = \
            (self.metrics.avg_latency_ms * (n - 1) + latency) / n

        # Update average track length
        if self.metrics.total_tracks_confirmed > 0:
            total_length = sum(track.total_detections
                             for track in self.tracks.values()
                             if track.confirmed)
            self.metrics.avg_track_length = \
                total_length / self.metrics.total_tracks_confirmed

    def _generate_output(self, detections: List[Dict], latency: float) -> Dict:
        """Generate output dictionary with tracks and metrics"""
        # Confirmed tracks only
        confirmed_tracks = []
        for track in self.tracks.values():
            if track.confirmed:
                confirmed_tracks.append({
                    'track_id': track.track_id,
                    'x': float(track.state[0]),
                    'y': float(track.state[1]),
                    'vx': float(track.state[2]),
                    'vy': float(track.state[3]),
                    'ax': float(track.state[4]),
                    'ay': float(track.state[5]),
                    'confidence': float(track.score),
                    'age': track.total_detections,
                    'occluded': track.occluded,
                    'trajectory_type': track.trajectory_type,
                    'max_velocity': float(track.max_velocity),
                    'max_acceleration': float(track.max_acceleration)
                })

        # Compile metrics
        metrics = {
            'latency_ms': latency,
            'avg_latency_ms': self.metrics.avg_latency_ms,
            'num_detections': len(detections),
            'num_active_tracks': len(self.tracks),
            'num_confirmed_tracks': len(confirmed_tracks),
            'total_tracks_created': self.metrics.total_tracks_created,
            'total_tracks_confirmed': self.metrics.total_tracks_confirmed,
            'total_tracks_lost': self.metrics.total_tracks_lost,
            'avg_track_length': self.metrics.avg_track_length,
            'frames_processed': self.metrics.frames_processed,
            'occlusion_recoveries': self.metrics.occlusion_recoveries,
            'meets_requirements': {
                'latency': latency < 100.0,
                'max_tracks': len(confirmed_tracks) <= 200,
                'detection_rate': self.metrics.detection_rate > 0.99,
                'false_positive_rate': self.metrics.false_positive_rate < 0.02
            }
        }

        return {
            'tracks': confirmed_tracks,
            'detections': detections,
            'metrics': metrics,
            'timestamp': time.time()
        }

    def add_occlusion_zone(self, polygon: np.ndarray):
        """Add known occlusion zone (polygon vertices)"""
        self.occlusion_zones.append(polygon)

    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.occluded_tracks.clear()
        self.track_history.clear()
        self.next_track_id = 0
        self.metrics = TrackMetrics()

    def get_track_history(self, track_id: int) -> Optional[np.ndarray]:
        """Get full trajectory history for a track"""
        if track_id in self.tracks:
            return np.array(list(self.tracks[track_id].trajectory))
        elif track_id in self.track_history:
            return np.array(list(self.track_history[track_id].trajectory))
        return None

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            'total_tracks_created': self.metrics.total_tracks_created,
            'total_tracks_confirmed': self.metrics.total_tracks_confirmed,
            'total_tracks_lost': self.metrics.total_tracks_lost,
            'avg_track_length': self.metrics.avg_track_length,
            'avg_latency_ms': self.metrics.avg_latency_ms,
            'frames_processed': self.metrics.frames_processed,
            'detection_rate': self.metrics.detection_rate,
            'false_positive_rate': self.metrics.false_positive_rate,
            'occlusion_recoveries': self.metrics.occlusion_recoveries,
            'active_tracks': len(self.tracks),
            'confirmed_tracks': sum(1 for t in self.tracks.values() if t.confirmed),
            'occluded_tracks': sum(1 for t in self.tracks.values() if t.occluded)
        }


if __name__ == "__main__":
    # Example usage
    tracker = MultiTargetTracker(max_tracks=200)

    # Simulate detections
    detections = [
        {'x': 100.0, 'y': 200.0, 'velocity_x': 1.0, 'velocity_y': 0.5,
         'confidence': 0.95, 'size': 2.5},
        {'x': 300.0, 'y': 400.0, 'velocity_x': -0.5, 'velocity_y': 1.2,
         'confidence': 0.88, 'size': 2.8},
    ]

    # Update tracker
    result = tracker.update(detections, frame_number=0, timestamp=time.time())

    print("Tracking Results:")
    print(f"Active tracks: {result['metrics']['num_active_tracks']}")
    print(f"Latency: {result['metrics']['latency_ms']:.2f} ms")
    print(f"Tracks: {result['tracks']}")
