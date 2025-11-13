"""
Fusion Manager - Coordinate thermal + monochrome camera fusion
Handles pair-wise fusion, target confirmation, false positive reduction, and thermal enhancement
"""

import numpy as np
import threading
import queue
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# Import C++ fusion engine (will be compiled with pybind11)
try:
    import thermal_mono_fusion as fusion_cpp
except ImportError:
    fusion_cpp = None
    logging.warning("C++ fusion module not available, using Python fallback")


@dataclass
class CameraPair:
    """Represents a thermal-monochrome camera pair"""
    thermal_id: int
    mono_id: int
    baseline_m: float  # Physical baseline between cameras
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    registration_params: Optional[Any] = None
    last_registration_time: float = 0.0
    registration_valid: bool = False


@dataclass
class FusionConfig:
    """Configuration for fusion processing"""
    target_fps: int = 30
    registration_update_interval_s: float = 1.0  # Re-register every N seconds
    thermal_threshold: float = 0.3
    mono_threshold: float = 0.2
    confidence_threshold: float = 0.6
    max_range_km: float = 5.0
    enable_thermal_enhancement: bool = True
    enable_false_positive_reduction: bool = True
    enable_cuda: bool = True
    thermal_palette: str = "iron"
    low_light_threshold: float = 0.1  # Below this, enable thermal enhancement


@dataclass
class FusedDetection:
    """Fused detection with multi-modal confirmation"""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    thermal_signature: float
    mono_signature: float
    track_id: int
    camera_pair_id: int
    timestamp: float
    confirmed_by_thermal: bool
    confirmed_by_mono: bool
    range_estimate_m: Optional[float] = None


class FusionManager:
    """
    Manages multi-camera thermal + monochrome fusion system
    Coordinates pair-wise fusion, target confirmation, and false positive reduction
    """

    def __init__(self, config: FusionConfig):
        self.config = config
        self.camera_pairs: List[CameraPair] = []
        self.active_tracks: Dict[int, List[FusedDetection]] = defaultdict(list)
        self.next_track_id = 0
        self.processing_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        self.running = False
        self.worker_threads: List[threading.Thread] = []

        # Performance metrics
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.false_positives_rejected = 0
        self.true_positives_confirmed = 0
        self.registration_errors = []

        # Thread safety
        self.lock = threading.Lock()

        logging.info(f"FusionManager initialized with config: {config}")

    def add_camera_pair(
        self,
        thermal_id: int,
        mono_id: int,
        baseline_m: float,
        calibration_params: Optional[Dict] = None
    ) -> int:
        """Add a thermal-monochrome camera pair"""
        pair = CameraPair(
            thermal_id=thermal_id,
            mono_id=mono_id,
            baseline_m=baseline_m,
            calibration_params=calibration_params or {}
        )

        with self.lock:
            pair_id = len(self.camera_pairs)
            self.camera_pairs.append(pair)

        logging.info(f"Added camera pair {pair_id}: thermal={thermal_id}, mono={mono_id}")
        return pair_id

    def start(self, num_workers: int = 4):
        """Start fusion processing threads"""
        if self.running:
            logging.warning("FusionManager already running")
            return

        self.running = True

        # Start worker threads for parallel processing
        for i in range(num_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)

        logging.info(f"Started {num_workers} fusion worker threads")

    def stop(self):
        """Stop fusion processing"""
        self.running = False

        # Wait for workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=1.0)

        self.worker_threads.clear()
        logging.info("FusionManager stopped")

    def process_frame_pair(
        self,
        pair_id: int,
        thermal_image: np.ndarray,
        mono_image: np.ndarray,
        timestamp: float
    ) -> Optional[List[FusedDetection]]:
        """
        Process a thermal-monochrome image pair
        Returns list of fused detections
        """
        if pair_id >= len(self.camera_pairs):
            logging.error(f"Invalid camera pair ID: {pair_id}")
            return None

        try:
            # Queue for processing
            self.processing_queue.put(
                (pair_id, thermal_image, mono_image, timestamp),
                timeout=0.1
            )

            # Try to get result (non-blocking)
            try:
                result_pair_id, detections, metrics = self.result_queue.get_nowait()
                return detections
            except queue.Empty:
                return None

        except queue.Full:
            logging.warning(f"Processing queue full, dropping frame for pair {pair_id}")
            return None

    def _worker_loop(self, worker_id: int):
        """Worker thread main loop"""
        logging.info(f"Fusion worker {worker_id} started")

        while self.running:
            try:
                # Get work from queue (blocking with timeout)
                pair_id, thermal_img, mono_img, timestamp = self.processing_queue.get(
                    timeout=0.1
                )

                # Process the frame pair
                detections, metrics = self._process_pair_internal(
                    pair_id, thermal_img, mono_img, timestamp
                )

                # Put result in queue
                try:
                    self.result_queue.put(
                        (pair_id, detections, metrics),
                        timeout=0.1
                    )
                except queue.Full:
                    logging.warning(f"Result queue full, dropping result from pair {pair_id}")

                self.processing_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}", exc_info=True)

    def _process_pair_internal(
        self,
        pair_id: int,
        thermal_image: np.ndarray,
        mono_image: np.ndarray,
        timestamp: float
    ) -> Tuple[List[FusedDetection], Dict[str, Any]]:
        """Internal processing for a camera pair"""
        start_time = time.time()
        pair = self.camera_pairs[pair_id]
        metrics = {}

        # Step 1: Check if we need to update registration
        if (not pair.registration_valid or
            (timestamp - pair.last_registration_time) > self.config.registration_update_interval_s):
            pair.registration_params = self._update_registration(
                thermal_image, mono_image, pair
            )
            pair.last_registration_time = timestamp
            pair.registration_valid = pair.registration_params.is_valid if pair.registration_params else False

        if not pair.registration_valid:
            logging.warning(f"Registration failed for pair {pair_id}")
            return [], metrics

        # Step 2: Check for low-light conditions and enable thermal enhancement
        mono_brightness = np.mean(mono_image)
        enable_thermal_enhancement = (
            self.config.enable_thermal_enhancement and
            mono_brightness < self.config.low_light_threshold
        )

        if enable_thermal_enhancement:
            mono_image = self._enhance_with_thermal(
                mono_image, thermal_image, pair.registration_params
            )

        # Step 3: Multi-spectral fusion detection
        if fusion_cpp is not None:
            # Use C++ accelerated fusion
            fusion_result = fusion_cpp.process_fusion(
                thermal_image,
                mono_image,
                pair.baseline_m,
                self.config.max_range_km,
                self.config.enable_cuda
            )

            detections = self._convert_cpp_detections(
                fusion_result.detections,
                pair_id,
                timestamp
            )

            metrics['fps'] = fusion_result.fps
            metrics['registration_error_mm'] = fusion_result.registration_error_mm
            metrics['false_positives_rejected'] = fusion_result.false_positives_rejected
            metrics['target_confirmation_rate'] = fusion_result.target_confirmation_rate

        else:
            # Python fallback
            detections = self._python_fusion_detection(
                thermal_image, mono_image, pair_id, timestamp
            )
            metrics['fps'] = 1.0 / (time.time() - start_time)

        # Step 4: Cross-validate detections across modalities
        if self.config.enable_false_positive_reduction:
            detections, fp_rejected = self._cross_validate_detections(
                detections, thermal_image, mono_image
            )
            metrics['false_positives_rejected'] = fp_rejected

        # Step 5: Track management
        detections = self._update_tracks(detections, pair_id)

        # Update metrics
        processing_time = time.time() - start_time
        with self.lock:
            self.frame_count += 1
            self.total_processing_time += processing_time
            if 'registration_error_mm' in metrics:
                self.registration_errors.append(metrics['registration_error_mm'])

        return detections, metrics

    def _update_registration(
        self,
        thermal_image: np.ndarray,
        mono_image: np.ndarray,
        pair: CameraPair
    ) -> Optional[Any]:
        """Update image registration parameters"""
        if fusion_cpp is not None:
            try:
                reg_params = fusion_cpp.estimate_homography(
                    thermal_image.astype(np.float64),
                    mono_image.astype(np.float64),
                    ransac_threshold=3.0,
                    max_iterations=1000
                )
                return reg_params
            except Exception as e:
                logging.error(f"Registration failed: {e}")
                return None
        else:
            # Python fallback - simple translation-only registration
            return self._python_registration(thermal_image, mono_image)

    def _python_registration(
        self,
        thermal_image: np.ndarray,
        mono_image: np.ndarray
    ) -> Any:
        """Python fallback for image registration"""
        # Simplified registration using cross-correlation
        from scipy import signal

        # Downsample for speed
        thermal_small = thermal_image[::4, ::4]
        mono_small = mono_image[::4, ::4]

        # Cross-correlation
        correlation = signal.correlate2d(mono_small, thermal_small, mode='same')
        y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)

        # Center the shifts
        y_shift = (y_shift - thermal_small.shape[0] // 2) * 4
        x_shift = (x_shift - thermal_small.shape[1] // 2) * 4

        # Create simple registration object
        class SimpleReg:
            def __init__(self, dx, dy):
                self.homography = [1, 0, dx, 0, 1, dy, 0, 0, 1]
                self.is_valid = True
                self.rmse = 1.0
                self.num_matches = 100

        return SimpleReg(x_shift, y_shift)

    def _enhance_with_thermal(
        self,
        mono_image: np.ndarray,
        thermal_image: np.ndarray,
        registration_params: Any
    ) -> np.ndarray:
        """Enhance low-light monochrome image with thermal data"""
        if fusion_cpp is not None:
            # Warp thermal to align with mono
            warped_thermal = fusion_cpp.warp_thermal_image(
                thermal_image,
                registration_params.homography,
                mono_image.shape[1],
                mono_image.shape[0]
            )
        else:
            # Simple alignment using shifts
            warped_thermal = thermal_image

        # Blend mono and thermal
        # In low light, thermal provides more signal
        alpha = 0.7  # Weight for thermal
        enhanced = alpha * warped_thermal + (1 - alpha) * mono_image

        return enhanced

    def _convert_cpp_detections(
        self,
        cpp_detections: List[Any],
        pair_id: int,
        timestamp: float
    ) -> List[FusedDetection]:
        """Convert C++ detections to Python FusedDetection objects"""
        detections = []

        for det in cpp_detections:
            fused = FusedDetection(
                x=det.x,
                y=det.y,
                width=det.width,
                height=det.height,
                confidence=det.confidence,
                thermal_signature=det.thermal_sig,
                mono_signature=det.confidence * 0.4,  # Estimate
                track_id=-1,  # Will be assigned later
                camera_pair_id=pair_id,
                timestamp=timestamp,
                confirmed_by_thermal=det.thermal_sig > self.config.thermal_threshold,
                confirmed_by_mono=True
            )
            detections.append(fused)

        return detections

    def _python_fusion_detection(
        self,
        thermal_image: np.ndarray,
        mono_image: np.ndarray,
        pair_id: int,
        timestamp: float
    ) -> List[FusedDetection]:
        """Python fallback for fusion detection"""
        detections = []

        # Simple threshold-based detection
        thermal_thresh = thermal_image > self.config.thermal_threshold
        mono_thresh = mono_image > self.config.mono_threshold

        # Find connected components (simplified)
        from scipy import ndimage

        thermal_labels, thermal_num = ndimage.label(thermal_thresh)
        mono_labels, mono_num = ndimage.label(mono_thresh)

        # For each thermal blob, check if there's a corresponding mono blob
        for i in range(1, thermal_num + 1):
            coords = np.where(thermal_labels == i)
            if len(coords[0]) == 0:
                continue

            y_center = np.mean(coords[0])
            x_center = np.mean(coords[1])
            height = np.ptp(coords[0])
            width = np.ptp(coords[1])

            # Check thermal signature
            thermal_sig = np.mean(thermal_image[coords])

            # Check if mono confirms
            mono_sig = np.mean(mono_image[coords])
            confirmed_by_mono = mono_sig > self.config.mono_threshold

            confidence = 0.6 if confirmed_by_mono else 0.3

            det = FusedDetection(
                x=x_center,
                y=y_center,
                width=width,
                height=height,
                confidence=confidence,
                thermal_signature=thermal_sig,
                mono_signature=mono_sig,
                track_id=-1,
                camera_pair_id=pair_id,
                timestamp=timestamp,
                confirmed_by_thermal=True,
                confirmed_by_mono=confirmed_by_mono
            )
            detections.append(det)

        return detections

    def _cross_validate_detections(
        self,
        detections: List[FusedDetection],
        thermal_image: np.ndarray,
        mono_image: np.ndarray
    ) -> Tuple[List[FusedDetection], int]:
        """Cross-validate detections to reduce false positives"""
        validated = []
        false_positives = 0

        for det in detections:
            # Extract region around detection
            x1 = int(max(0, det.x - det.width // 2))
            x2 = int(min(thermal_image.shape[1], det.x + det.width // 2))
            y1 = int(max(0, det.y - det.height // 2))
            y2 = int(min(thermal_image.shape[0], det.y + det.height // 2))

            if x2 <= x1 or y2 <= y1:
                false_positives += 1
                continue

            thermal_region = thermal_image[y1:y2, x1:x2]
            mono_region = mono_image[y1:y2, x1:x2]

            # Recompute signatures
            thermal_sig = np.mean(thermal_region)
            mono_sig = np.mean(mono_region)

            # Require confirmation from at least one modality strongly
            confirmed_thermal = thermal_sig > self.config.thermal_threshold
            confirmed_mono = mono_sig > self.config.mono_threshold

            if confirmed_thermal or confirmed_mono:
                det.thermal_signature = thermal_sig
                det.mono_signature = mono_sig
                det.confirmed_by_thermal = confirmed_thermal
                det.confirmed_by_mono = confirmed_mono

                # Update confidence based on cross-validation
                if confirmed_thermal and confirmed_mono:
                    det.confidence = min(1.0, det.confidence * 1.5)  # Both confirm - boost
                elif confirmed_thermal or confirmed_mono:
                    det.confidence = det.confidence  # One confirms - keep

                if det.confidence >= self.config.confidence_threshold:
                    validated.append(det)
                else:
                    false_positives += 1
            else:
                false_positives += 1

        return validated, false_positives

    def _update_tracks(
        self,
        detections: List[FusedDetection],
        pair_id: int
    ) -> List[FusedDetection]:
        """Update tracking IDs for detections"""
        with self.lock:
            # Get existing tracks for this camera pair
            existing_tracks = [
                track_history[-1] for track_id, track_history in self.active_tracks.items()
                if track_history and track_history[-1].camera_pair_id == pair_id
            ]

            # Match detections to existing tracks
            for det in detections:
                best_match_id = -1
                best_match_score = 0.0

                for prev_det in existing_tracks:
                    # Compute matching score
                    dx = det.x - prev_det.x
                    dy = det.y - prev_det.y
                    dist = np.sqrt(dx * dx + dy * dy)

                    if dist < 50:  # Max tracking distance
                        spatial_score = 1.0 - (dist / 50.0)
                        sig_diff = abs(det.thermal_signature - prev_det.thermal_signature)
                        sig_score = np.exp(-sig_diff * 5.0)

                        match_score = 0.7 * spatial_score + 0.3 * sig_score

                        if match_score > best_match_score:
                            best_match_score = match_score
                            best_match_id = prev_det.track_id

                # Assign track ID
                if best_match_id >= 0 and best_match_score > 0.5:
                    det.track_id = best_match_id
                else:
                    det.track_id = self.next_track_id
                    self.next_track_id += 1

                # Update track history
                self.active_tracks[det.track_id].append(det)

                # Keep only recent history
                if len(self.active_tracks[det.track_id]) > 30:
                    self.active_tracks[det.track_id].pop(0)

        return detections

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get fusion performance metrics"""
        with self.lock:
            if self.frame_count == 0:
                return {}

            avg_fps = self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0
            avg_reg_error = np.mean(self.registration_errors) if self.registration_errors else 0

            total_detections = self.true_positives_confirmed + self.false_positives_rejected
            confirmation_rate = (
                self.true_positives_confirmed / total_detections
                if total_detections > 0 else 0
            )

            fp_reduction = (
                self.false_positives_rejected / total_detections
                if total_detections > 0 else 0
            )

            return {
                'avg_fps': avg_fps,
                'frame_count': self.frame_count,
                'avg_registration_error_mm': avg_reg_error,
                'target_confirmation_rate': confirmation_rate,
                'false_positive_reduction_rate': fp_reduction,
                'active_tracks': len(self.active_tracks),
                'queue_size': self.processing_queue.qsize()
            }

    def apply_thermal_palette(
        self,
        thermal_image: np.ndarray,
        palette: str = "iron"
    ) -> np.ndarray:
        """Apply thermal color palette to image"""
        if fusion_cpp is not None:
            return fusion_cpp.apply_thermal_palette(
                thermal_image.astype(np.float64),
                palette
            )
        else:
            # Python fallback - simple grayscale
            normalized = (thermal_image - thermal_image.min()) / (thermal_image.max() - thermal_image.min() + 1e-8)
            return (normalized * 255).astype(np.uint8)
