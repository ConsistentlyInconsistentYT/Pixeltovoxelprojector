"""
Adaptive Performance Manager

Dynamically adjusts system parameters based on real-time performance metrics
to maintain target FPS and latency while maximizing quality.

Features:
- Adaptive resolution scaling
- Dynamic resource allocation
- Load-based quality adjustment
- Automatic parameter tuning
- Predictive performance management
"""

import time
import threading
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class PerformanceMode(Enum):
    """Performance operation modes"""
    MAX_QUALITY = "max_quality"  # Prioritize quality over FPS
    BALANCED = "balanced"  # Balance quality and performance
    MAX_PERFORMANCE = "max_performance"  # Prioritize FPS over quality
    LATENCY_CRITICAL = "latency_critical"  # Minimize latency above all
    POWER_SAVE = "power_save"  # Minimize power consumption


@dataclass
class PerformanceTarget:
    """Performance targets and thresholds"""
    target_fps: float = 30.0
    min_fps: float = 25.0
    max_latency_ms: float = 50.0
    critical_latency_ms: float = 80.0
    target_gpu_util: float = 90.0
    max_gpu_util: float = 98.0


@dataclass
class AdaptiveParameters:
    """Current adaptive parameters"""
    resolution_scale: float = 1.0
    num_streams: int = 10
    batch_size: int = 1
    detection_threshold: float = 0.15
    max_objects: int = 200
    enable_fusion: bool = True
    lod_distance: float = 5000.0
    frame_skip_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'resolution_scale': self.resolution_scale,
            'num_streams': self.num_streams,
            'batch_size': self.batch_size,
            'detection_threshold': self.detection_threshold,
            'max_objects': self.max_objects,
            'enable_fusion': self.enable_fusion,
            'lod_distance': self.lod_distance,
            'frame_skip_rate': self.frame_skip_rate
        }


class PerformancePredictor:
    """
    Predict future performance based on historical data
    Uses simple linear extrapolation and moving averages
    """

    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self.fps_history = deque(maxlen=history_size)
        self.latency_history = deque(maxlen=history_size)
        self.gpu_util_history = deque(maxlen=history_size)

    def update(self, fps: float, latency_ms: float, gpu_util: float):
        """Update with new measurements"""
        self.fps_history.append(fps)
        self.latency_history.append(latency_ms)
        self.gpu_util_history.append(gpu_util)

    def predict_fps(self, horizon_frames: int = 10) -> float:
        """Predict FPS in the near future"""
        if len(self.fps_history) < 5:
            return self.fps_history[-1] if self.fps_history else 30.0

        # Simple moving average with trend
        recent = list(self.fps_history)[-10:]
        avg = np.mean(recent)

        # Trend (first-order derivative)
        if len(recent) >= 2:
            trend = (recent[-1] - recent[0]) / len(recent)
            prediction = avg + trend * horizon_frames
        else:
            prediction = avg

        return float(prediction)

    def predict_latency(self, horizon_frames: int = 10) -> float:
        """Predict latency in the near future"""
        if len(self.latency_history) < 5:
            return self.latency_history[-1] if self.latency_history else 30.0

        recent = list(self.latency_history)[-10:]
        avg = np.mean(recent)

        if len(recent) >= 2:
            trend = (recent[-1] - recent[0]) / len(recent)
            prediction = avg + trend * horizon_frames
        else:
            prediction = avg

        return float(prediction)

    def is_trending_down(self, metric: str = 'fps') -> bool:
        """Check if metric is trending downward"""
        history = {
            'fps': self.fps_history,
            'latency': self.latency_history,
            'gpu_util': self.gpu_util_history
        }.get(metric, self.fps_history)

        if len(history) < 10:
            return False

        recent = list(history)[-10:]
        first_half = np.mean(recent[:5])
        second_half = np.mean(recent[5:])

        # For FPS, trending down means second half < first half
        if metric == 'fps':
            return second_half < first_half * 0.95
        # For latency/gpu_util, trending down means decreasing (good)
        else:
            return second_half < first_half * 0.95


class AdaptivePerformanceManager:
    """
    Main adaptive performance manager
    Monitors performance and adjusts parameters in real-time
    """

    def __init__(self,
                 mode: PerformanceMode = PerformanceMode.BALANCED,
                 targets: Optional[PerformanceTarget] = None,
                 update_interval_s: float = 1.0):
        """
        Initialize adaptive performance manager

        Args:
            mode: Performance mode (quality vs speed tradeoff)
            targets: Performance targets
            update_interval_s: How often to update parameters
        """
        self.mode = mode
        self.targets = targets or PerformanceTarget()
        self.update_interval = update_interval_s

        # Current parameters
        self.params = AdaptiveParameters()

        # Performance predictor
        self.predictor = PerformancePredictor()

        # Base parameters (initial values)
        self.base_resolution = (7680, 4320)
        self.min_resolution_scale = 0.5
        self.max_resolution_scale = 1.0

        self.min_streams = 4
        self.max_streams = 16

        # Adjustment rates (how aggressively to change parameters)
        self.aggressive_adjustment = 0.15
        self.normal_adjustment = 0.05
        self.conservative_adjustment = 0.02

        # State
        self.running = False
        self.adjustment_thread = None
        self.lock = threading.Lock()

        # Statistics
        self.adjustments_made = 0
        self.quality_reductions = 0
        self.quality_increases = 0

        # Callbacks for parameter changes
        self.parameter_callbacks: List[Callable[[AdaptiveParameters], None]] = []

        logger.info(f"Adaptive Performance Manager initialized: mode={mode.value}")

    def start(self):
        """Start adaptive adjustment thread"""
        if self.running:
            logger.warning("Adaptive manager already running")
            return

        self.running = True
        self.adjustment_thread = threading.Thread(
            target=self._adjustment_loop,
            daemon=True
        )
        self.adjustment_thread.start()
        logger.info("Adaptive performance manager started")

    def stop(self):
        """Stop adaptive adjustment"""
        self.running = False
        if self.adjustment_thread:
            self.adjustment_thread.join(timeout=5.0)
        logger.info("Adaptive performance manager stopped")

    def update_metrics(self, fps: float, latency_ms: float, gpu_utilization: float):
        """
        Update with current performance metrics

        Args:
            fps: Current frames per second
            latency_ms: Current pipeline latency in milliseconds
            gpu_utilization: GPU utilization percentage (0-100)
        """
        with self.lock:
            self.predictor.update(fps, latency_ms, gpu_utilization)

    def _adjustment_loop(self):
        """Main adjustment loop running in background thread"""
        while self.running:
            try:
                # Get predicted performance
                predicted_fps = self.predictor.predict_fps()
                predicted_latency = self.predictor.predict_latency()

                # Determine if adjustment needed
                needs_adjustment, reason = self._check_adjustment_needed(
                    predicted_fps, predicted_latency
                )

                if needs_adjustment:
                    logger.info(f"Adjustment needed: {reason}")
                    self._adjust_parameters(predicted_fps, predicted_latency, reason)

                # Sleep for interval
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in adjustment loop: {e}")
                time.sleep(self.update_interval)

    def _check_adjustment_needed(self, fps: float, latency_ms: float) -> Tuple[bool, str]:
        """Check if parameter adjustment is needed"""

        # Critical: FPS too low
        if fps < self.targets.min_fps:
            return True, f"FPS too low: {fps:.1f} < {self.targets.min_fps}"

        # Critical: Latency too high
        if latency_ms > self.targets.critical_latency_ms:
            return True, f"Latency too high: {latency_ms:.1f} > {self.targets.critical_latency_ms}"

        # Warning: FPS below target
        if fps < self.targets.target_fps * 0.95:
            return True, f"FPS below target: {fps:.1f} < {self.targets.target_fps}"

        # Warning: Latency above target
        if latency_ms > self.targets.max_latency_ms:
            return True, f"Latency above target: {latency_ms:.1f} > {self.targets.max_latency_ms}"

        # Opportunity: Can increase quality
        if fps > self.targets.target_fps * 1.15 and latency_ms < self.targets.max_latency_ms * 0.8:
            if self.params.resolution_scale < self.max_resolution_scale:
                return True, f"Can increase quality: FPS={fps:.1f}, latency={latency_ms:.1f}ms"

        # Predictive: Trending down
        if self.predictor.is_trending_down('fps'):
            return True, "FPS trending downward"

        return False, ""

    def _adjust_parameters(self, fps: float, latency_ms: float, reason: str):
        """Adjust parameters based on current performance"""
        with self.lock:
            old_params = AdaptiveParameters(**self.params.__dict__)

            # Determine adjustment magnitude based on severity
            if fps < self.targets.min_fps or latency_ms > self.targets.critical_latency_ms:
                adjustment_rate = self.aggressive_adjustment
            elif fps < self.targets.target_fps * 0.9:
                adjustment_rate = self.normal_adjustment
            else:
                adjustment_rate = self.conservative_adjustment

            # Adjust based on mode and performance
            if self.mode == PerformanceMode.MAX_PERFORMANCE:
                self._adjust_for_max_performance(fps, latency_ms, adjustment_rate)
            elif self.mode == PerformanceMode.MAX_QUALITY:
                self._adjust_for_max_quality(fps, latency_ms, adjustment_rate)
            elif self.mode == PerformanceMode.LATENCY_CRITICAL:
                self._adjust_for_latency(fps, latency_ms, adjustment_rate)
            else:  # BALANCED
                self._adjust_balanced(fps, latency_ms, adjustment_rate)

            # Track adjustments
            self.adjustments_made += 1

            if self.params.resolution_scale < old_params.resolution_scale:
                self.quality_reductions += 1
            elif self.params.resolution_scale > old_params.resolution_scale:
                self.quality_increases += 1

            # Notify callbacks
            self._notify_parameter_change()

            logger.info(f"Parameters adjusted: {old_params.to_dict()} -> {self.params.to_dict()}")

    def _adjust_for_max_performance(self, fps: float, latency_ms: float, rate: float):
        """Adjust parameters to maximize FPS"""

        # Reduce quality if below target
        if fps < self.targets.target_fps:
            # Reduce resolution
            self.params.resolution_scale = max(
                self.min_resolution_scale,
                self.params.resolution_scale - rate
            )

            # Reduce max objects
            self.params.max_objects = max(50, int(self.params.max_objects * 0.9))

            # Enable frame skipping if critical
            if fps < self.targets.min_fps:
                self.params.frame_skip_rate = min(0.3, self.params.frame_skip_rate + 0.1)

        # Increase streams if GPU not saturated
        elif fps > self.targets.target_fps and self.params.num_streams < self.max_streams:
            self.params.num_streams = min(self.max_streams, self.params.num_streams + 1)

    def _adjust_for_max_quality(self, fps: float, latency_ms: float, rate: float):
        """Adjust parameters to maximize quality while maintaining minimum FPS"""

        # Only reduce quality if below minimum FPS
        if fps < self.targets.min_fps:
            self.params.resolution_scale = max(
                self.min_resolution_scale,
                self.params.resolution_scale - rate * 0.5  # Slower adjustment
            )
        # Increase quality if well above target
        elif fps > self.targets.target_fps * 1.2:
            self.params.resolution_scale = min(
                self.max_resolution_scale,
                self.params.resolution_scale + rate * 0.5
            )

            # Increase max objects
            self.params.max_objects = min(250, int(self.params.max_objects * 1.1))

    def _adjust_for_latency(self, fps: float, latency_ms: float, rate: float):
        """Adjust parameters to minimize latency"""

        # Aggressively reduce latency
        if latency_ms > self.targets.max_latency_ms:
            # Reduce batch size
            self.params.batch_size = max(1, self.params.batch_size - 1)

            # Reduce resolution
            self.params.resolution_scale = max(
                self.min_resolution_scale,
                self.params.resolution_scale - rate
            )

            # Increase streams for parallelism
            self.params.num_streams = min(
                self.max_streams,
                self.params.num_streams + 1
            )

        # Increase quality if latency well below target
        elif latency_ms < self.targets.max_latency_ms * 0.7:
            self.params.resolution_scale = min(
                self.max_resolution_scale,
                self.params.resolution_scale + rate * 0.3
            )

    def _adjust_balanced(self, fps: float, latency_ms: float, rate: float):
        """Balanced adjustment considering both FPS and latency"""

        # Performance score (higher is better)
        perf_score = (fps / self.targets.target_fps) * \
                    (self.targets.max_latency_ms / max(latency_ms, 1.0))

        if perf_score < 0.8:
            # Reduce quality
            self.params.resolution_scale = max(
                self.min_resolution_scale,
                self.params.resolution_scale - rate
            )

            # Adjust streams based on GPU utilization
            # (would need GPU util as parameter - simplified here)
            if latency_ms > self.targets.max_latency_ms:
                self.params.num_streams = min(
                    self.max_streams,
                    self.params.num_streams + 1
                )

        elif perf_score > 1.2:
            # Increase quality
            self.params.resolution_scale = min(
                self.max_resolution_scale,
                self.params.resolution_scale + rate * 0.5
            )

            self.params.max_objects = min(250, int(self.params.max_objects * 1.05))

    def _notify_parameter_change(self):
        """Notify all registered callbacks of parameter changes"""
        for callback in self.parameter_callbacks:
            try:
                callback(self.params)
            except Exception as e:
                logger.error(f"Error in parameter callback: {e}")

    def register_callback(self, callback: Callable[[AdaptiveParameters], None]):
        """Register a callback for parameter changes"""
        self.parameter_callbacks.append(callback)
        logger.info(f"Registered parameter callback: {callback.__name__}")

    def get_current_parameters(self) -> AdaptiveParameters:
        """Get current adaptive parameters (thread-safe)"""
        with self.lock:
            return AdaptiveParameters(**self.params.__dict__)

    def get_current_resolution(self) -> Tuple[int, int]:
        """Get current resolution based on scale"""
        with self.lock:
            scale = self.params.resolution_scale
            return (
                int(self.base_resolution[0] * scale),
                int(self.base_resolution[1] * scale)
            )

    def set_mode(self, mode: PerformanceMode):
        """Change performance mode"""
        with self.lock:
            self.mode = mode
            logger.info(f"Performance mode changed to: {mode.value}")

    def get_statistics(self) -> Dict:
        """Get adjustment statistics"""
        with self.lock:
            return {
                'mode': self.mode.value,
                'adjustments_made': self.adjustments_made,
                'quality_reductions': self.quality_reductions,
                'quality_increases': self.quality_increases,
                'current_resolution_scale': self.params.resolution_scale,
                'current_streams': self.params.num_streams,
                'current_batch_size': self.params.batch_size,
                'current_max_objects': self.params.max_objects
            }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create adaptive manager
    targets = PerformanceTarget(
        target_fps=30.0,
        min_fps=25.0,
        max_latency_ms=50.0
    )

    manager = AdaptivePerformanceManager(
        mode=PerformanceMode.BALANCED,
        targets=targets,
        update_interval_s=0.5
    )

    # Register callback
    def on_param_change(params: AdaptiveParameters):
        print(f"Parameters changed: resolution_scale={params.resolution_scale:.2f}, "
              f"streams={params.num_streams}")

    manager.register_callback(on_param_change)

    # Start adaptive management
    manager.start()

    # Simulate performance updates
    try:
        for i in range(60):
            # Simulate varying performance
            fps = 30.0 + np.sin(i * 0.1) * 8.0  # Varies between 22-38 FPS
            latency = 45.0 + np.cos(i * 0.1) * 15.0  # Varies between 30-60ms
            gpu_util = 85.0 + np.random.rand() * 10.0

            manager.update_metrics(fps, latency, gpu_util)

            print(f"Frame {i}: FPS={fps:.1f}, Latency={latency:.1f}ms, GPU={gpu_util:.1f}%")

            # Get current resolution
            resolution = manager.get_current_resolution()
            print(f"  Current resolution: {resolution[0]}x{resolution[1]}")

            time.sleep(0.5)

    finally:
        manager.stop()

        # Print statistics
        stats = manager.get_statistics()
        print("\n=== Adaptive Manager Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
