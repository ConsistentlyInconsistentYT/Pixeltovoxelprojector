#!/usr/bin/env python3
"""
Metrics Analysis and Visualization for Fusion System

Provides detailed analysis of:
- Registration accuracy over time
- Detection performance metrics
- False positive reduction effectiveness
- FPS and latency analysis
- Occlusion handling statistics
"""

import numpy as np
from typing import List, Dict, Tuple
import json
import time
from dataclasses import dataclass, asdict


@dataclass
class FusionMetrics:
    """Complete metrics snapshot"""
    timestamp: float
    frame_id: int

    # Performance
    fps: float
    processing_time_ms: float
    registration_time_ms: float
    detection_time_ms: float

    # Registration
    registration_error_mm: float
    registration_error_pixels: float
    num_feature_matches: int
    registration_valid: bool

    # Detection
    num_detections: int
    num_confirmed: int
    num_rejected: int
    confirmation_rate: float
    false_positive_rate: float

    # Tracking
    num_active_tracks: int
    num_occluded_tracks: int
    num_recovered_tracks: int

    # Cross-validation
    cross_validation_rate: float
    thermal_only_rate: float
    mono_only_rate: float


class MetricsCollector:
    """Collects and analyzes fusion system metrics"""

    def __init__(self):
        self.metrics_history: List[FusionMetrics] = []
        self.start_time = time.time()

    def add_metrics(self, metrics: FusionMetrics):
        """Add metrics snapshot"""
        self.metrics_history.append(metrics)

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.metrics_history:
            return {}

        # Overall statistics
        total_frames = len(self.metrics_history)
        total_time = time.time() - self.start_time

        # FPS statistics
        fps_values = [m.fps for m in self.metrics_history]
        avg_fps = np.mean(fps_values)
        std_fps = np.std(fps_values)
        min_fps = np.min(fps_values)
        max_fps = np.max(fps_values)

        # Registration statistics
        reg_errors = [m.registration_error_mm for m in self.metrics_history
                     if m.registration_valid]
        avg_reg_error = np.mean(reg_errors) if reg_errors else 0
        max_reg_error = np.max(reg_errors) if reg_errors else 0
        registration_success_rate = len(reg_errors) / total_frames if total_frames > 0 else 0

        # Detection statistics
        total_detections = sum(m.num_detections for m in self.metrics_history)
        total_confirmed = sum(m.num_confirmed for m in self.metrics_history)
        total_rejected = sum(m.num_rejected for m in self.metrics_history)

        confirmation_rates = [m.confirmation_rate for m in self.metrics_history
                            if m.num_detections > 0]
        avg_confirmation = np.mean(confirmation_rates) if confirmation_rates else 0

        fp_rates = [m.false_positive_rate for m in self.metrics_history
                   if m.num_detections > 0]
        avg_fp_rate = np.mean(fp_rates) if fp_rates else 0

        # Tracking statistics
        max_tracks = max((m.num_active_tracks for m in self.metrics_history), default=0)
        total_occlusions = sum(m.num_occluded_tracks for m in self.metrics_history)
        total_recovered = sum(m.num_recovered_tracks for m in self.metrics_history)
        recovery_rate = total_recovered / total_occlusions if total_occlusions > 0 else 0

        # Cross-validation statistics
        cross_val_rates = [m.cross_validation_rate for m in self.metrics_history]
        avg_cross_val = np.mean(cross_val_rates)

        return {
            'overall': {
                'total_frames': total_frames,
                'total_time_s': total_time,
                'avg_fps': avg_fps,
                'std_fps': std_fps,
                'min_fps': min_fps,
                'max_fps': max_fps,
            },
            'registration': {
                'avg_error_mm': avg_reg_error,
                'max_error_mm': max_reg_error,
                'success_rate': registration_success_rate,
                'meets_spec': avg_reg_error < 5.0,  # <5mm @ 5km
            },
            'detection': {
                'total_detections': total_detections,
                'total_confirmed': total_confirmed,
                'total_rejected': total_rejected,
                'avg_confirmation_rate': avg_confirmation,
                'avg_false_positive_rate': avg_fp_rate,
                'meets_spec': avg_confirmation >= 0.99,  # 99%+ confirmation
            },
            'tracking': {
                'max_concurrent_tracks': max_tracks,
                'total_occlusions': total_occlusions,
                'total_recovered': total_recovered,
                'recovery_rate': recovery_rate,
            },
            'cross_validation': {
                'avg_rate': avg_cross_val,
            },
            'requirements': {
                'fps': {
                    'target': 30,
                    'actual': avg_fps,
                    'met': avg_fps >= 30
                },
                'registration': {
                    'target_mm': 5.0,
                    'actual_mm': avg_reg_error,
                    'met': avg_reg_error < 5.0
                },
                'confirmation': {
                    'target': 0.99,
                    'actual': avg_confirmation,
                    'met': avg_confirmation >= 0.99
                },
                'false_positive_reduction': {
                    'target': 0.90,
                    'actual': 1.0 - avg_fp_rate,
                    'met': (1.0 - avg_fp_rate) >= 0.90
                }
            }
        }

    def analyze_performance_over_time(self) -> Dict:
        """Analyze performance trends over time"""
        if len(self.metrics_history) < 10:
            return {'error': 'Insufficient data'}

        # Split into time windows
        window_size = min(30, len(self.metrics_history) // 5)
        num_windows = len(self.metrics_history) // window_size

        windows = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window_metrics = self.metrics_history[start:end]

            window_fps = np.mean([m.fps for m in window_metrics])
            window_reg_error = np.mean([m.registration_error_mm for m in window_metrics
                                       if m.registration_valid])

            windows.append({
                'window_id': i,
                'frame_range': (start, end),
                'avg_fps': window_fps,
                'avg_reg_error_mm': window_reg_error,
            })

        # Detect performance degradation
        fps_trend = [w['avg_fps'] for w in windows]
        fps_decreasing = fps_trend[0] > fps_trend[-1] + 5  # >5 FPS drop

        return {
            'windows': windows,
            'performance_degradation_detected': fps_decreasing,
            'fps_trend': 'decreasing' if fps_decreasing else 'stable',
        }

    def get_percentile_metrics(self, percentiles: List[int] = [50, 90, 95, 99]) -> Dict:
        """Get percentile metrics for key parameters"""
        if not self.metrics_history:
            return {}

        fps_values = [m.fps for m in self.metrics_history]
        reg_errors = [m.registration_error_mm for m in self.metrics_history
                     if m.registration_valid]
        proc_times = [m.processing_time_ms for m in self.metrics_history]

        result = {}
        for p in percentiles:
            result[f'p{p}'] = {
                'fps': np.percentile(fps_values, p),
                'registration_error_mm': np.percentile(reg_errors, p) if reg_errors else 0,
                'processing_time_ms': np.percentile(proc_times, p),
            }

        return result

    def export_to_json(self, filename: str):
        """Export metrics to JSON file"""
        data = {
            'collection_start': self.start_time,
            'collection_end': time.time(),
            'num_samples': len(self.metrics_history),
            'metrics': [asdict(m) for m in self.metrics_history],
            'summary': self.get_summary(),
            'percentiles': self.get_percentile_metrics(),
            'trends': self.analyze_performance_over_time(),
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Metrics exported to {filename}")

    def print_report(self):
        """Print detailed metrics report"""
        summary = self.get_summary()
        percentiles = self.get_percentile_metrics()
        trends = self.analyze_performance_over_time()

        print("\n" + "=" * 80)
        print("FUSION SYSTEM METRICS REPORT")
        print("=" * 80)

        # Overall performance
        print("\n--- Overall Performance ---")
        overall = summary['overall']
        print(f"Total frames processed: {overall['total_frames']}")
        print(f"Total time: {overall['total_time_s']:.2f} s")
        print(f"Average FPS: {overall['avg_fps']:.2f} ± {overall['std_fps']:.2f}")
        print(f"FPS range: [{overall['min_fps']:.2f}, {overall['max_fps']:.2f}]")

        # Registration
        print("\n--- Image Registration ---")
        reg = summary['registration']
        print(f"Average error: {reg['avg_error_mm']:.2f} mm at 5km range")
        print(f"Maximum error: {reg['max_error_mm']:.2f} mm")
        print(f"Success rate: {reg['success_rate']*100:.1f}%")
        print(f"Meets specification (<5mm): {reg['meets_spec']}")

        # Detection
        print("\n--- Target Detection ---")
        det = summary['detection']
        print(f"Total detections: {det['total_detections']}")
        print(f"Confirmed targets: {det['total_confirmed']}")
        print(f"Rejected false positives: {det['total_rejected']}")
        print(f"Confirmation rate: {det['avg_confirmation_rate']*100:.1f}%")
        print(f"False positive rate: {det['avg_false_positive_rate']*100:.1f}%")
        print(f"False positive reduction: {(1-det['avg_false_positive_rate'])*100:.1f}%")
        print(f"Meets specification (99%+ confirm): {det['meets_spec']}")

        # Tracking
        print("\n--- Multi-Object Tracking ---")
        track = summary['tracking']
        print(f"Max concurrent tracks: {track['max_concurrent_tracks']}")
        print(f"Total occlusions: {track['total_occlusions']}")
        print(f"Recovered from occlusion: {track['total_recovered']}")
        print(f"Recovery rate: {track['recovery_rate']*100:.1f}%")

        # Cross-validation
        print("\n--- Cross-Modal Validation ---")
        cv = summary['cross_validation']
        print(f"Cross-validation rate: {cv['avg_rate']*100:.1f}%")

        # Percentiles
        print("\n--- Latency Percentiles ---")
        print(f"{'Metric':<25} {'P50':>10} {'P90':>10} {'P95':>10} {'P99':>10}")
        print("-" * 70)
        print(f"{'FPS':<25} {percentiles['p50']['fps']:>10.1f} "
              f"{percentiles['p90']['fps']:>10.1f} {percentiles['p95']['fps']:>10.1f} "
              f"{percentiles['p99']['fps']:>10.1f}")
        print(f"{'Registration Error (mm)':<25} {percentiles['p50']['registration_error_mm']:>10.2f} "
              f"{percentiles['p90']['registration_error_mm']:>10.2f} "
              f"{percentiles['p95']['registration_error_mm']:>10.2f} "
              f"{percentiles['p99']['registration_error_mm']:>10.2f}")
        print(f"{'Processing Time (ms)':<25} {percentiles['p50']['processing_time_ms']:>10.2f} "
              f"{percentiles['p90']['processing_time_ms']:>10.2f} "
              f"{percentiles['p95']['processing_time_ms']:>10.2f} "
              f"{percentiles['p99']['processing_time_ms']:>10.2f}")

        # Requirements validation
        print("\n--- Requirements Validation ---")
        reqs = summary['requirements']
        print(f"{'Requirement':<30} {'Target':>15} {'Actual':>15} {'Status':>10}")
        print("-" * 75)

        status_icon = lambda met: "✓ PASS" if met else "✗ FAIL"

        print(f"{'Real-time FPS':<30} {reqs['fps']['target']:>15} "
              f"{reqs['fps']['actual']:>15.1f} {status_icon(reqs['fps']['met']):>10}")

        print(f"{'Registration accuracy (mm)':<30} {'<' + str(reqs['registration']['target_mm']):>15} "
              f"{reqs['registration']['actual_mm']:>15.2f} "
              f"{status_icon(reqs['registration']['met']):>10}")

        print(f"{'Target confirmation rate':<30} {reqs['confirmation']['target']*100:>14.0f}% "
              f"{reqs['confirmation']['actual']*100:>14.1f}% "
              f"{status_icon(reqs['confirmation']['met']):>10}")

        print(f"{'FP reduction rate':<30} {reqs['false_positive_reduction']['target']*100:>14.0f}% "
              f"{reqs['false_positive_reduction']['actual']*100:>14.1f}% "
              f"{status_icon(reqs['false_positive_reduction']['met']):>10}")

        # Performance trends
        if 'error' not in trends:
            print("\n--- Performance Trends ---")
            print(f"FPS trend: {trends['fps_trend']}")
            if trends['performance_degradation_detected']:
                print("⚠ Warning: Performance degradation detected over time")

        print("\n" + "=" * 80)


def generate_test_metrics(num_frames: int = 100) -> MetricsCollector:
    """Generate synthetic metrics for testing"""
    collector = MetricsCollector()

    for frame_id in range(num_frames):
        # Simulate realistic metrics
        base_fps = 32.0
        fps_noise = np.random.normal(0, 2)
        fps = max(20, base_fps + fps_noise)

        reg_error = np.random.normal(2.8, 0.8)  # Average ~2.8mm
        reg_error = max(0.5, min(8.0, reg_error))  # Clamp

        num_dets = np.random.randint(3, 10)
        num_confirmed = int(num_dets * np.random.uniform(0.95, 1.0))
        num_rejected = num_dets - num_confirmed

        metrics = FusionMetrics(
            timestamp=time.time(),
            frame_id=frame_id,
            fps=fps,
            processing_time_ms=1000.0 / fps,
            registration_time_ms=np.random.uniform(10, 20),
            detection_time_ms=np.random.uniform(15, 25),
            registration_error_mm=reg_error,
            registration_error_pixels=reg_error * 0.5,
            num_feature_matches=np.random.randint(50, 200),
            registration_valid=True,
            num_detections=num_dets,
            num_confirmed=num_confirmed,
            num_rejected=num_rejected,
            confirmation_rate=num_confirmed / num_dets if num_dets > 0 else 0,
            false_positive_rate=num_rejected / num_dets if num_dets > 0 else 0,
            num_active_tracks=np.random.randint(5, 15),
            num_occluded_tracks=np.random.randint(0, 3),
            num_recovered_tracks=np.random.randint(0, 2),
            cross_validation_rate=np.random.uniform(0.65, 0.85),
            thermal_only_rate=np.random.uniform(0.10, 0.20),
            mono_only_rate=np.random.uniform(0.05, 0.15),
        )

        collector.add_metrics(metrics)

    return collector


def main():
    """Demo metrics analysis"""
    print("Generating test metrics...")
    collector = generate_test_metrics(num_frames=300)

    print("Analyzing metrics...")
    collector.print_report()

    print("\nExporting to JSON...")
    collector.export_to_json('fusion_metrics.json')

    print("\nDone!")


if __name__ == '__main__':
    main()
