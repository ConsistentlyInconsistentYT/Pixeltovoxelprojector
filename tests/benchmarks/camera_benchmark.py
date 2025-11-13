#!/usr/bin/env python3
"""
Camera Pipeline Benchmarks for PixelToVoxelProjector

Benchmarks for:
- 8K video decode performance
- Motion extraction throughput
- Multi-camera synchronization accuracy
- Frame drop analysis
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading
import queue

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Camera benchmarks will be skipped.")


@dataclass
class CameraMetrics:
    """Metrics for camera pipeline performance"""
    camera_id: int
    decode_fps: float
    motion_fps: float
    frames_processed: int
    frames_dropped: int
    avg_decode_latency_ms: float
    avg_motion_latency_ms: float
    sync_accuracy_ms: float
    timestamp: str


@dataclass
class FrameTimestamp:
    """Frame timing information"""
    camera_id: int
    frame_idx: int
    capture_time: float
    decode_time: float
    process_time: float


class VideoDecoder:
    """Efficient video decoder with performance tracking"""

    def __init__(self, video_path: str, camera_id: int = 0):
        self.video_path = video_path
        self.camera_id = camera_id
        self.cap = None
        self.decode_times = []
        self.frame_count = 0

    def open(self) -> bool:
        """Open video file"""
        if not HAS_CV2:
            return False

        self.cap = cv2.VideoCapture(self.video_path)
        return self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read and decode a frame, return timing"""
        if not self.cap:
            return False, None, 0.0

        start = time.time()
        ret, frame = self.cap.read()
        decode_time = (time.time() - start) * 1000  # ms

        if ret:
            self.decode_times.append(decode_time)
            self.frame_count += 1

        return ret, frame, decode_time

    def get_metrics(self) -> Dict:
        """Get decoder metrics"""
        if not self.decode_times:
            return {
                'avg_decode_ms': 0,
                'min_decode_ms': 0,
                'max_decode_ms': 0,
                'decode_fps': 0,
            }

        avg_decode = np.mean(self.decode_times)
        return {
            'avg_decode_ms': avg_decode,
            'min_decode_ms': np.min(self.decode_times),
            'max_decode_ms': np.max(self.decode_times),
            'decode_fps': 1000.0 / avg_decode if avg_decode > 0 else 0,
        }

    def close(self):
        """Close video file"""
        if self.cap:
            self.cap.release()
            self.cap = None


class MotionDetector:
    """Motion detection with performance tracking"""

    def __init__(self, threshold: float = 25.0):
        self.threshold = threshold
        self.prev_frame = None
        self.motion_times = []
        self.motion_pixels = []

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Detect motion, return mask, processing time, and pixel count"""
        start = time.time()

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray, dtype=np.uint8), 0.0, 0

        # Compute difference
        diff = cv2.absdiff(self.prev_frame, gray)
        _, motion_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        motion_pixels = np.count_nonzero(motion_mask)
        process_time = (time.time() - start) * 1000  # ms

        self.motion_times.append(process_time)
        self.motion_pixels.append(motion_pixels)
        self.prev_frame = gray

        return motion_mask, process_time, motion_pixels

    def get_metrics(self) -> Dict:
        """Get motion detection metrics"""
        if not self.motion_times:
            return {
                'avg_motion_ms': 0,
                'motion_fps': 0,
                'avg_motion_pixels': 0,
            }

        avg_motion = np.mean(self.motion_times)
        return {
            'avg_motion_ms': avg_motion,
            'min_motion_ms': np.min(self.motion_times),
            'max_motion_ms': np.max(self.motion_times),
            'motion_fps': 1000.0 / avg_motion if avg_motion > 0 else 0,
            'avg_motion_pixels': np.mean(self.motion_pixels),
            'max_motion_pixels': np.max(self.motion_pixels),
        }


class MultiCameraSync:
    """Multi-camera synchronization tracker"""

    def __init__(self, num_cameras: int, sync_window_ms: float = 16.67):
        self.num_cameras = num_cameras
        self.sync_window_ms = sync_window_ms
        self.frame_timestamps: List[deque] = [deque(maxlen=100) for _ in range(num_cameras)]
        self.sync_errors = []

    def add_frame(self, camera_id: int, frame_idx: int, timestamp: float):
        """Add a frame timestamp"""
        self.frame_timestamps[camera_id].append((frame_idx, timestamp))

    def compute_sync_accuracy(self) -> Dict:
        """Compute synchronization accuracy across cameras"""
        if not all(self.frame_timestamps):
            return {'sync_error_ms': 0, 'sync_accuracy': 0}

        # Find common frame indices
        common_frames = set(f[0] for f in self.frame_timestamps[0])
        for cam_frames in self.frame_timestamps[1:]:
            common_frames &= set(f[0] for f in cam_frames)

        if not common_frames:
            return {'sync_error_ms': 0, 'sync_accuracy': 0}

        # Compute sync errors for common frames
        sync_errors = []
        for frame_idx in sorted(common_frames):
            timestamps = []
            for cam_frames in self.frame_timestamps:
                for f_idx, f_time in cam_frames:
                    if f_idx == frame_idx:
                        timestamps.append(f_time)
                        break

            if len(timestamps) == self.num_cameras:
                # Max time difference across cameras
                sync_error = (max(timestamps) - min(timestamps)) * 1000  # ms
                sync_errors.append(sync_error)

        if sync_errors:
            avg_error = np.mean(sync_errors)
            max_error = np.max(sync_errors)
            accuracy = np.mean([e <= self.sync_window_ms for e in sync_errors]) * 100

            return {
                'avg_sync_error_ms': avg_error,
                'max_sync_error_ms': max_error,
                'sync_accuracy_percent': accuracy,
                'frames_analyzed': len(sync_errors),
            }

        return {'sync_error_ms': 0, 'sync_accuracy': 0}


class FrameDropDetector:
    """Detect and analyze frame drops"""

    def __init__(self, expected_fps: float = 30.0):
        self.expected_fps = expected_fps
        self.expected_interval_ms = 1000.0 / expected_fps
        self.frame_times = []
        self.drops_detected = 0

    def add_frame(self, timestamp: float):
        """Add a frame timestamp"""
        self.frame_times.append(timestamp)

        if len(self.frame_times) >= 2:
            interval = (self.frame_times[-1] - self.frame_times[-2]) * 1000
            # Consider a drop if interval > 1.5x expected
            if interval > self.expected_interval_ms * 1.5:
                self.drops_detected += 1

    def get_metrics(self) -> Dict:
        """Get frame drop metrics"""
        if len(self.frame_times) < 2:
            return {
                'total_frames': len(self.frame_times),
                'drops_detected': 0,
                'drop_rate_percent': 0,
            }

        intervals = np.diff(self.frame_times) * 1000  # ms
        return {
            'total_frames': len(self.frame_times),
            'drops_detected': self.drops_detected,
            'drop_rate_percent': (self.drops_detected / len(self.frame_times)) * 100,
            'avg_interval_ms': np.mean(intervals),
            'max_interval_ms': np.max(intervals),
        }


class CameraBenchmark:
    """Main camera benchmark orchestrator"""

    def __init__(self, output_dir: str = "benchmark_results/camera"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def benchmark_8k_decode(self, iterations: int = 300) -> Dict:
        """Benchmark 8K video decode performance"""
        print("\n" + "="*60)
        print("Benchmarking 8K Video Decode Performance")
        print("="*60)

        if not HAS_CV2:
            print("OpenCV not available, skipping")
            return {}

        # Generate synthetic 8K video for testing
        width, height = 7680, 4320
        print(f"Generating synthetic 8K frames ({width}x{height})...")

        decode_times = []
        for i in range(iterations):
            # Simulate video decode by generating frame
            start = time.time()
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

            # Simulate H.264/H.265 decode overhead
            _ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            decode_time = (time.time() - start) * 1000
            decode_times.append(decode_time)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{iterations} frames")

        decode_times = np.array(decode_times)

        results = {
            'resolution': f"{width}x{height}",
            'frames_processed': iterations,
            'avg_decode_ms': np.mean(decode_times),
            'min_decode_ms': np.min(decode_times),
            'max_decode_ms': np.max(decode_times),
            'p95_decode_ms': np.percentile(decode_times, 95),
            'p99_decode_ms': np.percentile(decode_times, 99),
            'decode_fps': 1000.0 / np.mean(decode_times),
            'max_decode_fps': 1000.0 / np.min(decode_times),
        }

        print(f"\nResults:")
        print(f"  Avg Decode Time: {results['avg_decode_ms']:.2f} ms")
        print(f"  Decode FPS:      {results['decode_fps']:.2f}")
        print(f"  Max FPS:         {results['max_decode_fps']:.2f}")
        print(f"  p95 Latency:     {results['p95_decode_ms']:.2f} ms")
        print(f"  p99 Latency:     {results['p99_decode_ms']:.2f} ms")

        return results

    def benchmark_motion_extraction(self, iterations: int = 300) -> Dict:
        """Benchmark motion extraction throughput"""
        print("\n" + "="*60)
        print("Benchmarking Motion Extraction Throughput")
        print("="*60)

        if not HAS_CV2:
            print("OpenCV not available, skipping")
            return {}

        # Test different resolutions
        resolutions = [
            (7680, 4320, "8K"),
            (3840, 2160, "4K"),
            (1920, 1080, "1080p"),
        ]

        all_results = {}

        for width, height, name in resolutions:
            print(f"\nTesting {name} ({width}x{height})...")

            detector = MotionDetector(threshold=25.0)
            motion_times = []

            for i in range(iterations):
                # Generate frame with some motion
                frame = np.random.randint(0, 256, (height, width), dtype=np.uint8)

                start = time.time()
                motion_mask, _, motion_pixels = detector.detect(frame)
                motion_time = (time.time() - start) * 1000

                motion_times.append(motion_time)

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{iterations} frames")

            motion_times = np.array(motion_times[1:])  # Skip first (no prev frame)

            results = {
                'resolution': name,
                'frames_processed': iterations,
                'avg_motion_ms': np.mean(motion_times),
                'min_motion_ms': np.min(motion_times),
                'max_motion_ms': np.max(motion_times),
                'p95_motion_ms': np.percentile(motion_times, 95),
                'p99_motion_ms': np.percentile(motion_times, 99),
                'motion_fps': 1000.0 / np.mean(motion_times),
            }

            print(f"  Avg Motion Time: {results['avg_motion_ms']:.2f} ms")
            print(f"  Motion FPS:      {results['motion_fps']:.2f}")
            print(f"  p99 Latency:     {results['p99_motion_ms']:.2f} ms")

            all_results[name] = results

        return all_results

    def benchmark_multi_camera_sync(self, num_cameras: int = 8,
                                   frames_per_camera: int = 300) -> Dict:
        """Benchmark multi-camera synchronization accuracy"""
        print("\n" + "="*60)
        print(f"Benchmarking Multi-Camera Synchronization ({num_cameras} cameras)")
        print("="*60)

        sync_tracker = MultiCameraSync(num_cameras, sync_window_ms=16.67)

        # Simulate camera captures with slight timing variations
        print(f"Simulating {frames_per_camera} frames per camera...")

        base_time = time.time()
        frame_interval = 1.0 / 30.0  # 30 FPS

        for frame_idx in range(frames_per_camera):
            frame_time = base_time + frame_idx * frame_interval

            for camera_id in range(num_cameras):
                # Add random jitter (0-5ms)
                jitter = np.random.uniform(0, 0.005)
                timestamp = frame_time + jitter

                sync_tracker.add_frame(camera_id, frame_idx, timestamp)

            if (frame_idx + 1) % 50 == 0:
                print(f"  Processed {frame_idx+1}/{frames_per_camera} frames")

        # Compute synchronization metrics
        sync_metrics = sync_tracker.compute_sync_accuracy()

        print(f"\nResults:")
        print(f"  Avg Sync Error:  {sync_metrics.get('avg_sync_error_ms', 0):.2f} ms")
        print(f"  Max Sync Error:  {sync_metrics.get('max_sync_error_ms', 0):.2f} ms")
        print(f"  Sync Accuracy:   {sync_metrics.get('sync_accuracy_percent', 0):.1f}%")

        return {
            'num_cameras': num_cameras,
            'frames_per_camera': frames_per_camera,
            **sync_metrics
        }

    def benchmark_frame_drops(self, duration_sec: int = 10,
                             target_fps: float = 30.0) -> Dict:
        """Benchmark frame drop detection and analysis"""
        print("\n" + "="*60)
        print(f"Benchmarking Frame Drop Analysis ({duration_sec}s @ {target_fps} FPS)")
        print("="*60)

        drop_detector = FrameDropDetector(expected_fps=target_fps)

        # Simulate frame capture with occasional drops
        frame_interval = 1.0 / target_fps
        total_frames = int(duration_sec * target_fps)

        current_time = time.time()

        for i in range(total_frames):
            # Simulate occasional frame drops (5% chance)
            if np.random.random() < 0.05:
                # Skip this frame (simulate drop)
                current_time += frame_interval * 2  # Double interval
            else:
                drop_detector.add_frame(current_time)
                current_time += frame_interval

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{total_frames} frames")

        # Get metrics
        metrics = drop_detector.get_metrics()

        print(f"\nResults:")
        print(f"  Total Frames:    {metrics['total_frames']}")
        print(f"  Drops Detected:  {metrics['drops_detected']}")
        print(f"  Drop Rate:       {metrics['drop_rate_percent']:.2f}%")
        print(f"  Avg Interval:    {metrics['avg_interval_ms']:.2f} ms")

        return {
            'duration_sec': duration_sec,
            'target_fps': target_fps,
            **metrics
        }

    def benchmark_end_to_end_pipeline(self, iterations: int = 300) -> Dict:
        """Benchmark complete camera pipeline (decode + motion)"""
        print("\n" + "="*60)
        print("Benchmarking End-to-End Camera Pipeline")
        print("="*60)

        if not HAS_CV2:
            print("OpenCV not available, skipping")
            return {}

        width, height = 3840, 2160  # 4K
        detector = MotionDetector(threshold=25.0)

        pipeline_times = []
        decode_times = []
        motion_times = []

        print(f"Processing {iterations} 4K frames through pipeline...")

        for i in range(iterations):
            pipeline_start = time.time()

            # Decode simulation
            decode_start = time.time()
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            decode_time = (time.time() - decode_start) * 1000

            # Motion detection
            motion_start = time.time()
            motion_mask, _, _ = detector.detect(gray)
            motion_time = (time.time() - motion_start) * 1000

            pipeline_time = (time.time() - pipeline_start) * 1000

            decode_times.append(decode_time)
            motion_times.append(motion_time)
            pipeline_times.append(pipeline_time)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{iterations} frames")

        pipeline_times = np.array(pipeline_times)
        decode_times = np.array(decode_times)
        motion_times = np.array(motion_times[1:])  # Skip first

        results = {
            'resolution': f"{width}x{height}",
            'frames_processed': iterations,
            'pipeline_fps': 1000.0 / np.mean(pipeline_times),
            'avg_pipeline_ms': np.mean(pipeline_times),
            'p95_pipeline_ms': np.percentile(pipeline_times, 95),
            'p99_pipeline_ms': np.percentile(pipeline_times, 99),
            'decode_fraction': np.mean(decode_times) / np.mean(pipeline_times),
            'motion_fraction': np.mean(motion_times) / np.mean(pipeline_times),
        }

        print(f"\nResults:")
        print(f"  Pipeline FPS:    {results['pipeline_fps']:.2f}")
        print(f"  Avg Pipeline:    {results['avg_pipeline_ms']:.2f} ms")
        print(f"  p99 Latency:     {results['p99_pipeline_ms']:.2f} ms")
        print(f"  Decode %:        {results['decode_fraction']*100:.1f}%")
        print(f"  Motion %:        {results['motion_fraction']*100:.1f}%")

        return results

    def run_all_benchmarks(self):
        """Run complete camera benchmark suite"""
        results = {}

        results['8k_decode'] = self.benchmark_8k_decode(iterations=300)
        results['motion_extraction'] = self.benchmark_motion_extraction(iterations=300)
        results['multi_camera_sync'] = self.benchmark_multi_camera_sync(
            num_cameras=8, frames_per_camera=300
        )
        results['frame_drops'] = self.benchmark_frame_drops(duration_sec=10)
        results['end_to_end_pipeline'] = self.benchmark_end_to_end_pipeline(iterations=300)

        # Save results
        self.save_results(results)

        return results

    def save_results(self, results: Dict):
        """Save benchmark results to JSON"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"camera_benchmark_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}")


def main():
    """Run camera benchmarks"""
    benchmark = CameraBenchmark()

    print("="*60)
    print("Camera Pipeline Benchmark Suite")
    print("="*60)

    if not HAS_CV2:
        print("\nERROR: OpenCV is required for camera benchmarks")
        print("Install with: pip install opencv-python")
        return

    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
