"""
8K Video Processing Performance Benchmark

This script benchmarks the performance of the 8K video processing pipeline,
measuring FPS, latency, and resource utilization.
"""

import sys
import time
import numpy as np
import cv2
from typing import Dict, List
import json

try:
    from video_processor import VideoProcessor, VideoStream, MotionData
    from codec_handler import CodecHandler, VideoCodec, CodecBenchmark
except ImportError:
    print("Error: Unable to import video processing modules")
    print("Make sure you're running from the src directory")
    sys.exit(1)


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite"""

    def __init__(self):
        self.results = {}

    def create_test_video(
        self,
        output_path: str,
        width: int = 7680,
        height: int = 4320,
        fps: float = 30.0,
        duration: float = 10.0,
        codec: str = 'hevc'
    ):
        """
        Create synthetic 8K test video.

        Args:
            output_path: Output video file path
            width: Video width
            height: Video height
            fps: Frames per second
            duration: Video duration in seconds
            codec: Video codec (hevc, h264, etc.)
        """
        print(f"\n[Benchmark] Creating test video: {width}x{height} @ {fps} FPS")
        print(f"[Benchmark] Duration: {duration}s, Codec: {codec}")

        # Codec fourcc codes
        codec_map = {
            'hevc': cv2.VideoWriter_fourcc(*'X265'),
            'h265': cv2.VideoWriter_fourcc(*'X265'),
            'h264': cv2.VideoWriter_fourcc(*'X264'),
            'vp9': cv2.VideoWriter_fourcc(*'VP90'),
        }

        fourcc = codec_map.get(codec.lower(), cv2.VideoWriter_fourcc(*'X265'))

        # Create video writer
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)

        if not writer.isOpened():
            print("[Benchmark] Warning: Could not create video with hardware codec")
            print("[Benchmark] Falling back to MJPEG")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)

        num_frames = int(fps * duration)

        # Generate frames with moving patterns
        for i in range(num_frames):
            # Create synthetic motion pattern
            frame = self._generate_motion_frame(width, height, i, num_frames)
            writer.write(frame)

            if (i + 1) % 30 == 0:
                print(f"[Benchmark] Generated {i + 1}/{num_frames} frames")

        writer.release()
        print(f"[Benchmark] Test video created: {output_path}")

    def _generate_motion_frame(
        self,
        width: int,
        height: int,
        frame_num: int,
        total_frames: int
    ) -> np.ndarray:
        """Generate frame with synthetic motion"""
        frame = np.zeros((height, width), dtype=np.uint8)

        # Add moving objects
        num_objects = 10
        for obj_id in range(num_objects):
            # Calculate object position
            t = (frame_num + obj_id * total_frames // num_objects) / total_frames
            x = int((width * 0.8) * np.sin(2 * np.pi * t) + width // 2)
            y = int((height * 0.8) * np.cos(2 * np.pi * t) + height // 2)

            # Draw object
            size = 50 + obj_id * 10
            cv2.circle(frame, (x, y), size, 255, -1)

        # Add noise
        noise = np.random.randint(0, 20, (height, width), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        return frame

    def benchmark_codec_decoding(
        self,
        video_path: str,
        num_frames: int = 300
    ) -> Dict:
        """Benchmark video codec decoding performance"""
        print(f"\n{'='*80}")
        print("CODEC DECODING BENCHMARK")
        print(f"{'='*80}")

        results = {}

        # Test with hardware acceleration
        print("\n[Benchmark] Testing with hardware acceleration...")
        hw_results = CodecBenchmark.benchmark_decode(
            video_path,
            num_frames=num_frames,
            use_hw_accel=True
        )
        results['hardware_accel'] = hw_results

        print(f"  Frames decoded: {hw_results['frames_decoded']}")
        print(f"  FPS: {hw_results['fps']:.2f}")
        print(f"  ms/frame: {hw_results['ms_per_frame']:.2f}")

        # Test without hardware acceleration
        print("\n[Benchmark] Testing without hardware acceleration...")
        sw_results = CodecBenchmark.benchmark_decode(
            video_path,
            num_frames=num_frames,
            use_hw_accel=False
        )
        results['software_only'] = sw_results

        print(f"  Frames decoded: {sw_results['frames_decoded']}")
        print(f"  FPS: {sw_results['fps']:.2f}")
        print(f"  ms/frame: {sw_results['ms_per_frame']:.2f}")

        # Calculate speedup
        if sw_results['fps'] > 0:
            speedup = hw_results['fps'] / sw_results['fps']
            results['hw_speedup'] = speedup
            print(f"\n  Hardware acceleration speedup: {speedup:.2f}x")

        return results

    def benchmark_motion_extraction(
        self,
        video_path: str,
        duration: float = 10.0
    ) -> Dict:
        """Benchmark complete motion extraction pipeline"""
        print(f"\n{'='*80}")
        print("MOTION EXTRACTION BENCHMARK")
        print(f"{'='*80}")

        # Create processor
        processor = VideoProcessor(
            use_hardware_accel=True,
            target_fps=30.0,
            enable_profiling=False  # Disable continuous printing
        )

        # Add video stream
        success = processor.add_stream(
            'test_stream',
            VideoStream(
                path=video_path,
                stream_type='monochrome',
                codec=VideoCodec.HEVC
            )
        )

        if not success:
            return {'error': 'Failed to add stream'}

        # Collect metrics
        motion_count = 0
        latencies = []
        processing_times = []

        def motion_callback(motion_data: MotionData):
            nonlocal motion_count
            motion_count += 1
            # Store metrics if available

        processor.register_motion_callback(motion_callback)

        # Start processing
        print("\n[Benchmark] Starting motion extraction...")
        processor.start_processing()

        # Let it run for specified duration
        time.sleep(duration)

        # Get final metrics
        metrics = processor.get_metrics()

        # Stop processing
        processor.stop_processing()

        print(f"\n[Benchmark] Processing complete")
        print(f"  Frames processed: {metrics['frames_processed']}")
        print(f"  Current FPS: {metrics['current_fps']:.2f}")
        print(f"  Avg latency: {metrics['avg_latency_ms']:.2f} ms")
        print(f"  Decode time: {metrics['decode_time_ms']:.2f} ms")
        print(f"  Motion extraction time: {metrics['motion_extract_time_ms']:.2f} ms")

        return {
            'frames_processed': metrics['frames_processed'],
            'fps': metrics['current_fps'],
            'avg_latency_ms': metrics['avg_latency_ms'],
            'decode_time_ms': metrics['decode_time_ms'],
            'motion_extract_time_ms': metrics['motion_extract_time_ms'],
            'total_pipeline_time_ms': metrics['decode_time_ms'] + metrics['motion_extract_time_ms'],
            'meets_30fps_target': metrics['current_fps'] >= 30.0,
            'meets_latency_target': metrics['avg_latency_ms'] < 33.0
        }

    def benchmark_dual_stream(
        self,
        mono_video_path: str,
        thermal_video_path: str,
        duration: float = 10.0
    ) -> Dict:
        """Benchmark dual camera stream processing"""
        print(f"\n{'='*80}")
        print("DUAL STREAM SYNCHRONIZATION BENCHMARK")
        print(f"{'='*80}")

        processor = VideoProcessor(
            use_hardware_accel=True,
            target_fps=30.0,
            enable_profiling=False
        )

        # Add both streams
        processor.add_stream(
            'mono_cam',
            VideoStream(
                path=mono_video_path,
                stream_type='monochrome',
                codec=VideoCodec.HEVC
            )
        )

        processor.add_stream(
            'thermal_cam',
            VideoStream(
                path=thermal_video_path,
                stream_type='thermal',
                codec=VideoCodec.HEVC
            )
        )

        processor.start_processing()
        time.sleep(duration)

        metrics = processor.get_metrics()
        processor.stop_processing()

        print(f"\n[Benchmark] Dual stream processing complete")
        print(f"  Frames processed: {metrics['frames_processed']}")
        print(f"  FPS: {metrics['current_fps']:.2f}")
        print(f"  Frames dropped: {metrics['frames_dropped']}")

        return metrics

    def run_full_benchmark(
        self,
        test_video_path: str = '/tmp/test_8k.mp4'
    ):
        """Run complete benchmark suite"""
        print("\n" + "="*80)
        print("8K VIDEO PROCESSING - FULL BENCHMARK SUITE")
        print("="*80)

        # Create test video
        print("\n[1/3] Creating test video...")
        self.create_test_video(
            test_video_path,
            width=7680,
            height=4320,
            fps=30.0,
            duration=10.0,
            codec='hevc'
        )

        # Benchmark codec decoding
        print("\n[2/3] Benchmarking codec decoding...")
        codec_results = self.benchmark_codec_decoding(test_video_path, num_frames=300)
        self.results['codec_decoding'] = codec_results

        # Benchmark motion extraction
        print("\n[3/3] Benchmarking motion extraction...")
        motion_results = self.benchmark_motion_extraction(test_video_path, duration=10.0)
        self.results['motion_extraction'] = motion_results

        # Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        if 'codec_decoding' in self.results:
            codec = self.results['codec_decoding']
            print("\nCodec Decoding Performance:")
            if 'hardware_accel' in codec:
                hw = codec['hardware_accel']
                print(f"  Hardware Accelerated: {hw['fps']:.2f} FPS ({hw['ms_per_frame']:.2f} ms/frame)")
            if 'software_only' in codec:
                sw = codec['software_only']
                print(f"  Software Only: {sw['fps']:.2f} FPS ({sw['ms_per_frame']:.2f} ms/frame)")
            if 'hw_speedup' in codec:
                print(f"  Speedup: {codec['hw_speedup']:.2f}x")

        if 'motion_extraction' in self.results:
            motion = self.results['motion_extraction']
            print("\nMotion Extraction Performance:")
            print(f"  Processing FPS: {motion['fps']:.2f}")
            print(f"  Average Latency: {motion['avg_latency_ms']:.2f} ms")
            print(f"  Decode Time: {motion['decode_time_ms']:.2f} ms")
            print(f"  Motion Extract Time: {motion['motion_extract_time_ms']:.2f} ms")
            print(f"  Total Pipeline Time: {motion['total_pipeline_time_ms']:.2f} ms")
            print(f"\n  Meets 30 FPS target: {'YES' if motion['meets_30fps_target'] else 'NO'}")
            print(f"  Meets <33ms latency target: {'YES' if motion['meets_latency_target'] else 'NO'}")

        print("\n" + "="*80)

    def save_results(self, output_path: str = 'benchmark_results.json'):
        """Save benchmark results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n[Benchmark] Results saved to: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='8K Video Processing Benchmark')
    parser.add_argument('--video', type=str, help='Path to test video (will create if not exists)')
    parser.add_argument('--duration', type=float, default=10.0, help='Benchmark duration in seconds')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output JSON file')

    args = parser.parse_args()

    benchmark = PerformanceBenchmark()

    if args.video:
        # Use provided video
        benchmark.run_full_benchmark(args.video)
    else:
        # Create temporary test video
        benchmark.run_full_benchmark('/tmp/test_8k.mp4')

    benchmark.save_results(args.output)
