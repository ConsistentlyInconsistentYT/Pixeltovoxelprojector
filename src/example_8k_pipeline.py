"""
8K Video Analysis Pipeline - Example Usage

This example demonstrates how to use the 8K video processing pipeline
for real-time motion extraction from dual camera streams.
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

try:
    from video_processor import VideoProcessor, VideoStream, MotionData
    from codec_handler import VideoCodec
except ImportError:
    print("Error: Unable to import video processing modules")
    print("Please ensure you're in the src directory and dependencies are installed")
    sys.exit(1)


class MotionVisualization:
    """Visualization for motion detection results"""

    def __init__(self, output_path: str = None):
        self.output_path = output_path
        self.motion_history = []
        self.max_history = 100

    def visualize(self, frame: np.ndarray, motion_data: MotionData) -> np.ndarray:
        """
        Visualize motion detection results on frame.

        Args:
            frame: Input frame
            motion_data: Motion detection results

        Returns:
            Annotated frame
        """
        # Convert to color if grayscale
        if len(frame.shape) == 2:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis_frame = frame.copy()

        # Draw bounding boxes
        for bbox in motion_data.bounding_boxes:
            x, y, w, h = bbox
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw motion coordinates
        for coord in motion_data.coordinates:
            x, y = coord
            cv2.circle(vis_frame, (x, y), 5, (0, 0, 255), -1)

        # Draw velocity vectors
        for i, (coord, vel) in enumerate(zip(motion_data.coordinates, motion_data.velocities)):
            x, y = coord
            vx, vy = vel

            # Scale velocity for visualization
            end_x = int(x + vx * 5)
            end_y = int(y + vy * 5)

            cv2.arrowedLine(vis_frame, (x, y), (end_x, end_y), (255, 0, 0), 2)

        # Add info text
        info_text = [
            f"Objects: {len(motion_data.coordinates)}",
            f"Frame: {motion_data.frame_number}",
            f"Time: {motion_data.timestamp:.2f}s"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(
                vis_frame,
                text,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        return vis_frame


def example_single_stream():
    """Example: Process single 8K video stream"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single 8K Video Stream Processing")
    print("="*80)

    # Create processor
    processor = VideoProcessor(
        use_hardware_accel=True,
        target_fps=30.0,
        enable_profiling=True
    )

    # Add video stream
    video_path = "path/to/8k_video.mp4"  # Replace with actual path

    success = processor.add_stream(
        'main_camera',
        VideoStream(
            path=video_path,
            stream_type='monochrome',
            codec=VideoCodec.HEVC,
            buffer_size=30
        )
    )

    if not success:
        print(f"Error: Could not open video: {video_path}")
        return

    # Create visualization
    visualizer = MotionVisualization()

    # Motion detection callback
    motion_events = []

    def on_motion_detected(motion_data: MotionData):
        motion_events.append(motion_data)
        print(f"[Motion] Detected {len(motion_data.coordinates)} objects at frame {motion_data.frame_number}")

    processor.register_motion_callback(on_motion_detected)

    # Start processing
    print("\nStarting video processing...")
    processor.start_processing()

    try:
        # Process for 10 seconds
        duration = 10.0
        print(f"Processing for {duration} seconds...")
        time.sleep(duration)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Stop processing
        processor.stop_processing()

        # Print final metrics
        metrics = processor.get_metrics()
        print("\nFinal Metrics:")
        print(f"  Frames processed: {metrics['frames_processed']}")
        print(f"  Average FPS: {metrics['current_fps']:.2f}")
        print(f"  Average latency: {metrics['avg_latency_ms']:.2f} ms")
        print(f"  Motion events: {len(motion_events)}")


def example_dual_stream():
    """Example: Process dual camera streams (monochrome + thermal)"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Dual Camera Stream Processing")
    print("="*80)

    # Create processor
    processor = VideoProcessor(
        use_hardware_accel=True,
        target_fps=30.0,
        enable_profiling=True
    )

    # Add monochrome camera stream
    mono_video = "path/to/8k_mono.mp4"
    processor.add_stream(
        'mono_cam',
        VideoStream(
            path=mono_video,
            stream_type='monochrome',
            codec=VideoCodec.HEVC,
            buffer_size=30
        )
    )

    # Add thermal camera stream
    thermal_video = "path/to/8k_thermal.raw"
    processor.add_stream(
        'thermal_cam',
        VideoStream(
            path=thermal_video,
            stream_type='thermal',
            codec=VideoCodec.RAW_THERMAL,
            buffer_size=30
        )
    )

    # Motion detection callback
    def on_motion_detected(motion_data: MotionData):
        print(f"[DualCam] Frame {motion_data.frame_number}: {len(motion_data.coordinates)} objects")

    processor.register_motion_callback(on_motion_detected)

    # Start processing
    processor.start_processing()

    try:
        # Synchronize and process frames
        for _ in range(300):  # Process 300 synchronized frame pairs
            # Get synchronized frames from both cameras
            synced_frames = processor.synchronize_streams(
                ['mono_cam', 'thermal_cam'],
                max_time_diff_ms=10.0
            )

            if synced_frames:
                mono_frame = synced_frames['mono_cam']
                thermal_frame = synced_frames['thermal_cam']

                # Process synchronized frames
                print(f"Synced frames - Mono: {mono_frame.frame_number}, Thermal: {thermal_frame.frame_number}")

            time.sleep(1.0 / 30.0)  # 30 FPS

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        processor.stop_processing()


def example_motion_export():
    """Example: Export motion coordinates to file"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Export Motion Coordinates")
    print("="*80)

    processor = VideoProcessor(use_hardware_accel=True)

    video_path = "path/to/8k_video.mp4"
    processor.add_stream(
        'camera',
        VideoStream(path=video_path, stream_type='monochrome')
    )

    # Open output file for motion coordinates
    output_file = open('motion_coordinates.csv', 'w')
    output_file.write("frame,timestamp,object_id,x,y,vx,vy,confidence\n")

    object_id = 0

    def export_motion(motion_data: MotionData):
        nonlocal object_id

        for coord, vel, conf in zip(
            motion_data.coordinates,
            motion_data.velocities,
            motion_data.confidence
        ):
            x, y = coord
            vx, vy = vel

            output_file.write(
                f"{motion_data.frame_number},"
                f"{motion_data.timestamp},"
                f"{object_id},"
                f"{x},{y},"
                f"{vx},{vy},"
                f"{conf}\n"
            )

            object_id += 1

    processor.register_motion_callback(export_motion)
    processor.start_processing()

    try:
        time.sleep(10.0)  # Process for 10 seconds
    finally:
        processor.stop_processing()
        output_file.close()
        print(f"Motion coordinates exported to: motion_coordinates.csv")


def example_cpp_motion_extractor():
    """Example: Direct use of C++ motion extractor"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Direct C++ Motion Extractor Usage")
    print("="*80)

    try:
        from motion_extractor_cpp import MotionExtractor8K
    except ImportError:
        print("Error: C++ motion extractor module not built")
        print("Run: python setup_motion_extractor.py build_ext --inplace")
        return

    # Create extractor
    extractor = MotionExtractor8K(width=7680, height=4320)

    # Create synthetic 8K frame
    print("Creating synthetic 8K frame...")
    frame = np.zeros((4320, 7680), dtype=np.uint8)

    # Add some motion
    for i in range(10):
        x = np.random.randint(0, 7680 - 100)
        y = np.random.randint(0, 4320 - 100)
        size = np.random.randint(50, 150)
        cv2.circle(frame, (x, y), size, 255, -1)

    # Extract motion
    print("Extracting motion...")
    result = extractor.extract(frame)

    print(f"\nResults:")
    print(f"  Objects detected: {result['num_objects']}")
    print(f"  Processing time: {result['processing_time_ms']:.2f} ms")
    print(f"  Coordinates: {result['coordinates']}")
    print(f"  Bounding boxes: {result['bounding_boxes']}")

    # Process multiple frames
    print("\nProcessing 100 frames...")
    for i in range(100):
        # Modify frame to simulate motion
        frame = np.roll(frame, 10, axis=1)
        result = extractor.extract(frame)

    stats = extractor.getStats()
    print(f"\nPerformance Statistics:")
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Average processing time: {stats['avg_processing_time_ms']:.2f} ms")
    print(f"  Tracked objects: {stats['num_tracked_objects']}")


def example_performance_comparison():
    """Example: Compare Python vs C++ performance"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Performance Comparison (Python vs C++)")
    print("="*80)

    # Create test frame
    frame = np.random.randint(0, 255, (4320, 7680), dtype=np.uint8)

    try:
        from motion_extractor_cpp import MotionExtractor8K

        extractor = MotionExtractor8K()

        # Benchmark C++ implementation
        print("\nBenchmarking C++ implementation...")
        start_time = time.perf_counter()

        for i in range(100):
            result = extractor.extract(frame)

        cpp_time = time.perf_counter() - start_time
        cpp_fps = 100 / cpp_time

        print(f"  Total time: {cpp_time:.2f}s")
        print(f"  FPS: {cpp_fps:.2f}")
        print(f"  ms/frame: {cpp_time * 1000 / 100:.2f}")

        stats = extractor.getStats()
        print(f"  Average processing time: {stats['avg_processing_time_ms']:.2f} ms")

    except ImportError:
        print("C++ module not available for comparison")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='8K Video Processing Examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Example number to run (1-5)'
    )

    args = parser.parse_args()

    examples = {
        1: example_single_stream,
        2: example_dual_stream,
        3: example_motion_export,
        4: example_cpp_motion_extractor,
        5: example_performance_comparison
    }

    if args.example:
        examples[args.example]()
    else:
        # Run all examples
        print("\n8K VIDEO PROCESSING PIPELINE - EXAMPLES\n")
        print("Choose an example to run:")
        print("  1. Single 8K video stream processing")
        print("  2. Dual camera stream processing")
        print("  3. Export motion coordinates")
        print("  4. Direct C++ motion extractor usage")
        print("  5. Performance comparison (Python vs C++)")
        print("\nRun with --example <number> to run a specific example")
