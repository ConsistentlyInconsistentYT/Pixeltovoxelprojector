#!/usr/bin/env python3
"""
Network Infrastructure Benchmark
Tests latency, throughput, and scalability of distributed processing system
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import List, Dict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network import (
    RingBuffer,
    DataPipeline,
    FrameMetadata,
    ClusterConfig,
    DistributedProcessor,
    Task
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class NetworkBenchmark:
    """Benchmark suite for network infrastructure"""

    def __init__(self):
        self.results = {}

    def benchmark_ring_buffer(self, capacity: int = 64, num_frames: int = 1000):
        """Benchmark ring buffer performance"""
        print(f"\n{'='*60}")
        print("Ring Buffer Benchmark")
        print(f"{'='*60}")
        print(f"Configuration: capacity={capacity}, frames={num_frames}")

        # Create ring buffer
        buffer = RingBuffer(
            capacity=capacity,
            frame_shape=(1080, 1920, 3),
            dtype=np.float32
        )

        # Generate test data
        test_frame = np.random.rand(1080, 1920, 3).astype(np.float32)
        metadata = FrameMetadata(
            frame_id=0,
            camera_id=0,
            timestamp=0,
            width=1920,
            height=1080,
            channels=3,
            dtype='float32',
            compressed=False,
            checksum='',
            sequence_number=0
        )

        # Benchmark writes
        write_times = []
        for i in range(num_frames):
            start = time.perf_counter()
            success = buffer.write_frame(test_frame, metadata)
            end = time.perf_counter()

            if success:
                write_times.append((end - start) * 1e6)  # microseconds

            # Prevent buffer overflow
            if i % 10 == 0:
                buffer.read_frame()

        # Benchmark reads
        read_times = []
        for i in range(min(num_frames, capacity)):
            start = time.perf_counter()
            result = buffer.read_frame()
            end = time.perf_counter()

            if result:
                read_times.append((end - start) * 1e6)

        # Results
        stats = buffer.get_statistics()

        print(f"\nWrite Performance:")
        print(f"  Average: {np.mean(write_times):.2f} μs")
        print(f"  Median:  {np.median(write_times):.2f} μs")
        print(f"  P99:     {np.percentile(write_times, 99):.2f} μs")
        print(f"  Success: {len(write_times)}/{num_frames} ({len(write_times)/num_frames*100:.1f}%)")

        print(f"\nRead Performance:")
        print(f"  Average: {np.mean(read_times):.2f} μs")
        print(f"  Median:  {np.median(read_times):.2f} μs")
        print(f"  P99:     {np.percentile(read_times, 99):.2f} μs")

        print(f"\nBuffer Statistics:")
        print(f"  Capacity:       {stats['capacity']}")
        print(f"  Utilization:    {stats['utilization']*100:.1f}%")
        print(f"  Write failures: {stats['write_failures']}")
        print(f"  Read failures:  {stats['read_failures']}")

        self.results['ring_buffer'] = {
            'write_avg_us': np.mean(write_times),
            'read_avg_us': np.mean(read_times),
            'write_p99_us': np.percentile(write_times, 99),
            'utilization': stats['utilization']
        }

    def benchmark_data_pipeline(self, num_cameras: int = 4, frames_per_camera: int = 100):
        """Benchmark data pipeline throughput"""
        print(f"\n{'='*60}")
        print("Data Pipeline Benchmark")
        print(f"{'='*60}")
        print(f"Configuration: {num_cameras} cameras, {frames_per_camera} frames each")

        # Create pipeline
        pipeline = DataPipeline(
            buffer_capacity=64,
            frame_shape=(1080, 1920, 3),
            enable_rdma=False,  # Disable RDMA for benchmark
            enable_shared_memory=False
        )

        # Create buffers
        for camera_id in range(num_cameras):
            pipeline.create_ring_buffer(camera_id)

        # Generate test data
        test_frame = np.random.rand(1080, 1920, 3).astype(np.float32)

        # Benchmark
        start_time = time.time()
        total_bytes = 0

        for camera_id in range(num_cameras):
            for frame_id in range(frames_per_camera):
                metadata = FrameMetadata(
                    frame_id=frame_id,
                    camera_id=camera_id,
                    timestamp=time.time(),
                    width=1920,
                    height=1080,
                    channels=3,
                    dtype='float32',
                    compressed=False,
                    checksum='',
                    sequence_number=0
                )

                pipeline.write_frame(camera_id, test_frame, metadata)
                total_bytes += test_frame.nbytes

        end_time = time.time()
        duration = end_time - start_time

        # Results
        stats = pipeline.get_statistics()

        print(f"\nThroughput:")
        print(f"  Total frames:   {stats['frames_processed']}")
        print(f"  Duration:       {duration:.2f} seconds")
        print(f"  Frames/second:  {stats['frames_processed']/duration:.2f}")
        print(f"  Throughput:     {total_bytes/duration/1e9:.2f} GB/s")

        print(f"\nPipeline Statistics:")
        for camera_id in range(num_cameras):
            buffer_stats = stats['buffers'][f'camera_{camera_id}']
            print(f"  Camera {camera_id}:")
            print(f"    Utilization: {buffer_stats['utilization']*100:.1f}%")
            print(f"    Failures:    {buffer_stats['write_failures'] + buffer_stats['read_failures']}")

        self.results['data_pipeline'] = {
            'throughput_gbps': total_bytes / duration / 1e9,
            'frames_per_sec': stats['frames_processed'] / duration,
            'duration_sec': duration
        }

        pipeline.cleanup()

    def benchmark_task_scheduling(self, num_tasks: int = 1000):
        """Benchmark task scheduling overhead"""
        print(f"\n{'='*60}")
        print("Task Scheduling Benchmark")
        print(f"{'='*60}")
        print(f"Configuration: {num_tasks} tasks")

        # Create minimal system
        cluster = ClusterConfig(enable_rdma=False)
        cluster.start(is_master=True)
        time.sleep(1)

        pipeline = DataPipeline(
            buffer_capacity=32,
            frame_shape=(1080, 1920, 3),
            enable_rdma=False,
            enable_shared_memory=False
        )

        processor = DistributedProcessor(
            cluster_config=cluster,
            data_pipeline=pipeline,
            num_cameras=1,
            enable_fault_tolerance=False
        )

        # Simple task handler
        def dummy_task(task: Task):
            return {'result': 'ok'}

        processor.register_task_handler('dummy', dummy_task)
        processor.start()
        time.sleep(1)

        # Benchmark task submission
        start_time = time.time()
        task_ids = []

        for i in range(num_tasks):
            task = Task(
                task_id=f"task_{i}",
                task_type='dummy',
                camera_id=0,
                frame_ids=[i],
                input_data={},
                priority=1
            )
            processor.submit_task(task)
            task_ids.append(task.task_id)

        submission_time = time.time() - start_time

        # Wait for completion
        wait_start = time.time()
        completed = 0

        for task_id in task_ids:
            result = processor.wait_for_task(task_id, timeout=10.0)
            if result:
                completed += 1

        completion_time = time.time() - wait_start
        total_time = time.time() - start_time

        # Results
        stats = processor.get_statistics()

        print(f"\nScheduling Performance:")
        print(f"  Submission time: {submission_time:.3f} seconds")
        print(f"  Submission rate: {num_tasks/submission_time:.0f} tasks/sec")
        print(f"  Avg submit time: {submission_time/num_tasks*1000:.3f} ms")

        print(f"\nExecution Performance:")
        print(f"  Completion time: {completion_time:.3f} seconds")
        print(f"  Total time:      {total_time:.3f} seconds")
        print(f"  Throughput:      {completed/total_time:.0f} tasks/sec")
        print(f"  Success rate:    {stats['success_rate']*100:.1f}%")

        print(f"\nWorker Statistics:")
        print(f"  Total workers:   {stats['total_workers']}")
        print(f"  Avg task time:   {stats['avg_execution_time']*1000:.2f} ms")

        self.results['task_scheduling'] = {
            'submission_rate': num_tasks / submission_time,
            'throughput': completed / total_time,
            'success_rate': stats['success_rate']
        }

        # Cleanup
        processor.stop()
        cluster.stop()
        pipeline.cleanup()

    def benchmark_latency(self, num_iterations: int = 100):
        """Benchmark end-to-end latency"""
        print(f"\n{'='*60}")
        print("End-to-End Latency Benchmark")
        print(f"{'='*60}")
        print(f"Configuration: {num_iterations} iterations")

        # Setup
        cluster = ClusterConfig(enable_rdma=False)
        cluster.start(is_master=True)
        time.sleep(1)

        pipeline = DataPipeline(
            buffer_capacity=32,
            frame_shape=(1080, 1920, 3),
            enable_rdma=False,
            enable_shared_memory=False
        )

        processor = DistributedProcessor(
            cluster_config=cluster,
            data_pipeline=pipeline,
            num_cameras=1,
            enable_fault_tolerance=False
        )

        def timed_task(task: Task):
            # Simulate processing
            time.sleep(0.001)
            return {'timestamp': time.time()}

        processor.register_task_handler('timed', timed_task)
        processor.start()
        time.sleep(1)

        # Benchmark
        latencies = []
        test_frame = np.random.rand(1080, 1920, 3).astype(np.float32)

        for i in range(num_iterations):
            start = time.time()

            # Submit frame
            metadata = FrameMetadata(
                frame_id=i,
                camera_id=0,
                timestamp=start,
                width=1920,
                height=1080,
                channels=3,
                dtype='float32',
                compressed=False,
                checksum='',
                sequence_number=i
            )

            task_id = processor.submit_camera_frame(0, test_frame, metadata)

            # Wait for result
            result = processor.wait_for_task(task_id, timeout=5.0)

            if result:
                end = time.time()
                latencies.append((end - start) * 1000)  # milliseconds

        # Results
        print(f"\nLatency Distribution:")
        print(f"  Average:  {np.mean(latencies):.2f} ms")
        print(f"  Median:   {np.median(latencies):.2f} ms")
        print(f"  P50:      {np.percentile(latencies, 50):.2f} ms")
        print(f"  P95:      {np.percentile(latencies, 95):.2f} ms")
        print(f"  P99:      {np.percentile(latencies, 99):.2f} ms")
        print(f"  Min:      {np.min(latencies):.2f} ms")
        print(f"  Max:      {np.max(latencies):.2f} ms")

        self.results['latency'] = {
            'avg_ms': np.mean(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99)
        }

        # Cleanup
        processor.stop()
        cluster.stop()
        pipeline.cleanup()

    def print_summary(self):
        """Print benchmark summary"""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")

        if 'ring_buffer' in self.results:
            r = self.results['ring_buffer']
            print(f"\nRing Buffer:")
            print(f"  Write latency: {r['write_avg_us']:.2f} μs (P99: {r['write_p99_us']:.2f} μs)")
            print(f"  Read latency:  {r['read_avg_us']:.2f} μs")

        if 'data_pipeline' in self.results:
            r = self.results['data_pipeline']
            print(f"\nData Pipeline:")
            print(f"  Throughput: {r['throughput_gbps']:.2f} GB/s")
            print(f"  Frame rate: {r['frames_per_sec']:.2f} fps")

        if 'task_scheduling' in self.results:
            r = self.results['task_scheduling']
            print(f"\nTask Scheduling:")
            print(f"  Submission: {r['submission_rate']:.0f} tasks/sec")
            print(f"  Throughput: {r['throughput']:.0f} tasks/sec")
            print(f"  Success:    {r['success_rate']*100:.1f}%")

        if 'latency' in self.results:
            r = self.results['latency']
            print(f"\nEnd-to-End Latency:")
            print(f"  Average: {r['avg_ms']:.2f} ms")
            print(f"  P95:     {r['p95_ms']:.2f} ms")
            print(f"  P99:     {r['p99_ms']:.2f} ms")

        print(f"\n{'='*60}")


def main():
    """Run all benchmarks"""
    print("Network Infrastructure Benchmark Suite")
    print("This may take several minutes...")

    benchmark = NetworkBenchmark()

    try:
        # Run benchmarks
        benchmark.benchmark_ring_buffer(capacity=64, num_frames=1000)
        benchmark.benchmark_data_pipeline(num_cameras=4, frames_per_camera=100)
        benchmark.benchmark_task_scheduling(num_tasks=500)
        benchmark.benchmark_latency(num_iterations=50)

        # Summary
        benchmark.print_summary()

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
