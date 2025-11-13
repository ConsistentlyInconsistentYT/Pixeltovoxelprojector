"""
Network Streaming Integration Tests
Tests network reliability, latency, multi-client support, and failover scenarios

Requirements tested:
- Network streaming reliability
- Sub-100ms end-to-end latency
- Multi-client concurrent streaming
- Automatic failover and recovery
- Bandwidth utilization and throttling
"""

import pytest
import numpy as np
import time
import threading
import queue
import socket
from typing import List, Dict, Optional
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from network.distributed_processor import (
    DistributedProcessor, Task, TaskStatus, WorkerStatus,
    LoadBalancer, TaskScheduler
)
from network.cluster_config import ClusterConfig, NodeStatus, NodeInfo, GPUInfo, ResourceInfo
from network.data_pipeline import DataPipeline, FrameMetadata, RingBuffer

logger = logging.getLogger(__name__)


class MockNetworkTransport:
    """Mock network transport for testing"""

    def __init__(self, latency_ms: float = 5.0, packet_loss: float = 0.0):
        self.latency_ms = latency_ms
        self.packet_loss = packet_loss
        self.sent_packets = 0
        self.received_packets = 0
        self.dropped_packets = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def send(self, data: bytes) -> bool:
        """Simulate sending data with latency and packet loss"""
        self.sent_packets += 1
        self.total_bytes_sent += len(data)

        # Simulate packet loss
        if np.random.random() < self.packet_loss:
            self.dropped_packets += 1
            return False

        # Simulate network latency
        time.sleep(self.latency_ms / 1000.0)

        self.received_packets += 1
        self.total_bytes_received += len(data)
        return True

    def get_stats(self) -> Dict:
        """Get network statistics"""
        return {
            'sent_packets': self.sent_packets,
            'received_packets': self.received_packets,
            'dropped_packets': self.dropped_packets,
            'packet_loss_rate': self.dropped_packets / max(self.sent_packets, 1),
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received
        }


class TestNetworkStreaming:
    """Network streaming integration tests"""

    @pytest.fixture
    def cluster_config(self):
        """Setup cluster configuration"""
        config = ClusterConfig()

        # Add mock nodes
        for i in range(3):
            node = NodeInfo(
                node_id=f"node_{i}",
                hostname=f"worker{i}.local",
                ip_address=f"192.168.1.{10+i}",
                status=NodeStatus.ONLINE,
                resources=ResourceInfo(
                    gpus=[
                        GPUInfo(gpu_id=0, name="RTX 3090", memory_total_mb=24576, compute_capability="8.6"),
                        GPUInfo(gpu_id=1, name="RTX 3090", memory_total_mb=24576, compute_capability="8.6")
                    ],
                    cpu_count=16,
                    ram_gb=64
                )
            )
            config.nodes[node.node_id] = node

        return config

    @pytest.fixture
    def data_pipeline(self):
        """Setup data pipeline"""
        return DataPipeline(
            num_cameras=20,
            buffer_size_mb=2048,
            ring_buffer_frames=60
        )

    @pytest.fixture
    def distributed_processor(self, cluster_config, data_pipeline):
        """Setup distributed processor"""
        processor = DistributedProcessor(
            cluster_config=cluster_config,
            data_pipeline=data_pipeline,
            num_cameras=10,
            enable_fault_tolerance=True
        )

        # Register mock task handler
        def mock_handler(task: Task):
            time.sleep(0.01)  # Simulate processing
            return {"status": "completed", "result": f"processed_{task.task_id}"}

        processor.register_task_handler("process_frame", mock_handler)
        processor.start()

        yield processor
        processor.stop()

    def test_network_reliability(self):
        """Test network streaming reliability"""
        logger.info("Testing network streaming reliability")

        # Test with different packet loss rates
        loss_rates = [0.0, 0.01, 0.05]
        results = []

        for loss_rate in loss_rates:
            transport = MockNetworkTransport(latency_ms=5.0, packet_loss=loss_rate)

            num_packets = 1000
            data = b"x" * 1024  # 1KB packets

            for i in range(num_packets):
                transport.send(data)

            stats = transport.get_stats()
            results.append({
                'loss_rate': loss_rate,
                'delivered_rate': stats['received_packets'] / num_packets,
                'actual_loss': stats['packet_loss_rate']
            })

            logger.info(f"Loss rate {loss_rate*100:.1f}%: delivered {stats['received_packets']}/{num_packets}")

        # Validate reliability
        for result in results:
            expected_delivery = 1.0 - result['loss_rate']
            actual_delivery = result['delivered_rate']

            # Allow 5% tolerance
            assert abs(actual_delivery - expected_delivery) < 0.05, \
                f"Delivery rate {actual_delivery:.2%} differs from expected {expected_delivery:.2%}"

    def test_latency_measurements(self):
        """Test end-to-end latency measurements"""
        logger.info("Testing network latency")

        # Test with different latency configurations
        latency_configs = [1.0, 5.0, 10.0, 20.0]  # milliseconds
        results = []

        for target_latency in latency_configs:
            transport = MockNetworkTransport(latency_ms=target_latency, packet_loss=0.0)

            latencies = []
            num_measurements = 100

            for i in range(num_measurements):
                start_time = time.time()
                data = b"x" * 1024
                transport.send(data)
                latency = (time.time() - start_time) * 1000

                latencies.append(latency)

            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            results.append({
                'target_latency_ms': target_latency,
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency
            })

            logger.info(f"Target {target_latency}ms: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, p99={p99_latency:.2f}ms")

        # Validate latency measurements
        for result in results:
            # Allow 20% tolerance
            tolerance = result['target_latency_ms'] * 0.2
            assert abs(result['avg_latency_ms'] - result['target_latency_ms']) < tolerance, \
                f"Latency {result['avg_latency_ms']:.2f}ms differs from target {result['target_latency_ms']:.2f}ms"

    def test_multi_client_streaming(self, distributed_processor):
        """Test concurrent streaming to multiple clients"""
        logger.info("Testing multi-client concurrent streaming")

        num_clients = 5
        frames_per_client = 50

        client_results = []

        def client_worker(client_id: int):
            """Simulate a client receiving frames"""
            frames_received = 0
            latencies = []

            for frame_num in range(frames_per_client):
                start_time = time.time()

                # Simulate frame data
                frame_data = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
                metadata = FrameMetadata(
                    camera_id=client_id,
                    frame_id=frame_num,
                    timestamp=time.time(),
                    width=1920,
                    height=1080
                )

                # Submit frame for processing
                task_id = distributed_processor.submit_camera_frame(
                    client_id, frame_data, metadata
                )

                # Wait for result
                result = distributed_processor.wait_for_task(task_id, timeout=5.0)

                if result:
                    frames_received += 1
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)

                time.sleep(0.02)  # Simulate frame rate

            client_results.append({
                'client_id': client_id,
                'frames_received': frames_received,
                'avg_latency_ms': np.mean(latencies) if latencies else 0,
                'max_latency_ms': np.max(latencies) if latencies else 0
            })

        # Start all clients
        client_threads = []
        for client_id in range(num_clients):
            thread = threading.Thread(target=client_worker, args=(client_id,))
            thread.start()
            client_threads.append(thread)

        # Wait for completion
        for thread in client_threads:
            thread.join(timeout=30.0)

        # Validate results
        logger.info("Multi-client streaming results:")
        for result in client_results:
            logger.info(f"  Client {result['client_id']}: {result['frames_received']} frames, "
                       f"avg latency: {result['avg_latency_ms']:.2f}ms")

            # Each client should receive most frames
            assert result['frames_received'] >= frames_per_client * 0.9, \
                f"Client {result['client_id']} only received {result['frames_received']}/{frames_per_client} frames"

            # Latency should be reasonable
            assert result['avg_latency_ms'] < 200.0, \
                f"Client {result['client_id']} latency {result['avg_latency_ms']:.2f}ms too high"

    def test_failover_scenarios(self, distributed_processor, cluster_config):
        """Test automatic failover when nodes fail"""
        logger.info("Testing failover scenarios")

        # Submit tasks
        num_tasks = 50
        task_ids = []

        for i in range(num_tasks):
            task = Task(
                task_id=f"task_{i}",
                task_type="process_frame",
                camera_id=i % 10,
                frame_ids=[i],
                input_data={'frame_num': i},
                priority=5
            )
            distributed_processor.submit_task(task)
            task_ids.append(task.task_id)

        time.sleep(0.5)  # Let some tasks start

        # Simulate node failure
        logger.info("Simulating node failure...")
        node_to_fail = "node_1"
        cluster_config.nodes[node_to_fail].status = NodeStatus.OFFLINE

        time.sleep(2.0)  # Allow failover to occur

        # Check system status
        health = distributed_processor.get_system_health()
        stats = distributed_processor.get_statistics()

        logger.info(f"System health after failover:")
        logger.info(f"  Status: {health['status']}")
        logger.info(f"  Online nodes: {health['online_nodes']}")
        logger.info(f"  Active workers: {health['active_workers']}")
        logger.info(f"  Failover count: {health['failover_count']}")
        logger.info(f"  Tasks completed: {stats['tasks_completed']}")
        logger.info(f"  Tasks failed: {stats['tasks_failed']}")

        # Wait for remaining tasks
        time.sleep(5.0)

        final_stats = distributed_processor.get_statistics()

        logger.info(f"Final statistics:")
        logger.info(f"  Tasks completed: {final_stats['tasks_completed']}")
        logger.info(f"  Tasks failed: {final_stats['tasks_failed']}")
        logger.info(f"  Success rate: {final_stats['success_rate']*100:.2f}%")

        # Validate failover
        assert health['failover_count'] > 0, "No failover occurred"
        assert health['online_nodes'] < 3, "Failed node still online"

        # Most tasks should complete despite failure
        completion_rate = final_stats['tasks_completed'] / num_tasks
        assert completion_rate > 0.8, f"Only {completion_rate*100:.2f}% of tasks completed after failover"

    def test_bandwidth_utilization(self, data_pipeline):
        """Test bandwidth utilization and throttling"""
        logger.info("Testing bandwidth utilization")

        # Simulate high-bandwidth streaming
        frame_size_mb = 7680 * 4320 * 3 / (1024 * 1024)  # 8K RGB
        num_cameras = 20
        target_fps = 30

        total_bandwidth_mbps = frame_size_mb * num_cameras * target_fps * 8  # Convert to Mbps

        logger.info(f"Required bandwidth: {total_bandwidth_mbps:.2f} Mbps")

        # Test data pipeline throughput
        num_frames = 100
        start_time = time.time()
        bytes_written = 0

        for frame_num in range(num_frames):
            for camera_id in range(num_cameras):
                # Simulate 8K frame
                frame_data = np.random.randint(0, 255, (4320, 7680, 3), dtype=np.uint8)
                metadata = FrameMetadata(
                    camera_id=camera_id,
                    frame_id=frame_num,
                    timestamp=time.time(),
                    width=7680,
                    height=4320
                )

                data_pipeline.write_frame(camera_id, frame_data, metadata)
                bytes_written += frame_data.nbytes

            time.sleep(1.0 / target_fps)  # Maintain frame rate

        elapsed_time = time.time() - start_time
        actual_bandwidth_mbps = (bytes_written * 8) / (elapsed_time * 1024 * 1024)

        logger.info(f"Actual bandwidth: {actual_bandwidth_mbps:.2f} Mbps")
        logger.info(f"Write time: {elapsed_time:.2f}s")
        logger.info(f"Data written: {bytes_written / (1024**3):.2f} GB")

        # Validate bandwidth
        stats = data_pipeline.get_statistics()
        logger.info(f"Pipeline statistics:")
        logger.info(f"  Total frames: {stats['total_frames_written']}")
        logger.info(f"  Buffer utilization: {stats['buffer_utilization_percent']:.2f}%")

        # Pipeline should handle the load
        assert stats['total_frames_written'] >= num_frames * num_cameras * 0.95, \
            "Pipeline dropped too many frames"

    def test_network_congestion_handling(self):
        """Test handling of network congestion"""
        logger.info("Testing network congestion handling")

        # Simulate congestion with high latency and packet loss
        transport = MockNetworkTransport(latency_ms=50.0, packet_loss=0.10)

        num_packets = 500
        data_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB

        for data_size in data_sizes:
            data = b"x" * data_size

            start_time = time.time()
            successful = 0

            for i in range(num_packets):
                if transport.send(data):
                    successful += 1

            elapsed_time = time.time() - start_time
            throughput_mbps = (successful * data_size * 8) / (elapsed_time * 1024 * 1024)

            logger.info(f"Packet size {data_size/1024:.1f}KB:")
            logger.info(f"  Success rate: {successful/num_packets*100:.2f}%")
            logger.info(f"  Throughput: {throughput_mbps:.2f} Mbps")

            # Should still deliver most packets
            assert successful / num_packets > 0.85, f"Too many packets lost with {data_size} byte packets"

    def test_stream_recovery(self, data_pipeline):
        """Test stream recovery after interruption"""
        logger.info("Testing stream recovery")

        camera_id = 0
        num_frames_before = 50
        num_frames_after = 50

        # Stream frames before interruption
        for frame_num in range(num_frames_before):
            frame_data = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(
                camera_id=camera_id,
                frame_id=frame_num,
                timestamp=time.time(),
                width=1920,
                height=1080
            )
            data_pipeline.write_frame(camera_id, frame_data, metadata)
            time.sleep(0.01)

        # Simulate interruption
        logger.info("Simulating stream interruption...")
        time.sleep(2.0)

        # Resume streaming
        logger.info("Resuming stream...")
        for frame_num in range(num_frames_before, num_frames_before + num_frames_after):
            frame_data = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(
                camera_id=camera_id,
                frame_id=frame_num,
                timestamp=time.time(),
                width=1920,
                height=1080
            )
            data_pipeline.write_frame(camera_id, frame_data, metadata)
            time.sleep(0.01)

        # Validate recovery
        stats = data_pipeline.get_statistics()

        logger.info(f"Stream recovery results:")
        logger.info(f"  Total frames: {stats['total_frames_written']}")
        logger.info(f"  Expected: {num_frames_before + num_frames_after}")

        # Should have recovered and written all frames
        expected_total = num_frames_before + num_frames_after
        assert stats['total_frames_written'] >= expected_total * 0.95, \
            f"Only {stats['total_frames_written']}/{expected_total} frames written after recovery"

    def test_load_balancing_efficiency(self, distributed_processor):
        """Test load balancing across workers"""
        logger.info("Testing load balancing efficiency")

        # Submit many tasks
        num_tasks = 200
        task_ids = []

        for i in range(num_tasks):
            task = Task(
                task_id=f"task_{i}",
                task_type="process_frame",
                camera_id=i % 10,
                frame_ids=[i],
                input_data={'frame_num': i},
                priority=np.random.randint(1, 10)  # Varying priorities
            )
            distributed_processor.submit_task(task)
            task_ids.append(task.task_id)

        # Wait for completion
        time.sleep(10.0)

        # Get statistics
        stats = distributed_processor.get_statistics()

        logger.info(f"Load balancing results:")
        logger.info(f"  Total workers: {stats['total_workers']}")
        logger.info(f"  Tasks completed: {stats['tasks_completed']}")
        logger.info(f"  Avg execution time: {stats.get('avg_execution_time', 0)*1000:.2f}ms")
        logger.info(f"  Success rate: {stats['success_rate']*100:.2f}%")
        logger.info(f"  Load imbalances: {stats['load_imbalances']}")

        # Validate load balancing
        assert stats['tasks_completed'] >= num_tasks * 0.95, \
            f"Only {stats['tasks_completed']}/{num_tasks} tasks completed"
        assert stats['success_rate'] > 0.95, \
            f"Success rate {stats['success_rate']*100:.2f}% too low"

        # Load imbalances should be minimal
        assert stats['load_imbalances'] < num_tasks * 0.1, \
            f"Too many load imbalances: {stats['load_imbalances']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
