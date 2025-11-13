#!/usr/bin/env python3
"""
Example: Distributed Processing with Multi-GPU Support
Demonstrates the distributed processing infrastructure for 10 camera pairs
"""

import numpy as np
import time
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network import (
    ClusterConfig,
    DataPipeline,
    DistributedProcessor,
    Task,
    FrameMetadata
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_voxel_frame(task: Task):
    """
    Example task handler for voxel frame processing

    Args:
        task: Task containing frame data

    Returns:
        Processed voxel grid
    """
    frame_data = task.input_data['frame']
    metadata = task.input_data['metadata']

    logger.info(f"Processing frame {metadata.frame_id} from camera {task.camera_id}")

    # Simulate voxel processing (replace with actual processing)
    # In real implementation, this would:
    # 1. Cast rays through voxel grid
    # 2. Accumulate brightness values
    # 3. Update voxel occupancy

    time.sleep(0.1)  # Simulate processing time

    # Generate dummy voxel grid result
    voxel_grid = np.random.rand(100, 100, 100).astype(np.float32)

    return {
        'voxel_grid': voxel_grid,
        'camera_id': task.camera_id,
        'frame_id': metadata.frame_id,
        'timestamp': metadata.timestamp
    }


def generate_synthetic_frame(camera_id: int, frame_id: int, width: int = 3840, height: int = 2160):
    """Generate synthetic 8K camera frame"""
    # Generate synthetic frame data
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).astype(np.float32) / 255.0

    metadata = FrameMetadata(
        frame_id=frame_id,
        camera_id=camera_id,
        timestamp=time.time(),
        width=width,
        height=height,
        channels=3,
        dtype='float32',
        compressed=False,
        checksum='',
        sequence_number=0
    )

    return frame, metadata


def main():
    """Main demonstration"""
    logger.info("=" * 80)
    logger.info("Distributed Processing System Demo")
    logger.info("=" * 80)

    # Configuration
    num_cameras = 10
    frames_per_camera = 5

    # Initialize cluster configuration
    logger.info("\n[1/5] Initializing cluster configuration...")
    cluster = ClusterConfig(
        discovery_port=9999,
        heartbeat_interval=1.0,
        heartbeat_timeout=5.0,
        enable_rdma=True
    )

    # Start cluster services (as master node)
    cluster.start(is_master=True)
    time.sleep(2)  # Wait for node discovery

    # Initialize data pipeline
    logger.info("\n[2/5] Initializing data pipeline...")
    pipeline = DataPipeline(
        buffer_capacity=64,
        frame_shape=(2160, 3840, 3),  # 8K resolution
        enable_rdma=True,
        enable_shared_memory=True,
        shm_size_mb=2048  # 2GB shared memory
    )

    # Create ring buffers for each camera
    for camera_id in range(num_cameras):
        pipeline.create_ring_buffer(camera_id)

    # Initialize distributed processor
    logger.info("\n[3/5] Initializing distributed processor...")
    processor = DistributedProcessor(
        cluster_config=cluster,
        data_pipeline=pipeline,
        num_cameras=num_cameras,
        enable_fault_tolerance=True
    )

    # Register task handler
    processor.register_task_handler('process_frame', process_voxel_frame)

    # Start processing
    processor.start()
    time.sleep(2)  # Wait for workers to initialize

    # Display cluster status
    cluster_status = cluster.get_cluster_status()
    logger.info(f"\nCluster Status:")
    logger.info(f"  Total Nodes: {cluster_status['total_nodes']}")
    logger.info(f"  Online Nodes: {cluster_status['online_nodes']}")
    logger.info(f"  Total GPUs: {cluster_status['total_gpus']}")

    # Allocate cameras to nodes
    logger.info("\n[4/5] Allocating cameras to cluster nodes...")
    camera_allocation = cluster.allocate_cameras(num_cameras)

    for camera_id, node_id in camera_allocation.items():
        logger.info(f"  Camera {camera_id} -> Node {node_id}")

    # Submit processing tasks
    logger.info(f"\n[5/5] Submitting {num_cameras * frames_per_camera} processing tasks...")
    start_time = time.time()

    task_ids = []

    for camera_id in range(num_cameras):
        for frame_id in range(frames_per_camera):
            # Generate synthetic frame
            frame, metadata = generate_synthetic_frame(camera_id, frame_id)

            # Submit frame for processing
            task_id = processor.submit_camera_frame(camera_id, frame, metadata)
            task_ids.append(task_id)

            logger.debug(f"Submitted frame {frame_id} from camera {camera_id}")

    logger.info(f"Submitted {len(task_ids)} tasks")

    # Wait for tasks to complete
    logger.info("\nWaiting for tasks to complete...")
    completed = 0

    for task_id in task_ids:
        result = processor.wait_for_task(task_id, timeout=30.0)
        if result:
            completed += 1
            if completed % 10 == 0:
                logger.info(f"  Progress: {completed}/{len(task_ids)} tasks completed")

    total_time = time.time() - start_time

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("Processing Results")
    logger.info("=" * 80)

    stats = processor.get_statistics()

    logger.info(f"\nTask Statistics:")
    logger.info(f"  Tasks Submitted: {stats['tasks_submitted']}")
    logger.info(f"  Tasks Completed: {stats['tasks_completed']}")
    logger.info(f"  Tasks Failed: {stats['tasks_failed']}")
    logger.info(f"  Success Rate: {stats['success_rate']*100:.1f}%")

    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Total Processing Time: {total_time:.2f}s")
    logger.info(f"  Average Task Time: {stats['avg_execution_time']*1000:.2f}ms")
    logger.info(f"  Throughput: {stats['tasks_completed']/total_time:.2f} tasks/sec")
    logger.info(f"  Frames Per Second: {stats['tasks_completed']/total_time:.2f} fps")

    logger.info(f"\nWorker Statistics:")
    logger.info(f"  Total Workers: {stats['total_workers']}")
    logger.info(f"  Idle Workers: {stats['idle_workers']}")
    logger.info(f"  Busy Workers: {stats['busy_workers']}")
    logger.info(f"  Error Workers: {stats['error_workers']}")

    logger.info(f"\nReliability Metrics:")
    logger.info(f"  Failovers: {stats['failovers']}")
    logger.info(f"  Load Imbalances: {stats['load_imbalances']}")

    # Pipeline statistics
    pipeline_stats = stats['pipeline']
    logger.info(f"\nPipeline Statistics:")
    logger.info(f"  Frames Processed: {pipeline_stats['frames_processed']}")
    logger.info(f"  Bytes Transferred: {pipeline_stats['bytes_transferred']/1e9:.2f} GB")
    logger.info(f"  Average Transfer Time: {pipeline_stats['avg_transfer_time_ms']:.2f}ms")
    logger.info(f"  Zero-Copy Ratio: {pipeline_stats['zero_copy_ratio']*100:.1f}%")

    # System health
    health = processor.get_system_health()
    logger.info(f"\nSystem Health:")
    logger.info(f"  Status: {health['status'].upper()}")
    logger.info(f"  Active Workers: {health['active_workers']}")
    logger.info(f"  Average Latency: {health['avg_latency_ms']:.2f}ms")

    # Cleanup
    logger.info("\nShutting down...")
    processor.stop()
    cluster.stop()
    pipeline.cleanup()

    logger.info("Demo completed successfully!")


if __name__ == '__main__':
    main()
