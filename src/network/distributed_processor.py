"""
Distributed Processing System
Handles task distribution across multiple GPUs, load balancing, worker management,
and fault tolerance with automatic failover
"""

import numpy as np
import threading
import time
import queue
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
from enum import Enum
from collections import defaultdict, deque
import multiprocessing as mp

from .cluster_config import ClusterConfig, NodeStatus, GPUInfo
from .data_pipeline import DataPipeline, FrameMetadata, RingBuffer

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerStatus(Enum):
    """Worker status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class Task:
    """Distributed task representation"""
    task_id: str
    task_type: str
    camera_id: int
    frame_ids: List[int]
    input_data: Any
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    assigned_gpu: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    submitted_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds"""
        if self.start_time and self.completion_time:
            return self.completion_time - self.start_time
        return None


@dataclass
class WorkerNode:
    """Worker node for task execution"""
    worker_id: str
    node_id: str
    gpu_id: int
    status: WorkerStatus
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)


class LoadBalancer:
    """
    Load balancer for distributing tasks across workers
    Implements weighted round-robin and least-loaded strategies
    """

    def __init__(self, strategy: str = "least_loaded"):
        """
        Initialize load balancer

        Args:
            strategy: Load balancing strategy ("round_robin", "least_loaded", "weighted")
        """
        self.strategy = strategy
        self.worker_loads: Dict[str, float] = {}  # Worker ID -> load score
        self.worker_performance: Dict[str, float] = {}  # Worker ID -> avg execution time
        self.round_robin_index = 0

    def select_worker(self, workers: List[WorkerNode], task: Task) -> Optional[str]:
        """
        Select best worker for a task

        Args:
            workers: Available workers
            task: Task to assign

        Returns:
            Worker ID or None
        """
        available = [w for w in workers if w.status == WorkerStatus.IDLE]

        if not available:
            return None

        if self.strategy == "round_robin":
            return self._round_robin(available)
        elif self.strategy == "least_loaded":
            return self._least_loaded(available)
        elif self.strategy == "weighted":
            return self._weighted(available, task)
        else:
            return available[0].worker_id

    def _round_robin(self, workers: List[WorkerNode]) -> str:
        """Round-robin selection"""
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker.worker_id

    def _least_loaded(self, workers: List[WorkerNode]) -> str:
        """Select least loaded worker"""
        # Score based on current load
        scores = []
        for worker in workers:
            load = self.worker_loads.get(worker.worker_id, 0.0)
            scores.append((load, worker.worker_id))

        scores.sort()
        return scores[0][1]

    def _weighted(self, workers: List[WorkerNode], task: Task) -> str:
        """Weighted selection based on performance history"""
        scores = []

        for worker in workers:
            # Base score from load
            load = self.worker_loads.get(worker.worker_id, 0.0)

            # Performance factor
            avg_time = self.worker_performance.get(worker.worker_id, 1.0)
            performance_factor = 1.0 / max(avg_time, 0.1)

            # Priority factor
            priority_factor = task.priority / 10.0

            # Combined score (lower is better)
            score = load - performance_factor + priority_factor

            scores.append((score, worker.worker_id))

        scores.sort()
        return scores[0][1]

    def update_worker_load(self, worker_id: str, load: float):
        """Update worker load score"""
        self.worker_loads[worker_id] = load

    def update_worker_performance(self, worker_id: str, execution_time: float):
        """Update worker performance metrics"""
        if worker_id not in self.worker_performance:
            self.worker_performance[worker_id] = execution_time
        else:
            # Exponential moving average
            alpha = 0.3
            self.worker_performance[worker_id] = (
                alpha * execution_time +
                (1 - alpha) * self.worker_performance[worker_id]
            )


class TaskScheduler:
    """
    Task scheduler with priority queue and dependency resolution
    """

    def __init__(self):
        self.task_queue: List[Task] = []
        self.pending_tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        self.lock = threading.Lock()

    def submit_task(self, task: Task):
        """Submit a task to the scheduler"""
        with self.lock:
            self.pending_tasks[task.task_id] = task
            task.status = TaskStatus.QUEUED
            self._insert_task(task)

    def _insert_task(self, task: Task):
        """Insert task into priority queue"""
        # Insert in priority order (higher priority first)
        inserted = False
        for i, t in enumerate(self.task_queue):
            if task.priority > t.priority:
                self.task_queue.insert(i, task)
                inserted = True
                break

        if not inserted:
            self.task_queue.append(task)

    def get_next_task(self) -> Optional[Task]:
        """Get next task ready for execution"""
        with self.lock:
            for i, task in enumerate(self.task_queue):
                # Check if dependencies are met
                if self._dependencies_met(task):
                    self.task_queue.pop(i)
                    task.status = TaskStatus.RUNNING
                    task.start_time = time.time()
                    return task

            return None

    def _dependencies_met(self, task: Task) -> bool:
        """Check if task dependencies are completed"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True

    def complete_task(self, task_id: str, result: Any = None, error: str = None):
        """Mark task as completed"""
        with self.lock:
            if task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                task.completion_time = time.time()

                if error:
                    task.status = TaskStatus.FAILED
                    task.error = error
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    self.completed_tasks.add(task_id)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status"""
        with self.lock:
            if task_id in self.pending_tasks:
                return self.pending_tasks[task_id].status
            return None

    def get_queue_size(self) -> int:
        """Get number of queued tasks"""
        with self.lock:
            return len(self.task_queue)


class DistributedProcessor:
    """
    Main distributed processing coordinator
    Manages workers, schedules tasks, handles failures
    """

    def __init__(self,
                 cluster_config: ClusterConfig,
                 data_pipeline: DataPipeline,
                 num_cameras: int = 10,
                 enable_fault_tolerance: bool = True):
        """
        Initialize distributed processor

        Args:
            cluster_config: Cluster configuration manager
            data_pipeline: Data pipeline for frame management
            num_cameras: Number of camera pairs
            enable_fault_tolerance: Enable automatic failover
        """
        self.cluster = cluster_config
        self.pipeline = data_pipeline
        self.num_cameras = num_cameras
        self.enable_fault_tolerance = enable_fault_tolerance

        # Task management
        self.scheduler = TaskScheduler()
        self.load_balancer = LoadBalancer(strategy="weighted")

        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.task_registry: Dict[str, Task] = {}

        # Task handlers
        self.task_handlers: Dict[str, Callable] = {}

        # Camera stream assignments
        self.camera_workers: Dict[int, List[str]] = defaultdict(list)

        # Performance metrics
        self.metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'failovers': 0,
            'load_imbalances': 0
        }

        # Control
        self.running = False
        self.dispatcher_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()

        logger.info(f"Distributed processor initialized for {num_cameras} cameras")

    def register_task_handler(self, task_type: str, handler: Callable):
        """
        Register a task handler function

        Args:
            task_type: Type of task
            handler: Handler function (task) -> result
        """
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    def start(self):
        """Start distributed processing"""
        self.running = True

        # Create workers for each GPU in the cluster
        self._create_workers()

        # Start dispatcher and monitor threads
        self.dispatcher_thread = threading.Thread(target=self._dispatcher_loop, daemon=True)
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)

        self.dispatcher_thread.start()
        self.monitor_thread.start()

        logger.info("Distributed processor started")

    def stop(self):
        """Stop distributed processing"""
        self.running = False

        # Stop all workers
        for worker_id, worker in self.workers.items():
            worker.status = WorkerStatus.STOPPED

        # Wait for threads
        if self.dispatcher_thread:
            self.dispatcher_thread.join(timeout=5.0)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("Distributed processor stopped")

    def _create_workers(self):
        """Create worker nodes for available GPUs"""
        with self.lock:
            for node_id, node in self.cluster.nodes.items():
                if node.status != NodeStatus.ONLINE:
                    continue

                for gpu in node.resources.gpus:
                    worker_id = f"{node_id}_gpu{gpu.gpu_id}"

                    worker = WorkerNode(
                        worker_id=worker_id,
                        node_id=node_id,
                        gpu_id=gpu.gpu_id,
                        status=WorkerStatus.IDLE
                    )

                    self.workers[worker_id] = worker

                    # Start worker thread
                    thread = threading.Thread(
                        target=self._worker_loop,
                        args=(worker_id,),
                        daemon=True
                    )
                    thread.start()
                    self.worker_threads[worker_id] = thread

                    logger.info(f"Created worker: {worker_id}")

    def _worker_loop(self, worker_id: str):
        """Worker execution loop"""
        while self.running:
            try:
                worker = self.workers.get(worker_id)
                if not worker or worker.status == WorkerStatus.STOPPED:
                    break

                # Process assigned task
                if worker.current_task:
                    task = self.task_registry.get(worker.current_task)

                    if task:
                        result = self._execute_task(task)

                        if result is not None:
                            self.scheduler.complete_task(task.task_id, result=result)
                            worker.tasks_completed += 1

                            execution_time = task.execution_time()
                            if execution_time:
                                worker.total_execution_time += execution_time
                                self.load_balancer.update_worker_performance(worker_id, execution_time)

                            with self.lock:
                                self.metrics['tasks_completed'] += 1
                                self.metrics['total_execution_time'] += execution_time or 0

                        else:
                            # Task failed
                            error = task.error or "Unknown error"
                            self.scheduler.complete_task(task.task_id, error=error)
                            worker.tasks_failed += 1

                            with self.lock:
                                self.metrics['tasks_failed'] += 1

                            # Retry if possible
                            if task.retry_count < task.max_retries:
                                task.retry_count += 1
                                task.status = TaskStatus.PENDING
                                self.submit_task(task)
                                logger.warning(f"Retrying task {task.task_id} ({task.retry_count}/{task.max_retries})")

                    worker.current_task = None
                    worker.status = WorkerStatus.IDLE

                # Update heartbeat
                worker.last_heartbeat = time.time()

                # Small sleep to prevent busy waiting
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                worker.status = WorkerStatus.ERROR
                time.sleep(1.0)

    def _execute_task(self, task: Task) -> Optional[Any]:
        """
        Execute a task

        Args:
            task: Task to execute

        Returns:
            Task result or None if failed
        """
        try:
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                logger.error(f"No handler for task type: {task.task_type}")
                task.error = f"No handler for task type: {task.task_type}"
                return None

            # Execute handler
            result = handler(task)

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            task.error = str(e)
            return None

    def _dispatcher_loop(self):
        """Task dispatcher loop"""
        while self.running:
            try:
                # Get next task
                task = self.scheduler.get_next_task()

                if task:
                    # Find available worker
                    workers_list = list(self.workers.values())
                    worker_id = self.load_balancer.select_worker(workers_list, task)

                    if worker_id:
                        worker = self.workers[worker_id]

                        # Assign task to worker
                        task.assigned_worker = worker_id
                        task.assigned_gpu = worker.gpu_id
                        worker.current_task = task.task_id
                        worker.status = WorkerStatus.BUSY

                        # Update load balancer
                        load = 1.0 if worker.status == WorkerStatus.BUSY else 0.0
                        self.load_balancer.update_worker_load(worker_id, load)

                        logger.debug(f"Assigned task {task.task_id} to worker {worker_id}")
                    else:
                        # No workers available, requeue
                        self.scheduler.submit_task(task)

                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Dispatcher error: {e}")
                time.sleep(1.0)

    def _monitor_loop(self):
        """System monitoring and health check loop"""
        while self.running:
            try:
                current_time = time.time()

                with self.lock:
                    # Check for failed workers
                    failed_workers = []

                    for worker_id, worker in self.workers.items():
                        time_since_heartbeat = current_time - worker.last_heartbeat

                        # Worker timeout (10 seconds)
                        if time_since_heartbeat > 10.0 and worker.status != WorkerStatus.STOPPED:
                            logger.warning(f"Worker {worker_id} timed out")
                            worker.status = WorkerStatus.ERROR
                            failed_workers.append(worker_id)

                    # Handle worker failures
                    if failed_workers and self.enable_fault_tolerance:
                        self._handle_worker_failures(failed_workers)

                    # Load balancing check
                    self._check_load_balance()

                time.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5.0)

    def _handle_worker_failures(self, failed_worker_ids: List[str]):
        """
        Handle worker failures with automatic failover

        Args:
            failed_worker_ids: List of failed worker IDs
        """
        logger.info(f"Handling failures for workers: {failed_worker_ids}")

        for worker_id in failed_worker_ids:
            worker = self.workers.get(worker_id)
            if not worker:
                continue

            # Reassign current task
            if worker.current_task:
                task = self.task_registry.get(worker.current_task)
                if task:
                    # Reset task status
                    task.status = TaskStatus.PENDING
                    task.assigned_worker = None
                    task.assigned_gpu = None

                    # Resubmit
                    self.submit_task(task)

                    logger.info(f"Reassigned task {task.task_id} from failed worker {worker_id}")

                worker.current_task = None

            # Try to restart worker on same GPU
            node = self.cluster.nodes.get(worker.node_id)
            if node and node.status == NodeStatus.ONLINE:
                worker.status = WorkerStatus.IDLE
                worker.last_heartbeat = time.time()
                logger.info(f"Restarted worker {worker_id}")
            else:
                # Node is offline, remove worker
                worker.status = WorkerStatus.STOPPED
                logger.warning(f"Worker {worker_id} stopped (node offline)")

            with self.lock:
                self.metrics['failovers'] += 1

    def _check_load_balance(self):
        """Check and report load imbalance"""
        if not self.workers:
            return

        # Calculate load distribution
        loads = []
        for worker in self.workers.values():
            if worker.status != WorkerStatus.STOPPED:
                load = 1.0 if worker.status == WorkerStatus.BUSY else 0.0
                loads.append(load)

        if not loads:
            return

        avg_load = np.mean(loads)
        std_load = np.std(loads)

        # Significant imbalance threshold
        if std_load > 0.3 and avg_load > 0.5:
            with self.lock:
                self.metrics['load_imbalances'] += 1
            logger.warning(f"Load imbalance detected: avg={avg_load:.2f}, std={std_load:.2f}")

    def submit_task(self, task: Task):
        """
        Submit a task for execution

        Args:
            task: Task to submit
        """
        with self.lock:
            self.task_registry[task.task_id] = task
            self.metrics['tasks_submitted'] += 1

        self.scheduler.submit_task(task)
        logger.debug(f"Task submitted: {task.task_id}")

    def submit_camera_frame(self, camera_id: int, frame: np.ndarray, metadata: FrameMetadata) -> str:
        """
        Submit a camera frame for processing

        Args:
            camera_id: Camera ID
            frame: Frame data
            metadata: Frame metadata

        Returns:
            Task ID
        """
        # Write frame to pipeline
        self.pipeline.write_frame(camera_id, frame, metadata)

        # Create processing task
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type="process_frame",
            camera_id=camera_id,
            frame_ids=[metadata.frame_id],
            input_data={'frame': frame, 'metadata': metadata},
            priority=5
        )

        self.submit_task(task)

        return task_id

    def wait_for_task(self, task_id: str, timeout: float = 30.0) -> Optional[Any]:
        """
        Wait for task completion

        Args:
            task_id: Task ID
            timeout: Timeout in seconds

        Returns:
            Task result or None
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            task = self.task_registry.get(task_id)
            if task:
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    logger.error(f"Task {task_id} failed: {task.error}")
                    return None

            time.sleep(0.01)

        logger.warning(f"Task {task_id} timeout after {timeout}s")
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self.lock:
            stats = dict(self.metrics)

            # Worker statistics
            worker_stats = {
                'total_workers': len(self.workers),
                'idle_workers': sum(1 for w in self.workers.values() if w.status == WorkerStatus.IDLE),
                'busy_workers': sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY),
                'error_workers': sum(1 for w in self.workers.values() if w.status == WorkerStatus.ERROR),
            }

            stats.update(worker_stats)

            # Queue statistics
            stats['queue_size'] = self.scheduler.get_queue_size()

            # Performance metrics
            if stats['tasks_completed'] > 0:
                stats['avg_execution_time'] = stats['total_execution_time'] / stats['tasks_completed']
                stats['success_rate'] = stats['tasks_completed'] / (stats['tasks_completed'] + stats['tasks_failed'])
            else:
                stats['avg_execution_time'] = 0.0
                stats['success_rate'] = 1.0

            # Cluster statistics
            stats['cluster'] = self.cluster.get_cluster_status()

            # Pipeline statistics
            stats['pipeline'] = self.pipeline.get_statistics()

            return stats

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        stats = self.get_statistics()

        # Determine health status
        health = "healthy"

        if stats['error_workers'] > 0:
            health = "degraded"

        if stats['idle_workers'] == 0 and stats['queue_size'] > 100:
            health = "overloaded"

        if stats['cluster']['online_nodes'] == 0:
            health = "critical"

        return {
            'status': health,
            'online_nodes': stats['cluster']['online_nodes'],
            'total_gpus': stats['cluster']['total_gpus'],
            'active_workers': stats['busy_workers'] + stats['idle_workers'],
            'tasks_queued': stats['queue_size'],
            'success_rate': stats['success_rate'],
            'avg_latency_ms': stats.get('avg_execution_time', 0) * 1000,
            'failover_count': stats['failovers']
        }
