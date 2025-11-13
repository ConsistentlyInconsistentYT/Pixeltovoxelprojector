"""
Cluster Configuration and Management
Handles node discovery, registration, resource allocation, and performance monitoring
"""

import json
import socket
import threading
import time
import psutil
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum
import logging
import netifaces
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkType(Enum):
    """Network interface types"""
    INFINIBAND = "ib"
    TEN_GBE = "10gbe"
    ETHERNET = "eth"
    LOOPBACK = "lo"


class NodeStatus(Enum):
    """Node status enumeration"""
    DISCOVERING = "discovering"
    ONLINE = "online"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class NetworkInterface:
    """Network interface information"""
    name: str
    type: NetworkType
    ip_address: str
    bandwidth_gbps: float
    latency_ms: float
    mtu: int
    rdma_capable: bool

    def to_dict(self):
        return {
            'name': self.name,
            'type': self.type.value,
            'ip_address': self.ip_address,
            'bandwidth_gbps': self.bandwidth_gbps,
            'latency_ms': self.latency_ms,
            'mtu': self.mtu,
            'rdma_capable': self.rdma_capable
        }


@dataclass
class GPUInfo:
    """GPU resource information"""
    gpu_id: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    compute_capability: Tuple[int, int]
    utilization_percent: float
    temperature_c: float
    power_usage_w: float

    def to_dict(self):
        return asdict(self)


@dataclass
class NodeResources:
    """Node hardware resources"""
    cpu_count: int
    cpu_usage_percent: float
    memory_total_gb: float
    memory_available_gb: float
    gpus: List[GPUInfo]
    network_interfaces: List[NetworkInterface]
    disk_io_mbps: float

    def to_dict(self):
        return {
            'cpu_count': self.cpu_count,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_total_gb': self.memory_total_gb,
            'memory_available_gb': self.memory_available_gb,
            'gpus': [gpu.to_dict() for gpu in self.gpus],
            'network_interfaces': [iface.to_dict() for iface in self.network_interfaces],
            'disk_io_mbps': self.disk_io_mbps
        }


@dataclass
class ClusterNode:
    """Cluster node representation"""
    node_id: str
    hostname: str
    primary_ip: str
    status: NodeStatus
    resources: NodeResources
    last_heartbeat: float
    camera_assignments: List[int] = field(default_factory=list)
    gpu_assignments: Dict[int, List[str]] = field(default_factory=dict)  # GPU ID -> task IDs
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'primary_ip': self.primary_ip,
            'status': self.status.value,
            'resources': self.resources.to_dict(),
            'last_heartbeat': self.last_heartbeat,
            'camera_assignments': self.camera_assignments,
            'gpu_assignments': self.gpu_assignments,
            'performance_metrics': self.performance_metrics
        }


class ClusterConfig:
    """
    Cluster configuration and management system
    Handles node discovery, resource allocation, and network topology optimization
    """

    def __init__(self,
                 discovery_port: int = 9999,
                 heartbeat_interval: float = 1.0,
                 heartbeat_timeout: float = 5.0,
                 enable_rdma: bool = True):
        """
        Initialize cluster configuration

        Args:
            discovery_port: Port for node discovery broadcasts
            heartbeat_interval: Interval between heartbeats (seconds)
            heartbeat_timeout: Time before node considered offline (seconds)
            enable_rdma: Enable RDMA support for InfiniBand
        """
        self.discovery_port = discovery_port
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.enable_rdma = enable_rdma

        # Node registry
        self.nodes: Dict[str, ClusterNode] = {}
        self.local_node_id: Optional[str] = None
        self.is_master = False

        # Network topology
        self.latency_matrix: Dict[Tuple[str, str], float] = {}
        self.bandwidth_matrix: Dict[Tuple[str, str], float] = {}

        # Resource allocation
        self.camera_to_node: Dict[int, str] = {}  # Camera ID -> Node ID
        self.task_to_gpu: Dict[str, Tuple[str, int]] = {}  # Task ID -> (Node ID, GPU ID)

        # Performance monitoring
        self.metrics_history: Dict[str, List[Tuple[float, Dict]]] = defaultdict(list)
        self.max_history_size = 1000

        # Threading
        self.discovery_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.RLock()

        logger.info("Cluster configuration initialized")

    def start(self, is_master: bool = False):
        """Start cluster services"""
        self.is_master = is_master
        self.running = True

        # Register local node
        self._register_local_node()

        # Start service threads
        self.discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)

        self.discovery_thread.start()
        self.heartbeat_thread.start()
        self.monitoring_thread.start()

        logger.info(f"Cluster services started (master={is_master})")

    def stop(self):
        """Stop cluster services"""
        self.running = False

        if self.discovery_thread:
            self.discovery_thread.join(timeout=2.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2.0)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

        logger.info("Cluster services stopped")

    def _register_local_node(self):
        """Register the local node in the cluster"""
        hostname = socket.gethostname()
        self.local_node_id = f"{hostname}_{int(time.time())}"

        resources = self._detect_local_resources()
        primary_ip = self._get_primary_ip()

        local_node = ClusterNode(
            node_id=self.local_node_id,
            hostname=hostname,
            primary_ip=primary_ip,
            status=NodeStatus.ONLINE,
            resources=resources,
            last_heartbeat=time.time()
        )

        with self.lock:
            self.nodes[self.local_node_id] = local_node

        logger.info(f"Local node registered: {self.local_node_id} @ {primary_ip}")

    def _detect_local_resources(self) -> NodeResources:
        """Detect local hardware resources"""
        # CPU and memory
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        memory_total_gb = mem.total / (1024**3)
        memory_available_gb = mem.available / (1024**3)

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_mbps = (disk_io.read_bytes + disk_io.write_bytes) / (1024**2) if disk_io else 0

        # GPU detection
        gpus = self._detect_gpus()

        # Network interfaces
        network_interfaces = self._detect_network_interfaces()

        return NodeResources(
            cpu_count=cpu_count,
            cpu_usage_percent=cpu_usage,
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            gpus=gpus,
            network_interfaces=network_interfaces,
            disk_io_mbps=disk_io_mbps
        )

    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect available GPUs"""
        gpus = []

        try:
            import pynvml
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total_mb = mem_info.total // (1024**2)
                memory_free_mb = mem_info.free // (1024**2)

                compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_percent = utilization.gpu

                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0.0

                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    power_usage = 0.0

                gpus.append(GPUInfo(
                    gpu_id=i,
                    name=name,
                    memory_total_mb=memory_total_mb,
                    memory_free_mb=memory_free_mb,
                    compute_capability=compute_cap,
                    utilization_percent=utilization_percent,
                    temperature_c=temperature,
                    power_usage_w=power_usage
                ))

            pynvml.nvmlShutdown()

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")

        return gpus

    def _detect_network_interfaces(self) -> List[NetworkInterface]:
        """Detect network interfaces"""
        interfaces = []

        for iface_name in netifaces.interfaces():
            if iface_name == 'lo':
                continue

            addrs = netifaces.ifaddresses(iface_name)
            if netifaces.AF_INET not in addrs:
                continue

            ip_address = addrs[netifaces.AF_INET][0]['addr']

            # Determine interface type and capabilities
            if iface_name.startswith('ib'):
                iface_type = NetworkType.INFINIBAND
                bandwidth_gbps = 100.0  # InfiniBand EDR
                latency_ms = 0.5
                mtu = 65520
                rdma_capable = True
            elif iface_name.startswith('eth') or iface_name.startswith('en'):
                # Check if 10GbE
                iface_type = NetworkType.TEN_GBE
                bandwidth_gbps = 10.0
                latency_ms = 1.0
                mtu = 9000  # Jumbo frames
                rdma_capable = False
            else:
                iface_type = NetworkType.ETHERNET
                bandwidth_gbps = 1.0
                latency_ms = 2.0
                mtu = 1500
                rdma_capable = False

            interfaces.append(NetworkInterface(
                name=iface_name,
                type=iface_type,
                ip_address=ip_address,
                bandwidth_gbps=bandwidth_gbps,
                latency_ms=latency_ms,
                mtu=mtu,
                rdma_capable=rdma_capable and self.enable_rdma
            ))

        return interfaces

    def _get_primary_ip(self) -> str:
        """Get primary IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def _discovery_loop(self):
        """Node discovery broadcast loop"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.discovery_port))
        sock.settimeout(1.0)

        while self.running:
            try:
                # Broadcast presence
                if self.local_node_id and self.local_node_id in self.nodes:
                    node_info = self.nodes[self.local_node_id].to_dict()
                    message = json.dumps({
                        'type': 'discovery',
                        'node': node_info
                    }).encode('utf-8')

                    sock.sendto(message, ('<broadcast>', self.discovery_port))

                # Listen for other nodes
                try:
                    data, addr = sock.recvfrom(65535)
                    self._handle_discovery_message(data, addr)
                except socket.timeout:
                    pass

            except Exception as e:
                logger.error(f"Discovery error: {e}")

            time.sleep(self.heartbeat_interval)

    def _handle_discovery_message(self, data: bytes, addr: Tuple[str, int]):
        """Handle incoming discovery message"""
        try:
            message = json.loads(data.decode('utf-8'))

            if message['type'] == 'discovery':
                node_data = message['node']
                node_id = node_data['node_id']

                # Don't process our own messages
                if node_id == self.local_node_id:
                    return

                # Reconstruct node object
                resources_data = node_data['resources']
                gpus = [GPUInfo(**gpu) for gpu in resources_data['gpus']]

                network_interfaces = []
                for iface_data in resources_data['network_interfaces']:
                    iface_data['type'] = NetworkType(iface_data['type'])
                    network_interfaces.append(NetworkInterface(**iface_data))

                resources = NodeResources(
                    cpu_count=resources_data['cpu_count'],
                    cpu_usage_percent=resources_data['cpu_usage_percent'],
                    memory_total_gb=resources_data['memory_total_gb'],
                    memory_available_gb=resources_data['memory_available_gb'],
                    gpus=gpus,
                    network_interfaces=network_interfaces,
                    disk_io_mbps=resources_data['disk_io_mbps']
                )

                node = ClusterNode(
                    node_id=node_id,
                    hostname=node_data['hostname'],
                    primary_ip=node_data['primary_ip'],
                    status=NodeStatus(node_data['status']),
                    resources=resources,
                    last_heartbeat=time.time(),
                    camera_assignments=node_data.get('camera_assignments', []),
                    gpu_assignments=node_data.get('gpu_assignments', {}),
                    performance_metrics=node_data.get('performance_metrics', {})
                )

                with self.lock:
                    if node_id not in self.nodes:
                        logger.info(f"Discovered new node: {node_id} @ {node.primary_ip}")
                    self.nodes[node_id] = node

                    # Measure network latency
                    self._measure_latency(node_id, node.primary_ip)

        except Exception as e:
            logger.error(f"Failed to handle discovery message: {e}")

    def _heartbeat_loop(self):
        """Monitor node heartbeats"""
        while self.running:
            current_time = time.time()

            with self.lock:
                offline_nodes = []

                for node_id, node in self.nodes.items():
                    if node_id == self.local_node_id:
                        node.last_heartbeat = current_time
                        continue

                    time_since_heartbeat = current_time - node.last_heartbeat

                    if time_since_heartbeat > self.heartbeat_timeout:
                        if node.status != NodeStatus.OFFLINE:
                            logger.warning(f"Node {node_id} went offline (last heartbeat: {time_since_heartbeat:.1f}s ago)")
                            node.status = NodeStatus.OFFLINE
                            offline_nodes.append(node_id)

                # Trigger failover for offline nodes
                if offline_nodes and self.is_master:
                    self._handle_node_failures(offline_nodes)

            time.sleep(self.heartbeat_interval)

    def _monitoring_loop(self):
        """Performance monitoring loop"""
        while self.running:
            try:
                # Update local node resources
                if self.local_node_id:
                    resources = self._detect_local_resources()

                    with self.lock:
                        if self.local_node_id in self.nodes:
                            self.nodes[self.local_node_id].resources = resources

                            # Record metrics
                            metrics = {
                                'cpu_usage': resources.cpu_usage_percent,
                                'memory_available_gb': resources.memory_available_gb,
                                'gpu_utilization': np.mean([gpu.utilization_percent for gpu in resources.gpus]) if resources.gpus else 0,
                                'disk_io_mbps': resources.disk_io_mbps
                            }

                            self.nodes[self.local_node_id].performance_metrics = metrics

                            # Store in history
                            self.metrics_history[self.local_node_id].append((time.time(), metrics))
                            if len(self.metrics_history[self.local_node_id]) > self.max_history_size:
                                self.metrics_history[self.local_node_id].pop(0)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            time.sleep(5.0)  # Monitor every 5 seconds

    def _measure_latency(self, node_id: str, ip_address: str, port: int = 9998):
        """Measure network latency to another node"""
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((ip_address, port))
            sock.close()
            latency_ms = (time.time() - start_time) * 1000

            with self.lock:
                self.latency_matrix[(self.local_node_id, node_id)] = latency_ms

        except Exception:
            # If connection fails, estimate based on network type
            if self.local_node_id in self.nodes and node_id in self.nodes:
                local_ifaces = self.nodes[self.local_node_id].resources.network_interfaces
                if any(iface.type == NetworkType.INFINIBAND for iface in local_ifaces):
                    estimated_latency = 0.5
                else:
                    estimated_latency = 2.0

                with self.lock:
                    self.latency_matrix[(self.local_node_id, node_id)] = estimated_latency

    def _handle_node_failures(self, failed_node_ids: List[str]):
        """Handle node failures with automatic failover"""
        logger.info(f"Initiating failover for nodes: {failed_node_ids}")

        for failed_node_id in failed_node_ids:
            if failed_node_id not in self.nodes:
                continue

            failed_node = self.nodes[failed_node_id]

            # Reassign cameras
            for camera_id in failed_node.camera_assignments:
                new_node_id = self._find_best_node_for_camera(camera_id, exclude=[failed_node_id])
                if new_node_id:
                    self.camera_to_node[camera_id] = new_node_id
                    self.nodes[new_node_id].camera_assignments.append(camera_id)
                    logger.info(f"Reassigned camera {camera_id}: {failed_node_id} -> {new_node_id}")

            # Reassign GPU tasks
            for gpu_id, task_ids in failed_node.gpu_assignments.items():
                for task_id in task_ids:
                    new_assignment = self._find_best_gpu_for_task(task_id, exclude_nodes=[failed_node_id])
                    if new_assignment:
                        new_node_id, new_gpu_id = new_assignment
                        self.task_to_gpu[task_id] = (new_node_id, new_gpu_id)

                        if new_gpu_id not in self.nodes[new_node_id].gpu_assignments:
                            self.nodes[new_node_id].gpu_assignments[new_gpu_id] = []
                        self.nodes[new_node_id].gpu_assignments[new_gpu_id].append(task_id)

                        logger.info(f"Reassigned task {task_id}: {failed_node_id}/GPU{gpu_id} -> {new_node_id}/GPU{new_gpu_id}")

            # Clear assignments from failed node
            failed_node.camera_assignments.clear()
            failed_node.gpu_assignments.clear()

    def _find_best_node_for_camera(self, camera_id: int, exclude: List[str] = None) -> Optional[str]:
        """Find the best node to handle a camera stream"""
        exclude = exclude or []

        with self.lock:
            available_nodes = [
                (node_id, node) for node_id, node in self.nodes.items()
                if node.status == NodeStatus.ONLINE and node_id not in exclude
            ]

            if not available_nodes:
                return None

            # Score nodes based on available resources and current load
            scores = []
            for node_id, node in available_nodes:
                # Prefer nodes with more available memory and lower CPU usage
                score = (
                    node.resources.memory_available_gb * 10 +
                    (100 - node.resources.cpu_usage_percent) +
                    len(node.resources.gpus) * 5 -
                    len(node.camera_assignments) * 2
                )
                scores.append((score, node_id))

            scores.sort(reverse=True)
            return scores[0][1]

    def _find_best_gpu_for_task(self, task_id: str, exclude_nodes: List[str] = None) -> Optional[Tuple[str, int]]:
        """Find the best GPU for a task"""
        exclude_nodes = exclude_nodes or []

        with self.lock:
            candidates = []

            for node_id, node in self.nodes.items():
                if node.status != NodeStatus.ONLINE or node_id in exclude_nodes:
                    continue

                for gpu in node.resources.gpus:
                    # Score based on free memory and utilization
                    score = (
                        gpu.memory_free_mb / 1000 +
                        (100 - gpu.utilization_percent) / 10
                    )

                    # Penalize if GPU already has many tasks
                    task_count = len(node.gpu_assignments.get(gpu.gpu_id, []))
                    score -= task_count * 5

                    candidates.append((score, node_id, gpu.gpu_id))

            if not candidates:
                return None

            candidates.sort(reverse=True)
            return (candidates[0][1], candidates[0][2])

    def allocate_cameras(self, num_cameras: int) -> Dict[int, str]:
        """
        Allocate cameras across cluster nodes

        Args:
            num_cameras: Number of camera pairs to allocate

        Returns:
            Mapping of camera ID to node ID
        """
        allocation = {}

        with self.lock:
            for camera_id in range(num_cameras):
                node_id = self._find_best_node_for_camera(camera_id)
                if node_id:
                    allocation[camera_id] = node_id
                    self.camera_to_node[camera_id] = node_id
                    self.nodes[node_id].camera_assignments.append(camera_id)
                else:
                    logger.error(f"Failed to allocate camera {camera_id}")

        logger.info(f"Allocated {len(allocation)}/{num_cameras} cameras")
        return allocation

    def get_cluster_status(self) -> Dict:
        """Get current cluster status"""
        with self.lock:
            return {
                'total_nodes': len(self.nodes),
                'online_nodes': sum(1 for n in self.nodes.values() if n.status == NodeStatus.ONLINE),
                'total_gpus': sum(len(n.resources.gpus) for n in self.nodes.values()),
                'camera_allocations': dict(self.camera_to_node),
                'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()}
            }

    def optimize_network_topology(self):
        """Optimize network topology based on latency measurements"""
        logger.info("Optimizing network topology...")

        with self.lock:
            # Build latency graph
            nodes = list(self.nodes.keys())
            n = len(nodes)

            if n < 2:
                return

            # Create adjacency matrix
            latency_matrix = np.full((n, n), 999.0)

            for i, node_i in enumerate(nodes):
                latency_matrix[i, i] = 0.0
                for j, node_j in enumerate(nodes):
                    if i != j:
                        key = (node_i, node_j)
                        if key in self.latency_matrix:
                            latency_matrix[i, j] = self.latency_matrix[key]

            # Find optimal routing using Floyd-Warshall
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if latency_matrix[i, k] + latency_matrix[k, j] < latency_matrix[i, j]:
                            latency_matrix[i, j] = latency_matrix[i, k] + latency_matrix[k, j]

            # Update latency matrix with optimized routes
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if i != j:
                        self.latency_matrix[(node_i, node_j)] = latency_matrix[i, j]

            avg_latency = np.mean(latency_matrix[latency_matrix < 999.0])
            max_latency = np.max(latency_matrix[latency_matrix < 999.0])

            logger.info(f"Network topology optimized: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms")
