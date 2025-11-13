"""
System Monitor - Real-time Performance and Health Monitoring

Features:
- Real-time performance metrics at 10Hz
- GPU utilization and temperature monitoring
- Network bandwidth tracking
- Camera health status monitoring
- Detection accuracy tracking
- <1% performance overhead

Requirements:
- Monitor 20 cameras (10 pairs)
- Track 200+ drone targets
- 10Hz update rate
- Minimal performance impact
"""

import time
import threading
import psutil
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    """Resource health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class GPUMetrics:
    """GPU performance metrics"""
    gpu_id: int = 0
    utilization: float = 0.0  # Percentage
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature: float = 0.0  # Celsius
    power_draw: float = 0.0  # Watts
    clock_speed: int = 0  # MHz
    fan_speed: float = 0.0  # Percentage
    status: ResourceStatus = ResourceStatus.HEALTHY


@dataclass
class CPUMetrics:
    """CPU performance metrics"""
    utilization: float = 0.0  # Percentage
    per_core: List[float] = field(default_factory=list)
    frequency: float = 0.0  # MHz
    temperature: float = 0.0  # Celsius (if available)
    load_average: List[float] = field(default_factory=list)  # 1, 5, 15 min
    context_switches: int = 0
    status: ResourceStatus = ResourceStatus.HEALTHY


@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    total_mb: float = 0.0
    used_mb: float = 0.0
    available_mb: float = 0.0
    percent: float = 0.0
    swap_total_mb: float = 0.0
    swap_used_mb: float = 0.0
    swap_percent: float = 0.0
    status: ResourceStatus = ResourceStatus.HEALTHY


@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    send_rate_mbps: float = 0.0
    recv_rate_mbps: float = 0.0
    bandwidth_utilization: float = 0.0  # Percentage
    packet_loss_rate: float = 0.0
    latency_ms: float = 0.0
    interface: str = "eth0"
    status: ResourceStatus = ResourceStatus.HEALTHY


@dataclass
class CameraMetrics:
    """Camera health metrics"""
    camera_id: int
    status: str = "offline"  # offline, streaming, error
    fps: float = 0.0
    target_fps: float = 30.0
    frames_received: int = 0
    frames_dropped: int = 0
    drop_rate: float = 0.0
    latency_ms: float = 0.0
    temperature: float = 0.0
    packet_loss: float = 0.0
    health_status: ResourceStatus = ResourceStatus.OFFLINE


@dataclass
class DetectionMetrics:
    """Detection performance metrics"""
    total_detections: int = 0
    confirmed_tracks: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    avg_confidence: float = 0.0
    processing_time_ms: float = 0.0
    tracks_lost: int = 0
    occlusion_recoveries: int = 0


@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    timestamp: float = field(default_factory=time.time)

    # Hardware metrics
    cpu: CPUMetrics = field(default_factory=CPUMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    gpus: List[GPUMetrics] = field(default_factory=list)
    network: NetworkMetrics = field(default_factory=NetworkMetrics)

    # System metrics
    cameras: Dict[int, CameraMetrics] = field(default_factory=dict)
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)

    # Performance metrics
    system_fps: float = 0.0
    pipeline_latency_ms: float = 0.0
    total_throughput_mbps: float = 0.0

    # Overall health
    overall_status: ResourceStatus = ResourceStatus.HEALTHY
    active_alerts: int = 0


class SystemMonitor:
    """
    Real-time system monitoring at 10Hz with <1% overhead
    Monitors hardware, cameras, network, and detection performance
    """

    def __init__(self,
                 update_rate_hz: float = 10.0,
                 num_cameras: int = 20,
                 history_size: int = 300):  # 30 seconds at 10Hz
        """
        Initialize system monitor

        Args:
            update_rate_hz: Monitoring update rate (default 10Hz)
            num_cameras: Number of cameras to monitor
            history_size: Number of historical samples to keep
        """
        self.update_rate_hz = update_rate_hz
        self.update_interval = 1.0 / update_rate_hz
        self.num_cameras = num_cameras
        self.history_size = history_size

        # Current metrics
        self.current_metrics = SystemMetrics()

        # Historical metrics (time-series)
        self.metrics_history: deque = deque(maxlen=history_size)

        # GPU monitoring (try multiple libraries)
        self.gpu_available = False
        self.gpu_monitor = None
        self._init_gpu_monitoring()

        # Network monitoring baseline
        self.net_io_last = None
        self.net_io_timestamp = time.time()

        # Camera managers (external references)
        self.camera_manager = None
        self.tracker = None

        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.lock = threading.Lock()

        # Callbacks
        self.metric_callbacks: List[Callable[[SystemMetrics], None]] = []

        # Statistics
        self.samples_collected = 0
        self.monitoring_overhead_ms = 0.0

        logger.info(f"SystemMonitor initialized: {update_rate_hz}Hz, {num_cameras} cameras")

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring with available libraries"""
        # Try pynvml (NVIDIA)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self.gpu_monitor = 'pynvml'
            self.gpu_available = True
            logger.info(f"GPU monitoring enabled (NVIDIA): {device_count} devices")
            return
        except (ImportError, Exception) as e:
            logger.debug(f"pynvml not available: {e}")

        # Try GPUtil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_monitor = 'gputil'
                self.gpu_available = True
                logger.info(f"GPU monitoring enabled (GPUtil): {len(gpus)} devices")
                return
        except (ImportError, Exception) as e:
            logger.debug(f"GPUtil not available: {e}")

        logger.warning("GPU monitoring not available")

    def start(self):
        """Start monitoring thread"""
        if self.running:
            logger.warning("Monitor already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")

    def stop(self):
        """Stop monitoring thread"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            loop_start = time.time()

            try:
                # Collect all metrics
                metrics = self._collect_metrics()

                # Store current metrics
                with self.lock:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics)
                    self.samples_collected += 1

                # Call registered callbacks
                for callback in self.metric_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                # Calculate monitoring overhead
                loop_time = (time.time() - loop_start) * 1000.0
                self.monitoring_overhead_ms = 0.9 * self.monitoring_overhead_ms + 0.1 * loop_time

                # Sleep for remaining interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.update_interval)

    def _collect_metrics(self) -> SystemMetrics:
        """Collect all system metrics"""
        metrics = SystemMetrics()
        metrics.timestamp = time.time()

        # Collect CPU metrics
        metrics.cpu = self._collect_cpu_metrics()

        # Collect memory metrics
        metrics.memory = self._collect_memory_metrics()

        # Collect GPU metrics
        if self.gpu_available:
            metrics.gpus = self._collect_gpu_metrics()

        # Collect network metrics
        metrics.network = self._collect_network_metrics()

        # Collect camera metrics
        if self.camera_manager:
            metrics.cameras = self._collect_camera_metrics()

        # Collect detection metrics
        if self.tracker:
            metrics.detection = self._collect_detection_metrics()

        # Calculate overall system health
        metrics.overall_status = self._calculate_overall_status(metrics)

        return metrics

    def _collect_cpu_metrics(self) -> CPUMetrics:
        """Collect CPU performance metrics"""
        cpu = CPUMetrics()

        try:
            cpu.utilization = psutil.cpu_percent(interval=0)
            cpu.per_core = psutil.cpu_percent(interval=0, percpu=True)

            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu.frequency = cpu_freq.current

            load_avg = psutil.getloadavg()
            cpu.load_average = list(load_avg)

            # Try to get CPU temperature (Linux)
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu.temperature = np.mean([t.current for t in temps['coretemp']])
                elif temps:
                    first_sensor = list(temps.values())[0]
                    cpu.temperature = np.mean([t.current for t in first_sensor])
            except (AttributeError, Exception):
                pass

            # Determine status
            if cpu.utilization > 90:
                cpu.status = ResourceStatus.CRITICAL
            elif cpu.utilization > 75:
                cpu.status = ResourceStatus.WARNING
            else:
                cpu.status = ResourceStatus.HEALTHY

        except Exception as e:
            logger.error(f"CPU metrics error: {e}")
            cpu.status = ResourceStatus.OFFLINE

        return cpu

    def _collect_memory_metrics(self) -> MemoryMetrics:
        """Collect memory usage metrics"""
        mem = MemoryMetrics()

        try:
            vm = psutil.virtual_memory()
            mem.total_mb = vm.total / (1024 ** 2)
            mem.used_mb = vm.used / (1024 ** 2)
            mem.available_mb = vm.available / (1024 ** 2)
            mem.percent = vm.percent

            swap = psutil.swap_memory()
            mem.swap_total_mb = swap.total / (1024 ** 2)
            mem.swap_used_mb = swap.used / (1024 ** 2)
            mem.swap_percent = swap.percent

            # Determine status
            if mem.percent > 95:
                mem.status = ResourceStatus.CRITICAL
            elif mem.percent > 85:
                mem.status = ResourceStatus.WARNING
            else:
                mem.status = ResourceStatus.HEALTHY

        except Exception as e:
            logger.error(f"Memory metrics error: {e}")
            mem.status = ResourceStatus.OFFLINE

        return mem

    def _collect_gpu_metrics(self) -> List[GPUMetrics]:
        """Collect GPU performance metrics"""
        gpus = []

        try:
            if self.gpu_monitor == 'pynvml':
                gpus = self._collect_nvidia_metrics()
            elif self.gpu_monitor == 'gputil':
                gpus = self._collect_gputil_metrics()
        except Exception as e:
            logger.error(f"GPU metrics error: {e}")

        return gpus

    def _collect_nvidia_metrics(self) -> List[GPUMetrics]:
        """Collect NVIDIA GPU metrics using pynvml"""
        import pynvml

        gpus = []
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                gpu = GPUMetrics(gpu_id=i)

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu.utilization = util.gpu

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu.memory_total_mb = mem_info.total / (1024 ** 2)
                gpu.memory_used_mb = mem_info.used / (1024 ** 2)

                # Temperature
                gpu.temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                # Power
                try:
                    gpu.power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    pass

                # Clock speed
                try:
                    gpu.clock_speed = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS
                    )
                except:
                    pass

                # Fan speed
                try:
                    gpu.fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    pass

                # Determine status
                if gpu.temperature > 85 or gpu.utilization > 95:
                    gpu.status = ResourceStatus.CRITICAL
                elif gpu.temperature > 75 or gpu.utilization > 85:
                    gpu.status = ResourceStatus.WARNING
                else:
                    gpu.status = ResourceStatus.HEALTHY

                gpus.append(gpu)

            except Exception as e:
                logger.error(f"GPU {i} metrics error: {e}")

        return gpus

    def _collect_gputil_metrics(self) -> List[GPUMetrics]:
        """Collect GPU metrics using GPUtil"""
        import GPUtil

        gpus_list = GPUtil.getGPUs()
        gpus = []

        for gpu_obj in gpus_list:
            gpu = GPUMetrics(gpu_id=gpu_obj.id)
            gpu.utilization = gpu_obj.load * 100.0
            gpu.memory_total_mb = gpu_obj.memoryTotal
            gpu.memory_used_mb = gpu_obj.memoryUsed
            gpu.temperature = gpu_obj.temperature

            # Determine status
            if gpu.temperature > 85 or gpu.utilization > 95:
                gpu.status = ResourceStatus.CRITICAL
            elif gpu.temperature > 75 or gpu.utilization > 85:
                gpu.status = ResourceStatus.WARNING
            else:
                gpu.status = ResourceStatus.HEALTHY

            gpus.append(gpu)

        return gpus

    def _collect_network_metrics(self) -> NetworkMetrics:
        """Collect network performance metrics"""
        net = NetworkMetrics()

        try:
            net_io = psutil.net_io_counters()
            current_time = time.time()

            net.bytes_sent = net_io.bytes_sent
            net.bytes_recv = net_io.bytes_recv
            net.packets_sent = net_io.packets_sent
            net.packets_recv = net_io.packets_recv

            # Calculate rates
            if self.net_io_last:
                time_delta = current_time - self.net_io_timestamp
                if time_delta > 0:
                    bytes_sent_delta = net_io.bytes_sent - self.net_io_last.bytes_sent
                    bytes_recv_delta = net_io.bytes_recv - self.net_io_last.bytes_recv

                    net.send_rate_mbps = (bytes_sent_delta * 8) / (time_delta * 1e6)
                    net.recv_rate_mbps = (bytes_recv_delta * 8) / (time_delta * 1e6)

                    # Assume 10Gbps network for utilization calculation
                    total_rate_gbps = (net.send_rate_mbps + net.recv_rate_mbps) / 1000.0
                    net.bandwidth_utilization = (total_rate_gbps / 10.0) * 100.0

            # Calculate packet loss
            if hasattr(net_io, 'dropin') and hasattr(net_io, 'dropout'):
                total_packets = net_io.packets_recv + net_io.packets_sent
                dropped_packets = net_io.dropin + net_io.dropout
                if total_packets > 0:
                    net.packet_loss_rate = (dropped_packets / total_packets) * 100.0

            self.net_io_last = net_io
            self.net_io_timestamp = current_time

            # Determine status
            if net.bandwidth_utilization > 90 or net.packet_loss_rate > 5:
                net.status = ResourceStatus.CRITICAL
            elif net.bandwidth_utilization > 75 or net.packet_loss_rate > 1:
                net.status = ResourceStatus.WARNING
            else:
                net.status = ResourceStatus.HEALTHY

        except Exception as e:
            logger.error(f"Network metrics error: {e}")
            net.status = ResourceStatus.OFFLINE

        return net

    def _collect_camera_metrics(self) -> Dict[int, CameraMetrics]:
        """Collect camera health metrics from camera manager"""
        cameras = {}

        try:
            if hasattr(self.camera_manager, 'health'):
                for camera_id, health in self.camera_manager.health.items():
                    cam = CameraMetrics(camera_id=camera_id)

                    # Map camera health to metrics
                    cam.status = health.status.name.lower()
                    cam.fps = health.current_fps
                    cam.target_fps = health.target_fps
                    cam.frames_received = health.frames_received
                    cam.frames_dropped = health.frames_dropped

                    if cam.frames_received > 0:
                        cam.drop_rate = (cam.frames_dropped / cam.frames_received) * 100.0

                    cam.latency_ms = health.frame_latency
                    cam.temperature = health.temperature
                    cam.packet_loss = health.packet_loss_rate

                    # Determine health status
                    if cam.status == 'error' or cam.drop_rate > 10:
                        cam.health_status = ResourceStatus.CRITICAL
                    elif cam.drop_rate > 5 or abs(cam.fps - cam.target_fps) > 5:
                        cam.health_status = ResourceStatus.WARNING
                    elif cam.status == 'streaming':
                        cam.health_status = ResourceStatus.HEALTHY
                    else:
                        cam.health_status = ResourceStatus.OFFLINE

                    cameras[camera_id] = cam
        except Exception as e:
            logger.error(f"Camera metrics error: {e}")

        return cameras

    def _collect_detection_metrics(self) -> DetectionMetrics:
        """Collect detection performance metrics from tracker"""
        det = DetectionMetrics()

        try:
            if hasattr(self.tracker, 'metrics'):
                metrics = self.tracker.metrics

                det.total_detections = metrics.total_tracks_created
                det.confirmed_tracks = metrics.total_tracks_confirmed
                det.tracks_lost = metrics.total_tracks_lost
                det.detection_rate = metrics.detection_rate
                det.false_positive_rate = metrics.false_positive_rate
                det.processing_time_ms = metrics.avg_latency_ms
                det.occlusion_recoveries = metrics.occlusion_recoveries

                # Calculate average confidence
                if hasattr(self.tracker, 'tracks'):
                    confidences = [t.score for t in self.tracker.tracks.values() if t.confirmed]
                    if confidences:
                        det.avg_confidence = np.mean(confidences)

        except Exception as e:
            logger.error(f"Detection metrics error: {e}")

        return det

    def _calculate_overall_status(self, metrics: SystemMetrics) -> ResourceStatus:
        """Calculate overall system health status"""
        statuses = []

        # Add component statuses
        statuses.append(metrics.cpu.status)
        statuses.append(metrics.memory.status)
        statuses.append(metrics.network.status)

        for gpu in metrics.gpus:
            statuses.append(gpu.status)

        for cam in metrics.cameras.values():
            statuses.append(cam.health_status)

        # Determine worst status
        if ResourceStatus.CRITICAL in statuses:
            return ResourceStatus.CRITICAL
        elif ResourceStatus.WARNING in statuses:
            return ResourceStatus.WARNING
        elif ResourceStatus.OFFLINE in statuses:
            return ResourceStatus.WARNING
        else:
            return ResourceStatus.HEALTHY

    def register_callback(self, callback: Callable[[SystemMetrics], None]):
        """Register callback for metric updates"""
        self.metric_callbacks.append(callback)
        logger.info(f"Registered metric callback: {callback.__name__}")

    def set_camera_manager(self, camera_manager):
        """Set reference to camera manager"""
        self.camera_manager = camera_manager
        logger.info("Camera manager reference set")

    def set_tracker(self, tracker):
        """Set reference to tracker"""
        self.tracker = tracker
        logger.info("Tracker reference set")

    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics (thread-safe)"""
        with self.lock:
            return self.current_metrics

    def get_metrics_history(self, seconds: float = 30.0) -> List[SystemMetrics]:
        """Get historical metrics for specified duration"""
        with self.lock:
            num_samples = int(seconds * self.update_rate_hz)
            return list(self.metrics_history)[-num_samples:]

    def get_performance_overhead(self) -> float:
        """Get monitoring overhead as percentage of frame time"""
        frame_time_ms = 1000.0 / 30.0  # 30 FPS = 33.33ms
        return (self.monitoring_overhead_ms / frame_time_ms) * 100.0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of system status"""
        metrics = self.get_current_metrics()

        cameras_online = sum(1 for cam in metrics.cameras.values()
                           if cam.health_status == ResourceStatus.HEALTHY)

        return {
            'timestamp': metrics.timestamp,
            'overall_status': metrics.overall_status.value,
            'cpu_utilization': metrics.cpu.utilization,
            'memory_percent': metrics.memory.percent,
            'gpu_count': len(metrics.gpus),
            'avg_gpu_utilization': np.mean([g.utilization for g in metrics.gpus]) if metrics.gpus else 0,
            'network_send_mbps': metrics.network.send_rate_mbps,
            'network_recv_mbps': metrics.network.recv_rate_mbps,
            'cameras_total': len(metrics.cameras),
            'cameras_online': cameras_online,
            'confirmed_tracks': metrics.detection.confirmed_tracks,
            'detection_rate': metrics.detection.detection_rate,
            'processing_time_ms': metrics.detection.processing_time_ms,
            'monitoring_overhead_percent': self.get_performance_overhead(),
            'samples_collected': self.samples_collected
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("System Monitor - Real-time Performance Monitoring")
    print("=" * 60)

    # Create monitor
    monitor = SystemMonitor(update_rate_hz=10.0, num_cameras=20)

    # Register callback
    def print_metrics(metrics: SystemMetrics):
        print(f"\nTimestamp: {metrics.timestamp:.2f}")
        print(f"CPU: {metrics.cpu.utilization:.1f}% | "
              f"Memory: {metrics.memory.percent:.1f}% | "
              f"Network: {metrics.network.recv_rate_mbps:.2f} Mbps")
        if metrics.gpus:
            for gpu in metrics.gpus:
                print(f"GPU {gpu.gpu_id}: {gpu.utilization:.1f}% | "
                      f"{gpu.temperature:.1f}°C | "
                      f"{gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MB")

    monitor.register_callback(print_metrics)

    # Start monitoring
    monitor.start()

    try:
        # Monitor for 10 seconds
        time.sleep(10)

        # Print summary
        print("\n" + "=" * 60)
        summary = monitor.get_summary()
        print("System Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print(f"\nMonitoring overhead: {monitor.get_performance_overhead():.3f}%")

    finally:
        monitor.stop()

    print("\nMonitoring stopped")
