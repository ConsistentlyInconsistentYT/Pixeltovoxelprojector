"""
Pipeline Coordinator - Component lifecycle and orchestration

This module manages:
- Component initialization and shutdown
- Error handling and recovery
- Performance monitoring
- Resource allocation
- Health checks and alerts
"""

import time
import threading
import logging
import signal
import sys
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ComponentState(Enum):
    """Component lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERING = "recovering"


@dataclass
class ComponentStatus:
    """Status information for a component"""
    name: str
    state: ComponentState = ComponentState.UNINITIALIZED
    health: str = "unknown"  # healthy, warning, critical
    last_update: float = field(default_factory=time.time)
    error_count: int = 0
    last_error: Optional[str] = None
    restart_count: int = 0
    uptime_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinatorConfig:
    """Configuration for pipeline coordinator"""
    # Health check settings
    health_check_interval_s: float = 5.0
    component_timeout_s: float = 30.0

    # Recovery settings
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_delay_s: float = 5.0

    # Shutdown settings
    graceful_shutdown_timeout_s: float = 30.0
    force_shutdown_after_timeout: bool = True

    # Monitoring
    enable_watchdog: bool = True
    watchdog_interval_s: float = 1.0

    # Alerts
    enable_alerts: bool = True
    alert_callback: Optional[Callable] = None


class PipelineCoordinator:
    """
    Coordinates lifecycle of all system components

    Responsibilities:
    - Initialize components in correct order
    - Monitor component health
    - Handle component failures and recovery
    - Coordinate graceful shutdown
    - Resource allocation and management
    - Performance monitoring
    """

    def __init__(self, config: CoordinatorConfig):
        """
        Initialize pipeline coordinator

        Args:
            config: Coordinator configuration
        """
        self.config = config

        # Component registry
        self.components: Dict[str, Any] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        self.initialization_order: List[str] = []

        # State
        self.running = False
        self.shutdown_requested = False
        self.emergency_shutdown = False

        # Threads
        self.health_check_thread: Optional[threading.Thread] = None
        self.watchdog_thread: Optional[threading.Thread] = None

        # Synchronization
        self.lock = threading.Lock()

        # Callbacks
        self.state_change_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []

        # Statistics
        self.start_time: Optional[float] = None
        self.total_errors = 0
        self.total_recoveries = 0

        # Signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info("PipelineCoordinator initialized")

    def register_component(
        self,
        name: str,
        component: Any,
        dependencies: Optional[List[str]] = None
    ):
        """
        Register a component with the coordinator

        Args:
            name: Component name
            component: Component instance
            dependencies: List of component names this depends on
        """
        with self.lock:
            self.components[name] = component
            self.component_status[name] = ComponentStatus(name=name)
            self.component_dependencies[name] = dependencies or []

        logger.info(f"Registered component: {name}")

    def initialize_all(self) -> bool:
        """
        Initialize all components in dependency order

        Returns:
            True if all components initialized successfully
        """
        logger.info("Initializing all components...")

        # Determine initialization order based on dependencies
        self._calculate_initialization_order()

        # Initialize components in order
        for component_name in self.initialization_order:
            if not self._initialize_component(component_name):
                logger.error(f"Failed to initialize {component_name}")
                return False

        logger.info("All components initialized successfully")
        return True

    def start_all(self) -> bool:
        """
        Start all components

        Returns:
            True if all components started successfully
        """
        logger.info("Starting all components...")
        self.running = True
        self.start_time = time.time()

        # Start components in initialization order
        for component_name in self.initialization_order:
            if not self._start_component(component_name):
                logger.error(f"Failed to start {component_name}")
                self.stop_all()
                return False

        # Start health monitoring
        if self.config.enable_watchdog:
            self._start_health_monitoring()

        logger.info("All components started successfully")
        return True

    def stop_all(self, emergency: bool = False):
        """
        Stop all components

        Args:
            emergency: If True, force immediate shutdown
        """
        if emergency:
            logger.warning("EMERGENCY SHUTDOWN INITIATED")
            self.emergency_shutdown = True

        logger.info("Stopping all components...")
        self.shutdown_requested = True
        self.running = False

        # Stop health monitoring
        if self.health_check_thread:
            self.health_check_thread.join(timeout=2.0)

        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=2.0)

        # Stop components in reverse order
        for component_name in reversed(self.initialization_order):
            self._stop_component(component_name, emergency=emergency)

        # Print final statistics
        self._print_statistics()

        logger.info("All components stopped")

    def pause_all(self):
        """Pause all components"""
        logger.info("Pausing all components...")

        for component_name in self.initialization_order:
            self._pause_component(component_name)

        logger.info("All components paused")

    def resume_all(self):
        """Resume all components"""
        logger.info("Resuming all components...")

        for component_name in self.initialization_order:
            self._resume_component(component_name)

        logger.info("All components resumed")

    def restart_component(self, component_name: str) -> bool:
        """
        Restart a specific component

        Args:
            component_name: Name of component to restart

        Returns:
            True if restart successful
        """
        logger.info(f"Restarting component: {component_name}")

        status = self.component_status.get(component_name)
        if not status:
            logger.error(f"Component not found: {component_name}")
            return False

        # Stop component
        self._stop_component(component_name, emergency=False)

        # Wait a bit
        time.sleep(self.config.recovery_delay_s)

        # Restart
        if not self._start_component(component_name):
            logger.error(f"Failed to restart {component_name}")
            status.state = ComponentState.ERROR
            return False

        status.restart_count += 1
        logger.info(f"Component {component_name} restarted successfully")
        return True

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status

        Returns:
            Dictionary with system status information
        """
        with self.lock:
            component_states = {}
            healthy_count = 0
            warning_count = 0
            critical_count = 0
            error_count = 0

            for name, status in self.component_status.items():
                component_states[name] = {
                    'state': status.state.value,
                    'health': status.health,
                    'uptime': status.uptime_seconds,
                    'errors': status.error_count,
                    'restarts': status.restart_count
                }

                if status.health == 'healthy':
                    healthy_count += 1
                elif status.health == 'warning':
                    warning_count += 1
                elif status.health == 'critical':
                    critical_count += 1

                if status.state == ComponentState.ERROR:
                    error_count += 1

            # Determine overall health
            if critical_count > 0 or error_count > 0:
                overall_health = 'critical'
            elif warning_count > 0:
                overall_health = 'warning'
            else:
                overall_health = 'healthy'

            uptime = time.time() - self.start_time if self.start_time else 0

            return {
                'overall_health': overall_health,
                'uptime_seconds': uptime,
                'running': self.running,
                'components': component_states,
                'statistics': {
                    'healthy_components': healthy_count,
                    'warning_components': warning_count,
                    'critical_components': critical_count,
                    'error_components': error_count,
                    'total_errors': self.total_errors,
                    'total_recoveries': self.total_recoveries
                }
            }

    def _calculate_initialization_order(self):
        """Calculate component initialization order based on dependencies"""
        # Topological sort of dependencies
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            # Visit dependencies first
            for dep in self.component_dependencies.get(name, []):
                visit(dep)

            order.append(name)

        for component_name in self.components.keys():
            visit(component_name)

        self.initialization_order = order
        logger.info(f"Initialization order: {' -> '.join(order)}")

    def _initialize_component(self, component_name: str) -> bool:
        """Initialize a single component"""
        logger.info(f"Initializing component: {component_name}")

        status = self.component_status[component_name]
        component = self.components[component_name]

        status.state = ComponentState.INITIALIZING

        try:
            # Call initialization method if available
            if hasattr(component, 'initialize'):
                result = component.initialize()
                if result is False:
                    raise RuntimeError(f"Component {component_name} initialization returned False")
            elif hasattr(component, 'init'):
                result = component.init()
                if result is False:
                    raise RuntimeError(f"Component {component_name} init returned False")

            status.state = ComponentState.READY
            status.health = 'healthy'
            logger.info(f"Component {component_name} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {component_name}: {e}", exc_info=True)
            status.state = ComponentState.ERROR
            status.health = 'critical'
            status.error_count += 1
            status.last_error = str(e)
            self.total_errors += 1
            return False

    def _start_component(self, component_name: str) -> bool:
        """Start a single component"""
        logger.info(f"Starting component: {component_name}")

        status = self.component_status[component_name]
        component = self.components[component_name]

        try:
            # Call start method if available
            if hasattr(component, 'start'):
                result = component.start()
                if result is False:
                    raise RuntimeError(f"Component {component_name} start returned False")

            status.state = ComponentState.RUNNING
            status.health = 'healthy'
            logger.info(f"Component {component_name} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start {component_name}: {e}", exc_info=True)
            status.state = ComponentState.ERROR
            status.health = 'critical'
            status.error_count += 1
            status.last_error = str(e)
            self.total_errors += 1
            return False

    def _stop_component(self, component_name: str, emergency: bool = False):
        """Stop a single component"""
        logger.info(f"Stopping component: {component_name}")

        status = self.component_status[component_name]
        component = self.components[component_name]

        status.state = ComponentState.STOPPING

        try:
            # Call stop method if available
            if hasattr(component, 'stop'):
                if emergency:
                    # Force immediate stop
                    component.stop()
                else:
                    # Graceful stop with timeout
                    component.stop()

            status.state = ComponentState.STOPPED
            logger.info(f"Component {component_name} stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping {component_name}: {e}", exc_info=True)
            status.error_count += 1
            status.last_error = str(e)

    def _pause_component(self, component_name: str):
        """Pause a single component"""
        status = self.component_status[component_name]
        component = self.components[component_name]

        if hasattr(component, 'pause'):
            component.pause()
            status.state = ComponentState.PAUSED

    def _resume_component(self, component_name: str):
        """Resume a single component"""
        status = self.component_status[component_name]
        component = self.components[component_name]

        if hasattr(component, 'resume'):
            component.resume()
            status.state = ComponentState.RUNNING

    def _start_health_monitoring(self):
        """Start health monitoring threads"""
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="HealthCheckThread"
        )
        self.health_check_thread.start()

        if self.config.enable_watchdog:
            self.watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="WatchdogThread"
            )
            self.watchdog_thread.start()

        logger.info("Health monitoring started")

    def _health_check_loop(self):
        """Health check loop"""
        while self.running:
            try:
                for component_name, component in self.components.items():
                    self._check_component_health(component_name, component)

                time.sleep(self.config.health_check_interval_s)

            except Exception as e:
                logger.error(f"Health check error: {e}", exc_info=True)

    def _check_component_health(self, component_name: str, component: Any):
        """Check health of a single component"""
        status = self.component_status[component_name]

        try:
            # Get health status if available
            if hasattr(component, 'get_health'):
                health = component.get_health()
                if health:
                    status.health = health.get('status', 'unknown')
                    status.metrics = health.get('metrics', {})

            # Get component metrics if available
            elif hasattr(component, 'get_metrics'):
                metrics = component.get_metrics()
                status.metrics = metrics

            # Update uptime
            if status.state == ComponentState.RUNNING:
                status.uptime_seconds = time.time() - status.last_update

            # Check for critical conditions
            if status.error_count > 10:
                status.health = 'critical'

                # Attempt recovery if enabled
                if self.config.enable_auto_recovery and status.restart_count < self.config.max_recovery_attempts:
                    logger.warning(f"Component {component_name} is critical, attempting recovery...")
                    self._attempt_recovery(component_name)

            status.last_update = time.time()

        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            status.error_count += 1
            status.last_error = str(e)

    def _watchdog_loop(self):
        """Watchdog loop for detecting hangs"""
        while self.running:
            try:
                # Check if any components have stopped responding
                current_time = time.time()

                for component_name, status in self.component_status.items():
                    if status.state == ComponentState.RUNNING:
                        time_since_update = current_time - status.last_update

                        if time_since_update > self.config.component_timeout_s:
                            logger.warning(f"Component {component_name} appears to be hung "
                                         f"({time_since_update:.1f}s since last update)")

                            # Attempt recovery
                            if self.config.enable_auto_recovery:
                                self._attempt_recovery(component_name)

                time.sleep(self.config.watchdog_interval_s)

            except Exception as e:
                logger.error(f"Watchdog error: {e}", exc_info=True)

    def _attempt_recovery(self, component_name: str):
        """Attempt to recover a failed component"""
        status = self.component_status[component_name]

        if status.restart_count >= self.config.max_recovery_attempts:
            logger.error(f"Component {component_name} exceeded max recovery attempts")
            status.state = ComponentState.ERROR
            status.health = 'critical'
            return

        logger.info(f"Attempting recovery for {component_name} "
                   f"(attempt {status.restart_count + 1}/{self.config.max_recovery_attempts})")

        status.state = ComponentState.RECOVERING

        if self.restart_component(component_name):
            logger.info(f"Successfully recovered {component_name}")
            self.total_recoveries += 1
        else:
            logger.error(f"Failed to recover {component_name}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _print_statistics(self):
        """Print coordinator statistics"""
        status = self.get_system_status()

        logger.info("="*60)
        logger.info("Pipeline Coordinator Statistics")
        logger.info("="*60)
        logger.info(f"Overall health: {status['overall_health']}")
        logger.info(f"Uptime: {status['uptime_seconds']:.1f} seconds")
        logger.info(f"Total errors: {self.total_errors}")
        logger.info(f"Total recoveries: {self.total_recoveries}")
        logger.info("")
        logger.info("Component Status:")
        for name, comp_status in status['components'].items():
            logger.info(f"  {name}: {comp_status['state']} ({comp_status['health']}) - "
                       f"Errors: {comp_status['errors']}, Restarts: {comp_status['restarts']}")
        logger.info("="*60)

    def register_state_change_callback(self, callback: Callable):
        """Register callback for component state changes"""
        self.state_change_callbacks.append(callback)

    def register_error_callback(self, callback: Callable):
        """Register callback for component errors"""
        self.error_callbacks.append(callback)
