"""
Alert Manager - Intelligent Alert Management and Notification System

Features:
- Performance degradation alerts
- Camera failure detection
- Network issue alerts
- Automatic diagnostics
- Multi-channel notifications (email, SMS, webhook, log)
- Alert deduplication and rate limiting
- Alert severity escalation
- Automatic recovery detection

Requirements:
- Real-time alert generation
- <10ms alert processing overhead
- Configurable notification channels
- Alert history and analytics
- Automatic diagnostic information
"""

import time
import threading
import logging
import smtplib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class AlertCategory(Enum):
    """Alert categories"""
    PERFORMANCE = "performance"
    CAMERA = "camera"
    NETWORK = "network"
    DETECTION = "detection"
    HARDWARE = "hardware"
    CALIBRATION = "calibration"
    DATA_QUALITY = "data_quality"
    SYSTEM = "system"


class AlertChannel(Enum):
    """Alert notification channels"""
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    DATABASE = "database"


@dataclass
class Alert:
    """Represents a system alert"""
    alert_id: str
    level: AlertLevel
    category: AlertCategory
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)

    # Context
    component: Optional[str] = None
    camera_id: Optional[int] = None
    track_id: Optional[int] = None

    # Diagnostic information
    current_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    threshold: Optional[Any] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Suggested actions
    suggestions: List[str] = field(default_factory=list)

    # Alert management
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None
    escalated: bool = False
    notification_sent: bool = False

    def __hash__(self):
        """Hash for deduplication"""
        return hash(self.alert_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'level': self.level.name,
            'category': self.category.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp,
            'component': self.component,
            'camera_id': self.camera_id,
            'track_id': self.track_id,
            'current_value': self.current_value,
            'expected_value': self.expected_value,
            'threshold': self.threshold,
            'diagnostics': self.diagnostics,
            'suggestions': self.suggestions,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts"""
    rule_id: str
    category: AlertCategory
    level: AlertLevel
    condition: Callable[[Any], bool]
    message_template: str
    threshold: Optional[Any] = None
    cooldown_seconds: float = 60.0
    last_triggered: float = 0.0
    enabled: bool = True
    suggestions: List[str] = field(default_factory=list)


class EmailNotifier:
    """Email notification handler"""

    def __init__(self, smtp_host: str, smtp_port: int = 587,
                 username: str = "", password: str = "",
                 from_addr: str = "", to_addrs: List[str] = None):
        """
        Initialize email notifier

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_addr: From email address
            to_addrs: List of recipient email addresses
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.enabled = bool(smtp_host and from_addr and to_addrs)

        if self.enabled:
            logger.info(f"Email notifier enabled: {from_addr} -> {to_addrs}")
        else:
            logger.warning("Email notifier disabled (missing configuration)")

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not self.enabled:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            msg['Subject'] = f"[{alert.level.name}] {alert.title}"

            # Create email body
            body = f"""
Alert Details:
--------------
Level: {alert.level.name}
Category: {alert.category.value}
Component: {alert.component or 'N/A'}
Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}

Message:
{alert.message}

Current Value: {alert.current_value}
Expected Value: {alert.expected_value}
Threshold: {alert.threshold}

Diagnostics:
{json.dumps(alert.diagnostics, indent=2)}

Suggested Actions:
{chr(10).join('- ' + s for s in alert.suggestions) if alert.suggestions else 'None'}
"""

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email sent for alert {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookNotifier:
    """Webhook notification handler"""

    def __init__(self, webhook_url: str = "", timeout: float = 5.0):
        """
        Initialize webhook notifier

        Args:
            webhook_url: Webhook URL
            timeout: Request timeout
        """
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.enabled = bool(webhook_url)

        if self.enabled:
            logger.info(f"Webhook notifier enabled: {webhook_url}")
        else:
            logger.warning("Webhook notifier disabled (missing URL)")

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        if not self.enabled:
            return False

        try:
            import requests

            payload = alert.to_dict()

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            logger.info(f"Webhook sent for alert {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class AlertManager:
    """
    Main alert management system
    Handles alert generation, deduplication, notification, and tracking
    """

    def __init__(self,
                 max_alerts_per_minute: int = 100,
                 alert_history_size: int = 1000,
                 enable_auto_diagnostics: bool = True):
        """
        Initialize alert manager

        Args:
            max_alerts_per_minute: Maximum alerts per minute (rate limiting)
            alert_history_size: Number of historical alerts to keep
            enable_auto_diagnostics: Enable automatic diagnostics
        """
        self.max_alerts_per_minute = max_alerts_per_minute
        self.alert_history_size = alert_history_size
        self.enable_auto_diagnostics = enable_auto_diagnostics

        # Active alerts (unresolved)
        self.active_alerts: Dict[str, Alert] = {}

        # Alert history
        self.alert_history: deque = deque(maxlen=alert_history_size)

        # Alert rules
        self.rules: Dict[str, AlertRule] = {}

        # Notification channels
        self.notifiers: Dict[AlertChannel, Any] = {}

        # Deduplication tracking
        self.recent_alert_hashes: Set[str] = set()
        self.dedup_window_seconds = 300.0  # 5 minutes

        # Rate limiting
        self.alerts_last_minute: deque = deque(maxlen=max_alerts_per_minute)

        # Statistics
        self.total_alerts_generated = 0
        self.alerts_by_level: Dict[AlertLevel, int] = {}
        self.alerts_by_category: Dict[AlertCategory, int] = {}
        self.notifications_sent = 0
        self.notifications_failed = 0

        # Threading
        self.lock = threading.Lock()

        # System monitors (set externally)
        self.system_monitor = None

        logger.info("AlertManager initialized")

    def configure_email(self, smtp_host: str, smtp_port: int = 587,
                       username: str = "", password: str = "",
                       from_addr: str = "", to_addrs: List[str] = None):
        """Configure email notifications"""
        notifier = EmailNotifier(smtp_host, smtp_port, username, password, from_addr, to_addrs)
        self.notifiers[AlertChannel.EMAIL] = notifier

    def configure_webhook(self, webhook_url: str):
        """Configure webhook notifications"""
        notifier = WebhookNotifier(webhook_url)
        self.notifiers[AlertChannel.WEBHOOK] = notifier

    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        with self.lock:
            self.rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str):
        """Remove alert rule"""
        with self.lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")

    def create_alert(self,
                    level: AlertLevel,
                    category: AlertCategory,
                    title: str,
                    message: str,
                    component: Optional[str] = None,
                    camera_id: Optional[int] = None,
                    current_value: Optional[Any] = None,
                    expected_value: Optional[Any] = None,
                    threshold: Optional[Any] = None,
                    suggestions: List[str] = None) -> Optional[Alert]:
        """
        Create and process a new alert

        Args:
            level: Alert severity level
            category: Alert category
            title: Alert title
            message: Alert message
            component: Component identifier
            camera_id: Camera ID (if applicable)
            current_value: Current measured value
            expected_value: Expected value
            threshold: Alert threshold
            suggestions: Suggested actions

        Returns:
            Created Alert or None if deduplicated/rate-limited
        """
        # Generate alert ID
        alert_id = self._generate_alert_id(level, category, title, component, camera_id)

        # Check deduplication
        if self._is_duplicate(alert_id):
            logger.debug(f"Alert deduplicated: {alert_id}")
            return None

        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning("Alert rate limit exceeded")
            return None

        # Create alert
        alert = Alert(
            alert_id=alert_id,
            level=level,
            category=category,
            title=title,
            message=message,
            component=component,
            camera_id=camera_id,
            current_value=current_value,
            expected_value=expected_value,
            threshold=threshold,
            suggestions=suggestions or []
        )

        # Add diagnostics
        if self.enable_auto_diagnostics:
            alert.diagnostics = self._generate_diagnostics(alert)

        # Process alert
        with self.lock:
            # Add to active alerts
            self.active_alerts[alert_id] = alert

            # Add to history
            self.alert_history.append(alert)

            # Update statistics
            self.total_alerts_generated += 1
            self.alerts_by_level[level] = self.alerts_by_level.get(level, 0) + 1
            self.alerts_by_category[category] = self.alerts_by_category.get(category, 0) + 1

            # Track for deduplication
            self.recent_alert_hashes.add(alert_id)
            self.alerts_last_minute.append(time.time())

        # Send notifications
        self._send_notifications(alert)

        logger.info(f"Alert created: [{level.name}] {title}")

        return alert

    def _generate_alert_id(self, level: AlertLevel, category: AlertCategory,
                          title: str, component: Optional[str],
                          camera_id: Optional[int]) -> str:
        """Generate unique alert ID for deduplication"""
        key = f"{level.name}:{category.value}:{title}:{component}:{camera_id}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _is_duplicate(self, alert_id: str) -> bool:
        """Check if alert is a duplicate within deduplication window"""
        if alert_id not in self.recent_alert_hashes:
            return False

        # Check if alert is still active
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            age = time.time() - alert.timestamp
            return age < self.dedup_window_seconds

        return False

    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        current_time = time.time()

        # Remove old entries
        while self.alerts_last_minute and current_time - self.alerts_last_minute[0] > 60.0:
            self.alerts_last_minute.popleft()

        # Check limit
        return len(self.alerts_last_minute) < self.max_alerts_per_minute

    def _generate_diagnostics(self, alert: Alert) -> Dict[str, Any]:
        """Generate automatic diagnostic information"""
        diagnostics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp)),
            'system_time': time.time()
        }

        # Add system monitor data if available
        if self.system_monitor:
            try:
                summary = self.system_monitor.get_summary()
                diagnostics['system_status'] = summary['overall_status']
                diagnostics['cpu_utilization'] = summary['cpu_utilization']
                diagnostics['memory_percent'] = summary['memory_percent']
                diagnostics['cameras_online'] = summary['cameras_online']

                # Category-specific diagnostics
                if alert.category == AlertCategory.CAMERA and alert.camera_id is not None:
                    metrics = self.system_monitor.get_current_metrics()
                    if alert.camera_id in metrics.cameras:
                        cam = metrics.cameras[alert.camera_id]
                        diagnostics['camera_fps'] = cam.fps
                        diagnostics['camera_drop_rate'] = cam.drop_rate
                        diagnostics['camera_temperature'] = cam.temperature

                elif alert.category == AlertCategory.NETWORK:
                    metrics = self.system_monitor.get_current_metrics()
                    diagnostics['network_send_mbps'] = metrics.network.send_rate_mbps
                    diagnostics['network_recv_mbps'] = metrics.network.recv_rate_mbps
                    diagnostics['network_utilization'] = metrics.network.bandwidth_utilization
                    diagnostics['packet_loss'] = metrics.network.packet_loss_rate

                elif alert.category == AlertCategory.PERFORMANCE:
                    metrics = self.system_monitor.get_current_metrics()
                    diagnostics['processing_time_ms'] = metrics.detection.processing_time_ms
                    diagnostics['confirmed_tracks'] = metrics.detection.confirmed_tracks

            except Exception as e:
                logger.error(f"Failed to generate diagnostics: {e}")

        return diagnostics

    def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        # Determine which channels to use based on alert level
        channels = self._get_notification_channels(alert.level)

        for channel in channels:
            try:
                if channel == AlertChannel.LOG:
                    self._log_alert(alert)
                elif channel == AlertChannel.CONSOLE:
                    self._console_alert(alert)
                elif channel == AlertChannel.EMAIL:
                    if AlertChannel.EMAIL in self.notifiers:
                        success = self.notifiers[AlertChannel.EMAIL].send_alert(alert)
                        if success:
                            self.notifications_sent += 1
                        else:
                            self.notifications_failed += 1
                elif channel == AlertChannel.WEBHOOK:
                    if AlertChannel.WEBHOOK in self.notifiers:
                        success = self.notifiers[AlertChannel.WEBHOOK].send_alert(alert)
                        if success:
                            self.notifications_sent += 1
                        else:
                            self.notifications_failed += 1

                alert.notification_sent = True

            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {e}")
                self.notifications_failed += 1

    def _get_notification_channels(self, level: AlertLevel) -> List[AlertChannel]:
        """Determine notification channels based on alert level"""
        channels = [AlertChannel.LOG, AlertChannel.CONSOLE]

        if level == AlertLevel.CRITICAL:
            channels.extend([AlertChannel.EMAIL, AlertChannel.WEBHOOK, AlertChannel.SMS])
        elif level == AlertLevel.ERROR:
            channels.extend([AlertChannel.EMAIL, AlertChannel.WEBHOOK])
        elif level == AlertLevel.WARNING:
            channels.append(AlertChannel.WEBHOOK)

        return channels

    def _log_alert(self, alert: Alert):
        """Log alert to logging system"""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }[alert.level]

        logger.log(log_level, f"[{alert.category.value.upper()}] {alert.title}: {alert.message}")

    def _console_alert(self, alert: Alert):
        """Print alert to console"""
        color_codes = {
            AlertLevel.INFO: '\033[94m',  # Blue
            AlertLevel.WARNING: '\033[93m',  # Yellow
            AlertLevel.ERROR: '\033[91m',  # Red
            AlertLevel.CRITICAL: '\033[95m'  # Magenta
        }
        reset_code = '\033[0m'

        color = color_codes.get(alert.level, '')
        print(f"{color}[{alert.level.name}] {alert.title}: {alert.message}{reset_code}")

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = time.time()
                del self.active_alerts[alert_id]

                logger.info(f"Alert resolved: {alert_id}")

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")

    def check_rules(self, data: Any):
        """Check all alert rules against provided data"""
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            # Check cooldown
            if time.time() - rule.last_triggered < rule.cooldown_seconds:
                continue

            try:
                # Evaluate rule condition
                if rule.condition(data):
                    # Generate message from template
                    message = rule.message_template.format(data=data)

                    # Create alert
                    self.create_alert(
                        level=rule.level,
                        category=rule.category,
                        title=f"Rule triggered: {rule_id}",
                        message=message,
                        threshold=rule.threshold,
                        suggestions=rule.suggestions
                    )

                    rule.last_triggered = time.time()

            except Exception as e:
                logger.error(f"Error evaluating rule {rule_id}: {e}")

    def get_active_alerts(self, level: Optional[AlertLevel] = None,
                         category: Optional[AlertCategory] = None) -> List[Alert]:
        """Get active alerts, optionally filtered"""
        with self.lock:
            alerts = list(self.active_alerts.values())

        if level:
            alerts = [a for a in alerts if a.level == level]
        if category:
            alerts = [a for a in alerts if a.category == category]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_history(self, minutes: int = 60) -> List[Alert]:
        """Get alert history for specified duration"""
        cutoff_time = time.time() - (minutes * 60)

        with self.lock:
            return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        with self.lock:
            return {
                'total_alerts_generated': self.total_alerts_generated,
                'active_alerts': len(self.active_alerts),
                'alerts_by_level': {level.name: count for level, count in self.alerts_by_level.items()},
                'alerts_by_category': {cat.name: count for cat, count in self.alerts_by_category.items()},
                'notifications_sent': self.notifications_sent,
                'notifications_failed': self.notifications_failed,
                'notification_success_rate': self.notifications_sent / max(self.notifications_sent + self.notifications_failed, 1)
            }

    def set_system_monitor(self, system_monitor):
        """Set reference to system monitor"""
        self.system_monitor = system_monitor
        logger.info("System monitor reference set")


# Predefined alert rules
def create_default_rules() -> List[AlertRule]:
    """Create default alert rules"""
    rules = []

    # CPU overload rule
    rules.append(AlertRule(
        rule_id="cpu_overload",
        category=AlertCategory.PERFORMANCE,
        level=AlertLevel.WARNING,
        condition=lambda data: data.get('cpu_utilization', 0) > 90,
        message_template="CPU utilization critical: {data[cpu_utilization]:.1f}%",
        threshold=90.0,
        cooldown_seconds=60.0,
        suggestions=[
            "Check for resource-intensive processes",
            "Consider scaling to more nodes",
            "Review algorithm efficiency"
        ]
    ))

    # Memory pressure rule
    rules.append(AlertRule(
        rule_id="memory_pressure",
        category=AlertCategory.PERFORMANCE,
        level=AlertLevel.ERROR,
        condition=lambda data: data.get('memory_percent', 0) > 95,
        message_template="Memory usage critical: {data[memory_percent]:.1f}%",
        threshold=95.0,
        cooldown_seconds=60.0,
        suggestions=[
            "Check for memory leaks",
            "Reduce buffer sizes",
            "Clear cached data"
        ]
    ))

    # Camera offline rule
    rules.append(AlertRule(
        rule_id="camera_offline",
        category=AlertCategory.CAMERA,
        level=AlertLevel.CRITICAL,
        condition=lambda data: data.get('cameras_online', 20) < 18,
        message_template="Multiple cameras offline: {data[cameras_online]}/20",
        threshold=18,
        cooldown_seconds=120.0,
        suggestions=[
            "Check camera power and connections",
            "Verify network connectivity",
            "Review camera configuration"
        ]
    ))

    # Network saturation rule
    rules.append(AlertRule(
        rule_id="network_saturation",
        category=AlertCategory.NETWORK,
        level=AlertLevel.WARNING,
        condition=lambda data: data.get('network_utilization', 0) > 85,
        message_template="Network bandwidth saturated: {data[network_utilization]:.1f}%",
        threshold=85.0,
        cooldown_seconds=60.0,
        suggestions=[
            "Enable compression",
            "Reduce frame rate or resolution",
            "Check for network bottlenecks"
        ]
    ))

    # Detection rate drop rule
    rules.append(AlertRule(
        rule_id="detection_rate_drop",
        category=AlertCategory.DETECTION,
        level=AlertLevel.WARNING,
        condition=lambda data: data.get('detection_rate', 1.0) < 0.90,
        message_template="Detection rate below target: {data[detection_rate]:.1%}",
        threshold=0.90,
        cooldown_seconds=300.0,
        suggestions=[
            "Check camera focus and exposure",
            "Review detection thresholds",
            "Verify lighting conditions"
        ]
    ))

    return rules


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Alert Manager - Intelligent Alert System")
    print("=" * 60)

    # Create alert manager
    alert_mgr = AlertManager(
        max_alerts_per_minute=100,
        enable_auto_diagnostics=True
    )

    # Add default rules
    for rule in create_default_rules():
        alert_mgr.add_rule(rule)

    print(f"\nAdded {len(alert_mgr.rules)} alert rules")

    # Create some test alerts
    print("\nGenerating test alerts:")

    alert_mgr.create_alert(
        level=AlertLevel.WARNING,
        category=AlertCategory.CAMERA,
        title="Camera 5 FPS dropped",
        message="Camera 5 FPS dropped from 30 to 15",
        camera_id=5,
        current_value=15.0,
        expected_value=30.0,
        threshold=25.0,
        suggestions=["Check network connection", "Verify camera settings"]
    )

    alert_mgr.create_alert(
        level=AlertLevel.CRITICAL,
        category=AlertCategory.HARDWARE,
        title="GPU temperature critical",
        message="GPU 0 temperature exceeded safe limit",
        current_value=92.0,
        expected_value=75.0,
        threshold=85.0,
        suggestions=["Check cooling system", "Reduce workload"]
    )

    # Print active alerts
    print("\n" + "=" * 60)
    print("Active Alerts:")
    for alert in alert_mgr.get_active_alerts():
        print(f"\n[{alert.level.name}] {alert.title}")
        print(f"  Category: {alert.category.value}")
        print(f"  Message: {alert.message}")
        if alert.suggestions:
            print(f"  Suggestions: {', '.join(alert.suggestions)}")

    # Print statistics
    print("\n" + "=" * 60)
    stats = alert_mgr.get_statistics()
    print("Alert Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
