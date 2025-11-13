"""
Web Dashboard - Real-time Monitoring Dashboard

Features:
- Real-time system visualization
- Performance graphs and charts
- Camera view displays
- 3D voxel visualization
- Alert history and management
- Interactive controls

Requirements:
- Web-accessible interface (HTTP/HTTPS)
- Real-time updates via WebSocket
- Responsive design
- Low latency (<100ms update rate)
- Support for multiple concurrent viewers

Technology:
- Flask for web server
- Socket.IO for real-time updates
- Plotly/Chart.js for visualizations
- Three.js for 3D rendering
"""

import time
import threading
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np

logger = logging.getLogger(__name__)


# HTML Template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixel-to-Voxel Monitoring Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }

        .header h1 {
            color: white;
            font-size: 28px;
            margin-bottom: 5px;
        }

        .header .subtitle {
            color: rgba(255,255,255,0.8);
            font-size: 14px;
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .status-healthy { background: #4caf50; }
        .status-warning { background: #ff9800; }
        .status-critical { background: #f44336; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: #2d2d2d;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        }

        .card-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 8px;
            background: #383838;
            border-radius: 5px;
        }

        .metric-label {
            color: #b0b0b0;
            font-size: 14px;
        }

        .metric-value {
            font-size: 18px;
            font-weight: bold;
            color: #e0e0e0;
        }

        .metric-value.good { color: #4caf50; }
        .metric-value.warning { color: #ff9800; }
        .metric-value.bad { color: #f44336; }

        .progress-bar {
            width: 100%;
            height: 20px;
            background: #383838;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }

        .progress-fill {
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 10px;
        }

        .progress-fill.low { background: #4caf50; }
        .progress-fill.medium { background: #ff9800; }
        .progress-fill.high { background: #f44336; }

        .alert {
            background: #2d2d2d;
            border-left: 4px solid;
            padding: 12px;
            margin: 10px 0;
            border-radius: 5px;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .alert.info { border-color: #2196f3; }
        .alert.warning { border-color: #ff9800; }
        .alert.error { border-color: #f44336; }
        .alert.critical { border-color: #9c27b0; }

        .alert-title {
            font-weight: bold;
            margin-bottom: 5px;
        }

        .alert-time {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }

        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 15px;
        }

        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }

        .camera-card {
            background: #383838;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s;
        }

        .camera-card.online {
            border-color: #4caf50;
        }

        .camera-card.warning {
            border-color: #ff9800;
        }

        .camera-card.offline {
            border-color: #f44336;
        }

        .camera-id {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 5px;
        }

        .camera-status {
            font-size: 12px;
            color: #b0b0b0;
        }

        .controls {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }

        .button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin: 5px;
            transition: background 0.3s;
        }

        .button:hover {
            background: #764ba2;
        }

        .button:active {
            transform: scale(0.98);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <span class="status-indicator status-healthy" id="connection-status"></span>
            Pixel-to-Voxel Monitoring Dashboard
        </h1>
        <div class="subtitle">Real-time system monitoring and diagnostics</div>
    </div>

    <!-- System Overview -->
    <div class="grid">
        <!-- System Health -->
        <div class="card">
            <div class="card-header">System Health</div>
            <div class="metric">
                <span class="metric-label">Overall Status</span>
                <span class="metric-value" id="overall-status">Healthy</span>
            </div>
            <div class="metric">
                <span class="metric-label">CPU Utilization</span>
                <span class="metric-value" id="cpu-util">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill low" id="cpu-progress" style="width: 0%"></div>
            </div>
            <div class="metric">
                <span class="metric-label">Memory Usage</span>
                <span class="metric-value" id="memory-util">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill low" id="memory-progress" style="width: 0%"></div>
            </div>
            <div class="metric">
                <span class="metric-label">Active Alerts</span>
                <span class="metric-value" id="active-alerts">0</span>
            </div>
        </div>

        <!-- GPU Status -->
        <div class="card">
            <div class="card-header">GPU Status</div>
            <div id="gpu-metrics">
                <div class="metric">
                    <span class="metric-label">No GPU data</span>
                </div>
            </div>
        </div>

        <!-- Network Status -->
        <div class="card">
            <div class="card-header">Network Status</div>
            <div class="metric">
                <span class="metric-label">Send Rate</span>
                <span class="metric-value" id="network-send">0 Mbps</span>
            </div>
            <div class="metric">
                <span class="metric-label">Receive Rate</span>
                <span class="metric-value" id="network-recv">0 Mbps</span>
            </div>
            <div class="metric">
                <span class="metric-label">Bandwidth Utilization</span>
                <span class="metric-value" id="network-util">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill low" id="network-progress" style="width: 0%"></div>
            </div>
        </div>

        <!-- Detection Status -->
        <div class="card">
            <div class="card-header">Detection & Tracking</div>
            <div class="metric">
                <span class="metric-label">Confirmed Tracks</span>
                <span class="metric-value" id="confirmed-tracks">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Detection Rate</span>
                <span class="metric-value" id="detection-rate">0%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Processing Time</span>
                <span class="metric-value" id="processing-time">0 ms</span>
            </div>
        </div>
    </div>

    <!-- Camera Status -->
    <div class="card">
        <div class="card-header">Camera Status (Online: <span id="cameras-online">0</span>/20)</div>
        <div class="camera-grid" id="camera-grid"></div>
    </div>

    <!-- Performance Charts -->
    <div class="card">
        <div class="card-header">Performance History</div>
        <div class="chart-container">
            <canvas id="performance-chart"></canvas>
        </div>
    </div>

    <!-- Active Alerts -->
    <div class="card">
        <div class="card-header">Active Alerts</div>
        <div id="alerts-container">
            <div class="metric">
                <span class="metric-label">No active alerts</span>
            </div>
        </div>
    </div>

    <!-- Controls -->
    <div class="controls">
        <button class="button" onclick="clearAlerts()">Clear Alerts</button>
        <button class="button" onclick="refreshData()">Refresh Data</button>
        <button class="button" onclick="exportData()">Export Metrics</button>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();

        // Performance chart
        let performanceChart;
        const maxDataPoints = 60;
        const chartData = {
            labels: [],
            datasets: [
                {
                    label: 'CPU %',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                },
                {
                    label: 'Memory %',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                },
                {
                    label: 'Network %',
                    data: [],
                    borderColor: 'rgb(255, 205, 86)',
                    tension: 0.1
                }
            ]
        };

        // Initialize chart
        const ctx = document.getElementById('performance-chart').getContext('2d');
        performanceChart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: '#e0e0e0' },
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    },
                    x: {
                        ticks: { color: '#e0e0e0' },
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });

        // Connection status
        socket.on('connect', () => {
            document.getElementById('connection-status').className = 'status-indicator status-healthy';
            console.log('Connected to server');
        });

        socket.on('disconnect', () => {
            document.getElementById('connection-status').className = 'status-indicator status-critical';
            console.log('Disconnected from server');
        });

        // Update metrics
        socket.on('metrics_update', (data) => {
            updateSystemMetrics(data);
            updateCharts(data);
        });

        // Update alerts
        socket.on('alerts_update', (data) => {
            updateAlerts(data);
        });

        function updateSystemMetrics(metrics) {
            // System health
            document.getElementById('overall-status').textContent = metrics.overall_status;
            document.getElementById('overall-status').className =
                'metric-value ' + (metrics.overall_status === 'healthy' ? 'good' : 'bad');

            // CPU
            const cpuUtil = metrics.cpu_utilization.toFixed(1);
            document.getElementById('cpu-util').textContent = cpuUtil + '%';
            updateProgressBar('cpu-progress', metrics.cpu_utilization);

            // Memory
            const memUtil = metrics.memory_percent.toFixed(1);
            document.getElementById('memory-util').textContent = memUtil + '%';
            updateProgressBar('memory-progress', metrics.memory_percent);

            // Network
            document.getElementById('network-send').textContent =
                metrics.network_send_mbps.toFixed(2) + ' Mbps';
            document.getElementById('network-recv').textContent =
                metrics.network_recv_mbps.toFixed(2) + ' Mbps';
            const netUtil = ((metrics.network_send_mbps + metrics.network_recv_mbps) / 100).toFixed(1);
            document.getElementById('network-util').textContent = netUtil + '%';
            updateProgressBar('network-progress', parseFloat(netUtil));

            // Detection
            document.getElementById('confirmed-tracks').textContent = metrics.confirmed_tracks;
            document.getElementById('detection-rate').textContent =
                (metrics.detection_rate * 100).toFixed(1) + '%';
            document.getElementById('processing-time').textContent =
                metrics.processing_time_ms.toFixed(1) + ' ms';

            // Cameras
            document.getElementById('cameras-online').textContent = metrics.cameras_online;

            // Active alerts
            document.getElementById('active-alerts').textContent = metrics.active_alerts || 0;

            // GPU metrics
            if (metrics.gpu_metrics && metrics.gpu_metrics.length > 0) {
                updateGPUMetrics(metrics.gpu_metrics);
            }

            // Camera grid
            if (metrics.cameras) {
                updateCameraGrid(metrics.cameras);
            }
        }

        function updateProgressBar(id, value) {
            const bar = document.getElementById(id);
            bar.style.width = Math.min(value, 100) + '%';

            if (value < 70) {
                bar.className = 'progress-fill low';
            } else if (value < 85) {
                bar.className = 'progress-fill medium';
            } else {
                bar.className = 'progress-fill high';
            }
        }

        function updateGPUMetrics(gpus) {
            const container = document.getElementById('gpu-metrics');
            container.innerHTML = '';

            gpus.forEach(gpu => {
                container.innerHTML += `
                    <div class="metric">
                        <span class="metric-label">GPU ${gpu.gpu_id} Utilization</span>
                        <span class="metric-value">${gpu.utilization.toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Temperature</span>
                        <span class="metric-value ${gpu.temperature > 80 ? 'bad' : 'good'}">
                            ${gpu.temperature.toFixed(1)}°C
                        </span>
                    </div>
                `;
            });
        }

        function updateCameraGrid(cameras) {
            const grid = document.getElementById('camera-grid');
            grid.innerHTML = '';

            Object.values(cameras).forEach(cam => {
                const statusClass = cam.health_status === 'healthy' ? 'online' :
                                  cam.health_status === 'warning' ? 'warning' : 'offline';

                grid.innerHTML += `
                    <div class="camera-card ${statusClass}">
                        <div class="camera-id">Cam ${cam.camera_id}</div>
                        <div class="camera-status">${cam.fps.toFixed(1)} FPS</div>
                        <div class="camera-status">${cam.status}</div>
                    </div>
                `;
            });
        }

        function updateCharts(metrics) {
            const now = new Date().toLocaleTimeString();

            // Add new data point
            chartData.labels.push(now);
            chartData.datasets[0].data.push(metrics.cpu_utilization);
            chartData.datasets[1].data.push(metrics.memory_percent);
            chartData.datasets[2].data.push((metrics.network_send_mbps + metrics.network_recv_mbps) / 100);

            // Remove old data points
            if (chartData.labels.length > maxDataPoints) {
                chartData.labels.shift();
                chartData.datasets.forEach(dataset => dataset.data.shift());
            }

            performanceChart.update('none');
        }

        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');

            if (alerts.length === 0) {
                container.innerHTML = '<div class="metric"><span class="metric-label">No active alerts</span></div>';
                return;
            }

            container.innerHTML = '';
            alerts.forEach(alert => {
                const time = new Date(alert.timestamp * 1000).toLocaleString();
                container.innerHTML += `
                    <div class="alert ${alert.level.toLowerCase()}">
                        <div class="alert-title">[${alert.level}] ${alert.title}</div>
                        <div>${alert.message}</div>
                        <div class="alert-time">${time}</div>
                    </div>
                `;
            });
        }

        function clearAlerts() {
            socket.emit('clear_alerts');
        }

        function refreshData() {
            socket.emit('refresh_data');
        }

        function exportData() {
            socket.emit('export_metrics');
        }

        // Request initial data
        socket.emit('request_metrics');
    </script>
</body>
</html>
"""


class WebDashboard:
    """
    Web-based monitoring dashboard
    Provides real-time visualization of system metrics
    """

    def __init__(self,
                 host: str = '0.0.0.0',
                 port: int = 5000,
                 update_rate_hz: float = 2.0,
                 debug: bool = False):
        """
        Initialize web dashboard

        Args:
            host: Server host address
            port: Server port
            update_rate_hz: Metrics update rate (Hz)
            debug: Enable Flask debug mode
        """
        self.host = host
        self.port = port
        self.update_rate_hz = update_rate_hz
        self.update_interval = 1.0 / update_rate_hz
        self.debug = debug

        # Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'pixeltovoxel_monitoring_2024'

        # Socket.IO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')

        # References to monitoring components
        self.system_monitor = None
        self.alert_manager = None
        self.validator = None

        # Update thread
        self.running = False
        self.update_thread = None

        # Connected clients
        self.connected_clients = 0

        # Setup routes and events
        self._setup_routes()
        self._setup_socketio_events()

        logger.info(f"WebDashboard initialized: {host}:{port} @ {update_rate_hz}Hz")

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template_string(DASHBOARD_HTML)

        @self.app.route('/api/metrics')
        def api_metrics():
            """REST API endpoint for metrics"""
            if self.system_monitor:
                summary = self.system_monitor.get_summary()
                return jsonify(summary)
            return jsonify({'error': 'System monitor not configured'}), 503

        @self.app.route('/api/alerts')
        def api_alerts():
            """REST API endpoint for alerts"""
            if self.alert_manager:
                alerts = self.alert_manager.get_active_alerts()
                return jsonify([alert.to_dict() for alert in alerts])
            return jsonify({'error': 'Alert manager not configured'}), 503

        @self.app.route('/api/cameras')
        def api_cameras():
            """REST API endpoint for camera status"""
            if self.system_monitor:
                metrics = self.system_monitor.get_current_metrics()
                cameras = {}
                for cam_id, cam in metrics.cameras.items():
                    cameras[cam_id] = {
                        'camera_id': cam.camera_id,
                        'status': cam.status,
                        'fps': cam.fps,
                        'health_status': cam.health_status.value
                    }
                return jsonify(cameras)
            return jsonify({'error': 'System monitor not configured'}), 503

        @self.app.route('/api/statistics')
        def api_statistics():
            """REST API endpoint for statistics"""
            stats = {}

            if self.system_monitor:
                stats['monitoring'] = self.system_monitor.get_summary()

            if self.alert_manager:
                stats['alerts'] = self.alert_manager.get_statistics()

            if self.validator:
                stats['validation'] = self.validator.get_statistics()

            return jsonify(stats)

    def _setup_socketio_events(self):
        """Setup Socket.IO event handlers"""

        @self.socketio.on('connect')
        def handle_connect():
            """Client connected"""
            self.connected_clients += 1
            logger.info(f"Client connected (total: {self.connected_clients})")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Client disconnected"""
            self.connected_clients -= 1
            logger.info(f"Client disconnected (total: {self.connected_clients})")

        @self.socketio.on('request_metrics')
        def handle_request_metrics():
            """Client requesting metrics"""
            self._send_metrics_update()

        @self.socketio.on('clear_alerts')
        def handle_clear_alerts():
            """Clear all alerts"""
            if self.alert_manager:
                for alert_id in list(self.alert_manager.active_alerts.keys()):
                    self.alert_manager.resolve_alert(alert_id)
                logger.info("Alerts cleared via dashboard")

        @self.socketio.on('refresh_data')
        def handle_refresh_data():
            """Force refresh of all data"""
            self._send_metrics_update()
            self._send_alerts_update()

        @self.socketio.on('export_metrics')
        def handle_export_metrics():
            """Export metrics data"""
            if self.system_monitor:
                history = self.system_monitor.get_metrics_history(seconds=300)
                # Convert to serializable format
                export_data = []
                for metrics in history:
                    export_data.append({
                        'timestamp': metrics.timestamp,
                        'cpu_utilization': metrics.cpu.utilization,
                        'memory_percent': metrics.memory.percent,
                        'system_fps': metrics.system_fps
                    })

                logger.info(f"Exported {len(export_data)} metric samples")
                emit('export_data', {'metrics': export_data})

    def set_system_monitor(self, system_monitor):
        """Set reference to system monitor"""
        self.system_monitor = system_monitor
        logger.info("System monitor reference set")

    def set_alert_manager(self, alert_manager):
        """Set reference to alert manager"""
        self.alert_manager = alert_manager
        logger.info("Alert manager reference set")

    def set_validator(self, validator):
        """Set reference to validator"""
        self.validator = validator
        logger.info("Validator reference set")

    def start(self, blocking: bool = False):
        """
        Start web server

        Args:
            blocking: If True, blocks until server is stopped
        """
        # Start background update thread
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            logger.info("Dashboard update thread started")

        # Start Flask server
        logger.info(f"Starting web dashboard on http://{self.host}:{self.port}")

        if blocking:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug)
        else:
            # Run in separate thread
            server_thread = threading.Thread(
                target=lambda: self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug),
                daemon=True
            )
            server_thread.start()

    def stop(self):
        """Stop web server"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        logger.info("Dashboard stopped")

    def _update_loop(self):
        """Background loop for pushing updates to clients"""
        while self.running:
            if self.connected_clients > 0:
                try:
                    self._send_metrics_update()
                    self._send_alerts_update()
                except Exception as e:
                    logger.error(f"Update loop error: {e}")

            time.sleep(self.update_interval)

    def _send_metrics_update(self):
        """Send metrics update to all connected clients"""
        if not self.system_monitor:
            return

        try:
            # Get current metrics
            summary = self.system_monitor.get_summary()
            metrics = self.system_monitor.get_current_metrics()

            # Prepare data for transmission
            data = {
                'timestamp': time.time(),
                'overall_status': summary['overall_status'],
                'cpu_utilization': summary['cpu_utilization'],
                'memory_percent': summary['memory_percent'],
                'network_send_mbps': summary['network_send_mbps'],
                'network_recv_mbps': summary['network_recv_mbps'],
                'cameras_online': summary['cameras_online'],
                'cameras_total': summary['cameras_total'],
                'confirmed_tracks': summary['confirmed_tracks'],
                'detection_rate': summary['detection_rate'],
                'processing_time_ms': summary['processing_time_ms'],
                'active_alerts': len(self.alert_manager.active_alerts) if self.alert_manager else 0
            }

            # Add GPU metrics
            if metrics.gpus:
                data['gpu_metrics'] = [
                    {
                        'gpu_id': gpu.gpu_id,
                        'utilization': gpu.utilization,
                        'temperature': gpu.temperature,
                        'memory_used_mb': gpu.memory_used_mb,
                        'memory_total_mb': gpu.memory_total_mb
                    }
                    for gpu in metrics.gpus
                ]

            # Add camera details
            if metrics.cameras:
                data['cameras'] = {
                    cam_id: {
                        'camera_id': cam.camera_id,
                        'status': cam.status,
                        'fps': cam.fps,
                        'health_status': cam.health_status.value,
                        'drop_rate': cam.drop_rate
                    }
                    for cam_id, cam in metrics.cameras.items()
                }

            # Emit to all clients
            self.socketio.emit('metrics_update', data)

        except Exception as e:
            logger.error(f"Failed to send metrics update: {e}")

    def _send_alerts_update(self):
        """Send alerts update to all connected clients"""
        if not self.alert_manager:
            return

        try:
            # Get active alerts
            alerts = self.alert_manager.get_active_alerts()

            # Convert to serializable format
            alert_data = [
                {
                    'alert_id': alert.alert_id,
                    'level': alert.level.name,
                    'category': alert.category.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'camera_id': alert.camera_id
                }
                for alert in alerts[:10]  # Limit to 10 most recent
            ]

            # Emit to all clients
            self.socketio.emit('alerts_update', alert_data)

        except Exception as e:
            logger.error(f"Failed to send alerts update: {e}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Web Dashboard - Real-time Monitoring Interface")
    print("=" * 60)

    # Create dashboard
    dashboard = WebDashboard(
        host='0.0.0.0',
        port=5000,
        update_rate_hz=2.0
    )

    # Start server (blocking)
    print(f"\nDashboard available at: http://localhost:5000")
    print("Press Ctrl+C to stop\n")

    try:
        dashboard.start(blocking=True)
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        dashboard.stop()
