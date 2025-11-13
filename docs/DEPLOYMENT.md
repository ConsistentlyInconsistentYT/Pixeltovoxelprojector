# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the 8K Motion Tracking and Voxel Processing System in production environments, including hardware setup, network configuration, camera calibration, performance tuning, and troubleshooting.

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Network Configuration](#network-configuration)
3. [Camera Setup and Calibration](#camera-setup-and-calibration)
4. [Software Installation](#software-installation)
5. [System Configuration](#system-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Security Considerations](#security-considerations)
10. [Backup and Recovery](#backup-and-recovery)

---

## Hardware Requirements

### Minimum Configuration (Development/Testing)

#### Single Node Setup

- **CPU**: Intel Core i7-10700K or AMD Ryzen 7 3700X (8 cores)
- **RAM**: 32GB DDR4-3200
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **Storage**: 500GB NVMe SSD
- **Network**: 1GbE network adapter
- **Operating System**: Ubuntu 20.04 LTS or later

**Capacity**: 2 camera pairs (4 cameras)

### Recommended Configuration (Production)

#### Multi-Node Cluster

**Master Node**:
- **CPU**: Intel Core i9-13900K or AMD Ryzen 9 7950X (16+ cores)
- **RAM**: 64GB DDR5-5600
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) x1
- **Storage**: 2TB NVMe SSD (for OS and application)
- **Storage**: 8TB HDD (for data archival)
- **Network**: 10GbE or InfiniBand (100 Gbps)
- **Operating System**: Ubuntu 22.04 LTS

**Worker Nodes** (4+ nodes):
- **CPU**: Intel Core i9-13900K or AMD Ryzen 9 7950X
- **RAM**: 64GB DDR5-5600
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) x2-4
- **Storage**: 2TB NVMe SSD
- **Network**: 10GbE or InfiniBand
- **Operating System**: Ubuntu 22.04 LTS

**Capacity**: 10 camera pairs (20 cameras) per cluster

### Camera Hardware

**Per Camera Pair**:
- **Monochrome Camera**: 8K (7680x4320) GigE Vision camera
  - Frame rate: 30 FPS minimum
  - Sensor: Monochrome CMOS (1" or larger)
  - Interface: GigE Vision (10GbE preferred)
  - Triggering: Hardware trigger support
  - Example: FLIR Blackfly S BFS-U3-200S6M-C

- **Thermal Camera**: 8K thermal imaging camera
  - Frame rate: 30 FPS minimum
  - Sensor: Uncooled microbolometer
  - Interface: GigE Vision (10GbE preferred)
  - Triggering: Hardware trigger support
  - Temperature range: -20°C to 150°C
  - Example: FLIR A700 series

**Camera Mounting**:
- Rigid mounting bracket for camera pair
- Stereo baseline: 0.3-0.8 meters (adjustable)
- Vibration isolation mounts
- Weather-resistant enclosure (for outdoor deployment)

### Network Infrastructure

**Network Switches**:
- 10GbE managed switch with 24+ ports
- Jumbo frame support (MTU 9000)
- VLAN support
- QoS/traffic shaping
- Low latency (<1 microsecond)
- Example: Cisco Nexus 3000 series, Arista 7050 series

**Network Cables**:
- Cat6A or Cat7 Ethernet cables for 10GbE
- Maximum cable length: 100 meters for Cat6A
- Proper cable management and labeling

**Optional: InfiniBand**:
- Mellanox ConnectX-5 or newer adapters (100 Gbps)
- InfiniBand switches (e.g., Mellanox SN2700)
- QSFP28 cables

### Power Requirements

**Per Worker Node**:
- Power supply: 1200W+ 80 Plus Platinum
- UPS backup: 2000VA for critical nodes
- Power consumption: ~600-800W under load

**Total Cluster** (Master + 4 Workers):
- Estimated power: 3-4 kW
- Recommended UPS: 6000VA+ rack-mount UPS
- Redundant power supplies recommended

### Rack and Cooling

**Rack Requirements**:
- Standard 19" server rack
- 42U minimum height
- Proper cable management
- KVM switch for management

**Cooling**:
- Target ambient temperature: 20-22°C
- Airflow: Front-to-back
- Minimum 1000 CFM for 5-node cluster
- Hot aisle/cold aisle configuration recommended

---

## Network Configuration

### Network Topology

#### Option 1: Flat Network (Simple, <10 nodes)

```
┌─────────────────────────────────────────────────────────┐
│                   10GbE Switch                          │
│                                                          │
│  Port 1-10: Cameras (192.168.1.10-19)                  │
│  Port 11: Master Node (192.168.1.100)                  │
│  Port 12-15: Worker Nodes (192.168.1.101-104)          │
│  Port 20: Management/Internet Gateway                   │
└─────────────────────────────────────────────────────────┘
```

#### Option 2: Segmented Network (Production, 10+ nodes)

```
┌────────────────────────────────────────────────────────────┐
│                  Core Switch (10GbE)                       │
└───────┬──────────────────┬──────────────────┬─────────────┘
        │                  │                  │
┌───────▼──────────┐ ┌─────▼──────────┐ ┌───▼──────────────┐
│ Camera Network   │ │ Cluster Network│ │ Management Net   │
│ VLAN 10          │ │ VLAN 20        │ │ VLAN 99          │
│ 192.168.10.0/24  │ │ 10.0.0.0/24    │ │ 192.168.99.0/24  │
│                  │ │                │ │                  │
│ Cameras 1-20     │ │ Master + Workers│ │ Admin Access     │
└──────────────────┘ └────────────────┘ └──────────────────┘
```

### IP Address Assignment

**Camera Network** (192.168.10.0/24):
```
192.168.10.10-29    Monochrome cameras 0-9 (pairs 0-9)
192.168.10.30-49    Thermal cameras 0-9 (pairs 0-9)
192.168.10.1        Gateway/DHCP server
```

**Cluster Network** (10.0.0.0/24):
```
10.0.0.1            Master node
10.0.0.10-13        Worker nodes 0-3
10.0.0.100-103      Worker node IB/RDMA interfaces
```

**Management Network** (192.168.99.0/24):
```
192.168.99.1        Core switch management
192.168.99.10       Master node management
192.168.99.11-14    Worker node management interfaces
```

### Network Settings

#### MTU Configuration

Enable jumbo frames for better throughput:

```bash
# On all nodes and switches
sudo ip link set eth0 mtu 9000

# Make persistent (Ubuntu)
cat <<EOF | sudo tee -a /etc/netplan/01-network.yaml
network:
  version: 2
  ethernets:
    eth0:
      mtu: 9000
      dhcp4: no
      addresses:
        - 10.0.0.1/24
EOF

sudo netplan apply
```

#### Network Buffer Tuning

Optimize network buffers for high throughput:

```bash
# Increase network buffer sizes
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.rmem_default=26214400
sudo sysctl -w net.core.wmem_default=26214400

# TCP tuning
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr

# Make persistent
cat <<EOF | sudo tee -a /etc/sysctl.conf
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 26214400
net.core.wmem_default = 26214400
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
EOF
```

#### Switch Configuration

**Cisco Switch Example**:

```
! Enable jumbo frames
system mtu jumbo 9000

! Configure VLAN for cameras
vlan 10
  name Camera-Network

! Configure VLAN for cluster
vlan 20
  name Cluster-Network

! Configure port for cameras
interface GigabitEthernet1/0/1-20
  switchport mode access
  switchport access vlan 10
  mtu 9000
  spanning-tree portfast

! Configure port for cluster nodes
interface TenGigabitEthernet1/0/1-5
  switchport mode access
  switchport access vlan 20
  mtu 9000

! Enable QoS for real-time traffic
mls qos
class-map match-all CAMERA-TRAFFIC
  match vlan 10
policy-map CAMERA-POLICY
  class CAMERA-TRAFFIC
    priority
```

### Firewall Configuration

Allow required ports:

```bash
# Camera ports (GigE Vision)
sudo ufw allow from 192.168.10.0/24 to any port 3956 proto tcp
sudo ufw allow from 192.168.10.0/24 to any port 3956 proto udp

# Cluster communication
sudo ufw allow from 10.0.0.0/24 to any port 10000:11000 proto tcp
sudo ufw allow from 10.0.0.0/24 to any port 9999 proto udp

# Monitoring
sudo ufw allow from 192.168.99.0/24 to any port 22 proto tcp
sudo ufw allow from 192.168.99.0/24 to any port 9090 proto tcp  # Prometheus
sudo ufw allow from 192.168.99.0/24 to any port 3000 proto tcp  # Grafana

sudo ufw enable
```

---

## Camera Setup and Calibration

### Physical Installation

#### 1. Mounting Cameras

**Camera Pair Mounting**:

```
  Mono Camera          Thermal Camera
       │                      │
       │  ←── Baseline ────→  │
       │                      │
       └──────────┬───────────┘
                  │
           Mounting Bracket
```

**Requirements**:
- Rigid mounting bracket (aluminum or steel)
- Stereo baseline: 0.5 meters (adjustable 0.3-0.8m)
- Parallel optical axes (±0.1° tolerance)
- Vibration isolation dampers
- Cable strain relief

#### 2. Network Connection

```bash
# Connect cameras to switch
# Label cables: "Pair0-Mono", "Pair0-Thermal", etc.

# Verify camera discovery
arv-tool-0.8 -l

# Should list all connected cameras with IPs
```

#### 3. Initial Power-On

```bash
# Set static IP for each camera (via web interface or SDK)
# Mono cameras: 192.168.10.10-29
# Thermal cameras: 192.168.10.30-49

# Test connectivity
ping 192.168.10.10  # Pair 0 Mono
ping 192.168.10.30  # Pair 0 Thermal
```

### Camera Configuration

Create camera configuration file:

```json
{
  "num_pairs": 10,
  "cameras": {
    "0": {
      "camera_id": 0,
      "pair_id": 0,
      "camera_type": "MONO",
      "connection": "GIGE_VISION",
      "ip_address": "192.168.10.10",
      "mac_address": "00:11:1C:XX:XX:XX",
      "width": 7680,
      "height": 4320,
      "frame_rate": 30.0,
      "exposure_time": 10000.0,
      "gain": 1.0,
      "trigger_mode": "Hardware",
      "trigger_source": "Line1",
      "packet_size": 9000,
      "packet_delay": 1000,
      "position": [0.0, 0.0, 10.0],
      "orientation": [[1,0,0], [0,1,0], [0,0,1]],
      "intrinsic_file": "/opt/calibration/camera0_intrinsic.json",
      "extrinsic_file": "/opt/calibration/camera0_extrinsic.json"
    },
    "1": {
      "camera_id": 1,
      "pair_id": 0,
      "camera_type": "THERMAL",
      "connection": "GIGE_VISION",
      "ip_address": "192.168.10.30",
      "mac_address": "00:11:1C:XX:XX:XX",
      "width": 7680,
      "height": 4320,
      "frame_rate": 30.0,
      "exposure_time": 10000.0,
      "gain": 1.0,
      "trigger_mode": "Hardware",
      "trigger_source": "Line1",
      "packet_size": 9000,
      "packet_delay": 1000,
      "position": [0.5, 0.0, 10.0],
      "orientation": [[1,0,0], [0,1,0], [0,0,1]],
      "intrinsic_file": "/opt/calibration/camera1_intrinsic.json",
      "extrinsic_file": "/opt/calibration/camera1_extrinsic.json"
    }
  },
  "pairs": {
    "0": {
      "pair_id": 0,
      "mono_camera_id": 0,
      "thermal_camera_id": 1,
      "stereo_baseline": 0.5
    }
  }
}
```

### Intrinsic Calibration

Calibrate each camera's intrinsic parameters:

```bash
# 1. Collect calibration images (30+ images of checkerboard)
python scripts/collect_calibration_images.py \
    --camera-ip 192.168.10.10 \
    --output-dir /tmp/calibration/camera0 \
    --num-images 30

# 2. Run calibration
python scripts/calibrate_camera.py \
    --input-dir /tmp/calibration/camera0 \
    --output /opt/calibration/camera0_intrinsic.json \
    --board-size 9x6 \
    --square-size 0.025  # 25mm squares

# Output format:
{
  "camera_matrix": [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
  ],
  "distortion_coefficients": [k1, k2, p1, p2, k3],
  "image_size": [7680, 4320],
  "reprojection_error": 0.32
}
```

### Stereo Calibration

Calibrate stereo pairs:

```bash
# 1. Collect synchronized calibration images
python scripts/collect_stereo_calibration.py \
    --mono-ip 192.168.10.10 \
    --thermal-ip 192.168.10.30 \
    --output-dir /tmp/calibration/pair0 \
    --num-images 30

# 2. Run stereo calibration
python scripts/calibrate_stereo_pair.py \
    --input-dir /tmp/calibration/pair0 \
    --mono-intrinsic /opt/calibration/camera0_intrinsic.json \
    --thermal-intrinsic /opt/calibration/camera1_intrinsic.json \
    --output /opt/calibration/pair0_stereo.json

# Output format:
{
  "rotation_matrix": [[...], [...], [...]],
  "translation_vector": [tx, ty, tz],
  "essential_matrix": [[...], [...], [...]],
  "fundamental_matrix": [[...], [...], [...]],
  "stereo_baseline": 0.502,  # meters
  "reprojection_error": 0.45
}
```

### Extrinsic Calibration

Calibrate camera positions in world coordinates:

```bash
# Use reference targets at known positions
python scripts/calibrate_extrinsic.py \
    --camera-config /opt/config/camera_config.json \
    --reference-targets /opt/calibration/reference_targets.json \
    --output-dir /opt/calibration/

# reference_targets.json format:
{
  "targets": [
    {
      "id": 0,
      "position": [0.0, 0.0, 0.0],  # Known 3D position
      "size": 0.5  # Target size in meters
    },
    ...
  ]
}
```

### Validation

Validate calibration quality:

```bash
python scripts/validate_calibration.py \
    --camera-config /opt/config/camera_config.json \
    --calibration-dir /opt/calibration/ \
    --validation-targets /opt/calibration/validation_targets.json

# Expected output:
# Camera 0 reprojection error: 0.32 pixels (good)
# Camera 1 reprojection error: 0.38 pixels (good)
# Pair 0 stereo error: 0.45 pixels (good)
# Pair 0 depth accuracy: 98.5% (excellent)
```

---

## Software Installation

### Operating System Setup

```bash
# 1. Install Ubuntu 22.04 LTS Server
# Download from: https://ubuntu.com/download/server

# 2. Update system
sudo apt-get update
sudo apt-get upgrade -y

# 3. Install build essentials
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    net-tools
```

### CUDA Installation

```bash
# 1. Install NVIDIA drivers
sudo apt-get install -y nvidia-driver-535

# 2. Download CUDA Toolkit 12.0
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run

# 3. Install CUDA
sudo sh cuda_12.0.0_525.60.13_linux.run --silent --toolkit

# 4. Set environment variables
cat <<EOF >> ~/.bashrc
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
EOF

source ~/.bashrc

# 5. Verify installation
nvcc --version
nvidia-smi
```

### Python and Dependencies

```bash
# 1. Install Python 3.10
sudo apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip

# 2. Create virtual environment
python3.10 -m venv /opt/motion-tracking/venv
source /opt/motion-tracking/venv/bin/activate

# 3. Install Python packages
pip install --upgrade pip setuptools wheel
pip install \
    numpy==1.24.3 \
    opencv-python==4.7.0 \
    pybind11==2.10.4 \
    protobuf==4.22.3 \
    grpcio==1.53.0 \
    grpcio-tools==1.53.0 \
    pyzmq==25.0.2 \
    scipy==1.10.1 \
    scikit-image==0.20.0
```

### Application Installation

```bash
# 1. Clone repository
cd /opt
sudo git clone <repository-url> motion-tracking
cd motion-tracking

# 2. Set ownership
sudo chown -R $USER:$USER /opt/motion-tracking

# 3. Build C++ extensions
python setup.py build_ext --inplace

# 4. Verify build
python verify_tracking_system.py

# 5. Install systemd service
sudo cp scripts/motion-tracking.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable motion-tracking
```

### RDMA/InfiniBand Setup (Optional)

```bash
# 1. Install RDMA packages
sudo apt-get install -y \
    rdma-core \
    libibverbs1 \
    librdmacm1 \
    ibverbs-utils \
    infiniband-diags

# 2. Load kernel modules
sudo modprobe ib_uverbs
sudo modprobe rdma_ucm
sudo modprobe mlx5_core
sudo modprobe mlx5_ib

# 3. Verify RDMA devices
ibv_devices
# Should list Mellanox adapters

# 4. Test RDMA performance
ib_write_bw -d mlx5_0

# 5. Configure persistent modules
cat <<EOF | sudo tee -a /etc/modules
ib_uverbs
rdma_ucm
mlx5_core
mlx5_ib
EOF
```

---

## System Configuration

### Directory Structure

```bash
# Create standard directory structure
sudo mkdir -p /opt/motion-tracking/{bin,lib,config,calibration,logs,data}
sudo mkdir -p /var/log/motion-tracking
sudo mkdir -p /var/lib/motion-tracking

# Set permissions
sudo chown -R motion-tracking:motion-tracking /opt/motion-tracking
sudo chown -R motion-tracking:motion-tracking /var/log/motion-tracking
sudo chown -R motion-tracking:motion-tracking /var/lib/motion-tracking
```

### Configuration Files

**Main Configuration** (`/opt/motion-tracking/config/system.yaml`):

```yaml
system:
  deployment_mode: production  # development, production
  log_level: INFO  # DEBUG, INFO, WARNING, ERROR
  log_file: /var/log/motion-tracking/system.log

cluster:
  node_id: master
  is_master: true
  discovery_port: 9999
  data_port_range: [10000, 11000]
  heartbeat_interval: 1.0
  heartbeat_timeout: 5.0
  enable_rdma: true
  rdma_device: mlx5_0

cameras:
  config_file: /opt/motion-tracking/config/camera_config.json
  num_pairs: 10
  frame_rate: 30.0
  buffer_size: 60
  auto_reconnect: true
  max_reconnect_attempts: 10

video_processing:
  use_hardware_accel: true
  codec: hevc
  enable_profiling: false
  motion_threshold: 20
  min_object_area: 100

fusion:
  enable_cuda: true
  registration_update_interval: 1.0
  thermal_threshold: 0.3
  mono_threshold: 0.2
  confidence_threshold: 0.6
  enable_false_positive_reduction: true

voxel:
  resolution: 512
  voxel_size: 0.1  # meters
  use_cuda: true

monitoring:
  enable_metrics: true
  metrics_port: 9090
  collect_interval: 1.0
  enable_grafana: true
  grafana_port: 3000
```

### Systemd Service

**Service File** (`/etc/systemd/system/motion-tracking.service`):

```ini
[Unit]
Description=8K Motion Tracking System
After=network.target

[Service]
Type=simple
User=motion-tracking
Group=motion-tracking
WorkingDirectory=/opt/motion-tracking
Environment="PATH=/opt/motion-tracking/venv/bin:/usr/local/cuda/bin:/usr/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
Environment="PYTHONPATH=/opt/motion-tracking/src"
ExecStart=/opt/motion-tracking/venv/bin/python /opt/motion-tracking/src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Start the service:

```bash
sudo systemctl start motion-tracking
sudo systemctl status motion-tracking
sudo journalctl -u motion-tracking -f
```

---

## Performance Tuning

### GPU Optimization

```bash
# 1. Set GPU performance mode
sudo nvidia-smi -pm 1

# 2. Set maximum clock speeds
sudo nvidia-smi -ac 7001,1980  # Memory,GPU clocks (RTX 4090)

# 3. Disable ECC (for maximum performance, data center only)
sudo nvidia-smi -e 0

# 4. Set persistence mode
sudo nvidia-smi -pm 1

# 5. Create startup script
cat <<EOF | sudo tee /etc/systemd/system/gpu-performance.service
[Unit]
Description=GPU Performance Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -ac 7001,1980
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable gpu-performance
sudo systemctl start gpu-performance
```

### CPU Optimization

```bash
# 1. Set CPU governor to performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu
done

# 2. Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# 3. Configure NUMA (if applicable)
sudo apt-get install numactl

# Run application with NUMA binding
numactl --cpubind=0 --membind=0 python main.py

# 4. Set process priority
sudo renice -n -10 -p $(pgrep -f motion-tracking)
```

### Memory Optimization

```bash
# 1. Increase shared memory
sudo sysctl -w kernel.shmmax=68719476736  # 64GB
sudo sysctl -w kernel.shmall=16777216

# 2. Configure huge pages
sudo sysctl -w vm.nr_hugepages=2048
echo 2048 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# 3. Reduce swappiness
sudo sysctl -w vm.swappiness=10

# 4. Make persistent
cat <<EOF | sudo tee -a /etc/sysctl.conf
kernel.shmmax = 68719476736
kernel.shmall = 16777216
vm.nr_hugepages = 2048
vm.swappiness = 10
EOF
```

### I/O Optimization

```bash
# 1. Set I/O scheduler to deadline
echo deadline | sudo tee /sys/block/nvme0n1/queue/scheduler

# 2. Increase read-ahead
sudo blockdev --setra 8192 /dev/nvme0n1

# 3. Configure file system (XFS recommended)
sudo mkfs.xfs -f -K /dev/nvme0n1p1
sudo mount -o noatime,nodiratime,logbufs=8 /dev/nvme0n1p1 /data
```

---

## Monitoring and Maintenance

### Monitoring Setup

**Install Prometheus**:

```bash
# 1. Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.42.0/prometheus-2.42.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# 2. Configure Prometheus
cat <<EOF > prometheus.yml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'motion-tracking'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
EOF

# 3. Start Prometheus
./prometheus --config.file=prometheus.yml

# 4. Access UI: http://localhost:9090
```

**Install Grafana**:

```bash
# 1. Install Grafana
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana

# 2. Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# 3. Access UI: http://localhost:3000 (admin/admin)
```

### Health Checks

**Create health check script** (`/opt/motion-tracking/scripts/health_check.sh`):

```bash
#!/bin/bash

# Check if service is running
if ! systemctl is-active --quiet motion-tracking; then
    echo "ERROR: Service not running"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: GPU not accessible"
    exit 1
fi

# Check camera connectivity
python3 <<EOF
from src.camera import CameraManager
mgr = CameraManager()
mgr.load_configuration()
if not mgr.initialize_all_cameras():
    print("ERROR: Camera initialization failed")
    exit(1)
EOF

# Check disk space
DISK_USAGE=$(df -h /opt/motion-tracking | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "WARNING: Disk usage above 90%"
fi

echo "Health check passed"
```

Run periodically:

```bash
# Add to crontab
crontab -e

# Run health check every 5 minutes
*/5 * * * * /opt/motion-tracking/scripts/health_check.sh >> /var/log/motion-tracking/health_check.log 2>&1
```

### Log Rotation

```bash
# Configure logrotate
sudo cat <<EOF > /etc/logrotate.d/motion-tracking
/var/log/motion-tracking/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 motion-tracking motion-tracking
    sharedscripts
    postrotate
        systemctl reload motion-tracking > /dev/null 2>&1 || true
    endscript
}
EOF
```

---

## Troubleshooting Guide

### Common Issues

#### Issue 1: Camera Connection Failures

**Symptoms**:
- Cameras not detected
- Connection timeouts
- Packet loss

**Diagnosis**:
```bash
# Check network connectivity
ping 192.168.10.10

# Check camera discovery
arv-tool-0.8 -l

# Check packet loss
ping -c 100 -s 8000 192.168.10.10
```

**Solutions**:
```bash
# 1. Check network settings
sudo ethtool eth0

# 2. Increase network buffers (see Network Configuration section)

# 3. Verify jumbo frames
ip link show eth0 | grep mtu

# 4. Check firewall
sudo ufw status

# 5. Restart network
sudo systemctl restart networking
```

#### Issue 2: Low FPS / High Latency

**Symptoms**:
- Processing FPS < 30
- Latency > 33ms
- GPU utilization low

**Diagnosis**:
```bash
# Check GPU usage
nvidia-smi dmon -s u

# Check CPU usage
htop

# Check application metrics
curl http://localhost:9090/metrics
```

**Solutions**:
```bash
# 1. Enable hardware acceleration
# Edit /opt/motion-tracking/config/system.yaml
# Set use_hardware_accel: true

# 2. Reduce buffer size
# Set buffer_size: 30 (from 60)

# 3. Check GPU clocks
nvidia-smi -q -d CLOCK

# 4. Increase processing threads
# Edit config, set num_processing_threads: 16

# 5. Profile application
python -m cProfile -o profile.stats main.py
```

#### Issue 3: Memory Issues

**Symptoms**:
- Out of memory errors
- High swap usage
- System instability

**Diagnosis**:
```bash
# Check memory usage
free -h

# Check GPU memory
nvidia-smi

# Check process memory
ps aux --sort=-%mem | head -20
```

**Solutions**:
```bash
# 1. Reduce buffer sizes
# Edit config: buffer_size: 30

# 2. Enable sparse voxel storage
# Edit config: enable_sparse_storage: true

# 3. Reduce number of camera pairs
# Edit config: num_pairs: 5

# 4. Add swap space
sudo dd if=/dev/zero of=/swapfile bs=1G count=32
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Issue 4: Calibration Errors

**Symptoms**:
- High reprojection errors
- Poor stereo matching
- Fusion alignment issues

**Diagnosis**:
```bash
# Validate calibration
python scripts/validate_calibration.py

# Check calibration files
cat /opt/calibration/camera0_intrinsic.json
```

**Solutions**:
```bash
# 1. Recalibrate cameras
python scripts/calibrate_camera.py ...

# 2. Use more calibration images (50+)

# 3. Ensure good lighting conditions

# 4. Use larger calibration target

# 5. Check camera mounting rigidity
```

#### Issue 5: Network Issues

**Symptoms**:
- Node disconnections
- High latency between nodes
- Data loss

**Diagnosis**:
```bash
# Check network latency
ping -c 100 10.0.0.10

# Check network bandwidth
iperf3 -c 10.0.0.10 -t 30

# Check network errors
netstat -i
```

**Solutions**:
```bash
# 1. Enable RDMA (if available)
# Edit config: enable_rdma: true

# 2. Adjust TCP settings (see Network Configuration)

# 3. Check switch configuration

# 4. Reduce network traffic on VLAN

# 5. Use dedicated network for cluster
```

### Diagnostic Commands

```bash
# System health
/opt/motion-tracking/scripts/health_check.sh

# GPU status
nvidia-smi

# Service status
sudo systemctl status motion-tracking

# View logs
sudo journalctl -u motion-tracking -f

# Check metrics
curl http://localhost:9090/metrics

# Network diagnostics
sudo netstat -tuln | grep LISTEN
sudo ss -s

# Process information
ps aux | grep motion-tracking
top -p $(pgrep -f motion-tracking)
```

---

## Security Considerations

### Network Security

```bash
# 1. Segment networks with VLANs

# 2. Enable firewall
sudo ufw enable

# 3. Restrict SSH access
sudo ufw limit ssh
sudo ufw allow from 192.168.99.0/24 to any port 22

# 4. Disable unused services
sudo systemctl disable bluetooth
sudo systemctl disable cups
```

### Application Security

```bash
# 1. Run as non-root user
sudo useradd -r -s /bin/false motion-tracking

# 2. Set file permissions
sudo chmod 700 /opt/motion-tracking/config
sudo chmod 600 /opt/motion-tracking/config/*.yaml

# 3. Encrypt calibration data
gpg --encrypt /opt/calibration/camera0_intrinsic.json

# 4. Enable audit logging
sudo apt-get install auditd
sudo auditctl -w /opt/motion-tracking -p wa
```

---

## Backup and Recovery

### Backup Strategy

```bash
# 1. Backup configuration
tar czf config-$(date +%Y%m%d).tar.gz /opt/motion-tracking/config

# 2. Backup calibration
tar czf calibration-$(date +%Y%m%d).tar.gz /opt/calibration

# 3. Backup data
rsync -av /var/lib/motion-tracking /backup/

# 4. Automated backups
cat <<EOF > /opt/motion-tracking/scripts/backup.sh
#!/bin/bash
DATE=\$(date +%Y%m%d)
tar czf /backup/config-\$DATE.tar.gz /opt/motion-tracking/config
tar czf /backup/calibration-\$DATE.tar.gz /opt/calibration
find /backup -name "*.tar.gz" -mtime +30 -delete
EOF

chmod +x /opt/motion-tracking/scripts/backup.sh

# Add to crontab: Daily at 2 AM
0 2 * * * /opt/motion-tracking/scripts/backup.sh
```

### Recovery Procedures

```bash
# 1. Restore configuration
tar xzf config-20240101.tar.gz -C /

# 2. Restore calibration
tar xzf calibration-20240101.tar.gz -C /

# 3. Rebuild application
cd /opt/motion-tracking
python setup.py build_ext --inplace

# 4. Restart service
sudo systemctl restart motion-tracking
```

---

## Deployment Checklist

- [ ] Hardware installed and powered on
- [ ] Network configured (IP addresses, MTU, VLANs)
- [ ] Cameras mounted and connected
- [ ] CUDA and drivers installed
- [ ] Application installed and built
- [ ] Configuration files created
- [ ] Cameras calibrated (intrinsic + extrinsic)
- [ ] Stereo pairs calibrated
- [ ] System service configured and enabled
- [ ] Firewall configured
- [ ] Monitoring setup (Prometheus + Grafana)
- [ ] Log rotation configured
- [ ] Health checks configured
- [ ] Backup system configured
- [ ] Documentation reviewed
- [ ] Performance testing completed
- [ ] Security audit completed

---

## Support and Maintenance

### Regular Maintenance Tasks

**Daily**:
- Monitor system health dashboard
- Check logs for errors
- Verify camera connectivity

**Weekly**:
- Review performance metrics
- Check disk space
- Verify backups

**Monthly**:
- Update system packages
- Review and optimize configuration
- Test backup restoration
- Recalibrate cameras (if needed)

**Quarterly**:
- Hardware inspection
- Cable management review
- Security audit
- Performance benchmarking

### Contact Information

For deployment support, please contact:
- Technical Support: support@example.com
- Emergency Hotline: +1-XXX-XXX-XXXX
- Documentation: https://docs.example.com

---

## Appendix

### A. Network Bandwidth Calculation

For 10 camera pairs at 8K resolution:

```
Frame size: 7680 x 4320 x 1 byte = 33.2 MB
Frame rate: 30 FPS
Cameras: 20

Bandwidth per camera: 33.2 MB × 30 FPS = 996 MB/s = 7.97 Gbps
Total bandwidth: 7.97 Gbps × 20 = 159.4 Gbps

Recommendation: 10GbE per 2 cameras, or 100 Gbps InfiniBand for full system
```

### B. Storage Requirements

```
Recording rate: 33.2 MB/frame × 30 FPS × 20 cameras = 19.9 GB/s
Per hour: 71.6 TB
Per day: 1.72 PB

Recommendation:
- Real-time processing without storage
- Record detections only (~100 MB/hour)
- Archive calibration data (~10 GB)
```

### C. Power Consumption

```
Master node: 600W
Worker nodes: 700W × 4 = 2800W
Cameras: 30W × 20 = 600W
Network: 200W
Total: 4200W

UPS recommendation: 6000VA (1 hour runtime)
```
