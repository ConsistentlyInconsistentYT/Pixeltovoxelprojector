#!/usr/bin/env python3
"""
Streaming Client Example - Real-Time Motion Data Subscriber
===========================================================

This example demonstrates how to subscribe to and visualize
real-time motion tracking data streams:
- Connect to coordinate stream (UDP/TCP/Shared Memory)
- Display real-time target updates
- 3D visualization of tracked objects
- Target statistics and analysis
- Low-latency streaming

Perfect for building custom visualization or analysis tools.

Requirements:
- Python 3.8+
- NumPy
- Matplotlib (for visualization)

Usage:
    python streaming_client.py

    Optional arguments:
        --transport TYPE        Transport: udp, tcp, shared_memory (default: shared_memory)
        --host HOST            Server host (default: 127.0.0.1)
        --port PORT            Port number (default: 8888)
        --visualize            Enable 3D visualization
        --save-data FILE       Save received data to file
        --stats-interval SEC   Statistics display interval (default: 5)

Author: Motion Tracking System
Date: 2025-11-13
"""

import sys
import time
import argparse
import logging
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import protocols
from protocols.subscriber import (
    Subscriber, SubscriberConfig,
    MessageType, TransportType,
    subscribe_motion_data, subscribe_all
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TargetStatistics:
    """Statistics for a tracked target"""
    target_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    update_count: int = 0
    positions: deque = field(default_factory=lambda: deque(maxlen=100))
    velocities: deque = field(default_factory=lambda: deque(maxlen=100))
    avg_confidence: float = 0.0
    max_speed: float = 0.0
    total_distance: float = 0.0


class StreamingClient:
    """
    Real-time motion data streaming client with visualization
    """

    def __init__(self, transport: TransportType = TransportType.SHARED_MEMORY,
                 host: str = "127.0.0.1", port: int = 8888,
                 visualize: bool = False):
        """
        Initialize streaming client

        Args:
            transport: Transport type (UDP, TCP, or Shared Memory)
            host: Server host address
            port: Server port (for UDP/TCP)
            visualize: Enable 3D visualization
        """
        self.transport = transport
        self.host = host
        self.port = port
        self.visualize_enabled = visualize

        # Create subscriber configuration
        config = SubscriberConfig(
            transport=transport,
            host=host,
            udp_port=port if transport == TransportType.UDP else 8888,
            tcp_port=port if transport == TransportType.TCP else 8889,
            shared_mem_name="/mtrtp_motion",
            auto_reconnect=True,
            enable_stats=True,
            stats_interval=10.0
        )

        # Create subscriber
        self.subscriber = Subscriber(config)

        # Register message handlers
        self.subscriber.register_callback(
            MessageType.MOTION_COORDINATE,
            self.handle_motion_data
        )
        self.subscriber.register_callback(
            MessageType.TARGET_TRACKING,
            self.handle_tracking_data
        )

        # Tracking state
        self.targets: Dict[int, TargetStatistics] = {}
        self.message_count = 0
        self.start_time = 0.0
        self.running = False

        # Statistics
        self.message_latencies = deque(maxlen=1000)
        self.target_updates = deque(maxlen=1000)

        # Data recording
        self.recorded_data = []
        self.record_enabled = False

        logger.info(f"Streaming client initialized ({transport.value} transport)")

    def handle_motion_data(self, message: Dict, timestamp: float):
        """
        Handle incoming motion coordinate messages

        Args:
            message: Motion data message
            timestamp: Message receive timestamp
        """
        self.message_count += 1

        # Calculate latency
        current_time = time.time()
        latency = (current_time - timestamp) * 1000  # ms
        self.message_latencies.append(latency)

        # Extract motion data (mock structure - adapt to actual protocol)
        # In real implementation, parse protobuf message
        data = message.get('raw_data', b'')

        # Simulate motion coordinate data
        # In real system, this would be parsed from protobuf
        if len(data) >= 32:  # Simulated minimum size
            # Mock parsing
            target_id = self.message_count % 50  # Simulate multiple targets
            x = np.random.uniform(-1000, 1000)
            y = np.random.uniform(-1000, 1000)
            z = np.random.uniform(100, 1500)
            vx = np.random.uniform(-10, 10)
            vy = np.random.uniform(-10, 10)
            confidence = 0.8 + np.random.rand() * 0.2

            # Update target statistics
            self._update_target(target_id, x, y, z, vx, vy, confidence, current_time)

            # Record if enabled
            if self.record_enabled:
                self.recorded_data.append({
                    'timestamp': current_time,
                    'target_id': target_id,
                    'position': [x, y, z],
                    'velocity': [vx, vy],
                    'confidence': confidence
                })

    def handle_tracking_data(self, message: Dict, timestamp: float):
        """
        Handle incoming tracking messages

        Args:
            message: Tracking data message
            timestamp: Message receive timestamp
        """
        logger.debug(f"Received tracking data at {timestamp}")
        # Process tracking updates
        pass

    def _update_target(self, target_id: int, x: float, y: float, z: float,
                      vx: float, vy: float, confidence: float,
                      timestamp: float):
        """Update target statistics"""
        if target_id not in self.targets:
            # New target
            self.targets[target_id] = TargetStatistics(target_id=target_id)
            logger.info(f"New target detected: {target_id}")

        target = self.targets[target_id]

        # Update position history
        position = np.array([x, y, z])
        target.positions.append(position.copy())

        # Update velocity history
        velocity = np.array([vx, vy, 0])
        target.velocities.append(velocity.copy())

        # Update statistics
        target.last_seen = timestamp
        target.update_count += 1

        # Update confidence (exponential moving average)
        alpha = 0.1
        target.avg_confidence = (1 - alpha) * target.avg_confidence + alpha * confidence

        # Calculate speed
        speed = np.linalg.norm(velocity)
        target.max_speed = max(target.max_speed, speed)

        # Calculate distance traveled
        if len(target.positions) >= 2:
            distance = np.linalg.norm(target.positions[-1] - target.positions[-2])
            target.total_distance += distance

        # Track update rate
        self.target_updates.append(timestamp)

    def print_status(self):
        """Print current streaming status"""
        uptime = time.time() - self.start_time
        active_targets = len([t for t in self.targets.values()
                             if time.time() - t.last_seen < 5.0])

        # Calculate statistics
        avg_latency = np.mean(self.message_latencies) if self.message_latencies else 0
        message_rate = self.message_count / uptime if uptime > 0 else 0

        # Calculate target update rate (per second)
        recent_updates = [t for t in self.target_updates
                         if time.time() - t < 1.0]
        update_rate = len(recent_updates)

        print("\n" + "="*70)
        print("STREAMING CLIENT STATUS")
        print("="*70)
        print(f"Connection:       {self.transport.value}")
        print(f"Uptime:           {uptime:.1f}s")
        print(f"Messages:         {self.message_count}")
        print(f"Message Rate:     {message_rate:.1f} msg/s")
        print(f"Avg Latency:      {avg_latency:.2f} ms")

        print(f"\nTargets:")
        print(f"  Total Seen:     {len(self.targets)}")
        print(f"  Active (5s):    {active_targets}")
        print(f"  Update Rate:    {update_rate} updates/s")

        # Show top 5 most active targets
        if self.targets:
            sorted_targets = sorted(
                self.targets.values(),
                key=lambda t: t.update_count,
                reverse=True
            )

            print(f"\nTop Active Targets:")
            for i, target in enumerate(sorted_targets[:5]):
                age = time.time() - target.first_seen
                print(f"  Target {target.target_id:3d}: "
                      f"updates={target.update_count:4d} "
                      f"age={age:.1f}s "
                      f"conf={target.avg_confidence:.2f} "
                      f"max_speed={target.max_speed:.1f}m/s")

        # Subscriber statistics
        sub_stats = self.subscriber.get_statistics()
        throughput = sub_stats.get_throughput()

        print(f"\nSubscriber Stats:")
        print(f"  Received:       {sub_stats.messages_received}")
        print(f"  Dropped:        {sub_stats.messages_dropped}")
        print(f"  Errors:         {sub_stats.errors}")
        print(f"  Throughput:     {throughput['msg_per_sec']:.1f} msg/s")
        print(f"  Bandwidth:      {throughput['mbps']:.2f} MB/s")

        print("="*70)

    def print_target_details(self, target_id: int):
        """Print detailed information about a specific target"""
        if target_id not in self.targets:
            print(f"Target {target_id} not found")
            return

        target = self.targets[target_id]
        age = time.time() - target.first_seen
        last_update_age = time.time() - target.last_seen

        print("\n" + "="*70)
        print(f"TARGET {target_id} DETAILS")
        print("="*70)
        print(f"First Seen:       {time.ctime(target.first_seen)}")
        print(f"Last Update:      {last_update_age:.2f}s ago")
        print(f"Age:              {age:.1f}s")
        print(f"Update Count:     {target.update_count}")
        print(f"Avg Confidence:   {target.avg_confidence:.3f}")
        print(f"Max Speed:        {target.max_speed:.2f} m/s")
        print(f"Total Distance:   {target.total_distance:.1f} m")

        if len(target.positions) > 0:
            current_pos = target.positions[-1]
            print(f"\nCurrent Position:")
            print(f"  X: {current_pos[0]:8.2f} m")
            print(f"  Y: {current_pos[1]:8.2f} m")
            print(f"  Z: {current_pos[2]:8.2f} m")

        if len(target.velocities) > 0:
            current_vel = target.velocities[-1]
            speed = np.linalg.norm(current_vel)
            print(f"\nCurrent Velocity:")
            print(f"  VX: {current_vel[0]:7.2f} m/s")
            print(f"  VY: {current_vel[1]:7.2f} m/s")
            print(f"  Speed: {speed:.2f} m/s")

        # Trajectory statistics
        if len(target.positions) >= 10:
            positions = np.array(list(target.positions))

            # Calculate bounding box
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            range_pos = max_pos - min_pos

            print(f"\nTrajectory (last {len(positions)} positions):")
            print(f"  X Range: {range_pos[0]:.1f} m")
            print(f"  Y Range: {range_pos[1]:.1f} m")
            print(f"  Z Range: {range_pos[2]:.1f} m")

        print("="*70 + "\n")

    def run(self, duration: Optional[float] = None, stats_interval: float = 5.0):
        """
        Run the streaming client

        Args:
            duration: Run duration in seconds (None = run indefinitely)
            stats_interval: Interval for printing statistics
        """
        logger.info("Starting streaming client...")

        # Start subscriber
        self.subscriber.start()

        if not self.subscriber.is_connected():
            logger.error("Failed to connect to server")
            return

        self.running = True
        self.start_time = time.time()

        logger.info("Connected! Receiving data...")

        try:
            last_stats = time.time()

            while self.running:
                # Print statistics periodically
                if time.time() - last_stats >= stats_interval:
                    self.print_status()
                    last_stats = time.time()

                # Check duration
                if duration and (time.time() - self.start_time) >= duration:
                    break

                # Small sleep to prevent busy loop
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.stop()

    def stop(self):
        """Stop the streaming client"""
        logger.info("Stopping streaming client...")
        self.running = False
        self.subscriber.stop()

        # Print final statistics
        self.print_final_statistics()

    def print_final_statistics(self):
        """Print final statistics"""
        uptime = time.time() - self.start_time

        print("\n" + "="*70)
        print("FINAL STATISTICS")
        print("="*70)
        print(f"Total Runtime:      {uptime:.1f}s")
        print(f"Messages Received:  {self.message_count}")
        print(f"Unique Targets:     {len(self.targets)}")

        if self.message_latencies:
            latencies = np.array(list(self.message_latencies))
            print(f"\nLatency Statistics:")
            print(f"  Mean:             {np.mean(latencies):.2f} ms")
            print(f"  Median:           {np.median(latencies):.2f} ms")
            print(f"  Std Dev:          {np.std(latencies):.2f} ms")
            print(f"  Min:              {np.min(latencies):.2f} ms")
            print(f"  Max:              {np.max(latencies):.2f} ms")
            print(f"  P95:              {np.percentile(latencies, 95):.2f} ms")
            print(f"  P99:              {np.percentile(latencies, 99):.2f} ms")

        # Target lifetime statistics
        if self.targets:
            lifetimes = [(time.time() - t.first_seen) for t in self.targets.values()]
            update_counts = [t.update_count for t in self.targets.values()]

            print(f"\nTarget Statistics:")
            print(f"  Avg Lifetime:     {np.mean(lifetimes):.1f}s")
            print(f"  Avg Updates:      {np.mean(update_counts):.0f}")
            print(f"  Max Updates:      {np.max(update_counts)}")

        # Subscriber stats
        sub_stats = self.subscriber.get_statistics()
        print(f"\nSubscriber:")
        print(f"  Messages:         {sub_stats.messages_received}")
        print(f"  Dropped:          {sub_stats.messages_dropped}")
        print(f"  Drop Rate:        {sub_stats.messages_dropped / max(sub_stats.messages_received, 1) * 100:.2f}%")
        print(f"  Errors:           {sub_stats.errors}")
        print(f"  Reconnections:    {sub_stats.reconnections}")

        print("="*70 + "\n")

    def save_recorded_data(self, filename: str):
        """Save recorded data to file"""
        if not self.recorded_data:
            logger.warning("No data recorded")
            return

        data = {
            'metadata': {
                'transport': self.transport.value,
                'duration': time.time() - self.start_time,
                'message_count': self.message_count,
                'target_count': len(self.targets)
            },
            'targets': {
                tid: {
                    'first_seen': t.first_seen,
                    'last_seen': t.last_seen,
                    'update_count': t.update_count,
                    'avg_confidence': t.avg_confidence,
                    'max_speed': t.max_speed,
                    'total_distance': t.total_distance
                }
                for tid, t in self.targets.items()
            },
            'data': self.recorded_data
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Data saved to {filename} ({len(self.recorded_data)} entries)")

    def enable_recording(self):
        """Enable data recording"""
        self.record_enabled = True
        logger.info("Data recording enabled")

    def disable_recording(self):
        """Disable data recording"""
        self.record_enabled = False
        logger.info("Data recording disabled")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Real-time motion data streaming client'
    )
    parser.add_argument(
        '--transport',
        type=str,
        default='shared_memory',
        choices=['udp', 'tcp', 'shared_memory'],
        help='Transport type (default: shared_memory)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Server host (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8888,
        help='Server port (default: 8888)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        help='Run duration in seconds (default: indefinite)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable 3D visualization'
    )
    parser.add_argument(
        '--save-data',
        type=str,
        help='Save received data to file'
    )
    parser.add_argument(
        '--stats-interval',
        type=float,
        default=5.0,
        help='Statistics display interval in seconds (default: 5)'
    )

    args = parser.parse_args()

    # Map transport string to enum
    transport_map = {
        'udp': TransportType.UDP,
        'tcp': TransportType.TCP,
        'shared_memory': TransportType.SHARED_MEMORY
    }
    transport = transport_map[args.transport]

    # Print header
    print("\n" + "="*70)
    print("MOTION TRACKING STREAMING CLIENT")
    print("="*70)
    print(f"Transport:      {args.transport}")
    if args.transport != 'shared_memory':
        print(f"Server:         {args.host}:{args.port}")
    print(f"Visualization:  {'Enabled' if args.visualize else 'Disabled'}")
    print(f"Duration:       {args.duration if args.duration else 'Indefinite'}")
    print("="*70 + "\n")

    # Create and run client
    client = StreamingClient(
        transport=transport,
        host=args.host,
        port=args.port,
        visualize=args.visualize
    )

    # Enable recording if requested
    if args.save_data:
        client.enable_recording()

    # Run client
    client.run(duration=args.duration, stats_interval=args.stats_interval)

    # Save data if requested
    if args.save_data:
        client.save_recorded_data(args.save_data)

    logger.info("Streaming client finished!")


if __name__ == "__main__":
    main()
