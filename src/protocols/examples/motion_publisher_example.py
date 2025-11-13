#!/usr/bin/env python3
"""
Motion Data Publisher Example

Demonstrates how to publish motion tracking data using MTRTP.
This example simulates a motion tracking system generating
real-time 3D coordinate data.

Usage:
    python motion_publisher_example.py [--transport TYPE] [--rate HZ]

Examples:
    python motion_publisher_example.py --transport shared_memory --rate 1000
    python motion_publisher_example.py --transport udp --rate 100
"""

import argparse
import time
import math
import numpy as np
from dataclasses import dataclass
from typing import List

# Note: In real implementation, would import actual protobuf messages
# from motion_protocol_pb2 import *


@dataclass
class MotionPoint:
    """Simulated motion point"""
    point_id: int
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    confidence: float = 1.0


class MotionSimulator:
    """Simulates motion tracking data"""

    def __init__(self, num_points: int = 10):
        self.num_points = num_points
        self.time = 0.0
        self.points = []

        # Initialize points in circular pattern
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            radius = 1.0

            point = MotionPoint(
                point_id=i,
                x=radius * math.cos(angle),
                y=radius * math.sin(angle),
                z=0.0
            )
            self.points.append(point)

    def update(self, dt: float):
        """Update motion simulation"""
        self.time += dt

        for i, point in enumerate(self.points):
            # Circular motion
            angle = (2 * math.pi * i) / self.num_points + self.time
            radius = 1.0 + 0.2 * math.sin(self.time * 2)

            # Update position
            new_x = radius * math.cos(angle)
            new_y = radius * math.sin(angle)
            new_z = 0.3 * math.sin(self.time * 3 + i)

            # Calculate velocity
            point.vx = (new_x - point.x) / dt
            point.vy = (new_y - point.y) / dt
            point.vz = (new_z - point.z) / dt

            point.x = new_x
            point.y = new_y
            point.z = new_z

            # Vary confidence based on position
            point.confidence = 0.8 + 0.2 * math.sin(self.time + i)

    def get_points(self) -> List[MotionPoint]:
        """Get current motion points"""
        return self.points


class MockPublisher:
    """Mock publisher for demonstration (would use actual StreamManager)"""

    def __init__(self, transport: str, config: dict):
        self.transport = transport
        self.config = config
        self.message_count = 0
        self.start_time = time.time()

    def publish_motion_batch(self, points: List[MotionPoint]):
        """Publish a batch of motion points"""
        # In real implementation, would serialize to protobuf and send
        self.message_count += 1

        if self.message_count % 100 == 0:
            elapsed = time.time() - self.start_time
            rate = self.message_count / elapsed
            print(f"Published {self.message_count} messages "
                  f"({rate:.1f} msg/s, {len(points)} points/msg)")

    def get_statistics(self):
        """Get publisher statistics"""
        elapsed = time.time() - self.start_time
        return {
            'messages': self.message_count,
            'elapsed': elapsed,
            'rate': self.message_count / elapsed if elapsed > 0 else 0
        }


def main():
    parser = argparse.ArgumentParser(
        description='Motion tracking data publisher example'
    )
    parser.add_argument(
        '--transport',
        choices=['shared_memory', 'udp', 'tcp'],
        default='shared_memory',
        help='Transport mechanism'
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=100.0,
        help='Publishing rate (Hz)'
    )
    parser.add_argument(
        '--num-points',
        type=int,
        default=10,
        help='Number of tracking points'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Run duration (seconds)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8888,
        help='UDP/TCP port'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Remote host for UDP/TCP'
    )

    args = parser.parse_args()

    print("="*60)
    print("MTRTP Motion Publisher Example")
    print("="*60)
    print(f"Transport: {args.transport}")
    print(f"Rate: {args.rate} Hz")
    print(f"Points: {args.num_points}")
    print(f"Duration: {args.duration} seconds")
    print("="*60)

    # Create simulator
    simulator = MotionSimulator(num_points=args.num_points)

    # Create publisher
    config = {
        'host': args.host,
        'port': args.port,
        'rate': args.rate
    }
    publisher = MockPublisher(args.transport, config)

    # Calculate timing
    dt = 1.0 / args.rate
    start_time = time.time()
    next_publish = start_time

    print("\nPublishing motion data... (Ctrl+C to stop)\n")

    try:
        while (time.time() - start_time) < args.duration:
            current_time = time.time()

            if current_time >= next_publish:
                # Update simulation
                simulator.update(dt)

                # Get points
                points = simulator.get_points()

                # Publish
                publisher.publish_motion_batch(points)

                # Schedule next publish
                next_publish += dt

                # Maintain timing
                if current_time > next_publish:
                    # Fell behind, catch up
                    next_publish = current_time

            else:
                # Sleep until next publish time
                sleep_time = next_publish - current_time
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.001))

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    # Print statistics
    stats = publisher.get_statistics()
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    print(f"Messages published: {stats['messages']}")
    print(f"Duration: {stats['elapsed']:.2f} seconds")
    print(f"Average rate: {stats['rate']:.1f} msg/s")
    print(f"Points per message: {args.num_points}")
    print(f"Total points: {stats['messages'] * args.num_points}")
    print("="*60)


if __name__ == '__main__':
    main()
