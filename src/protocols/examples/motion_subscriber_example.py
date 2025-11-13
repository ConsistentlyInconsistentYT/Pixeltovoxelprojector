#!/usr/bin/env python3
"""
Motion Data Subscriber Example

Demonstrates how to subscribe to motion tracking data using MTRTP.
Shows callback-based event handling, filtering, and statistics.

Usage:
    python motion_subscriber_example.py [--transport TYPE] [--filter EXPR]

Examples:
    python motion_subscriber_example.py --transport shared_memory
    python motion_subscriber_example.py --transport udp --port 8888
    python motion_subscriber_example.py --filter "confidence > 0.9"
"""

import argparse
import time
import sys
from collections import defaultdict, deque
from typing import Dict, List
import numpy as np

# Add parent directory to path to import subscriber
sys.path.insert(0, '..')

try:
    from subscriber import (
        Subscriber,
        SubscriberConfig,
        MessageType,
        TransportType
    )
except ImportError:
    print("Warning: Could not import subscriber module")
    print("This example shows the intended usage")


class MotionDataProcessor:
    """Process and analyze received motion data"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_history = defaultdict(lambda: deque(maxlen=window_size))
        self.message_count = 0
        self.point_count = 0
        self.start_time = time.time()
        self.latencies = deque(maxlen=1000)

    def process_motion_message(self, message: dict, timestamp: float):
        """Process incoming motion coordinate message"""
        self.message_count += 1

        # Calculate latency
        latency_ms = (time.time() - timestamp) * 1000
        self.latencies.append(latency_ms)

        # Extract point data (mock implementation)
        point_id = message.get('point_id', 0)
        self.point_count += 1

        # Store in history
        self.data_history[point_id].append({
            'timestamp': timestamp,
            'position': message.get('position', [0, 0, 0]),
            'velocity': message.get('velocity', [0, 0, 0]),
            'confidence': message.get('confidence', 1.0)
        })

        # Print periodic updates
        if self.message_count % 100 == 0:
            self.print_status()

    def process_camera_pose(self, message: dict, timestamp: float):
        """Process camera pose message"""
        print(f"Camera pose received at {timestamp:.3f}")
        # Process camera data

    def process_target_tracking(self, message: dict, timestamp: float):
        """Process target tracking message"""
        print(f"Target tracking received at {timestamp:.3f}")
        # Process tracking data

    def print_status(self):
        """Print current processing status"""
        elapsed = time.time() - self.start_time
        rate = self.message_count / elapsed if elapsed > 0 else 0

        avg_latency = np.mean(self.latencies) if self.latencies else 0
        max_latency = np.max(self.latencies) if self.latencies else 0

        print(f"\rProcessed: {self.message_count} msgs, "
              f"Rate: {rate:.1f} msg/s, "
              f"Latency: {avg_latency:.2f}ms (max: {max_latency:.2f}ms)",
              end='', flush=True)

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        elapsed = time.time() - self.start_time

        return {
            'messages': self.message_count,
            'points': self.point_count,
            'elapsed': elapsed,
            'rate': self.message_count / elapsed if elapsed > 0 else 0,
            'avg_latency_ms': np.mean(self.latencies) if self.latencies else 0,
            'max_latency_ms': np.max(self.latencies) if self.latencies else 0,
            'unique_points': len(self.data_history)
        }

    def print_statistics(self):
        """Print detailed statistics"""
        stats = self.get_statistics()

        print("\n\n" + "="*60)
        print("Processing Statistics")
        print("="*60)
        print(f"Messages processed: {stats['messages']}")
        print(f"Points tracked: {stats['points']}")
        print(f"Unique points: {stats['unique_points']}")
        print(f"Duration: {stats['elapsed']:.2f} seconds")
        print(f"Average rate: {stats['rate']:.1f} msg/s")
        print(f"Average latency: {stats['avg_latency_ms']:.2f} ms")
        print(f"Max latency: {stats['max_latency_ms']:.2f} ms")
        print("="*60)


def create_filter_function(filter_expr: str):
    """Create a filter function from expression"""
    if not filter_expr:
        return lambda msg: True

    # Simple filter implementation
    # In production, would use proper expression parser
    def filter_func(message):
        try:
            # Example: "confidence > 0.9"
            confidence = message.get('confidence', 1.0)

            if 'confidence' in filter_expr:
                if '>' in filter_expr:
                    threshold = float(filter_expr.split('>')[-1].strip())
                    return confidence > threshold
                elif '<' in filter_expr:
                    threshold = float(filter_expr.split('<')[-1].strip())
                    return confidence < threshold

            return True
        except:
            return True

    return filter_func


def main():
    parser = argparse.ArgumentParser(
        description='Motion tracking data subscriber example'
    )
    parser.add_argument(
        '--transport',
        choices=['shared_memory', 'udp', 'tcp'],
        default='shared_memory',
        help='Transport mechanism'
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
    parser.add_argument(
        '--shared-mem-name',
        default='/mtrtp_default',
        help='Shared memory name'
    )
    parser.add_argument(
        '--filter',
        default='',
        help='Message filter expression'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Run duration (seconds, 0=infinite)'
    )
    parser.add_argument(
        '--stats-interval',
        type=float,
        default=5.0,
        help='Statistics print interval (seconds)'
    )

    args = parser.parse_args()

    print("="*60)
    print("MTRTP Motion Subscriber Example")
    print("="*60)
    print(f"Transport: {args.transport}")

    if args.transport == 'shared_memory':
        print(f"Shared memory: {args.shared_mem_name}")
    else:
        print(f"Endpoint: {args.host}:{args.port}")

    if args.filter:
        print(f"Filter: {args.filter}")

    print("="*60)

    # Create data processor
    processor = MotionDataProcessor()

    # Create filter
    filter_func = create_filter_function(args.filter)

    # Configure subscriber
    transport_map = {
        'shared_memory': TransportType.SHARED_MEMORY,
        'udp': TransportType.UDP,
        'tcp': TransportType.TCP
    }

    config = SubscriberConfig(
        transport=transport_map[args.transport],
        host=args.host,
        udp_port=args.port if args.transport == 'udp' else 8888,
        tcp_port=args.port if args.transport == 'tcp' else 8889,
        shared_mem_name=args.shared_mem_name,
        auto_reconnect=True,
        enable_stats=True,
        stats_interval=args.stats_interval
    )

    # Create subscriber
    subscriber = Subscriber(config)

    # Register callbacks using decorator pattern
    @subscriber.on_message(MessageType.MOTION_COORDINATE)
    def handle_motion(message, timestamp):
        if filter_func(message):
            processor.process_motion_message(message, timestamp)

    @subscriber.on_message(MessageType.CAMERA_POSE)
    def handle_camera(message, timestamp):
        processor.process_camera_pose(message, timestamp)

    @subscriber.on_message(MessageType.TARGET_TRACKING)
    def handle_tracking(message, timestamp):
        processor.process_target_tracking(message, timestamp)

    # Alternative: Register callbacks directly
    # subscriber.register_callback(
    #     MessageType.MOTION_COORDINATE,
    #     lambda msg, ts: processor.process_motion_message(msg, ts)
    # )

    print("\nSubscribing to motion data... (Ctrl+C to stop)\n")

    # Start subscriber
    subscriber.start()

    try:
        # Run for specified duration
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            # Run indefinitely
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        # Stop subscriber
        print("\nStopping subscriber...")
        subscriber.stop()

        # Print processor statistics
        processor.print_statistics()

        # Print subscriber statistics
        print("\nSubscriber Statistics:")
        subscriber.get_statistics().print_stats()


if __name__ == '__main__':
    main()
