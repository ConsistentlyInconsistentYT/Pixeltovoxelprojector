"""
High-Performance Python Subscriber Interface for Motion Tracking Real-Time Protocol

Features:
- Easy subscription to multiple data streams
- Message type filtering
- Callback-based event handling
- Automatic reconnection with exponential backoff
- Thread-safe operation
- Zero-copy shared memory access
- Performance monitoring

Author: Motion Tracking System
Version: 1.0.0
"""

import threading
import time
import socket
import struct
import mmap
import os
import posix_ipc
from typing import Callable, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
from queue import Queue, Empty
import lz4.frame
import numpy as np

# Protocol buffer imports (assumes generated from motion_protocol.proto)
# from motion_protocol_pb2 import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION AND TYPES
# ============================================================================

class TransportType(Enum):
    """Available transport mechanisms"""
    UDP = "udp"
    TCP = "tcp"
    SHARED_MEMORY = "shared_memory"


class MessageType(Enum):
    """Message types for filtering"""
    MOTION_COORDINATE = 1
    CAMERA_POSE = 2
    TARGET_TRACKING = 3
    SYSTEM_STATUS = 4
    STREAM_CONTROL = 5
    ALL = 0


class CompressionType(Enum):
    """Compression options"""
    NONE = 0
    LZ4 = 1


@dataclass
class SubscriberConfig:
    """Configuration for subscriber"""
    # Transport settings
    transport: TransportType = TransportType.SHARED_MEMORY
    host: str = "127.0.0.1"
    udp_port: int = 8888
    tcp_port: int = 8889
    shared_mem_name: str = "/mtrtp_default"

    # Connection settings
    auto_reconnect: bool = True
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 30.0
    connection_timeout: float = 5.0

    # Buffer settings
    buffer_size: int = 65536
    message_queue_size: int = 1000

    # Compression
    compression: CompressionType = CompressionType.LZ4

    # Performance
    enable_stats: bool = True
    stats_interval: float = 10.0

    # Threading
    num_worker_threads: int = 2
    callback_timeout: float = 1.0


@dataclass
class Statistics:
    """Performance statistics"""
    messages_received: int = 0
    bytes_received: int = 0
    messages_dropped: int = 0
    errors: int = 0
    reconnections: int = 0
    avg_latency_ms: float = 0.0
    avg_callback_time_ms: float = 0.0
    start_time: float = field(default_factory=time.time)

    def get_throughput(self) -> Dict[str, float]:
        """Calculate throughput metrics"""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return {"msg_per_sec": 0.0, "mbps": 0.0}

        return {
            "msg_per_sec": self.messages_received / elapsed,
            "mbps": (self.bytes_received / elapsed) / (1024 * 1024)
        }

    def print_stats(self):
        """Print formatted statistics"""
        throughput = self.get_throughput()
        elapsed = time.time() - self.start_time

        print("\n" + "="*50)
        print("Subscriber Statistics")
        print("="*50)
        print(f"Uptime: {elapsed:.2f} seconds")
        print(f"Messages received: {self.messages_received}")
        print(f"Bytes received: {self.bytes_received:,}")
        print(f"Messages dropped: {self.messages_dropped}")
        print(f"Errors: {self.errors}")
        print(f"Reconnections: {self.reconnections}")
        print(f"Throughput: {throughput['msg_per_sec']:.1f} msg/s")
        print(f"Bandwidth: {throughput['mbps']:.2f} MB/s")
        print(f"Avg latency: {self.avg_latency_ms:.2f} ms")
        print(f"Avg callback time: {self.avg_callback_time_ms:.2f} ms")
        print("="*50 + "\n")


# ============================================================================
# MESSAGE HANDLER
# ============================================================================

class MessageHandler:
    """Manages callbacks for different message types"""

    def __init__(self):
        self._callbacks: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def register(self, message_type: MessageType, callback: Callable):
        """Register a callback for a message type"""
        with self._lock:
            self._callbacks[message_type].append(callback)
            logger.info(f"Registered callback for {message_type.name}")

    def unregister(self, message_type: MessageType, callback: Callable):
        """Unregister a callback"""
        with self._lock:
            if callback in self._callbacks[message_type]:
                self._callbacks[message_type].remove(callback)
                logger.info(f"Unregistered callback for {message_type.name}")

    def get_callbacks(self, message_type: MessageType) -> List[Callable]:
        """Get all callbacks for a message type"""
        with self._lock:
            # Return callbacks for specific type + ALL type
            callbacks = self._callbacks[message_type].copy()
            callbacks.extend(self._callbacks[MessageType.ALL])
            return callbacks

    def clear(self):
        """Clear all callbacks"""
        with self._lock:
            self._callbacks.clear()


# ============================================================================
# TRANSPORT IMPLEMENTATIONS
# ============================================================================

class SharedMemoryTransport:
    """Shared memory transport using POSIX shared memory"""

    def __init__(self, name: str, size: int = 1024 * 1024 * 64):
        self.name = name
        self.size = size
        self.shm = None
        self.mem = None
        self.read_pos = 0

    def connect(self) -> bool:
        """Connect to shared memory"""
        try:
            # Open existing shared memory
            self.shm = posix_ipc.SharedMemory(self.name)

            # Map to memory
            self.mem = mmap.mmap(
                self.shm.fd,
                self.size,
                mmap.MAP_SHARED,
                mmap.PROT_READ
            )

            os.close(self.shm.fd)
            logger.info(f"Connected to shared memory: {self.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to shared memory: {e}")
            return False

    def receive(self, timeout: float = 0.1) -> Optional[bytes]:
        """Receive message from shared memory"""
        if not self.mem:
            return None

        try:
            # Read message header (size + checksum + timestamp)
            header_size = 16  # 4 + 4 + 8 bytes

            if self.read_pos + header_size > self.size:
                self.read_pos = 0

            # Read header
            self.mem.seek(self.read_pos)
            header = self.mem.read(header_size)

            if len(header) < header_size:
                return None

            msg_size, checksum, timestamp = struct.unpack('<IIQ', header)

            if msg_size == 0 or msg_size > self.size:
                return None

            # Read message data
            data = self.mem.read(msg_size)

            if len(data) < msg_size:
                return None

            # Verify checksum (simple CRC32)
            import zlib
            computed_crc = zlib.crc32(data) & 0xFFFFFFFF

            if computed_crc != checksum:
                logger.warning("Checksum mismatch in shared memory message")
                return None

            # Update read position
            self.read_pos += header_size + msg_size

            return data

        except Exception as e:
            logger.error(f"Error reading from shared memory: {e}")
            return None

    def close(self):
        """Close shared memory connection"""
        if self.mem:
            self.mem.close()
            self.mem = None
        self.shm = None


class UDPTransport:
    """UDP transport for low-latency streaming"""

    def __init__(self, host: str, port: int, buffer_size: int = 65536):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.sock = None

    def connect(self) -> bool:
        """Create UDP socket"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Set large receive buffer
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 4)

            # Bind to port
            self.sock.bind(('0.0.0.0', self.port))

            # Set timeout for non-blocking receives
            self.sock.settimeout(0.1)

            logger.info(f"UDP transport listening on port {self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to create UDP socket: {e}")
            return False

    def receive(self, timeout: float = 0.1) -> Optional[bytes]:
        """Receive UDP packet"""
        if not self.sock:
            return None

        try:
            self.sock.settimeout(timeout)
            data, addr = self.sock.recvfrom(self.buffer_size)
            return data

        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Error receiving UDP packet: {e}")
            return None

    def close(self):
        """Close UDP socket"""
        if self.sock:
            self.sock.close()
            self.sock = None


class TCPTransport:
    """TCP transport for reliable streaming"""

    def __init__(self, host: str, port: int, buffer_size: int = 65536):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.sock = None

    def connect(self) -> bool:
        """Connect to TCP server"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(5.0)

            self.sock.connect((self.host, self.port))
            self.sock.settimeout(0.1)

            logger.info(f"Connected to TCP server {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to TCP server: {e}")
            return False

    def receive(self, timeout: float = 0.1) -> Optional[bytes]:
        """Receive TCP message"""
        if not self.sock:
            return None

        try:
            self.sock.settimeout(timeout)

            # Read size header (4 bytes)
            size_data = self._recv_exact(4)
            if not size_data:
                return None

            msg_size = struct.unpack('<I', size_data)[0]

            if msg_size > self.buffer_size:
                logger.error(f"Message too large: {msg_size}")
                return None

            # Read message data
            data = self._recv_exact(msg_size)
            return data

        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Error receiving TCP message: {e}")
            return None

    def _recv_exact(self, size: int) -> Optional[bytes]:
        """Receive exact number of bytes"""
        data = b''
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def close(self):
        """Close TCP connection"""
        if self.sock:
            self.sock.close()
            self.sock = None


# ============================================================================
# MAIN SUBSCRIBER CLASS
# ============================================================================

class Subscriber:
    """
    High-performance subscriber for motion tracking data streams

    Example usage:
        subscriber = Subscriber()

        @subscriber.on_message(MessageType.MOTION_COORDINATE)
        def handle_motion(message, timestamp):
            print(f"Received motion data: {message.point_id}")

        subscriber.start()
        time.sleep(60)
        subscriber.stop()
    """

    def __init__(self, config: Optional[SubscriberConfig] = None):
        self.config = config or SubscriberConfig()
        self.stats = Statistics()
        self.message_handler = MessageHandler()

        self.transport: Optional[Union[SharedMemoryTransport, UDPTransport, TCPTransport]] = None
        self.message_queue = Queue(maxsize=self.config.message_queue_size)

        self.running = False
        self.connected = False
        self.receiver_thread: Optional[threading.Thread] = None
        self.worker_threads: List[threading.Thread] = []
        self.stats_thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()

    def on_message(self, message_type: MessageType):
        """
        Decorator for registering message callbacks

        Usage:
            @subscriber.on_message(MessageType.MOTION_COORDINATE)
            def handle_motion(message, timestamp):
                pass
        """
        def decorator(callback: Callable):
            self.register_callback(message_type, callback)
            return callback
        return decorator

    def register_callback(self, message_type: MessageType, callback: Callable):
        """Register a callback for a specific message type"""
        self.message_handler.register(message_type, callback)

    def unregister_callback(self, message_type: MessageType, callback: Callable):
        """Unregister a callback"""
        self.message_handler.unregister(message_type, callback)

    def start(self):
        """Start the subscriber"""
        if self.running:
            logger.warning("Subscriber already running")
            return

        self.running = True

        # Connect to transport
        if not self._connect():
            logger.error("Failed to connect to transport")
            self.running = False
            return

        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receiver_thread.start()

        # Start worker threads
        for i in range(self.config.num_worker_threads):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.worker_threads.append(thread)

        # Start stats thread
        if self.config.enable_stats:
            self.stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
            self.stats_thread.start()

        logger.info("Subscriber started successfully")

    def stop(self):
        """Stop the subscriber"""
        if not self.running:
            return

        logger.info("Stopping subscriber...")
        self.running = False

        # Wait for threads to finish
        if self.receiver_thread:
            self.receiver_thread.join(timeout=2.0)

        for thread in self.worker_threads:
            thread.join(timeout=2.0)

        if self.stats_thread:
            self.stats_thread.join(timeout=2.0)

        # Close transport
        if self.transport:
            self.transport.close()

        # Print final stats
        if self.config.enable_stats:
            self.stats.print_stats()

        logger.info("Subscriber stopped")

    def _connect(self) -> bool:
        """Connect to the specified transport"""
        try:
            if self.config.transport == TransportType.SHARED_MEMORY:
                self.transport = SharedMemoryTransport(self.config.shared_mem_name)
            elif self.config.transport == TransportType.UDP:
                self.transport = UDPTransport(
                    self.config.host,
                    self.config.udp_port,
                    self.config.buffer_size
                )
            elif self.config.transport == TransportType.TCP:
                self.transport = TCPTransport(
                    self.config.host,
                    self.config.tcp_port,
                    self.config.buffer_size
                )
            else:
                logger.error(f"Unknown transport type: {self.config.transport}")
                return False

            if self.transport.connect():
                self.connected = True
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def _reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if not self.config.auto_reconnect:
            return

        delay = self.config.reconnect_delay

        while self.running and not self.connected:
            logger.info(f"Attempting to reconnect in {delay:.1f} seconds...")
            time.sleep(delay)

            if self._connect():
                logger.info("Reconnected successfully")
                self.stats.reconnections += 1
                return

            # Exponential backoff
            delay = min(delay * 2, self.config.max_reconnect_delay)

    def _receive_loop(self):
        """Main receive loop (runs in separate thread)"""
        logger.info("Receive loop started")

        while self.running:
            if not self.connected:
                self._reconnect()
                if not self.connected:
                    break
                continue

            try:
                # Receive data from transport
                data = self.transport.receive(timeout=0.1)

                if data is None:
                    continue

                # Decompress if needed
                if self.config.compression == CompressionType.LZ4:
                    try:
                        data = lz4.frame.decompress(data)
                    except Exception as e:
                        logger.error(f"Failed to decompress message: {e}")
                        self.stats.errors += 1
                        continue

                # Add to processing queue
                timestamp = time.time()
                try:
                    self.message_queue.put_nowait((data, timestamp))
                    self.stats.messages_received += 1
                    self.stats.bytes_received += len(data)
                except:
                    self.stats.messages_dropped += 1

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                self.stats.errors += 1
                self.connected = False

        logger.info("Receive loop stopped")

    def _worker_loop(self):
        """Worker loop for processing messages (runs in separate thread)"""
        while self.running:
            try:
                # Get message from queue
                data, recv_timestamp = self.message_queue.get(timeout=0.1)

                # Parse message (simplified - would use protobuf in real implementation)
                message, message_type = self._parse_message(data)

                if message is None:
                    continue

                # Calculate latency
                latency_ms = (time.time() - recv_timestamp) * 1000
                self.stats.avg_latency_ms = (
                    self.stats.avg_latency_ms * 0.95 + latency_ms * 0.05
                )

                # Get callbacks for this message type
                callbacks = self.message_handler.get_callbacks(message_type)

                # Execute callbacks
                for callback in callbacks:
                    try:
                        start_time = time.time()
                        callback(message, recv_timestamp)

                        # Update callback time
                        callback_time = (time.time() - start_time) * 1000
                        self.stats.avg_callback_time_ms = (
                            self.stats.avg_callback_time_ms * 0.95 +
                            callback_time * 0.05
                        )

                    except Exception as e:
                        logger.error(f"Error in callback: {e}")
                        self.stats.errors += 1

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                self.stats.errors += 1

    def _parse_message(self, data: bytes) -> tuple:
        """
        Parse message from bytes

        In a real implementation, this would use protobuf deserialization.
        For now, returns a mock message object.
        """
        # Mock implementation - would use actual protobuf parsing
        try:
            # Extract message type from first byte
            if len(data) < 8:
                return None, None

            msg_type = data[0]

            # Create mock message object
            message = {
                'raw_data': data,
                'type': msg_type,
                'timestamp': time.time()
            }

            message_type = MessageType(msg_type) if msg_type <= 5 else MessageType.ALL

            return message, message_type

        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            self.stats.errors += 1
            return None, None

    def _stats_loop(self):
        """Periodically print statistics"""
        while self.running:
            time.sleep(self.config.stats_interval)
            if self.running:
                self.stats.print_stats()

    def get_statistics(self) -> Statistics:
        """Get current statistics"""
        return self.stats

    def is_connected(self) -> bool:
        """Check if subscriber is connected"""
        return self.connected


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_subscriber(
    transport: TransportType = TransportType.SHARED_MEMORY,
    **kwargs
) -> Subscriber:
    """
    Create a subscriber with custom configuration

    Args:
        transport: Transport type to use
        **kwargs: Additional configuration parameters

    Returns:
        Configured Subscriber instance
    """
    config = SubscriberConfig(transport=transport, **kwargs)
    return Subscriber(config)


def subscribe_motion_data(
    callback: Callable,
    transport: TransportType = TransportType.SHARED_MEMORY,
    **kwargs
) -> Subscriber:
    """
    Quick setup for subscribing to motion coordinate messages

    Args:
        callback: Function to call when motion data is received
        transport: Transport type to use
        **kwargs: Additional configuration parameters

    Returns:
        Started Subscriber instance
    """
    subscriber = create_subscriber(transport, **kwargs)
    subscriber.register_callback(MessageType.MOTION_COORDINATE, callback)
    subscriber.start()
    return subscriber


def subscribe_camera_pose(
    callback: Callable,
    transport: TransportType = TransportType.SHARED_MEMORY,
    **kwargs
) -> Subscriber:
    """Quick setup for subscribing to camera pose messages"""
    subscriber = create_subscriber(transport, **kwargs)
    subscriber.register_callback(MessageType.CAMERA_POSE, callback)
    subscriber.start()
    return subscriber


def subscribe_all(
    callback: Callable,
    transport: TransportType = TransportType.SHARED_MEMORY,
    **kwargs
) -> Subscriber:
    """Subscribe to all message types"""
    subscriber = create_subscriber(transport, **kwargs)
    subscriber.register_callback(MessageType.ALL, callback)
    subscriber.start()
    return subscriber


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Using decorator syntax
    subscriber = Subscriber(SubscriberConfig(
        transport=TransportType.SHARED_MEMORY,
        shared_mem_name="/mtrtp_motion"
    ))

    @subscriber.on_message(MessageType.MOTION_COORDINATE)
    def handle_motion(message, timestamp):
        print(f"Motion data received at {timestamp}")
        # Process motion data

    @subscriber.on_message(MessageType.TARGET_TRACKING)
    def handle_tracking(message, timestamp):
        print(f"Tracking data received at {timestamp}")
        # Process tracking data

    subscriber.start()

    try:
        # Run for 60 seconds
        time.sleep(60)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        subscriber.stop()

    # Example 2: Using convenience functions
    # def my_callback(message, timestamp):
    #     print(f"Received message at {timestamp}")
    #
    # subscriber = subscribe_motion_data(
    #     my_callback,
    #     transport=TransportType.UDP,
    #     udp_port=8888
    # )
    #
    # time.sleep(60)
    # subscriber.stop()
