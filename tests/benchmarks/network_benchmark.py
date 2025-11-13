#!/usr/bin/env python3
"""
Network Benchmarks for PixelToVoxelProjector

Benchmarks for:
- Streaming latency measurements
- Throughput testing
- Multi-client scalability
- Packet loss analysis
- TCP/UDP performance comparison
"""

import os
import sys
import time
import json
import socket
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
import struct
import select


@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    protocol: str
    throughput_mbps: float
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    packet_loss_percent: float
    jitter_ms: float
    bytes_sent: int
    bytes_received: int
    duration_sec: float
    num_clients: int


@dataclass
class PacketStats:
    """Packet-level statistics"""
    sent: int
    received: int
    lost: int
    out_of_order: int
    duplicates: int


class NetworkBenchmark:
    """Network benchmark orchestrator"""

    def __init__(self, output_dir: str = "benchmark_results/network"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def benchmark_tcp_throughput(self,
                                 host: str = "127.0.0.1",
                                 port: int = 9999,
                                 duration_sec: int = 10,
                                 chunk_size: int = 65536) -> Dict:
        """Benchmark TCP throughput"""
        print("\n" + "="*60)
        print(f"Benchmarking TCP Throughput ({duration_sec}s)")
        print("="*60)

        # Start server in background thread
        server_thread = threading.Thread(
            target=self._tcp_server,
            args=(host, port, duration_sec + 5)
        )
        server_thread.daemon = True
        server_thread.start()

        time.sleep(0.5)  # Give server time to start

        # Client
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))

            data = b'X' * chunk_size
            bytes_sent = 0
            start_time = time.time()

            while time.time() - start_time < duration_sec:
                sock.sendall(data)
                bytes_sent += len(data)

            end_time = time.time()
            sock.close()

            duration = end_time - start_time
            throughput_mbps = (bytes_sent * 8) / (duration * 1e6)

            results = {
                'protocol': 'TCP',
                'duration_sec': duration,
                'bytes_sent': bytes_sent,
                'throughput_mbps': throughput_mbps,
                'chunk_size': chunk_size,
            }

            print(f"\nResults:")
            print(f"  Bytes Sent:    {bytes_sent:,}")
            print(f"  Duration:      {duration:.2f} s")
            print(f"  Throughput:    {throughput_mbps:.2f} Mbps")

            return results

        except Exception as e:
            print(f"Error: {e}")
            return {}

    def _tcp_server(self, host: str, port: int, timeout: int):
        """TCP server for throughput test"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(1)
            sock.settimeout(timeout)

            conn, addr = sock.accept()

            # Receive data
            while True:
                data = conn.recv(65536)
                if not data:
                    break

            conn.close()
            sock.close()

        except Exception:
            pass

    def benchmark_udp_throughput(self,
                                 host: str = "127.0.0.1",
                                 port: int = 9998,
                                 duration_sec: int = 10,
                                 packet_size: int = 1400) -> Dict:
        """Benchmark UDP throughput with packet loss tracking"""
        print("\n" + "="*60)
        print(f"Benchmarking UDP Throughput ({duration_sec}s)")
        print("="*60)

        # Shared results
        results = {'packets_sent': 0, 'packets_received': 0}

        # Start server
        server_thread = threading.Thread(
            target=self._udp_server,
            args=(host, port, duration_sec + 5, results)
        )
        server_thread.daemon = True
        server_thread.start()

        time.sleep(0.5)

        # Client
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            packets_sent = 0
            bytes_sent = 0
            start_time = time.time()

            # Packet format: [seq_num (8 bytes)] [timestamp (8 bytes)] [data]
            data_size = packet_size - 16
            data = b'X' * data_size

            while time.time() - start_time < duration_sec:
                seq = packets_sent
                timestamp = time.time()

                packet = struct.pack('!Qd', seq, timestamp) + data
                sock.sendto(packet, (host, port))

                packets_sent += 1
                bytes_sent += len(packet)

                # Small delay to avoid overwhelming receiver
                time.sleep(0.0001)

            end_time = time.time()

            # Wait for server to finish receiving
            time.sleep(1.0)

            sock.close()

            duration = end_time - start_time
            throughput_mbps = (bytes_sent * 8) / (duration * 1e6)

            packets_received = results['packets_received']
            packet_loss = ((packets_sent - packets_received) / packets_sent) * 100 if packets_sent > 0 else 0

            results = {
                'protocol': 'UDP',
                'duration_sec': duration,
                'packets_sent': packets_sent,
                'packets_received': packets_received,
                'packet_loss_percent': packet_loss,
                'bytes_sent': bytes_sent,
                'throughput_mbps': throughput_mbps,
                'packet_size': packet_size,
            }

            print(f"\nResults:")
            print(f"  Packets Sent:     {packets_sent:,}")
            print(f"  Packets Received: {packets_received:,}")
            print(f"  Packet Loss:      {packet_loss:.2f}%")
            print(f"  Throughput:       {throughput_mbps:.2f} Mbps")

            return results

        except Exception as e:
            print(f"Error: {e}")
            return {}

    def _udp_server(self, host: str, port: int, timeout: int, results: Dict):
        """UDP server for throughput test"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.settimeout(1.0)

            packets_received = 0
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    data, addr = sock.recvfrom(65536)
                    packets_received += 1
                except socket.timeout:
                    continue
                except Exception:
                    break

            results['packets_received'] = packets_received
            sock.close()

        except Exception:
            pass

    def benchmark_latency(self,
                         host: str = "127.0.0.1",
                         port: int = 9997,
                         num_pings: int = 1000,
                         protocol: str = "TCP") -> Dict:
        """Benchmark network latency (ping-pong test)"""
        print("\n" + "="*60)
        print(f"Benchmarking {protocol} Latency ({num_pings} pings)")
        print("="*60)

        if protocol.upper() == "TCP":
            return self._benchmark_tcp_latency(host, port, num_pings)
        else:
            return self._benchmark_udp_latency(host, port, num_pings)

    def _benchmark_tcp_latency(self, host: str, port: int, num_pings: int) -> Dict:
        """TCP latency benchmark"""
        # Start echo server
        server_thread = threading.Thread(
            target=self._tcp_echo_server,
            args=(host, port, num_pings + 100)
        )
        server_thread.daemon = True
        server_thread.start()

        time.sleep(0.5)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))

            latencies = []
            data = b'PING' * 64  # 256 bytes

            for i in range(num_pings):
                start = time.time()
                sock.sendall(data)
                response = sock.recv(len(data))
                end = time.time()

                if response:
                    latency = (end - start) * 1000  # ms
                    latencies.append(latency)

                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{num_pings}")

            sock.close()

            latencies = np.array(latencies)

            results = {
                'protocol': 'TCP',
                'num_pings': num_pings,
                'latency_avg_ms': np.mean(latencies),
                'latency_min_ms': np.min(latencies),
                'latency_max_ms': np.max(latencies),
                'latency_p50_ms': np.percentile(latencies, 50),
                'latency_p95_ms': np.percentile(latencies, 95),
                'latency_p99_ms': np.percentile(latencies, 99),
                'latency_std_ms': np.std(latencies),
            }

            print(f"\nResults:")
            print(f"  Avg Latency:  {results['latency_avg_ms']:.2f} ms")
            print(f"  p50 Latency:  {results['latency_p50_ms']:.2f} ms")
            print(f"  p95 Latency:  {results['latency_p95_ms']:.2f} ms")
            print(f"  p99 Latency:  {results['latency_p99_ms']:.2f} ms")

            return results

        except Exception as e:
            print(f"Error: {e}")
            return {}

    def _tcp_echo_server(self, host: str, port: int, timeout: int):
        """TCP echo server"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(1)
            sock.settimeout(timeout)

            conn, addr = sock.accept()

            while True:
                data = conn.recv(4096)
                if not data:
                    break
                conn.sendall(data)

            conn.close()
            sock.close()

        except Exception:
            pass

    def benchmark_multi_client(self,
                               host: str = "127.0.0.1",
                               port: int = 9996,
                               num_clients: int = 10,
                               duration_sec: int = 10) -> Dict:
        """Benchmark multi-client scalability"""
        print("\n" + "="*60)
        print(f"Benchmarking Multi-Client Scalability ({num_clients} clients)")
        print("="*60)

        # Shared results
        results = {
            'clients_completed': 0,
            'total_bytes': 0,
            'client_results': [],
        }
        results_lock = threading.Lock()

        # Start server
        server_thread = threading.Thread(
            target=self._multi_client_server,
            args=(host, port, num_clients, duration_sec + 5)
        )
        server_thread.daemon = True
        server_thread.start()

        time.sleep(0.5)

        # Start clients
        client_threads = []
        for i in range(num_clients):
            t = threading.Thread(
                target=self._multi_client_worker,
                args=(host, port, i, duration_sec, results, results_lock)
            )
            t.daemon = True
            t.start()
            client_threads.append(t)

        # Wait for clients
        for t in client_threads:
            t.join()

        # Calculate aggregate metrics
        total_bytes = results['total_bytes']
        throughput_mbps = (total_bytes * 8) / (duration_sec * 1e6)

        final_results = {
            'num_clients': num_clients,
            'duration_sec': duration_sec,
            'clients_completed': results['clients_completed'],
            'total_bytes': total_bytes,
            'aggregate_throughput_mbps': throughput_mbps,
            'avg_throughput_per_client_mbps': throughput_mbps / num_clients if num_clients > 0 else 0,
        }

        print(f"\nResults:")
        print(f"  Clients Completed:    {results['clients_completed']}/{num_clients}")
        print(f"  Total Bytes:          {total_bytes:,}")
        print(f"  Aggregate Throughput: {throughput_mbps:.2f} Mbps")
        print(f"  Per-Client Avg:       {final_results['avg_throughput_per_client_mbps']:.2f} Mbps")

        return final_results

    def _multi_client_worker(self, host: str, port: int, client_id: int,
                            duration_sec: int, results: Dict, lock: threading.Lock):
        """Worker for multi-client test"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))

            data = b'X' * 8192
            bytes_sent = 0
            start_time = time.time()

            while time.time() - start_time < duration_sec:
                sock.sendall(data)
                bytes_sent += len(data)

            sock.close()

            with lock:
                results['clients_completed'] += 1
                results['total_bytes'] += bytes_sent

        except Exception as e:
            print(f"Client {client_id} error: {e}")

    def _multi_client_server(self, host: str, port: int, num_clients: int, timeout: int):
        """Server for multi-client test"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(num_clients)
            sock.settimeout(timeout)

            connections = []

            # Accept connections
            for _ in range(num_clients):
                try:
                    conn, addr = sock.accept()
                    connections.append(conn)
                except socket.timeout:
                    break

            # Handle clients
            for conn in connections:
                t = threading.Thread(target=self._handle_client, args=(conn,))
                t.daemon = True
                t.start()

            time.sleep(timeout)

            for conn in connections:
                try:
                    conn.close()
                except:
                    pass

            sock.close()

        except Exception:
            pass

    def _handle_client(self, conn):
        """Handle single client connection"""
        try:
            while True:
                data = conn.recv(65536)
                if not data:
                    break
        except:
            pass
        finally:
            try:
                conn.close()
            except:
                pass

    def benchmark_streaming_latency(self,
                                   host: str = "127.0.0.1",
                                   port: int = 9995,
                                   stream_duration_sec: int = 10,
                                   packet_size: int = 1400,
                                   target_fps: int = 30) -> Dict:
        """Benchmark streaming latency (simulating voxel data streaming)"""
        print("\n" + "="*60)
        print(f"Benchmarking Streaming Latency ({target_fps} FPS)")
        print("="*60)

        # Shared results
        results = {
            'latencies': [],
            'packets_received': 0,
        }
        results_lock = threading.Lock()

        # Start receiver
        receiver_thread = threading.Thread(
            target=self._streaming_receiver,
            args=(host, port, results, results_lock)
        )
        receiver_thread.daemon = True
        receiver_thread.start()

        time.sleep(0.5)

        # Sender
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            frame_interval = 1.0 / target_fps
            data_size = packet_size - 16
            data = b'X' * data_size

            packets_sent = 0
            start_time = time.time()
            next_frame_time = start_time

            while time.time() - start_time < stream_duration_sec:
                current_time = time.time()

                if current_time >= next_frame_time:
                    # Send packet with timestamp
                    timestamp = time.time()
                    packet = struct.pack('!Qd', packets_sent, timestamp) + data
                    sock.sendto(packet, (host, port))

                    packets_sent += 1
                    next_frame_time += frame_interval
                else:
                    # Small sleep to avoid busy waiting
                    time.sleep(0.001)

            end_time = time.time()
            time.sleep(0.5)  # Wait for remaining packets

            sock.close()

            # Calculate metrics
            with results_lock:
                latencies = results['latencies']
                packets_received = results['packets_received']

            if latencies:
                latencies = np.array(latencies)
                jitter = np.std(latencies)

                packet_loss = ((packets_sent - packets_received) / packets_sent) * 100 if packets_sent > 0 else 0

                final_results = {
                    'protocol': 'UDP',
                    'target_fps': target_fps,
                    'duration_sec': end_time - start_time,
                    'packets_sent': packets_sent,
                    'packets_received': packets_received,
                    'packet_loss_percent': packet_loss,
                    'latency_avg_ms': np.mean(latencies),
                    'latency_p50_ms': np.percentile(latencies, 50),
                    'latency_p95_ms': np.percentile(latencies, 95),
                    'latency_p99_ms': np.percentile(latencies, 99),
                    'jitter_ms': jitter,
                }

                print(f"\nResults:")
                print(f"  Packets Sent:     {packets_sent}")
                print(f"  Packets Received: {packets_received}")
                print(f"  Packet Loss:      {packet_loss:.2f}%")
                print(f"  Avg Latency:      {final_results['latency_avg_ms']:.2f} ms")
                print(f"  p99 Latency:      {final_results['latency_p99_ms']:.2f} ms")
                print(f"  Jitter:           {jitter:.2f} ms")

                return final_results

            return {}

        except Exception as e:
            print(f"Error: {e}")
            return {}

    def _streaming_receiver(self, host: str, port: int, results: Dict, lock: threading.Lock):
        """Receiver for streaming latency test"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.settimeout(1.0)

            while True:
                try:
                    data, addr = sock.recvfrom(65536)
                    receive_time = time.time()

                    if len(data) >= 16:
                        seq, send_time = struct.unpack('!Qd', data[:16])
                        latency = (receive_time - send_time) * 1000  # ms

                        with lock:
                            results['latencies'].append(latency)
                            results['packets_received'] += 1

                except socket.timeout:
                    continue
                except Exception:
                    break

            sock.close()

        except Exception:
            pass

    def run_all_benchmarks(self):
        """Run complete network benchmark suite"""
        all_results = {}

        all_results['tcp_throughput'] = self.benchmark_tcp_throughput(
            duration_sec=10
        )

        all_results['udp_throughput'] = self.benchmark_udp_throughput(
            duration_sec=10
        )

        all_results['tcp_latency'] = self.benchmark_latency(
            num_pings=1000,
            protocol="TCP"
        )

        all_results['multi_client'] = self.benchmark_multi_client(
            num_clients=10,
            duration_sec=10
        )

        all_results['streaming_latency'] = self.benchmark_streaming_latency(
            stream_duration_sec=10,
            target_fps=30
        )

        # Save results
        self.save_results(all_results)

        return all_results

    def save_results(self, results: Dict):
        """Save benchmark results"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"network_benchmark_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}")


def main():
    """Run network benchmarks"""
    benchmark = NetworkBenchmark()

    print("="*60)
    print("Network Benchmark Suite")
    print("="*60)
    print("\nNote: Running on localhost (127.0.0.1)")
    print("For accurate results, test between separate machines\n")

    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
