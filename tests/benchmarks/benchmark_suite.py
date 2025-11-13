#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Suite for PixelToVoxelProjector

This suite provides end-to-end performance testing including:
- Pipeline benchmarking
- Component-level performance tests
- GPU utilization monitoring
- Memory bandwidth measurements
- Latency profiling
- Performance regression detection
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import subprocess
import threading
from datetime import datetime

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    print("Warning: pynvml not available. GPU monitoring disabled.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Some benchmarks will be skipped.")


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    name: str
    duration_ms: float
    throughput_fps: float
    memory_mb: float
    gpu_utilization_percent: float
    gpu_memory_mb: float
    cpu_utilization_percent: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    timestamp: str
    metadata: Dict


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection"""
    name: str
    min_throughput_fps: float
    max_latency_p99_ms: float
    max_memory_mb: float
    max_gpu_memory_mb: float


class GPUMonitor:
    """Monitor GPU utilization and memory usage"""

    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.thread = None

        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.available = True
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")
                self.available = False
        else:
            self.available = False

    def start(self):
        """Start monitoring in background thread"""
        if not self.available:
            return

        self.monitoring = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> Dict:
        """Stop monitoring and return statistics"""
        if not self.available:
            return {"utilization": 0, "memory_mb": 0}

        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)

        if not self.samples:
            return {"utilization": 0, "memory_mb": 0}

        utils = [s['util'] for s in self.samples]
        mems = [s['mem'] for s in self.samples]

        return {
            "utilization": np.mean(utils),
            "memory_mb": np.mean(mems),
            "max_utilization": np.max(utils),
            "max_memory_mb": np.max(mems),
        }

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                self.samples.append({
                    'util': util.gpu,
                    'mem': mem.used / (1024 ** 2),  # Convert to MB
                })
            except Exception:
                pass

            time.sleep(0.1)  # Sample every 100ms

    def __del__(self):
        if HAS_NVML and self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class CPUMonitor:
    """Monitor CPU utilization and memory"""

    def __init__(self):
        self.process = psutil.Process()
        self.samples = []
        self.monitoring = False
        self.thread = None

    def start(self):
        """Start monitoring"""
        self.monitoring = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> Dict:
        """Stop monitoring and return stats"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)

        if not self.samples:
            return {"cpu_percent": 0, "memory_mb": 0}

        cpu_vals = [s['cpu'] for s in self.samples]
        mem_vals = [s['mem'] for s in self.samples]

        return {
            "cpu_percent": np.mean(cpu_vals),
            "memory_mb": np.mean(mem_vals),
            "max_cpu_percent": np.max(cpu_vals),
            "max_memory_mb": np.max(mem_vals),
        }

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                cpu = self.process.cpu_percent(interval=0.1)
                mem = self.process.memory_info().rss / (1024 ** 2)  # MB

                self.samples.append({
                    'cpu': cpu,
                    'mem': mem,
                })
            except Exception:
                pass

            time.sleep(0.1)


class BenchmarkSuite:
    """Main benchmark suite orchestrator"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[BenchmarkResult] = []
        self.gpu_monitor = GPUMonitor()
        self.cpu_monitor = CPUMonitor()

        # Performance baselines for regression detection
        self.baselines = self._load_baselines()

    def _load_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Load performance baselines from file"""
        baseline_file = self.output_dir / "baselines.json"
        if not baseline_file.exists():
            return {}

        try:
            with open(baseline_file, 'r') as f:
                data = json.load(f)
            return {k: PerformanceBaseline(**v) for k, v in data.items()}
        except Exception as e:
            print(f"Warning: Could not load baselines: {e}")
            return {}

    def save_baselines(self):
        """Save current results as baselines"""
        baselines = {}
        for result in self.results:
            baselines[result.name] = PerformanceBaseline(
                name=result.name,
                min_throughput_fps=result.throughput_fps * 0.9,  # 10% tolerance
                max_latency_p99_ms=result.latency_p99_ms * 1.1,
                max_memory_mb=result.memory_mb * 1.1,
                max_gpu_memory_mb=result.gpu_memory_mb * 1.1,
            )

        baseline_file = self.output_dir / "baselines.json"
        with open(baseline_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in baselines.items()}, f, indent=2)

        print(f"Saved {len(baselines)} baselines to {baseline_file}")

    def check_regression(self, result: BenchmarkResult) -> List[str]:
        """Check for performance regressions"""
        if result.name not in self.baselines:
            return []

        baseline = self.baselines[result.name]
        regressions = []

        if result.throughput_fps < baseline.min_throughput_fps:
            regressions.append(
                f"Throughput regression: {result.throughput_fps:.2f} < {baseline.min_throughput_fps:.2f} FPS"
            )

        if result.latency_p99_ms > baseline.max_latency_p99_ms:
            regressions.append(
                f"Latency regression: {result.latency_p99_ms:.2f} > {baseline.max_latency_p99_ms:.2f} ms"
            )

        if result.memory_mb > baseline.max_memory_mb:
            regressions.append(
                f"Memory regression: {result.memory_mb:.2f} > {baseline.max_memory_mb:.2f} MB"
            )

        if result.gpu_memory_mb > baseline.max_gpu_memory_mb:
            regressions.append(
                f"GPU memory regression: {result.gpu_memory_mb:.2f} > {baseline.max_gpu_memory_mb:.2f} MB"
            )

        return regressions

    def run_benchmark(self, name: str, benchmark_fn, iterations: int = 100,
                     warmup: int = 10, **kwargs) -> BenchmarkResult:
        """Run a single benchmark with monitoring"""
        print(f"\n{'='*60}")
        print(f"Running benchmark: {name}")
        print(f"{'='*60}")

        # Warmup
        print(f"Warmup ({warmup} iterations)...")
        for i in range(warmup):
            benchmark_fn(**kwargs)

        # Start monitoring
        self.gpu_monitor.start()
        self.cpu_monitor.start()

        # Run benchmark
        print(f"Running ({iterations} iterations)...")
        latencies = []
        start_time = time.time()

        for i in range(iterations):
            iter_start = time.time()
            benchmark_fn(**kwargs)
            iter_end = time.time()
            latencies.append((iter_end - iter_start) * 1000)  # ms

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{iterations}")

        end_time = time.time()
        total_duration = (end_time - start_time) * 1000  # ms

        # Stop monitoring
        gpu_stats = self.gpu_monitor.stop()
        cpu_stats = self.cpu_monitor.stop()

        # Calculate statistics
        latencies_np = np.array(latencies)

        result = BenchmarkResult(
            name=name,
            duration_ms=total_duration,
            throughput_fps=iterations / (total_duration / 1000),
            memory_mb=cpu_stats.get('memory_mb', 0),
            gpu_utilization_percent=gpu_stats.get('utilization', 0),
            gpu_memory_mb=gpu_stats.get('memory_mb', 0),
            cpu_utilization_percent=cpu_stats.get('cpu_percent', 0),
            latency_p50_ms=np.percentile(latencies_np, 50),
            latency_p95_ms=np.percentile(latencies_np, 95),
            latency_p99_ms=np.percentile(latencies_np, 99),
            timestamp=datetime.now().isoformat(),
            metadata={
                'iterations': iterations,
                'warmup': warmup,
                'max_gpu_util': gpu_stats.get('max_utilization', 0),
                'max_gpu_mem_mb': gpu_stats.get('max_memory_mb', 0),
                'max_cpu_percent': cpu_stats.get('max_cpu_percent', 0),
                'max_memory_mb': cpu_stats.get('max_memory_mb', 0),
                **kwargs
            }
        )

        # Print results
        print(f"\n{'='*60}")
        print(f"Results for: {name}")
        print(f"{'='*60}")
        print(f"Duration:          {result.duration_ms:.2f} ms")
        print(f"Throughput:        {result.throughput_fps:.2f} FPS")
        print(f"Latency (p50):     {result.latency_p50_ms:.2f} ms")
        print(f"Latency (p95):     {result.latency_p95_ms:.2f} ms")
        print(f"Latency (p99):     {result.latency_p99_ms:.2f} ms")
        print(f"CPU Util:          {result.cpu_utilization_percent:.1f}%")
        print(f"Memory:            {result.memory_mb:.2f} MB")
        print(f"GPU Util:          {result.gpu_utilization_percent:.1f}%")
        print(f"GPU Memory:        {result.gpu_memory_mb:.2f} MB")

        # Check for regressions
        regressions = self.check_regression(result)
        if regressions:
            print(f"\nWARNING: Performance regressions detected:")
            for reg in regressions:
                print(f"  - {reg}")
        else:
            print(f"\nNo performance regressions detected.")

        self.results.append(result)
        return result

    def save_results(self):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_file = self.output_dir / f"results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"\nSaved results to {json_file}")

        # Save CSV
        csv_file = self.output_dir / f"results_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            if self.results:
                # Header
                f.write("name,duration_ms,throughput_fps,latency_p50_ms,latency_p95_ms,latency_p99_ms,")
                f.write("cpu_percent,memory_mb,gpu_percent,gpu_memory_mb,timestamp\n")

                # Data
                for r in self.results:
                    f.write(f"{r.name},{r.duration_ms:.2f},{r.throughput_fps:.2f},")
                    f.write(f"{r.latency_p50_ms:.2f},{r.latency_p95_ms:.2f},{r.latency_p99_ms:.2f},")
                    f.write(f"{r.cpu_utilization_percent:.1f},{r.memory_mb:.2f},")
                    f.write(f"{r.gpu_utilization_percent:.1f},{r.gpu_memory_mb:.2f},{r.timestamp}\n")
        print(f"Saved CSV to {csv_file}")

    def generate_report(self):
        """Generate HTML performance report with graphs"""
        if not self.results:
            print("No results to report")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create comparison plots
        self._plot_throughput_comparison()
        self._plot_latency_distribution()
        self._plot_resource_utilization()

        # Generate HTML report
        html_file = self.output_dir / f"report_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(self._generate_html_report())

        print(f"\nGenerated report: {html_file}")

    def _plot_throughput_comparison(self):
        """Plot throughput comparison across benchmarks"""
        if not self.results:
            return

        names = [r.name for r in self.results]
        throughputs = [r.throughput_fps for r in self.results]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(names)), throughputs, color='steelblue')
        plt.xlabel('Benchmark')
        plt.ylabel('Throughput (FPS)')
        plt.title('Throughput Comparison')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{throughputs[i]:.1f}',
                    ha='center', va='bottom', fontsize=9)

        plt.savefig(self.output_dir / 'throughput_comparison.png', dpi=150)
        plt.close()

    def _plot_latency_distribution(self):
        """Plot latency percentiles"""
        if not self.results:
            return

        names = [r.name for r in self.results]
        p50 = [r.latency_p50_ms for r in self.results]
        p95 = [r.latency_p95_ms for r in self.results]
        p99 = [r.latency_p99_ms for r in self.results]

        x = np.arange(len(names))
        width = 0.25

        plt.figure(figsize=(14, 6))
        plt.bar(x - width, p50, width, label='p50', color='lightgreen')
        plt.bar(x, p95, width, label='p95', color='orange')
        plt.bar(x + width, p99, width, label='p99', color='red')

        plt.xlabel('Benchmark')
        plt.ylabel('Latency (ms)')
        plt.title('Latency Distribution (Percentiles)')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.output_dir / 'latency_distribution.png', dpi=150)
        plt.close()

    def _plot_resource_utilization(self):
        """Plot CPU/GPU utilization"""
        if not self.results:
            return

        names = [r.name for r in self.results]
        cpu_util = [r.cpu_utilization_percent for r in self.results]
        gpu_util = [r.gpu_utilization_percent for r in self.results]

        x = np.arange(len(names))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # CPU/GPU utilization
        ax1.bar(x - width/2, cpu_util, width, label='CPU %', color='cornflowerblue')
        ax1.bar(x + width/2, gpu_util, width, label='GPU %', color='orange')
        ax1.set_ylabel('Utilization (%)')
        ax1.set_title('CPU vs GPU Utilization')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Memory usage
        cpu_mem = [r.memory_mb for r in self.results]
        gpu_mem = [r.gpu_memory_mb for r in self.results]

        ax2.bar(x - width/2, cpu_mem, width, label='CPU Memory (MB)', color='steelblue')
        ax2.bar(x + width/2, gpu_mem, width, label='GPU Memory (MB)', color='darkorange')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Memory Usage')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'resource_utilization.png', dpi=150)
        plt.close()

    def _generate_html_report(self) -> str:
        """Generate HTML report content"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>PixelToVoxel Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; background: white; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .metric-label { font-size: 14px; color: #666; }
        .chart { margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; }
        .warning { color: #ff6b6b; font-weight: bold; }
        .success { color: #51cf66; font-weight: bold; }
    </style>
</head>
<body>
    <h1>PixelToVoxel Performance Benchmark Report</h1>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

    <h2>Summary</h2>
"""

        if self.results:
            avg_throughput = np.mean([r.throughput_fps for r in self.results])
            avg_latency = np.mean([r.latency_p50_ms for r in self.results])
            avg_cpu = np.mean([r.cpu_utilization_percent for r in self.results])
            avg_gpu = np.mean([r.gpu_utilization_percent for r in self.results])

            html += f"""
    <div class="metric">
        <div class="metric-value">{avg_throughput:.1f}</div>
        <div class="metric-label">Avg Throughput (FPS)</div>
    </div>
    <div class="metric">
        <div class="metric-value">{avg_latency:.1f}</div>
        <div class="metric-label">Avg Latency (ms)</div>
    </div>
    <div class="metric">
        <div class="metric-value">{avg_cpu:.0f}%</div>
        <div class="metric-label">Avg CPU Usage</div>
    </div>
    <div class="metric">
        <div class="metric-value">{avg_gpu:.0f}%</div>
        <div class="metric-label">Avg GPU Usage</div>
    </div>
"""

        html += """
    <h2>Performance Charts</h2>
    <div class="chart">
        <h3>Throughput Comparison</h3>
        <img src="throughput_comparison.png" alt="Throughput Comparison">
    </div>
    <div class="chart">
        <h3>Latency Distribution</h3>
        <img src="latency_distribution.png" alt="Latency Distribution">
    </div>
    <div class="chart">
        <h3>Resource Utilization</h3>
        <img src="resource_utilization.png" alt="Resource Utilization">
    </div>

    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Throughput (FPS)</th>
            <th>p50 (ms)</th>
            <th>p95 (ms)</th>
            <th>p99 (ms)</th>
            <th>CPU %</th>
            <th>GPU %</th>
            <th>Memory (MB)</th>
            <th>Status</th>
        </tr>
"""

        for result in self.results:
            regressions = self.check_regression(result)
            status = '<span class="warning">REGRESSION</span>' if regressions else '<span class="success">PASS</span>'

            html += f"""
        <tr>
            <td>{result.name}</td>
            <td>{result.throughput_fps:.2f}</td>
            <td>{result.latency_p50_ms:.2f}</td>
            <td>{result.latency_p95_ms:.2f}</td>
            <td>{result.latency_p99_ms:.2f}</td>
            <td>{result.cpu_utilization_percent:.1f}</td>
            <td>{result.gpu_utilization_percent:.1f}</td>
            <td>{result.memory_mb:.1f}</td>
            <td>{status}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""
        return html


# Example benchmark functions
def benchmark_voxel_ray_casting(grid_size=500, num_rays=1000):
    """Benchmark voxel ray casting performance"""
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # Simulate ray casting
    for _ in range(num_rays):
        # Random ray origin and direction
        origin = np.random.rand(3) * grid_size
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)

        # Simple DDA-like traversal
        t = 0
        step = 1.0
        max_t = grid_size * 1.414  # Diagonal

        while t < max_t:
            pos = origin + direction * t
            idx = np.clip(pos.astype(int), 0, grid_size - 1)

            if np.all(idx >= 0) and np.all(idx < grid_size):
                grid[idx[0], idx[1], idx[2]] += 1.0

            t += step


def benchmark_motion_detection(width=7680, height=4320):
    """Benchmark motion detection on 8K frames"""
    if not HAS_CV2:
        return

    # Generate synthetic frames
    frame1 = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    frame2 = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # Motion detection
    diff = cv2.absdiff(frame1, frame2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)


def benchmark_voxel_update(grid_size=500, num_updates=10000):
    """Benchmark voxel grid update performance"""
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # Random updates
    indices = np.random.randint(0, grid_size, (num_updates, 3))
    values = np.random.rand(num_updates).astype(np.float32)

    for idx, val in zip(indices, values):
        grid[idx[0], idx[1], idx[2]] += val


def main():
    """Run the complete benchmark suite"""
    suite = BenchmarkSuite(output_dir="benchmark_results")

    print("="*60)
    print("PixelToVoxel Performance Benchmark Suite")
    print("="*60)

    # Run benchmarks
    suite.run_benchmark(
        "Voxel Ray Casting (500^3)",
        benchmark_voxel_ray_casting,
        iterations=50,
        warmup=5,
        grid_size=500,
        num_rays=1000
    )

    if HAS_CV2:
        suite.run_benchmark(
            "Motion Detection (8K)",
            benchmark_motion_detection,
            iterations=50,
            warmup=5,
            width=7680,
            height=4320
        )

    suite.run_benchmark(
        "Voxel Grid Updates",
        benchmark_voxel_update,
        iterations=100,
        warmup=10,
        grid_size=500,
        num_updates=10000
    )

    # Save results and generate report
    suite.save_results()
    suite.generate_report()

    # Ask if user wants to save as baseline
    print("\n" + "="*60)
    response = input("Save these results as performance baseline? (y/n): ")
    if response.lower() == 'y':
        suite.save_baselines()

    print("\n" + "="*60)
    print("Benchmark suite completed!")
    print("="*60)


if __name__ == "__main__":
    main()
