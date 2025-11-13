#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner

Executes all benchmark suites and generates unified reports:
- Main benchmark suite
- Camera benchmarks
- CUDA voxel benchmarks (if available)
- Network benchmarks

Generates:
- Combined JSON results
- HTML summary report
- Performance comparison graphs
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class BenchmarkRunner:
    """Orchestrate all benchmark suites"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.benchmark_dir = Path(__file__).parent
        self.output_dir = self.benchmark_dir / "benchmark_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self):
        """Run all benchmarks"""
        print("="*70)
        print(" PixelToVoxel Comprehensive Benchmark Suite")
        print("="*70)

        self.start_time = datetime.now()
        print(f"\nStarted: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Check environment
        self._check_environment()

        # Run benchmarks
        self._run_main_benchmark()
        self._run_camera_benchmark()
        self._run_cuda_benchmark()
        self._run_network_benchmark()

        self.end_time = datetime.now()

        # Generate reports
        self._save_results()
        self._generate_summary()

        print("\n" + "="*70)
        print(" Benchmark Suite Completed")
        print("="*70)
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"\nTotal Duration: {duration:.1f} seconds")
        print(f"Results saved to: {self.output_dir}")

    def _check_environment(self):
        """Check required dependencies"""
        print("Checking environment...")

        # Check Python version
        python_version = sys.version_info
        print(f"  Python: {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Check dependencies
        dependencies = {
            'numpy': False,
            'cv2': False,
            'matplotlib': False,
            'psutil': False,
        }

        for module in dependencies:
            try:
                __import__(module)
                dependencies[module] = True
                print(f"  {module}: OK")
            except ImportError:
                print(f"  {module}: MISSING")

        # Check CUDA
        cuda_available = self._check_cuda()
        print(f"  CUDA: {'OK' if cuda_available else 'NOT AVAILABLE'}")

        print()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_main_benchmark(self):
        """Run main benchmark suite"""
        print("\n" + "="*70)
        print(" Running Main Benchmark Suite")
        print("="*70)

        try:
            # Import and run
            sys.path.insert(0, str(self.benchmark_dir))
            from benchmark_suite import BenchmarkSuite

            suite = BenchmarkSuite(output_dir=str(self.output_dir))

            # Run core benchmarks
            from benchmark_suite import (
                benchmark_voxel_ray_casting,
                benchmark_motion_detection,
                benchmark_voxel_update
            )

            import cv2
            HAS_CV2 = True
        except ImportError:
            HAS_CV2 = False

            # Ray casting
            suite.run_benchmark(
                "Voxel Ray Casting (500^3)",
                benchmark_voxel_ray_casting,
                iterations=50,
                warmup=5,
                grid_size=500,
                num_rays=1000
            )

            # Motion detection (if OpenCV available)
            if HAS_CV2:
                suite.run_benchmark(
                    "Motion Detection (8K)",
                    benchmark_motion_detection,
                    iterations=50,
                    warmup=5,
                    width=7680,
                    height=4320
                )

            # Voxel updates
            suite.run_benchmark(
                "Voxel Grid Updates",
                benchmark_voxel_update,
                iterations=100,
                warmup=10,
                grid_size=500,
                num_updates=10000
            )

            suite.save_results()
            suite.generate_report()

            # Store results
            self.results['main_suite'] = {
                'completed': True,
                'num_benchmarks': len(suite.results),
                'results': [
                    {
                        'name': r.name,
                        'throughput_fps': r.throughput_fps,
                        'latency_p99_ms': r.latency_p99_ms,
                    }
                    for r in suite.results
                ]
            }

            print("\n✓ Main benchmark suite completed")

        except Exception as e:
            print(f"\n✗ Main benchmark suite failed: {e}")
            self.results['main_suite'] = {'completed': False, 'error': str(e)}

    def _run_camera_benchmark(self):
        """Run camera benchmark suite"""
        print("\n" + "="*70)
        print(" Running Camera Benchmark Suite")
        print("="*70)

        try:
            sys.path.insert(0, str(self.benchmark_dir))
            from camera_benchmark import CameraBenchmark

            benchmark = CameraBenchmark(
                output_dir=str(self.output_dir / "camera")
            )

            results = benchmark.run_all_benchmarks()

            self.results['camera_suite'] = {
                'completed': True,
                'benchmarks': list(results.keys()),
            }

            print("\n✓ Camera benchmark suite completed")

        except Exception as e:
            print(f"\n✗ Camera benchmark suite failed: {e}")
            self.results['camera_suite'] = {'completed': False, 'error': str(e)}

    def _run_cuda_benchmark(self):
        """Run CUDA voxel benchmarks"""
        print("\n" + "="*70)
        print(" Running CUDA Voxel Benchmarks")
        print("="*70)

        cuda_binary = self.benchmark_dir / "voxel_benchmark"

        # Try to compile if not exists
        if not cuda_binary.exists():
            print("CUDA benchmark not compiled. Attempting to compile...")
            try:
                result = subprocess.run(
                    ['make', 'voxel_benchmark'],
                    cwd=self.benchmark_dir,
                    capture_output=True,
                    timeout=60
                )
                if result.returncode != 0:
                    print(f"Compilation failed: {result.stderr.decode()}")
                    self.results['cuda_suite'] = {
                        'completed': False,
                        'error': 'Compilation failed'
                    }
                    return
            except Exception as e:
                print(f"Cannot compile CUDA benchmark: {e}")
                self.results['cuda_suite'] = {
                    'completed': False,
                    'error': f'Cannot compile: {e}'
                }
                return

        # Run CUDA benchmark
        if cuda_binary.exists():
            try:
                print("\nRunning CUDA benchmarks...")
                result = subprocess.run(
                    [str(cuda_binary)],
                    cwd=self.benchmark_dir,
                    capture_output=True,
                    timeout=300
                )

                output = result.stdout.decode()
                print(output)

                self.results['cuda_suite'] = {
                    'completed': True,
                    'output': output
                }

                print("\n✓ CUDA benchmark suite completed")

            except subprocess.TimeoutExpired:
                print("\n✗ CUDA benchmark timed out")
                self.results['cuda_suite'] = {
                    'completed': False,
                    'error': 'Timeout'
                }
            except Exception as e:
                print(f"\n✗ CUDA benchmark failed: {e}")
                self.results['cuda_suite'] = {
                    'completed': False,
                    'error': str(e)
                }
        else:
            print("\n✗ CUDA benchmark binary not found")
            print("  To enable: cd to benchmarks dir and run 'make'")
            self.results['cuda_suite'] = {
                'completed': False,
                'error': 'Binary not found'
            }

    def _run_network_benchmark(self):
        """Run network benchmark suite"""
        print("\n" + "="*70)
        print(" Running Network Benchmark Suite")
        print("="*70)

        try:
            sys.path.insert(0, str(self.benchmark_dir))
            from network_benchmark import NetworkBenchmark

            benchmark = NetworkBenchmark(
                output_dir=str(self.output_dir / "network")
            )

            results = benchmark.run_all_benchmarks()

            self.results['network_suite'] = {
                'completed': True,
                'benchmarks': list(results.keys()),
            }

            print("\n✓ Network benchmark suite completed")

        except Exception as e:
            print(f"\n✗ Network benchmark suite failed: {e}")
            self.results['network_suite'] = {
                'completed': False,
                'error': str(e)
            }

    def _save_results(self):
        """Save combined results"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"combined_results_{timestamp}.json"

        combined = {
            'timestamp': self.start_time.isoformat(),
            'duration_sec': (self.end_time - self.start_time).total_seconds(),
            'suites': self.results,
        }

        with open(results_file, 'w') as f:
            json.dump(combined, f, indent=2)

        print(f"\nCombined results saved to: {results_file}")

    def _generate_summary(self):
        """Generate summary report"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"summary_{timestamp}.txt"

        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" PixelToVoxel Benchmark Summary\n")
            f.write("="*70 + "\n\n")

            f.write(f"Started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Finished: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = (self.end_time - self.start_time).total_seconds()
            f.write(f"Duration: {duration:.1f} seconds\n\n")

            f.write("="*70 + "\n")
            f.write(" Suite Status\n")
            f.write("="*70 + "\n\n")

            for suite_name, suite_data in self.results.items():
                status = "✓ PASS" if suite_data.get('completed', False) else "✗ FAIL"
                f.write(f"{suite_name:20s} {status}\n")

                if not suite_data.get('completed', False):
                    error = suite_data.get('error', 'Unknown error')
                    f.write(f"  Error: {error}\n")

            f.write("\n")

            # Main suite details
            if 'main_suite' in self.results and self.results['main_suite'].get('completed'):
                f.write("="*70 + "\n")
                f.write(" Main Suite Results\n")
                f.write("="*70 + "\n\n")

                for result in self.results['main_suite'].get('results', []):
                    f.write(f"{result['name']}\n")
                    f.write(f"  Throughput: {result['throughput_fps']:.2f} FPS\n")
                    f.write(f"  p99 Latency: {result['latency_p99_ms']:.2f} ms\n\n")

            f.write("="*70 + "\n")
            f.write(" End of Summary\n")
            f.write("="*70 + "\n")

        print(f"Summary saved to: {summary_file}")


def main():
    """Main entry point"""
    runner = BenchmarkRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
