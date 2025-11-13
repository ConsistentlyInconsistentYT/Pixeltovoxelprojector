#!/usr/bin/env python3
"""
Quick Benchmark - Fast performance check

Runs a subset of benchmarks for quick performance verification.
Use this for rapid testing during development.

For comprehensive benchmarks, use: python run_all_benchmarks.py
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_suite import BenchmarkSuite, benchmark_voxel_ray_casting, benchmark_voxel_update


def main():
    """Run quick benchmarks"""
    print("="*60)
    print(" Quick Performance Benchmark")
    print("="*60)
    print("\nRunning subset of benchmarks for quick verification...")
    print("For full benchmarks, use: python run_all_benchmarks.py\n")

    suite = BenchmarkSuite(output_dir="benchmark_results")

    # Quick tests with fewer iterations
    suite.run_benchmark(
        "Quick Voxel Ray Casting",
        benchmark_voxel_ray_casting,
        iterations=20,
        warmup=3,
        grid_size=256,  # Smaller grid
        num_rays=500    # Fewer rays
    )

    suite.run_benchmark(
        "Quick Voxel Updates",
        benchmark_voxel_update,
        iterations=30,
        warmup=3,
        grid_size=256,
        num_updates=5000
    )

    # Save results
    suite.save_results()

    print("\n" + "="*60)
    print(" Quick Benchmark Complete")
    print("="*60)
    print("\nResults saved to: benchmark_results/")
    print("\nFor detailed reports and full benchmarks:")
    print("  python run_all_benchmarks.py")


if __name__ == "__main__":
    main()
