#!/usr/bin/env python3
"""
Benchmark Comparison Tool

Compare benchmark results to detect performance regressions.
Compares current benchmark results against a baseline and reports
any significant performance degradation.

Usage:
    python compare_benchmarks.py --baseline baseline.json --current latest.json
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkComparison:
    """Comparison result for a single benchmark"""
    name: str
    baseline_fps: float
    current_fps: float
    change_percent: float
    is_regression: bool


class BenchmarkComparator:
    """Compare benchmark results and detect regressions"""

    def __init__(self, threshold_percent: float = 10.0):
        """
        Initialize comparator

        Args:
            threshold_percent: Performance degradation threshold as percentage
        """
        self.threshold = threshold_percent
        self.comparisons: List[BenchmarkComparison] = []

    def load_results(self, filepath: Path) -> Dict:
        """Load benchmark results from JSON file"""
        if not filepath.exists():
            raise FileNotFoundError(f"Benchmark file not found: {filepath}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def compare(self, baseline_file: Path, current_file: Path) -> bool:
        """
        Compare two benchmark result files

        Args:
            baseline_file: Path to baseline results
            current_file: Path to current results

        Returns:
            True if no regressions detected, False otherwise
        """
        print("="*70)
        print(" Benchmark Comparison")
        print("="*70)
        print()

        # Load results
        baseline = self.load_results(baseline_file)
        current = self.load_results(current_file)

        print(f"Baseline: {baseline_file}")
        print(f"Current:  {current_file}")
        print(f"Threshold: {self.threshold}%")
        print()

        # Extract benchmark data
        baseline_benchmarks = self._extract_benchmarks(baseline)
        current_benchmarks = self._extract_benchmarks(current)

        # Compare common benchmarks
        common_names = set(baseline_benchmarks.keys()) & set(current_benchmarks.keys())

        if not common_names:
            print("Warning: No common benchmarks found")
            return True

        print(f"Comparing {len(common_names)} benchmarks...")
        print()

        # Perform comparisons
        has_regression = False
        for name in sorted(common_names):
            baseline_fps = baseline_benchmarks[name]
            current_fps = current_benchmarks[name]

            # Calculate percentage change (negative means worse performance)
            change_percent = ((current_fps - baseline_fps) / baseline_fps) * 100
            is_regression = change_percent < -self.threshold

            comparison = BenchmarkComparison(
                name=name,
                baseline_fps=baseline_fps,
                current_fps=current_fps,
                change_percent=change_percent,
                is_regression=is_regression
            )

            self.comparisons.append(comparison)

            if is_regression:
                has_regression = True

        # Print results
        self._print_results()

        return not has_regression

    def _extract_benchmarks(self, results: Dict) -> Dict[str, float]:
        """Extract benchmark names and FPS values from results"""
        benchmarks = {}

        # Handle different result formats
        if 'suites' in results:
            # Combined results format
            for suite_name, suite_data in results['suites'].items():
                if suite_data.get('completed') and 'results' in suite_data:
                    for result in suite_data['results']:
                        name = result.get('name', 'unknown')
                        fps = result.get('throughput_fps', 0.0)
                        benchmarks[name] = fps

        elif 'results' in results:
            # Single suite format
            for result in results['results']:
                name = result.get('name', 'unknown')
                fps = result.get('throughput_fps', 0.0)
                benchmarks[name] = fps

        elif 'benchmarks' in results:
            # Direct benchmark format
            for name, data in results['benchmarks'].items():
                fps = data.get('fps', data.get('throughput_fps', 0.0))
                benchmarks[name] = fps

        return benchmarks

    def _print_results(self):
        """Print comparison results in a formatted table"""
        print("="*70)
        print(" Results")
        print("="*70)
        print()

        # Print header
        print(f"{'Benchmark':<40} {'Baseline':>10} {'Current':>10} {'Change':>10}")
        print("-"*70)

        # Print each comparison
        for comp in sorted(self.comparisons, key=lambda x: x.change_percent):
            status = "REGRESSION" if comp.is_regression else "OK"
            status_symbol = "✗" if comp.is_regression else "✓"

            color_code = ""
            if comp.is_regression:
                color_code = "\033[91m"  # Red
            elif comp.change_percent > 5:
                color_code = "\033[92m"  # Green (improvement)
            reset_code = "\033[0m" if color_code else ""

            print(f"{status_symbol} {comp.name:<38} "
                  f"{comp.baseline_fps:>9.2f} "
                  f"{comp.current_fps:>9.2f} "
                  f"{color_code}{comp.change_percent:>+9.1f}%{reset_code}")

        print()

        # Print summary
        regressions = [c for c in self.comparisons if c.is_regression]
        improvements = [c for c in self.comparisons if c.change_percent > 5]

        print("="*70)
        print(" Summary")
        print("="*70)
        print()
        print(f"Total benchmarks: {len(self.comparisons)}")
        print(f"Regressions:      {len(regressions)}")
        print(f"Improvements:     {len(improvements)}")
        print()

        if regressions:
            print("\033[91m✗ Performance regressions detected!\033[0m")
            print()
            print("Benchmarks with regressions:")
            for comp in regressions:
                print(f"  - {comp.name}: {comp.change_percent:+.1f}%")
            print()
        else:
            print("\033[92m✓ No performance regressions detected\033[0m")
            print()

    def export_report(self, output_file: Path):
        """Export comparison report to file"""
        report = {
            'threshold_percent': self.threshold,
            'total_benchmarks': len(self.comparisons),
            'regressions_count': len([c for c in self.comparisons if c.is_regression]),
            'comparisons': [
                {
                    'name': c.name,
                    'baseline_fps': c.baseline_fps,
                    'current_fps': c.current_fps,
                    'change_percent': c.change_percent,
                    'is_regression': c.is_regression,
                }
                for c in self.comparisons
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report exported to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Compare benchmark results and detect performance regressions'
    )

    parser.add_argument(
        '--baseline',
        type=Path,
        required=True,
        help='Path to baseline benchmark results (JSON)'
    )

    parser.add_argument(
        '--current',
        type=Path,
        required=True,
        help='Path to current benchmark results (JSON)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=10.0,
        help='Performance degradation threshold percentage (default: 10.0)'
    )

    parser.add_argument(
        '--fail-on-regression',
        action='store_true',
        help='Exit with error code if regressions detected'
    )

    parser.add_argument(
        '--export',
        type=Path,
        help='Export comparison report to file'
    )

    args = parser.parse_args()

    # Create comparator
    comparator = BenchmarkComparator(threshold_percent=args.threshold)

    try:
        # Perform comparison
        no_regression = comparator.compare(args.baseline, args.current)

        # Export report if requested
        if args.export:
            comparator.export_report(args.export)

        # Exit with appropriate code
        if args.fail_on_regression and not no_regression:
            print("Exiting with error code due to performance regressions")
            sys.exit(1)
        else:
            sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
