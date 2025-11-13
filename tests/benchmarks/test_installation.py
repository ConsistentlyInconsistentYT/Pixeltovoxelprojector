#!/usr/bin/env python3
"""
Test Benchmark Installation

Verifies that all required dependencies are installed and working.
Run this before running benchmarks to ensure everything is set up correctly.
"""

import sys
import subprocess
from pathlib import Path


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info

    if version.major >= 3 and version.minor >= 7:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (requires >= 3.7)")
        return False


def check_module(module_name, description=""):
    """Check if a Python module is installed"""
    try:
        __import__(module_name)
        version = ""
        try:
            mod = __import__(module_name)
            if hasattr(mod, '__version__'):
                version = f" ({mod.__version__})"
        except:
            pass

        desc = f" - {description}" if description else ""
        print_success(f"{module_name}{version}{desc}")
        return True
    except ImportError:
        desc = f" - {description}" if description else ""
        print_error(f"{module_name} not installed{desc}")
        return False


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")

    # Check nvcc
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.decode()
            # Extract version
            for line in output.split('\n'):
                if 'release' in line.lower():
                    print_success(f"CUDA Compiler: {line.strip()}")
                    break
        else:
            print_warning("nvcc found but returned error")
            return False
    except FileNotFoundError:
        print_warning("nvcc not found - CUDA benchmarks will not be available")
        return False
    except Exception as e:
        print_warning(f"Cannot check nvcc: {e}")
        return False

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.decode().strip()
            if output:
                gpu_info = output.split(',')
                print_success(f"GPU: {gpu_info[0].strip()}")
                print_success(f"Driver: {gpu_info[1].strip()}")
                print_success(f"Memory: {gpu_info[2].strip()}")
            return True
        else:
            print_warning("nvidia-smi found but returned error")
            return False
    except FileNotFoundError:
        print_warning("nvidia-smi not found - cannot detect GPU")
        return False
    except Exception as e:
        print_warning(f"Cannot check nvidia-smi: {e}")
        return False


def check_benchmark_files():
    """Check that benchmark files exist"""
    print("\nChecking benchmark files...")

    benchmark_dir = Path(__file__).parent
    files = [
        ('benchmark_suite.py', 'Main benchmark suite'),
        ('camera_benchmark.py', 'Camera benchmarks'),
        ('voxel_benchmark.cu', 'CUDA voxel benchmarks'),
        ('network_benchmark.py', 'Network benchmarks'),
        ('run_all_benchmarks.py', 'Comprehensive runner'),
        ('Makefile', 'CUDA build file'),
    ]

    all_exist = True
    for filename, description in files:
        filepath = benchmark_dir / filename
        if filepath.exists():
            print_success(f"{filename} - {description}")
        else:
            print_error(f"{filename} - {description} (MISSING)")
            all_exist = False

    return all_exist


def check_output_directory():
    """Check/create output directory"""
    print("\nChecking output directory...")

    output_dir = Path(__file__).parent / "benchmark_results"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Output directory: {output_dir}")
        return True
    except Exception as e:
        print_error(f"Cannot create output directory: {e}")
        return False


def main():
    """Run all installation tests"""
    print_header("Benchmark Installation Test")

    results = {}

    # Python version
    print_header("Python Environment")
    results['python'] = check_python_version()

    # Required modules
    print("\nChecking required modules...")
    required_modules = {
        'numpy': 'Numerical computing',
        'matplotlib': 'Plotting and graphs',
        'psutil': 'System monitoring',
    }

    for module, desc in required_modules.items():
        results[module] = check_module(module, desc)

    # Optional modules
    print("\nChecking optional modules...")
    optional_modules = {
        'cv2': 'OpenCV - required for camera benchmarks',
        'pynvml': 'NVIDIA ML - required for GPU monitoring',
    }

    for module, desc in optional_modules.items():
        check_module(module, desc)

    # CUDA
    results['cuda'] = check_cuda()

    # Files
    results['files'] = check_benchmark_files()

    # Output directory
    results['output'] = check_output_directory()

    # Summary
    print_header("Installation Summary")

    required_ok = all([
        results['python'],
        results['numpy'],
        results['matplotlib'],
        results['psutil'],
        results['files'],
        results['output'],
    ])

    if required_ok:
        print_success("All required components are installed")
        print("\nYou can now run benchmarks:")
        print(f"  {Colors.BOLD}python quick_benchmark.py{Colors.RESET}      - Quick test")
        print(f"  {Colors.BOLD}python run_all_benchmarks.py{Colors.RESET}  - Full suite")
    else:
        print_error("Some required components are missing")
        print("\nTo install missing Python packages:")
        print(f"  {Colors.BOLD}pip install -r requirements.txt{Colors.RESET}")

    if not results.get('cuda', False):
        print_warning("\nCUDA not available - CUDA benchmarks will be skipped")
        print("To enable CUDA benchmarks:")
        print("  1. Install NVIDIA CUDA Toolkit")
        print("  2. Ensure nvcc is in your PATH")
        print("  3. Run 'make' to compile voxel_benchmark")

    print()


if __name__ == "__main__":
    main()
