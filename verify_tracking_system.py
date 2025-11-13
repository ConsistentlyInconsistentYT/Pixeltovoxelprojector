#!/usr/bin/env python3
"""
Verification script for Camera Tracking System

This script verifies that all components are properly installed and functional.
"""

import sys
import os
import importlib.util

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: NOT FOUND - {filepath}")
        return False


def check_python_import(module_name, description):
    """Check if a Python module can be imported"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"✓ {description}: {module_name}")
            return True
        else:
            print(f"✗ {description}: NOT FOUND - {module_name}")
            return False
    except Exception as e:
        print(f"✗ {description}: ERROR - {e}")
        return False


def main():
    print("=" * 80)
    print("Camera Tracking System - Verification")
    print("=" * 80)
    print()

    all_checks_passed = True

    # Check main implementation files
    print("1. Checking implementation files...")
    files_to_check = [
        ("/home/user/Pixeltovoxelprojector/src/camera/pose_tracker.py", "Pose Tracker"),
        ("/home/user/Pixeltovoxelprojector/src/camera/orientation_manager.cpp", "Orientation Manager"),
        ("/home/user/Pixeltovoxelprojector/src/camera/position_broadcast.py", "Position Broadcast"),
        ("/home/user/Pixeltovoxelprojector/src/camera/__init__.py", "Package Init"),
    ]

    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    print()

    # Check documentation
    print("2. Checking documentation...")
    doc_files = [
        ("/home/user/Pixeltovoxelprojector/src/camera/README.md", "README"),
        ("/home/user/Pixeltovoxelprojector/CAMERA_TRACKING_IMPLEMENTATION.md", "Implementation Doc"),
        ("/home/user/Pixeltovoxelprojector/examples/camera_tracking_example.py", "Example Script"),
    ]

    for filepath, description in doc_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    print()

    # Check build files
    print("3. Checking build configuration...")
    build_files = [
        ("/home/user/Pixeltovoxelprojector/src/camera/CMakeLists.txt", "CMake Config"),
        ("/home/user/Pixeltovoxelprojector/src/camera/build.sh", "Build Script"),
        ("/home/user/Pixeltovoxelprojector/src/camera/requirements.txt", "Requirements"),
    ]

    for filepath, description in build_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    print()

    # Check Python dependencies
    print("4. Checking Python dependencies...")
    dependencies = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("zmq", "PyZMQ"),
    ]

    for module, description in dependencies:
        if not check_python_import(module, description):
            all_checks_passed = False
            print(f"   Install with: pip install {module}")
    print()

    # Try importing the camera tracking modules
    print("5. Checking camera tracking modules...")
    try:
        from camera.pose_tracker import CameraPoseTracker
        print("✓ Can import CameraPoseTracker")
    except Exception as e:
        print(f"✗ Cannot import CameraPoseTracker: {e}")
        all_checks_passed = False

    try:
        from camera.position_broadcast import PositionBroadcaster
        print("✓ Can import PositionBroadcaster")
    except Exception as e:
        print(f"✗ Cannot import PositionBroadcaster: {e}")
        all_checks_passed = False
    print()

    # File statistics
    print("6. Implementation statistics...")
    try:
        import subprocess
        result = subprocess.run(
            ["wc", "-l",
             "/home/user/Pixeltovoxelprojector/src/camera/pose_tracker.py",
             "/home/user/Pixeltovoxelprojector/src/camera/orientation_manager.cpp",
             "/home/user/Pixeltovoxelprojector/src/camera/position_broadcast.py"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"  Could not get statistics: {e}")
    print()

    # Final summary
    print("=" * 80)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("Next steps:")
        print("  1. Install dependencies: cd src/camera && pip install -r requirements.txt")
        print("  2. Build C++ code: cd src/camera && ./build.sh")
        print("  3. Run example: python examples/camera_tracking_example.py")
        print("  4. Read documentation: src/camera/README.md")
    else:
        print("✗ SOME CHECKS FAILED")
        print()
        print("Please install missing dependencies and ensure all files are present.")
    print("=" * 80)

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
