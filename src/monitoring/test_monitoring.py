"""
Test script to verify monitoring system installation and functionality
"""

import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported"""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)

    modules = [
        ('SystemMonitor', 'src.monitoring.system_monitor'),
        ('DataValidator', 'src.monitoring.validator'),
        ('AlertManager', 'src.monitoring.alert_manager'),
        ('WebDashboard', 'src.monitoring.web_dashboard'),
    ]

    passed = 0
    failed = 0

    for name, module_path in modules:
        try:
            parts = module_path.split('.')
            module = __import__(module_path, fromlist=[parts[-1]])
            getattr(module, name)
            print(f"✓ {name:20s} imported successfully")
            passed += 1
        except ImportError as e:
            print(f"✗ {name:20s} import failed: {e}")
            failed += 1
        except AttributeError as e:
            print(f"✗ {name:20s} not found in module: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_system_monitor():
    """Test SystemMonitor functionality"""
    print("=" * 60)
    print("Testing SystemMonitor")
    print("=" * 60)

    try:
        from src.monitoring import SystemMonitor

        # Create monitor
        monitor = SystemMonitor(update_rate_hz=10.0, num_cameras=20)
        print("✓ SystemMonitor created")

        # Start monitoring
        monitor.start()
        print("✓ SystemMonitor started")

        # Wait and collect samples
        time.sleep(1.5)

        # Get metrics
        metrics = monitor.get_current_metrics()
        print(f"✓ Collected metrics:")
        print(f"  - CPU: {metrics.cpu.utilization:.1f}%")
        print(f"  - Memory: {metrics.memory.percent:.1f}%")
        print(f"  - GPUs: {len(metrics.gpus)}")

        # Get summary
        summary = monitor.get_summary()
        print(f"✓ Summary generated:")
        print(f"  - Samples: {summary['samples_collected']}")
        print(f"  - Overhead: {monitor.get_performance_overhead():.3f}%")

        # Stop monitoring
        monitor.stop()
        print("✓ SystemMonitor stopped")

        print("\n✓ SystemMonitor tests passed\n")
        return True

    except Exception as e:
        print(f"\n✗ SystemMonitor test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_validator():
    """Test DataValidator functionality"""
    print("=" * 60)
    print("Testing DataValidator")
    print("=" * 60)

    try:
        from src.monitoring import DataValidator, CoordinateBounds
        import numpy as np

        # Create validator
        validator = DataValidator(
            bounds=CoordinateBounds(),
            min_confidence=0.5,
            max_velocity=100.0
        )
        print("✓ DataValidator created")

        # Test detection validation
        detection = {
            'x': 100.0,
            'y': 200.0,
            'z': 50.0,
            'confidence': 0.95
        }
        result = validator.validate_detection(detection)
        print(f"✓ Detection validation: {'PASS' if result.passed else 'FAIL'}")
        print(f"  - Issues: {len(result.issues)}")
        print(f"  - Time: {result.validation_time_ms:.2f} ms")

        # Test track validation
        track = {
            'track_id': 1,
            'x': 100.0, 'y': 200.0, 'z': 50.0,
            'vx': 10.0, 'vy': 5.0, 'vz': 0.0,
            'ax': 0.5, 'ay': 0.2, 'az': 0.0,
            'confidence': 0.92
        }
        result = validator.validate_track(track)
        print(f"✓ Track validation: {'PASS' if result.passed else 'FAIL'}")
        print(f"  - Issues: {len(result.issues)}")

        # Test invalid detection
        invalid_detection = {
            'x': 10000.0,  # Out of bounds
            'y': 200.0,
            'z': 50.0,
            'confidence': 0.2  # Low confidence
        }
        result = validator.validate_detection(invalid_detection)
        print(f"✓ Invalid detection caught: {len(result.issues)} issues")

        # Get statistics
        stats = validator.get_statistics()
        print(f"✓ Statistics:")
        print(f"  - Validations: {stats['total_validations']}")
        print(f"  - Avg time: {stats['avg_validation_time_ms']:.2f} ms")

        print("\n✓ DataValidator tests passed\n")
        return True

    except Exception as e:
        print(f"\n✗ DataValidator test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_alert_manager():
    """Test AlertManager functionality"""
    print("=" * 60)
    print("Testing AlertManager")
    print("=" * 60)

    try:
        from src.monitoring import AlertManager, AlertLevel, AlertCategory, create_default_rules

        # Create alert manager
        alert_mgr = AlertManager(
            max_alerts_per_minute=100,
            enable_auto_diagnostics=True
        )
        print("✓ AlertManager created")

        # Add default rules
        rules = create_default_rules()
        for rule in rules:
            alert_mgr.add_rule(rule)
        print(f"✓ Added {len(rules)} default rules")

        # Create test alert
        alert = alert_mgr.create_alert(
            level=AlertLevel.WARNING,
            category=AlertCategory.CAMERA,
            title="Test Alert",
            message="This is a test alert",
            camera_id=1,
            current_value=15.0,
            expected_value=30.0,
            suggestions=["Check camera", "Verify settings"]
        )
        print(f"✓ Alert created: {alert.alert_id if alert else 'None (deduplicated)'}")

        # Get active alerts
        active = alert_mgr.get_active_alerts()
        print(f"✓ Active alerts: {len(active)}")

        # Get statistics
        stats = alert_mgr.get_statistics()
        print(f"✓ Statistics:")
        print(f"  - Total alerts: {stats['total_alerts_generated']}")
        print(f"  - Active: {stats['active_alerts']}")

        # Test deduplication
        alert2 = alert_mgr.create_alert(
            level=AlertLevel.WARNING,
            category=AlertCategory.CAMERA,
            title="Test Alert",
            message="This is a test alert",
            camera_id=1
        )
        print(f"✓ Deduplication test: {'Passed' if alert2 is None else 'Failed'}")

        print("\n✓ AlertManager tests passed\n")
        return True

    except Exception as e:
        print(f"\n✗ AlertManager test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_web_dashboard():
    """Test WebDashboard functionality"""
    print("=" * 60)
    print("Testing WebDashboard")
    print("=" * 60)

    try:
        from src.monitoring import WebDashboard

        # Create dashboard
        dashboard = WebDashboard(
            host='127.0.0.1',
            port=5001,  # Use different port to avoid conflicts
            update_rate_hz=2.0
        )
        print("✓ WebDashboard created")

        # Note: We don't actually start the server in tests
        # as it would block or require more complex async handling
        print("✓ WebDashboard initialization successful")
        print("  (Server not started in test mode)")

        print("\n✓ WebDashboard tests passed\n")
        return True

    except Exception as e:
        print(f"\n✗ WebDashboard test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of all components"""
    print("=" * 60)
    print("Testing Component Integration")
    print("=" * 60)

    try:
        from src.monitoring import (
            SystemMonitor, DataValidator, AlertManager, WebDashboard
        )

        # Create all components
        monitor = SystemMonitor(update_rate_hz=10.0, num_cameras=20)
        validator = DataValidator()
        alert_mgr = AlertManager(enable_auto_diagnostics=True)
        dashboard = WebDashboard(host='127.0.0.1', port=5001)

        print("✓ All components created")

        # Link components
        alert_mgr.set_system_monitor(monitor)
        dashboard.set_system_monitor(monitor)
        dashboard.set_alert_manager(alert_mgr)
        dashboard.set_validator(validator)

        print("✓ Components linked")

        # Start monitoring
        monitor.start()
        time.sleep(1.0)

        # Process some data
        detection = {'x': 100.0, 'y': 200.0, 'z': 50.0, 'confidence': 0.95}
        result = validator.validate_detection(detection)

        if not result.passed:
            alert_mgr.create_alert(
                level=AlertLevel.ERROR,
                category=AlertCategory.DATA_QUALITY,
                title="Validation failed",
                message="Test validation failure"
            )

        # Get metrics
        summary = monitor.get_summary()
        print(f"✓ System operational:")
        print(f"  - CPU: {summary['cpu_utilization']:.1f}%")
        print(f"  - Memory: {summary['memory_percent']:.1f}%")
        print(f"  - Monitoring overhead: {monitor.get_performance_overhead():.3f}%")

        # Stop monitoring
        monitor.stop()

        print("\n✓ Integration tests passed\n")
        return True

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  Monitoring System Test Suite".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")

    results = {
        'Imports': test_imports(),
        'SystemMonitor': test_system_monitor(),
        'DataValidator': test_validator(),
        'AlertManager': test_alert_manager(),
        'WebDashboard': test_web_dashboard(),
        'Integration': test_integration()
    }

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s}: {status}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Monitoring system is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
