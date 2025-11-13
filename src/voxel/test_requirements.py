"""
test_requirements.py

Automated test suite to verify the voxel grid system meets all requirements:
1. Cover 5km x 5km x 2km volume minimum
2. <1m resolution at 5km distance
3. Support 200 tracked objects
4. <500MB memory footprint
5. Real-time updates at 30Hz
"""

import numpy as np
import time
import sys
from grid_manager import VoxelGridManager, GridConfig, LODLevel


class RequirementsTest:
    """Test suite for system requirements"""

    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0

    def run_test(self, name: str, test_func):
        """Run a single test"""
        print(f"\nTest: {name}")
        print("-" * 60)

        try:
            result = test_func()
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            self.results[name] = result

            if result['passed']:
                self.passed += 1
            else:
                self.failed += 1

            print(f"Status: {status}")
            if 'details' in result:
                for detail in result['details']:
                    print(f"  {detail}")

            return result['passed']

        except Exception as e:
            print(f"Status: ✗ ERROR - {e}")
            self.failed += 1
            self.results[name] = {'passed': False, 'error': str(e)}
            return False

    def test_volume_coverage(self):
        """Test 1: Verify 5km x 5km x 2km volume coverage"""
        print("Testing volume coverage...")

        config = GridConfig(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([5000.0, 5000.0, 2000.0]),
            base_resolution=1.0
        )
        manager = VoxelGridManager(config)

        # Test points at boundaries
        test_points = [
            (0, 0, 0, True),           # Center
            (2500, 2500, 1000, True),  # Corner
            (-2500, -2500, -1000, True), # Opposite corner
            (2499, 2499, 999, True),   # Just inside
            (2501, 2501, 1001, False), # Just outside
        ]

        all_correct = True
        details = []

        for x, y, z, should_be_inside in test_points:
            pos = np.array([float(x), float(y), float(z)])
            inside = np.all(np.abs(pos - manager.grid_center) <= manager.grid_size / 2)

            if inside == should_be_inside:
                details.append(f"  ✓ Point ({x}, {y}, {z}): {'inside' if inside else 'outside'} as expected")
            else:
                details.append(f"  ✗ Point ({x}, {y}, {z}): expected {'inside' if should_be_inside else 'outside'}, got {'inside' if inside else 'outside'}")
                all_correct = False

        # Check actual grid size
        details.append(f"Grid size: {manager.grid_size}")
        size_correct = np.all(manager.grid_size >= [5000.0, 5000.0, 2000.0])

        return {
            'passed': all_correct and size_correct,
            'details': details,
            'grid_size': manager.grid_size.tolist()
        }

    def test_resolution_at_5km(self):
        """Test 2: Verify <1m resolution at 5km distance"""
        print("Testing resolution at 5km...")

        config = GridConfig(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([5000.0, 5000.0, 2000.0]),
            base_resolution=1.0,
            enable_lod=True
        )
        manager = VoxelGridManager(config)

        # Test resolution at different distances
        test_distances = [100, 500, 1000, 2500, 5000]
        details = []

        max_resolution = 0.0

        for distance in test_distances:
            lod = manager._get_lod_for_distance(distance)
            resolution = lod.resolution

            details.append(f"  Distance {distance:4d}m: {resolution:.2f}m resolution (LOD: {lod.name})")

            if distance == 5000:
                max_resolution = resolution

        passed = max_resolution < 1.0

        return {
            'passed': passed,
            'details': details,
            'resolution_at_5km': max_resolution
        }

    def test_object_count(self):
        """Test 3: Verify support for 200 tracked objects"""
        print("Testing 200 tracked objects...")

        config = GridConfig(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([5000.0, 5000.0, 2000.0]),
            base_resolution=1.0
        )
        manager = VoxelGridManager(config)

        # Add 200 objects
        num_objects = 200
        start_time = time.time()

        for i in range(num_objects):
            angle = (i / num_objects) * 2 * np.pi
            distance = 1000 + i * 20
            position = np.array([
                np.cos(angle) * distance,
                np.sin(angle) * distance,
                (i % 20) * 50.0
            ])
            manager.add_tracked_object(i, position, size=2.0)

        add_time = time.time() - start_time

        # Verify all objects added
        stats = manager.get_statistics()
        actual_count = stats['tracked_objects']

        details = [
            f"  Target objects: {num_objects}",
            f"  Actual objects: {actual_count}",
            f"  Add time: {add_time*1000:.2f} ms",
            f"  Avg per object: {add_time*1000/num_objects:.3f} ms"
        ]

        return {
            'passed': actual_count == num_objects,
            'details': details,
            'object_count': actual_count
        }

    def test_memory_footprint(self):
        """Test 4: Verify <500MB memory footprint"""
        print("Testing memory footprint...")

        config = GridConfig(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([5000.0, 5000.0, 2000.0]),
            base_resolution=1.0,
            max_memory_mb=500,
            enable_lod=True
        )
        manager = VoxelGridManager(config)

        # Add 200 objects to populate grid
        for i in range(200):
            angle = (i / 200) * 2 * np.pi
            distance = 1000 + i * 20
            position = np.array([
                np.cos(angle) * distance,
                np.sin(angle) * distance,
                (i % 20) * 50.0
            ])
            manager.add_tracked_object(i, position, size=2.0)

        # Get memory usage
        stats = manager.get_statistics()
        memory_mb = stats['memory_usage_mb']

        details = [
            f"  Memory usage: {memory_mb:.2f} MB",
            f"  Memory limit: {config.max_memory_mb} MB",
            f"  Utilization: {memory_mb/config.max_memory_mb*100:.1f}%"
        ]

        # Check LOD levels
        for lod_level in LODLevel:
            key = f'active_voxels_{lod_level.name}'
            if key in stats:
                details.append(f"  {lod_level.name}: {stats[key]:,} active voxels")

        return {
            'passed': memory_mb < config.max_memory_mb,
            'details': details,
            'memory_mb': memory_mb
        }

    def test_update_rate(self):
        """Test 5: Verify real-time updates at 30Hz"""
        print("Testing 30Hz update rate...")

        config = GridConfig(
            center=np.array([0.0, 0.0, 0.0]),
            size=np.array([5000.0, 5000.0, 2000.0]),
            base_resolution=1.0,
            enable_lod=True
        )
        manager = VoxelGridManager(config)

        # Add 200 objects
        for i in range(200):
            angle = (i / 200) * 2 * np.pi
            distance = 1000 + i * 20
            position = np.array([
                np.cos(angle) * distance,
                np.sin(angle) * distance,
                (i % 20) * 50.0
            ])
            manager.add_tracked_object(i, position, size=2.0)

        # Benchmark update performance
        num_frames = 30
        frame_times = []

        print(f"  Running {num_frames} frame updates...")

        for frame in range(num_frames):
            frame_start = time.time()

            # Update all objects
            for i in range(200):
                obj = manager.tracked_objects[i]
                new_pos = obj.position + np.array([1.0, 0.0, 0.0])
                manager.update_object_position(i, new_pos)

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

        avg_frame_time = np.mean(frame_times)
        max_frame_time = np.max(frame_times)
        avg_fps = 1.0 / avg_frame_time

        details = [
            f"  Frames tested: {num_frames}",
            f"  Avg frame time: {avg_frame_time*1000:.2f} ms",
            f"  Max frame time: {max_frame_time*1000:.2f} ms",
            f"  Average FPS: {avg_fps:.1f}",
            f"  Target FPS: 30.0"
        ]

        return {
            'passed': avg_fps >= 30.0,
            'details': details,
            'avg_fps': avg_fps,
            'avg_frame_time_ms': avg_frame_time * 1000
        }

    def run_all_tests(self):
        """Run all requirement tests"""
        print("="*60)
        print("Voxel Grid System - Requirements Test Suite")
        print("="*60)

        # Run all tests
        self.run_test("1. Volume Coverage (5km x 5km x 2km)", self.test_volume_coverage)
        self.run_test("2. Resolution at 5km (<1m)", self.test_resolution_at_5km)
        self.run_test("3. Support 200 Tracked Objects", self.test_object_count)
        self.run_test("4. Memory Footprint (<500MB)", self.test_memory_footprint)
        self.run_test("5. Update Rate (30Hz)", self.test_update_rate)

        # Print summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Passed: {self.passed}/{self.passed + self.failed}")
        print(f"Failed: {self.failed}/{self.passed + self.failed}")

        if self.failed == 0:
            print("\n✓ All requirements met!")
        else:
            print(f"\n✗ {self.failed} requirement(s) not met")

        print("="*60)

        # Print detailed results
        print("\nDetailed Results:")
        print("-"*60)

        for test_name, result in self.results.items():
            status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
            print(f"\n{test_name}: {status}")

            if 'error' in result:
                print(f"  Error: {result['error']}")

        return self.failed == 0


def main():
    """Main entry point"""
    tester = RequirementsTest()
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
