"""
Test suite for Multi-Camera Synchronization System

Tests core functionality of synchronization, management, and calibration modules.
"""

import time
import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from camera_sync import (
    CameraSynchronizer, SyncMode, FrameMetadata,
    PTPManager, HardwareTriggerController
)
from camera_manager import (
    CameraManager, CameraConfiguration, CameraType,
    ConnectionType, CameraPair
)
from pair_calibration import (
    PairCalibrationManager, IntrinsicParameters,
    CalibrationEngine, CalibrationTarget
)


class TestCameraSynchronizer(unittest.TestCase):
    """Test camera synchronization functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.num_pairs = 2
        self.sync = CameraSynchronizer(num_pairs=self.num_pairs, sync_mode=SyncMode.HYBRID)

    def tearDown(self):
        """Clean up after tests"""
        if self.sync.running:
            self.sync.stop()

    def test_initialization(self):
        """Test synchronizer initialization"""
        self.assertEqual(self.sync.num_pairs, 2)
        self.assertEqual(self.sync.num_cameras, 4)
        self.assertEqual(len(self.sync.frame_buffers), 4)
        self.assertEqual(len(self.sync.sync_stats), 4)

    def test_start_stop(self):
        """Test starting and stopping synchronizer"""
        self.sync.start()
        self.assertTrue(self.sync.running)
        time.sleep(0.5)  # Let it run briefly
        self.sync.stop()
        self.assertFalse(self.sync.running)

    def test_add_frame(self):
        """Test adding frames to synchronizer"""
        metadata = FrameMetadata(
            camera_id=0,
            pair_id=0,
            frame_number=1,
            timestamp=time.time(),
            system_time=time.time(),
            trigger_id=1
        )
        self.sync.add_frame(0, metadata)

        # Check frame was added to buffer
        frame = self.sync.frame_buffers[0].get_latest_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.frame_number, 1)

    def test_frame_synchronization(self):
        """Test frame synchronization between camera pairs"""
        self.sync.start()
        time.sleep(0.1)

        # Add synchronized frames for a pair
        timestamp = time.time()
        for camera_id in [0, 1]:
            metadata = FrameMetadata(
                camera_id=camera_id,
                pair_id=0,
                frame_number=1,
                timestamp=timestamp + 0.0001,  # 0.1ms offset
                system_time=timestamp,
                trigger_id=1
            )
            self.sync.add_frame(camera_id, metadata)

        # Wait for synchronization
        time.sleep(0.2)

        # Get synchronized frame
        synced_set = self.sync.get_synced_frame_set(timeout=1.0)
        self.assertIsNotNone(synced_set)
        self.assertEqual(synced_set.pair_id, 0)
        self.assertLess(synced_set.sync_error, 1.0)  # < 1ms

        self.sync.stop()

    def test_sync_tolerance(self):
        """Test synchronization tolerance setting"""
        self.sync.set_sync_tolerance(0.5)  # 0.5ms
        self.assertAlmostEqual(self.sync.sync_tolerance, 0.0005, places=6)

    def test_metrics(self):
        """Test synchronization metrics retrieval"""
        metrics = self.sync.get_synchronization_metrics()
        self.assertIn('sync_mode', metrics)
        self.assertIn('num_pairs', metrics)
        self.assertIn('overall', metrics)
        self.assertIn('camera_stats', metrics)


class TestPTPManager(unittest.TestCase):
    """Test PTP synchronization manager"""

    def setUp(self):
        """Set up test fixtures"""
        self.ptp = PTPManager()

    def test_initialization(self):
        """Test PTP manager initialization"""
        self.assertEqual(self.ptp.master_offset, 0.0)
        self.assertFalse(self.ptp.is_synchronized)

    def test_offset_update(self):
        """Test updating PTP offset"""
        local_time = time.time()
        master_time = local_time + 0.001  # 1ms offset

        self.ptp.update_master_offset(master_time, local_time)
        self.assertGreater(len(self.ptp.offset_history), 0)

    def test_ptp_time(self):
        """Test getting PTP-synchronized time"""
        ptp_time = self.ptp.get_ptp_time()
        self.assertIsInstance(ptp_time, float)
        self.assertGreater(ptp_time, 0)

    def test_sync_convergence(self):
        """Test PTP synchronization convergence"""
        # Simulate multiple sync updates
        for i in range(20):
            local_time = time.time()
            master_time = local_time + 0.001  # Constant 1ms offset
            self.ptp.update_master_offset(master_time, local_time)

        # After 20 updates, should be synchronized
        quality = self.ptp.get_sync_quality()
        self.assertIn('is_synced', quality)
        self.assertIn('offset', quality)
        self.assertIn('jitter', quality)


class TestHardwareTrigger(unittest.TestCase):
    """Test hardware trigger controller"""

    def setUp(self):
        """Set up test fixtures"""
        self.trigger = HardwareTriggerController(num_cameras=4)

    def tearDown(self):
        """Clean up after tests"""
        if self.trigger.running:
            self.trigger.stop_hardware_trigger()

    def test_initialization(self):
        """Test trigger controller initialization"""
        self.assertEqual(self.trigger.num_cameras, 4)
        self.assertEqual(self.trigger.trigger_counter, 0)
        self.assertFalse(self.trigger.running)

    def test_start_stop(self):
        """Test starting and stopping trigger"""
        self.trigger.start_hardware_trigger(rate=30.0)
        self.assertTrue(self.trigger.running)
        time.sleep(0.2)
        self.trigger.stop_hardware_trigger()
        self.assertFalse(self.trigger.running)

    def test_trigger_generation(self):
        """Test trigger signal generation"""
        self.trigger.start_hardware_trigger(rate=30.0)
        time.sleep(0.5)  # Let it generate some triggers

        stats = self.trigger.get_trigger_stats()
        self.assertGreater(stats['total_triggers'], 0)

        self.trigger.stop_hardware_trigger()

    def test_trigger_response(self):
        """Test registering camera responses"""
        self.trigger.trigger_counter = 1
        self.trigger.trigger_map[1] = {
            'timestamp': time.time(),
            'expected_cameras': set(range(4)),
            'received_cameras': set()
        }

        # Register response
        result = self.trigger.register_trigger_response(0, 1)
        self.assertTrue(result)

        # Check response was recorded
        self.assertIn(0, self.trigger.trigger_map[1]['received_cameras'])


class TestCameraManager(unittest.TestCase):
    """Test camera management system"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = CameraManager(num_pairs=2)

    def tearDown(self):
        """Clean up after tests"""
        if self.manager.monitoring_active:
            self.manager.stop_health_monitoring()
        self.manager.disconnect_all()

    def test_initialization(self):
        """Test camera manager initialization"""
        self.assertEqual(self.manager.num_pairs, 2)
        self.assertEqual(self.manager.num_cameras, 4)

    def test_add_camera(self):
        """Test adding camera configuration"""
        config = CameraConfiguration(
            camera_id=0,
            pair_id=0,
            camera_type=CameraType.MONO,
            connection=ConnectionType.GIGE_VISION,
            ip_address="192.168.1.10"
        )
        self.manager.add_camera(config)

        self.assertIn(0, self.manager.cameras)
        self.assertIn(0, self.manager.configurations)
        self.assertIn(0, self.manager.health)

    def test_add_pair(self):
        """Test adding camera pair"""
        # Add cameras first
        for i in range(2):
            config = CameraConfiguration(
                camera_id=i,
                pair_id=0,
                camera_type=CameraType.MONO if i == 0 else CameraType.THERMAL,
                connection=ConnectionType.GIGE_VISION,
                ip_address=f"192.168.1.{10+i}"
            )
            self.manager.add_camera(config)

        # Add pair
        pair = CameraPair(pair_id=0, mono_camera_id=0, thermal_camera_id=1)
        result = self.manager.add_pair(pair)

        self.assertTrue(result)
        self.assertIn(0, self.manager.pairs)

    def test_health_monitoring(self):
        """Test health monitoring"""
        # Add a camera
        config = CameraConfiguration(
            camera_id=0,
            pair_id=0,
            camera_type=CameraType.MONO,
            connection=ConnectionType.GIGE_VISION
        )
        self.manager.add_camera(config)

        # Start monitoring
        self.manager.start_health_monitoring()
        self.assertTrue(self.manager.monitoring_active)

        time.sleep(0.5)

        # Stop monitoring
        self.manager.stop_health_monitoring()
        self.assertFalse(self.manager.monitoring_active)

    def test_system_health(self):
        """Test system health metrics"""
        health = self.manager.get_system_health()

        self.assertIn('total_cameras', health)
        self.assertIn('streaming', health)
        self.assertIn('avg_fps', health)
        self.assertEqual(health['total_cameras'], 4)

    def test_configuration_persistence(self):
        """Test saving and loading configuration"""
        # Add camera
        config = CameraConfiguration(
            camera_id=0,
            pair_id=0,
            camera_type=CameraType.MONO,
            connection=ConnectionType.GIGE_VISION,
            ip_address="192.168.1.10"
        )
        self.manager.add_camera(config)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            self.manager.save_configuration(temp_path)

            # Load in new manager
            new_manager = CameraManager(num_pairs=2)
            result = new_manager.load_configuration(temp_path)

            self.assertTrue(result)
            self.assertIn(0, new_manager.cameras)
        finally:
            Path(temp_path).unlink()


class TestCalibrationSystem(unittest.TestCase):
    """Test calibration system"""

    def setUp(self):
        """Set up test fixtures"""
        self.calib = PairCalibrationManager(num_pairs=2)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests"""
        if self.calib.update_enabled:
            self.calib.stop_realtime_updates()
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test calibration manager initialization"""
        self.assertEqual(self.calib.num_pairs, 2)
        self.assertEqual(self.calib.num_cameras, 4)
        self.assertEqual(len(self.calib.calibration_status), 2)

    def test_intrinsic_calibration(self):
        """Test intrinsic calibration"""
        # Simulate calibration images
        images = [np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
                 for _ in range(10)]
        points = [np.random.rand(54, 2) * [1920, 1080] for _ in range(10)]

        result = self.calib.calibrate_camera_intrinsic(0, images, points)

        self.assertTrue(result)
        self.assertIn(0, self.calib.intrinsics)

        intrinsic = self.calib.get_intrinsics(0)
        self.assertIsNotNone(intrinsic)
        self.assertGreater(intrinsic.fx, 0)
        self.assertGreater(intrinsic.fy, 0)

    def test_stereo_calibration(self):
        """Test stereo calibration"""
        # First calibrate intrinsics
        for camera_id in [0, 1]:
            images = [np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
                     for _ in range(10)]
            points = [np.random.rand(54, 2) * [1920, 1080] for _ in range(10)]
            self.calib.calibrate_camera_intrinsic(camera_id, images, points)

        # Stereo calibration
        images1 = [np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
                  for _ in range(10)]
        images2 = [np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
                  for _ in range(10)]
        points1 = [np.random.rand(54, 2) * [1920, 1080] for _ in range(10)]
        points2 = [np.random.rand(54, 2) * [1920, 1080] for _ in range(10)]

        result = self.calib.calibrate_stereo_pair(0, images1, images2, points1, points2)

        self.assertTrue(result)
        self.assertIn(0, self.calib.stereo_calibrations)

        stereo = self.calib.get_stereo_calibration(0)
        self.assertIsNotNone(stereo)
        self.assertGreater(stereo.baseline, 0)

    def test_mono_thermal_registration(self):
        """Test mono-thermal registration"""
        # Calibrate intrinsics first
        for camera_id in [0, 1]:
            images = [np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
                     for _ in range(10)]
            points = [np.random.rand(54, 2) * [1920, 1080] for _ in range(10)]
            self.calib.calibrate_camera_intrinsic(camera_id, images, points)

        # Registration
        mono_img = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
        thermal_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        result = self.calib.register_mono_thermal_pair(0, mono_img, thermal_img)

        self.assertTrue(result)
        self.assertIn(0, self.calib.registrations)

        reg = self.calib.get_registration(0)
        self.assertIsNotNone(reg)

    def test_calibration_validity(self):
        """Test calibration validity checking"""
        # Without calibration
        is_valid, reason = self.calib.check_calibration_validity(0)
        self.assertFalse(is_valid)

    def test_calibration_persistence(self):
        """Test saving and loading calibration"""
        # Create intrinsic calibration
        images = [np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
                 for _ in range(10)]
        points = [np.random.rand(54, 2) * [1920, 1080] for _ in range(10)]
        self.calib.calibrate_camera_intrinsic(0, images, points)

        # Save
        self.calib.save_calibration(0, directory=self.temp_dir)

        # Load in new manager
        new_calib = PairCalibrationManager(num_pairs=2)
        result = new_calib.load_calibration(0, directory=self.temp_dir)

        self.assertTrue(result)
        self.assertIn(0, new_calib.intrinsics)

    def test_calibration_report(self):
        """Test calibration report generation"""
        report = self.calib.get_calibration_report()

        self.assertIn('num_pairs', report)
        self.assertIn('pairs', report)
        self.assertIn('summary', report)
        self.assertEqual(report['num_pairs'], 2)


class TestCalibrationEngine(unittest.TestCase):
    """Test calibration computation engine"""

    def test_intrinsic_calibration(self):
        """Test intrinsic calibration computation"""
        image_points = [np.random.rand(54, 2) * [1920, 1080] for _ in range(10)]
        object_points = [np.zeros((54, 3), np.float32) for _ in range(10)]

        intrinsic = CalibrationEngine.calibrate_intrinsic(
            image_points, object_points, (1920, 1080)
        )

        self.assertGreater(intrinsic.fx, 0)
        self.assertGreater(intrinsic.fy, 0)
        self.assertGreater(intrinsic.reprojection_error, 0)

    def test_stereo_calibration(self):
        """Test stereo calibration computation"""
        intrinsic1 = IntrinsicParameters(camera_id=0, width=1920, height=1080)
        intrinsic1.fx = intrinsic1.fy = 2000
        intrinsic1.cx = 960
        intrinsic1.cy = 540

        intrinsic2 = IntrinsicParameters(camera_id=1, width=1920, height=1080)
        intrinsic2.fx = intrinsic2.fy = 2000
        intrinsic2.cx = 960
        intrinsic2.cy = 540

        image_points1 = [np.random.rand(54, 2) * [1920, 1080] for _ in range(10)]
        image_points2 = [np.random.rand(54, 2) * [1920, 1080] for _ in range(10)]
        object_points = [np.zeros((54, 3), np.float32) for _ in range(10)]

        stereo = CalibrationEngine.calibrate_stereo(
            intrinsic1, intrinsic2,
            image_points1, image_points2, object_points
        )

        self.assertGreater(stereo.baseline, 0)
        self.assertGreater(stereo.reprojection_error, 0)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCameraSynchronizer))
    suite.addTests(loader.loadTestsFromTestCase(TestPTPManager))
    suite.addTests(loader.loadTestsFromTestCase(TestHardwareTrigger))
    suite.addTests(loader.loadTestsFromTestCase(TestCameraManager))
    suite.addTests(loader.loadTestsFromTestCase(TestCalibrationSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestCalibrationEngine))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
