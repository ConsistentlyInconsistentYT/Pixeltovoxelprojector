#!/usr/bin/env python3
"""
Camera Calibration Tool - Interactive Calibration Interface
===========================================================

This example provides an interactive tool for camera calibration:
- Intrinsic calibration for individual cameras
- Stereo calibration for camera pairs
- Mono-thermal registration
- Checkerboard/pattern detection
- Calibration validation and quality assessment
- Parameter export to JSON

Perfect for setting up new camera systems or recalibrating existing ones.

Requirements:
- Python 3.8+
- NumPy
- OpenCV (cv2) for real calibration (optional for demo)

Usage:
    python calibration_tool.py

    Optional arguments:
        --pair-id ID            Camera pair to calibrate (default: 0)
        --target TYPE           Calibration target: checkerboard, circles (default: checkerboard)
        --num-images NUM        Number of calibration images (default: 20)
        --validate              Run validation after calibration
        --export-dir DIR        Export calibration files to directory
        --interactive           Interactive mode with prompts

Author: Motion Tracking System
Date: 2025-11-13
"""

import sys
import time
import argparse
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import calibration components
from camera.pair_calibration import (
    PairCalibrationManager, CalibrationTarget, CalibrationStatus,
    IntrinsicParameters, StereoCalibration, MonoThermalRegistration,
    CalibrationEngine
)
from camera.camera_manager import CameraManager, CameraConfiguration, CameraPair, CameraType, ConnectionType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalibrationTool:
    """
    Interactive camera calibration tool
    """

    def __init__(self, num_pairs: int = 10):
        """
        Initialize calibration tool

        Args:
            num_pairs: Number of camera pairs in system
        """
        self.num_pairs = num_pairs

        # Initialize calibration manager
        logger.info(f"Initializing calibration manager for {num_pairs} pairs...")
        self.calib_manager = PairCalibrationManager(num_pairs=num_pairs)

        # Initialize camera manager (for live capture)
        self.camera_manager = CameraManager(num_pairs=num_pairs)

        # Calibration state
        self.current_pair = 0
        self.calibration_images = []

        logger.info("Calibration tool initialized")

    def setup_calibration_target(self, target_type: str = "checkerboard",
                                rows: int = 9, cols: int = 6,
                                square_size: float = 0.025) -> CalibrationTarget:
        """
        Setup calibration target

        Args:
            target_type: Type of target (checkerboard, circles)
            rows: Number of rows in pattern
            cols: Number of columns in pattern
            square_size: Size of squares/circles in meters

        Returns:
            CalibrationTarget object
        """
        target = CalibrationTarget(
            target_type=target_type,
            rows=rows,
            cols=cols,
            square_size=square_size
        )

        self.calib_manager.target = target

        logger.info(f"Calibration target: {target_type} {cols}x{rows}, "
                   f"size={square_size*1000:.1f}mm")

        return target

    def simulate_calibration_images(self, num_images: int = 20) -> Tuple[List, List]:
        """
        Simulate calibration image capture

        In real implementation, this would:
        1. Capture images from cameras
        2. Detect calibration pattern
        3. Extract corner/circle positions

        Args:
            num_images: Number of images to capture

        Returns:
            Tuple of (mono_images, thermal_images) lists
        """
        logger.info(f"Simulating capture of {num_images} calibration images...")

        mono_images = []
        thermal_images = []

        for i in range(num_images):
            # Simulate image capture
            # In real system: grab frames from cameras
            mono_img = np.random.randint(0, 255, (4320, 7680), dtype=np.uint8)
            thermal_img = np.random.randint(0, 255, (512, 640), dtype=np.uint8)

            mono_images.append(mono_img)
            thermal_images.append(thermal_img)

            if (i + 1) % 5 == 0:
                logger.info(f"  Captured {i+1}/{num_images} images...")

        logger.info(f"Capture complete: {num_images} image pairs")

        return mono_images, thermal_images

    def detect_calibration_pattern(self, images: List[np.ndarray],
                                   target: CalibrationTarget) -> List[np.ndarray]:
        """
        Detect calibration pattern in images

        In real implementation using OpenCV:
        - cv2.findChessboardCorners() for checkerboard
        - cv2.findCirclesGrid() for circle grid

        Args:
            images: List of calibration images
            target: Calibration target specification

        Returns:
            List of detected point arrays
        """
        logger.info(f"Detecting {target.target_type} pattern in {len(images)} images...")

        detected_points = []

        for i, img in enumerate(images):
            # Simulate pattern detection
            # In real system: use cv2.findChessboardCorners()

            # Generate simulated detected points
            points = np.random.rand(target.rows * target.cols, 2)
            points[:, 0] *= img.shape[1]  # Scale to image width
            points[:, 1] *= img.shape[0]  # Scale to image height

            # Add small noise
            points += np.random.randn(*points.shape) * 0.5

            detected_points.append(points.astype(np.float32))

        success_rate = len(detected_points) / len(images) * 100
        logger.info(f"Pattern detection: {len(detected_points)}/{len(images)} "
                   f"successful ({success_rate:.1f}%)")

        return detected_points

    def calibrate_intrinsic(self, pair_id: int, camera_type: str,
                           num_images: int = 20) -> bool:
        """
        Calibrate intrinsic parameters for a camera

        Args:
            pair_id: Camera pair ID
            camera_type: "mono" or "thermal"
            num_images: Number of calibration images

        Returns:
            Success status
        """
        camera_id = pair_id * 2 + (1 if camera_type == "thermal" else 0)

        print("\n" + "="*70)
        print(f"INTRINSIC CALIBRATION - Pair {pair_id} {camera_type.upper()} Camera")
        print("="*70)

        # Simulate image capture
        print(f"\n1. Capturing {num_images} calibration images...")
        if camera_type == "mono":
            images = [np.random.randint(0, 255, (4320, 7680), dtype=np.uint8)
                     for _ in range(num_images)]
        else:
            images = [np.random.randint(0, 255, (512, 640), dtype=np.uint8)
                     for _ in range(num_images)]

        # Detect pattern
        print(f"2. Detecting calibration pattern...")
        detected_points = self.detect_calibration_pattern(
            images,
            self.calib_manager.target
        )

        if len(detected_points) < num_images * 0.8:
            logger.error(f"Insufficient detections ({len(detected_points)}/{num_images})")
            return False

        # Calibrate
        print(f"3. Computing camera calibration...")
        success = self.calib_manager.calibrate_camera_intrinsic(
            camera_id,
            images,
            detected_points
        )

        if success:
            intrinsic = self.calib_manager.get_intrinsics(camera_id)
            print(f"\n✓ Calibration successful!")
            print(f"\nIntrinsic Parameters:")
            print(f"  Focal Length:     fx={intrinsic.fx:.1f}, fy={intrinsic.fy:.1f}")
            print(f"  Principal Point:  cx={intrinsic.cx:.1f}, cy={intrinsic.cy:.1f}")
            print(f"  Distortion:       {intrinsic.distortion}")
            print(f"  Reprojection Error: {intrinsic.reprojection_error:.3f} pixels")
        else:
            print(f"\n✗ Calibration failed!")

        print("="*70 + "\n")

        return success

    def calibrate_stereo(self, pair_id: int, num_images: int = 20) -> bool:
        """
        Calibrate stereo geometry for camera pair

        Args:
            pair_id: Camera pair ID
            num_images: Number of calibration images

        Returns:
            Success status
        """
        print("\n" + "="*70)
        print(f"STEREO CALIBRATION - Pair {pair_id}")
        print("="*70)

        # Check intrinsics
        mono_id = pair_id * 2
        thermal_id = pair_id * 2 + 1

        if mono_id not in self.calib_manager.intrinsics:
            print("\n✗ Mono camera intrinsics not calibrated!")
            print("  Run intrinsic calibration first.")
            return False

        if thermal_id not in self.calib_manager.intrinsics:
            print("\n✗ Thermal camera intrinsics not calibrated!")
            print("  Run intrinsic calibration first.")
            return False

        # Simulate synchronized image capture
        print(f"\n1. Capturing {num_images} synchronized image pairs...")
        mono_images, thermal_images = self.simulate_calibration_images(num_images)

        # Detect patterns in both cameras
        print(f"2. Detecting patterns in both cameras...")
        mono_points = self.detect_calibration_pattern(
            mono_images,
            self.calib_manager.target
        )
        thermal_points = self.detect_calibration_pattern(
            thermal_images,
            self.calib_manager.target
        )

        # Must have matching detections
        min_detections = min(len(mono_points), len(thermal_points))
        if min_detections < num_images * 0.8:
            logger.error(f"Insufficient matching detections ({min_detections}/{num_images})")
            return False

        # Use only matching images
        mono_points = mono_points[:min_detections]
        thermal_points = thermal_points[:min_detections]

        # Calibrate stereo
        print(f"3. Computing stereo calibration...")
        success = self.calib_manager.calibrate_stereo_pair(
            pair_id,
            mono_images[:min_detections],
            thermal_images[:min_detections],
            mono_points,
            thermal_points
        )

        if success:
            stereo = self.calib_manager.get_stereo_calibration(pair_id)
            print(f"\n✓ Stereo calibration successful!")
            print(f"\nStereo Parameters:")
            print(f"  Baseline:         {stereo.baseline*1000:.1f} mm")
            print(f"  Rotation:         {np.degrees(np.arctan2(stereo.rotation[1,0], stereo.rotation[0,0])):.2f}°")
            print(f"  Translation:      {stereo.translation}")
            print(f"  Reprojection Error: {stereo.reprojection_error:.3f} pixels")
            print(f"  Epipolar Error:   {stereo.epipolar_error:.3f} pixels")
        else:
            print(f"\n✗ Stereo calibration failed!")

        print("="*70 + "\n")

        return success

    def register_mono_thermal(self, pair_id: int) -> bool:
        """
        Register mono and thermal cameras

        Args:
            pair_id: Camera pair ID

        Returns:
            Success status
        """
        print("\n" + "="*70)
        print(f"MONO-THERMAL REGISTRATION - Pair {pair_id}")
        print("="*70)

        # Capture synchronized images
        print(f"\n1. Capturing synchronized mono-thermal images...")
        mono_img = np.random.randint(0, 255, (4320, 7680), dtype=np.uint8)
        thermal_img = np.random.randint(0, 255, (512, 640), dtype=np.uint8)

        # Register
        print(f"2. Computing registration (feature matching + homography)...")
        success = self.calib_manager.register_mono_thermal_pair(
            pair_id,
            mono_img,
            thermal_img
        )

        if success:
            reg = self.calib_manager.get_registration(pair_id)
            print(f"\n✓ Registration successful!")
            print(f"\nRegistration Parameters:")
            print(f"  Translation:      {reg.translation}")
            print(f"  Rotation:         {np.degrees(np.arctan2(reg.rotation[1,0], reg.rotation[0,0])):.2f}°")
            print(f"  Registration Error: {reg.registration_error:.3f} pixels")
            print(f"  Mutual Info:      {reg.mutual_information:.3f}")
            print(f"  Feature Matches:  {reg.feature_matches}")
        else:
            print(f"\n✗ Registration failed!")

        print("="*70 + "\n")

        return success

    def validate_calibration(self, pair_id: int) -> Dict:
        """
        Validate calibration quality

        Args:
            pair_id: Camera pair ID

        Returns:
            Validation results dictionary
        """
        print("\n" + "="*70)
        print(f"CALIBRATION VALIDATION - Pair {pair_id}")
        print("="*70)

        is_valid, reason = self.calib_manager.check_calibration_validity(pair_id)

        results = {
            'pair_id': pair_id,
            'is_valid': is_valid,
            'reason': reason,
            'checks': {}
        }

        # Check intrinsics
        mono_id = pair_id * 2
        thermal_id = pair_id * 2 + 1

        print("\n1. Intrinsic Calibration:")

        if mono_id in self.calib_manager.intrinsics:
            mono_intrinsic = self.calib_manager.intrinsics[mono_id]
            mono_ok = mono_intrinsic.reprojection_error < 1.0
            results['checks']['mono_intrinsic'] = mono_ok
            print(f"   Mono:    {'✓' if mono_ok else '✗'} "
                  f"(error={mono_intrinsic.reprojection_error:.3f}px)")
        else:
            results['checks']['mono_intrinsic'] = False
            print(f"   Mono:    ✗ Not calibrated")

        if thermal_id in self.calib_manager.intrinsics:
            thermal_intrinsic = self.calib_manager.intrinsics[thermal_id]
            thermal_ok = thermal_intrinsic.reprojection_error < 1.0
            results['checks']['thermal_intrinsic'] = thermal_ok
            print(f"   Thermal: {'✓' if thermal_ok else '✗'} "
                  f"(error={thermal_intrinsic.reprojection_error:.3f}px)")
        else:
            results['checks']['thermal_intrinsic'] = False
            print(f"   Thermal: ✗ Not calibrated")

        # Check stereo calibration
        print("\n2. Stereo Calibration:")

        if pair_id in self.calib_manager.stereo_calibrations:
            stereo = self.calib_manager.stereo_calibrations[pair_id]
            stereo_ok = (stereo.reprojection_error < 1.0 and
                        stereo.epipolar_error < 1.0)
            results['checks']['stereo'] = stereo_ok
            print(f"   Status:  {'✓' if stereo_ok else '✗'}")
            print(f"   Reproj:  {stereo.reprojection_error:.3f}px")
            print(f"   Epipolar: {stereo.epipolar_error:.3f}px")
        else:
            results['checks']['stereo'] = False
            print(f"   Status:  ✗ Not calibrated")

        # Check registration
        print("\n3. Mono-Thermal Registration:")

        if pair_id in self.calib_manager.registrations:
            reg = self.calib_manager.registrations[pair_id]
            reg_ok = (reg.registration_error < 3.0 and
                     reg.mutual_information > 0.5)
            results['checks']['registration'] = reg_ok
            print(f"   Status:  {'✓' if reg_ok else '✗'}")
            print(f"   Error:   {reg.registration_error:.3f}px")
            print(f"   MI:      {reg.mutual_information:.3f}")
        else:
            results['checks']['registration'] = False
            print(f"   Status:  ✗ Not calibrated")

        # Overall result
        print("\n" + "-"*70)
        print(f"Overall: {'✓ VALID' if is_valid else '✗ INVALID'}")
        if not is_valid:
            print(f"Reason:  {reason}")
        print("="*70 + "\n")

        return results

    def export_calibration(self, pair_id: int, export_dir: str = "calibration"):
        """
        Export calibration parameters to files

        Args:
            pair_id: Camera pair ID
            export_dir: Export directory path
        """
        print(f"\nExporting calibration for pair {pair_id} to {export_dir}/")

        # Save calibration
        self.calib_manager.save_calibration(pair_id, directory=export_dir)

        print(f"✓ Calibration exported successfully")

    def calibrate_full_pair(self, pair_id: int, num_images: int = 20) -> bool:
        """
        Complete calibration workflow for a camera pair

        Args:
            pair_id: Camera pair ID
            num_images: Number of calibration images

        Returns:
            Success status
        """
        print("\n" + "="*70)
        print(f"COMPLETE CALIBRATION WORKFLOW - Pair {pair_id}")
        print("="*70)
        print(f"\nThis will perform:")
        print(f"  1. Mono camera intrinsic calibration")
        print(f"  2. Thermal camera intrinsic calibration")
        print(f"  3. Stereo pair calibration")
        print(f"  4. Mono-thermal registration")
        print(f"  5. Validation")
        print("="*70 + "\n")

        # Step 1: Mono intrinsic
        if not self.calibrate_intrinsic(pair_id, "mono", num_images):
            return False

        # Step 2: Thermal intrinsic
        if not self.calibrate_intrinsic(pair_id, "thermal", num_images):
            return False

        # Step 3: Stereo calibration
        if not self.calibrate_stereo(pair_id, num_images):
            return False

        # Step 4: Mono-thermal registration
        if not self.register_mono_thermal(pair_id):
            return False

        # Step 5: Validation
        results = self.validate_calibration(pair_id)

        return results['is_valid']

    def print_calibration_report(self):
        """Print comprehensive calibration report for all pairs"""
        report = self.calib_manager.get_calibration_report()

        print("\n" + "="*70)
        print("CALIBRATION SYSTEM REPORT")
        print("="*70)
        print(f"Total Pairs:      {report['num_pairs']}")
        print(f"Timestamp:        {time.ctime(report['timestamp'])}")

        summary = report['summary']
        print(f"\nSummary:")
        print(f"  Calibrated:     {summary['calibrated_pairs']}")
        print(f"  Needs Update:   {summary['needs_update']}")
        print(f"  Not Calibrated: {summary['not_calibrated']}")

        if summary['avg_stereo_error'] > 0:
            print(f"  Avg Stereo Error: {summary['avg_stereo_error']:.3f}px")
        if summary['avg_registration_error'] > 0:
            print(f"  Avg Reg Error:    {summary['avg_registration_error']:.3f}px")

        print("\nPer-Pair Status:")
        for pair_id, pair_info in report['pairs'].items():
            status_symbol = "✓" if pair_info['is_valid'] else "✗"
            print(f"\n  Pair {pair_id}: {status_symbol} {pair_info['status']}")

            if pair_info['stereo_available']:
                quality = pair_info['stereo_quality']
                print(f"    Stereo: error={quality['reprojection_error']:.3f}px, "
                      f"age={quality['age_days']:.1f}d")

            if pair_info['registration_available']:
                quality = pair_info['registration_quality']
                print(f"    Registration: error={quality['registration_error']:.3f}px, "
                      f"MI={quality['mutual_information']:.3f}")

        print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Interactive camera calibration tool'
    )
    parser.add_argument(
        '--pair-id',
        type=int,
        default=0,
        help='Camera pair ID to calibrate (default: 0)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='checkerboard',
        choices=['checkerboard', 'circles'],
        help='Calibration target type (default: checkerboard)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=20,
        help='Number of calibration images (default: 20)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after calibration'
    )
    parser.add_argument(
        '--export-dir',
        type=str,
        default='calibration',
        help='Export directory for calibration files'
    )
    parser.add_argument(
        '--all-pairs',
        action='store_true',
        help='Calibrate all 10 pairs'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Show calibration report'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*70)
    print("CAMERA CALIBRATION TOOL")
    print("="*70)
    print(f"Target Type:    {args.target}")
    print(f"Num Images:     {args.num_images}")
    if not args.all_pairs:
        print(f"Pair ID:        {args.pair_id}")
    print("="*70 + "\n")

    # Create calibration tool
    tool = CalibrationTool(num_pairs=10)

    # Setup calibration target
    if args.target == "checkerboard":
        tool.setup_calibration_target("checkerboard", rows=9, cols=6, square_size=0.025)
    else:
        tool.setup_calibration_target("circles", rows=7, cols=7, square_size=0.025)

    if args.report:
        # Just show report
        tool.print_calibration_report()
        return

    if args.all_pairs:
        # Calibrate all pairs
        print("Calibrating all 10 camera pairs...")
        print("This will take some time...\n")

        for pair_id in range(10):
            success = tool.calibrate_full_pair(pair_id, args.num_images)

            if success:
                # Export calibration
                tool.export_calibration(pair_id, args.export_dir)
            else:
                logger.warning(f"Pair {pair_id} calibration failed")

        # Print final report
        tool.print_calibration_report()

    else:
        # Calibrate single pair
        success = tool.calibrate_full_pair(args.pair_id, args.num_images)

        if success and args.validate:
            tool.validate_calibration(args.pair_id)

        if success:
            # Export calibration
            tool.export_calibration(args.pair_id, args.export_dir)

    logger.info("Calibration tool finished!")


if __name__ == "__main__":
    main()
