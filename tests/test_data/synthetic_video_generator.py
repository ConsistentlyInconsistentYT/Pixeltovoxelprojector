"""
Synthetic 8K Video Generator
Generates realistic 8K video frames with simulated drone targets for testing
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DroneTarget:
    """Simulated drone target"""
    target_id: int
    position: Tuple[float, float, float]  # x, y, z (meters)
    velocity: Tuple[float, float, float]  # vx, vy, vz (m/s)
    size: float  # meters
    temperature: float  # Kelvin
    trajectory_type: str  # "linear", "circular", "evasive"


class SyntheticVideoGenerator:
    """Generate synthetic 8K video with simulated drone targets"""

    def __init__(
        self,
        width: int = 7680,
        height: int = 4320,
        fov_horizontal_deg: float = 50.0,
        camera_range_m: float = 5000.0,
        frame_rate: float = 30.0
    ):
        """
        Initialize synthetic video generator

        Args:
            width: Frame width in pixels (8K = 7680)
            height: Frame height in pixels (8K = 4320)
            fov_horizontal_deg: Horizontal field of view in degrees
            camera_range_m: Maximum detection range in meters
            frame_rate: Video frame rate
        """
        self.width = width
        self.height = height
        self.fov_horizontal = np.deg2rad(fov_horizontal_deg)
        self.fov_vertical = np.deg2rad(fov_horizontal_deg * height / width)
        self.camera_range = camera_range_m
        self.frame_rate = frame_rate

        # Background texture
        self.background = None

        logger.info(f"Synthetic video generator initialized: {width}x{height}, {fov_horizontal_deg}° FOV")

    def generate_background(self, sky_type: str = "clear") -> np.ndarray:
        """Generate background sky texture"""
        if sky_type == "clear":
            # Clear blue sky with gradient
            background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(self.height):
                intensity = int(200 - 50 * (y / self.height))
                background[y, :] = [intensity, intensity + 20, 255]

        elif sky_type == "cloudy":
            # Cloudy sky with noise
            background = np.ones((self.height, self.width, 3), dtype=np.uint8) * 180

            # Add cloud texture
            cloud_noise = np.random.randint(-30, 30, (self.height // 10, self.width // 10))
            cloud_noise = cv2.resize(cloud_noise, (self.width, self.height))

            for c in range(3):
                background[:, :, c] = np.clip(background[:, :, c] + cloud_noise, 0, 255)

        elif sky_type == "night":
            # Night sky
            background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            background[:, :] = [10, 10, 20]

            # Add stars
            num_stars = 1000
            for _ in range(num_stars):
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                brightness = np.random.randint(100, 255)
                background[y, x] = [brightness, brightness, brightness]

        else:
            background = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.background = background
        return background

    def project_3d_to_2d(
        self, position_3d: Tuple[float, float, float]
    ) -> Tuple[Optional[float], Optional[float], float]:
        """
        Project 3D world position to 2D pixel coordinates

        Args:
            position_3d: (x, y, z) in meters from camera

        Returns:
            (pixel_x, pixel_y, distance) or (None, None, distance) if out of view
        """
        x, y, z = position_3d

        # Calculate distance
        distance = np.sqrt(x**2 + y**2 + z**2)

        if distance > self.camera_range or z <= 0:
            return None, None, distance

        # Project to camera plane
        # Assuming camera looks along +z axis
        horizontal_angle = np.arctan2(x, z)
        vertical_angle = np.arctan2(y, z)

        # Check if within FOV
        if abs(horizontal_angle) > self.fov_horizontal / 2:
            return None, None, distance
        if abs(vertical_angle) > self.fov_vertical / 2:
            return None, None, distance

        # Convert to pixel coordinates
        pixel_x = (horizontal_angle / self.fov_horizontal + 0.5) * self.width
        pixel_y = (0.5 - vertical_angle / self.fov_vertical) * self.height

        return pixel_x, pixel_y, distance

    def calculate_pixel_size(self, object_size_m: float, distance_m: float) -> float:
        """Calculate pixel size of object at given distance"""
        if distance_m <= 0:
            return 0

        angular_size = np.arctan(object_size_m / distance_m)
        pixel_size = (angular_size / self.fov_horizontal) * self.width

        return max(1.0, pixel_size)

    def render_drone(
        self,
        frame: np.ndarray,
        drone: DroneTarget,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Render a drone target on the frame

        Returns:
            Updated frame and detection metadata (if visible)
        """
        # Project to 2D
        pixel_x, pixel_y, distance = self.project_3d_to_2d(drone.position)

        if pixel_x is None or pixel_y is None:
            return frame, None

        # Calculate pixel size
        pixel_size = self.calculate_pixel_size(drone.size, distance)

        if pixel_size < 0.5:
            return frame, None

        # Draw drone
        center_x = int(pixel_x)
        center_y = int(pixel_y)
        radius = max(1, int(pixel_size / 2))

        # Calculate brightness based on distance
        brightness = int(255 * (1.0 - distance / self.camera_range))
        brightness = max(30, min(255, brightness))

        # Draw drone as a bright spot
        cv2.circle(frame, (center_x, center_y), radius, (brightness, brightness, brightness), -1)

        # Add motion blur if moving fast
        velocity_magnitude = np.sqrt(sum(v**2 for v in drone.velocity))
        if velocity_magnitude > 10:
            # Calculate motion direction in image
            vx, vy, vz = drone.velocity
            motion_angle = np.arctan2(vy, vx)
            blur_length = min(10, int(velocity_magnitude / 2))

            end_x = int(center_x + blur_length * np.cos(motion_angle))
            end_y = int(center_y + blur_length * np.sin(motion_angle))

            cv2.line(frame, (center_x, center_y), (end_x, end_y),
                    (brightness // 2, brightness // 2, brightness // 2), max(1, radius // 2))

        # Add noise
        if add_noise:
            noise_region = max(3, radius * 2)
            x1 = max(0, center_x - noise_region)
            x2 = min(self.width, center_x + noise_region)
            y1 = max(0, center_y - noise_region)
            y2 = min(self.height, center_y + noise_region)

            noise = np.random.randint(-10, 10, (y2 - y1, x2 - x1, 3), dtype=np.int16)
            frame[y1:y2, x1:x2] = np.clip(frame[y1:y2, x1:x2].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Generate detection metadata
        metadata = {
            'target_id': drone.target_id,
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'pixel_size': pixel_size,
            'distance_m': distance,
            'brightness': brightness / 255.0,
            'velocity': drone.velocity,
            'position_3d': drone.position
        }

        return frame, metadata

    def render_thermal_signature(
        self,
        frame: np.ndarray,
        drone: DroneTarget
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Render thermal signature of drone

        Returns:
            Thermal frame and metadata
        """
        # Project to 2D
        pixel_x, pixel_y, distance = self.project_3d_to_2d(drone.position)

        if pixel_x is None or pixel_y is None:
            return frame, None

        # Calculate pixel size
        pixel_size = self.calculate_pixel_size(drone.size * 1.5, distance)  # Thermal bloom

        if pixel_size < 0.5:
            return frame, None

        # Draw thermal signature
        center_x = int(pixel_x)
        center_y = int(pixel_y)
        radius = max(1, int(pixel_size / 2))

        # Temperature to intensity (assuming 300-320K range)
        temp_normalized = (drone.temperature - 300) / 20.0
        intensity = int(255 * temp_normalized * (1.0 - distance / self.camera_range))
        intensity = max(20, min(255, intensity))

        # Draw thermal hotspot with gradient
        for r in range(radius, 0, -1):
            alpha = (r / radius)
            color = int(intensity * alpha)
            cv2.circle(frame, (center_x, center_y), r, (color, color, color), 1)

        metadata = {
            'target_id': drone.target_id,
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'pixel_size': pixel_size,
            'distance_m': distance,
            'temperature': drone.temperature,
            'thermal_intensity': intensity / 255.0
        }

        return frame, metadata

    def generate_frame(
        self,
        drones: List[DroneTarget],
        frame_type: str = "monochrome",
        add_noise: bool = True,
        noise_level: float = 0.02
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate a complete frame with all drones

        Args:
            drones: List of drone targets to render
            frame_type: "monochrome" or "thermal"
            add_noise: Whether to add sensor noise
            noise_level: Noise standard deviation (0.0 - 1.0)

        Returns:
            Frame image and list of detection metadata
        """
        # Start with background
        if self.background is None:
            self.generate_background("clear")

        frame = self.background.copy()

        # Render all drones
        detections = []

        for drone in drones:
            if frame_type == "monochrome":
                frame, metadata = self.render_drone(frame, drone, add_noise=False)
            else:  # thermal
                frame, metadata = self.render_thermal_signature(frame, drone)

            if metadata:
                detections.append(metadata)

        # Add global noise
        if add_noise:
            noise = np.random.normal(0, noise_level * 255, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Convert to grayscale for monochrome
        if frame_type == "monochrome":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame, detections

    def save_frame(self, frame: np.ndarray, filename: str, quality: int = 95):
        """Save frame to file"""
        # Use JPEG compression for 8K frames
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        logger.info(f"Saved frame to {filename}")

    def generate_video_sequence(
        self,
        drones: List[DroneTarget],
        num_frames: int,
        output_dir: str,
        update_positions: bool = True
    ) -> List[List[Dict]]:
        """
        Generate a sequence of video frames

        Args:
            drones: Initial drone targets
            num_frames: Number of frames to generate
            output_dir: Directory to save frames
            update_positions: Whether to update drone positions based on velocity

        Returns:
            List of detection metadata for each frame
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        all_detections = []
        dt = 1.0 / self.frame_rate

        for frame_num in range(num_frames):
            # Generate monochrome frame
            mono_frame, mono_detections = self.generate_frame(drones, "monochrome")
            mono_path = os.path.join(output_dir, f"mono_frame_{frame_num:06d}.jpg")
            self.save_frame(mono_frame, mono_path)

            # Generate thermal frame
            thermal_frame, thermal_detections = self.generate_frame(drones, "thermal")
            thermal_path = os.path.join(output_dir, f"thermal_frame_{frame_num:06d}.jpg")
            self.save_frame(thermal_frame, thermal_path)

            all_detections.append({
                'frame_num': frame_num,
                'mono_detections': mono_detections,
                'thermal_detections': thermal_detections
            })

            # Update drone positions
            if update_positions:
                for drone in drones:
                    x, y, z = drone.position
                    vx, vy, vz = drone.velocity

                    drone.position = (
                        x + vx * dt,
                        y + vy * dt,
                        z + vz * dt
                    )

                    # Keep drones in valid range
                    if drone.position[2] < 100 or drone.position[2] > self.camera_range:
                        drone.velocity = (vx, vy, -vz)

            if (frame_num + 1) % 10 == 0:
                logger.info(f"Generated {frame_num + 1}/{num_frames} frames")

        logger.info(f"Video sequence generation complete: {num_frames} frames")
        return all_detections


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create generator
    generator = SyntheticVideoGenerator(
        width=1920,  # Use lower resolution for testing
        height=1080,
        fov_horizontal_deg=50.0,
        camera_range_m=5000.0
    )

    # Create test drones
    drones = [
        DroneTarget(
            target_id=0,
            position=(100, 50, 2000),
            velocity=(5, 0, 0),
            size=0.2,
            temperature=310.0,
            trajectory_type="linear"
        ),
        DroneTarget(
            target_id=1,
            position=(-200, 100, 3000),
            velocity=(-3, 2, 5),
            size=0.25,
            temperature=315.0,
            trajectory_type="circular"
        )
    ]

    # Generate test frame
    generator.generate_background("clear")
    mono_frame, mono_dets = generator.generate_frame(drones, "monochrome")
    thermal_frame, thermal_dets = generator.generate_frame(drones, "thermal")

    print(f"Generated test frames:")
    print(f"  Monochrome detections: {len(mono_dets)}")
    print(f"  Thermal detections: {len(thermal_dets)}")
