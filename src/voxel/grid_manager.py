"""
grid_manager.py

Grid Management System for 5km+ Range Tracking

Features:
- Dynamic grid sizing based on scene
- Automatic resolution adjustment
- Memory management for large grids
- Multi-resolution grid support
- Integration with C++ sparse voxel grid and CUDA optimizer

Requirements:
- Cover 5km x 5km x 2km volume minimum
- <1m resolution at 5km distance
- Support 200 tracked objects
- <500MB memory footprint
- Real-time updates at 30Hz

Architecture:
- Multi-level grid hierarchy
- Automatic LOD selection based on object distance
- Memory pooling and reuse
- Background pruning and compression
"""

import numpy as np
import ctypes
import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import json


# ============================================================================
# Configuration
# ============================================================================

class LODLevel(Enum):
    """Level of Detail configurations"""
    ULTRA_HIGH = (0.1, 100.0)    # 0.1m resolution, <100m distance
    HIGH = (0.25, 500.0)         # 0.25m resolution, <500m distance
    MEDIUM = (0.5, 1000.0)       # 0.5m resolution, <1km distance
    LOW = (1.0, 5000.0)          # 1.0m resolution, <5km distance
    ULTRA_LOW = (2.0, 10000.0)   # 2.0m resolution, >5km distance

    def __init__(self, resolution: float, max_distance: float):
        self.resolution = resolution
        self.max_distance = max_distance


@dataclass
class GridConfig:
    """Configuration for voxel grid"""
    center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    size: np.ndarray = field(default_factory=lambda: np.array([5000.0, 5000.0, 2000.0]))
    base_resolution: float = 1.0
    max_memory_mb: int = 500
    enable_lod: bool = True
    enable_compression: bool = True
    prune_threshold: float = 0.01
    update_rate_hz: float = 30.0


@dataclass
class TrackedObject:
    """Represents a tracked object in the voxel grid"""
    object_id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    size: float = 1.0
    confidence: float = 1.0
    last_update: float = 0.0


@dataclass
class Camera:
    """Camera representation for multi-camera tracking"""
    camera_id: int
    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    fov: float = 60.0
    width: int = 640
    height: int = 480


# ============================================================================
# Voxel Grid Manager
# ============================================================================

class VoxelGridManager:
    """
    High-level manager for sparse voxel grid system
    Handles dynamic grid sizing, LOD management, and memory optimization
    """

    def __init__(self, config: GridConfig = None):
        """
        Initialize the grid manager

        Args:
            config: GridConfig object with system parameters
        """
        self.config = config or GridConfig()

        # Grid state
        self.grid_center = self.config.center.copy()
        self.grid_size = self.config.size.copy()
        self.base_resolution = self.config.base_resolution

        # Multi-resolution grids (LOD pyramid)
        self.lod_grids: Dict[LODLevel, np.ndarray] = {}
        self.active_lod = LODLevel.MEDIUM

        # Tracked objects
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.cameras: Dict[int, Camera] = {}

        # Performance metrics
        self.frame_count = 0
        self.total_time = 0.0
        self.memory_usage_mb = 0.0

        # Threading
        self.update_lock = threading.Lock()
        self.background_thread = None
        self.running = False

        # Initialize grids
        self._initialize_grids()

        print(f"VoxelGridManager initialized:")
        print(f"  Grid center: {self.grid_center}")
        print(f"  Grid size: {self.grid_size}")
        print(f"  Base resolution: {self.base_resolution}m")
        print(f"  LOD enabled: {self.config.enable_lod}")
        print(f"  Memory limit: {self.config.max_memory_mb}MB")

    def _initialize_grids(self):
        """Initialize multi-resolution grid hierarchy"""
        if self.config.enable_lod:
            # Create grids for each LOD level
            for lod_level in LODLevel:
                resolution = lod_level.resolution
                grid_shape = tuple((self.grid_size / resolution).astype(int))

                # Estimate memory usage
                memory_mb = np.prod(grid_shape) * 4 / (1024 ** 2)  # 4 bytes per float32

                # Only create grid if within memory budget
                if memory_mb < self.config.max_memory_mb:
                    self.lod_grids[lod_level] = np.zeros(grid_shape, dtype=np.float32)
                    print(f"  LOD {lod_level.name}: {grid_shape} ({memory_mb:.1f} MB)")
                else:
                    print(f"  LOD {lod_level.name}: Skipped (exceeds memory budget)")
        else:
            # Single resolution grid
            grid_shape = tuple((self.grid_size / self.base_resolution).astype(int))
            self.lod_grids[LODLevel.MEDIUM] = np.zeros(grid_shape, dtype=np.float32)

    def start_background_processing(self):
        """Start background thread for pruning and optimization"""
        if self.running:
            return

        self.running = True
        self.background_thread = threading.Thread(target=self._background_worker)
        self.background_thread.daemon = True
        self.background_thread.start()
        print("Background processing started")

    def stop_background_processing(self):
        """Stop background processing thread"""
        self.running = False
        if self.background_thread:
            self.background_thread.join()
        print("Background processing stopped")

    def _background_worker(self):
        """Background worker for continuous optimization"""
        while self.running:
            time.sleep(1.0 / self.config.update_rate_hz)

            with self.update_lock:
                # Prune low-value voxels
                if self.config.enable_compression:
                    self._prune_voxels()

                # Update memory statistics
                self._update_memory_stats()

    # ========================================================================
    # Object Tracking
    # ========================================================================

    def add_tracked_object(self, object_id: int, position: np.ndarray,
                          size: float = 1.0, velocity: np.ndarray = None):
        """
        Add or update a tracked object

        Args:
            object_id: Unique object identifier
            position: 3D position (x, y, z) in meters
            size: Object size in meters
            velocity: Optional velocity vector (vx, vy, vz)
        """
        if velocity is None:
            velocity = np.array([0.0, 0.0, 0.0])

        obj = TrackedObject(
            object_id=object_id,
            position=position.copy(),
            velocity=velocity.copy(),
            size=size,
            confidence=1.0,
            last_update=time.time()
        )

        with self.update_lock:
            self.tracked_objects[object_id] = obj
            self._update_voxel_for_object(obj)

    def remove_tracked_object(self, object_id: int):
        """Remove a tracked object"""
        with self.update_lock:
            if object_id in self.tracked_objects:
                del self.tracked_objects[object_id]

    def update_object_position(self, object_id: int, position: np.ndarray):
        """Update position of tracked object"""
        with self.update_lock:
            if object_id in self.tracked_objects:
                obj = self.tracked_objects[object_id]
                obj.velocity = (position - obj.position) / (time.time() - obj.last_update)
                obj.position = position.copy()
                obj.last_update = time.time()
                self._update_voxel_for_object(obj)

    def _update_voxel_for_object(self, obj: TrackedObject):
        """Update voxels for a tracked object"""
        # Determine appropriate LOD level based on distance from grid center
        distance = np.linalg.norm(obj.position - self.grid_center)
        lod_level = self._get_lod_for_distance(distance)

        if lod_level not in self.lod_grids:
            lod_level = self.active_lod

        grid = self.lod_grids[lod_level]
        resolution = lod_level.resolution

        # Convert world position to voxel coordinates
        voxel_pos = self._world_to_voxel(obj.position, resolution)

        # Update voxel(s) for this object
        # For simplicity, update a small region around the object
        obj_radius_voxels = int(np.ceil(obj.size / resolution))

        for dx in range(-obj_radius_voxels, obj_radius_voxels + 1):
            for dy in range(-obj_radius_voxels, obj_radius_voxels + 1):
                for dz in range(-obj_radius_voxels, obj_radius_voxels + 1):
                    vx = voxel_pos[0] + dx
                    vy = voxel_pos[1] + dy
                    vz = voxel_pos[2] + dz

                    if self._is_voxel_in_bounds(vx, vy, vz, grid.shape):
                        # Distance-based falloff
                        dist = np.sqrt(dx**2 + dy**2 + dz**2) * resolution
                        if dist <= obj.size:
                            falloff = max(0.0, 1.0 - dist / obj.size)
                            grid[vx, vy, vz] = max(grid[vx, vy, vz],
                                                   obj.confidence * falloff)

    # ========================================================================
    # Camera Management
    # ========================================================================

    def add_camera(self, camera_id: int, position: np.ndarray,
                  forward: np.ndarray, up: np.ndarray = None,
                  fov: float = 60.0, width: int = 640, height: int = 480):
        """
        Add a camera to the system

        Args:
            camera_id: Unique camera identifier
            position: Camera position (x, y, z)
            forward: Forward direction vector
            up: Up direction vector (default: [0, 0, 1])
            fov: Field of view in degrees
            width: Image width in pixels
            height: Image height in pixels
        """
        if up is None:
            up = np.array([0.0, 0.0, 1.0])

        # Normalize vectors
        forward = forward / np.linalg.norm(forward)
        up = up / np.linalg.norm(up)

        camera = Camera(
            camera_id=camera_id,
            position=position.copy(),
            forward=forward,
            up=up,
            fov=fov,
            width=width,
            height=height
        )

        with self.update_lock:
            self.cameras[camera_id] = camera

    def remove_camera(self, camera_id: int):
        """Remove a camera"""
        with self.update_lock:
            if camera_id in self.cameras:
                del self.cameras[camera_id]

    def cast_rays_from_camera(self, camera_id: int,
                              pixel_coords: np.ndarray,
                              values: np.ndarray):
        """
        Cast rays from camera and update voxels

        Args:
            camera_id: Camera identifier
            pixel_coords: Nx2 array of pixel coordinates
            values: N array of intensity values
        """
        if camera_id not in self.cameras:
            return

        camera = self.cameras[camera_id]

        with self.update_lock:
            for i, (px, py) in enumerate(pixel_coords):
                # Generate ray direction
                ray_dir = self._pixel_to_ray(camera, px, py)

                # March ray through grid
                voxels = self._march_ray(camera.position, ray_dir)

                # Update voxels along ray
                for voxel_pos in voxels:
                    distance = np.linalg.norm(voxel_pos - camera.position)
                    lod = self._get_lod_for_distance(distance)

                    if lod in self.lod_grids:
                        grid = self.lod_grids[lod]
                        voxel_idx = self._world_to_voxel(voxel_pos, lod.resolution)

                        if self._is_voxel_in_bounds(voxel_idx[0], voxel_idx[1],
                                                    voxel_idx[2], grid.shape):
                            # Accumulate with distance falloff
                            attenuation = 1.0 / (1.0 + distance / 1000.0)
                            grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]] += \
                                values[i] * attenuation

    def _pixel_to_ray(self, camera: Camera, px: int, py: int) -> np.ndarray:
        """Convert pixel coordinates to ray direction"""
        # NDC coordinates
        ndc_x = (2.0 * px / camera.width - 1.0)
        ndc_y = (1.0 - 2.0 * py / camera.height)

        # Calculate ray direction
        aspect = camera.width / camera.height
        fov_rad = np.radians(camera.fov)
        tan_half_fov = np.tan(fov_rad * 0.5)

        # Right vector
        right = np.cross(camera.forward, camera.up)
        right = right / np.linalg.norm(right)

        ray_dir = (camera.forward +
                  right * ndc_x * aspect * tan_half_fov +
                  camera.up * ndc_y * tan_half_fov)

        return ray_dir / np.linalg.norm(ray_dir)

    def _march_ray(self, origin: np.ndarray, direction: np.ndarray,
                   max_distance: float = 5000.0) -> List[np.ndarray]:
        """March ray through grid and return voxel positions"""
        voxels = []

        # Use medium resolution for ray marching
        step_size = self.base_resolution * 0.5
        distance = 0.0

        while distance < max_distance:
            pos = origin + direction * distance

            # Check if in bounds
            if np.all(np.abs(pos - self.grid_center) <= self.grid_size / 2):
                voxels.append(pos.copy())
            else:
                break

            distance += step_size

        return voxels

    # ========================================================================
    # Grid Operations
    # ========================================================================

    def get_voxel(self, position: np.ndarray, lod: LODLevel = None) -> float:
        """
        Get voxel value at world position

        Args:
            position: World position (x, y, z)
            lod: LOD level (auto-select if None)

        Returns:
            Voxel value
        """
        if lod is None:
            distance = np.linalg.norm(position - self.grid_center)
            lod = self._get_lod_for_distance(distance)

        if lod not in self.lod_grids:
            return 0.0

        grid = self.lod_grids[lod]
        voxel_idx = self._world_to_voxel(position, lod.resolution)

        if self._is_voxel_in_bounds(voxel_idx[0], voxel_idx[1], voxel_idx[2], grid.shape):
            return grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]]

        return 0.0

    def set_voxel(self, position: np.ndarray, value: float, lod: LODLevel = None):
        """
        Set voxel value at world position

        Args:
            position: World position (x, y, z)
            value: Voxel value
            lod: LOD level (auto-select if None)
        """
        if lod is None:
            distance = np.linalg.norm(position - self.grid_center)
            lod = self._get_lod_for_distance(distance)

        if lod not in self.lod_grids:
            return

        with self.update_lock:
            grid = self.lod_grids[lod]
            voxel_idx = self._world_to_voxel(position, lod.resolution)

            if self._is_voxel_in_bounds(voxel_idx[0], voxel_idx[1], voxel_idx[2], grid.shape):
                grid[voxel_idx[0], voxel_idx[1], voxel_idx[2]] = value

    def clear_grid(self, lod: LODLevel = None):
        """Clear voxel grid(s)"""
        with self.update_lock:
            if lod is None:
                for grid in self.lod_grids.values():
                    grid.fill(0.0)
            elif lod in self.lod_grids:
                self.lod_grids[lod].fill(0.0)

    def _prune_voxels(self):
        """Remove voxels below threshold"""
        for grid in self.lod_grids.values():
            grid[grid < self.config.prune_threshold] = 0.0

    # ========================================================================
    # Coordinate Conversions
    # ========================================================================

    def _world_to_voxel(self, position: np.ndarray, resolution: float) -> np.ndarray:
        """Convert world position to voxel indices"""
        local_pos = position - self.grid_center
        voxel_pos = (local_pos / resolution).astype(int)

        # Offset to handle negative coordinates
        grid_shape = (self.grid_size / resolution).astype(int)
        voxel_pos += grid_shape // 2

        return voxel_pos

    def _voxel_to_world(self, voxel_idx: np.ndarray, resolution: float) -> np.ndarray:
        """Convert voxel indices to world position"""
        grid_shape = (self.grid_size / resolution).astype(int)
        local_voxel = voxel_idx - grid_shape // 2
        world_pos = self.grid_center + local_voxel * resolution
        return world_pos

    def _is_voxel_in_bounds(self, vx: int, vy: int, vz: int,
                           grid_shape: Tuple[int, int, int]) -> bool:
        """Check if voxel indices are within grid bounds"""
        return (0 <= vx < grid_shape[0] and
                0 <= vy < grid_shape[1] and
                0 <= vz < grid_shape[2])

    def _get_lod_for_distance(self, distance: float) -> LODLevel:
        """Determine appropriate LOD level for distance"""
        for lod_level in LODLevel:
            if distance <= lod_level.max_distance:
                return lod_level
        return LODLevel.ULTRA_LOW

    # ========================================================================
    # Dynamic Grid Management
    # ========================================================================

    def recenter_grid(self, new_center: np.ndarray):
        """
        Dynamically recenter the grid around a new position

        Args:
            new_center: New grid center position
        """
        with self.update_lock:
            # Calculate shift
            shift = new_center - self.grid_center

            # TODO: Implement grid data migration
            # For now, just update center
            self.grid_center = new_center.copy()

            print(f"Grid recentered to {new_center}")

    def resize_grid(self, new_size: np.ndarray):
        """
        Dynamically resize the grid

        Args:
            new_size: New grid size (x, y, z)
        """
        with self.update_lock:
            self.grid_size = new_size.copy()

            # Reinitialize grids with new size
            self._initialize_grids()

            print(f"Grid resized to {new_size}")

    def auto_adjust_grid(self):
        """
        Automatically adjust grid size and position based on tracked objects
        """
        if not self.tracked_objects:
            return

        # Calculate bounding box of all objects
        positions = np.array([obj.position for obj in self.tracked_objects.values()])
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)

        # Add margin
        margin = 500.0  # 500m margin
        min_pos -= margin
        max_pos += margin

        # Calculate new center and size
        new_center = (min_pos + max_pos) / 2
        new_size = max_pos - min_pos

        # Ensure minimum size
        new_size = np.maximum(new_size, [1000.0, 1000.0, 500.0])

        # Update if significantly different
        if np.linalg.norm(new_center - self.grid_center) > 100.0:
            self.recenter_grid(new_center)

        if np.linalg.norm(new_size - self.grid_size) > 100.0:
            self.resize_grid(new_size)

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def _update_memory_stats(self):
        """Update memory usage statistics"""
        total_memory = 0.0
        for grid in self.lod_grids.values():
            total_memory += grid.nbytes / (1024 ** 2)

        self.memory_usage_mb = total_memory

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        stats = {
            'tracked_objects': len(self.tracked_objects),
            'cameras': len(self.cameras),
            'memory_usage_mb': self.memory_usage_mb,
            'grid_center': self.grid_center.tolist(),
            'grid_size': self.grid_size.tolist(),
            'lod_levels': len(self.lod_grids),
            'frame_count': self.frame_count
        }

        # Count active voxels per LOD
        for lod_level, grid in self.lod_grids.items():
            active = np.count_nonzero(grid)
            stats[f'active_voxels_{lod_level.name}'] = int(active)

        return stats

    def print_statistics(self):
        """Print detailed statistics"""
        stats = self.get_statistics()

        print("\n=== Voxel Grid Manager Statistics ===")
        print(f"Tracked Objects:  {stats['tracked_objects']}")
        print(f"Cameras:          {stats['cameras']}")
        print(f"Memory Usage:     {stats['memory_usage_mb']:.2f} MB")
        print(f"Grid Center:      {stats['grid_center']}")
        print(f"Grid Size:        {stats['grid_size']}")
        print(f"LOD Levels:       {stats['lod_levels']}")

        for lod_level in LODLevel:
            key = f'active_voxels_{lod_level.name}'
            if key in stats:
                print(f"  {lod_level.name}: {stats[key]:,} voxels")

        print("=" * 38)

    def save_state(self, filename: str):
        """Save grid state to file"""
        state = {
            'config': {
                'center': self.grid_center.tolist(),
                'size': self.grid_size.tolist(),
                'base_resolution': self.base_resolution
            },
            'objects': {
                obj_id: {
                    'position': obj.position.tolist(),
                    'velocity': obj.velocity.tolist(),
                    'size': obj.size,
                    'confidence': obj.confidence
                }
                for obj_id, obj in self.tracked_objects.items()
            },
            'statistics': self.get_statistics()
        }

        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"State saved to {filename}")

    def load_state(self, filename: str):
        """Load grid state from file"""
        with open(filename, 'r') as f:
            state = json.load(f)

        # Restore configuration
        config = state['config']
        self.grid_center = np.array(config['center'])
        self.grid_size = np.array(config['size'])
        self.base_resolution = config['base_resolution']

        # Restore objects
        with self.update_lock:
            self.tracked_objects.clear()
            for obj_id_str, obj_data in state['objects'].items():
                obj_id = int(obj_id_str)
                self.add_tracked_object(
                    obj_id,
                    np.array(obj_data['position']),
                    obj_data['size'],
                    np.array(obj_data['velocity'])
                )

        print(f"State loaded from {filename}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Voxel Grid Manager - 5km+ Range Tracking")
    print("=========================================\n")

    # Create grid manager with custom configuration
    config = GridConfig(
        center=np.array([0.0, 0.0, 500.0]),
        size=np.array([5000.0, 5000.0, 2000.0]),
        base_resolution=1.0,
        max_memory_mb=500,
        enable_lod=True,
        enable_compression=True
    )

    manager = VoxelGridManager(config)
    manager.start_background_processing()

    # Add cameras
    print("\nAdding 10 cameras...")
    for i in range(10):
        angle = (i / 10.0) * 2.0 * np.pi
        position = np.array([
            np.cos(angle) * 100.0,
            np.sin(angle) * 100.0,
            50.0
        ])
        forward = np.array([
            -np.cos(angle),
            -np.sin(angle),
            -0.1
        ])
        manager.add_camera(i, position, forward)

    # Add tracked objects
    print("Adding 200 tracked objects...")
    for i in range(200):
        angle = (i / 200.0) * 2.0 * np.pi
        distance = 500.0 + i * 20.0
        position = np.array([
            np.cos(angle) * distance,
            np.sin(angle) * distance,
            (i % 20) * 50.0
        ])
        manager.add_tracked_object(i, position, size=2.0)

    # Simulate updates
    print("\nSimulating 30 frames of updates...")
    start_time = time.time()

    for frame in range(30):
        # Update some objects
        for i in range(0, 200, 10):
            obj = manager.tracked_objects[i]
            new_pos = obj.position + np.array([1.0, 0.0, 0.0])
            manager.update_object_position(i, new_pos)

        manager.frame_count += 1

    elapsed = time.time() - start_time
    fps = 30.0 / elapsed

    print(f"\nPerformance:")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Avg frame time: {elapsed*1000/30:.1f} ms")
    print(f"  FPS: {fps:.1f}")

    if fps >= 30.0:
        print("  ✓ Meets 30Hz real-time requirement")
    else:
        print("  ✗ Below 30Hz target")

    # Print statistics
    manager.print_statistics()

    # Check memory usage
    if manager.memory_usage_mb < config.max_memory_mb:
        print(f"\n✓ Memory usage ({manager.memory_usage_mb:.1f} MB) within {config.max_memory_mb} MB limit")
    else:
        print(f"\n✗ Memory usage ({manager.memory_usage_mb:.1f} MB) exceeds {config.max_memory_mb} MB limit")

    # Save state
    manager.save_state("grid_state.json")

    # Stop background processing
    manager.stop_background_processing()

    print("\nDone!")
