## File Structure
`ray_voxel.cpp`
C++ executable for motion detection, ray casting, and voxel grid accumulation from standard images.

`process_image.cpp`
C++ code (with pybind11) for advanced image processing, including celestial sphere mapping and voxel grid updates. Built as a Python extension.

`setup.py`
Python build script for compiling process_image.cpp into a Python module.

`Makefile`
Build script for compiling C++ executables.

`spacevoxelviewer.py`
Python script for processing FITS images, accumulating into a voxel grid, and visualizing results.

`voxelmotionviewer.py`
Python script for interactive visualization of voxel grids.


## Metadata Format
The metadata.json file should be a JSON array, with each entry containing:

```JSON
[
  {
    "camera_index": 0,
    "frame_index": 0,
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "fov_degrees": 60.0,
    "image_file": "frame_000.png",
    "camera_position": [0.0, 0.0, 0.0]
    // Optional: "object_name", "object_location"
  }
]
```