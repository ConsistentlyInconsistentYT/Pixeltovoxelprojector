# Pixeltovoxelprojector

Projects motion of pixels to a voxel

## Overview

PixelToVoxelProjector reconstructs 3D motion or object trajectories from multi-camera image sequences using voxel-based ray accumulation. It provides both C++ and Python tools for processing images, detecting motion, casting rays into a 3D voxel grid, and visualizing results.

## Features

1. Multi-camera motion detection: Detects moving pixels between consecutive frames for each camera.
1. Ray casting: Projects rays from camera pixels into a shared 3D voxel grid using voxel DDA.
1. Voxel accumulation: Aggregates brightness/motion evidence in a 3D grid.
1. Flexible metadata: Camera parameters and image info are loaded from a JSON file.
1. Python visualization: Visualizes the reconstructed 3D scene, camera positions, and motion using matplotlib and pyvista.
1. Astro support: Includes tools for processing astronomical FITS images and mapping sky coordinates.

## Requirements
1. C++17 compiler
1. Python 3.8+

## Example Workflow
1. Prepare metadata.json and images in a folder.
1. Run the C++ pipeline to generate voxel_grid.bin.
1. Use Python scripts to visualize and analyze the voxel grid.


## File Structure and metadata format

See [docs](./docs/file_structure.md).

## Build Instructions
C++ Executable
```bash
make
```
This builds `ray_voxel`.

Python Extension (`process_image_cpp`)
```bash
# Optionally create a virtual env (recommended)
python3 -m venv .venv
# or with uv:
# uv venv
# Then activate with (linux)
source .venv/bin/activate
# Windows
# .venv/bin/activate.bat or .venv/bin/Activate.ps1

# Install the dependencies
pip install pybind11 setuptools
python setup.py build_ext --inplace
```
This builds the `process_image_cpp` Python module.

## Usage
C++ Pipeline
```bash
./build/ray_voxel metadata.json images_folder output_voxel_grid.bin
```
- Processes images and metadata, outputs a binary voxel grid.

Python Visualization
```bash
python spacevoxelviewer.py
```
- Processes FITS images, accumulates into a voxel grid, and visualizes results.

```bash
python voxelmotionviewer.py
```
- Loads and interactively visualizes a voxel grid.
