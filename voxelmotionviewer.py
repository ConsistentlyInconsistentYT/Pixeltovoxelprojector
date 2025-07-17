#!/usr/bin/env python3

"""
pyvista_interactive_view_with_rotation_history.py

Requirements:
    pip install pyvista numpy

Usage:
    python pyvista_interactive_view_with_rotation_history.py

Description:
    1) Loads voxel_grid.bin (written by your C++ code).
    2) Interprets the 3D array shape as (Z, Y, X).
    3) Extracts top percentile of brightness.
    4) Applies an additional Euler rotation to the entire cloud (user-defined).
    5) Displays them interactively in a PyVista window,
       so you can orbit, zoom, and pan with the mouse.
    6) On closing the window, saves a 1920×1080 screenshot named 'voxel_####.png'
       in a 'screenshots/' folder, so you can keep a history of runs.
"""

import os
import re
import math
import numpy as np
import pyvista as pv


def load_voxel_grid(filename):
    """
    Reads a voxel grid from a binary file with the following layout:
      1) int32: N (size of the NxNxN grid)
      2) float32: voxel_size
      3) N*N*N float32: the voxel data in row-major order
    Returns:
       voxel_grid (N x N x N),
       voxel_size
    """
    with open(filename, "rb") as f:
        # read N
        raw = f.read(4)
        N = np.frombuffer(raw, dtype=np.int32)[0]

        # read voxel_size
        raw = f.read(4)
        voxel_size = np.frombuffer(raw, dtype=np.float32)[0]

        # read the voxel data
        count = N*N*N
        raw = f.read(count*4)
        data = np.frombuffer(raw, dtype=np.float32)
        voxel_grid = data.reshape((N, N, N))

    return voxel_grid, voxel_size


def extract_top_percentile_z_up(voxel_grid, voxel_size, grid_center,
                                percentile=99.5, use_hard_thresh=False, hard_thresh=700):
    """
    Extract the top 'percentile' bright voxels (or above 'hard_thresh').
    We interpret the array shape as (Z, Y, X).

    index: voxel[z, y, x]

    We'll produce an Nx3 array of points in (x, y, z).
    Then we'll produce intensities as a separate array.
    """
    N = voxel_grid.shape[0]  # assume shape is (N,N,N)
    half_side = (N * voxel_size) * 0.5
    grid_min = grid_center - half_side

    # Flatten to find threshold
    flat_vals = voxel_grid.ravel()
    if use_hard_thresh:
        thresh = hard_thresh
    else:
        thresh = np.percentile(flat_vals, percentile)

    coords = np.argwhere(voxel_grid > thresh)
    if coords.size == 0:
        print(f"No voxels above threshold {thresh}. Nothing to display.")
        return None, None

    intensities = voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]]

    # Because we're now treating 0 -> z, 1 -> y, 2 -> x:
    z_idx = coords[:, 0] + 0.5
    y_idx = coords[:, 1] + 0.5
    x_idx = coords[:, 2] + 0.5

    # Convert to world coords
    x_world = grid_min[0] + x_idx * voxel_size
    y_world = grid_min[1] + y_idx * voxel_size
    z_world = grid_min[2] + z_idx * voxel_size

    points = np.column_stack((x_world, y_world, z_world))
    return points, intensities


def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    """
    Build a rotation matrix (3x3) for Euler angles (rx, ry, rz) in degrees,
    applied in X->Y->Z order. That is:
      R = Rz(rz) * Ry(ry) * Rx(rx)
    so we rotate first by rx around X, then ry around Y, then rz around Z.
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # Rx
    Rx = np.array([
        [1,   0,   0],
        [0,  cx, -sx],
        [0,  sx,  cx]
    ], dtype=np.float32)

    # Ry
    Ry = np.array([
        [ cy,  0,  sy],
        [  0,  1,   0],
        [-sy,  0,  cy]
    ], dtype=np.float32)

    # Rz
    Rz = np.array([
        [ cz, -sz,  0],
        [ sz,  cz,  0],
        [  0,   0,  1]
    ], dtype=np.float32)

    # Combined: Rz * Ry * Rx
    Rtemp = Rz @ Ry
    Rfinal = Rtemp @ Rx
    return Rfinal


def get_next_image_index(folder, prefix="voxel_", suffix=".png"):
    """
    Scan 'folder' for files named like 'voxel_XXXX.png'.
    Find the largest XXXX as int, return that + 1.
    If none found, return 1.
    """
    if not os.path.exists(folder):
        return 1

    pattern = re.compile(rf"^{prefix}(\d+){suffix}$")
    max_index = 0
    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx
    return max_index + 1


def main():
    # 1) Load the voxel grid
    voxel_grid, vox_size = load_voxel_grid("voxel_grid.bin")
    print("Loaded voxel grid:", voxel_grid.shape, "voxel_size=", vox_size)
    print("Max voxel value:", voxel_grid.max())

    # 2) Define the grid center (x,y,z)
    grid_center = np.array([30, 0, 14000], dtype=np.float32)

    # 3) Extract top percentile (Z-up)
    percentile_to_show = 99.9
    points, intensities = extract_top_percentile_z_up(
        voxel_grid,
        voxel_size=vox_size,
        grid_center=grid_center,
        percentile=percentile_to_show,
        use_hard_thresh=False,
        hard_thresh=700
    )
    if points is None:
        return  # nothing to show

    # 4) Optional rotation
    # e.g. rotate to fix orientation
    rx_deg = 90
    ry_deg = 270
    rz_deg = 0

    R = rotation_matrix_xyz(rx_deg, ry_deg, rz_deg)  # shape (3,3)
    points_rot = points @ R.T
    # 5) PyVista Plotter (interactive)
    plotter = pv.Plotter(off_screen=True,)
    plotter.set_background("white")
    plotter.enable_terrain_style()


    # Convert to PolyData with scalars
    cloud = pv.PolyData(points_rot)
    cloud["intensity"] = intensities

    # Add points
    plotter.add_points(
        cloud,
        scalars="intensity",
        cmap="hot",
        point_size=4.0,
        render_points_as_spheres=True,
        opacity=0.1,
    )

    plotter.add_scalar_bar(
        title="Brightness",
        n_labels=5
    )

    # 6) Determine the next screenshot index
    screenshot_folder = "screenshots"
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)
    next_idx = get_next_image_index(screenshot_folder, prefix="voxel_", suffix=".png")
    out_name = f"voxel_{next_idx:04d}.png"
    out_path = os.path.join(screenshot_folder, out_name)
    # 7) Show the interactive window at 1920x1080 and save final screenshot
    #    The screenshot is generated when you close the plot window.
    plotter.show(window_size=[3840, 2160], auto_close=False, screenshot=out_path)



    print(f"[Info] Saved screenshot to {out_path}")
    plotter = pv.Plotter(off_screen=False, )
    plotter.set_background("white")
    plotter.enable_terrain_style()

    # Convert to PolyData with scalars
    cloud = pv.PolyData(points_rot)
    cloud["intensity"] = intensities

    # Add points
    plotter.add_points(
        cloud,
        scalars="intensity",
        cmap="hot",
        point_size=4.0,
        render_points_as_spheres=True,
        opacity=0.05,
    )

    # plotter.add_scalar_bar(
    #     title="Brightness",
    #     n_labels=5
    # )
    print("[Done]")
    plotter.show(window_size=[1920, 1080], auto_close=False)




if __name__ == "__main__":
    main()
