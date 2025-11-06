#!/usr/bin/env python3
"""
Demo script for the Pixeltovoxelprojector
========================================

This script demonstrates how to use the process_image_cpp library to:
1. Process astronomical images
2. Convert pixel data to voxel space
3. Update celestial sphere textures
4. Handle motion processing

The demo creates synthetic astronomical data and shows typical usage patterns.
"""

import numpy as np
import sys
import os
import math

# Convert degrees to radians
def deg_to_rad(deg):
    return deg * math.pi / 180.0

# Convert radians to degrees
def rad_to_deg(rad):
    return rad * 180.0 / math.pi

def create_synthetic_astronomical_image(width, height, star_positions, star_magnitudes):
    """
    Create a synthetic astronomical image with point sources (stars).
    """
    image = np.zeros((height, width), dtype=np.float64)

    # Create a Gaussian PSF for each star
    psf_sigma = 2.0

    y_coords, x_coords = np.ogrid[:height, :width]

    for pos, mag in zip(star_positions, star_magnitudes):
        x_star, y_star = pos
        if 0 <= x_star < width and 0 <= y_star < height:
            # Convert magnitude to intensity (brighter means negative magnitude)
            intensity = 10 ** (-mag / 2.5)

            # Create Gaussian PSF
            dist_sq = (x_coords - x_star)**2 + (y_coords - y_star)**2
            psf = intensity * np.exp(-dist_sq / (2 * psf_sigma**2))
            image += psf

    return image

def create_voxel_grid(grid_size):
    """
    Create an empty 3D voxel grid.
    """
    return np.zeros(grid_size, dtype=np.float64)

def create_celestial_sphere_texture(texture_size):
    """
    Create an empty celestial sphere texture.
    """
    return np.zeros((texture_size[1], texture_size[0]), dtype=np.float64)

def demo_basic_image_processing():
    """
    Demonstrate basic image processing functionality.
    """
    print("=" * 60)
    print("Pixeltovoxelprojector - Basic Image Processing Demo")
    print("=" * 60)

    # Image parameters
    image_width = 1024
    image_height = 768

    # Create synthetic star field
    np.random.seed(42)  # For reproducible results
    num_stars = 50
    star_positions = np.random.rand(num_stars, 2) * [image_width, image_height]
    star_magnitudes = np.random.uniform(0, 8, num_stars)  # Apparent magnitudes

    # Generate synthetic astronomical image
    image = create_synthetic_astronomical_image(
        image_width, image_height,
        star_positions, star_magnitudes
    )

    print(f"Created synthetic image: {image.shape}")
    print(".2f")
    print(f"Number of synthetic stars: {num_stars}")
    print(f"Star magnitude range: {star_magnitudes.min():.2f} to {star_magnitudes.max():.2f}")

    # Camera parameters (simulating a horizontal camera looking at galactic center)
    earth_position = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # AU units
    pointing_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # Looking up
    fov = deg_to_rad(45.0)  # 45-degree field of view

    print("\nCamera parameters:")
    print(f"Earth position: {earth_position}")
    print(f"Pointing direction: {pointing_direction}")
    print(".1f")

    # Voxel grid parameters
    voxel_grid_size = (50, 50, 50)
    voxel_grid_extent = [
        (-1000.0, 1000.0),   # X: -1000 to 1000 space units
        (-1000.0, 1000.0),   # Y: -1000 to 1000 space units
        (-1000.0, 1000.0)    # Z: -1000 to 1000 space units
    ]

    voxel_grid = create_voxel_grid(voxel_grid_size)

    # Celestial sphere texture
    texture_size = (360, 180)  # RA x Dec degrees
    celestial_sphere_texture = create_celestial_sphere_texture(texture_size)

    # Processing parameters
    max_distance = 2000.0
    num_steps = 100
    center_ra_rad = 0.0  # 0h RA = galactic center for demo
    center_dec_rad = 0.0  # 0° Dec
    angular_width_rad = deg_to_rad(90.0)   # 90° angular coverage
    angular_height_rad = deg_to_rad(45.0)  # 45° angular coverage

    print("\nVoxel grid shape:", voxel_grid.shape)
    print(f"Voxel extent: {voxel_grid_extent}")
    print(".1f")
    print(f"Texture size: {texture_size}")

    # Try to call the actual compiled library function
    print("\n--- Attempting to call compiled library function ---")
    try:
        # Import the compiled library
        import sys
        import os

        # Add build/Debug path to Python path
        build_path = './build/Debug'
        if build_path not in sys.path:
            sys.path.append(build_path)

        print(f"Looking for library in: {os.path.abspath(build_path)}")
        print(f"Available files: {os.listdir(build_path) if os.path.exists(build_path) else 'Path not found'}")

        from process_image_cpp import process_image_cpp

        print("✓ Successfully imported process_image_cpp library")

        # Call the main processing function
        print("\nExecuting process_image_cpp function...")
        process_image_cpp(
            image,                           # Input astronomical image
            earth_position,                  # Earth position vector
            pointing_direction,              # Camera pointing direction
            fov,                             # Field of view in radians
            image_width,                     # Image width in pixels
            image_height,                    # Image height in pixels
            voxel_grid,                      # 3D voxel grid (modified in place)
            voxel_grid_extent,               # Spatial extents for each axis
            max_distance,                    # Maximum ray marching distance
            num_steps,                       # Number of integration steps
            celestial_sphere_texture,        # Celestial sphere texture (modified)
            center_ra_rad,                   # Center RA of sky patch
            center_dec_rad,                  # Center Dec of sky patch
            angular_width_rad,               # Angular width of sky patch
            angular_height_rad,              # Angular height of sky patch
            True,                            # Update celestial sphere
            False                            # Perform background subtraction
        )

        print("✓ Function executed successfully!")

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("This could mean the library wasn't built or path is incorrect")
        print("Make sure to build with: cd build && cmake --build .")

    except Exception as e:
        print(f"✗ Function execution error: {e}")

    print("\n--- Function Call Parameters ---")
    print("process_image_cpp(")
    print("    image = synthetic_astro_image,")
    print("    earth_position = [1.0, 0.0, 0.0],")
    print("    pointing_direction = [0.0, 0.0, 1.0],")
    print("    fov = 0.785 radians (45°),")
    print(f"    image_width = {image_width},")
    print(f"    image_height = {image_height},")
    print(f"    voxel_grid = {voxel_grid.shape} array,")
    print(f"    voxel_grid_extent = {voxel_grid_extent},")
    print(f"    max_distance = {max_distance},")
    print(f"    num_steps = {num_steps},")
    print(f"    celestial_sphere_texture = {texture_size} array,")
    print("    center_ra_rad = 0.0,")
    print("    center_dec_rad = 0.0,")
    print("    angular_width_rad = 1.57 (90°),")
    print("    angular_height_rad = 0.785 (45°),")
    print("    update_celestial_sphere = True,")
    print("    perform_background_subtraction = False")
    print(")")

    return {
        'image': image,
        'voxel_grid': voxel_grid,
        'celestial_sphere_texture': celestial_sphere_texture,
        'params': {
            'image_shape': (image_height, image_width),
            'voxel_shape': voxel_grid_size,
            'texture_shape': texture_size
        }
    }

def demo_data_visualization(demo_data):
    """
    Demonstrate data visualization and analysis.
    """
    print("\n" + "=" * 60)
    print("Data Visualization & Analysis")
    print("=" * 60)

    image = demo_data['image']
    voxel_grid = demo_data['voxel_grid']
    celestial_sphere_texture = demo_data['celestial_sphere_texture']

    # Analyze image statistics
    print("Image Statistics:")
    print(f"  Mean brightness: {image.mean():.6f}")
    print(f"  Max brightness: {image.max():.6f}")
    print(".6f")
    print(f"  Standard deviation: {image.std():.6f}")

    # Find brightest pixels (potential star locations)
    threshold = image.mean() + 2 * image.std()
    bright_pixels = np.where(image > threshold)
    print(f"\nDetected {len(bright_pixels[0])} bright regions above 2σ threshold")

    if len(bright_pixels[0]) > 0:
        print("Bright pixel coordinates (first 10):")
        for i in range(min(10, len(bright_pixels[0]))):
            y, x = bright_pixels[0][i], bright_pixels[1][i]
            print(".6f")

    # Voxel grid analysis
    print("\nVoxel Grid Status:")
    print(f"  Grid shape: {voxel_grid.shape}")
    print(f"  Total voxels: {voxel_grid.size}")
    print(f"  Non-zero voxels: {np.count_nonzero(voxel_grid)}")
    print(".6f")
    print(".6f")

    # Celestial sphere analysis
    print("\nCelestial Sphere Texture:")
    print(f"  Texture shape: {celestial_sphere_texture.shape}")
    print(f"  Total pixels: {celestial_sphere_texture.size}")
    print(f"  Non-zero pixels: {np.count_nonzero(celestial_sphere_texture)}")
    print(".6f")

def save_demo_data(demo_data, output_dir="./demo_output"):
    """
    Save demo data to files for visualization.
    """
    print("\n" + "=" * 60)
    print("Saving Demo Data")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Save synthetic image
        np.save(os.path.join(output_dir, 'synthetic_image.npy'), demo_data['image'])
        print("✓ Saved synthetic image to synthetic_image.npy")

        # Save voxel grid
        np.save(os.path.join(output_dir, 'voxel_grid.npy'), demo_data['voxel_grid'])
        print("✓ Saved voxel grid to voxel_grid.npy")

        # Save celestial sphere texture
        np.save(os.path.join(output_dir, 'celestial_sphere_texture.npy'),
                demo_data['celestial_sphere_texture'])
        print("✓ Saved celestial sphere texture to celestial_sphere_texture.npy")

        # Save parameters as JSON
        import json
        with open(os.path.join(output_dir, 'demo_parameters.json'), 'w') as f:
            json.dump(demo_data['params'], f, indent=2)
        print("✓ Saved parameters to demo_parameters.json")

        print(f"\nAll demo data saved to: {output_dir}")

    except Exception as e:
        print(f"Error saving data: {e}")

def print_usage_instructions():
    """
    Print instructions for using the compiled library.
    """
    print("\n" + "=" * 60)
    print("Usage Instructions")
    print("=" * 60)
    print("""
To use the compiled library:

1. Build the C++ library:
   cd build
   cmake ..
   cmake --build .

2. Run this demo with Python:
   python demo_pixeltovoxel.py

3. To use the library in your own code:
   ```python
   import sys
   sys.path.append('./build/Debug')  # Path to compiled DLL
   from process_image_cpp import process_image_cpp, process_motion

   # Call the function with your data
   process_image_cpp(
       your_image_array,
       earth_position_array,
       pointing_direction_array,
       # ... other parameters
   )
   ```

4. Data formats expected:
   - Images: 2D numpy arrays of type float64
   - Voxel grids: 3D numpy arrays of type float64
   - Celestial sphere textures: 2D numpy arrays of type float64
   - All angles in radians
   - Spatial coordinates in consistent units

5. For motion processing:
   ```python
   process_motion(
       metadata_json_path,
       images_folder_path,
       output_binary_path,
       N_grid_size,
       voxel_size,
       grid_center,
       motion_threshold,
       alpha_blend_factor
   )
   ```
""")

def main():
    """
    Main demo function.
    """
    print("Pixeltovoxelprojector - Complete Demo")
    print("====================================")

    # Run the basic demonstration
    demo_data = demo_basic_image_processing()

    # Show data analysis
    demo_data_visualization(demo_data)

    # Save data for further analysis
    save_demo_data(demo_data)

    # Print usage instructions
    print_usage_instructions()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("Check the ./demo_output directory for saved data.")
    print("=" * 60)

if __name__ == "__main__":
    main()