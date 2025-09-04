#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits  # For reading FITS files
from astropy.time import Time  # For handling observation times
from astropy.coordinates import get_body_barycentric, SkyCoord, solar_system_ephemeris
import astropy.units as u
import os

# Import the compiled C++ module
import process_image_cpp

# -------------------------------------------------------------------------------------
# Configurable Parameters
# -------------------------------------------------------------------------------------

# The voxel grid is a 3D array where we accumulate brightness values from rays cast
# through space. Adjust voxel_grid_size and grid_extent for your scenario.
voxel_grid_size = (400, 400, 400)  # Number of voxels in (x, y, z) directions
grid_extent = 3e12  # Half the length of one side of the voxel cube (in meters)

# The position and orientation of the voxel grid is determined based on RA/Dec.
# We choose a RA/Dec and assume the voxel grid is centered along that line-of-sight
# at distance_from_sun.
distance_from_sun = 1.496e+11 * 41.714231  # Approximately 1 AU in meters
center_ra = 280.50  # Center RA in degrees
center_dec = -20.  # Center Dec in degrees

# distance_from_sun = 1.496e+11 * 34  # Approximately 1 AU in meters
# center_ra = 287.967022  # Center RA in degrees
# center_dec = -20.713745  # Center Dec in degrees

# Threshold used to identify "significant" voxels (e.g., top 90% brightness).
brightness_threshold_percentile = .1

# Ray casting parameters: how far and how finely we cast rays into space.
max_distance = distance_from_sun * 100# Maximum distance (in meters) for ray casting
num_steps = 20000     # Number of steps along each ray

# Visualization parameters for plots.
marker_size = 5  # Marker size for scatter plots
alpha = 0.5      # Transparency for scatter points

# Default field-of-view (FOV) if not found in FITS header (in arcminutes).
default_fov_arcminutes = 2.7

# Directory containing FITS files
fits_directory = 'fits'

# Define the sky patch parameters for building a celestial sphere texture:
# We'll project the image directions onto this sky patch.
angular_width = 5.0    # Angular width of sky patch in degrees
angular_height = 5.0   # Angular height of sky patch in degrees
texture_width = 1024   # Texture width (pixels)
texture_height = 1024  # Texture height (pixels)

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------

def get_earth_position_icrs(obs_time):
    """
    Compute Earth's heliocentric position in ICRS coordinates for a given observation time.

    Parameters:
    - obs_time: Astropy Time object for the observation time.

    Returns:
    - earth_pos: Numpy array [x, y, z] in meters representing Earth's position in ICRS frame.
    """
    with solar_system_ephemeris.set('builtin'):
        earth_barycentric = get_body_barycentric('earth', obs_time)
        earth_icrs = earth_barycentric

    earth_pos = earth_icrs.get_xyz().to(u.meter).value
    earth_pos = np.array(earth_pos).flatten()
    return earth_pos

def get_telescope_pointing(header):
    """
    Get the telescope's pointing direction from FITS header RA_TARG and DEC_TARG.

    Parameters:
    - header: FITS header containing RA_TARG and DEC_TARG.

    Returns:
    - direction: Unit vector [x, y, z] in ICRS frame.
    """
    ra = header.get('RA_TARG')
    dec = header.get('DEC_TARG')
    if ra is None or dec is None:
        raise ValueError("RA_TARG and DEC_TARG not found in FITS header.")

    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    cartesian = coord.represent_as('cartesian')
    direction = np.array([cartesian.x.value, cartesian.y.value, cartesian.z.value])
    direction /= np.linalg.norm(direction)
    return direction

def get_observation_time(fits_file):
    """
    Extract the observation time (DATE-OBS, TIME-OBS) from a FITS file and return as an Astropy Time object.

    Parameters:
    - fits_file: Path to the FITS file.

    Returns:
    - obs_time: Astropy Time object representing observation time.
    """
    with fits.open(fits_file) as hdulist:
        header = hdulist[0].header
        date_obs = header.get('DATE-OBS')
        time_obs = header.get('TIME-OBS')

        if date_obs is None or time_obs is None:
            raise ValueError(f"DATE-OBS and TIME-OBS not found in FITS header of {fits_file}.")

        obs_time_str = f"{date_obs}T{time_obs}"
        obs_time = Time(obs_time_str, format='isot', scale='utc')
    return obs_time

def process_image(fits_file, voxel_grid, voxel_grid_extent, celestial_sphere_texture):
    """
    Process a single FITS image:
    - Compute Earth's position and telescope pointing.
    - Read and normalize image data.
    - Call the C++ function to cast rays, update voxel grid and celestial sphere texture.

    Parameters:
    - fits_file: Path to the FITS file.
    - voxel_grid: 3D numpy array for voxel accumulation.
    - voxel_grid_extent: Spatial extents of voxel grid.
    - celestial_sphere_texture: 2D numpy array for celestial sphere.

    Returns:
    - earth_position: Earth's position at obs time.
    - pointing_direction: Telescope pointing direction.
    - obs_time: Astropy Time object for observation time.
    """
    with fits.open(fits_file) as hdulist:
        print(f"Processing FITS file: {fits_file}")
        hdulist.info()

        header = hdulist[0].header
        date_obs = header.get('DATE-OBS')
        time_obs = header.get('TIME-OBS')

        if date_obs is None or time_obs is None:
            raise ValueError("DATE-OBS and TIME-OBS not found in FITS header.")

        obs_time_str = f"{date_obs}T{time_obs}"
        obs_time = Time(obs_time_str, format='isot', scale='utc')

        # Compute Earth position and telescope pointing
        earth_position = get_earth_position_icrs(obs_time)
        pointing_direction = get_telescope_pointing(header)

        # Find image data
        image_data = None
        if hdulist[0].data is not None:
            image_data = hdulist[0].data
            print("Found image data in Primary HDU.")
        elif 'SCI' in hdulist:
            image_data = hdulist['SCI'].data
            print("Found image data in 'SCI' extension.")
        else:
            for hdu in hdulist:
                if isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)):
                    if hdu.data is not None:
                        image_data = hdu.data
                        print(f"Found image data in extension '{hdu.name}'.")
                        break

        if image_data is None:
            raise ValueError("No image data found in the FITS file.")

        if image_data.ndim != 2:
            raise ValueError("Image data is not 2D.")

        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        image_min = np.min(image_data)
        image_max = np.max(image_data)
        if image_max - image_min == 0:
            raise ValueError("Image data has zero dynamic range.")
        image = (image_data - image_min) / (image_max - image_min)

        # Optional visualization of the FITS image
        # plt.figure(figsize=(8, 6))
        # plt.imshow(image, cmap='gray', origin='lower')
        # plt.title(f"FITS Image: {os.path.basename(fits_file)}")
        # plt.xlabel('Pixel X')
        # plt.ylabel('Pixel Y')
        # plt.colorbar(label='Normalized Intensity')
        # plt.show()

        height, width = image.shape

        # Determine field of view
        fov = header.get('FOV')
        if fov is None:
            cd1_1 = header.get('CD1_1')
            cd1_2 = header.get('CD1_2')
            cd2_1 = header.get('CD2_1')
            cd2_2 = header.get('CD2_2')
            if cd1_1 is not None and cd1_2 is not None and cd2_1 is not None and cd2_2 is not None:
                pixel_scale_x = np.sqrt(cd1_1**2 + cd2_1**2)
                pixel_scale_y = np.sqrt(cd1_2**2 + cd2_2**2)
                fov_x = pixel_scale_x * width
                fov_y = pixel_scale_y * height
                fov = max(fov_x, fov_y)
            else:
                fov = default_fov_arcminutes / 60  # degrees
        else:
            fov = float(fov)

        fov_rad = np.deg2rad(fov)

        # Convert Python data to lists for C++
        earth_position_list = earth_position.tolist()
        pointing_direction_list = pointing_direction.tolist()
        voxel_grid_extent_list = [
            (voxel_grid_extent[0][0], voxel_grid_extent[0][1]),
            (voxel_grid_extent[1][0], voxel_grid_extent[1][1]),
            (voxel_grid_extent[2][0], voxel_grid_extent[2][1])
        ]

        # Define sky patch in radians
        c_ra_rad = np.deg2rad(center_ra)
        c_dec_rad = np.deg2rad(center_dec)
        aw_rad = np.deg2rad(angular_width)
        ah_rad = np.deg2rad(angular_height)

        # Call C++ function to process the image
        process_image_cpp.process_image_cpp(
            image.astype(np.float64),
            earth_position_list,
            pointing_direction_list,
            fov_rad,
            width,
            height,
            voxel_grid,
            voxel_grid_extent_list,
            max_distance,
            num_steps,
            celestial_sphere_texture,
            c_ra_rad,
            c_dec_rad,
            aw_rad,
            ah_rad,
            True,   # update_celestial_sphere: True to accumulate celestial sphere brightness
            False   # perform_background_subtraction: False for now (no background subtraction)
        )

    return earth_position, pointing_direction, obs_time

def main():
    """
    Main function:
    1. Set up voxel grid and celestial sphere texture.
    2. Find and process FITS files (one pass).
    3. Analyze voxel grid, find brightest point, visualize results.
    """

    # Compute voxel grid center from RA/Dec
    center_coord = SkyCoord(ra=center_ra*u.degree, dec=center_dec*u.degree, frame='icrs')
    direction_vector = center_coord.cartesian.xyz.value
    voxel_grid_center = direction_vector * distance_from_sun

    voxel_grid_extent = (
        (voxel_grid_center[0] - grid_extent, voxel_grid_center[0] + grid_extent),
        (voxel_grid_center[1] - grid_extent, voxel_grid_center[1] + grid_extent),
        (voxel_grid_center[2] - grid_extent, voxel_grid_center[2] + grid_extent)
    )

    # Initialize voxel grid and celestial sphere texture
    voxel_grid = np.zeros(voxel_grid_size, dtype=np.float64)
    celestial_sphere_texture = np.zeros((texture_height, texture_width), dtype=np.float64)

    # List FITS files
    fits_files = [os.path.join(fits_directory, f) for f in os.listdir(fits_directory) if f.endswith('.fits')]

    if not fits_files:
        print(f"No FITS files found in directory '{fits_directory}'.")
        return

    # Sort FITS files by observation time
    fits_files_with_times = []
    for fits_file in fits_files:
        try:
            obs_time = get_observation_time(fits_file)
            fits_files_with_times.append((fits_file, obs_time))
            print(obs_time)
        except ValueError as e:
            print(e)


    fits_files_sorted = sorted(fits_files_with_times, key=lambda x: x[1])
    fits_files_sorted = [(f[0], f[1]) for f in fits_files_sorted]

    earth_positions = []
    pointing_directions = []
    observation_times = []

    # Process each FITS file once
    for fits_file, obs_time in fits_files_sorted:
        earth_pos, p_dir, obs_time = process_image(
            fits_file,
            voxel_grid,
            voxel_grid_extent,
            celestial_sphere_texture
        )
        earth_positions.append(earth_pos)
        pointing_directions.append(p_dir)
        observation_times.append(obs_time)

    # Create a background model (optional step)
    # In this example, we simply show the result after one pass.
    background_model = celestial_sphere_texture / len(fits_files_sorted)

    # Analyze the voxel grid
    voxel_grid_avg = voxel_grid / len(fits_files_sorted)

    # Threshold for significant voxels
    if np.any(voxel_grid_avg > 0):
        threshold = np.percentile(voxel_grid_avg[voxel_grid_avg > 0], brightness_threshold_percentile)
    else:
        threshold = 0

    object_voxels = voxel_grid_avg > threshold
    x_indices, y_indices, z_indices = np.nonzero(object_voxels)

    nx, ny, nz = voxel_grid_avg.shape
    x_min, x_max = voxel_grid_extent[0]
    y_min, y_max = voxel_grid_extent[1]
    z_min, z_max = voxel_grid_extent[2]

    x_coords = x_indices / nx * (x_max - x_min) + x_min
    y_coords = y_indices / ny * (y_max - y_min) + y_min
    z_coords = z_indices / nz * (z_max - z_min) + z_min

    intensities = voxel_grid_avg[object_voxels]

    # Find brightest point
    if intensities.size > 0:
        brightest_idx = np.argmax(intensities)
        brightest_x = x_coords[brightest_idx]
        brightest_y = y_coords[brightest_idx]
        brightest_z = z_coords[brightest_idx]

        brightest_coord = SkyCoord(
            x=brightest_x * u.meter,
            y=brightest_y * u.meter,
            z=brightest_z * u.meter,
            representation_type='cartesian',
            frame='icrs'
        )

        brightest_sph = brightest_coord.represent_as('spherical')
        brightest_ra = brightest_sph.lon.degree
        brightest_dec = brightest_sph.lat.degree
        distance_to_origin = np.sqrt(brightest_x**2 + brightest_y**2 + brightest_z**2)
        distance_to_origin_au = distance_to_origin / 1.496e+11

        print(f"Brightest Point Coordinates:")
        print(f"RA: {brightest_ra:.6f} degrees")
        print(f"Dec: {brightest_dec:.6f} degrees")
        print(f"Distance from Origin: {distance_to_origin_au:.6f} AU")
    else:
        print("No significant voxels found to identify brightest point.")

    # 3D Visualization of the voxel grid
    if len(x_coords) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(x_coords, y_coords, z_coords, c=intensities, cmap='hot', marker='o', s=marker_size, alpha=alpha)
        if intensities.size > 0:
            ax.scatter([brightest_x], [brightest_y], [brightest_z], c='blue', marker='*', s=100, label='Brightest Point')

        # Plot Earth positions
        earth_x = [pos[0] for pos in earth_positions]
        earth_y = [pos[1] for pos in earth_positions]
        earth_z = [pos[2] for pos in earth_positions]
        ax.scatter(earth_x, earth_y, earth_z, c='green', marker='o', s=200, label='Earth Positions')

        # Plot Voxel Grid Center
        ax.scatter([voxel_grid_center[0]], [voxel_grid_center[1]], [voxel_grid_center[2]],
                   c='purple', marker='x', s=100, label='Voxel Grid Center')

        # Draw arrows representing camera pointing directions over time
        times = np.array([t.mjd for t in observation_times])
        if len(times) > 1 and times.max() != times.min():
            times_norm = (times - times.min()) / (times.max() - times.min())
        else:
            times_norm = np.zeros_like(times)

        cmap = plt.cm.get_cmap('viridis')

        # Draw arrows from Earth positions in the direction of the camera pointing
        arrow_length = grid_extent * 0.5
        for idx, (pos, dir_vec, time_norm) in enumerate(zip(earth_positions, pointing_directions, times_norm)):
            x0, y0, z0 = pos
            dx = dir_vec[0] * arrow_length
            dy = dir_vec[1] * arrow_length
            dz = dir_vec[2] * arrow_length
            color = cmap(time_norm)
            ax.quiver(x0, y0, z0, dx, dy, dz, color=color, length=1.0, normalize=False, arrow_length_ratio=0.1)

        # Add a color bar for observation times
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(times)
        plt.colorbar(mappable, ax=ax, label='Observation Time (MJD)')

        ax.legend()
        plt.colorbar(sc, ax=ax, label='Average Brightness')
        ax.set_title('3D Visualization of Detected Objects, Earth Positions, and Camera Directions')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        plt.show()
    else:
        print("No significant voxels found to visualize.")

    # Visualize a 2D slice of the voxel grid
    voxel_grid_avg = voxel_grid / len(fits_files_sorted)
    z_slice_index = voxel_grid_avg.shape[2] // 2
    voxel_slice = voxel_grid_avg[:, :, z_slice_index]
    plt.figure(figsize=(8, 6))
    plt.imshow(voxel_slice.T, origin='lower', cmap='hot', extent=(x_min, x_max, y_min, y_max))
    plt.colorbar(label='Average Brightness')
    z_slice_pos = z_min + z_slice_index * (z_max - z_min) / voxel_grid_avg.shape[2]
    plt.title(f'Voxel Grid Slice at z = {z_slice_pos:.2f} meters')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()

if __name__ == '__main__':
    main()
