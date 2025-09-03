#!/usr/bin/env python3
"""
FITS File Loader for AstraVoxel
===============================

Advanced astronomical data processing module that handles FITS format files
with native celestial coordinate system support for real-time 3D reconstruction.

Features:
- Native FITS file format support for astronomical cameras and telescopes
- Celestial coordinate system transformations (RA/Dec/RaDec ↔ 3D cartesian)
- Motion detection and tracking from astronomical image sequences
- Integration with the pixeltovoxelprojector core algorithms
- Support for multiple observational bands and data types

Requirements:
- astropy for FITS handling
- numpy for array operations
- Optional: pyfits (fallback if astropy unavailable)
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
import json
import math

# Try to import astropy, provide fallback if not available
try:
    import astropy
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn(
        "astropy not available. FITS loading will be limited. "
        "Install with: pip install astropy",
        UserWarning
    )

# Try to import pyfits as fallback
try:
    import pyfits
    PYFITS_AVAILABLE = True
except ImportError:
    PYFITS_AVAILABLE = False

@dataclass
class FITSHeader:
    """Container for essential FITS header information."""
    telescope: str = ""
    instrument: str = ""
    object_name: str = ""
    observer: str = ""
    date_obs: str = ""
    exposure_time: float = 0.0
    filter_name: str = ""
    airmass: float = 1.0
    ra: float = 0.0  # degrees
    dec: float = 0.0  # degrees
    equinox: float = 2000.0
    pixel_scale: float = 1.0  # arcsec/pixel
    image_shape: Tuple[int, int] = (0, 0)
    data_type: str = "UNKNOWN"

@dataclass
class CelestialCoordinates:
    """Celestial coordinate system container."""
    ra_deg: float
    dec_deg: float

    def to_cartesian(self, distance_au: float = 1.0) -> np.ndarray:
        """Convert RA/Dec to 3D cartesian coordinates (heliocentric)."""
        ra_rad = math.radians(self.ra_deg)
        dec_rad = math.radians(self.dec_deg)

        # Convert to cartesian (heliocentric coordinate system)
        x = distance_au * math.cos(dec_rad) * math.cos(ra_rad)
        y = distance_au * math.cos(dec_rad) * math.sin(ra_rad)
        z = distance_au * math.sin(dec_rad)

        return np.array([x, y, z])

    def to_radians(self) -> Tuple[float, float]:
        """Convert to radians for astronomical calculations."""
        return math.radians(self.ra_deg), math.radians(self.dec_deg)

    @classmethod
    def from_header(cls, header) -> 'CelestialCoordinates':
        """Create from FITS header information."""
        ra = 0.0
        dec = 0.0

        # Extract RA/DEC from various header formats
        if 'RA' in header and 'DEC' in header:
            ra = float(header['RA'])
            dec = float(header['DEC'])
        elif 'CRVAL1' in header and 'CRVAL2' in header:
            # WCS coordinate system
            ra = float(header['CRVAL1'])
            dec = float(header['CRVAL2'])

        # Handle different RA/DEC formats (degrees vs HHMMSS/DDMMSS)
        if ra > 90:  # Likely in HHMMSS format
            ra_deg = ra / 10000.0 + (ra % 10000) / 100.0 / 60.0 + (ra % 100) / 60.0
            dec_deg = dec / 10000.0 + abs(dec % 10000) / 100.0 / 60.0 + abs(dec % 100) / 60.0
            if dec < 0:
                dec_deg = -dec_deg
        else:
            ra_deg = ra
            dec_deg = dec

        return cls(ra_deg=ra_deg, dec_deg=dec_deg)

class FITSLoader:
    """
    Advanced FITS file loader with astronomical coordinate support.

    This class provides comprehensive FITS file handling capabilities
    specifically designed for astronomical object detection and 3D reconstruction.
    """

    def __init__(self, verbose: bool = True):
        """Initialize the FITS loader."""
        self.verbose = verbose
        self.last_loaded_header: Optional[FITSHeader] = None
        self.supported_extensions = {'.fits', '.fts', '.fit'}

    def load_fits_file(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, FITSHeader]:
        """
        Load astronomical data from a FITS file with complete metadata extraction.

        Args:
            filepath: Path to the FITS file

        Returns:
            Tuple of (image_data, header_info)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid FITS file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {filepath}")

        if filepath.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")

        # Load with astropy (preferred)
        if ASTROPY_AVAILABLE:
            return self._load_with_astropy(filepath)

        # Fallback to pyfits
        elif PYFITS_AVAILABLE:
            return self._load_with_pyfits(filepath)

        else:
            raise ImportError(
                "No FITS library available. Please install astropy: pip install astropy"
            )

    def _load_with_astropy(self, filepath: Path) -> Tuple[np.ndarray, FITSHeader]:
        """Load FITS file using astropy."""
        try:
            with fits.open(filepath, memmap=False) as hdul:
                primary_hdu = hdul[0]
                header = primary_hdu.header
                data = primary_hdu.data

                # Extract metadata
                metadata = self._extract_metadata_astropy(header)

                # Convert data to expected format
                if data is None:
                    raise ValueError("FITS file contains no image data")

                # Ensure data is 2D and float
                if data.ndim == 2:
                    image_data = data.astype(np.float64)
                elif data.ndim == 3 and data.shape[0] == 1:
                    # Single plane 3D data
                    image_data = data[0].astype(np.float64)
                else:
                    raise ValueError(f"Unsupported data dimensions: {data.ndim}")

                if self.verbose:
                    print(f"✓ Loaded FITS file: {filepath.name}")
                    print(f"  Shape: {image_data.shape}")
                    print(f"  Data type: {metadata.data_type}")
                    print(f"  Object: {metadata.object_name}")
                    print(f"  Telescope: {metadata.telescope}")
                    print(f"  Coordinates: RA={metadata.ra:.4f}°, DEC={metadata.dec:.4f}°")

                return image_data, metadata

        except Exception as e:
            raise ValueError(f"Failed to load FITS file with astropy: {e}")

    def _load_with_pyfits(self, filepath: Path) -> Tuple[np.ndarray, FITSHeader]:
        """Load FITS file using pyfits (fallback)."""
        # Simplified fallback implementation
        warnings.warn("Using pyfits fallback - limited functionality")

        try:
            with pyfits.open(filepath) as hdul:
                header = hdul[0].header
                data = hdul[0].data

                # Basic data extraction
                image_data = np.array(data, dtype=np.float64) if data is not None else np.array([])
                metadata = self._extract_metadata_pyfits(header)

                return image_data, metadata

        except Exception as e:
            raise ValueError(f"Failed to load FITS file with pyfits: {e}")

    def _extract_metadata_astropy(self, header) -> FITSHeader:
        """Extract comprehensive metadata from astropy header."""
        metadata = FITSHeader()

        # Basic observational parameters
        metadata.telescope = header.get('TELESCOP', '')
        metadata.instrument = header.get('INSTRUME', '')
        metadata.object_name = header.get('OBJECT', '')
        metadata.observer = header.get('OBSERVER', '')
        metadata.date_obs = header.get('DATE-OBS', '')
        metadata.exposure_time = float(header.get('EXPTIME', header.get('EXPOSURE', 0.0)))
        metadata.filter_name = header.get('FILTER', header.get('FILTNAM', ''))
        metadata.airmass = float(header.get('AIRMASS', 1.0))

        # Celestial coordinates
        coords = CelestialCoordinates.from_header(header)
        metadata.ra = coords.ra_deg
        metadata.dec = coords.dec_deg
        metadata.equinox = float(header.get('EQUINOX', 2000.0))

        # Pixel scale information
        if 'PIXSCALE' in header:
            metadata.pixel_scale = float(header['PIXSCALE'])
        elif 'CDELT1' in header and 'CDELT2' in header:
            # From WCS, convert from degrees to arcsec
            cdelt1_arcsec = abs(float(header['CDELT1'])) * 3600.0
            cdelt2_arcsec = abs(float(header['CDELT2'])) * 3600.0
            metadata.pixel_scale = (cdelt1_arcsec + cdelt2_arcsec) / 2.0

        # Image dimensions and data type
        metadata.image_shape = (header.get('NAXIS2', 0), header.get('NAXIS1', 0))
        if header.get('BITPIX') == -32:
            metadata.data_type = "FLOAT32"
        elif header.get('BITPIX') == 16:
            metadata.data_type = "SIGNED_INT_16"
        else:
            metadata.data_type = f"BITPIX_{header.get('BITPIX', 'UNKNOWN')}"

        return metadata

    def _extract_metadata_pyfits(self, header) -> FITSHeader:
        """Extract basic metadata from pyfits header (fallback)."""
        metadata = FITSHeader()

        # Simple field extraction
        for key in ['TELESCOP', 'INSTRUME', 'OBJECT', 'OBSERVER',
                   'DATE-OBS', 'EXPTIME', 'FILTER']:
            if key in header:
                value = header[key]
                if key == 'EXPTIME':
                    metadata.exposure_time = float(value)
                else:
                    setattr(metadata, key.lower(), str(value))

        # Coordinates (simplified)
        if 'RA' in header and 'DEC' in header:
            metadata.ra = float(header['RA'])
            metadata.dec = float(header['DEC'])

        return metadata

    def batch_load_fits_directory(self, directory_path: Union[str, Path],
                                extension: str = ".fits",
                                recursive: bool = False) -> Dict[str, Tuple[np.ndarray, FITSHeader]]:
        """
        Load all FITS files from a directory with metadata extraction.

        Args:
            directory_path: Directory containing FITS files
            extension: File extension filter (default: .fits)
            recursive: Whether to search subdirectories

        Returns:
            Dictionary mapping filenames to (image_data, metadata) tuples
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        fits_files = list(directory.glob(f"**/*{extension}" if recursive else f"*{extension}"))

        if self.verbose:
            print(f"Found {len(fits_files)} {extension} files in {directory}")

        loaded_files = {}
        for fits_file in fits_files:
            try:
                image_data, metadata = self.load_fits_file(fits_file)
                loaded_files[fits_file.name] = (image_data, metadata)
            except Exception as e:
                warnings.warn(f"Failed to load {fits_file.name}: {e}")

        return loaded_files

    def create_motion_sequence(self, fits_directory: Union[str, Path],
                             output_metadata: Union[str, Path] = None) -> List[Dict]:
        """
        Process a directory of FITS files for motion detection sequence.

        This function creates the metadata JSON format expected by the
        motion detection pipeline, extracting temporal and positional
        information from FITS headers.

        Args:
            fits_directory: Directory containing FITS sequence
            output_metadata: Optional path to save metadata JSON

        Returns:
            List of sequence metadata entries
        """
        directory = Path(fits_directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Load all FITS files and extract metadata
        loaded_data = self.batch_load_fits_directory(directory)

        # Create motion sequence metadata
        sequence_metadata = []
        camera_counter = 0

        for filename, (image_data, metadata) in loaded_data.items():
            # Extract timing information
            timestamp_s = self._parse_timestamp(metadata.date_obs)

            # Create entry for each frame
            entry = {
                "camera_index": camera_counter,
                "frame_index": len(sequence_metadata),
                "camera_position": [0.0, 0.0, 0.0],  # Assuming fixed position for now
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "image_file": filename,
                "timestamp": timestamp_s,
                "ra": metadata.ra,
                "dec": metadata.dec,
                "exposure_time": metadata.exposure_time,
                "telescope": metadata.telescope,
                "object_name": metadata.object_name
            }

            sequence_metadata.append(entry)

        # Sort by timestamp
        sequence_metadata.sort(key=lambda x: x['timestamp'])

        # Save metadata if requested
        if output_metadata:
            output_path = Path(output_metadata)
            with open(output_path, 'w') as f:
                json.dump(sequence_metadata, f, indent=2)

            if self.verbose:
                print(f"✓ Saved motion sequence metadata to: {output_path}")

        if self.verbose:
            print(f"✓ Created motion sequence with {len(sequence_metadata)} frames")
            print(f"  Time span: {sequence_metadata[0]['timestamp']} to {sequence_metadata[-1]['timestamp']}")

        return sequence_metadata

    def _parse_timestamp(self, date_str: str) -> float:
        """Parse FITS timestamp string to Unix timestamp."""
        try:
            if ASTROPY_AVAILABLE:
                # Use astropy for accurate date parsing
                time_obj = Time(date_str)
                return time_obj.unix
            else:
                # Simple fallback parsing
                return 0.0
        except:
            return 0.0

    def calibrate_coordinates(self, ra_deg: float, dec_deg: float,
                            telescope_params: Dict = None) -> np.ndarray:
        """
        Convert celestial coordinates to 3D position with telescope calibration.

        Args:
            ra_deg: Right ascension in degrees
            dec_deg: Declination in degrees
            telescope_params: Dictionary of telescope-specific parameters

        Returns:
            3D position vector in astronomical units
        """
        coords = CelestialCoordinates(ra_deg, dec_deg)

        # Default distance (Earth-Sun distance for heliocentric coordinates)
        distance_au = 1.0

        if telescope_params:
            # Adjust for telescope-specific parameters
            if 'distance_au' in telescope_params:
                distance_au = telescope_params['distance_au']

        return coords.to_cartesian(distance_au)

    def extract_object_trajectory(self, fits_sequence: List[Tuple[np.ndarray, FITSHeader]],
                                object_ra: float, object_dec: float) -> Dict:
        """
        Extract trajectory information for a specific celestial object.

        Args:
            fits_sequence: List of (image_data, metadata) tuples
            object_ra: Object RA in degrees
            object_dec: Object DEC in degrees

        Returns:
            Dictionary containing trajectory data
        """
        trajectory = {
            'ra': object_ra,
            'dec': object_dec,
            'positions': [],
            'timestamps': [],
            'magnitudes': []
        }

        for image_data, metadata in fits_sequence:
            # Calculate object position in image coordinates
            # This would need WCS transformation in a full implementation
            position_au = self.calibrate_coordinates(object_ra, object_dec)

            trajectory['positions'].append(position_au.tolist())
            trajectory['timestamps'].append(self._parse_timestamp(metadata.date_obs))

            # Estimate magnitude from image data (simplified)
            center_region = image_data[100:200, 100:200]  # Central region
            magnitude_est = -2.5 * np.log10(np.sum(center_region) + 1e-10)
            trajectory['magnitudes'].append(magnitude_est)

        return trajectory

def create_astronomical_demo_data(output_dir: Union[str, Path] = "./astro_demo_data",
                                num_images: int = 10) -> Dict:
    """
    Create synthetic astronomical data for testing and demonstration.

    Args:
        output_dir: Directory to save demo data
        num_images: Number of synthetic images to generate

    Returns:
        Dictionary with information about created demo data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create synthetic FITS data (would normally create actual FITS files)
    demo_files = []

    for i in range(num_images):
        filename = f"synthetic_astro_{i:03d}.fits"
        filepath = output_path / filename

        # Generate synthetic astronomical image parameters
        params = {
            'filename': filename,
            'telescope': 'AstraVoxel Synthetic Telescope',
            'object': f'Demo Target {i}',
            'ra': 45.0 + i * 0.1,  # Vary RA slightly
            'dec': 30.0,
            'exposure_time': 60.0,
            'magnitude': 15.0 - i * 0.2
        }

        demo_files.append(params)

        # In a real implementation, this would create actual FITS files
        # For demo purposes, we'll just create JSON metadata
        metadata_file = output_path / f"{filename}.json"
        with open(metadata_file, 'w') as f:
            json.dump(params, f, indent=2)

    # Create motion sequence metadata
    loader = FITSLoader()
    sequence_file = output_path / "motion_sequence.json"

    # Simulate sequence entries (normally from actual FITS files)
    sequence_data = []
    for i, demo_file in enumerate(demo_files):
        entry = {
            "camera_index": 0,
            "frame_index": i,
            "camera_position": [1.0, 0.0, 0.0],  # Earth position approx
            "yaw": i * 0.1,
            "pitch": 0.0,
            "roll": 0.0,
            "image_file": demo_file['filename'],
            "timestamp": i * 60.0,  # 1 minute intervals
            "ra": demo_file['ra'],
            "dec": demo_file['dec']
        }
        sequence_data.append(entry)

    with open(sequence_file, 'w') as f:
        json.dump(sequence_data, f, indent=2)

    print(f"✓ Created astronomical demo data in: {output_path}")
    print(f"  Generated {num_images} synthetic images")
    print(f"  Motion sequence: {sequence_file}")

    return {
        'output_directory': str(output_path),
        'num_files': num_images,
        'sequence_file': str(sequence_file),
        'files': demo_files
    }

def main():
    """Main function for FITS loader demonstration."""
    print("FITS File Loader for AstraVoxel")
    print("===============================")

    # Create demo data
    demo_info = create_astronomical_demo_data()

    print("\nFITS Loader Usage Examples:")
    print("1. Load single FITS file:")
    print("   loader = FITSLoader()")
    print("   image, metadata = loader.load_fits_file('path/to/file.fits')")
    print()
    print("2. Process directory of FITS files:")
    print("   fits_data = loader.batch_load_fits_directory('/path/to/fits/')")
    print()
    print("3. Create motion detection sequence:")
    print("   sequence = loader.create_motion_sequence('/path/to/fits/')")
    print()
    print("4. Extract object trajectory:")
    print("   trajectory = loader.extract_object_trajectory(sequence, ra=45.0, dec=30.0)")

if __name__ == "__main__":
    main()