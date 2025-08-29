#!/usr/bin/env python3
"""
Visualization Script for Pixeltovoxelprojector Demo Results
==========================================================

This script loads the saved demo data and creates visualizations using matplotlib.
It provides multiple visualization modes for astronomical images, voxel grids,
and celestial sphere textures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Info: seaborn not available - using matplotlib defaults")

try:
    from matplotlib.animation import FuncAnimation
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False
    print("Info: matplotlib animation not available")

class PixeltovoxelVisualizer:
    """
    A comprehensive visualizer for astronomical voxel data.
    """

    def __init__(self, data_dir="./demo_output"):
        """
        Initialize the visualizer with data from the specified directory.
        """
        self.data_dir = data_dir
        self.check_data_availability()
        self.load_data()

        # Set up pretty plotting style
        if HAS_SEABORN:
            sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14

    def check_data_availability(self):
        """Check if required data files exist."""
        required_files = [
            'synthetic_image.npy',
            'voxel_grid.npy',
            'celestial_sphere_texture.npy',
            'demo_parameters.json'
        ]

        missing_files = []
        for filename in required_files:
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)

        if missing_files:
            raise FileNotFoundError(f"Missing required data files: {missing_files}")

        print(f"✓ Found all required data files in: {self.data_dir}")

    def load_data(self):
        """Load all the demo data from numpy and json files."""
        print("Loading demo data...")

        # Load numpy arrays
        self.image = np.load(os.path.join(self.data_dir, 'synthetic_image.npy'))
        self.voxel_grid = np.load(os.path.join(self.data_dir, 'voxel_grid.npy'))
        self.celestial_texture = np.load(os.path.join(self.data_dir, 'celestial_sphere_texture.npy'))

        # Load parameters
        with open(os.path.join(self.data_dir, 'demo_parameters.json'), 'r') as f:
            self.params = json.load(f)

        print(f"✓ Loaded image: {self.image.shape}")
        print(f"✓ Loaded voxel grid: {self.voxel_grid.shape}")
        print(f"✓ Loaded celestial texture: {self.celestial_texture.shape}")

    def visualize_astronomical_image(self, save_path=None):
        """
        Create a comprehensive visualization of the synthetic astronomical image.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Synthetic Astronomical Image Analysis', fontsize=16, fontweight='bold')

        # 1. Main image
        im1 = axes[0, 0].imshow(self.image, cmap='inferno', origin='lower')
        axes[0, 0].set_title('Synthetic Astronomical Image')
        axes[0, 0].set_xlabel('X pixel')
        axes[0, 0].set_ylabel('Y pixel')
        plt.colorbar(im1, ax=axes[0, 0], label='Brightness')

        # 2. Image histogram
        non_zero = self.image[self.image > 0]
        axes[0, 1].hist(non_zero.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title('Image Brightness Distribution')
        axes[0, 1].set_xlabel('Brightness')
        axes[0, 1].set_ylabel('Pixel Count')
        axes[0, 1].set_yscale('log')

        # 3. Bright star detection
        threshold = np.mean(non_zero) + 2 * np.std(non_zero)
        bright_mask = self.image > threshold
        bright_coords = np.where(bright_mask)

        axes[1, 0].imshow(self.image, cmap='gray', origin='lower', alpha=0.7)
        if len(bright_coords[0]) > 0:
            axes[1, 0].scatter(bright_coords[1], bright_coords[0],
                              c='red', s=10, alpha=0.8, label='Bright regions')
            axes[1, 0].legend()
        axes[1, 0].set_title(f'Detected Bright Regions (>{threshold:.4f})')
        axes[1, 0].set_xlabel('X pixel')
        axes[1, 0].set_ylabel('Y pixel')

        # 4. Image statistics
        axes[1, 1].axis('off')
        stats_text = f"""Image Statistics:

Mean Brightness: {self.image.mean():.6f}
Max Brightness: {self.image.max():.6f}
Min Brightness: {self.image[self.image > 0].min():.6f} (non-zero)
Std Deviation: {self.image.std():.6f}

Star Detection:
Bright Regions: {np.sum(bright_mask)}
Detection Threshold: {threshold:.6f}
Total Pixels: {self.image.size}
Image Shape: {self.image.shape}
"""

        axes[1, 1].text(0, 0.5, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved astronomical image visualization to: {save_path}")

        return fig

    def visualize_voxel_grid(self, save_path=None, threshold_percentile=95):
        """
        Create 3D visualization of the voxel grid.
        """
        fig = plt.figure(figsize=(16, 8))

        # Calculate threshold for visualization
        if self.voxel_grid.size > 0 and np.max(self.voxel_grid) > 0:
            threshold = np.percentile(self.voxel_grid[self.voxel_grid > 0], threshold_percentile)
        else:
            threshold = 0

        # Get non-zero voxels
        nonzero_mask = self.voxel_grid > threshold
        if np.sum(nonzero_mask) == 0:
            # No significant voxels - show summary
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                   '.10f',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14,
                   bbox=dict(boxstyle="round,pad=1", facecolor="lightcoral"))
            ax.set_title('Voxel Grid Analysis - No Significant Data')
            ax.axis('off')
        else:
            # Get coordinates of non-zero voxels
            coords = np.where(nonzero_mask)
            values = self.voxel_grid[nonzero_mask]

            # Create 3D scatter plot
            ax = fig.add_subplot(121, projection='3d')

            scatter = ax.scatter(coords[2], coords[1], coords[0],
                               c=values, cmap='plasma', alpha=0.7,
                               s=20, edgecolors='black', linewidth=0.5)

            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_zlabel('Z coordinate')
            ax.set_title(f'3D Voxel Grid Visualization\n(Threshold: {threshold:.2e})')

            plt.colorbar(scatter, ax=ax, label='Voxel Value', shrink=0.7)

            # Add 2D projections
            ax2 = fig.add_subplot(222)
            ax2.scatter(coords[2], coords[0], c=values, cmap='plasma', alpha=0.6, s=10)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Z')
            ax2.set_title('X-Z Projection')

            ax3 = fig.add_subplot(224)
            ax3.scatter(coords[1], coords[0], c=values, cmap='plasma', alpha=0.6, s=10)
            ax3.set_xlabel('Y')
            ax3.set_ylabel('Z')
            ax3.set_title('Y-Z Projection')

            ax4 = fig.add_subplot(223)
            ax4.scatter(coords[1], coords[2], c=values, cmap='plasma', alpha=0.6, s=10)
            ax4.set_xlabel('Y')
            ax4.set_ylabel('X')
            ax4.set_title('Y-X Projection (top view)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved voxel grid visualization to: {save_path}")

        return fig

    def visualize_celestial_sphere(self, save_path=None):
        """
        Visualize the celestial sphere texture with astronomical coordinates.
        """
        fig = plt.figure(figsize=(14, 6))

        # Create a RA/Dec grid for the texture
        ra_deg = np.linspace(0, 360, self.celestial_texture.shape[1])
        dec_deg = np.linspace(-90, 90, self.celestial_texture.shape[0])

        # Plot the celestial sphere
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(self.celestial_texture, extent=[0, 360, -90, 90],
                        aspect='auto', cmap='viridis', origin='lower')
        ax1.set_xlabel('Right Ascension (degrees)')
        ax1.set_ylabel('Declination (degrees)')
        ax1.set_title('Celestial Sphere Texture')
        plt.colorbar(im1, ax=ax1, label='Intensity')

        # Add equatorial line
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Celestial Equator')
        ax1.legend()

        # Plot intensity distribution
        ax2 = fig.add_subplot(222)
        ax2.hist(self.celestial_texture.flatten(), bins=50, alpha=0.7,
                color='blue', edgecolor='black')
        ax2.set_title('Intensity Distribution')
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Pixel Count')
        ax2.set_yscale('log')

        # Polar plot of celestial sphere
        ax3 = fig.add_subplot(224, polar=True)

        # Convert to polar coordinates
        nonzero = self.celestial_texture[self.celestial_texture > 0]
        if len(nonzero) > 0:
            # Find brightest region in celestial coordinates
            bright_idx = np.unravel_index(np.argmax(self.celestial_texture),
                                        self.celestial_texture.shape)
            bright_ra = ra_deg[bright_idx[1]]
            bright_dec = dec_deg[bright_idx[0]]

            # Convert to polar coordinates
            r = 90 - bright_dec  # Distance from north celestial pole
            theta = np.radians(bright_ra)

            ax3.scatter(theta, r, s=100, c='red', alpha=0.8, edgecolors='black')
            ax3.set_title('Brightest Region\n(celestial coordinates)')

            # Set up polar plot properly
            ax3.set_rlim(0, 90)
            ax3.set_rticks([30, 60, 90])
            ax3.set_rlabel_position(135)
            ax3.grid(True, alpha=0.3)

        # Statistics
        ax4 = fig.add_subplot(223)
        ax4.axis('off')
        stats_text = ".2f"f"""Celestial Sphere Stats:

Shape: {self.celestial_texture.shape}
Total Pixels: {self.celestial_texture.size}
Max Intensity: {self.celestial_texture.max():.6f}
Min Intensity: {self.celestial_texture.min():.6f}
Non-zero Pixels: {np.count_nonzero(self.celestial_texture)}

Coverage: 360° RA × 180° Dec
Resolution: {360/self.celestial_texture.shape[1]:.2f}°/pixel
"""

        ax4.text(0, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved celestial sphere visualization to: {save_path}")

        return fig

    def create_comprehensive_report(self, save_dir="./visualizations"):
        """
        Generate a comprehensive visual report with all data visualizations.
        """
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
        print("="*60)

        # Astronomical image visualization
        print("\n1. Creating astronomical image visualization...")
        fig1 = self.visualize_astronomical_image()
        fig1.savefig(os.path.join(save_dir, 'astronomical_image_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Voxel grid visualization
        print("\n2. Creating 3D voxel grid visualization...")
        fig2 = self.visualize_voxel_grid()
        if np.sum(self.voxel_grid > 0) > 0:  # Only save if there's data
            fig2.savefig(os.path.join(save_dir, 'voxel_grid_3d.png'),
                        dpi=300, bbox_inches='tight')
        else:
            print("   → Skipping voxel grid (no significant data)")
        plt.close(fig2)

        # Celestial sphere visualization
        print("\n3. Creating celestial sphere visualization...")
        fig3 = self.visualize_celestial_sphere()
        fig3.savefig(os.path.join(save_dir, 'celestial_sphere_texture.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # Create summary plot
        print("\n4. Creating summary dashboard...")
        self.create_summary_dashboard(save_dir)

        print(f"\n✓ All visualizations saved to: {save_dir}")
        print("✓ Report generation completed!")

    def create_summary_dashboard(self, save_dir):
        """
        Create a summary dashboard with key metrics and small previews.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pixeltovoxelprojector Demo Results - Summary Dashboard',
                    fontsize=16, fontweight='bold')

        # 1. Image preview
        axes[0, 0].imshow(self.image, cmap='inferno', origin='lower')
        axes[0, 0].set_title('Astronomical Image')
        axes[0, 0].set_xlabel('X pixel')
        axes[0, 0].set_ylabel('Y pixel')

        # 2. Image histogram
        non_zero = self.image[self.image > 0]
        axes[1, 0].hist(non_zero.flatten(), bins=30, alpha=0.7, color='green')
        axes[1, 0].set_title('Brightness Histogram')
        axes[1, 0].set_xlabel('Brightness')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_yscale('log')

        # 3. Voxel grid scatter
        voxel_coords = np.where(self.voxel_grid > 0)
        if len(voxel_coords[0]) > 0:
            axes[0, 1].scatter(voxel_coords[2], voxel_coords[1], voxel_coords[0],
                              c=self.voxel_grid[voxel_coords], cmap='plasma',
                              alpha=0.6, s=5)
            axes[0, 1].set_title(f'Voxel Grid ({len(voxel_coords[0])} points)')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
            axes[0, 1].set_zlabel('Z')
        else:
            axes[0, 1].text(0.5, 0.5, 'No significant\nvoxel data',
                           transform=axes[0, 1].transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle="round", facecolor="lightcoral"))

        # 4. Celestial sphere polar plot
        axes[1, 1].remove()
        ax_polar = fig.add_subplot(2, 3, 5, polar=True)
        if np.max(self.celestial_texture) > 0:
            ax_polar.text(0, 0.5, 'Celestial\nSphere\nAvailable',
                         transform=ax_polar.transAxes, ha='center', va='center')
        else:
            ax_polar.text(0, 0.5, 'Celestial\nSphere\n(Not processed)',
                         transform=ax_polar.transAxes, ha='center', va='center')
        ax_polar.set_title('Celestial Coverage', pad=20)

        # 5. Statistics summary
        axes[0, 2].axis('off')
        stats_text = f"""SUMMARY STATISTICS

Image Metrics:
  Shape: {self.params['image_shape']}
  Mean: {self.image.mean():.4e}
  Max: {self.image.max():.4e}
  Std: {self.image.std():.4e}

Voxel Grid:
  Shape: {self.params['voxel_shape']}
  Total Voxels: {self.voxel_grid.size:,}
  Non-zero: {np.count_nonzero(self.voxel_grid):,}
  Max Value: {self.voxel_grid.max():.4e}

Celestial Texture:
  Shape: {self.params['texture_shape']}
  Total Pixels: {self.celestial_texture.size:,}
  Non-zero: {np.count_nonzero(self.celestial_texture):,}
  Max Value: {self.celestial_texture.max():.4e}
"""

        axes[0, 2].text(0, 0.5, stats_text, transform=axes[0, 2].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))

        # 6. Processing status
        axes[1, 2].axis('off')
        status_text = """PROCESSING STATUS

✓ Synthetic astronomical image generated
✓ Star field simulation completed
✓ Voxel grid initialized
✓ Celestial sphere texture initialized
✓ Data export completed

Note: Actual voxel processing requires
compiled C++ library to be called.

Use: python demo_pixeltovoxel.py
"""

        axes[1, 2].text(0, 0.5, status_text, transform=axes[1, 2].transAxes,
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, 'summary_dashboard.png'),
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    def show_interactive_voxel_slice(self):
        """
        Create an interactive voxel grid slice viewer.
        Requires matplotlib animation support.
        """
        if not HAS_ANIMATION:
            print("Matplotlib animation not available. Skipping interactive viewer.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        def update_frame(frame):
            ax.clear()
            slice_data = self.voxel_grid[:, :, frame]
            im = ax.imshow(slice_data, cmap='plasma', origin='lower')
            ax.set_title(f'Voxel Grid Slice Z={frame}')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            plt.colorbar(im, ax=ax, label='Intensity')
            return [im]

        # Create animation
        num_frames = self.voxel_grid.shape[2]
        if num_frames > 1:
            anim = FuncAnimation(fig, update_frame, frames=num_frames,
                               interval=200, blit=False)

            print("Interactive voxel slice viewer created!")
            return anim
        else:
            print("No animation needed - only one slice available.")
            return None

def main():
    """
    Main function to run all visualizations.
    """
    print("Pixeltovoxelprojector - Results Visualization")
    print("=" * 50)

    try:
        # Create visualizer
        visualizer = PixeltovoxelVisualizer()

        # Generate comprehensive report
        visualizer.create_comprehensive_report()

        # Try to show individual visualizations
        print("\n" + "="*50)
        print("INDIVIDUAL VISUALIZATIONS AVAILABLE:")
        print("1. Astronomical image analysis")
        print("2. 3D voxel grid visualization")
        print("3. Celestial sphere texture")
        print("4. Summary dashboard")
        print("="*50)

        # Optional: Show interactive visualization
        if HAS_ANIMATION and input("\nShow interactive voxel slice viewer? (y/N): ").lower().startswith('y'):
            anim = visualizer.show_interactive_voxel_slice()
            if anim:
                plt.show()
        else:
            print("\nVisualization complete! Check the ./visualizations/ directory for all images.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you've run the demo script first:")
        print("python demo_pixeltovoxel.py")
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()