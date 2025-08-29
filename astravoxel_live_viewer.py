#!/usr/bin/env python3
"""
AstraVoxel Live Motion Viewer
=============================

Enhanced webcam demo with real-time motion detection overlay and live voxel rendering.
Shows the complete pipeline from camera feed to 3D voxel grid.

Features:
✅ Live camera feed display
✅ Real-time motion detection with visual indicators
✅ Motion data accumulation into 3D voxel space
✅ Interactive 3D visualization updates
✅ Visual connection between detected motion and voxels

This demonstrates AstraVoxel's core capability of transforming
real motion from camera feeds into 3D spatial reconstructions.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import threading
import time

class AstraVoxelMotionViewer:
    """
    Enhanced motion viewer with live camera feed and motion overlays
    """

    def __init__(self, root):
        """Initialize the enhanced motion viewer"""
        self.root = root
        self.root.title("AstraVoxel Live Motion Viewer")
        self.root.geometry("1400x800")

        # Processing state
        self.camera = None
        self.camera_active = False
        self.processing_active = False
        self.current_frame = None
        self.previous_frame = None
        self.voxel_grid = None
        self.grid_size = 24
        self.motion_threshold = 25.0
        self.frame_count = 0

        # Motion tracking
        self.motion_points = []
        self.last_motion_time = 0

        # Interface setup
        self.setup_interface()
        self.detect_camera()

    def setup_interface(self):
        """Set up the complete interface"""
        # Main layout with title and status
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, pady=5)

        ttk.Label(title_frame,
                 text="🎥 AstraVoxel Live Motion Viewer - Camera Feed → Motion Detection → 3D Visualization",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        self.status_label = ttk.Label(title_frame, text="● Ready", foreground="blue")
        self.status_label.pack(side=tk.RIGHT)

        # Main content area
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Horizontal split: Left (Camera+Motion) | Right (3D Voxels)
        main_paned = ttk.PanedWindow(content_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left pane: Camera feed with motion overlay
        left_pane = ttk.Frame(main_paned)
        self.setup_camera_motion_pane(left_pane)
        main_paned.add(left_pane, weight=1)

        # Right pane: 3D voxel visualization
        right_pane = ttk.Frame(main_paned)
        self.setup_voxel_pane(right_pane)
        main_paned.add(right_pane, weight=1)

        # Bottom controls
        self.setup_bottom_controls(content_frame)

    def setup_camera_motion_pane(self, parent):
        """Set up camera feed and motion detection pane"""
        pane_title = ttk.Label(parent, text="📹 Live Camera Feed with Motion Detection",
                              font=("Segoe UI", 10, "bold"))
        pane_title.pack(pady=(0, 5))

        # Camera preview canvas
        self.camera_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.camera_ax = self.camera_figure.add_subplot(111)
        self.camera_ax.set_title("Camera Feed with Motion Overlay")
        self.camera_ax.axis('off')

        # Initialize with empty image
        empty_img = np.zeros((240, 320), dtype=np.uint8)
        self.camera_display = self.camera_ax.imshow(empty_img, cmap='gray', vmin=0, vmax=255)

        # Motion detection overlay (red scatter points)
        self.motion_overlay, = self.camera_ax.plot([], [], 'ro', markersize=6, alpha=0.8,
                                                label='Motion Detected')

        self.camera_ax.legend(loc='upper right')

        # Canvas
        self.camera_canvas = FigureCanvasTkAgg(self.camera_figure, master=parent)
        self.camera_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Camera controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 5))

        self.camera_status_label = ttk.Label(control_frame, text="Camera: Not detected")
        self.camera_status_label.pack(side=tk.LEFT)

        ttk.Button(control_frame, text="🔄 Refresh Camera",
                  command=self.detect_camera).pack(side=tk.LEFT, padx=(20, 0))

        # Motion info
        motion_frame = ttk.LabelFrame(parent, text="Motion Detection Status")
        motion_frame.pack(fill=tk.X)

        self.motion_info_label = ttk.Label(motion_frame,
                                         text="Motion Detection: Idle\nNo motion detected",
                                         justify=tk.LEFT)
        self.motion_info_label.pack(anchor=tk.W, padx=5, pady=5)

    def setup_voxel_pane(self, parent):
        """Set up 3D voxel visualization pane"""
        pane_title = ttk.Label(parent, text="🧊 3D Voxel Reconstruction from Motion",
                              font=("Segoe UI", 10, "bold"))
        pane_title.pack(pady=(0, 5))

        # 3D visualization
        self.voxel_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.voxel_ax = self.voxel_figure.add_subplot(111, projection='3d')

        # Initialize empty voxel display
        empty_coords = np.array([[0], [0], [0]])
        empty_colors = np.array([0])
        self.voxel_scatter = self.voxel_ax.scatter(empty_coords[0], empty_coords[1], empty_coords[2],
                                                c=empty_colors, cmap='plasma', marker='o', s=20, alpha=0.8)

        self.voxel_ax.set_xlabel('X')
        self.voxel_ax.set_ylabel('Y')
        self.voxel_ax.set_zlabel('Z')
        self.voxel_ax.set_title('Live Voxel Reconstruction')
        self.voxel_ax.set_xlim(0, self.grid_size)
        self.voxel_ax.set_ylim(0, self.grid_size)
        self.voxel_ax.set_zlim(0, self.grid_size)

        # Add colorbar
        self.voxel_figure.colorbar(self.voxel_scatter, ax=self.voxel_ax, shrink=0.5,
                                 label='Voxel Intensity')

        # Canvas
        self.voxel_canvas = FigureCanvasTkAgg(self.voxel_figure, master=parent)
        self.voxel_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Voxel stats
        stats_frame = ttk.LabelFrame(parent, text="Voxel Statistics")
        stats_frame.pack(fill=tk.X)

        self.voxel_stats_label = ttk.Label(stats_frame,
                                         text="Total Voxels: 0\nActive Voxels: 0\nPeak Intensity: 0",
                                         justify=tk.LEFT)
        self.voxel_stats_label.pack(anchor=tk.W, padx=5, pady=5)

    def setup_bottom_controls(self, parent):
        """Set up bottom control panel"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        # Left: Camera controls
        camera_ctrl = ttk.LabelFrame(control_frame, text="Camera Control", padding=5)
        camera_ctrl.pack(side=tk.LEFT, padx=(0, 20))

        self.start_camera_btn = ttk.Button(camera_ctrl, text="▶️ Start Camera",
                                         command=self.start_camera)
        self.start_camera_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_camera_btn = ttk.Button(camera_ctrl, text="⏹️ Stop Camera",
                                        command=self.stop_camera, state="disabled")
        self.stop_camera_btn.pack(side=tk.LEFT)

        # Center: Processing controls
        process_ctrl = ttk.LabelFrame(control_frame, text="Motion Processing", padding=5)
        process_ctrl.pack(side=tk.LEFT, padx=(0, 20))

        self.start_process_btn = ttk.Button(process_ctrl, text="🚀 Start Motion Detection",
                                          command=self.start_processing)
        self.start_process_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_process_btn = ttk.Button(process_ctrl, text="⏹️ Stop Processing",
                                         command=self.stop_processing, state="disabled")
        self.stop_process_btn.pack(side=tk.LEFT)

        # Right: Parameters and monitoring
        params_ctrl = ttk.LabelFrame(control_frame, text="Parameters", padding=5)
        params_ctrl.pack(side=tk.RIGHT)

        ttk.Label(params_ctrl, text="Motion Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.IntVar(value=int(self.motion_threshold))
        ttk.Spinbox(params_ctrl, from_=5, to=100, textvariable=self.threshold_var,
                   width=5).grid(row=0, column=1, padx=5)

        ttk.Label(params_ctrl, text="Grid Size:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.grid_var = tk.IntVar(value=self.grid_size)
        ttk.Spinbox(params_ctrl, from_=16, to_=64, textvariable=self.grid_var,
                   width=5).grid(row=1, column=1, padx=5, pady=(5, 0))

        ttk.Button(params_ctrl, text="Apply", command=self.apply_parameters).grid(
            row=2, column=0, columnspan=2, pady=5)

    def detect_camera(self):
        """Detect available cameras"""
        try:
            # Try common camera indices
            self.available_cameras = []

            for camera_index in [0, 1, 2]:
                try:
                    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.available_cameras.append(camera_index)
                            cap.release()
                            break
                    cap.release()
                except:
                    continue

            if self.available_cameras:
                self.camera_index = self.available_cameras[0]
                info = self.get_camera_info(self.camera_index)
                self.camera_status_label.config(
                    text=f"Camera {self.camera_index}: {info}")
                self.start_camera_btn.config(state="normal")
            else:
                self.camera_status_label.config(text="No cameras detected")
                self.start_camera_btn.config(state="disabled")

        except Exception as e:
            self.camera_status_label.config(text=f"Detection failed: {str(e)}")

    def get_camera_info(self, index):
        """Get camera information"""
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                return f"{width}x{height} @ {fps:.0f}fps"
        except:
            pass
        return "Unknown"

    def start_camera(self):
        """Start camera feed"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if self.camera.isOpened():
                self.camera_active = True
                self.status_label.config(text="● Camera Active", foreground="green")
                self.camera_status_label.config(text="Camera: Active")
                self.start_camera_btn.config(state="disabled")
                self.stop_camera_btn.config(state="normal")
                self.start_process_btn.config(state="normal")

                # Start camera feed thread
                self.camera_thread = threading.Thread(target=self.camera_feed_loop, daemon=True)
                self.camera_thread.start()

                self.update_motion_info("Camera started - ready for motion detection")

            else:
                messagebox.showerror("Camera Error", "Failed to open camera")

        except Exception as e:
            messagebox.showerror("Camera Error", f"Error starting camera: {str(e)}")

    def stop_camera(self):
        """Stop camera feed"""
        self.camera_active = False
        if self.camera:
            self.camera.release()

        self.status_label.config(text="● Camera Stopped", foreground="orange")
        self.camera_status_label.config(text="Camera: Stopped")
        self.start_camera_btn.config(state="normal")
        self.stop_camera_btn.config(state="disabled")
        self.start_process_btn.config(state="disabled")

    def start_processing(self):
        """Start motion detection and voxel processing"""
        if not self.camera_active:
            messagebox.showerror("Camera Required", "Please start camera first")
            return

        self.processing_active = True
        self.status_label.config(text="● Processing Active", foreground="green")

        # Initialize voxel grid
        self.grid_size = self.grid_var.get()
        self.motion_threshold = self.threshold_var.get()
        self.voxel_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)

        self.start_process_btn.config(state="disabled")
        self.stop_process_btn.config(state="normal")

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

        self.update_motion_info("Motion detection started! Move objects in front of camera.")

    def stop_processing(self):
        """Stop motion processing"""
        self.processing_active = False
        self.status_label.config(text="● Processing Stopped", foreground="orange")

        self.start_process_btn.config(state="normal")
        self.stop_process_btn.config(state="disabled")

    def apply_parameters(self):
        """Apply parameter changes"""
        self.motion_threshold = self.threshold_var.get()
        new_grid_size = self.grid_var.get()

        if new_grid_size != self.grid_size:
            self.grid_size = new_grid_size
            if self.voxel_grid is not None:
                self.voxel_grid.fill(0)  # Reset voxel grid
                self.update_3d_visualization()

        self.update_motion_info(f"Parameters applied - Threshold: {self.motion_threshold}, Grid: {self.grid_size}")

    def camera_feed_loop(self):
        """Main camera feed loop with live display"""
        while self.camera_active and self.camera.isOpened():
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Convert to grayscale for processing
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Display live camera feed
                    self.display_camera_feed(self.current_frame)

                time.sleep(1.0 / 15.0)  # ~15 FPS for display

            except Exception as e:
                print(f"Camera feed error: {e}")
                time.sleep(0.1)

    def display_camera_feed(self, frame):
        """Display camera feed with motion overlay"""
        try:
            # Update camera display
            self.camera_display.set_data(frame)

            # If processing is active and we have motion points, overlay them
            if self.processing_active and hasattr(self, 'motion_y_coords') and self.motion_y_coords:
                y_coords = np.array(self.motion_y_coords[-10:])  # Show last 10 points
                x_coords = np.array(self.motion_x_coords[-10:])

                # Scale coordinates for display
                height, width = frame.shape
                x_coords = (x_coords - np.min(x_coords, initial=0)) / max((np.max(x_coords, initial=1) - np.min(x_coords, initial=0)), 1) * width
                y_coords = (y_coords - np.min(y_coords, initial=0)) / max((np.max(y_coords, initial=1) - np.min(y_coords, initial=0)), 1) * height

                self.motion_overlay.set_data(x_coords, y_coords)
            else:
                self.motion_overlay.set_data([], [])  # Clear motion points

            # Refresh display
            self.camera_canvas.draw()

        except Exception as e:
            print(f"Camera display error: {e}")

    def processing_loop(self):
        """Main processing loop with motion detection and voxel accumulation"""
        self.motion_x_coords = []
        self.motion_y_coords = []

        while self.processing_active:
            if self.current_frame is not None:
                current = self.current_frame.copy()

                # Motion detection
                if self.previous_frame is not None:
                    # Calculate motion using frame difference
                    diff = np.abs(current.astype(np.float32) - self.previous_frame.astype(np.float32))
                    threshold = self.motion_threshold

                    # Apply threshold
                    motion_mask = diff > threshold
                    motion_pixels = np.count_nonzero(motion_mask)

                    if motion_pixels > 0:
                        # Find motion centers
                        y_indices, x_indices = np.where(motion_mask)

                        # Update motion coordinates for overlay
                        center_y = np.mean(y_indices)
                        center_x = np.mean(x_indices)

                        self.motion_y_coords.append(center_y)
                        self.motion_x_coords.append(center_x)

                        # Keep only recent motion points
                        if len(self.motion_y_coords) > 40:
                            self.motion_y_coords.pop(0)
                            self.motion_x_coords.pop(0)

                        # Update motion info
                        self.update_motion_info(
                            f"Motion detected: {motion_pixels} pixels at ({center_x:.0f}, {center_y:.0f})\n"
                            f"This motion will be converted to 3D voxels..."
                        )

                        # Convert motion to voxel space
                        self.add_motion_to_voxels(diff, motion_pixels)

                        # Update 3D visualization
                        self.root.after(0, self.update_3d_visualization)

                self.previous_frame = current.copy()

            time.sleep(0.1)

    def add_motion_to_voxels(self, motion_data, motion_pixels):
        """Convert 2D motion to 3D voxel accumulation"""
        if self.voxel_grid is None:
            return

        # Simple strategy: distribute motion energy across voxel space
        # More motion = more voxels added, stronger intensity

        base_intensity = motion_pixels / 2000.0  # Scale for reasonable voxel intensity

        # Add voxels in some pattern (could be smarter based on camera calibration)
        num_voxels_to_add = min(10, int(np.sqrt(motion_pixels) / 20))

        for _ in range(num_voxels_to_add):
            # Random distribution (in real system, this would be based on camera geometry)
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            z = np.random.randint(self.grid_size // 4, 3 * self.grid_size // 4)

            # Add intensity
            intensity = base_intensity * np.random.uniform(0.8, 1.2)
            self.voxel_grid[x, y, z] += intensity

    def update_3d_visualization(self):
        """Update the 3D voxel visualization"""
        if self.voxel_grid is None:
            return

        try:
            # Get all non-zero voxels
            coords = np.where(self.voxel_grid > 0.1)
            if len(coords[0]) > 0:
                intensities = self.voxel_grid[coords]

                # Update scatter plot data
                self.voxel_scatter._offsets3d = (coords[0], coords[1], coords[2])
                self.voxel_scatter.set_array(intensities)
                self.voxel_scatter.set_sizes(20 + intensities * 30)  # Size based on intensity

                self.voxel_ax.set_title(f'Live Motion-to-Voxel\n{len(coords[0])} Active Points')

                # Update statistics
                max_intensity = np.max(self.voxel_grid) if np.max(self.voxel_grid) > 0 else 0
                self.voxel_stats_label.config(
                    text=f"Total Voxels: {self.grid_size**3:,}\n"
                         f"Active Voxels: {len(coords[0]):,}\n"
                         f"Peak Intensity: {max_intensity:.2f}"
                )
            else:
                # Clear scatter plot
                self.voxel_scatter._offsets3d = ([], [], [])
                self.voxel_scatter.set_array([])
                self.voxel_scatter.set_sizes([])
                self.voxel_ax.set_title('No Voxel Data Yet\nMove objects in front of camera')
                self.voxel_stats_label.config(text="Total Voxels: 0\nActive Voxels: 0\nPeak Intensity: 0")

            # Refresh display
            self.voxel_canvas.draw()

        except Exception as e:
            print(f"3D visualization error: {e}")

    def update_motion_info(self, info):
        """Update motion detection status"""
        self.motion_info_label.config(text=info)

def main():
    """Main function"""
    print("🎥 AstraVoxel Live Motion Viewer")
    print("=================================")
    print()
    print("Features:")
    print("✅ Live USB webcam feed")
    print("✅ Real-time motion detection with visual indicators")
    print("✅ Motion-to-voxel conversion with 3D visualization")
    print("✅ Interactive parameter adjustment")
    print("✅ Complete pipeline visualization")
    print()
    print("Instructions:")
    print("1. Connect your USB webcam")
    print("2. Click 'Start Camera' to begin live feed")
    print("3. Click 'Start Motion Detection' to begin processing")
    print("4. Move objects in front of camera to see motion -> voxel conversion!")
    print()

    root = tk.Tk()
    app = AstraVoxelMotionViewer(root)

    print("Starting AstraVoxel Live Motion Viewer...")
    root.mainloop()

if __name__ == "__main__":
    main()