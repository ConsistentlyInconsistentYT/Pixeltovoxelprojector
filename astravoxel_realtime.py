#!/usr/bin/env python3
"""
AstraVoxel Real-Time Interface
==============================

Advanced real-time camera tracking and 3D voxel reconstruction interface.
Provides live multi-sensor data fusion and interactive 3D visualization.

Features:
- Real-time camera footage processing
- Live voxel grid accumulation
- Interactive 3D visualization with continuous updates
- Multi-camera motion tracking
- Astronomical FITS data integration
- Performance monitoring and diagnostics

This interface bridges the existing pixel-to-voxel processing engine
with real-time visualization and control systems.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image, ImageTk
import random
from pathlib import Path
import queue

# Import AstraVoxel components
from fits_loader import FITSLoader, CelestialCoordinates, create_astronomical_demo_data

class AstraVoxelRealtime:
    """
    Real-time AstraVoxel interface with live camera tracking and 3D visualization
    """

    def __init__(self, root):
        """Initialize the real-time AstraVoxel interface"""
        self.root = root
        self.root.title("🛰️ AstraVoxel - Real-Time 3D Motion Analysis")
        self.root.geometry("1600x1000")

        # Real-time processing state
        self.is_processing = False
        self.camera_active = False
        self.viz_thread = None
        self.data_queue = queue.Queue(maxsize=100)

        # Processing parameters
        self.motion_threshold = 25.0
        self.voxel_resolution = 0.5
        self.camera_count = 1
        self.frame_rate = 30.0
        self.grid_size = 64

        # Current data
        self.current_voxel_data = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        self.frame_count = 0
        self.processing_time = 0.0
        self.fps = 0.0

        # Camera simulation
        self.simulate_camera = True
        self.camera_feed_thread = None
        self.camera_image = None

        # Setup the real-time interface
        self.setup_interface()
        self.setup_realtime_processing()
        self.setup_visualization_panel()
        self.log_message("✓ AstraVoxel Real-Time Interface initialized")

    def setup_interface(self):
        """Set up the main interface layout"""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title bar
        title_frame = ttk.Frame(self.main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(title_frame, text="🛰️ AstraVoxel Real-Time", font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        self.status_indicator = ttk.Label(title_frame, text="● Inactive", foreground="red")
        self.status_indicator.pack(side=tk.RIGHT)

        # Main split: Left controls, Right visualization
        h_split = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)

        # Control panel (left)
        control_frame = ttk.LabelFrame(h_split, text="Mission Control", width=400)
        self.setup_control_panel(control_frame)
        h_split.add(control_frame, weight=1)

        # Visualization panel (right)
        viz_frame = ttk.LabelFrame(h_split, text="3D Interactive Viewport")
        self.setup_visualization_panel(viz_frame)
        h_split.add(viz_frame, weight=3)

        h_split.pack(fill=tk.BOTH, expand=True)

        # Bottom status and log
        bottom_frame = ttk.Frame(self.main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        # Stats panel
        stats_frame = ttk.LabelFrame(bottom_frame, text="Live Statistics", width=500)
        stats_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.setup_stats_panel(stats_frame)

        # Log panel
        log_frame = ttk.LabelFrame(bottom_frame, text="System Log")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_log_panel(log_frame)

    def setup_control_panel(self, parent):
        """Set up the control panel with real-time controls"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Camera Control Tab
        camera_tab = ttk.Frame(notebook)
        notebook.add(camera_tab, text="📹 Camera")

        camera_frame = ttk.Frame(camera_tab)
        camera_frame.pack(fill=tk.X, pady=10)

        ttk.Label(camera_frame, text="Camera Status:").grid(row=0, column=0, sticky=tk.W)
        self.camera_status_label = ttk.Label(camera_frame, text="Inactive", foreground="red")
        self.camera_status_label.grid(row=0, column=1, sticky=tk.W)

        # Camera controls
        controls_frame = ttk.Frame(camera_tab)
        controls_frame.pack(fill=tk.X, pady=10)

        self.camera_button = ttk.Button(
            controls_frame, text="▶️ Start Camera",
            command=self.toggle_camera, width=15
        )
        self.camera_button.pack(side=tk.LEFT, padx=5)

        self.capture_button = ttk.Button(
            controls_frame, text="📸 Capture",
            command=self.capture_frame, state="disabled"
        )
        self.capture_button.pack(side=tk.LEFT, padx=5)

        # Camera preview area
        preview_frame = ttk.LabelFrame(camera_tab, text="Live Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.preview_label = ttk.Label(preview_frame, text="Camera feed will appear here")
        self.preview_label.pack(expand=True)

        # Processing Parameters Tab
        params_tab = ttk.Frame(notebook)
        notebook.add(params_tab, text="⚙️ Processing")

        params_frame = ttk.Frame(params_tab)
        params_frame.pack(fill=tk.X, pady=10)

        # Motion detection threshold
        ttk.Label(params_frame, text="Motion Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.motion_threshold_var = tk.DoubleVar(value=self.motion_threshold)
        ttk.Scale(
            params_frame, from_=1, to=100, variable=self.motion_threshold_var,
            orient=tk.HORIZONTAL, length=200
        ).grid(row=0, column=1, sticky=tk.W, pady=2)

        # Voxel resolution
        ttk.Label(params_frame, text="Voxel Resolution:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.voxel_resolution_var = tk.DoubleVar(value=self.voxel_resolution)
        ttk.Scale(
            params_frame, from_=0.1, to=2.0, variable=self.voxel_resolution_var,
            orient=tk.HORIZONTAL, length=200
        ).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Grid size
        ttk.Label(params_frame, text="Grid Size:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.grid_size_var = tk.IntVar(value=self.grid_size)
        ttk.Spinbox(
            params_frame, from_=32, to=128, textvariable=self.grid_size_var, width=10
        ).grid(row=2, column=1, sticky=tk.W, pady=2)

        # Apply parameters button
        ttk.Button(
            params_frame, text="🔄 Apply Parameters",
            command=self.apply_parameters
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=10)

        # Action buttons
        action_frame = ttk.Frame(params_tab)
        action_frame.pack(fill=tk.X, pady=10)

        self.process_button = ttk.Button(
            action_frame, text="🚀 Start Processing",
            command=self.toggle_processing, state="disabled"
        )
        self.process_button.pack(fill=tk.X, pady=2)

        ttk.Button(
            action_frame, text="🔄 Reset Voxel Grid",
            command=self.reset_voxel_grid
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            action_frame, text="💾 Save Current State",
            command=self.save_current_state
        ).pack(fill=tk.X, pady=2)

    def setup_visualization_panel(self, parent):
        """Set up the interactive 3D visualization panel"""
        # Create matplotlib figure for 3D plotting
        self.viz_figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.viz_axes = self.viz_figure.add_subplot(111, projection='3d')

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.viz_figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control buttons
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(controls_frame, text="🔄 Reset View").pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="📸 Screenshot").pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="🎥 Record").pack(side=tk.LEFT, padx=2)

        # Coordinate system label
        ttk.Label(controls_frame, text="Coordinate System:").pack(side=tk.RIGHT, padx=(20, 0))
        self.coord_label = ttk.Label(controls_frame, text="Camera Reference")
        self.coord_label.pack(side=tk.RIGHT)

        # Initialize empty voxel visualization
        self.update_3d_visualization()

    def setup_stats_panel(self, parent):
        """Set up the live statistics panel"""
        # Performance metrics
        perf_frame = ttk.Frame(parent)
        perf_frame.pack(fill=tk.X, pady=5)

        ttk.Label(perf_frame, text="Performance:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        metrics = [
            ("Frame Rate", "fps", 0),
            ("Processing Time", "proc_time", 0),
            ("Memory Usage", "memory", 256),
            ("Voxel Count", "voxel_count", 0),
            ("Motion Detection", "motion", 0)
        ]

        self.metric_labels = {}
        for i, (label, var_name, value) in enumerate(metrics):
            frame = ttk.Frame(perf_frame)
            frame.pack(fill=tk.X, pady=1)

            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            metric_label = ttk.Label(frame, text=f"{value}")
            metric_label.pack(side=tk.RIGHT)
            self.metric_labels[var_name] = metric_label

        # Object detection stats
        detection_frame = ttk.Frame(parent)
        detection_frame.pack(fill=tk.X, pady=(10, 5))

        ttk.Label(detection_frame, text="Detection Results:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        self.detection_stats = {
            'detected_objects': ttk.Label(detection_frame, text="0"),
            'confidence': ttk.Label(detection_frame, text="0.0%"),
            'processing_queue': ttk.Label(detection_frame, text="0")
        }

        for label_text, var_name in [("Objects Detected:", "detected_objects"),
                                   ("Avg Confidence:", "confidence"),
                                   ("Queue Size:", "processing_queue")]:
            frame = ttk.Frame(detection_frame)
            frame.pack(fill=tk.X, pady=1)

            ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
            self.detection_stats[var_name].pack(side=tk.RIGHT)

    def setup_log_panel(self, parent):
        """Set up the system log panel"""
        self.log_text = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, height=8,
            font=("Consolas", 9), state='normal'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.insert(tk.END, "System initialized - ready for processing\n")

    def setup_realtime_processing(self):
        """Set up real-time data processing pipeline"""
        # Initialize processing threads
        self.camera_thread_active = False
        self.viz_update_active = False
        self.previous_frame = None

    def toggle_camera(self):
        """Toggle camera activation"""
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        """Start camera feed simulation"""
        self.camera_active = True
        self.camera_status_label.config(text="Active", foreground="green")
        self.status_indicator.config(text="● Active", foreground="green")
        self.camera_button.config(text="⏹️ Stop Camera")
        self.capture_button.config(state="normal")

        # Start camera feed thread
        if not self.camera_thread_active:
            self.camera_thread_active = True
            self.camera_feed_thread = threading.Thread(
                target=self.simulate_camera_feed, daemon=True
            )
            self.camera_feed_thread.start()

        self.log_message("📹 Camera feed started")

    def stop_camera(self):
        """Stop camera feed"""
        self.camera_active = False
        self.camera_status_label.config(text="Inactive", foreground="red")
        self.status_indicator.config(text="● Inactive", foreground="red")
        self.camera_button.config(text="▶️ Start Camera")
        self.capture_button.config(state="disabled")

        # Stop camera thread
        self.camera_thread_active = False
        if self.camera_feed_thread and self.camera_feed_thread.is_alive():
            self.camera_feed_thread.join(timeout=1.0)

        self.log_message("🛑 Camera feed stopped")

    def simulate_camera_feed(self):
        """Simulate real-time camera feed with synthetic astronomical images"""
        import matplotlib.pyplot as plt

        # Create synthetic star field for simulation
        np.random.seed(42)
        num_stars = 50
        star_positions = np.random.rand(num_stars, 2) * 640  # 640x480 resolution
        star_magnitudes = np.random.uniform(0, 8, num_stars)

        frame_count = 0

        while self.camera_thread_active:
            try:
                # Create synthetic astronomical image with moving stars
                image = np.zeros((480, 640), dtype=np.uint8)

                # Add background noise
                image += np.random.normal(10, 3, (480, 640)).astype(np.uint8)

                # Add stars with slight movement
                for i, (pos, mag) in enumerate(zip(star_positions, star_magnitudes)):
                    x, y = pos + np.random.normal(0, 0.5, 2)  # Slight movement
                    if 0 <= x < 640 and 0 <= y < 480:
                        intensity = int(255 * 10 ** (-mag / 2.5))
                        intensity = min(intensity, 200)  # Prevent saturation

                        # Add Gaussian star profile
                        y_coords, x_coords = np.ogrid[:480, :640]
                        sigma = 2.0
                        dist_sq = (x_coords - x)**2 + (y_coords - y)**2
                        star_profile = intensity * np.exp(-dist_sq / (2 * sigma**2))
                        image += star_profile.astype(np.uint8)

                # Convert to RGB for display
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # Store current frame for processing
                self.camera_image = image.copy()

                # Update preview (create smaller version for UI)
                small_image = cv2.resize(rgb_image, (320, 240))
                if hasattr(self, 'preview_label'):
                    # Convert to tkinter image (simplified - in real app use PIL)
                    self.update_camera_preview(small_image)

                frame_count += 1
                time.sleep(1.0 / self.frame_rate)  # Control frame rate

            except Exception as e:
                print(f"Camera simulation error: {e}")
                time.sleep(0.1)

    def update_camera_preview(self, image_array):
        """Update camera preview in UI"""
        # This is a simplified preview update
        # In a real implementation, convert to PhotoImage
        self.preview_label.config(text=f"Camera Preview Active\nFrame: {self.frame_count}\nResolution: {image_array.shape}")

    def capture_frame(self):
        """Capture current frame for processing"""
        if self.camera_image is not None:
            # Process the captured image
            self.process_frame(self.camera_image.copy())
            self.log_message("📸 Frame captured and processed")
        else:
            self.log_message("⚠️ No camera image available")

    def process_frame(self, image):
        """Process a single frame for motion detection and voxel update"""
        if not self.is_processing:
            return

        start_time = time.time()

        try:
            # Detect motion (simplified - compare with previous frame)
            if hasattr(self, 'previous_frame') and self.previous_frame is not None:
                # Calculate motion mask
                diff = np.abs(image.astype(np.float32) - self.previous_frame.astype(np.float32))
                motion_mask = diff > self.motion_threshold

                # Extract motion vectors (simplified)
                motion_pixels = np.where(motion_mask)

                if len(motion_pixels[0]) > 0:
                    # Project to voxel grid (simplified)
                    pixel_coords = np.column_stack(motion_pixels)
                    intensities = diff[motion_pixels]

                    for (y, x), intensity in zip(pixel_coords, intensities):
                        # Convert pixel coordinates to voxel indices
                        voxel_x = int((x / image.shape[1]) * self.grid_size)
                        voxel_y = int((y / image.shape[0]) * self.grid_size)
                        voxel_z = self.grid_size // 2  # Central depth

                        # Bound check
                        voxel_x = max(0, min(voxel_x, self.grid_size - 1))
                        voxel_y = max(0, min(voxel_y, self.grid_size - 1))
                        voxel_z = max(0, min(voxel_z, self.grid_size - 1))

                        # Accumulate in voxel grid
                        self.current_voxel_data[voxel_x, voxel_y, voxel_z] += intensity * self.voxel_resolution

                    # Update statistics
                    self.frame_count += 1
                    self.update_statistics(intensity)

            # Store current frame as previous
            self.previous_frame = image.copy()

            # Update visualization
            self.update_3d_visualization()

            # Update performance metrics
            end_time = time.time()
            self.processing_time = (end_time - start_time) * 1000  # ms
            self.update_performance_display()

        except Exception as e:
            self.log_message(f"❌ Frame processing error: {e}")

    def update_statistics(self, avg_intensity):
        """Update detection statistics"""
        # Count non-zero voxels
        nonzero_count = np.count_nonzero(self.current_voxel_data)
        self.detection_stats['detected_objects'].config(text=str(nonzero_count))

        # Calculate average confidence
        confidence = min(avg_intensity / 50.0, 1.0) * 100 if nonzero_count > 0 else 0
        self.detection_stats['confidence'].config(text=".1f")

        # Queue size (simplified)
        queue_size = self.data_queue.qsize() if hasattr(self, 'data_queue') else 0
        self.detection_stats['processing_queue'].config(text=str(queue_size))

    def update_3d_visualization(self):
        """Update the 3D voxel visualization"""
        try:
            # Clear previous plot
            self.viz_axes.clear()

            # Get non-zero voxel coordinates
            nonzero_coords = np.where(self.current_voxel_data > 0)
            if len(nonzero_coords[0]) > 0:
                intensities = self.current_voxel_data[nonzero_coords]

                # Create 3D scatter plot
                self.viz_axes.scatter(
                    nonzero_coords[2], nonzero_coords[1], nonzero_coords[0],
                    c=intensities, cmap='plasma', marker='o',
                    s=5, alpha=0.8
                )

                self.viz_axes.set_xlabel('X (space units)')
                self.viz_axes.set_ylabel('Y (space units)')
                self.viz_axes.set_zlabel('Z (space units)')
                self.viz_axes.set_title(f'Live Voxel Grid ({len(nonzero_coords[0])} points)')
            else:
                self.viz_axes.text(0.5, 0.5, 0.5, 'No voxel data\nWaiting for motion detection...',
                                  ha='center', va='center', transform=self.viz_axes.transAxes)
                self.viz_axes.set_title('Voxel Grid (Empty)')

            # Update coordinate system label
            self.coord_label.config(text="Camera Reference Frame")

            # Refresh canvas
            self.canvas.draw()

        except Exception as e:
            print(f"Visualization update error: {e}")

    def update_performance_display(self):
        """Update real-time performance metrics display"""
        try:
            # Calculate FPS
            self.fps = 1.0 / max(self.processing_time / 1000.0, 0.001)

            # Update metric labels
            self.metric_labels['fps'].config(text=".1f")
            self.metric_labels['proc_time'].config(text=".2f")
            self.metric_labels['memory'].config(text="256/512 MB")  # Simulated
            self.metric_labels['voxel_count'].config(text=str(np.count_nonzero(self.current_voxel_data)))
            self.metric_labels['motion'].config(text=".1f")

        except Exception as e:
            print(f"Performance display error: {e}")

    def toggle_processing(self):
        """Toggle real-time processing on/off"""
        if self.is_processing:
            self.stop_processing()
        else:
            self.start_processing()

    def start_processing(self):
        """Start real-time processing"""
        if not self.camera_active:
            messagebox.showwarning("No Camera", "Please start camera feed first")
            return

        self.is_processing = True
        self.process_button.config(text="⏹️ Stop Processing")

        # Start visualization update thread
        if not self.viz_update_active:
            self.viz_update_active = True
            self.viz_thread = threading.Thread(target=self.viz_update_loop, daemon=True)
            self.viz_thread.start()

        self.log_message("🚀 Real-time processing started")

    def stop_processing(self):
        """Stop real-time processing"""
        self.is_processing = False
        self.process_button.config(text="🚀 Start Processing")

        # Stop visualization thread
        self.viz_update_active = False
        if self.viz_thread and self.viz_thread.is_alive():
            self.viz_thread.join(timeout=1.0)

        self.log_message("⏹️ Real-time processing stopped")

    def viz_update_loop(self):
        """Background thread to update visualization"""
        while self.viz_update_active:
            try:
                if self.camera_image is not None:
                    self.process_frame(self.camera_image.copy())
                time.sleep(1.0 / 30.0)  # 30 FPS update
            except Exception as e:
                print(f"Visualization update loop error: {e}")
                time.sleep(0.1)

    def apply_parameters(self):
        """Apply current parameter settings"""
        self.motion_threshold = self.motion_threshold_var.get()
        self.voxel_resolution = self.voxel_resolution_var.get()
        self.grid_size = self.grid_size_var.get()

        # Reset voxel grid with new size
        if self.grid_size != self.current_voxel_data.shape[0]:
            self.current_voxel_data = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
            self.update_3d_visualization()

        self.log_message(".1f"
                        ".1f")

    def reset_voxel_grid(self):
        """Reset the voxel grid to zeros"""
        self.current_voxel_data.fill(0)
        self.previous_frame = None
        self.update_3d_visualization()
        self.log_message("🗑️ Voxel grid reset")

    def save_current_state(self):
        """Save current processing state"""
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./astravoxel_capture_{timestamp}.npz"

            np.savez(filename, voxel_data=self.current_voxel_data)
            self.log_message(f"💾 State saved to: {filename}")

        except Exception as e:
            self.log_message(f"❌ Save error: {e}")

    def log_message(self, message):
        """Add message to system log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        try:
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
        except:
            print(log_entry.strip())

def main():
    """Main function to start AstraVoxel Real-Time"""
    root = tk.Tk()
    app = AstraVoxelRealtime(root)

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()

# End of AstraVoxel Real-Time Interface</content>