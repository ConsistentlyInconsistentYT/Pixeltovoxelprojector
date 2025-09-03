#!/usr/bin/env python3
"""
AstraVoxel Simple Webcam Demo
=============================

Clean, simple version that focuses on basic USB webcam usage.
Avoids complex camera systems and focuses on essential functionality.

Features:
- Simple USB webcam detection
- Basic motion detection
- 3D voxel visualization
- No advanced camera systems (just standard webcams)
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

class AstraVoxelSimpleWebcam:
    """
    Simple webcam demo focusing on basic functionality
    """

    def __init__(self, root):
        self.root = root
        self.root.title("AstraVoxel Webcam Demo")
        self.root.geometry("1000x700")

        # Camera and processing state
        self.camera = None
        self.camera_active = False
        self.processing_active = False
        self.previous_frame = None
        self.voxel_grid = None
        self.grid_size = 24

        # Create simple interface
        self.setup_interface()
        self.detect_simple_camera()

    def setup_interface(self):
        """Set up simple interface"""
        # Title
        title_label = ttk.Label(self.root, text="🎥 AstraVoxel Webcam Demo",
                               font=("Segoe UI", 14, "bold"))
        title_label.pack(pady=10)

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        self.setup_controls(left_panel)

        # Right panel - Visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_visualization(right_panel)

    def setup_controls(self, parent):
        """Set up control panel"""
        # Camera section
        camera_frame = ttk.LabelFrame(parent, text="Camera", padding=10)
        camera_frame.pack(fill=tk.X, pady=(0, 20))

        self.camera_status_label = ttk.Label(camera_frame,
                                           text="Detecting camera...",
                                           foreground="orange")
        self.camera_status_label.pack(anchor=tk.W)

        self.start_camera_btn = ttk.Button(camera_frame,
                                         text="▶️ Connect Camera",
                                         command=self.start_simple_camera)
        self.start_camera_btn.pack(fill=tk.X, pady=5)

        self.stop_camera_btn = ttk.Button(camera_frame,
                                        text="⏹️ Stop Camera",
                                        command=self.stop_simple_camera,
                                        state="disabled")
        self.stop_camera_btn.pack(fill=tk.X, pady=5)

        # Processing section
        process_frame = ttk.LabelFrame(parent, text="Motion Processing", padding=10)
        process_frame.pack(fill=tk.X, pady=(0, 20))

        self.start_process_btn = ttk.Button(process_frame,
                                          text="🚀 Start Processing",
                                          command=self.start_processing)
        self.start_process_btn.pack(fill=tk.X, pady=5)

        self.stop_process_btn = ttk.Button(process_frame,
                                         text="⏹️ Stop Processing",
                                         command=self.stop_processing)
        self.stop_process_btn.pack(fill=tk.X, pady=5)

        # Parameters
        param_frame = ttk.LabelFrame(parent, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X)

        ttk.Label(param_frame, text="Motion Threshold:").pack(anchor=tk.W)
        self.threshold_var = tk.IntVar(value=25)
        threshold_scale = ttk.Scale(param_frame, from_=5, to=100,
                                  variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(param_frame, text="Grid Size:").pack(anchor=tk.W)
        self.grid_size_var = tk.IntVar(value=self.grid_size)
        grid_scale = ttk.Scale(param_frame, from_=16, to_=48,
                             variable=self.grid_size_var, orient=tk.HORIZONTAL)
        grid_scale.pack(fill=tk.X)

        # Status
        self.control_status_label = ttk.Label(parent,
                                            text="Ready to use",
                                            foreground="green")
        self.control_status_label.pack(pady=(20, 0))

    def setup_visualization(self, parent):
        """Set up 3D visualization"""
        viz_frame = ttk.LabelFrame(parent, text="3D Motion Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Create 3D plot
        self.figure = plt.Figure(figsize=(6, 5))
        self.ax = self.figure.add_subplot(111, projection='3d')

        # Initialize empty plot
        self.update_3d_visualization()

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status
        self.viz_status_label = ttk.Label(viz_frame, text="No voxel data")
        self.viz_status_label.pack(pady=(5, 0))

    def detect_simple_camera(self):
        """Simple camera detection"""
        try:
            # Try to find a basic webcam (usually index 0 or 1)
            for camera_index in [0, 1]:
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow backend

                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()

                    if ret and frame is not None:
                        self.available_camera_index = camera_index
                        self.camera_status_label.config(
                            text=f"Camera {camera_index} detected",
                            foreground="green"
                        )
                        self.control_status_label.config(
                            text="Camera detected - ready to use",
                            foreground="green"
                        )
                        return True

            # No camera found
            self.camera_status_label.config(
                text="No camera detected",
                foreground="red"
            )
            self.control_status_label.config(
                text="Please connect a USB webcam",
                foreground="red"
            )
            return False

        except Exception as e:
            self.camera_status_label.config(
                text="Camera detection failed",
                foreground="red"
            )
            return False

    def start_simple_camera(self):
        """Start webcam"""
        try:
            # Use CAP_DSHOW backend to avoid conflicts
            self.camera = cv2.VideoCapture(self.available_camera_index, cv2.CAP_DSHOW)

            if self.camera.isOpened():
                self.camera_active = True
                self.camera_status_label.config(
                    text="Camera Active",
                    foreground="green"
                )
                self.start_camera_btn.config(state="disabled")
                self.stop_camera_btn.config(state="normal")
                self.control_status_label.config(
                    text="Camera started - ready for motion detection",
                    foreground="green"
                )

                # Start camera preview thread
                self.camera_thread = threading.Thread(target=self.camera_preview_loop, daemon=True)
                self.camera_thread.start()

            else:
                messagebox.showerror("Camera Error", "Failed to open camera")

        except Exception as e:
            messagebox.showerror("Camera Error", f"Error starting camera: {e}")

    def stop_simple_camera(self):
        """Stop camera"""
        self.camera_active = False
        if self.camera:
            self.camera.release()

        self.camera_status_label.config(
            text="Camera Stopped",
            foreground="orange"
        )
        self.start_camera_btn.config(state="normal")
        self.stop_camera_btn.config(state="disabled")
        self.control_status_label.config(
            text="Camera stopped",
            foreground="orange"
        )

    def camera_preview_loop(self):
        """Simple camera preview"""
        while self.camera_active and self.camera.isOpened():
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Convert to grayscale for processing
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                print(f"Camera preview error: {e}")
                break

    def start_processing(self):
        """Start motion processing"""
        if not self.camera_active:
            messagebox.showerror("Camera Required", "Please start camera first")
            return

        self.processing_active = True
        self.start_process_btn.config(state="disabled")
        self.stop_process_btn.config(state="normal")

        # Initialize voxel grid
        self.grid_size = self.grid_size_var.get()
        self.voxel_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)

        self.control_status_label.config(
            text="Processing motion...",
            foreground="blue"
        )

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

    def stop_processing(self):
        """Stop motion processing"""
        self.processing_active = False
        self.start_process_btn.config(state="normal")
        self.stop_process_btn.config(state="disabled")
        self.control_status_label.config(
            text="Processing stopped",
            foreground="orange"
        )

    def processing_loop(self):
        """Motion detection and voxel accumulation"""
        while self.processing_active:
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                current_frame = self.current_frame.copy()

                if self.previous_frame is not None:
                    # Simple motion detection
                    diff = np.abs(current_frame.astype(np.float32) -
                                self.previous_frame.astype(np.float32))

                    threshold = self.threshold_var.get()
                    motion_mask = diff > threshold
                    motion_pixels = np.count_nonzero(motion_mask)

                    # If significant motion detected, add to voxel grid
                    if motion_pixels > 100:  # Minimum motion threshold
                        self.add_motion_to_voxels(motion_pixels)

                        # Update visualization occasionally
                        self.root.after(0, self.update_3d_visualization)

                self.previous_frame = current_frame

            time.sleep(0.1)

    def add_motion_to_voxels(self, motion_intensity):
        """Simple voxel accumulation"""
        if self.voxel_grid is None:
            return

        # Add voxels in a simple pattern
        base_intensity = motion_intensity / 5000.0

        # Add some structure by distributing motion across grid
        for _ in range(5):  # Add multiple points per motion event
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            z = np.random.randint(self.grid_size // 4, 3 * self.grid_size // 4)

            self.voxel_grid[x, y, z] += base_intensity * np.random.uniform(0.5, 1.5)

    def update_3d_visualization(self):
        """Update 3D visualization"""
        if self.voxel_grid is None:
            return

        try:
            self.ax.clear()

            # Get active voxels
            coords = np.where(self.voxel_grid > 0.1)
            if len(coords[0]) > 0:
                intensities = self.voxel_grid[coords]

                self.ax.scatter(
                    coords[0], coords[1], coords[2],
                    c=intensities, cmap='plasma', marker='o', s=10, alpha=0.8
                )

                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Y')
                self.ax.set_zlabel('Z')
                self.ax.set_title(f'Motion Detection Results\n{len(coords[0])} Points')

                self.viz_status_label.config(text=f"Active Voxels: {len(coords[0])}")

            else:
                self.ax.text(0.5, 0.5, 0.5, 'Waiting for motion...\nMove objects in front of camera',
                           ha='center', va='center', transform=self.ax.transAxes)
                self.ax.set_title('3D Motion Space')
                self.viz_status_label.config(text="No motion detected yet")

            # Set axis limits
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_zlim(0, self.grid_size)

            self.canvas.draw()

        except Exception as e:
            print(f"Visualization error: {e}")

def main():
    """Main function"""
    print("🎥 AstraVoxel Simple Webcam Demo")
    print("===================================")
    print("This version focuses on basic USB webcam functionality.")
    print("It avoids complex camera systems and provides simple motion detection.")
    print()
    print("Requirements:")
    print("- Install opencv-python if needed: pip install opencv-python")
    print("- Use with standard USB webcams only")
    print()

    root = tk.Tk()
    app = AstraVoxelSimpleWebcam(root)

    print("Starting AstraVoxel Simple Webcam Demo...")
    root.mainloop()

if __name__ == "__main__":
    main()