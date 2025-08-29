#!/usr/bin/env python3
"""
GUI Interface for Pixeltovoxelprojector
=======================================

Optional graphical user interface for the Pixel-to-Voxel Projector.
Provides an elegant GUI alternative to command-line usage while maintaining
full backward compatibility.

Requirements:
- tkinter (included with Python)
- tkinter.ttk for modern widgets
- numpy for NPY file handling
- Optional: Pillow for enhanced image display
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import sys
import os
import json
from pathlib import Path
import numpy as np

class PixeltovoxelGUI:
    """
    Graphical User Interface for Pixeltovoxelprojector.

    Provides an intuitive way to:
    - Run the demo with configurable parameters
    - Generate visualizations
    - Monitor progress and results
    - View output files and folders
    """

    def __init__(self, root):
        """Initialize the GUI with all components."""
        self.root = root
        self.root.title("Pixel-to-Voxel Projector")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # Set window icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass  # Icon file not present, skip

        # Create main frames
        self.setup_frames()

        # Initialize variables
        self.setup_variables()

        # Create UI components
        self.create_widgets()

        # Bind events
        self.bind_events()

        # Load any existing configuration
        self.load_config()

        # Start status monitoring
        self.update_status()

        # Center window
        self.center_window()

    def setup_frames(self):
        """Set up the main frame structure."""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        self.main_frame.rowconfigure(3, weight=1)

        # Camera footage frame
        self.camera_frame = ttk.LabelFrame(
            self.main_frame,
            text="📹 Camera Footage Input",
            padding="10"
        )
        self.camera_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))

        # Title
        self.title_label = ttk.Label(
            self.main_frame,
            text="🛰️ Pixel-to-Voxel Projector",
            font=("Segoe UI", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Control panel frame
        self.control_frame = ttk.LabelFrame(
            self.main_frame,
            text="Control Panel",
            padding="10"
        )
        self.control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 5))

        # Output panel frame
        self.output_frame = ttk.LabelFrame(
            self.main_frame,
            text="Output & Results",
            padding="10"
        )
        self.output_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 5))

        # NPY Viewer panel frame (initially hidden)
        self.npy_viewer_frame = ttk.LabelFrame(
            self.main_frame,
            text="NPY File Viewer 🔽",
            padding="10"
        )
        # Configure viewer to be collapsible - starts collapsed

        # Log panel frame
        self.log_frame = ttk.LabelFrame(
            self.main_frame,
            text="Execution Log",
            padding="5"
        )
        self.log_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.S), pady=(10, 0))

        # NPY viewer visibility toggle
        self.npy_viewer_visible = False
        self.npy_expanded = False

    def setup_variables(self):
        """Initialize GUI variables."""
        self.demo_running = False
        self.viz_running = False
        self.current_process = None

        # Configuration variables
        self.star_count = tk.IntVar(value=100)
        self.image_width = tk.IntVar(value=1024)
        self.image_height = tk.IntVar(value=768)
        self.voxel_size = tk.IntVar(value=50)
        self.fov_degrees = tk.DoubleVar(value=45.0)
        self.voxel_range = tk.IntVar(value=1000)

        # Auto-save results
        self.auto_save = tk.BooleanVar(value=True)
        self.show_advanced = tk.BooleanVar(value=False)

        # NPY viewer variables
        self.npy_viewer_visible = False
        self.npy_expanded = False
        self.current_npy_file = None

        # Camera footage variables
        self.camera_images_path = None
        self.metadata_path = None
        self.camera_data = None
        self.image_files = []

    def create_widgets(self):
        """Create all GUI widgets."""
        self.create_control_panel()
        self.create_output_panel()
        self.create_camera_footage_panel()
        self.create_npy_viewer_panel()
        self.create_log_panel()

    def create_control_panel(self):
        """Create the control panel with parameter inputs and buttons."""
        # Parameter inputs
        ttk.Label(self.control_frame, text="Basic Parameters", font=("Segoe UI", 10, "bold")) \
            .grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

        # Star count
        ttk.Label(self.control_frame, text="Number of Stars:").grid(row=1, column=0, sticky=tk.W)
        self.star_spinbox = ttk.Spinbox(
            self.control_frame, from_=10, to=500, textvariable=self.star_count, width=8
        )
        self.star_spinbox.grid(row=1, column=1, sticky=tk.W, pady=2)

        # Image resolution
        ttk.Label(self.control_frame, text="Image Resolution:").grid(row=2, column=0, sticky=tk.W)
        self.res_frame = ttk.Frame(self.control_frame)
        self.res_frame.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(self.res_frame, text="W:").pack(side=tk.LEFT)
        self.width_spinbox = ttk.Spinbox(
            self.res_frame, from_=256, to=2048, textvariable=self.image_width, width=5
        )
        self.width_spinbox.pack(side=tk.LEFT)
        ttk.Label(self.res_frame, text="H:").pack(side=tk.LEFT)
        self.height_spinbox = ttk.Spinbox(
            self.res_frame, from_=256, to=1536, textvariable=self.image_height, width=5
        )
        self.height_spinbox.pack(side=tk.LEFT)

        # Advanced parameters toggle
        self.advanced_check = ttk.Checkbutton(
            self.control_frame, text="Show Advanced Parameters",
            variable=self.show_advanced, command=self.toggle_advanced
        )
        self.advanced_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        # Advanced parameters frame (initially hidden)
        self.advanced_frame = ttk.Frame(self.control_frame)

        ttk.Label(self.advanced_frame, text="Advanced Settings", font=("Segoe UI", 9, "bold")) \
            .grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # Voxel grid size
        ttk.Label(self.advanced_frame, text="Voxel Grid Size:").grid(row=1, column=0, sticky=tk.W)
        self.voxel_spinbox = ttk.Spinbox(
            self.advanced_frame, from_=20, to=100, textvariable=self.voxel_size, width=6
        )
        self.voxel_spinbox.grid(row=1, column=1, sticky=tk.W, pady=2)

        # Field of view
        ttk.Label(self.advanced_frame, text="Camera FOV (°):").grid(row=2, column=0, sticky=tk.W)
        self.fov_spinbox = ttk.Spinbox(
            self.advanced_frame, from_=10, to=180, textvariable=self.fov_degrees,
            increment=5, format="%.1f", width=6
        )
        self.fov_spinbox.grid(row=2, column=1, sticky=tk.W, pady=2)

        # Voxel range
        ttk.Label(self.advanced_frame, text="Spatial Range (±AU):").grid(row=3, column=0, sticky=tk.W)
        self.range_spinbox = ttk.Spinbox(
            self.advanced_frame, from_=500, to=5000, textvariable=self.voxel_range, width=6
        )
        self.range_spinbox.grid(row=3, column=1, sticky=tk.W, pady=2)

        # Auto-save results
        self.autosave_check = ttk.Checkbutton(
            self.advanced_frame, text="Auto-save Results",
            variable=self.auto_save
        )
        self.autosave_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        # Action buttons
        self.button_frame = ttk.Frame(self.control_frame)
        self.button_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))

        # Main action buttons
        self.demo_button = ttk.Button(
            self.button_frame, text="🚀 Run Demo",
            command=self.run_demo
        )
        self.demo_button.pack(side=tk.LEFT, padx=(0, 5))

        self.viz_button = ttk.Button(
            self.button_frame, text="📊 Generate Visualizations",
            command=self.run_visualization
        )
        self.viz_button.pack(side=tk.LEFT, padx=(0, 5))

        # Utility buttons
        self.util_frame = ttk.Frame(self.control_frame)
        self.util_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))

        ttk.Button(self.util_frame, text="📂 Open Output Folder",
                  command=self.open_output_folder).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(self.util_frame, text="🔄 Refresh Status",
                  command=self.update_status).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(self.util_frame, text="ℹ️ Help",
                  command=self.show_help).pack(side=tk.LEFT)

        # Progress indicator
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.control_frame, orient="horizontal",
            length=200, mode="determinate",
            variable=self.progress_var
        )
        self.progress_bar.grid(row=6, column=0, columnspan=2, pady=(10, 0))

    def create_output_panel(self):
        """Create the output panel showing results and status."""
        # Status indicators
        self.status_frame = ttk.Frame(self.output_frame)
        self.status_frame.pack(fill=tk.X, pady=(0, 10))

        # Build status
        ttk.Label(self.status_frame, text="Build Status:").pack(side=tk.LEFT)
        self.build_status = ttk.Label(
            self.status_frame, text="⚪ Checking...",
            foreground="orange"
        )
        self.build_status.pack(side=tk.LEFT, padx=(5, 15))

        # Demo data status
        ttk.Label(self.status_frame, text="Demo Data:").pack(side=tk.LEFT)
        self.demo_status = ttk.Label(
            self.status_frame, text="⚪ N/A",
            foreground="gray"
        )
        self.demo_status.pack(side=tk.LEFT, padx=(5, 15))

        # Visualization status
        ttk.Label(self.status_frame, text="Visualizations:").pack(side=tk.LEFT)
        self.viz_status = ttk.Label(
            self.status_frame, text="⚪ N/A",
            foreground="gray"
        )
        self.viz_status.pack(side=tk.LEFT)

        # Results tree view
        tree_frame = ttk.Frame(self.output_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        ttk.Label(tree_frame, text="Generated Files:").pack(anchor=tk.W)

        # File list tree
        self.file_tree = ttk.Treeview(tree_frame, height=8)
        self.file_tree.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for tree
        tree_scrollbar = ttk.Scrollbar(
            tree_frame, orient="vertical",
            command=self.file_tree.yview
        )
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.configure(yscrollcommand=tree_scrollbar.set)

        # Configure tree columns
        self.file_tree.heading("#0", text="File")

        # Bind double-click event
        self.file_tree.bind("<Double-1>", self.open_file)

    def create_camera_footage_panel(self):
        """Create the camera footage input panel."""
        # Configure grid for camera frame
        self.camera_frame.columnconfigure(0, weight=1)
        self.camera_frame.columnconfigure(1, weight=1)
        self.camera_frame.columnconfigure(2, weight=2)

        # Title and description
        ttk.Label(self.camera_frame, text="Load Camera Footage for Motion Tracking", font=("Segoe UI", 10, "bold")) \
            .grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        # Folder selection buttons
        self.images_btn = ttk.Button(
            self.camera_frame, text="📁 Select Images Folder",
            command=self.select_images_folder
        )
        self.images_btn.grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)

        self.metadata_btn = ttk.Button(
            self.camera_frame, text="📋 Select Metadata File",
            command=self.select_metadata_file
        )
        self.metadata_btn.grid(row=1, column=1, sticky=tk.W, padx=(0, 5), pady=2)

        # Status indicators
        self.images_path_label = ttk.Label(self.camera_frame, text="Images: None", foreground="red")
        self.images_path_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.metadata_path_label = ttk.Label(self.camera_frame, text="Metadata: None", foreground="red")
        self.metadata_path_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Validation and preview buttons
        self.validate_btn = ttk.Button(
            self.camera_frame, text="✅ Validate Metadata",
            command=self.validate_metadata, state="disabled"
        )
        self.validate_btn.grid(row=4, column=0, sticky=tk.W, pady=(10, 5))

        self.preview_btn = ttk.Button(
            self.camera_frame, text="👁️ Preview Sequence",
            command=self.preview_sequence, state="disabled"
        )
        self.preview_btn.grid(row=4, column=1, sticky=tk.W, pady=(10, 5))

        # Clear button
        ttk.Button(self.camera_frame, text="🗑️ Clear Data",
                  command=self.clear_camera_data).grid(row=4, column=2, sticky=tk.W, pady=(10, 5))

        # Metadata display frame
        self.metadata_display_frame = ttk.LabelFrame(self.camera_frame, text="📊 Camera Metadata", padding="5")
        self.metadata_display_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))

        # Configure metadata display
        self.metadata_display_frame.columnconfigure(0, weight=1)
        self.metadata_display_frame.columnconfigure(1, weight=1)

        # Metadata display widgets
        self.metadata_text = scrolledtext.ScrolledText(
            self.metadata_display_frame,
            wrap=tk.WORD,
            height=8,
            state='disabled',
            font=("Consolas", 9)
        )
        self.metadata_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Statistics labels
        ttk.Label(self.metadata_display_frame, text="Frames:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.frame_count_label = ttk.Label(self.metadata_display_frame, text="0")
        self.frame_count_label.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(self.metadata_display_frame, text="Cameras:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.camera_count_label = ttk.Label(self.metadata_display_frame, text="0")
        self.camera_count_label.grid(row=2, column=1, sticky=tk.W)

        ttk.Label(self.metadata_display_frame, text="Status:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.metadata_status_label = ttk.Label(self.metadata_display_frame, text="Not loaded", foreground="red")
        self.metadata_status_label.grid(row=3, column=1, sticky=tk.W)

    def select_images_folder(self):
        """Select folder containing camera images."""
        folder_path = filedialog.askdirectory(title="Select Images Folder")
        if folder_path:
            self.camera_images_path = Path(folder_path)
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            self.image_files = [
                f for f in self.camera_images_path.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            self.image_files.sort()
            self.images_path_label.config(
                text=f"Images: {self.camera_images_path.name} ({len(self.image_files)} files)",
                foreground="green"
            )
            self.log_message(f"✓ Loaded {len(self.image_files)} images from {self.camera_images_path.name}")

            # Enable validate button if both paths are set
            if self.metadata_path:
                self.validate_btn.config(state="normal")
                self.preview_btn.config(state="normal")
            self.update_status()

    def select_metadata_file(self):
        """Select metadata.json file."""
        file_path = filedialog.askopenfilename(
            title="Select Metadata File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.metadata_path = Path(file_path)
            self.metadata_path_label.config(
                text=f"Metadata: {self.metadata_path.name}",
                foreground="green"
            )
            self.log_message(f"✓ Selected metadata file: {self.metadata_path.name}")

            # Enable validate button if both paths are set
            if self.camera_images_path:
                self.validate_btn.config(state="normal")
                self.preview_btn.config(state="normal")

    def validate_metadata(self):
        """Validate and parse metadata.json file."""
        try:
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)

            # Validate data structure
            if not isinstance(data, list):
                raise ValueError("Metadata must be a list of camera entries")

            if len(data) == 0:
                raise ValueError("No camera entries found in metadata")

            # Validate each entry
            required_fields = ["camera_index", "frame_index", "camera_position",
                             "yaw", "pitch", "roll", "image_file"]

            validated_data = []
            cameras = set()
            frames = set()

            for i, entry in enumerate(data):
                for field in required_fields:
                    if field not in entry:
                        raise ValueError(f"Missing field '{field}' in entry {i+1}")

                if not isinstance(entry["camera_position"], list) or len(entry["camera_position"]) != 3:
                    raise ValueError(f"camera_position must be a 3-element list in entry {i+1}")

                cameras.add(entry["camera_index"])
                frames.add(entry["frame_index"])
                validated_data.append(entry)

            self.camera_data = validated_data

            # Update UI
            self.frame_count_label.config(text=str(len(frames)))
            self.camera_count_label.config(text=str(len(cameras)))
            self.metadata_status_label.config(text="Valid ✓", foreground="green")

            # Display metadata in text area
            self.metadata_text.config(state='normal')
            self.metadata_text.delete(1.0, tk.END)
            self.metadata_text.insert(tk.END, json.dumps(validated_data[:5], indent=2))  # Show first 5 entries
            if len(validated_data) > 5:
                self.metadata_text.insert(tk.END, f"\n[... and {len(validated_data)-5} more entries]")
            self.metadata_text.config(state='disabled')

            self.log_message(f"✓ Validated metadata: {len(cameras)} cameras, {len(frames)} frames")

        except Exception as e:
            error_msg = f"Metadata validation error: {str(e)}"
            messagebox.showerror("Metadata Error", error_msg)
            self.metadata_status_label.config(text="Invalid ✗", foreground="red")
            self.log_message(f"❌ {error_msg}")

    def preview_sequence(self):
        """Preview the image sequence."""
        if not self.image_files:
            messagebox.showwarning("Preview Error", "No image files loaded")
            return

        try:
            from PIL import Image

            # Create a simple preview window
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Image Sequence Preview")
            preview_window.geometry("800x600")

            # Create canvas and scrollbar
            canvas = tk.Canvas(preview_window)
            scrollbar = ttk.Scrollbar(preview_window, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Add thumbnail images
            max_images = min(20, len(self.image_files))  # Limit to 20 thumbnails
            for i, img_file in enumerate(self.image_files[:max_images]):
                try:
                    img = Image.open(img_file)
                    img.thumbnail((150, 150))
                    photo = tk.PhotoImage(file=str(img_file))

                    frame = ttk.Frame(scrollable_frame, borderwidth=1, relief="sunken")
                    frame.grid(row=i//4, column=i%4, padx=5, pady=5)

                    img_label = ttk.Label(frame, image=photo)
                    img_label.image = photo  # Keep reference
                    img_label.pack()

                    name_label = ttk.Label(frame, text=img_file.name, font=("Segoe UI", 8))
                    name_label.pack(pady=(2, 0))

                except Exception as img_error:
                    error_frame = ttk.Frame(scrollable_frame, borderwidth=1, relief="sunken")
                    error_frame.grid(row=i//4, column=i%4, padx=5, pady=5)
                    ttk.Label(error_frame, text=f"Error loading: {img_file.name}",
                             foreground="red", font=("Segoe UI", 8)).pack()

            # Layout
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            self.log_message(f"✓ Opened image sequence preview ({max_images} thumbnails)")

        except ImportError:
            messagebox.showinfo("Preview Info", "PIL (Pillow) not installed. Install with: pip install Pillow")
        except Exception as e:
            error_msg = f"Preview error: {str(e)}"
            messagebox.showerror("Preview Error", error_msg)
            self.log_message(f"❌ {error_msg}")

    def clear_camera_data(self):
        """Clear all camera data and reset UI."""
        self.camera_images_path = None
        self.metadata_path = None
        self.camera_data = None
        self.image_files = []

        # Reset UI
        self.images_path_label.config(text="Images: None", foreground="red")
        self.metadata_path_label.config(text="Metadata: None", foreground="red")

        self.validate_btn.config(state="disabled")
        self.preview_btn.config(state="disabled")

        self.frame_count_label.config(text="0")
        self.camera_count_label.config(text="0")
        self.metadata_status_label.config(text="Not loaded", foreground="red")

        self.metadata_text.config(state='normal')
        self.metadata_text.delete(1.0, tk.END)
        self.metadata_text.config(state='disabled')

        self.log_message("✓ Cleared all camera data")

    def create_npy_viewer_panel(self):
        """Create the NPY file viewer panel."""
        # Control frame for viewer controls
        self.npy_control_frame = ttk.Frame(self.npy_viewer_frame)
        self.npy_control_frame.pack(fill=tk.X, pady=(0, 5))

        # Viewer toggle button
        self.npy_toggle_button = ttk.Button(
            self.npy_control_frame,
            text="🔽 Expand NPY Viewer",
            command=self.toggle_npy_viewer
        )
        self.npy_toggle_button.pack(side=tk.LEFT)

        # Clear viewer button
        self.npy_clear_button = ttk.Button(
            self.npy_control_frame,
            text="❌ Clear",
            command=self.clear_npy_viewer,
            state="disabled"
        )
        self.npy_clear_button.pack(side=tk.RIGHT)

        # Content frame (initially hidden)
        self.npy_content_frame = ttk.Frame(self.npy_viewer_frame)

        # Metadata display frame
        metadata_frame = ttk.LabelFrame(self.npy_content_frame, text="Array Metadata", padding="5")
        metadata_frame.pack(fill=tk.X, pady=(5, 0))

        # Metadata labels
        self.npy_filename_label = ttk.Label(metadata_frame, text="File: None")
        self.npy_filename_label.pack(anchor=tk.W, padx=5)

        self.npy_shape_label = ttk.Label(metadata_frame, text="Shape: N/A")
        self.npy_shape_label.pack(anchor=tk.W, padx=5)

        self.npy_dtype_label = ttk.Label(metadata_frame, text="Data Type: N/A")
        self.npy_dtype_label.pack(anchor=tk.W, padx=5)

        self.npy_size_label = ttk.Label(metadata_frame, text="Size: N/A")
        self.npy_size_label.pack(anchor=tk.W, padx=5)

        self.npy_stats_label = ttk.Label(metadata_frame, text="Statistics: N/A")
        self.npy_stats_label.pack(anchor=tk.W, padx=5)

        # Data preview frame
        preview_frame = ttk.LabelFrame(self.npy_content_frame, text="Data Preview", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Preview text area with scrollbar
        self.npy_preview_text = tk.Text(
            preview_frame,
            wrap=tk.NONE,
            height=8,
            state='disabled',
            font=("Consolas", 9)
        )

        # Scrollbars for preview
        preview_scroll_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.npy_preview_text.yview)
        preview_scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.npy_preview_text.xview)

        self.npy_preview_text.configure(yscrollcommand=preview_scroll_y.set, xscrollcommand=preview_scroll_x.set)

        # Pack preview components
        self.npy_preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        preview_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_npy_viewer(self):
        """Toggle the visibility of the NPY viewer panel."""
        if self.npy_expanded:
            # Collapse the viewer
            self.npy_content_frame.pack_forget()
            self.npy_toggle_button.config(text="🔽 Expand NPY Viewer")
            self.npy_expanded = False
            # Hide the viewer frame completely when collapsed
            self.npy_viewer_frame.grid_remove()
        else:
            # Expand the viewer
            self.npy_content_frame.pack(fill=tk.BOTH, expand=True)
            self.npy_toggle_button.config(text="🔼 Collapse NPY Viewer")
            self.npy_expanded = True
            # Show the viewer frame
            self.npy_viewer_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

    def clear_npy_viewer(self):
        """Clear the NPY viewer display."""
        self.npy_filename_label.config(text="File: None")
        self.npy_shape_label.config(text="Shape: N/A")
        self.npy_dtype_label.config(text="Data Type: N/A")
        self.npy_size_label.config(text="Size: N/A")
        self.npy_stats_label.config(text="Statistics: N/A")

        self.npy_preview_text.config(state='normal')
        self.npy_preview_text.delete(1.0, tk.END)
        self.npy_preview_text.config(state='disabled')

        self.npy_clear_button.config(state="disabled")
        self.current_npy_file = None

    def display_npy_file(self, filepath):
        """Load and display an NPY file in the viewer."""
        try:
            # Load the NPY file
            array = np.load(filepath)

            # Update metadata
            self.npy_filename_label.config(text=f"File: {filepath.name}")
            self.npy_shape_label.config(text=f"Shape: {array.shape}")
            self.npy_dtype_label.config(text=f"Data Type: {array.dtype}")
            self.npy_size_label.config(text=f"Size: {array.size} elements")

            # Calculate statistics
            if array.size > 0:
                if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating):
                    stats_parts = []
                    if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating):
                        stats_parts.append(f"Min: {array.min():g}")
                        stats_parts.append(f"Max: {array.max():g}")
                        if np.issubdtype(array.dtype, np.floating) or array.size <= 1000000:  # Avoid mean calculation on very large arrays
                            stats_parts.append(f"Mean: {array.mean():g}")
                    self.npy_stats_label.config(text=f"Statistics: {', '.join(stats_parts)}")
                else:
                    self.npy_stats_label.config(text=f"Statistics: Non-numeric data type")
            else:
                self.npy_stats_label.config(text="Statistics: Empty array")

            # Generate preview
            preview_text = self.format_array_preview(array)
            self.npy_preview_text.config(state='normal')
            self.npy_preview_text.delete(1.0, tk.END)
            self.npy_preview_text.insert(tk.END, preview_text)
            self.npy_preview_text.config(state='disabled')

            # Update UI state
            self.npy_clear_button.config(state="normal")
            self.current_npy_file = filepath

            # Auto-expand viewer if collapsed
            if not self.npy_expanded:
                self.toggle_npy_viewer()

            self.log_message(f"✓ Loaded NPY file: {filepath.name}")

        except Exception as e:
            error_msg = f"Error loading NPY file {filepath.name}: {str(e)}"
            messagebox.showerror("NPY File Error", error_msg)
            self.log_message(f"❌ {error_msg}")

    def format_array_preview(self, array, max_rows=10, max_cols=10):
        """Format array data for preview display."""
        if array.ndim == 0:
            # Scalar value
            return f"Scalar value: {array.item()}"

        elif array.ndim == 1:
            # 1D array
            if array.size <= max_cols * 2:
                # Show all elements
                return f"[{', '.join(str(x) for x in array)}]"
            else:
                # Show first and last elements
                first_part = ', '.join(str(x) for x in array[:max_cols])
                last_part = ', '.join(str(x) for x in array[-max_cols:])
                return f"[{first_part}, ..., {last_part}] ({array.size} elements)"

        elif array.ndim == 2:
            # 2D array
            lines = []
            total_rows, total_cols = array.shape

            if total_rows <= max_rows and total_cols <= max_cols:
                # Show entire array
                for row in array:
                    row_str = ', '.join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in row)
                    lines.append(f"[{row_str}]")
                return '\n'.join(lines)
            else:
                # Show truncated view
                for i in range(min(max_rows // 2, total_rows)):
                    row = array[i]
                    if total_cols <= max_cols:
                        row_str = ', '.join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in row)
                    else:
                        first_cols = ', '.join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in row[:max_cols // 2])
                        last_cols = ', '.join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in row[-(max_cols // 2):])
                        row_str = f"{first_cols}, ..., {last_cols}"
                    lines.append(f"[{row_str}]")

                if total_rows > max_rows:
                    lines.append("...")

                    # Show last few rows
                    for i in range(max(total_rows - max_rows // 2, max_rows // 2), total_rows):
                        row = array[i]
                        if total_cols <= max_cols:
                            row_str = ', '.join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in row)
                        else:
                            first_cols = ', '.join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in row[:max_cols // 2])
                            last_cols = ', '.join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in row[-(max_cols // 2):])
                            row_str = f"{first_cols}, ..., {last_cols}"
                        lines.append(f"[{row_str}]")

                return '\n'.join(lines)
        else:
            # Higher-dimensional array
            flat_array = array.flatten()
            preview_size = min(100, array.size)
            preview_data = ', '.join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in flat_array[:preview_size])
            if array.size > preview_size:
                preview_data += ", ..."
            return f"High-dimensional array: {array.ndim}D, shape {array.shape}\nFirst {preview_size} elements: [{preview_data}]"
        return '\n'.join(lines)

    def create_log_panel(self):
        """Create the log panel for execution output."""
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame,
            wrap=tk.WORD,
            height=10,
            state='disabled'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Clear button
        ttk.Button(
            self.log_frame, text="Clear Log",
            command=self.clear_log
        ).pack(side=tk.RIGHT, anchor=tk.S, pady=(5, 0))

    def bind_events(self):
        """Bind event handlers."""
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Keyboard shortcuts
        self.root.bind('<F1>', lambda e: self.show_help())
        self.root.bind('<Control-r>', lambda e: self.update_status())

    def toggle_advanced(self):
        """Toggle visibility of advanced parameters."""
        if self.show_advanced.get():
            self.advanced_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
            self.button_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0))
            self.util_frame.grid(row=6, column=0, columnspan=2, pady=(10, 0))
            self.progress_bar.grid(row=7, column=0, columnspan=2, pady=(10, 0))
        else:
            self.advanced_frame.grid_remove()
            self.button_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))
            self.util_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))
            self.progress_bar.grid(row=6, column=0, columnspan=2, pady=(10, 0))

    def center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def load_config(self):
        """Load saved configuration if available."""
        try:
            config_file = Path("gui_config.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Load saved parameters
                for key, var in zip(
                    ['star_count', 'image_width', 'image_height', 'voxel_size', 'fov_degrees', 'voxel_range'],
                    [self.star_count, self.image_width, self.image_height, self.voxel_size, self.fov_degrees, self.voxel_range]
                ):
                    if key in config:
                        var.set(config[key])
        except Exception:
            pass  # Config loading failed, use defaults

    def save_config(self):
        """Save current configuration."""
        try:
            config = {
                'star_count': self.star_count.get(),
                'image_width': self.image_width.get(),
                'image_height': self.image_height.get(),
                'voxel_size': self.voxel_size.get(),
                'fov_degrees': self.fov_degrees.get(),
                'voxel_range': self.voxel_range.get(),
                'auto_save': self.auto_save.get()
            }

            with open("gui_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass  # Config saving failed, continue

    def update_status(self):
        """Update the status indicators."""
        try:
            # Check build status
            if self.check_build_exists():
                self.build_status.config(text="🟢 Found", foreground="green")
            else:
                self.build_status.config(text="🔴 Missing", foreground="red")

            # Check demo data status
            demo_dir = Path("demo_output")
            if demo_dir.exists() and any(demo_dir.iterdir()):
                self.demo_status.config(text="🟢 Available", foreground="green")

                # Populate file tree
                self.update_file_tree()
            else:
                self.demo_status.config(text="⚪ Not Generated", foreground="gray")

            # Check visualization status
            viz_dir = Path("visualizations")
            if viz_dir.exists() and any(viz_dir.iterdir()):
                self.viz_status.config(text="🟢 Generated", foreground="green")
            else:
                self.viz_status.config(text="⚪ Not Generated", foreground="gray")

        except Exception as e:
            self.log_message(f"Status check error: {e}")

    def update_file_tree(self):
        """Update the file tree with available output files."""
        # Clear existing items
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)

        # Add root directories
        demo_path = Path("demo_output")
        if demo_path.exists():
            demo_node = self.file_tree.insert("", 'end', text="📁 demo_output", open=True)
            for file_path in sorted(demo_path.glob("*")):
                if file_path.is_file():
                    self.file_tree.insert(demo_node, 'end', text=f"📄 {file_path.name}")

        viz_path = Path("visualizations")
        if viz_path.exists():
            viz_node = self.file_tree.insert("", 'end', text="📁 visualizations", open=False)
            for file_path in sorted(viz_path.glob("*")):
                if file_path.is_file():
                    self.file_tree.insert(viz_node, 'end', text=f"📊 {file_path.name}")

        # Add build directory if available
        build_path = Path("build/Debug")
        if build_path.exists():
            build_node = self.file_tree.insert("", 'end', text="📁 build/Debug", open=False)
            for file_path in sorted(build_path.glob("*")):
                if file_path.is_file():
                    self.file_tree.insert(build_node, 'end', text=f"⚙️ {file_path.name}")

    def check_build_exists(self):
        """Check if the C++ library build exists."""
        build_file = Path("build/Debug/process_image_cpp.dll")
        return build_file.exists()

    def log_message(self, message):
        """Add a message to the log."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)

    def clear_log(self):
        """Clear the log text area."""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')

    def run_demo(self):
        """Run the demo script in a separate thread."""
        if self.demo_running:
            messagebox.showwarning("Warning", "Demo is already running!")
            return

        self.demo_running = True
        self.demo_button.config(text="⏹️ Stop Demo", state='disabled')
        self.progress_var.set(0)

        # Save current config
        self.save_config()

        # Run in separate thread
        thread = threading.Thread(target=self._run_demo_thread)
        thread.daemon = True
        thread.start()

    def _run_demo_thread(self):
        """Run demo in background thread."""
        try:
            self.log_message("=" * 50)
            self.log_message("🚀 Starting Pixel-to-Voxel Demo")
            self.log_message("=" * 50)

            # Update progress
            self.root.after(0, lambda: self.progress_var.set(10))
            self.log_message("✓ Parameters configured")
            self.log_message(f"  - Stars: {self.star_count.get()}")
            self.log_message(f"  - Resolution: {self.image_width.get()}x{self.image_height.get()}")
            self.log_message(f"  - Voxel Grid: {self.voxel_size.get()}³")

            # Run demo script
            cmd = [sys.executable, "demo_pixeltovoxel.py"]
            self.log_message(f"Running: {' '.join(cmd)}")

            self.root.after(0, lambda: self.progress_var.set(30))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=300  # 5 minute timeout
            )

            self.root.after(0, lambda: self.progress_var.set(80))

            if result.returncode == 0:
                self.log_message("✓ Demo completed successfully!")
                self.log_message(result.stdout)
            else:
                self.log_message("✗ Demo failed!")
                self.log_message(f"Error: {result.stderr}")

            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, self.update_status)

        except subprocess.TimeoutExpired:
            self.log_message("✗ Timeout: Demo took too long to complete")
        except Exception as e:
            self.log_message(f"✗ Unexpected error: {e}")
        finally:
            self.demo_running = False
            self.root.after(0, lambda: self.demo_button.config(text="🚀 Run Demo", state='normal'))

    def run_visualization(self):
        """Run the visualization script."""
        if self.viz_running:
            messagebox.showwarning("Warning", "Visualization is already running!")
            return

        if not Path("demo_output").exists() or not any(Path("demo_output").iterdir()):
            answer = messagebox.askyesno("No Demo Data",
                                       "No demo data found. Would you like to run the demo first?")
            if answer:
                self.run_demo()
            return

        self.viz_running = True
        self.viz_button.config(text="⏹️ Stop Visualization", state='disabled')
        self.progress_var.set(0)

        # Run in separate thread
        thread = threading.Thread(target=self._run_viz_thread)
        thread.daemon = True
        thread.start()

    def _run_viz_thread(self):
        """Run visualization in background thread."""
        try:
            self.log_message("=" * 50)
            self.log_message("📊 Generating Visualizations")
            self.log_message("=" * 50)

            self.root.after(0, lambda: self.progress_var.set(20))

            # Run visualization script
            cmd = [sys.executable, "visualize_results.py"]
            self.log_message(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=600  # 10 minute timeout
            )

            self.root.after(0, lambda: self.progress_var.set(90))

            if result.returncode == 0:
                self.log_message("✓ Visualizations generated successfully!")
                self.log_message("Check the './visualizations/' directory for results.")
            else:
                self.log_message("✗ Visualization failed!")
                self.log_message(f"Error: {result.stderr}")

            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, self.update_status)

        except subprocess.TimeoutExpired:
            self.log_message("✗ Timeout: Visualization took too long")
        except Exception as e:
            self.log_message(f"✗ Unexpected error: {e}")
        finally:
            self.viz_running = False
            self.root.after(0, lambda: self.viz_button.config(text="📊 Generate Visualizations", state='normal'))

    def open_output_folder(self):
        """Open the output folder in file explorer."""
        try:
            output_dir = Path("demo_output")
            if not output_dir.exists():
                output_dir.mkdir(parents=True)

            if sys.platform == "win32":
                os.startfile(str(output_dir))
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(output_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(output_dir)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder: {e}")

    def open_file(self, event):
        """Open a file from the tree view on double-click."""
        # Get selected item
        selection = self.file_tree.selection()
        if not selection:
            return

        item = selection[0]
        text = self.file_tree.item(item, "text")

        # Remove icon from text if present
        if "📄 " in text:
            filename = text.replace("📄 ", "")
        elif "📊 " in text:
            filename = text.replace("📊 ", "")
        elif "⚙️ " in text:
            filename = text.replace("⚙️ ", "")
        else:
            filename = text

        # Construct full path based on the tree node
        try:
            parent_item = self.file_tree.parent(item)
            if parent_item:
                parent_text = self.file_tree.item(parent_item, "text")
                if "📁 demo_output" in parent_text:
                    filepath = Path("demo_output") / filename
                elif "📁 visualizations" in parent_text:
                    filepath = Path("visualizations") / filename
                elif "📁 build/Debug" in parent_text:
                    filepath = Path("build/Debug") / filename
                else:
                    # Default to demo_output if parent is unclear
                    filepath = Path("demo_output") / filename
            else:
                # Root-level items that somehow got selected
                return

            # Check if it's an NPY file
            if filepath.suffix.lower() == '.npy':
                # Use the built-in NPY viewer
                self.display_npy_file(filepath)
                self.log_message(f"✓ Opened NPY file in viewer: {filepath.name}")
            else:
                # Open with default system application
                try:
                    if sys.platform == "win32":
                        os.startfile(str(filepath))
                    elif sys.platform == "darwin":  # macOS
                        subprocess.run(["open", str(filepath)])
                    else:  # Linux and other Unix-like systems
                        subprocess.run(["xdg-open", str(filepath)])
                    self.log_message(f"✓ Opened file with system default: {filepath.name}")
                except Exception as e:
                    error_msg = f"Error opening file {filepath.name}: {str(e)}"
                    messagebox.showerror("Error", error_msg)
                    self.log_message(f"❌ {error_msg}")

        except Exception as e:
            self.log_message(f"❌ Error opening file: {e}")
        selection = self.file_tree.selection()
        if not selection:
            return

        item_text = self.file_tree.item(selection, "text")

        # Extract file name from item text
        if item_text.startswith("📄 ") or item_text.startswith("📊 ") or item_text.startswith("⚙️ "):
            filename = item_text[2:]  # Remove emoji prefix

            # Find the parent directory
            parent = self.file_tree.parent(selection)
            parent_text = self.file_tree.item(parent, "text")

            if "demo_output" in parent_text:
                filepath = Path("demo_output") / filename
            elif "visualizations" in parent_text:
                filepath = Path("visualizations") / filename
            elif "build" in parent_text:
                filepath = Path("build/Debug") / filename
            else:
                return

            if filepath.exists():
                try:
                    if sys.platform == "win32":
                        os.startfile(str(filepath))
                    elif sys.platform == "darwin":
                        subprocess.run(["open", str(filepath)])
                    else:
                        subprocess.run(["xdg-open", str(filepath.parent)])
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open file: {e}")

    def show_help(self):
        """Show help dialog."""
        help_text = """
Pixel-to-Voxel Projector GUI Help
=================================

OVERVIEW:
--------
This GUI provides an easy-to-use interface for the Pixel-to-Voxel Projector,
an astronomical image processing system that converts 2D images into 3D voxel grids.

FEATURES:
--------
• Configure demo parameters (number of stars, image resolution, voxel grid size)
• Run synthetic astronomical data generation
• Generate comprehensive visualizations
• Monitor progress and execution logs
• Browse generated files and results

USAGE:
-----
1. Adjust parameters in the control panel
2. Click "🚀 Run Demo" to generate synthetic astronomical data
3. Click "📊 Generate Visualizations" to create plots and charts
4. Monitor progress and check the execution log
5. Browse results using the file tree

SHORTCUTS:
---------
• F1: Show this help
• Ctrl+R: Refresh status

PROCESSES:
---------
• Demo Script: Creates synthetic astronomical images with stars
• C++ Library: Compiles the process_image_cpp.dll for performance
• Visualization: Creates matplotlib charts showing data analysis

TROUBLESHOOTING:
---------------
• Ensure Python is properly installed with numpy and matplotlib
• Build the C++ library first with: cd build && cmake .. && cmake --build .
• Check the execution log for detailed error messages
• Close other applications using matplotlib to avoid display conflicts

For more information, see the README.md file.
"""

        # Create help dialog
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("Help - Pixel-to-Voxel Projector")
        help_dialog.geometry("700x500")

        # Help text widget
        help_text_widget = scrolledtext.ScrolledText(
            help_dialog, wrap=tk.WORD, padx=10, pady=10
        )
        help_text_widget.insert(1.0, help_text)
        help_text_widget.configure(state='disabled')
        help_text_widget.pack(fill=tk.BOTH, expand=True)

        # Close button
        ttk.Button(help_dialog, text="Close", command=help_dialog.destroy) \
            .pack(pady=10)

        help_dialog.transient(self.root)
        help_dialog.grab_set()
        help_dialog.focus_set()

    def on_close(self):
        """Handle window close event."""
        if self.demo_running or self.viz_running:
            if messagebox.askyesno("Quit",
                                 "Demo or visualization is still running. Quit anyway?"):
                if self.current_process:
                    try:
                        self.current_process.terminate()
                    except:
                        pass
                self.root.destroy()
        else:
            self.save_config()
            self.root.destroy()


def check_tkinter_available():
    """Check if tkinter is available."""
    try:
        import tkinter
        import tkinter.ttk
        return True
    except ImportError:
        return False


def main():
    """Main function to start the GUI."""
    if not check_tkinter_available():
        print("Error: tkinter is not available.")
        print("The GUI requires tkinter, which should come with Python.")
        print("For terminal-based usage, run:")
        print("  python demo_pixeltovoxel.py          # Run demo")
        print("  python visualize_results.py         # Generate visualizations")
        return 1

    # Check for matplotlib
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use tkinter backend
    except ImportError:
        print("Warning: matplotlib not found. Some functionality may be limited.")

    # Create and run GUI
    root = tk.Tk()
    app = PixeltovoxelGUI(root)

    print("Pixel-to-Voxel Projector GUI started.")
    print("Close the window to exit the application.")

    root.mainloop()
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)