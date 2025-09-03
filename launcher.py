#!/usr/bin/env python3
"""
Launcher for Pixel-to-Voxel Projector
=====================================

Universal entry point that detects available interfaces and guides users
to the appropriate usage method.

This launcher:
- Tries to start the GUI if tkinter is available
- Falls back to terminal instructions if GUI is unavailable
- Provides a consistent entry point for users
- Auto-detects system configuration
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

class PixeltovoxelLauncher:
    """Launcher that chooses between GUI and terminal interfaces."""

    def __init__(self):
        self.project_name = "Pixel-to-Voxel Projector"
        self.gui_script = "gui_interface.py"
        self.demo_script = "demo_pixeltovoxel.py"
        self.viz_script = "visualize_results.py"

    def check_tkinter(self):
        """Check if tkinter is available."""
        try:
            import tkinter
            import tkinter.ttk
            # Try to create a basic window to ensure tkinter works
            root = tkinter.Tk()
            root.destroy()
            return True
        except Exception as e:
            print(f"Tkinter check failed: {e}")
            return False

    def check_system_requirements(self):
        """Check basic system requirements."""
        requirements = {
            "Python Version": sys.version_info >= (3, 6),
            "Build Directory": Path("build/Debug/process_image_cpp.dll").exists(),
            "Demo Output": Path("demo_output").exists(),
            "Visualizations": Path("visualizations").exists(),
        }
        return requirements

    def start_gui(self):
        """Start the GUI interface."""
        try:
            print(f"🚀 Starting {self.project_name} GUI...")
            result = subprocess.run([sys.executable, self.gui_script],
                                  check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ GUI failed to start: {e}")
            return False
        except KeyboardInterrupt:
            print("\n👋 GUI closed by user")
            return True

    def show_terminal_usage(self):
        """Show terminal-based usage instructions."""
        print(f"\n{'='*60}")
        print(f"📊 {self.project_name} - Terminal Usage")
        print(f"{'='*60}")

        print("\n🖥️  GUI NOT AVAILABLE")
        print("   The graphical interface requires tkinter.")
        print("   You can install it or use the terminal commands below.")

        print("\n🚀 QUICK START (Terminal Mode):")
        print(f"   1. Run the demo: python {self.demo_script}")
        print(f"   2. Generate visualizations: python {self.viz_script}")
        print("   3. Check results in ./demo_output/ and ./visualizations/")

        print("\n📁 PROJECT SCRIPTS:")
        print(f"   • {self.demo_script}        - Run complete astronomical demo")
        print(f"   • {self.viz_script}        - Generate visualizations")
        print(f"   • {self.gui_script}        - GUI interface (when available)")
        print("   • CMake-based build system for C++ library")

        print("\n🛠️  SYSTEM REQUIREMENTS:")
        print("   • Python 3.6+ with numpy and matplotlib")
        print("   • C++17 compiler (GCC/Clang/MSVC)")
        print("   • CMake 3.12+")
        print("   • tkinter (for GUI) - optional")

        print("\n🔧 BUILDING THE PROJECT:")
        print("   mkdir build && cd build")
        print("   cmake ..")
        print("   cmake --build .")

    def show_system_status(self):
        """Show current system status."""
        print(f"\n{'='*60}")
        print("🔍 SYSTEM STATUS CHECK")
        print(f"{'='*60}")

        requirements = self.check_system_requirements()

        for component, status in requirements.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}")

        # Show GUI availability
        gui_ok = self.check_tkinter()
        gui_status = "✅ Available" if gui_ok else "❌ Not Available"
        print(f"   {gui_status} Graphical Interface (tkinter)")

    def interactive_menu(self):
        """Show interactive menu for terminal users."""
        while True:
            print(f"\n{'='*60}")
            print(f"🎛️  {self.project_name} - Interactive Menu")
            print(f"{'='*60}")
            print("1. Run Demo Script")
            print("2. Generate Visualizations")
            print("3. Open Output Directory")
            print("4. Check System Status")
            print("5. Help")
            print("6. Exit")
            print(f"{'='*60}")

            try:
                choice = input("Choose an option (1-6): ").strip()

                if choice == "1":
                    print("\n🚀 Running demo script...")
                    try:
                        subprocess.run([sys.executable, self.demo_script], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Demo failed: {e}")

                elif choice == "2":
                    print("\n📊 Generating visualizations...")
                    if not Path("demo_output").exists():
                        print("❌ No demo data found. Run the demo first (option 1).")
                        continue
                    try:
                        subprocess.run([sys.executable, self.viz_script], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Visualization failed: {e}")

                elif choice == "3":
                    print("\n📂 Opening output directories...")
                    directories = ["demo_output", "visualizations"]

                    for dir_name in directories:
                        dir_path = Path(dir_name)
                        if not dir_path.exists():
                            dir_path.mkdir(parents=True)

                        try:
                            if platform.system() == "Windows":
                                os.startfile(str(dir_path))
                            elif platform.system() == "Darwin":
                                subprocess.run(["open", str(dir_path)])
                            else:
                                subprocess.run(["xdg-open", str(dir_path)])
                        except Exception as e:
                            print(f"Could not open {dir_name}: {e}")

                elif choice == "4":
                    self.show_system_status()

                elif choice == "5":
                    self.show_terminal_usage()

                elif choice == "6":
                    print("\n👋 Goodbye!")
                    break

                else:
                    print("❌ Invalid choice. Please enter 1-6.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                # Handle cases where input is not available (e.g., piped input)
                print("\n👋 Input not available. Exiting.")
                break

    def run(self):
        """Main launcher function."""
        print(f"🛰️  Welcome to {self.project_name}!")

        # Show system status first
        self.show_system_status()

        # Try to start GUI
        gui_available = self.check_tkinter()

        if gui_available:
            print(f"\n🖥️  GUI AVAILABLE - Starting graphical interface...")
            success = self.start_gui()

            if not success:
                print("❌ GUI failed to start. Falling back to terminal mode...")
                self.interactive_menu()
        else:
            print(f"\n⚙️  FALLING BACK TO TERMINAL MODE")
            print("tkinter not available. Starting interactive terminal menu...")

            # Small delay for readability
            import time
            time.sleep(1)

            self.interactive_menu()


def main():
    """Main entry point."""
    try:
        launcher = PixeltovoxelLauncher()
        launcher.run()
    except KeyboardInterrupt:
        print("\n👋 Program terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n🔧 DEPENDENCY CHECKS:")
        print("   Make sure you have:")
        print("   - Python 3.6+")
        print("   - numpy, matplotlib installed")
        print("   - tkinter (for GUI) - optional")
        sys.exit(1)


if __name__ == "__main__":
    main()