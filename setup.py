from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
import pybind11
import sys
import os
import subprocess
from pathlib import Path
import glob
import shutil

# ============================================================================
# CUDA Configuration
# ============================================================================

def find_cuda():
    """Find CUDA installation path"""
    cuda_paths = [
        '/usr/local/cuda',
        '/usr/local/cuda-12.0',
        '/usr/local/cuda-11.8',
        '/usr/local/cuda-11.0',
        '/opt/cuda',
        os.environ.get('CUDA_HOME', ''),
        os.environ.get('CUDA_PATH', ''),
    ]

    for path in cuda_paths:
        if path and os.path.exists(path):
            nvcc_path = os.path.join(path, 'bin', 'nvcc')
            if os.path.exists(nvcc_path):
                return path
    return None

def get_cuda_compute_capabilities():
    """Detect GPU compute capabilities"""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit

        capabilities = []
        for i in range(cuda.Device.count()):
            device = cuda.Device(i)
            capability = device.compute_capability()
            cap_str = f"{capability[0]}{capability[1]}"
            capabilities.append((capability[0], capability[1], cap_str))
            print(f"GPU {i}: {device.name()} - Compute Capability {capability[0]}.{capability[1]}")
        return capabilities
    except Exception as e:
        print(f"Could not detect GPU capabilities: {e}")
        # Default to RTX 3090/4090 compute capabilities
        return [(8, 6, '86'), (8, 9, '89')]

CUDA_HOME = find_cuda()
CUDA_AVAILABLE = CUDA_HOME is not None

if CUDA_AVAILABLE:
    print(f"✓ CUDA found at: {CUDA_HOME}")
    CUDA_COMPUTE_CAPS = get_cuda_compute_capabilities()
else:
    print("⚠ WARNING: CUDA not found. CUDA extensions will not be built.")
    print("  Set CUDA_HOME environment variable if CUDA is installed.")
    CUDA_COMPUTE_CAPS = []

# ============================================================================
# Protocol Buffer Compilation
# ============================================================================

def compile_protobuf():
    """Compile protocol buffer definitions"""
    proto_dir = Path('src/protocols')
    proto_files = list(proto_dir.glob('*.proto'))

    if not proto_files:
        print("No protocol buffer files found")
        return

    try:
        import grpc_tools.protoc

        for proto_file in proto_files:
            print(f"Compiling protobuf: {proto_file}")
            result = grpc_tools.protoc.main([
                'grpc_tools.protoc',
                f'-I{proto_dir}',
                f'--python_out={proto_dir}',
                f'--grpc_python_out={proto_dir}',
                str(proto_file)
            ])
            if result != 0:
                print(f"⚠ Warning: Failed to compile {proto_file}")

        print("✓ Protocol buffers compiled successfully")
    except ImportError:
        print("⚠ grpcio-tools not found. Install with: pip install grpcio-tools")
    except Exception as e:
        print(f"⚠ Error compiling protocol buffers: {e}")

# Custom build command that compiles protobufs first
class CustomBuildPy(build_py):
    def run(self):
        compile_protobuf()
        build_py.run(self)

# ============================================================================
# Custom Build Extension for CUDA
# ============================================================================

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/O2', '/openmp', '/std:c++17'],
        'unix': ['-O3', '-fopenmp', '-std=c++17', '-march=native', '-ffast-math'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        # Filter out extensions based on availability
        filtered_extensions = []
        for ext in self.extensions:
            # Skip CUDA extensions if CUDA not available
            if 'cuda' in ext.name and not CUDA_AVAILABLE:
                print(f"⊗ Skipping {ext.name} (CUDA not available)")
                continue

            if 'cuda' not in ext.name:
                ext.extra_compile_args = opts
                ext.extra_link_args = link_opts

            filtered_extensions.append(ext)

        self.extensions = filtered_extensions
        build_ext.build_extensions(self)

    def build_extension(self, ext):
        if 'cuda' in ext.name and CUDA_AVAILABLE:
            self._build_cuda_extension(ext)
        else:
            build_ext.build_extension(self, ext)

    def _build_cuda_extension(self, ext):
        """Custom build for CUDA extensions"""
        sources = ext.sources
        cuda_sources = [s for s in sources if s.endswith('.cu')]
        cpp_sources = [s for s in sources if s.endswith('.cpp') or s.endswith('.cc')]

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        objects = []

        # Compile CUDA sources with nvcc
        if cuda_sources:
            nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')

            # CUDA compilation flags optimized for detected GPUs
            nvcc_flags = [
                '-c',
                '--std=c++17',
                '-O3',
                '--use_fast_math',
                '-Xcompiler', '-fPIC,-fopenmp',
            ]

            # Add compute capabilities for detected GPUs
            for major, minor, cap_str in CUDA_COMPUTE_CAPS:
                nvcc_flags.extend([
                    '-gencode', f'arch=compute_{cap_str},code=sm_{cap_str}',
                ])
            # Add PTX for forward compatibility
            if CUDA_COMPUTE_CAPS:
                last_cap = CUDA_COMPUTE_CAPS[-1][2]
                nvcc_flags.extend([
                    '-gencode', f'arch=compute_{last_cap},code=compute_{last_cap}',
                ])

            # Optimization flags
            nvcc_flags.extend([
                '--ptxas-options=-v',
                '-lineinfo',
                '-maxrregcount=128',
                f'-I{CUDA_HOME}/include',
            ])

            # Add custom include directories
            for inc_dir in ext.include_dirs:
                nvcc_flags.extend(['-I', inc_dir])

            for cu_file in cuda_sources:
                obj_file = build_temp / (Path(cu_file).stem + '.o')
                cmd = [nvcc] + nvcc_flags + [cu_file, '-o', str(obj_file)]
                print(f"  Compiling CUDA: {Path(cu_file).name}")
                subprocess.check_call(cmd)
                objects.append(str(obj_file))

        # Compile C++ sources with g++
        if cpp_sources:
            for cpp_file in cpp_sources:
                obj_file = build_temp / (Path(cpp_file).stem + '.o')
                cmd = [
                    'g++', '-c', cpp_file, '-o', str(obj_file),
                    '-O3', '-fPIC', '-std=c++17', '-fopenmp',
                    '-march=native', '-ffast-math',
                    f'-I{CUDA_HOME}/include',
                ]
                for inc_dir in ext.include_dirs:
                    cmd.extend(['-I', inc_dir])
                print(f"  Compiling C++: {Path(cpp_file).name}")
                subprocess.check_call(cmd)
                objects.append(str(obj_file))

        # Link everything together
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        link_cmd = [
            'g++', '-shared', '-o', ext_path,
        ] + objects + [
            f'-L{CUDA_HOME}/lib64',
            '-lcudart',
            '-lcuda',
            '-fopenmp',
        ]

        for lib_dir in ext.library_dirs:
            link_cmd.extend(['-L', lib_dir])
        for lib in ext.libraries:
            link_cmd.extend(['-l', lib])

        print(f"  Linking: {ext.name}")
        subprocess.check_call(link_cmd)

# ============================================================================
# Extension Modules
# ============================================================================

# Common include directories
common_includes = [
    pybind11.get_include(),
    'src',
    'cuda',
]

if CUDA_AVAILABLE:
    common_includes.append(os.path.join(CUDA_HOME, 'include'))

# C++ Extensions (non-CUDA)
cpp_extensions = [
    Extension(
        'process_image_cpp',
        sources=['process_image.cpp'],
        include_dirs=common_includes,
        language='c++',
    ),
    Extension(
        'motion_extractor_cpp',
        sources=['src/motion_extractor.cpp'],
        include_dirs=common_includes,
        language='c++',
    ),
    Extension(
        'sparse_voxel_grid',
        sources=['src/voxel/sparse_voxel_grid.cpp'],
        include_dirs=common_includes,
        language='c++',
    ),
    Extension(
        'stream_manager',
        sources=['src/protocols/stream_manager.cpp'],
        include_dirs=common_includes,
        language='c++',
    ),
    Extension(
        'drone_detector',
        sources=['src/detection/drone_detector.cpp'],
        include_dirs=common_includes,
        language='c++',
    ),
    Extension(
        'thermal_mono_fusion',
        sources=['src/fusion/thermal_mono_fusion.cpp'],
        include_dirs=common_includes,
        language='c++',
    ),
    Extension(
        'orientation_manager',
        sources=['src/camera/orientation_manager.cpp'],
        include_dirs=common_includes,
        language='c++',
    ),
]

# CUDA Extensions
cuda_extensions = []
if CUDA_AVAILABLE:
    cuda_library_dirs = [os.path.join(CUDA_HOME, 'lib64')]
    cuda_libraries = ['cudart', 'cuda']

    cuda_extensions = [
        Extension(
            'voxel_cuda',
            sources=[
                'cuda/voxel_cuda.cu',
                'cuda/voxel_cuda_wrapper.cpp',
            ],
            include_dirs=common_includes,
            library_dirs=cuda_library_dirs,
            libraries=cuda_libraries,
            language='c++',
        ),
        Extension(
            'voxel_optimizer_cuda',
            sources=['src/voxel/voxel_optimizer.cu'],
            include_dirs=common_includes,
            library_dirs=cuda_library_dirs,
            libraries=cuda_libraries,
            language='c++',
        ),
        Extension(
            'small_object_detector_cuda',
            sources=['src/detection/small_object_detector.cu'],
            include_dirs=common_includes,
            library_dirs=cuda_library_dirs,
            libraries=cuda_libraries,
            language='c++',
        ),
    ]

# Combine all extensions
ext_modules = cpp_extensions + cuda_extensions

print(f"\n{'='*70}")
print(f"Build Configuration:")
print(f"  C++ Extensions: {len(cpp_extensions)}")
print(f"  CUDA Extensions: {len(cuda_extensions)}")
print(f"  Total Extensions: {len(ext_modules)}")
print(f"{'='*70}\n")

# ============================================================================
# Setup
# ============================================================================

# Read long description from README
long_description = ""
readme_path = Path('README.md')
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

# Read requirements from requirements.txt
def read_requirements(filename='requirements.txt'):
    """Read requirements from file"""
    req_path = Path(filename)
    if req_path.exists():
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

install_requires = read_requirements('requirements.txt')

setup(
    name='pixeltovoxelprojector',
    version='1.0.0',
    description='High-performance 8K video motion tracking and voxel processing system',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Voxel Processing Team',
    url='https://github.com/yourusername/Pixeltovoxelprojector',

    # Package discovery
    packages=find_packages(where='src') + find_packages(where='.', include=['cuda*']),
    package_dir={'': 'src', 'cuda': 'cuda'},

    # Extension modules
    ext_modules=ext_modules,

    # Custom build commands
    cmdclass={
        'build_ext': BuildExt,
        'build_py': CustomBuildPy,
    },

    # Dependencies
    install_requires=install_requires,

    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'pytest-benchmark>=3.4.0',
            'black>=21.6b0',
            'flake8>=3.9.0',
            'mypy>=0.910',
        ],
        'cuda': [
            'cupy-cuda11x>=10.0.0',
            'pycuda>=2021.1',
        ],
        'full': [
            'opencv-contrib-python>=4.5.0',
            'pypylon>=1.7.0',
            'simple-pyspin>=0.2.0',
            'pyzmq>=22.0.0',
            'protobuf>=3.19.0',
            'grpcio>=1.40.0',
            'grpcio-tools>=1.40.0',
        ],
    },

    # Python version requirement
    python_requires='>=3.8',

    # Project metadata
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Multimedia :: Video',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Operating System :: POSIX :: Linux',
        'License :: OSI Approved :: MIT License',
    ],

    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'voxel-viewer=spacevoxelviewer:main',
            'motion-viewer=voxelmotionviewer:main',
            'run-benchmark=tests.benchmarks.run_all_benchmarks:main',
        ],
    },

    # Include additional files
    include_package_data=True,
    zip_safe=False,
)

print("\n✓ Setup configuration complete!")
print("  Run 'pip install -e .' for development installation")
print("  Run 'pip install -e .[full,dev]' for full installation with dev tools\n")
