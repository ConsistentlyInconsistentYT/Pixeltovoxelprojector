"""
Setup script for building the thermal_mono_fusion C++ extension
Run with: python setup_fusion.py build_ext --inplace
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys
import os


class BuildExt(build_ext):
    """Custom build extension to add compiler-specific flags"""

    c_opts = {
        'msvc': ['/O2', '/openmp', '/std:c++17'],
        'unix': ['-O3', '-fopenmp', '-std=c++17', '-march=native'],
    }

    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }

    cuda_opts = {
        'nvcc': ['-O3', '--use_fast_math', '-arch=sm_70', '--std=c++17'],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        # Check for CUDA availability
        cuda_available = os.system('which nvcc > /dev/null 2>&1') == 0

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

            if cuda_available and 'WITH_CUDA' not in ext.define_macros:
                print("CUDA detected, building with CUDA support")
                ext.define_macros.append(('WITH_CUDA', '1'))
                # Add CUDA include and library paths
                ext.include_dirs.append('/usr/local/cuda/include')
                ext.library_dirs.append('/usr/local/cuda/lib64')
                ext.libraries.extend(['cudart', 'cublas'])
            else:
                print("CUDA not detected, building CPU-only version")

        build_ext.build_extensions(self)


# Define the extension module
ext_modules = [
    Extension(
        'thermal_mono_fusion',
        ['thermal_mono_fusion.cpp'],
        include_dirs=[
            pybind11.get_include(),
            '/usr/local/cuda/include',  # CUDA include path
        ],
        library_dirs=[
            '/usr/local/cuda/lib64',  # CUDA library path
        ],
        libraries=[],  # Will be added by BuildExt if CUDA available
        language='c++',
        extra_compile_args=[],  # Will be added by BuildExt
        extra_link_args=[],  # Will be added by BuildExt
    ),
]

setup(
    name='thermal_mono_fusion',
    version='1.0.0',
    author='Pixeltovoxelprojector Team',
    description='High-performance thermal + monochrome camera fusion',
    long_description=open('README_FUSION.md', 'r').read() if os.path.exists('README_FUSION.md') else '',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
    ],
    python_requires='>=3.7',
    zip_safe=False,
)
