"""
Setup script for building the C++ motion extractor module
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys
import os


class BuildExt(build_ext):
    """Custom build extension with optimizations"""

    c_opts = {
        'msvc': ['/O2', '/openmp', '/arch:AVX2'],
        'unix': ['-O3', '-fopenmp', '-march=native', '-ffast-math'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append('-std=c++14')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        'motion_extractor_cpp',
        ['motion_extractor.cpp'],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=[],
        extra_link_args=[],
    ),
]

setup(
    name='motion_extractor_cpp',
    version='1.0.0',
    author='Pixel to Voxel Project',
    description='Optimized 8K motion extraction module',
    long_description='High-performance motion detection and tracking for 8K video streams',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=['pybind11>=2.6.0'],
    zip_safe=False,
    python_requires='>=3.6',
)
