cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ============================================================================
# Project Configuration
# ============================================================================
project(PixelToVoxelProjector
    VERSION 1.0.0
    DESCRIPTION "High-performance 8K video motion tracking and voxel processing"
    LANGUAGES CXX CUDA
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# Options
# ============================================================================
option(BUILD_CUDA "Build CUDA extensions" ON)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(USE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_FAST_MATH "Enable fast math optimizations" ON)

# ============================================================================
# Dependencies
# ============================================================================

# Find Python
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Python3: ${Python3_VERSION}")
endif()

# Find pybind11
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found via find_package, trying Python module")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(PYBIND11_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
            find_package(pybind11 CONFIG REQUIRED)
        else()
            message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
        endif()
    endif()
    message(STATUS "pybind11: ${pybind11_VERSION}")
endif()

# Find OpenMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")
    endif()
endif()

# Find CUDA
if(BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_TARGET_DIR}")

    # Detect GPU architecture
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Default to RTX 3090/4090 compute capabilities
        set(CMAKE_CUDA_ARCHITECTURES "86;89" CACHE STRING "CUDA architectures")
    endif()
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ============================================================================
# Compiler Flags
# ============================================================================

# C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    if(ENABLE_FAST_MATH)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
    endif()
    message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    message(STATUS "Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

# CUDA compiler flags
if(BUILD_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --ptxas-options=-v")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -maxrregcount=128")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")
    endif()
endif()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/cuda
    ${CMAKE_SOURCE_DIR}/include
)

if(BUILD_PYTHON_BINDINGS)
    include_directories(${Python3_INCLUDE_DIRS})
endif()

if(BUILD_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# ============================================================================
# Source Files
# ============================================================================

# C++ source files
set(CPP_SOURCES
    src/motion_extractor.cpp
    src/voxel/sparse_voxel_grid.cpp
    src/protocols/stream_manager.cpp
    src/detection/drone_detector.cpp
    src/fusion/thermal_mono_fusion.cpp
    src/camera/orientation_manager.cpp
    process_image.cpp
)

# CUDA source files
if(BUILD_CUDA)
    set(CUDA_SOURCES
        cuda/voxel_cuda.cu
        src/voxel/voxel_optimizer.cu
        src/detection/small_object_detector.cu
    )
endif()

# ============================================================================
# Libraries
# ============================================================================

# C++ Library
add_library(voxel_processing_cpp STATIC ${CPP_SOURCES})
target_include_directories(voxel_processing_cpp PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/cuda
)

if(USE_OPENMP)
    target_link_libraries(voxel_processing_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

# CUDA Library
if(BUILD_CUDA)
    add_library(voxel_processing_cuda STATIC ${CUDA_SOURCES})
    target_include_directories(voxel_processing_cuda PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/cuda
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(voxel_processing_cuda PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
    )

    if(USE_OPENMP)
        target_link_libraries(voxel_processing_cuda PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # Set CUDA architecture
    set_target_properties(voxel_processing_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

# ============================================================================
# Python Bindings
# ============================================================================

if(BUILD_PYTHON_BINDINGS)
    # Process image binding
    pybind11_add_module(process_image_cpp
        process_image.cpp
    )
    target_link_libraries(process_image_cpp PRIVATE voxel_processing_cpp)

    # Motion extractor binding
    pybind11_add_module(motion_extractor_cpp
        src/motion_extractor.cpp
    )
    target_link_libraries(motion_extractor_cpp PRIVATE voxel_processing_cpp)

    # Sparse voxel grid binding
    pybind11_add_module(sparse_voxel_grid
        src/voxel/sparse_voxel_grid.cpp
    )
    target_link_libraries(sparse_voxel_grid PRIVATE voxel_processing_cpp)

    # Stream manager binding
    pybind11_add_module(stream_manager
        src/protocols/stream_manager.cpp
    )
    target_link_libraries(stream_manager PRIVATE voxel_processing_cpp)

    # Drone detector binding
    pybind11_add_module(drone_detector
        src/detection/drone_detector.cpp
    )
    target_link_libraries(drone_detector PRIVATE voxel_processing_cpp)

    # Thermal mono fusion binding
    pybind11_add_module(thermal_mono_fusion
        src/fusion/thermal_mono_fusion.cpp
    )
    target_link_libraries(thermal_mono_fusion PRIVATE voxel_processing_cpp)

    # Orientation manager binding
    pybind11_add_module(orientation_manager
        src/camera/orientation_manager.cpp
    )
    target_link_libraries(orientation_manager PRIVATE voxel_processing_cpp)

    # CUDA bindings
    if(BUILD_CUDA)
        # Voxel CUDA binding
        pybind11_add_module(voxel_cuda
            cuda/voxel_cuda.cu
            cuda/voxel_cuda_wrapper.cpp
        )
        target_link_libraries(voxel_cuda PRIVATE
            voxel_processing_cuda
            CUDA::cudart
        )
        set_target_properties(voxel_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )

        # Voxel optimizer CUDA binding
        pybind11_add_module(voxel_optimizer_cuda
            src/voxel/voxel_optimizer.cu
        )
        target_link_libraries(voxel_optimizer_cuda PRIVATE
            voxel_processing_cuda
            CUDA::cudart
        )
        set_target_properties(voxel_optimizer_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )

        # Small object detector CUDA binding
        pybind11_add_module(small_object_detector_cuda
            src/detection/small_object_detector.cu
        )
        target_link_libraries(small_object_detector_cuda PRIVATE
            voxel_processing_cuda
            CUDA::cudart
        )
        set_target_properties(small_object_detector_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )
    endif()
endif()

# ============================================================================
# Tests
# ============================================================================

if(BUILD_TESTS)
    enable_testing()

    # Add test targets here
    message(STATUS "Tests enabled")
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(BUILD_BENCHMARKS)
    # Add benchmark targets here
    message(STATUS "Benchmarks enabled")
endif()

# ============================================================================
# Installation
# ============================================================================

# Install C++ libraries
install(TARGETS voxel_processing_cpp
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

if(BUILD_CUDA)
    install(TARGETS voxel_processing_cuda
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

# Install header files
install(DIRECTORY src/ cuda/
    DESTINATION include/pixeltovoxel
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install Python modules
if(BUILD_PYTHON_BINDINGS)
    install(TARGETS
        process_image_cpp
        motion_extractor_cpp
        sparse_voxel_grid
        stream_manager
        drone_detector
        thermal_mono_fusion
        orientation_manager
        LIBRARY DESTINATION ${Python3_SITELIB}
    )

    if(BUILD_CUDA)
        install(TARGETS
            voxel_cuda
            voxel_optimizer_cuda
            small_object_detector_cuda
            LIBRARY DESTINATION ${Python3_SITELIB}
        )
    endif()
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Configuration Summary")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA enabled: ${BUILD_CUDA}")
if(BUILD_CUDA)
    message(STATUS "  CUDA version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "OpenMP enabled: ${USE_OPENMP}")
message(STATUS "Python bindings: ${BUILD_PYTHON_BINDINGS}")
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "  Python version: ${Python3_VERSION}")
endif()
message(STATUS "Tests: ${BUILD_TESTS}")
message(STATUS "Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "========================================")
message(STATUS "")
