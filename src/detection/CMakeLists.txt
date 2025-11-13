# CMakeLists.txt for drone detection system
cmake_minimum_required(VERSION 3.18)
project(drone_detector LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Find required packages
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# CUDA settings for optimal performance
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -O3;
    --use_fast_math;
    --ptxas-options=-v;
    -lineinfo;
)

# Enable separable compilation for device linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# CUDA small object detector library
add_library(small_object_detector STATIC
    small_object_detector.cu
)

set_target_properties(small_object_detector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(small_object_detector PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        -Xcompiler=-fPIC
    >
)

# Python bindings for drone detector
pybind11_add_module(drone_detector
    drone_detector.cpp
)

target_link_libraries(drone_detector PRIVATE
    small_object_detector
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cufft_LIBRARY}
    Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(drone_detector PRIVATE OpenMP::OpenMP_CXX)
endif()

# Optimization flags
target_compile_options(drone_detector PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -ffast-math
    >
)

# Installation
install(TARGETS drone_detector
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
