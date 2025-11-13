cmake_minimum_required(VERSION 3.18)
project(CalibrationSystem LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# CUDA architecture settings (RTX 3090/4090)
set(CMAKE_CUDA_ARCHITECTURES 86 89)

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# ============================================================================
# C++ Library - Coordinate Transformer
# ============================================================================

add_library(coordinate_transformer SHARED
    coordinate_transformer.cpp
)

target_link_libraries(coordinate_transformer
    ${EIGEN3_LIBRARIES}
)

# ============================================================================
# CUDA Library - Transform Optimizer
# ============================================================================

add_library(transform_optimizer SHARED
    transform_optimizer.cu
)

target_link_libraries(transform_optimizer
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
)

set_target_properties(transform_optimizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDA compiler flags for optimization
target_compile_options(transform_optimizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --ptxas-options=-v
        -lineinfo
        -maxrregcount=128
    >
)

# ============================================================================
# Example Executables (Optional)
# ============================================================================

# C++ example
add_executable(coordinate_transformer_example
    coordinate_transformer.cpp
)

target_compile_definitions(coordinate_transformer_example PRIVATE BUILD_EXAMPLE)

target_link_libraries(coordinate_transformer_example
    ${EIGEN3_LIBRARIES}
)

# CUDA example
add_executable(transform_optimizer_example
    transform_optimizer.cu
)

target_compile_definitions(transform_optimizer_example PRIVATE BUILD_EXAMPLE)

target_link_libraries(transform_optimizer_example
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

set_target_properties(transform_optimizer_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# Installation
# ============================================================================

install(TARGETS coordinate_transformer transform_optimizer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    coordinate_transformer.cpp
    transform_optimizer.cu
    DESTINATION include/calibration
)

# ============================================================================
# Testing
# ============================================================================

enable_testing()

add_executable(test_coordinate_transformer
    coordinate_transformer.cpp
)

target_compile_definitions(test_coordinate_transformer PRIVATE BUILD_EXAMPLE)
target_link_libraries(test_coordinate_transformer ${EIGEN3_LIBRARIES})

add_test(NAME CoordinateTransformerTest
    COMMAND test_coordinate_transformer
)
