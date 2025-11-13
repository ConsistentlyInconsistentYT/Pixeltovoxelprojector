cmake_minimum_required(VERSION 3.10)
project(OrientationManager VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found - building without parallel support")
endif()

# Standalone executable
add_executable(orientation_manager orientation_manager.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(orientation_manager PUBLIC OpenMP::OpenMP_CXX)
endif()

# Optional: Python bindings with pybind11
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    # Find pybind11
    find_package(pybind11 CONFIG)

    if(pybind11_FOUND)
        message(STATUS "Building Python bindings")

        pybind11_add_module(orientation_manager_cpp
            orientation_manager_bindings.cpp
            orientation_manager.cpp
        )

        if(OpenMP_CXX_FOUND)
            target_link_libraries(orientation_manager_cpp PRIVATE OpenMP::OpenMP_CXX)
        endif()
    else()
        message(WARNING "pybind11 not found - skipping Python bindings")
    endif()
endif()

# Installation
install(TARGETS orientation_manager
    RUNTIME DESTINATION bin
)

# Testing
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_quaternion test_quaternion.cpp)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_quaternion PUBLIC OpenMP::OpenMP_CXX)
    endif()

    add_test(NAME QuaternionTest COMMAND test_quaternion)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP support: ${OpenMP_CXX_FOUND}")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "")
