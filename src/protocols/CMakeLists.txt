# CMake configuration for Motion Tracking Real-Time Protocol (MTRTP)
# High-performance streaming protocol implementation

cmake_minimum_required(VERSION 3.15)
project(MTRTP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# COMPILER OPTIONS
# ============================================================================

# Performance optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Enable link-time optimization for release builds
if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ============================================================================
# DEPENDENCIES
# ============================================================================

# Protocol Buffers
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# LZ4 compression
find_library(LZ4_LIBRARY lz4)
if(NOT LZ4_LIBRARY)
    message(FATAL_ERROR "LZ4 library not found. Install with: sudo apt-get install liblz4-dev")
endif()

# Threads
find_package(Threads REQUIRED)

# POSIX real-time extensions
find_library(RT_LIBRARY rt)

# ============================================================================
# GENERATE PROTOCOL BUFFERS
# ============================================================================

set(PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/motion_protocol.proto)
set(PROTO_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

file(MAKE_DIRECTORY ${PROTO_OUTPUT_DIR})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Add generated files to include path
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROTO_OUTPUT_DIR})

# ============================================================================
# LIBRARY TARGET
# ============================================================================

add_library(mtrtp STATIC
    stream_manager.cpp
    ${PROTO_SRCS}
)

target_include_directories(mtrtp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROTO_OUTPUT_DIR}
)

target_link_libraries(mtrtp
    ${Protobuf_LIBRARIES}
    ${LZ4_LIBRARY}
    ${RT_LIBRARY}
    Threads::Threads
)

# ============================================================================
# EXAMPLE EXECUTABLES
# ============================================================================

# Motion data publisher example
add_executable(motion_publisher examples/motion_publisher.cpp)
target_link_libraries(motion_publisher mtrtp)

# Motion data subscriber example
add_executable(motion_subscriber examples/motion_subscriber.cpp)
target_link_libraries(motion_subscriber mtrtp)

# Performance benchmark
add_executable(stream_benchmark examples/stream_benchmark.cpp)
target_link_libraries(stream_benchmark mtrtp)

# ============================================================================
# TESTS
# ============================================================================

option(BUILD_TESTS "Build test suite" ON)

if(BUILD_TESTS)
    enable_testing()

    # Find GTest
    find_package(GTest)

    if(GTest_FOUND)
        add_executable(protocol_tests tests/protocol_tests.cpp)
        target_link_libraries(protocol_tests mtrtp GTest::GTest GTest::Main)
        add_test(NAME protocol_tests COMMAND protocol_tests)

        add_executable(performance_tests tests/performance_tests.cpp)
        target_link_libraries(performance_tests mtrtp GTest::GTest GTest::Main)
        add_test(NAME performance_tests COMMAND performance_tests)
    else()
        message(WARNING "GTest not found. Tests will not be built.")
    endif()
endif()

# ============================================================================
# INSTALLATION
# ============================================================================

install(TARGETS mtrtp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    ${PROTO_HDRS}
    stream_manager.cpp
    DESTINATION include/mtrtp
)

install(FILES
    motion_protocol.proto
    DESTINATION share/mtrtp/proto
)

# ============================================================================
# DOCUMENTATION
# ============================================================================

option(BUILD_DOCS "Build documentation" OFF)

if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            @ONLY
        )
        add_custom_target(docs
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
        )
    endif()
endif()

# ============================================================================
# PACKAGE CONFIGURATION
# ============================================================================

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MTRTPConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/MTRTPConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MTRTPConfig.cmake"
    INSTALL_DESTINATION lib/cmake/MTRTP
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MTRTPConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MTRTPConfigVersion.cmake"
    DESTINATION lib/cmake/MTRTP
)

# ============================================================================
# STATUS MESSAGES
# ============================================================================

message(STATUS "")
message(STATUS "MTRTP Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Protobuf Version: ${Protobuf_VERSION}")
message(STATUS "  LZ4 Library: ${LZ4_LIBRARY}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "")
