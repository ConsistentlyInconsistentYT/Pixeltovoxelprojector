cmake_minimum_required(VERSION 3.12)
project(Pixeltovoxelprojector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonLibs REQUIRED)

# Add pybind11
add_subdirectory(pybind11)

add_library(process_image_cpp SHARED process_image.cpp)
target_link_libraries(process_image_cpp PRIVATE pybind11::embed)

# Add this line to include the nlohmann directory
target_include_directories(process_image_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
