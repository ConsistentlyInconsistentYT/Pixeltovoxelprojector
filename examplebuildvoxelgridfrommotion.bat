@echo off

rem Build with g++
g++ -W -Wall -std=c++17 -O2 ray_voxel.cpp -o ray_voxel
if %ERRORLEVEL% NEQ 0 (
    echo [Error] Compilation failed
    pause
    exit /b %ERRORLEVEL%
)

echo [Info] Compilation succeeded.

rem Now run the compiled program:
rem Usage: ray_voxel <metadata.json> <image_folder> <output_voxel_bin>
ray_voxel motionimages/metadata.json motionimages voxel_grid.bin
