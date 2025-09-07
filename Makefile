CPPFLAGS =
CFLAGS = -Wall -W -std=c++17 -O2
LDFLAGS =

all: ray_voxel

example: ray_voxel
	ray_voxel motionimages/metadata.json motionimages voxel_grid.bin

ray_voxel:
	$(CXX) $(CPPFLAGS) $(CFLAGS) ray_voxel.cpp -o ray_voxel

clean:
	$(RM) ray_voxel
