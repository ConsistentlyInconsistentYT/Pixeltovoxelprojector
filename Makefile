CPPFLAGS =
CFLAGS = -Wall -W -std=c++17 -O3
LDFLAGS =

BINDIR = build

all: $(BINDIR)/ray_voxel

$(BINDIR)/ray_voxel: ray_voxel.cpp | $(BINDIR)
	$(CXX) $(CPPFLAGS) $(CFLAGS) ray_voxel.cpp -o $(BINDIR)/ray_voxel

$(BINDIR):
	mkdir -p $(BINDIR)

example: $(BINDIR)/ray_voxel
	./build/ray_voxel motionimages/metadata.json motionimages voxel_grid.bin

clean:
	$(RM) $(BINDIR)/ray_voxel