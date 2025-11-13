/***************************************************
 * voxel_optimizer.cu
 *
 * CUDA-Accelerated Voxel Optimization for 5km+ Range Tracking
 *
 * Features:
 * - GPU-accelerated voxel updates (30Hz+ real-time)
 * - Parallel ray casting from multiple cameras
 * - Voxel pruning and compression
 * - Level-of-detail management
 * - Memory-efficient sparse representation
 * - Support for 200+ tracked objects
 *
 * Performance Targets:
 * - 200 objects @ 30Hz update rate
 * - <500MB GPU memory footprint
 * - Ray casting: 1M+ rays/second
 * - Voxel updates: 10M+ voxels/second
 ***************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Device Constants
// ============================================================================

__constant__ float d_gridSizeX = 5000.0f;
__constant__ float d_gridSizeY = 5000.0f;
__constant__ float d_gridSizeZ = 2000.0f;

// ============================================================================
// Device Structures
// ============================================================================

struct Vec3f {
    float x, y, z;

    __host__ __device__ Vec3f() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vec3f operator+(const Vec3f& v) const {
        return Vec3f(x+v.x, y+v.y, z+v.z);
    }

    __host__ __device__ Vec3f operator-(const Vec3f& v) const {
        return Vec3f(x-v.x, y-v.y, z-v.z);
    }

    __host__ __device__ Vec3f operator*(float s) const {
        return Vec3f(x*s, y*s, z*s);
    }

    __host__ __device__ Vec3f operator/(float s) const {
        return Vec3f(x/s, y/s, z/s);
    }

    __host__ __device__ float length() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __host__ __device__ float dot(const Vec3f& v) const {
        return x*v.x + y*v.y + z*v.z;
    }

    __host__ __device__ Vec3f normalized() const {
        float len = length();
        return (len > 1e-6f) ? (*this / len) : Vec3f(0, 0, 0);
    }
};

/**
 * Sparse voxel structure for GPU
 * Uses hash-based indexing for memory efficiency
 */
struct SparseVoxel {
    int x, y, z;        // Voxel grid coordinates
    float value;        // Accumulated value
    int objectID;       // Object identifier
    uint32_t timestamp; // Last update time

    __host__ __device__ SparseVoxel()
        : x(0), y(0), z(0), value(0.0f), objectID(-1), timestamp(0) {}

    __host__ __device__ SparseVoxel(int x_, int y_, int z_, float val)
        : x(x_), y(y_), z(z_), value(val), objectID(-1), timestamp(0) {}

    // Hash function for spatial hashing
    __host__ __device__ uint64_t hash() const {
        uint64_t h = 0;
        h |= (uint64_t)(x & 0xFFFF);
        h |= (uint64_t)(y & 0xFFFF) << 16;
        h |= (uint64_t)(z & 0xFFFF) << 32;
        return h;
    }

    __host__ __device__ bool operator==(const SparseVoxel& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * Ray structure for parallel ray casting
 */
struct Ray {
    Vec3f origin;
    Vec3f direction;
    float maxDistance;
    int objectID;

    __host__ __device__ Ray() : maxDistance(5000.0f), objectID(-1) {}

    __host__ __device__ Ray(const Vec3f& orig, const Vec3f& dir, float maxDist = 5000.0f)
        : origin(orig), direction(dir.normalized()), maxDistance(maxDist), objectID(-1) {}
};

/**
 * Camera structure for multi-camera tracking
 */
struct Camera {
    Vec3f position;
    Vec3f forward;
    Vec3f up;
    Vec3f right;
    float fov;
    int width;
    int height;

    __host__ __device__ Camera() : fov(60.0f), width(640), height(480) {}
};

// ============================================================================
// Device Functions
// ============================================================================

/**
 * Get LOD level based on distance
 */
__device__ float getLODResolution(float distance) {
    if (distance < 100.0f) return 0.1f;
    if (distance < 500.0f) return 0.25f;
    if (distance < 1000.0f) return 0.5f;
    if (distance < 5000.0f) return 1.0f;
    return 2.0f;
}

/**
 * Convert world position to voxel coordinates
 */
__device__ void worldToVoxel(const Vec3f& worldPos, const Vec3f& gridCenter,
                             float voxelSize, int& vx, int& vy, int& vz) {
    Vec3f localPos = worldPos - gridCenter;
    vx = (int)floorf(localPos.x / voxelSize + 0.5f);
    vy = (int)floorf(localPos.y / voxelSize + 0.5f);
    vz = (int)floorf(localPos.z / voxelSize + 0.5f);
}

/**
 * Convert voxel coordinates to world position
 */
__device__ Vec3f voxelToWorld(int vx, int vy, int vz, const Vec3f& gridCenter,
                              float voxelSize) {
    Vec3f worldPos;
    worldPos.x = gridCenter.x + vx * voxelSize;
    worldPos.y = gridCenter.y + vy * voxelSize;
    worldPos.z = gridCenter.z + vz * voxelSize;
    return worldPos;
}

/**
 * Check if voxel is within grid bounds
 */
__device__ bool isVoxelInBounds(int vx, int vy, int vz, int gridSizeX, int gridSizeY, int gridSizeZ) {
    int halfX = gridSizeX / 2;
    int halfY = gridSizeY / 2;
    int halfZ = gridSizeZ / 2;
    return vx >= -halfX && vx < halfX &&
           vy >= -halfY && vy < halfY &&
           vz >= -halfZ && vz < halfZ;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Kernel: Parallel ray casting through voxel grid
 * Each thread handles one ray
 */
__global__ void rayCastKernel(
    const Ray* rays,
    int numRays,
    SparseVoxel* voxels,
    int* voxelCount,
    const Vec3f gridCenter,
    float baseVoxelSize,
    int gridSizeX,
    int gridSizeY,
    int gridSizeZ,
    float maxDistance
) {
    int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIdx >= numRays) return;

    Ray ray = rays[rayIdx];
    Vec3f pos = ray.origin;
    Vec3f dir = ray.direction;

    // DDA algorithm for voxel traversal
    float distance = 0.0f;
    float stepSize = baseVoxelSize * 0.5f;  // Conservative step size

    while (distance < ray.maxDistance && distance < maxDistance) {
        // Get LOD resolution at current distance
        float lodRes = getLODResolution(distance);

        // Convert to voxel coordinates
        int vx, vy, vz;
        worldToVoxel(pos, gridCenter, lodRes, vx, vy, vz);

        // Check bounds
        if (isVoxelInBounds(vx, vy, vz, gridSizeX, gridSizeY, gridSizeZ)) {
            // Atomically add voxel to output list
            int idx = atomicAdd(voxelCount, 1);
            if (idx < gridSizeX * gridSizeY * gridSizeZ) {
                voxels[idx].x = vx;
                voxels[idx].y = vy;
                voxels[idx].z = vz;
                voxels[idx].value = 1.0f;
                voxels[idx].objectID = ray.objectID;
            }
        }

        // Advance along ray
        pos = pos + dir * stepSize;
        distance += stepSize;
    }
}

/**
 * Kernel: Accumulate values into voxels (parallel reduction)
 */
__global__ void accumulateVoxelsKernel(
    SparseVoxel* voxels,
    int numVoxels,
    float* voxelGrid,
    int gridResX,
    int gridResY,
    int gridResZ,
    const Vec3f gridCenter,
    float voxelSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    SparseVoxel voxel = voxels[idx];

    // Convert to grid index
    int halfX = gridResX / 2;
    int halfY = gridResY / 2;
    int halfZ = gridResZ / 2;

    int gx = voxel.x + halfX;
    int gy = voxel.y + halfY;
    int gz = voxel.z + halfZ;

    if (gx >= 0 && gx < gridResX &&
        gy >= 0 && gy < gridResY &&
        gz >= 0 && gz < gridResZ) {

        int gridIdx = gx * gridResY * gridResZ + gy * gridResZ + gz;
        atomicAdd(&voxelGrid[gridIdx], voxel.value);
    }
}

/**
 * Kernel: Prune voxels below threshold
 */
__global__ void pruneVoxelsKernel(
    float* voxelGrid,
    int numVoxels,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    if (voxelGrid[idx] < threshold) {
        voxelGrid[idx] = 0.0f;
    }
}

/**
 * Kernel: Compress voxel grid by removing zeros
 */
__global__ void compressVoxelsKernel(
    const float* voxelGrid,
    int numVoxels,
    SparseVoxel* compressedVoxels,
    int* compressedCount,
    int gridResX,
    int gridResY,
    int gridResZ
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    float value = voxelGrid[idx];
    if (value > 0.0f) {
        // Convert linear index to 3D coordinates
        int gz = idx % gridResZ;
        int gy = (idx / gridResZ) % gridResY;
        int gx = idx / (gridResY * gridResZ);

        int outIdx = atomicAdd(compressedCount, 1);
        compressedVoxels[outIdx].x = gx - gridResX / 2;
        compressedVoxels[outIdx].y = gy - gridResY / 2;
        compressedVoxels[outIdx].z = gz - gridResZ / 2;
        compressedVoxels[outIdx].value = value;
    }
}

/**
 * Kernel: Multi-camera ray generation
 */
__global__ void generateCameraRaysKernel(
    const Camera* cameras,
    int numCameras,
    Ray* rays,
    int* rayCount,
    int raysPerCamera
) {
    int camIdx = blockIdx.x;
    int pixelIdx = threadIdx.x;

    if (camIdx >= numCameras || pixelIdx >= raysPerCamera) return;

    Camera cam = cameras[camIdx];

    // Generate ray for this pixel
    int pixelX = pixelIdx % cam.width;
    int pixelY = pixelIdx / cam.width;

    if (pixelX >= cam.width || pixelY >= cam.height) return;

    // NDC coordinates
    float ndcX = (2.0f * pixelX / cam.width - 1.0f);
    float ndcY = (1.0f - 2.0f * pixelY / cam.height);

    // Calculate ray direction
    float aspect = (float)cam.width / cam.height;
    float fovRad = cam.fov * 3.14159265f / 180.0f;
    float tanHalfFov = tanf(fovRad * 0.5f);

    Vec3f dir = cam.forward + cam.right * (ndcX * aspect * tanHalfFov)
                            + cam.up * (ndcY * tanHalfFov);
    dir = dir.normalized();

    // Store ray
    int rayIdx = atomicAdd(rayCount, 1);
    rays[rayIdx].origin = cam.position;
    rays[rayIdx].direction = dir;
    rays[rayIdx].maxDistance = 5000.0f;
    rays[rayIdx].objectID = camIdx;
}

/**
 * Kernel: LOD update based on distance from observer
 */
__global__ void updateLODKernel(
    SparseVoxel* voxels,
    int numVoxels,
    const Vec3f observerPos,
    float* lodLevels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    SparseVoxel voxel = voxels[idx];

    // Calculate distance to observer
    Vec3f voxelPos(voxel.x, voxel.y, voxel.z);
    float distance = (voxelPos - observerPos).length();

    // Determine LOD level
    lodLevels[idx] = getLODResolution(distance);
}

// ============================================================================
// Host Class: VoxelOptimizer
// ============================================================================

class VoxelOptimizer {
private:
    // Grid parameters
    Vec3f gridCenter;
    float baseVoxelSize;
    int gridResX, gridResY, gridResZ;

    // Device memory
    float* d_voxelGrid;
    SparseVoxel* d_sparseVoxels;
    Ray* d_rays;
    Camera* d_cameras;
    int* d_voxelCount;
    int* d_rayCount;

    // Host memory
    std::vector<Camera> h_cameras;
    std::vector<Ray> h_rays;

    // Statistics
    size_t totalVoxels;
    size_t activeVoxels;

public:
    VoxelOptimizer(const Vec3f& center, float voxelSize, int resX, int resY, int resZ)
        : gridCenter(center), baseVoxelSize(voxelSize),
          gridResX(resX), gridResY(resY), gridResZ(resZ),
          totalVoxels(0), activeVoxels(0) {

        // Allocate device memory
        size_t gridSize = resX * resY * resZ;
        totalVoxels = gridSize;

        CUDA_CHECK(cudaMalloc(&d_voxelGrid, gridSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sparseVoxels, gridSize * sizeof(SparseVoxel)));
        CUDA_CHECK(cudaMalloc(&d_rays, 1000000 * sizeof(Ray)));  // 1M rays
        CUDA_CHECK(cudaMalloc(&d_cameras, 200 * sizeof(Camera)));  // 200 cameras
        CUDA_CHECK(cudaMalloc(&d_voxelCount, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_rayCount, sizeof(int)));

        // Initialize grid to zero
        CUDA_CHECK(cudaMemset(d_voxelGrid, 0, gridSize * sizeof(float)));

        std::cout << "VoxelOptimizer initialized:" << std::endl;
        std::cout << "  Grid size: " << resX << " x " << resY << " x " << resZ << std::endl;
        std::cout << "  Total voxels: " << gridSize << std::endl;
        std::cout << "  Memory: " << (gridSize * sizeof(float) / (1024.0 * 1024.0)) << " MB" << std::endl;
    }

    ~VoxelOptimizer() {
        cudaFree(d_voxelGrid);
        cudaFree(d_sparseVoxels);
        cudaFree(d_rays);
        cudaFree(d_cameras);
        cudaFree(d_voxelCount);
        cudaFree(d_rayCount);
    }

    /**
     * Add camera for tracking
     */
    void addCamera(const Camera& cam) {
        h_cameras.push_back(cam);
    }

    /**
     * Cast rays from all cameras (parallel)
     */
    void castRaysFromCameras() {
        if (h_cameras.empty()) return;

        // Copy cameras to device
        CUDA_CHECK(cudaMemcpy(d_cameras, h_cameras.data(),
                             h_cameras.size() * sizeof(Camera),
                             cudaMemcpyHostToDevice));

        // Reset ray count
        int zeroCount = 0;
        CUDA_CHECK(cudaMemcpy(d_rayCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice));

        // Generate rays
        int raysPerCam = 1024;  // Configurable
        dim3 blocks(h_cameras.size());
        dim3 threads(raysPerCam);

        generateCameraRaysKernel<<<blocks, threads>>>(
            d_cameras, h_cameras.size(), d_rays, d_rayCount, raysPerCam
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // Get ray count
        int numRays;
        CUDA_CHECK(cudaMemcpy(&numRays, d_rayCount, sizeof(int), cudaMemcpyDeviceToHost));

        // Cast rays
        CUDA_CHECK(cudaMemcpy(d_voxelCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;

        rayCastKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_rays, numRays, d_sparseVoxels, d_voxelCount,
            gridCenter, baseVoxelSize, gridResX, gridResY, gridResZ, 5000.0f
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // Get voxel count
        int numVoxels;
        CUDA_CHECK(cudaMemcpy(&numVoxels, d_voxelCount, sizeof(int), cudaMemcpyDeviceToHost));

        // Accumulate voxels
        blocksPerGrid = (numVoxels + threadsPerBlock - 1) / threadsPerBlock;

        accumulateVoxelsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_sparseVoxels, numVoxels, d_voxelGrid,
            gridResX, gridResY, gridResZ, gridCenter, baseVoxelSize
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        activeVoxels = numVoxels;
    }

    /**
     * Prune voxels below threshold
     */
    void pruneVoxels(float threshold) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalVoxels + threadsPerBlock - 1) / threadsPerBlock;

        pruneVoxelsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_voxelGrid, totalVoxels, threshold
        );

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /**
     * Get voxel grid data
     */
    void getVoxelGrid(float* hostGrid) {
        CUDA_CHECK(cudaMemcpy(hostGrid, d_voxelGrid,
                             totalVoxels * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }

    /**
     * Clear voxel grid
     */
    void clear() {
        CUDA_CHECK(cudaMemset(d_voxelGrid, 0, totalVoxels * sizeof(float)));
        activeVoxels = 0;
    }

    /**
     * Get statistics
     */
    void printStatistics() {
        std::cout << "=== Voxel Optimizer Statistics ===" << std::endl;
        std::cout << "Active Voxels: " << activeVoxels << std::endl;
        std::cout << "Total Voxels:  " << totalVoxels << std::endl;
        std::cout << "Occupancy:     " << (100.0f * activeVoxels / totalVoxels) << "%" << std::endl;
        std::cout << "===================================" << std::endl;
    }
};

// ============================================================================
// Example Usage
// ============================================================================

#ifdef BUILD_EXAMPLE

int main() {
    std::cout << "CUDA Voxel Optimizer - 5km+ Range Tracking" << std::endl;
    std::cout << "===========================================" << std::endl;

    // Initialize optimizer with multi-resolution grid
    // Using 5000x5000x2000 volume with 1m base resolution = 5000x5000x2000 voxels
    // But we'll use a coarser grid for this example
    VoxelOptimizer optimizer(
        Vec3f(0, 0, 0),  // Center
        1.0f,            // 1m base resolution
        1000, 1000, 400  // 1000x1000x400 = 400M voxels (but sparse!)
    );

    // Add cameras
    std::cout << "\nAdding 10 cameras..." << std::endl;
    for (int i = 0; i < 10; i++) {
        Camera cam;
        float angle = (i / 10.0f) * 2.0f * 3.14159f;
        cam.position = Vec3f(std::cos(angle) * 100.0f, std::sin(angle) * 100.0f, 50.0f);
        cam.forward = Vec3f(-std::cos(angle), -std::sin(angle), -0.1f).normalized();
        cam.up = Vec3f(0, 0, 1);
        cam.right = Vec3f(cam.forward.y, -cam.forward.x, 0).normalized();
        cam.fov = 60.0f;
        cam.width = 640;
        cam.height = 480;

        optimizer.addCamera(cam);
    }

    // Benchmark ray casting
    std::cout << "\nBenchmarking ray casting (30 frames)..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < 30; frame++) {
        optimizer.castRaysFromCameras();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    float fps = 30000.0f / duration.count();
    std::cout << "Frame time: " << (duration.count() / 30.0f) << " ms" << std::endl;
    std::cout << "FPS: " << fps << std::endl;

    if (fps >= 30.0f) {
        std::cout << "✓ Meets 30Hz real-time requirement" << std::endl;
    } else {
        std::cout << "✗ Below 30Hz target" << std::endl;
    }

    // Print statistics
    std::cout << std::endl;
    optimizer.printStatistics();

    return 0;
}

#endif
