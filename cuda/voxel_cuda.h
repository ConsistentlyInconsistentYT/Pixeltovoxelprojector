#ifndef VOXEL_CUDA_H
#define VOXEL_CUDA_H

#include <cuda_runtime.h>
#include <vector>

// ============================================================================
// Structure Definitions
// ============================================================================

struct Vec3f {
    float x, y, z;

    __host__ __device__ Vec3f() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Mat3f {
    float m[9];
};

struct CameraParams {
    Vec3f position;
    Mat3f rotation;
    float fov_rad;
    int width;
    int height;
    int camera_id;
};

struct VoxelGridParams {
    int N;              // Grid size (NxNxN)
    float voxel_size;
    Vec3f grid_center;
    float* data;        // Device pointer to voxel data
};

struct MotionDetectionParams {
    float threshold;
    int width;
    int height;
};

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Core CUDA Functions
// ============================================================================

/**
 * Initialize CUDA device and streams
 * @param num_streams Number of CUDA streams to create
 * @param streams Output array of CUDA streams
 */
void initCudaStreams(int num_streams, cudaStream_t** streams);

/**
 * Cleanup CUDA streams
 * @param num_streams Number of streams
 * @param streams Array of CUDA streams to destroy
 */
void cleanupCudaStreams(int num_streams, cudaStream_t* streams);

/**
 * Allocate voxel grid on GPU
 * @param N Grid size (NxNxN)
 * @param d_voxel_grid Output device pointer
 */
void allocateVoxelGrid(int N, float** d_voxel_grid);

/**
 * Free voxel grid on GPU
 * @param d_voxel_grid Device pointer to free
 */
void freeVoxelGrid(float* d_voxel_grid);

/**
 * Clear/reset voxel grid to zeros
 * @param d_voxel_grid Device pointer to voxel grid
 * @param N Grid size
 * @param stream CUDA stream for async execution
 */
void clearVoxelGrid(float* d_voxel_grid, int N, cudaStream_t stream);

/**
 * Copy voxel grid from device to host
 * @param d_voxel_grid Device pointer
 * @param h_voxel_grid Host pointer
 * @param N Grid size
 */
void copyVoxelGridToHost(float* d_voxel_grid, float* h_voxel_grid, int N);

// ============================================================================
// Motion Detection Functions
// ============================================================================

/**
 * GPU-accelerated motion detection between two frames
 * @param d_prev_frame Previous frame on device
 * @param d_curr_frame Current frame on device
 * @param d_motion_mask Output motion mask (bool array)
 * @param d_diff Output difference values
 * @param width Frame width
 * @param height Frame height
 * @param threshold Motion detection threshold
 * @param stream CUDA stream for async execution
 */
void detectMotionGPU(
    const float* d_prev_frame,
    const float* d_curr_frame,
    bool* d_motion_mask,
    float* d_diff,
    int width,
    int height,
    float threshold,
    cudaStream_t stream
);

/**
 * Count number of changed pixels (for statistics)
 * @param d_motion_mask Motion mask on device
 * @param width Frame width
 * @param height Frame height
 * @param stream CUDA stream
 * @return Number of changed pixels
 */
int countChangedPixels(
    const bool* d_motion_mask,
    int width,
    int height,
    cudaStream_t stream
);

// ============================================================================
// Ray-Casting Functions
// ============================================================================

/**
 * GPU-accelerated voxel ray-casting with motion detection
 * Casts rays for pixels that have changed and accumulates into voxel grid
 * Uses shared memory optimization for voxel access
 *
 * @param d_frame Current frame data on device
 * @param d_motion_mask Motion mask on device
 * @param d_diff Difference values on device
 * @param camera Camera parameters
 * @param voxel_params Voxel grid parameters
 * @param stream CUDA stream for async execution
 */
void castRaysMotionGPU(
    const float* d_frame,
    const bool* d_motion_mask,
    const float* d_diff,
    const CameraParams& camera,
    const VoxelGridParams& voxel_params,
    cudaStream_t stream
);

/**
 * GPU-accelerated full-frame ray-casting (no motion detection)
 * Casts rays for all pixels in the frame
 *
 * @param d_frame Frame data on device
 * @param camera Camera parameters
 * @param voxel_params Voxel grid parameters
 * @param stream CUDA stream for async execution
 */
void castRaysFullFrameGPU(
    const float* d_frame,
    const CameraParams& camera,
    const VoxelGridParams& voxel_params,
    cudaStream_t stream
);

// ============================================================================
// Multi-Camera Processing
// ============================================================================

/**
 * Process multiple cameras in parallel using CUDA streams
 * Each camera gets its own stream for concurrent processing
 *
 * @param h_prev_frames Host array of previous frames (one per camera)
 * @param h_curr_frames Host array of current frames (one per camera)
 * @param cameras Array of camera parameters
 * @param voxel_params Voxel grid parameters
 * @param num_cameras Number of cameras
 * @param motion_threshold Motion detection threshold
 * @param streams Array of CUDA streams
 */
void processMultipleCameras(
    const std::vector<float*>& h_prev_frames,
    const std::vector<float*>& h_curr_frames,
    const std::vector<CameraParams>& cameras,
    const VoxelGridParams& voxel_params,
    int num_cameras,
    float motion_threshold,
    cudaStream_t* streams
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get CUDA device properties and print info
 * @param device_id Device ID (default 0)
 */
void printCudaDeviceInfo(int device_id = 0);

/**
 * Check if device supports required compute capability
 * @param required_major Required major version
 * @param required_minor Required minor version
 * @param device_id Device ID
 * @return true if supported
 */
bool checkComputeCapability(int required_major, int required_minor, int device_id = 0);

/**
 * Optimize CUDA settings for 8K video processing
 * Sets cache preferences and shared memory configurations
 */
void optimizeFor8K();

/**
 * Get optimal block and grid dimensions for given image size
 * @param width Image width
 * @param height Image height
 * @param block_dim Output block dimensions
 * @param grid_dim Output grid dimensions
 */
void getOptimalDimensions(
    int width,
    int height,
    dim3& block_dim,
    dim3& grid_dim
);

/**
 * Benchmark function to measure ray-casting performance
 * @param width Frame width
 * @param height Frame height
 * @param num_cameras Number of cameras
 * @param voxel_grid_size Voxel grid size
 * @param num_iterations Number of iterations to run
 */
void benchmarkRayCasting(
    int width,
    int height,
    int num_cameras,
    int voxel_grid_size,
    int num_iterations = 100
);

// ============================================================================
// Advanced Features
// ============================================================================

/**
 * Apply 3D Gaussian blur to voxel grid on GPU
 * @param d_voxel_grid Input/output voxel grid
 * @param d_temp_grid Temporary buffer (same size as voxel grid)
 * @param N Grid size
 * @param sigma Gaussian sigma
 * @param stream CUDA stream
 */
void applyGaussianBlurGPU(
    float* d_voxel_grid,
    float* d_temp_grid,
    int N,
    float sigma,
    cudaStream_t stream
);

/**
 * Find local maxima in voxel grid (for object detection)
 * @param d_voxel_grid Input voxel grid
 * @param d_maxima Output maxima locations
 * @param d_maxima_values Output maxima values
 * @param N Grid size
 * @param threshold Minimum threshold for maxima
 * @param stream CUDA stream
 * @return Number of maxima found
 */
int findLocalMaximaGPU(
    const float* d_voxel_grid,
    int* d_maxima,
    float* d_maxima_values,
    int N,
    float threshold,
    cudaStream_t stream
);

/**
 * Compute histogram of voxel values on GPU
 * @param d_voxel_grid Input voxel grid
 * @param d_histogram Output histogram
 * @param N Grid size
 * @param num_bins Number of histogram bins
 * @param stream CUDA stream
 */
void computeHistogramGPU(
    const float* d_voxel_grid,
    int* d_histogram,
    int N,
    int num_bins,
    cudaStream_t stream
);

#endif // VOXEL_CUDA_H
