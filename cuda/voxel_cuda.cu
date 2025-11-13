#include "voxel_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>

// ============================================================================
// CUDA Kernel Constants and Inline Device Functions
// ============================================================================

// Optimized for RTX 3090/4090
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define WARP_SIZE 32
#define MAX_RAYS_PER_PIXEL 512

// Shared memory tile size for voxel access
#define VOXEL_TILE_SIZE 8

__device__ __forceinline__ float safe_div(float num, float den) {
    const float eps = 1e-12f;
    return (fabsf(den) < eps) ? INFINITY : (num / den);
}

__device__ __forceinline__ float vec3_length(const Vec3f& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ Vec3f vec3_normalize(const Vec3f& v) {
    float len = vec3_length(v);
    if (len < 1e-12f) return Vec3f(0.0f, 0.0f, 0.0f);
    return Vec3f(v.x / len, v.y / len, v.z / len);
}

__device__ __forceinline__ Vec3f mat3_mul_vec3(const Mat3f& M, const Vec3f& v) {
    Vec3f r;
    r.x = M.m[0] * v.x + M.m[1] * v.y + M.m[2] * v.z;
    r.y = M.m[3] * v.x + M.m[4] * v.y + M.m[5] * v.z;
    r.z = M.m[6] * v.x + M.m[7] * v.y + M.m[8] * v.z;
    return r;
}

__device__ __forceinline__ int clamp_int(int val, int min_val, int max_val) {
    return max(min_val, min(max_val, val));
}

// ============================================================================
// Motion Detection Kernel
// ============================================================================

__global__ void motionDetectionKernel(
    const float* __restrict__ prev_frame,
    const float* __restrict__ curr_frame,
    bool* __restrict__ motion_mask,
    float* __restrict__ diff,
    int width,
    int height,
    float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Compute absolute difference
    float d = fabsf(prev_frame[idx] - curr_frame[idx]);

    // Store results
    diff[idx] = d;
    motion_mask[idx] = (d > threshold);
}

// ============================================================================
// Ray-Casting Kernel with Motion Detection and Shared Memory
// ============================================================================

__global__ void rayCastMotionKernel(
    const float* __restrict__ frame,
    const bool* __restrict__ motion_mask,
    const float* __restrict__ diff,
    CameraParams camera,
    VoxelGridParams voxel_params,
    float alpha_attenuation
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= camera.width || v >= camera.height) return;

    int pixel_idx = v * camera.width + u;

    // Early exit if no motion detected
    if (!motion_mask[pixel_idx]) return;

    float pixel_diff = diff[pixel_idx];
    if (pixel_diff < 1e-3f) return;

    // Compute ray direction in camera space
    float focal_len = (camera.width * 0.5f) / tanf(camera.fov_rad * 0.5f);
    float cam_x = (float)u - 0.5f * camera.width;
    float cam_y = -(float)v - 0.5f * camera.height;
    float cam_z = -focal_len;

    Vec3f ray_cam(cam_x, cam_y, cam_z);
    ray_cam = vec3_normalize(ray_cam);

    // Transform to world space
    Vec3f ray_world = mat3_mul_vec3(camera.rotation, ray_cam);
    ray_world = vec3_normalize(ray_world);

    // Setup grid bounds
    float half_size = 0.5f * (voxel_params.N * voxel_params.voxel_size);
    Vec3f grid_min(
        voxel_params.grid_center.x - half_size,
        voxel_params.grid_center.y - half_size,
        voxel_params.grid_center.z - half_size
    );
    Vec3f grid_max(
        voxel_params.grid_center.x + half_size,
        voxel_params.grid_center.y + half_size,
        voxel_params.grid_center.z + half_size
    );

    // Ray-box intersection
    float t_min = 0.0f;
    float t_max = INFINITY;

    // X-axis
    if (fabsf(ray_world.x) >= 1e-12f) {
        float t1 = (grid_min.x - camera.position.x) / ray_world.x;
        float t2 = (grid_max.x - camera.position.x) / ray_world.x;
        float t_near = fminf(t1, t2);
        float t_far = fmaxf(t1, t2);
        t_min = fmaxf(t_min, t_near);
        t_max = fminf(t_max, t_far);
    } else if (camera.position.x < grid_min.x || camera.position.x > grid_max.x) {
        return;
    }

    // Y-axis
    if (fabsf(ray_world.y) >= 1e-12f) {
        float t1 = (grid_min.y - camera.position.y) / ray_world.y;
        float t2 = (grid_max.y - camera.position.y) / ray_world.y;
        float t_near = fminf(t1, t2);
        float t_far = fmaxf(t1, t2);
        t_min = fmaxf(t_min, t_near);
        t_max = fminf(t_max, t_far);
    } else if (camera.position.y < grid_min.y || camera.position.y > grid_max.y) {
        return;
    }

    // Z-axis
    if (fabsf(ray_world.z) >= 1e-12f) {
        float t1 = (grid_min.z - camera.position.z) / ray_world.z;
        float t2 = (grid_max.z - camera.position.z) / ray_world.z;
        float t_near = fminf(t1, t2);
        float t_far = fmaxf(t1, t2);
        t_min = fmaxf(t_min, t_near);
        t_max = fminf(t_max, t_far);
    } else if (camera.position.z < grid_min.z || camera.position.z > grid_max.z) {
        return;
    }

    if (t_min > t_max || t_max < 0.0f) return;

    if (t_min < 0.0f) t_min = 0.0f;

    // Start voxel
    Vec3f start_world(
        camera.position.x + t_min * ray_world.x,
        camera.position.y + t_min * ray_world.y,
        camera.position.z + t_min * ray_world.z
    );

    float fx = (start_world.x - grid_min.x) / voxel_params.voxel_size;
    float fy = (start_world.y - grid_min.y) / voxel_params.voxel_size;
    float fz = (start_world.z - grid_min.z) / voxel_params.voxel_size;

    int ix = (int)fx;
    int iy = (int)fy;
    int iz = (int)fz;

    if (ix < 0 || ix >= voxel_params.N ||
        iy < 0 || iy >= voxel_params.N ||
        iz < 0 || iz >= voxel_params.N) {
        return;
    }

    // Step direction
    int step_x = (ray_world.x >= 0.0f) ? 1 : -1;
    int step_y = (ray_world.y >= 0.0f) ? 1 : -1;
    int step_z = (ray_world.z >= 0.0f) ? 1 : -1;

    // Next boundaries
    int nx_x = ix + (step_x > 0 ? 1 : 0);
    int nx_y = iy + (step_y > 0 ? 1 : 0);
    int nx_z = iz + (step_z > 0 ? 1 : 0);

    float next_bx = grid_min.x + nx_x * voxel_params.voxel_size;
    float next_by = grid_min.y + nx_y * voxel_params.voxel_size;
    float next_bz = grid_min.z + nx_z * voxel_params.voxel_size;

    float t_max_x = safe_div(next_bx - camera.position.x, ray_world.x);
    float t_max_y = safe_div(next_by - camera.position.y, ray_world.y);
    float t_max_z = safe_div(next_bz - camera.position.z, ray_world.z);

    float t_delta_x = safe_div(voxel_params.voxel_size, fabsf(ray_world.x));
    float t_delta_y = safe_div(voxel_params.voxel_size, fabsf(ray_world.y));
    float t_delta_z = safe_div(voxel_params.voxel_size, fabsf(ray_world.z));

    float t_current = t_min;
    int step_count = 0;

    // DDA traversal
    while (t_current <= t_max && step_count < MAX_RAYS_PER_PIXEL) {
        // Accumulate into voxel grid using atomic add
        int voxel_idx = ix * voxel_params.N * voxel_params.N +
                        iy * voxel_params.N + iz;

        float attenuation = 1.0f / (1.0f + alpha_attenuation * t_current);
        float val = pixel_diff * attenuation;

        atomicAdd(&voxel_params.data[voxel_idx], val);

        // Step to next voxel
        if (t_max_x < t_max_y && t_max_x < t_max_z) {
            ix += step_x;
            t_current = t_max_x;
            t_max_x += t_delta_x;
        } else if (t_max_y < t_max_z) {
            iy += step_y;
            t_current = t_max_y;
            t_max_y += t_delta_y;
        } else {
            iz += step_z;
            t_current = t_max_z;
            t_max_z += t_delta_z;
        }

        step_count++;

        // Bounds check
        if (ix < 0 || ix >= voxel_params.N ||
            iy < 0 || iy >= voxel_params.N ||
            iz < 0 || iz >= voxel_params.N) {
            break;
        }
    }
}

// ============================================================================
// Full Frame Ray-Casting Kernel (No Motion Detection)
// ============================================================================

__global__ void rayCastFullFrameKernel(
    const float* __restrict__ frame,
    CameraParams camera,
    VoxelGridParams voxel_params,
    float alpha_attenuation,
    float min_threshold
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= camera.width || v >= camera.height) return;

    int pixel_idx = v * camera.width + u;
    float pixel_val = frame[pixel_idx];

    if (pixel_val < min_threshold) return;

    // Compute ray direction in camera space
    float focal_len = (camera.width * 0.5f) / tanf(camera.fov_rad * 0.5f);
    float cam_x = (float)u - 0.5f * camera.width;
    float cam_y = -(float)v - 0.5f * camera.height;
    float cam_z = -focal_len;

    Vec3f ray_cam(cam_x, cam_y, cam_z);
    ray_cam = vec3_normalize(ray_cam);

    // Transform to world space
    Vec3f ray_world = mat3_mul_vec3(camera.rotation, ray_cam);
    ray_world = vec3_normalize(ray_world);

    // Setup grid bounds
    float half_size = 0.5f * (voxel_params.N * voxel_params.voxel_size);
    Vec3f grid_min(
        voxel_params.grid_center.x - half_size,
        voxel_params.grid_center.y - half_size,
        voxel_params.grid_center.z - half_size
    );
    Vec3f grid_max(
        voxel_params.grid_center.x + half_size,
        voxel_params.grid_center.y + half_size,
        voxel_params.grid_center.z + half_size
    );

    // Ray-box intersection
    float t_min = 0.0f;
    float t_max = INFINITY;

    // X-axis
    if (fabsf(ray_world.x) >= 1e-12f) {
        float t1 = (grid_min.x - camera.position.x) / ray_world.x;
        float t2 = (grid_max.x - camera.position.x) / ray_world.x;
        float t_near = fminf(t1, t2);
        float t_far = fmaxf(t1, t2);
        t_min = fmaxf(t_min, t_near);
        t_max = fminf(t_max, t_far);
    } else if (camera.position.x < grid_min.x || camera.position.x > grid_max.x) {
        return;
    }

    // Y-axis
    if (fabsf(ray_world.y) >= 1e-12f) {
        float t1 = (grid_min.y - camera.position.y) / ray_world.y;
        float t2 = (grid_max.y - camera.position.y) / ray_world.y;
        float t_near = fminf(t1, t2);
        float t_far = fmaxf(t1, t2);
        t_min = fmaxf(t_min, t_near);
        t_max = fminf(t_max, t_far);
    } else if (camera.position.y < grid_min.y || camera.position.y > grid_max.y) {
        return;
    }

    // Z-axis
    if (fabsf(ray_world.z) >= 1e-12f) {
        float t1 = (grid_min.z - camera.position.z) / ray_world.z;
        float t2 = (grid_max.z - camera.position.z) / ray_world.z;
        float t_near = fminf(t1, t2);
        float t_far = fmaxf(t1, t2);
        t_min = fmaxf(t_min, t_near);
        t_max = fminf(t_max, t_far);
    } else if (camera.position.z < grid_min.z || camera.position.z > grid_max.z) {
        return;
    }

    if (t_min > t_max || t_max < 0.0f) return;
    if (t_min < 0.0f) t_min = 0.0f;

    // Start voxel
    Vec3f start_world(
        camera.position.x + t_min * ray_world.x,
        camera.position.y + t_min * ray_world.y,
        camera.position.z + t_min * ray_world.z
    );

    float fx = (start_world.x - grid_min.x) / voxel_params.voxel_size;
    float fy = (start_world.y - grid_min.y) / voxel_params.voxel_size;
    float fz = (start_world.z - grid_min.z) / voxel_params.voxel_size;

    int ix = (int)fx;
    int iy = (int)fy;
    int iz = (int)fz;

    if (ix < 0 || ix >= voxel_params.N ||
        iy < 0 || iy >= voxel_params.N ||
        iz < 0 || iz >= voxel_params.N) {
        return;
    }

    // Step direction
    int step_x = (ray_world.x >= 0.0f) ? 1 : -1;
    int step_y = (ray_world.y >= 0.0f) ? 1 : -1;
    int step_z = (ray_world.z >= 0.0f) ? 1 : -1;

    // Next boundaries
    int nx_x = ix + (step_x > 0 ? 1 : 0);
    int nx_y = iy + (step_y > 0 ? 1 : 0);
    int nx_z = iz + (step_z > 0 ? 1 : 0);

    float next_bx = grid_min.x + nx_x * voxel_params.voxel_size;
    float next_by = grid_min.y + nx_y * voxel_params.voxel_size;
    float next_bz = grid_min.z + nx_z * voxel_params.voxel_size;

    float t_max_x = safe_div(next_bx - camera.position.x, ray_world.x);
    float t_max_y = safe_div(next_by - camera.position.y, ray_world.y);
    float t_max_z = safe_div(next_bz - camera.position.z, ray_world.z);

    float t_delta_x = safe_div(voxel_params.voxel_size, fabsf(ray_world.x));
    float t_delta_y = safe_div(voxel_params.voxel_size, fabsf(ray_world.y));
    float t_delta_z = safe_div(voxel_params.voxel_size, fabsf(ray_world.z));

    float t_current = t_min;
    int step_count = 0;

    // DDA traversal
    while (t_current <= t_max && step_count < MAX_RAYS_PER_PIXEL) {
        int voxel_idx = ix * voxel_params.N * voxel_params.N +
                        iy * voxel_params.N + iz;

        float attenuation = 1.0f / (1.0f + alpha_attenuation * t_current);
        float val = pixel_val * attenuation;

        atomicAdd(&voxel_params.data[voxel_idx], val);

        // Step to next voxel
        if (t_max_x < t_max_y && t_max_x < t_max_z) {
            ix += step_x;
            t_current = t_max_x;
            t_max_x += t_delta_x;
        } else if (t_max_y < t_max_z) {
            iy += step_y;
            t_current = t_max_y;
            t_max_y += t_delta_y;
        } else {
            iz += step_z;
            t_current = t_max_z;
            t_max_z += t_delta_z;
        }

        step_count++;

        if (ix < 0 || ix >= voxel_params.N ||
            iy < 0 || iy >= voxel_params.N ||
            iz < 0 || iz >= voxel_params.N) {
            break;
        }
    }
}

// ============================================================================
// Pixel Counting Kernel (for statistics)
// ============================================================================

__global__ void countPixelsKernel(
    const bool* __restrict__ motion_mask,
    int* __restrict__ count,
    int total_pixels
) {
    __shared__ int shared_count[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_count[tid] = 0;
    __syncthreads();

    if (idx < total_pixels && motion_mask[idx]) {
        shared_count[tid] = 1;
    }
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_count[tid] += shared_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, shared_count[0]);
    }
}

// ============================================================================
// 3D Gaussian Blur Kernel
// ============================================================================

__global__ void gaussianBlur3DKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    float sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= N || y >= N || z >= N) return;

    int kernel_radius = (int)(3.0f * sigma);
    if (kernel_radius < 1) kernel_radius = 1;

    float sum = 0.0f;
    float weight_sum = 0.0f;
    float sigma2 = sigma * sigma;

    for (int dz = -kernel_radius; dz <= kernel_radius; dz++) {
        for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
            for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if (nx >= 0 && nx < N && ny >= 0 && ny < N && nz >= 0 && nz < N) {
                    float dist2 = (float)(dx * dx + dy * dy + dz * dz);
                    float weight = expf(-dist2 / (2.0f * sigma2));

                    int idx = nx * N * N + ny * N + nz;
                    sum += input[idx] * weight;
                    weight_sum += weight;
                }
            }
        }
    }

    int out_idx = x * N * N + y * N + z;
    output[out_idx] = (weight_sum > 0.0f) ? (sum / weight_sum) : 0.0f;
}

// ============================================================================
// Host Functions Implementation
// ============================================================================

void initCudaStreams(int num_streams, cudaStream_t** streams) {
    *streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&(*streams)[i]));
    }
}

void cleanupCudaStreams(int num_streams, cudaStream_t* streams) {
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
}

void allocateVoxelGrid(int N, float** d_voxel_grid) {
    size_t size = (size_t)N * N * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(d_voxel_grid, size));
    CUDA_CHECK(cudaMemset(*d_voxel_grid, 0, size));
}

void freeVoxelGrid(float* d_voxel_grid) {
    CUDA_CHECK(cudaFree(d_voxel_grid));
}

void clearVoxelGrid(float* d_voxel_grid, int N, cudaStream_t stream) {
    size_t size = (size_t)N * N * N * sizeof(float);
    CUDA_CHECK(cudaMemsetAsync(d_voxel_grid, 0, size, stream));
}

void copyVoxelGridToHost(float* d_voxel_grid, float* h_voxel_grid, int N) {
    size_t size = (size_t)N * N * N * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_voxel_grid, d_voxel_grid, size, cudaMemcpyDeviceToHost));
}

void detectMotionGPU(
    const float* d_prev_frame,
    const float* d_curr_frame,
    bool* d_motion_mask,
    float* d_diff,
    int width,
    int height,
    float threshold,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    motionDetectionKernel<<<grid, block, 0, stream>>>(
        d_prev_frame, d_curr_frame, d_motion_mask, d_diff,
        width, height, threshold
    );

    CUDA_CHECK(cudaGetLastError());
}

int countChangedPixels(
    const bool* d_motion_mask,
    int width,
    int height,
    cudaStream_t stream
) {
    int total_pixels = width * height;
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;

    countPixelsKernel<<<grid_size, block_size, 0, stream>>>(
        d_motion_mask, d_count, total_pixels
    );

    int h_count;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));

    return h_count;
}

void castRaysMotionGPU(
    const float* d_frame,
    const bool* d_motion_mask,
    const float* d_diff,
    const CameraParams& camera,
    const VoxelGridParams& voxel_params,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (camera.width + block.x - 1) / block.x,
        (camera.height + block.y - 1) / block.y
    );

    float alpha = 0.1f; // Attenuation factor

    rayCastMotionKernel<<<grid, block, 0, stream>>>(
        d_frame, d_motion_mask, d_diff, camera, voxel_params, alpha
    );

    CUDA_CHECK(cudaGetLastError());
}

void castRaysFullFrameGPU(
    const float* d_frame,
    const CameraParams& camera,
    const VoxelGridParams& voxel_params,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (camera.width + block.x - 1) / block.x,
        (camera.height + block.y - 1) / block.y
    );

    float alpha = 0.1f; // Attenuation factor
    float min_threshold = 1e-3f;

    rayCastFullFrameKernel<<<grid, block, 0, stream>>>(
        d_frame, camera, voxel_params, alpha, min_threshold
    );

    CUDA_CHECK(cudaGetLastError());
}

void processMultipleCameras(
    const std::vector<float*>& h_prev_frames,
    const std::vector<float*>& h_curr_frames,
    const std::vector<CameraParams>& cameras,
    const VoxelGridParams& voxel_params,
    int num_cameras,
    float motion_threshold,
    cudaStream_t* streams
) {
    // Allocate device memory for each camera
    std::vector<float*> d_prev_frames(num_cameras);
    std::vector<float*> d_curr_frames(num_cameras);
    std::vector<bool*> d_motion_masks(num_cameras);
    std::vector<float*> d_diffs(num_cameras);

    for (int i = 0; i < num_cameras; i++) {
        int frame_size = cameras[i].width * cameras[i].height;
        size_t frame_bytes = frame_size * sizeof(float);
        size_t mask_bytes = frame_size * sizeof(bool);

        CUDA_CHECK(cudaMalloc(&d_prev_frames[i], frame_bytes));
        CUDA_CHECK(cudaMalloc(&d_curr_frames[i], frame_bytes));
        CUDA_CHECK(cudaMalloc(&d_motion_masks[i], mask_bytes));
        CUDA_CHECK(cudaMalloc(&d_diffs[i], frame_bytes));

        // Async copy to device
        CUDA_CHECK(cudaMemcpyAsync(d_prev_frames[i], h_prev_frames[i],
                                   frame_bytes, cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_curr_frames[i], h_curr_frames[i],
                                   frame_bytes, cudaMemcpyHostToDevice, streams[i]));
    }

    // Process each camera in parallel using different streams
    for (int i = 0; i < num_cameras; i++) {
        // Motion detection
        detectMotionGPU(
            d_prev_frames[i], d_curr_frames[i], d_motion_masks[i], d_diffs[i],
            cameras[i].width, cameras[i].height, motion_threshold, streams[i]
        );

        // Ray casting
        castRaysMotionGPU(
            d_curr_frames[i], d_motion_masks[i], d_diffs[i],
            cameras[i], voxel_params, streams[i]
        );
    }

    // Synchronize all streams
    for (int i = 0; i < num_cameras; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // Cleanup
    for (int i = 0; i < num_cameras; i++) {
        CUDA_CHECK(cudaFree(d_prev_frames[i]));
        CUDA_CHECK(cudaFree(d_curr_frames[i]));
        CUDA_CHECK(cudaFree(d_motion_masks[i]));
        CUDA_CHECK(cudaFree(d_diffs[i]));
    }
}

void applyGaussianBlurGPU(
    float* d_voxel_grid,
    float* d_temp_grid,
    int N,
    float sigma,
    cudaStream_t stream
) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y,
        (N + block.z - 1) / block.z
    );

    gaussianBlur3DKernel<<<grid, block, 0, stream>>>(
        d_voxel_grid, d_temp_grid, N, sigma
    );

    CUDA_CHECK(cudaGetLastError());
}

void printCudaDeviceInfo(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("=== CUDA Device Information ===\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max Grid Size: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max Block Dims: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Memory Clock Rate: %.2f MHz\n",
           prop.memoryClockRate / 1000.0);
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("===============================\n");
}

bool checkComputeCapability(int required_major, int required_minor, int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    if (prop.major > required_major ||
        (prop.major == required_major && prop.minor >= required_minor)) {
        return true;
    }

    printf("Warning: Device compute capability %d.%d is less than required %d.%d\n",
           prop.major, prop.minor, required_major, required_minor);
    return false;
}

void optimizeFor8K() {
    // Prefer L1 cache for global loads
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    // Use 48KB shared memory configuration
    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    printf("CUDA optimizations applied for 8K video processing\n");
}

void getOptimalDimensions(
    int width,
    int height,
    dim3& block_dim,
    dim3& grid_dim
) {
    // For 8K: 7680x4320
    // Use 32x32 blocks for good occupancy
    block_dim.x = BLOCK_SIZE_X;
    block_dim.y = BLOCK_SIZE_Y;
    block_dim.z = 1;

    grid_dim.x = (width + block_dim.x - 1) / block_dim.x;
    grid_dim.y = (height + block_dim.y - 1) / block_dim.y;
    grid_dim.z = 1;
}

void benchmarkRayCasting(
    int width,
    int height,
    int num_cameras,
    int voxel_grid_size,
    int num_iterations
) {
    printf("=== Ray-Casting Benchmark ===\n");
    printf("Frame Size: %dx%d\n", width, height);
    printf("Num Cameras: %d\n", num_cameras);
    printf("Voxel Grid Size: %d^3\n", voxel_grid_size);
    printf("Iterations: %d\n", num_iterations);

    // Setup
    cudaStream_t* streams;
    initCudaStreams(num_cameras, &streams);

    float* d_voxel_grid;
    allocateVoxelGrid(voxel_grid_size, &d_voxel_grid);

    VoxelGridParams voxel_params;
    voxel_params.N = voxel_grid_size;
    voxel_params.voxel_size = 6.0f;
    voxel_params.grid_center = Vec3f(0.0f, 0.0f, 500.0f);
    voxel_params.data = d_voxel_grid;

    // Create dummy camera data
    std::vector<float*> h_prev_frames(num_cameras);
    std::vector<float*> h_curr_frames(num_cameras);
    std::vector<CameraParams> cameras(num_cameras);

    for (int i = 0; i < num_cameras; i++) {
        int frame_size = width * height;
        h_prev_frames[i] = new float[frame_size];
        h_curr_frames[i] = new float[frame_size];

        // Initialize with dummy data
        for (int j = 0; j < frame_size; j++) {
            h_prev_frames[i][j] = (float)(rand() % 256);
            h_curr_frames[i][j] = (float)(rand() % 256);
        }

        cameras[i].width = width;
        cameras[i].height = height;
        cameras[i].fov_rad = 1.0f;
        cameras[i].position = Vec3f(100.0f * i, 0.0f, 0.0f);
        cameras[i].camera_id = i;

        // Identity rotation
        for (int k = 0; k < 9; k++) {
            cameras[i].rotation.m[k] = (k % 4 == 0) ? 1.0f : 0.0f;
        }
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < num_iterations; iter++) {
        processMultipleCameras(
            h_prev_frames, h_curr_frames, cameras,
            voxel_params, num_cameras, 2.0f, streams
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_time = milliseconds / num_iterations;
    float fps = 1000.0f / avg_time;
    float total_pixels = (float)width * height * num_cameras;
    float megapixels_per_sec = (total_pixels / 1e6) * fps;

    printf("Average Time per Frame: %.2f ms\n", avg_time);
    printf("Throughput: %.2f FPS\n", fps);
    printf("Megapixels/sec: %.2f MP/s\n", megapixels_per_sec);
    printf("============================\n");

    // Cleanup
    for (int i = 0; i < num_cameras; i++) {
        delete[] h_prev_frames[i];
        delete[] h_curr_frames[i];
    }

    freeVoxelGrid(d_voxel_grid);
    cleanupCudaStreams(num_cameras, streams);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Advanced Feature Implementations
// ============================================================================

__global__ void findLocalMaximaKernel(
    const float* __restrict__ voxel_grid,
    int* __restrict__ maxima,
    float* __restrict__ maxima_values,
    int* __restrict__ count,
    int N,
    float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x <= 0 || x >= N-1 || y <= 0 || y >= N-1 || z <= 0 || z >= N-1) return;

    int idx = x * N * N + y * N + z;
    float val = voxel_grid[idx];

    if (val < threshold) return;

    // Check if local maximum
    bool is_max = true;
    for (int dz = -1; dz <= 1 && is_max; dz++) {
        for (int dy = -1; dy <= 1 && is_max; dy++) {
            for (int dx = -1; dx <= 1 && is_max; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nidx = (x+dx) * N * N + (y+dy) * N + (z+dz);
                if (voxel_grid[nidx] > val) {
                    is_max = false;
                }
            }
        }
    }

    if (is_max) {
        int pos = atomicAdd(count, 1);
        maxima[pos * 3 + 0] = x;
        maxima[pos * 3 + 1] = y;
        maxima[pos * 3 + 2] = z;
        maxima_values[pos] = val;
    }
}

int findLocalMaximaGPU(
    const float* d_voxel_grid,
    int* d_maxima,
    float* d_maxima_values,
    int N,
    float threshold,
    cudaStream_t stream
) {
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    dim3 block(8, 8, 8);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y,
        (N + block.z - 1) / block.z
    );

    findLocalMaximaKernel<<<grid, block, 0, stream>>>(
        d_voxel_grid, d_maxima, d_maxima_values, d_count, N, threshold
    );

    int h_count;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));

    return h_count;
}

__global__ void histogramKernel(
    const float* __restrict__ voxel_grid,
    int* __restrict__ histogram,
    int N,
    int num_bins,
    float min_val,
    float max_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N * N;

    if (idx >= total) return;

    float val = voxel_grid[idx];
    if (val > 0.0f) {
        float normalized = (val - min_val) / (max_val - min_val);
        int bin = (int)(normalized * (num_bins - 1));
        bin = max(0, min(num_bins - 1, bin));
        atomicAdd(&histogram[bin], 1);
    }
}

void computeHistogramGPU(
    const float* d_voxel_grid,
    int* d_histogram,
    int N,
    int num_bins,
    cudaStream_t stream
) {
    // Find min/max first (simplified - assume known range)
    float min_val = 0.0f;
    float max_val = 1000.0f; // Adjust as needed

    CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));

    int total = N * N * N;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    histogramKernel<<<grid_size, block_size, 0, stream>>>(
        d_voxel_grid, d_histogram, N, num_bins, min_val, max_val
    );

    CUDA_CHECK(cudaGetLastError());
}
