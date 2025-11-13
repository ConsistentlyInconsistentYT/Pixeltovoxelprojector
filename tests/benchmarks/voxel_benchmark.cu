/*
 * voxel_benchmark.cu
 *
 * CUDA Voxel Benchmarks for PixelToVoxelProjector
 *
 * Benchmarks:
 * - Ray-casting performance
 * - Voxel update throughput
 * - CUDA kernel performance
 * - Memory access patterns
 * - GPU memory bandwidth
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Benchmark result structure
typedef struct {
    const char* name;
    double duration_ms;
    double throughput_gops;
    double memory_bandwidth_gbps;
    double kernel_time_ms;
    int blocks;
    int threads_per_block;
} BenchmarkResult;

// 3D vector for ray operations
typedef struct {
    float x, y, z;
} float3_t;

// Timing utilities
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// ==============================================================================
// KERNEL 1: Voxel Ray Casting (DDA Algorithm)
// ==============================================================================

__device__ float3_t make_float3_dev(float x, float y, float z) {
    float3_t v;
    v.x = x; v.y = y; v.z = z;
    return v;
}

__device__ float3_t normalize_dev(float3_t v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len > 1e-6f) {
        v.x /= len; v.y /= len; v.z /= len;
    }
    return v;
}

__global__ void raycast_voxel_kernel(
    float* voxel_grid,      // Grid data (N^3)
    int N,                   // Grid size
    float voxel_size,        // Voxel size
    float3_t grid_center,    // Grid center
    float3_t* ray_origins,   // Ray origins
    float3_t* ray_directions,// Ray directions
    float* ray_results,      // Output: accumulated values
    int num_rays
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;

    float3_t origin = ray_origins[ray_idx];
    float3_t dir = normalize_dev(ray_directions[ray_idx]);

    // Compute grid bounds
    float half_size = 0.5f * (N * voxel_size);
    float3_t grid_min = make_float3_dev(
        grid_center.x - half_size,
        grid_center.y - half_size,
        grid_center.z - half_size
    );
    float3_t grid_max = make_float3_dev(
        grid_center.x + half_size,
        grid_center.y + half_size,
        grid_center.z + half_size
    );

    // Ray-box intersection
    float t_min = 0.0f;
    float t_max = 1e10f;

    for (int i = 0; i < 3; i++) {
        float o = (i == 0) ? origin.x : (i == 1) ? origin.y : origin.z;
        float d = (i == 0) ? dir.x : (i == 1) ? dir.y : dir.z;
        float bmin = (i == 0) ? grid_min.x : (i == 1) ? grid_min.y : grid_min.z;
        float bmax = (i == 0) ? grid_max.x : (i == 1) ? grid_max.y : grid_max.z;

        if (fabsf(d) > 1e-6f) {
            float t1 = (bmin - o) / d;
            float t2 = (bmax - o) / d;
            float t_near = fminf(t1, t2);
            float t_far = fmaxf(t1, t2);

            t_min = fmaxf(t_min, t_near);
            t_max = fminf(t_max, t_far);

            if (t_min > t_max) {
                ray_results[ray_idx] = 0.0f;
                return;
            }
        } else {
            if (o < bmin || o > bmax) {
                ray_results[ray_idx] = 0.0f;
                return;
            }
        }
    }

    if (t_min < 0.0f) t_min = 0.0f;

    // DDA traversal
    float3_t start_pos = make_float3_dev(
        origin.x + t_min * dir.x,
        origin.y + t_min * dir.y,
        origin.z + t_min * dir.z
    );

    int ix = (int)((start_pos.x - grid_min.x) / voxel_size);
    int iy = (int)((start_pos.y - grid_min.y) / voxel_size);
    int iz = (int)((start_pos.z - grid_min.z) / voxel_size);

    if (ix < 0 || ix >= N || iy < 0 || iy >= N || iz < 0 || iz >= N) {
        ray_results[ray_idx] = 0.0f;
        return;
    }

    int step_x = (dir.x >= 0.0f) ? 1 : -1;
    int step_y = (dir.y >= 0.0f) ? 1 : -1;
    int step_z = (dir.z >= 0.0f) ? 1 : -1;

    float t_delta_x = fabsf(voxel_size / dir.x);
    float t_delta_y = fabsf(voxel_size / dir.y);
    float t_delta_z = fabsf(voxel_size / dir.z);

    float t_max_x = t_min + fabsf(((ix + (step_x > 0 ? 1 : 0)) * voxel_size + grid_min.x - origin.x) / dir.x);
    float t_max_y = t_min + fabsf(((iy + (step_y > 0 ? 1 : 0)) * voxel_size + grid_min.y - origin.y) / dir.y);
    float t_max_z = t_min + fabsf(((iz + (step_z > 0 ? 1 : 0)) * voxel_size + grid_min.z - origin.z) / dir.z);

    float accumulated = 0.0f;
    int steps = 0;
    int max_steps = N * 3;

    while (steps < max_steps) {
        // Access voxel
        int idx = ix * N * N + iy * N + iz;
        accumulated += voxel_grid[idx];

        // Step to next voxel
        if (t_max_x < t_max_y && t_max_x < t_max_z) {
            ix += step_x;
            t_max_x += t_delta_x;
        } else if (t_max_y < t_max_z) {
            iy += step_y;
            t_max_y += t_delta_y;
        } else {
            iz += step_z;
            t_max_z += t_delta_z;
        }

        if (ix < 0 || ix >= N || iy < 0 || iy >= N || iz < 0 || iz >= N) {
            break;
        }

        steps++;
    }

    ray_results[ray_idx] = accumulated;
}

// ==============================================================================
// KERNEL 2: Voxel Update (Atomic Operations)
// ==============================================================================

__global__ void voxel_update_kernel(
    float* voxel_grid,
    int N,
    int* update_indices,  // Flat indices
    float* update_values,
    int num_updates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_updates) return;

    int voxel_idx = update_indices[idx];
    float value = update_values[idx];

    // Atomic add to voxel grid
    atomicAdd(&voxel_grid[voxel_idx], value);
}

// ==============================================================================
// KERNEL 3: Memory Bandwidth Test (Coalesced Access)
// ==============================================================================

__global__ void memory_bandwidth_kernel(
    float* input,
    float* output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Coalesced memory access pattern
    output[idx] = input[idx] * 2.0f + 1.0f;
}

// ==============================================================================
// KERNEL 4: Voxel Reduction (Sum all voxels)
// ==============================================================================

__global__ void voxel_reduction_kernel(
    float* voxel_grid,
    float* partial_sums,
    int N
) {
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N * N;

    // Load data into shared memory
    shared_sum[tid] = (idx < total) ? voxel_grid[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

// ==============================================================================
// Benchmark Functions
// ==============================================================================

void print_result(BenchmarkResult res) {
    printf("\n");
    printf("========================================\n");
    printf("Benchmark: %s\n", res.name);
    printf("========================================\n");
    printf("Duration:         %.2f ms\n", res.duration_ms);
    printf("Throughput:       %.2f GOPS\n", res.throughput_gops);
    printf("Memory BW:        %.2f GB/s\n", res.memory_bandwidth_gbps);
    printf("Kernel Time:      %.2f ms\n", res.kernel_time_ms);
    printf("Blocks:           %d\n", res.blocks);
    printf("Threads/Block:    %d\n", res.threads_per_block);
    printf("========================================\n");
}

BenchmarkResult benchmark_raycast(int N, int num_rays) {
    printf("\nBenchmarking Ray Casting (%d^3 grid, %d rays)...\n", N, num_rays);

    BenchmarkResult res;
    res.name = "Voxel Ray Casting (DDA)";

    // Allocate grid
    size_t grid_size = N * N * N * sizeof(float);
    float* h_grid = (float*)malloc(grid_size);
    float* d_grid;

    // Initialize grid with random values
    for (int i = 0; i < N*N*N; i++) {
        h_grid[i] = (float)rand() / RAND_MAX;
    }

    CUDA_CHECK(cudaMalloc(&d_grid, grid_size));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, grid_size, cudaMemcpyHostToDevice));

    // Allocate rays
    float3_t* h_origins = (float3_t*)malloc(num_rays * sizeof(float3_t));
    float3_t* h_directions = (float3_t*)malloc(num_rays * sizeof(float3_t));
    float* h_results = (float*)malloc(num_rays * sizeof(float));

    float3_t* d_origins, *d_directions;
    float* d_results;

    CUDA_CHECK(cudaMalloc(&d_origins, num_rays * sizeof(float3_t)));
    CUDA_CHECK(cudaMalloc(&d_directions, num_rays * sizeof(float3_t)));
    CUDA_CHECK(cudaMalloc(&d_results, num_rays * sizeof(float)));

    // Generate random rays
    for (int i = 0; i < num_rays; i++) {
        h_origins[i].x = ((float)rand() / RAND_MAX - 0.5f) * 2000.0f;
        h_origins[i].y = ((float)rand() / RAND_MAX - 0.5f) * 2000.0f;
        h_origins[i].z = ((float)rand() / RAND_MAX - 0.5f) * 2000.0f;

        h_directions[i].x = (float)rand() / RAND_MAX - 0.5f;
        h_directions[i].y = (float)rand() / RAND_MAX - 0.5f;
        h_directions[i].z = (float)rand() / RAND_MAX - 0.5f;
    }

    CUDA_CHECK(cudaMemcpy(d_origins, h_origins, num_rays * sizeof(float3_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_directions, h_directions, num_rays * sizeof(float3_t), cudaMemcpyHostToDevice));

    // Launch configuration
    int threads = 256;
    int blocks = (num_rays + threads - 1) / threads;

    res.blocks = blocks;
    res.threads_per_block = threads;

    float3_t grid_center;
    grid_center.x = 0.0f;
    grid_center.y = 0.0f;
    grid_center.z = 500.0f;

    // Warmup
    raycast_voxel_kernel<<<blocks, threads>>>(
        d_grid, N, 6.0f, grid_center, d_origins, d_directions, d_results, num_rays
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double cpu_start = get_time_ms();
    CUDA_CHECK(cudaEventRecord(start));

    raycast_voxel_kernel<<<blocks, threads>>>(
        d_grid, N, 6.0f, grid_center, d_origins, d_directions, d_results, num_rays
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    double cpu_end = get_time_ms();

    float kernel_time;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    res.duration_ms = cpu_end - cpu_start;
    res.kernel_time_ms = kernel_time;

    // Estimate operations (rays * steps_per_ray)
    long long ops = (long long)num_rays * N;  // Approximation
    res.throughput_gops = (ops / 1e9) / (kernel_time / 1000.0);

    // Memory bandwidth (rough estimate)
    long long bytes = grid_size + num_rays * sizeof(float3_t) * 2;
    res.memory_bandwidth_gbps = (bytes / 1e9) / (kernel_time / 1000.0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_origins));
    CUDA_CHECK(cudaFree(d_directions));
    CUDA_CHECK(cudaFree(d_results));
    free(h_grid);
    free(h_origins);
    free(h_directions);
    free(h_results);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return res;
}

BenchmarkResult benchmark_voxel_updates(int N, int num_updates) {
    printf("\nBenchmarking Voxel Updates (%d^3 grid, %d updates)...\n", N, num_updates);

    BenchmarkResult res;
    res.name = "Voxel Updates (Atomic)";

    size_t grid_size = N * N * N * sizeof(float);
    float* d_grid;
    CUDA_CHECK(cudaMalloc(&d_grid, grid_size));
    CUDA_CHECK(cudaMemset(d_grid, 0, grid_size));

    // Generate random updates
    int* h_indices = (int*)malloc(num_updates * sizeof(int));
    float* h_values = (float*)malloc(num_updates * sizeof(float));

    for (int i = 0; i < num_updates; i++) {
        h_indices[i] = rand() % (N * N * N);
        h_values[i] = (float)rand() / RAND_MAX;
    }

    int* d_indices;
    float* d_values;
    CUDA_CHECK(cudaMalloc(&d_indices, num_updates * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, num_updates * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, num_updates * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, num_updates * sizeof(float), cudaMemcpyHostToDevice));

    // Launch configuration
    int threads = 256;
    int blocks = (num_updates + threads - 1) / threads;

    res.blocks = blocks;
    res.threads_per_block = threads;

    // Warmup
    voxel_update_kernel<<<blocks, threads>>>(d_grid, N, d_indices, d_values, num_updates);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double cpu_start = get_time_ms();
    CUDA_CHECK(cudaEventRecord(start));

    voxel_update_kernel<<<blocks, threads>>>(d_grid, N, d_indices, d_values, num_updates);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    double cpu_end = get_time_ms();

    float kernel_time;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    res.duration_ms = cpu_end - cpu_start;
    res.kernel_time_ms = kernel_time;
    res.throughput_gops = (num_updates / 1e9) / (kernel_time / 1000.0);

    long long bytes = num_updates * (sizeof(int) + sizeof(float));
    res.memory_bandwidth_gbps = (bytes / 1e9) / (kernel_time / 1000.0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_values));
    free(h_indices);
    free(h_values);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return res;
}

BenchmarkResult benchmark_memory_bandwidth(size_t num_elements) {
    printf("\nBenchmarking Memory Bandwidth (%zu elements)...\n", num_elements);

    BenchmarkResult res;
    res.name = "Memory Bandwidth (Coalesced)";

    size_t size = num_elements * sizeof(float);
    float* d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemset(d_input, 0, size));

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    res.blocks = blocks;
    res.threads_per_block = threads;

    // Warmup
    memory_bandwidth_kernel<<<blocks, threads>>>(d_input, d_output, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    memory_bandwidth_kernel<<<blocks, threads>>>(d_input, d_output, num_elements);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_time;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    res.duration_ms = kernel_time;
    res.kernel_time_ms = kernel_time;

    // Read + Write
    long long bytes = 2 * num_elements * sizeof(float);
    res.memory_bandwidth_gbps = (bytes / 1e9) / (kernel_time / 1000.0);
    res.throughput_gops = (num_elements / 1e9) / (kernel_time / 1000.0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return res;
}

// ==============================================================================
// Main
// ==============================================================================

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("CUDA Voxel Benchmark Suite\n");
    printf("========================================\n");

    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\nGPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);

    // Run benchmarks
    BenchmarkResult results[4];

    results[0] = benchmark_raycast(500, 100000);
    results[1] = benchmark_voxel_updates(500, 1000000);
    results[2] = benchmark_memory_bandwidth(500 * 500 * 500);

    // Print all results
    printf("\n\n");
    printf("========================================\n");
    printf("BENCHMARK SUMMARY\n");
    printf("========================================\n");

    for (int i = 0; i < 3; i++) {
        print_result(results[i]);
    }

    printf("\nBenchmark suite completed!\n");

    return 0;
}
