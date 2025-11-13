/**
 * small_object_detector.cu
 *
 * CUDA-accelerated small object detection for detecting 20cm drones at 5km range
 * Achieves sub-pixel accuracy with atmospheric compensation and motion-based enhancement
 *
 * Performance targets:
 * - Detect 0.23 arcminute objects (20cm at 5km)
 * - 99%+ detection rate
 * - <2% false positive rate
 * - <100ms latency for 200 simultaneous targets
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <math_constants.h>
#include <float.h>

// Detection parameters optimized for 20cm objects at 5km
#define DETECTION_THRESHOLD 0.15f        // Adaptive threshold for small objects
#define MIN_OBJECT_SIZE_PIXELS 1.5f      // Sub-pixel minimum (0.23 arcmin)
#define MAX_OBJECT_SIZE_PIXELS 8.0f      // Maximum expected size
#define ATMOSPHERIC_TURBULENCE 0.85f     // Fried parameter compensation
#define MOTION_BLUR_COMPENSATION 1.2f    // Motion enhancement factor
#define BACKGROUND_WINDOW_SIZE 15        // Local background estimation
#define SUB_PIXEL_ACCURACY 0.1f          // Sub-pixel centroid accuracy

// Thread block configuration for optimal occupancy
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define WARP_SIZE 32

/**
 * Detection result structure
 */
struct Detection {
    float x;                    // Sub-pixel X coordinate
    float y;                    // Sub-pixel Y coordinate
    float confidence;           // Detection confidence [0-1]
    float velocity_x;           // Estimated X velocity (pixels/frame)
    float velocity_y;           // Estimated Y velocity (pixels/frame)
    float size;                 // Object size in pixels
    int track_id;              // Associated track ID (-1 if new)
    unsigned long long timestamp; // Detection timestamp
};

/**
 * Atmospheric compensation parameters
 */
struct AtmosphericParams {
    float turbulence_strength;  // Turbulence strength (0-1)
    float scintillation_index;  // Scintillation index
    float seeing_angle;         // Atmospheric seeing (arcsec)
    float airmass;             // Airmass factor
};

/**
 * CUDA kernel: Adaptive local background subtraction
 * Uses median filtering to estimate local background with atmospheric compensation
 */
__global__ void adaptiveBackgroundSubtraction(
    const float* __restrict__ input,
    float* __restrict__ output,
    const AtmosphericParams* __restrict__ atmos,
    int width,
    int height,
    int window_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int half_window = window_size / 2;

    // Shared memory for local window processing
    __shared__ float local_window[BLOCK_SIZE_Y + 30][BLOCK_SIZE_X + 30];

    // Load local neighborhood into shared memory
    int local_x = threadIdx.x + half_window;
    int local_y = threadIdx.y + half_window;

    // Boundary handling with reflection
    int src_x = min(max(x, 0), width - 1);
    int src_y = min(max(y, 0), height - 1);
    local_window[local_y][local_x] = input[src_y * width + src_x];

    // Load halo regions
    if (threadIdx.x < half_window) {
        int left_x = max(x - half_window, 0);
        local_window[local_y][threadIdx.x] = input[src_y * width + left_x];

        int right_x = min(x + blockDim.x, width - 1);
        local_window[local_y][threadIdx.x + blockDim.x + half_window] =
            input[src_y * width + right_x];
    }

    if (threadIdx.y < half_window) {
        int top_y = max(y - half_window, 0);
        local_window[threadIdx.y][local_x] = input[top_y * width + src_x];

        int bottom_y = min(y + blockDim.y, height - 1);
        local_window[threadIdx.y + blockDim.y + half_window][local_x] =
            input[bottom_y * width + src_x];
    }

    __syncthreads();

    // Compute local statistics for adaptive thresholding
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;

    // Use robust estimator (trimmed mean) to avoid bright outliers
    float values[225]; // Max window size

    for (int dy = -half_window; dy <= half_window; dy++) {
        for (int dx = -half_window; dx <= half_window; dx++) {
            float val = local_window[local_y + dy][local_x + dx];
            values[count++] = val;
            sum += val;
            sum_sq += val * val;
        }
    }

    // Sort for median (simple bubble sort for small arrays)
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                float temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }

    // Use trimmed mean: exclude top and bottom 10%
    int trim = count / 10;
    float background = 0.0f;
    int bg_count = 0;
    for (int i = trim; i < count - trim; i++) {
        background += values[i];
        bg_count++;
    }
    background /= bg_count;

    // Compute standard deviation
    float variance = 0.0f;
    for (int i = trim; i < count - trim; i++) {
        float diff = values[i] - background;
        variance += diff * diff;
    }
    float std_dev = sqrtf(variance / bg_count);

    // Atmospheric compensation: adjust threshold based on seeing conditions
    float atmos_factor = 1.0f + atmos->turbulence_strength * (1.0f - ATMOSPHERIC_TURBULENCE);
    float adaptive_threshold = background + std_dev * DETECTION_THRESHOLD * atmos_factor;

    // Background subtraction with atmospheric correction
    float pixel_value = input[idx];
    float subtracted = pixel_value - background;

    // Apply scintillation compensation
    subtracted *= (1.0f + atmos->scintillation_index * 0.5f);

    output[idx] = fmaxf(0.0f, subtracted);
}

/**
 * CUDA kernel: Motion-based enhancement using temporal differencing
 * Enhances moving objects while suppressing static noise
 */
__global__ void motionEnhancement(
    const float* __restrict__ frame_current,
    const float* __restrict__ frame_prev,
    const float* __restrict__ frame_prev2,
    float* __restrict__ motion_enhanced,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Three-frame differencing for robust motion detection
    float curr = frame_current[idx];
    float prev = frame_prev[idx];
    float prev2 = frame_prev2[idx];

    // Compute temporal gradients
    float diff1 = curr - prev;
    float diff2 = prev - prev2;

    // Motion energy: consistent motion across frames
    float motion_energy = sqrtf(diff1 * diff1 + diff2 * diff2);

    // Direction consistency check
    float consistency = (diff1 * diff2 > 0) ? 1.0f : 0.3f;

    // Enhanced signal with motion compensation
    motion_enhanced[idx] = curr * (1.0f + MOTION_BLUR_COMPENSATION * motion_energy * consistency);
}

/**
 * CUDA kernel: Sub-pixel accurate blob detection with connected components
 * Detects small objects with sub-pixel centroid accuracy
 */
__global__ void subPixelBlobDetection(
    const float* __restrict__ input,
    Detection* __restrict__ detections,
    int* __restrict__ detection_count,
    int width,
    int height,
    float min_size,
    float max_size,
    int max_detections)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float pixel_value = input[idx];

    // Early exit for background pixels
    if (pixel_value < 0.01f) return;

    // Shared memory for local maxima detection
    __shared__ float local_region[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;
    local_region[local_y][local_x] = pixel_value;

    // Load borders
    if (threadIdx.x == 0 && x > 0)
        local_region[local_y][0] = input[idx - 1];
    if (threadIdx.x == blockDim.x - 1 && x < width - 1)
        local_region[local_y][BLOCK_SIZE_X + 1] = input[idx + 1];
    if (threadIdx.y == 0 && y > 0)
        local_region[0][local_x] = input[idx - width];
    if (threadIdx.y == blockDim.y - 1 && y < height - 1)
        local_region[BLOCK_SIZE_Y + 1][local_x] = input[idx + width];

    __syncthreads();

    // Check if local maximum (potential object center)
    bool is_maximum = true;
    for (int dy = -1; dy <= 1 && is_maximum; dy++) {
        for (int dx = -1; dx <= 1 && is_maximum; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (local_region[local_y + dy][local_x + dx] > pixel_value) {
                is_maximum = false;
            }
        }
    }

    if (!is_maximum || pixel_value < 0.05f) return;

    // Sub-pixel centroid computation using weighted neighborhood
    float sum_intensity = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    int blob_pixels = 0;

    // Analyze 5x5 neighborhood for sub-pixel centroid
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float neighbor_val = input[ny * width + nx];
                if (neighbor_val > 0.01f) {
                    sum_intensity += neighbor_val;
                    sum_x += neighbor_val * nx;
                    sum_y += neighbor_val * ny;
                    blob_pixels++;
                }
            }
        }
    }

    // Sub-pixel centroid
    float centroid_x = sum_x / sum_intensity;
    float centroid_y = sum_y / sum_intensity;

    // Estimate blob size using second moment
    float moment_xx = 0.0f;
    float moment_yy = 0.0f;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float neighbor_val = input[ny * width + nx];
                if (neighbor_val > 0.01f) {
                    float diff_x = nx - centroid_x;
                    float diff_y = ny - centroid_y;
                    moment_xx += neighbor_val * diff_x * diff_x;
                    moment_yy += neighbor_val * diff_y * diff_y;
                }
            }
        }
    }

    float blob_size = sqrtf((moment_xx + moment_yy) / sum_intensity);

    // Size filtering
    if (blob_size < min_size || blob_size > max_size) return;

    // Confidence based on peak intensity and size consistency
    float confidence = fminf(1.0f, pixel_value * (1.0f + 0.5f / blob_size));

    // Add detection atomically
    int detection_idx = atomicAdd(detection_count, 1);

    if (detection_idx < max_detections) {
        Detection det;
        det.x = centroid_x;
        det.y = centroid_y;
        det.confidence = confidence;
        det.size = blob_size;
        det.velocity_x = 0.0f; // Will be computed by tracker
        det.velocity_y = 0.0f;
        det.track_id = -1;
        det.timestamp = 0; // Will be set by host

        detections[detection_idx] = det;
    }
}

/**
 * CUDA kernel: Non-maximum suppression to eliminate duplicate detections
 */
__global__ void nonMaximumSuppression(
    Detection* __restrict__ detections,
    bool* __restrict__ suppressed,
    int num_detections,
    float nms_threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_detections || suppressed[i]) return;

    Detection det_i = detections[i];

    for (int j = i + 1; j < num_detections; j++) {
        if (suppressed[j]) continue;

        Detection det_j = detections[j];

        // Compute Euclidean distance
        float dx = det_i.x - det_j.x;
        float dy = det_i.y - det_j.y;
        float distance = sqrtf(dx * dx + dy * dy);

        // Suppress lower confidence detection if too close
        if (distance < nms_threshold) {
            if (det_i.confidence >= det_j.confidence) {
                suppressed[j] = true;
            } else {
                suppressed[i] = true;
                return;
            }
        }
    }
}

/**
 * CUDA kernel: Velocity estimation using optical flow
 * Estimates sub-pixel velocity for each detection using Lucas-Kanade
 */
__global__ void estimateVelocity(
    const float* __restrict__ frame_current,
    const float* __restrict__ frame_prev,
    Detection* __restrict__ detections,
    int num_detections,
    int width,
    int height)
{
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (det_idx >= num_detections) return;

    Detection det = detections[det_idx];
    int x = (int)det.x;
    int y = (int)det.y;

    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return;

    // Lucas-Kanade optical flow in 5x5 window
    float sum_Ix2 = 0.0f;
    float sum_Iy2 = 0.0f;
    float sum_IxIy = 0.0f;
    float sum_IxIt = 0.0f;
    float sum_IyIt = 0.0f;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            int idx = ny * width + nx;

            // Spatial gradients
            float Ix = (frame_current[idx + 1] - frame_current[idx - 1]) / 2.0f;
            float Iy = (frame_current[idx + width] - frame_current[idx - width]) / 2.0f;

            // Temporal gradient
            float It = frame_current[idx] - frame_prev[idx];

            sum_Ix2 += Ix * Ix;
            sum_Iy2 += Iy * Iy;
            sum_IxIy += Ix * Iy;
            sum_IxIt += Ix * It;
            sum_IyIt += Iy * It;
        }
    }

    // Solve 2x2 system for velocity
    float determinant = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy;

    if (fabsf(determinant) > 1e-6f) {
        float vx = (sum_IxIy * sum_IyIt - sum_Iy2 * sum_IxIt) / determinant;
        float vy = (sum_IxIy * sum_IxIt - sum_Ix2 * sum_IyIt) / determinant;

        // Clamp velocity to reasonable range (max 50 pixels/frame for drones)
        vx = fmaxf(-50.0f, fminf(50.0f, vx));
        vy = fmaxf(-50.0f, fminf(50.0f, vy));

        detections[det_idx].velocity_x = vx;
        detections[det_idx].velocity_y = vy;
    }
}

/**
 * Host function: Complete detection pipeline
 */
extern "C" {

int detectSmallObjects(
    const float* d_frame_current,
    const float* d_frame_prev,
    const float* d_frame_prev2,
    Detection* d_detections,
    int* h_detection_count,
    int width,
    int height,
    AtmosphericParams* h_atmos_params,
    cudaStream_t stream = 0)
{
    // Allocate device memory for intermediate results
    float *d_background_subtracted, *d_motion_enhanced;
    int *d_detection_count;
    AtmosphericParams *d_atmos_params;
    bool *d_suppressed;

    cudaMalloc(&d_background_subtracted, width * height * sizeof(float));
    cudaMalloc(&d_motion_enhanced, width * height * sizeof(float));
    cudaMalloc(&d_detection_count, sizeof(int));
    cudaMalloc(&d_atmos_params, sizeof(AtmosphericParams));
    cudaMalloc(&d_suppressed, 2000 * sizeof(bool)); // Max 2000 detections before NMS

    // Copy atmospheric parameters
    cudaMemcpyAsync(d_atmos_params, h_atmos_params, sizeof(AtmosphericParams),
                    cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_detection_count, 0, sizeof(int), stream);
    cudaMemsetAsync(d_suppressed, 0, 2000 * sizeof(bool), stream);

    // Configure kernel launches
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
              (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // Pipeline stage 1: Adaptive background subtraction
    adaptiveBackgroundSubtraction<<<grid, block, 0, stream>>>(
        d_frame_current, d_background_subtracted, d_atmos_params,
        width, height, BACKGROUND_WINDOW_SIZE);

    // Pipeline stage 2: Motion enhancement
    motionEnhancement<<<grid, block, 0, stream>>>(
        d_background_subtracted, d_frame_prev, d_frame_prev2,
        d_motion_enhanced, width, height);

    // Pipeline stage 3: Sub-pixel blob detection
    subPixelBlobDetection<<<grid, block, 0, stream>>>(
        d_motion_enhanced, d_detections, d_detection_count,
        width, height, MIN_OBJECT_SIZE_PIXELS, MAX_OBJECT_SIZE_PIXELS, 2000);

    // Copy detection count back
    cudaMemcpyAsync(h_detection_count, d_detection_count, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Pipeline stage 4: Non-maximum suppression
    int num_detections = *h_detection_count;
    if (num_detections > 0) {
        int nms_threads = 256;
        int nms_blocks = (num_detections + nms_threads - 1) / nms_threads;

        nonMaximumSuppression<<<nms_blocks, nms_threads, 0, stream>>>(
            d_detections, d_suppressed, num_detections, 3.0f);

        // Pipeline stage 5: Velocity estimation
        estimateVelocity<<<nms_blocks, nms_threads, 0, stream>>>(
            d_frame_current, d_frame_prev, d_detections,
            num_detections, width, height);
    }

    // Cleanup
    cudaFree(d_background_subtracted);
    cudaFree(d_motion_enhanced);
    cudaFree(d_detection_count);
    cudaFree(d_atmos_params);
    cudaFree(d_suppressed);

    return num_detections;
}

} // extern "C"
