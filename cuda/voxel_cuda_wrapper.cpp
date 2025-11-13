#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "voxel_cuda.h"

namespace py = pybind11;

// ============================================================================
// Python Wrapper Classes
// ============================================================================

class VoxelGridGPU {
private:
    float* d_voxel_grid;
    VoxelGridParams params;
    bool allocated;

public:
    VoxelGridGPU(int N, float voxel_size, py::array_t<float> grid_center) {
        auto center = grid_center.unchecked<1>();
        if (center.shape(0) != 3) {
            throw std::runtime_error("grid_center must have 3 elements");
        }

        params.N = N;
        params.voxel_size = voxel_size;
        params.grid_center.x = center(0);
        params.grid_center.y = center(1);
        params.grid_center.z = center(2);

        allocateVoxelGrid(N, &d_voxel_grid);
        params.data = d_voxel_grid;
        allocated = true;
    }

    ~VoxelGridGPU() {
        if (allocated) {
            freeVoxelGrid(d_voxel_grid);
            allocated = false;
        }
    }

    void clear(int stream_id = 0) {
        cudaStream_t stream = 0;
        clearVoxelGrid(d_voxel_grid, params.N, stream);
    }

    py::array_t<float> toHost() {
        size_t total_size = (size_t)params.N * params.N * params.N;
        py::array_t<float> result({params.N, params.N, params.N});

        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);

        copyVoxelGridToHost(d_voxel_grid, ptr, params.N);

        return result;
    }

    VoxelGridParams getParams() const {
        return params;
    }

    int getN() const { return params.N; }
    float getVoxelSize() const { return params.voxel_size; }
};

// ============================================================================
// Camera Manager with Multi-Stream Support
// ============================================================================

class CameraStreamManager {
private:
    int num_streams;
    cudaStream_t* streams;
    std::vector<CameraParams> cameras;

public:
    CameraStreamManager(int num_cameras) : num_streams(num_cameras) {
        initCudaStreams(num_streams, &streams);
        cameras.resize(num_cameras);
    }

    ~CameraStreamManager() {
        cleanupCudaStreams(num_streams, streams);
    }

    void setCamera(int cam_id,
                   py::array_t<float> position,
                   py::array_t<float> rotation_matrix,
                   float fov_rad,
                   int width,
                   int height) {
        if (cam_id < 0 || cam_id >= num_streams) {
            throw std::runtime_error("Invalid camera ID");
        }

        auto pos = position.unchecked<1>();
        auto rot = rotation_matrix.unchecked<1>();

        if (pos.shape(0) != 3) {
            throw std::runtime_error("position must have 3 elements");
        }
        if (rot.shape(0) != 9) {
            throw std::runtime_error("rotation_matrix must have 9 elements (flattened 3x3)");
        }

        cameras[cam_id].position.x = pos(0);
        cameras[cam_id].position.y = pos(1);
        cameras[cam_id].position.z = pos(2);

        for (int i = 0; i < 9; i++) {
            cameras[cam_id].rotation.m[i] = rot(i);
        }

        cameras[cam_id].fov_rad = fov_rad;
        cameras[cam_id].width = width;
        cameras[cam_id].height = height;
        cameras[cam_id].camera_id = cam_id;
    }

    void processFrames(
        py::array_t<float> prev_frames,
        py::array_t<float> curr_frames,
        VoxelGridGPU& voxel_grid,
        float motion_threshold = 2.0f) {

        // Validate input shapes
        if (prev_frames.ndim() != 3 || curr_frames.ndim() != 3) {
            throw std::runtime_error("Frame arrays must be 3D (num_cameras, height, width)");
        }

        auto prev_buf = prev_frames.request();
        auto curr_buf = curr_frames.request();

        int num_cams = prev_buf.shape[0];
        if (num_cams != num_streams) {
            throw std::runtime_error("Number of frames doesn't match number of cameras");
        }

        // Prepare host pointers
        std::vector<float*> h_prev_frames(num_cams);
        std::vector<float*> h_curr_frames(num_cams);

        float* prev_data = static_cast<float*>(prev_buf.ptr);
        float* curr_data = static_cast<float*>(curr_buf.ptr);

        for (int i = 0; i < num_cams; i++) {
            int frame_size = cameras[i].width * cameras[i].height;
            h_prev_frames[i] = prev_data + i * frame_size;
            h_curr_frames[i] = curr_data + i * frame_size;
        }

        // Process on GPU
        processMultipleCameras(
            h_prev_frames,
            h_curr_frames,
            cameras,
            voxel_grid.getParams(),
            num_cams,
            motion_threshold,
            streams
        );
    }

    void processSingleFrame(
        int cam_id,
        py::array_t<float> frame,
        VoxelGridGPU& voxel_grid,
        float min_threshold = 1e-3f) {

        if (cam_id < 0 || cam_id >= num_streams) {
            throw std::runtime_error("Invalid camera ID");
        }

        // Allocate device memory
        auto frame_buf = frame.request();
        int frame_size = cameras[cam_id].width * cameras[cam_id].height;

        float* d_frame;
        cudaMalloc(&d_frame, frame_size * sizeof(float));
        cudaMemcpyAsync(d_frame, frame_buf.ptr, frame_size * sizeof(float),
                        cudaMemcpyHostToDevice, streams[cam_id]);

        // Cast rays
        castRaysFullFrameGPU(
            d_frame,
            cameras[cam_id],
            voxel_grid.getParams(),
            streams[cam_id]
        );

        cudaStreamSynchronize(streams[cam_id]);
        cudaFree(d_frame);
    }

    int getNumStreams() const { return num_streams; }
};

// ============================================================================
// Standalone Utility Functions
// ============================================================================

void py_printDeviceInfo(int device_id = 0) {
    printCudaDeviceInfo(device_id);
}

bool py_checkComputeCapability(int major, int minor, int device_id = 0) {
    return checkComputeCapability(major, minor, device_id);
}

void py_optimizeFor8K() {
    optimizeFor8K();
}

py::array_t<float> py_detectMotion(
    py::array_t<float> prev_frame,
    py::array_t<float> curr_frame,
    float threshold = 2.0f) {

    auto prev_buf = prev_frame.request();
    auto curr_buf = curr_frame.request();

    if (prev_buf.ndim != 2 || curr_buf.ndim != 2) {
        throw std::runtime_error("Frames must be 2D arrays");
    }

    int height = prev_buf.shape[0];
    int width = prev_buf.shape[1];
    int frame_size = width * height;

    // Allocate device memory
    float *d_prev, *d_curr, *d_diff;
    bool *d_mask;

    cudaMalloc(&d_prev, frame_size * sizeof(float));
    cudaMalloc(&d_curr, frame_size * sizeof(float));
    cudaMalloc(&d_diff, frame_size * sizeof(float));
    cudaMalloc(&d_mask, frame_size * sizeof(bool));

    cudaMemcpy(d_prev, prev_buf.ptr, frame_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr, curr_buf.ptr, frame_size * sizeof(float), cudaMemcpyHostToDevice);

    // Detect motion
    detectMotionGPU(d_prev, d_curr, d_mask, d_diff, width, height, threshold, 0);

    // Copy result back
    py::array_t<float> result({height, width});
    auto result_buf = result.request();
    cudaMemcpy(result_buf.ptr, d_diff, frame_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_diff);
    cudaFree(d_mask);

    return result;
}

void py_benchmark(int width = 7680, int height = 4320, int num_cameras = 10,
                  int voxel_size = 500, int iterations = 100) {
    benchmarkRayCasting(width, height, num_cameras, voxel_size, iterations);
}

py::array_t<float> py_applyGaussianBlur(
    py::array_t<float> voxel_grid,
    float sigma = 1.0f) {

    auto buf = voxel_grid.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Voxel grid must be 3D");
    }

    int N = buf.shape[0];
    if (buf.shape[1] != N || buf.shape[2] != N) {
        throw std::runtime_error("Voxel grid must be cubic (NxNxN)");
    }

    size_t size = (size_t)N * N * N * sizeof(float);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, buf.ptr, size, cudaMemcpyHostToDevice);

    // Apply blur
    applyGaussianBlurGPU(d_input, d_output, N, sigma, 0);

    // Copy result back
    py::array_t<float> result({N, N, N});
    auto result_buf = result.request();
    cudaMemcpy(result_buf.ptr, d_output, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}

// ============================================================================
// Pybind11 Module Definition
// ============================================================================

PYBIND11_MODULE(voxel_cuda, m) {
    m.doc() = "CUDA-accelerated voxel processing for multi-camera systems";

    // VoxelGridGPU class
    py::class_<VoxelGridGPU>(m, "VoxelGridGPU")
        .def(py::init<int, float, py::array_t<float>>(),
             py::arg("N"),
             py::arg("voxel_size"),
             py::arg("grid_center"),
             "Initialize GPU voxel grid\n\n"
             "Parameters:\n"
             "  N: Grid size (creates NxNxN grid)\n"
             "  voxel_size: Size of each voxel in world units\n"
             "  grid_center: 3D array [x, y, z] for grid center position")
        .def("clear", &VoxelGridGPU::clear,
             py::arg("stream_id") = 0,
             "Clear voxel grid to zeros")
        .def("to_host", &VoxelGridGPU::toHost,
             "Copy voxel grid from GPU to CPU as numpy array")
        .def("get_N", &VoxelGridGPU::getN,
             "Get grid size")
        .def("get_voxel_size", &VoxelGridGPU::getVoxelSize,
             "Get voxel size")
        .def("__repr__", [](const VoxelGridGPU &grid) {
            return "<VoxelGridGPU N=" + std::to_string(grid.getN()) +
                   " voxel_size=" + std::to_string(grid.getVoxelSize()) + ">";
        });

    // CameraStreamManager class
    py::class_<CameraStreamManager>(m, "CameraStreamManager")
        .def(py::init<int>(),
             py::arg("num_cameras"),
             "Initialize camera stream manager for parallel processing\n\n"
             "Parameters:\n"
             "  num_cameras: Number of cameras (and CUDA streams)")
        .def("set_camera", &CameraStreamManager::setCamera,
             py::arg("cam_id"),
             py::arg("position"),
             py::arg("rotation_matrix"),
             py::arg("fov_rad"),
             py::arg("width"),
             py::arg("height"),
             "Set camera parameters\n\n"
             "Parameters:\n"
             "  cam_id: Camera ID (0 to num_cameras-1)\n"
             "  position: 3D array [x, y, z]\n"
             "  rotation_matrix: Flattened 3x3 rotation matrix (9 elements)\n"
             "  fov_rad: Field of view in radians\n"
             "  width: Frame width in pixels\n"
             "  height: Frame height in pixels")
        .def("process_frames", &CameraStreamManager::processFrames,
             py::arg("prev_frames"),
             py::arg("curr_frames"),
             py::arg("voxel_grid"),
             py::arg("motion_threshold") = 2.0f,
             "Process multiple camera frames with motion detection\n\n"
             "Parameters:\n"
             "  prev_frames: 3D array (num_cameras, height, width)\n"
             "  curr_frames: 3D array (num_cameras, height, width)\n"
             "  voxel_grid: VoxelGridGPU instance\n"
             "  motion_threshold: Pixel difference threshold for motion")
        .def("process_single_frame", &CameraStreamManager::processSingleFrame,
             py::arg("cam_id"),
             py::arg("frame"),
             py::arg("voxel_grid"),
             py::arg("min_threshold") = 1e-3f,
             "Process single frame without motion detection\n\n"
             "Parameters:\n"
             "  cam_id: Camera ID\n"
             "  frame: 2D array (height, width)\n"
             "  voxel_grid: VoxelGridGPU instance\n"
             "  min_threshold: Minimum pixel value to process")
        .def("get_num_streams", &CameraStreamManager::getNumStreams,
             "Get number of CUDA streams")
        .def("__repr__", [](const CameraStreamManager &mgr) {
            return "<CameraStreamManager streams=" + std::to_string(mgr.getNumStreams()) + ">";
        });

    // Utility functions
    m.def("print_device_info", &py_printDeviceInfo,
          py::arg("device_id") = 0,
          "Print CUDA device information");

    m.def("check_compute_capability", &py_checkComputeCapability,
          py::arg("major"),
          py::arg("minor"),
          py::arg("device_id") = 0,
          "Check if device supports required compute capability");

    m.def("optimize_for_8k", &py_optimizeFor8K,
          "Optimize CUDA settings for 8K video processing");

    m.def("detect_motion", &py_detectMotion,
          py::arg("prev_frame"),
          py::arg("curr_frame"),
          py::arg("threshold") = 2.0f,
          "GPU-accelerated motion detection between two frames\n\n"
          "Returns: 2D array of absolute differences");

    m.def("benchmark", &py_benchmark,
          py::arg("width") = 7680,
          py::arg("height") = 4320,
          py::arg("num_cameras") = 10,
          py::arg("voxel_size") = 500,
          py::arg("iterations") = 100,
          "Run performance benchmark");

    m.def("apply_gaussian_blur", &py_applyGaussianBlur,
          py::arg("voxel_grid"),
          py::arg("sigma") = 1.0f,
          "Apply 3D Gaussian blur to voxel grid on GPU\n\n"
          "Returns: Blurred voxel grid");

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("CUDA_ENABLED") = true;
}
