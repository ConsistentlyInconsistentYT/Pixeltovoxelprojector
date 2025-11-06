// process_image.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include "nlohmann/json.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

// For convenience
using json = nlohmann::json;

//----------------------------------------------
// 1) Data Structures
//----------------------------------------------
struct Vec3 {
    float x, y, z;
};

struct Mat3 {
    float m[9];
};

struct FrameInfo {
    int camera_index;
    int frame_index;
    Vec3 camera_position;
    float yaw, pitch, roll;
    float fov_degrees;
    std::string image_file;
    // Optionally we store object_name, object_location if needed
};

//----------------------------------------------
// 2) Basic Math Helpers
//----------------------------------------------
static inline float deg2rad(float deg) {
    return deg * 3.14159265358979323846f / 180.0f;
}

static inline Vec3 normalize(const Vec3 &v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if(len < 1e-12f) {
        return {0.f, 0.f, 0.f};
    }
    return { v.x/len, v.y/len, v.z/len };
}

// Multiply 3x3 matrix by Vec3
static inline Vec3 mat3_mul_vec3(const Mat3 &M, const Vec3 &v) {
    Vec3 r;
    r.x = M.m[0]*v.x + M.m[1]*v.y + M.m[2]*v.z;
    r.y = M.m[3]*v.x + M.m[4]*v.y + M.m[5]*v.z;
    r.z = M.m[6]*v.x + M.m[7]*v.y + M.m[8]*v.z;
    return r;
}

//----------------------------------------------
// 3) Euler -> Rotation Matrix
//----------------------------------------------
Mat3 rotation_matrix_yaw_pitch_roll(float yaw_deg, float pitch_deg, float roll_deg) {
    float y = deg2rad(yaw_deg);
    float p = deg2rad(pitch_deg);
    float r = deg2rad(roll_deg);

    // Build each sub-rotation
    // Rz(yaw)
    float cy = std::cos(y), sy = std::sin(y);
    float Rz[9] = {
        cy, -sy, 0.f,
        sy,  cy, 0.f,
        0.f, 0.f, 1.f
    };

    // Ry(roll)
    float cr = std::cos(r), sr = std::sin(r);
    float Ry[9] = {
        cr,  0.f, sr,
        0.f, 1.f, 0.f,
        -sr, 0.f, cr
    };

    // Rx(pitch)
    float cp = std::cos(p), sp = std::sin(p);
    float Rx[9] = {
        1.f,  0.f,  0.f,
        0.f,  cp,  -sp,
        0.f,  sp,   cp
    };

    // Helper to multiply 3x3
    auto matmul3x3 = [&](const float A[9], const float B[9], float C[9]){
        for(int row=0; row<3; ++row) {
            for(int col=0; col<3; ++col) {
                C[row*3+col] = 
                    A[row*3+0]*B[0*3+col] +
                    A[row*3+1]*B[1*3+col] +
                    A[row*3+2]*B[2*3+col];
            }
        }
    };

    float Rtemp[9], Rfinal[9];
    matmul3x3(Rz, Ry, Rtemp);    // Rz * Ry
    matmul3x3(Rtemp, Rx, Rfinal); // (Rz*Ry)*Rx

    Mat3 out;
    for(int i=0; i<9; i++){
        out.m[i] = Rfinal[i];
    }
    return out;
}

//----------------------------------------------
// 4) Load JSON Metadata
//----------------------------------------------
std::vector<FrameInfo> load_metadata(const std::string &json_path) {
    std::vector<FrameInfo> frames;

    std::ifstream ifs(json_path);
    if(!ifs.is_open()){
        std::cerr << "ERROR: Cannot open " << json_path << std::endl;
        return frames;
    }
    json j;
    ifs >> j;
    if(!j.is_array()){
        std::cerr << "ERROR: JSON top level is not an array.\n";
        return frames;
    }

    for(const auto &entry : j) {
        FrameInfo fi;
        fi.camera_index   = entry.value("camera_index", 0);
        fi.frame_index    = entry.value("frame_index", 0);
        fi.yaw            = entry.value("yaw", 0.f);
        fi.pitch          = entry.value("pitch", 0.f);
        fi.roll           = entry.value("roll", 0.f);
        fi.fov_degrees    = entry.value("fov_degrees", 60.f);
        fi.image_file     = entry.value("image_file", "");

        // camera_position array
        if(entry.contains("camera_position") && entry["camera_position"].is_array()){
            auto arr = entry["camera_position"];
            if(arr.size()>=3){
                fi.camera_position.x = arr[0].get<float>();
                fi.camera_position.y = arr[1].get<float>();
                fi.camera_position.z = arr[2].get<float>();
            }
        }
        frames.push_back(fi);
    }

    return frames;
}

//----------------------------------------------
// 5) Image Loading (Gray) & Motion Detection
//----------------------------------------------
struct ImageGray {
    int width;
    int height;
    std::vector<float> pixels;  // grayscale float
};

#include <random>  // for std::mt19937, std::uniform_real_distribution

// Load image in grayscale (0-255 float) and add uniform noise.
bool load_image_gray(const std::string &img_path, ImageGray &out) {
    int w, h, channels;
    // stbi_load returns 8-bit data by default
    unsigned char* data = stbi_load(img_path.c_str(), &w, &h, &channels, 1);
    if (!data) {
        std::cerr << "Failed to load image: " << img_path << std::endl;
        return false;
    }

    out.width = w;
    out.height = h;
    out.pixels.resize(w * h);

    // Prepare random noise generator
    static std::random_device rd;
    static std::mt19937 gen(rd());
    // Noise in [-3, +3]
    std::uniform_real_distribution<float> noise_dist(-1.0f, 1.0f);

    // Copy pixels and add noise
    for (int i = 0; i < w * h; i++) {
        float val = static_cast<float>(data[i]);  // 0..255
        // Add uniform noise
        val += noise_dist(gen);
        // Clamp to [0, 255]
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        // Store in out.pixels
        out.pixels[i] = val;
    }

    stbi_image_free(data);
    return true;
}

// Detect motion by absolute difference
// Returns a boolean mask + the difference for each pixel
struct MotionMask {
    int width;
    int height;
    std::vector<bool> changed;
    std::vector<float> diff; // absolute difference
};

MotionMask detect_motion(const ImageGray &prev, const ImageGray &next, float threshold) {
    MotionMask mm;
    if(prev.width != next.width || prev.height != next.height) {
        std::cerr << "Images differ in size. Can\'t do motion detection!\n";
        mm.width = 0;
        mm.height = 0;
        return mm;
    }
    mm.width = prev.width;
    mm.height = prev.height;
    mm.changed.resize(mm.width * mm.height, false);
    mm.diff.resize(mm.width * mm.height, 0.f);

    for(int i=0; i < mm.width*mm.height; i++){
        float d = std::fabs(prev.pixels[i] - next.pixels[i]);
        mm.diff[i] = d;
        mm.changed[i] = (d > threshold);
    }
    return mm;
}

bool ray_aabb_intersection(const Vec3& ray_origin, const Vec3& ray_direction, const Vec3& aabb_min, const Vec3& aabb_max, float& t_entry, float& t_exit) {
    Vec3 inv_direction = {1.0f / ray_direction.x, 1.0f / ray_direction.y, 1.0f / ray_direction.z};
    Vec3 tmin = {(aabb_min.x - ray_origin.x) * inv_direction.x,
                 (aabb_min.y - ray_origin.y) * inv_direction.y,
                 (aabb_min.z - ray_origin.z) * inv_direction.z};
    Vec3 tmax = {(aabb_max.x - ray_origin.x) * inv_direction.x,
                 (aabb_max.y - ray_origin.y) * inv_direction.y,
                 (aabb_max.z - ray_origin.z) * inv_direction.z};

    Vec3 t1 = {std::min(tmin.x, tmax.x), std::min(tmin.y, tmax.y), std::min(tmin.z, tmax.z)};
    Vec3 t2 = {std::max(tmin.x, tmax.x), std::max(tmin.y, tmax.y), std::max(tmin.z, tmax.z)};

    float t_near = std::max(std::max(t1.x, t1.y), t1.z);
    float t_far = std::min(std::min(t2.x, t2.y), t2.z);

    if (t_near > t_far || t_far < 0.0f) {
        return false;
    }

    t_entry = t_near;
    t_exit = t_far;
    return true;
}

/**
 * Convert a 3D point to voxel indices within the voxel grid.
 *
 * Given a point in space and the voxel grid extents, compute which voxel it falls into.
 */
std::tuple<pybind11::ssize_t, pybind11::ssize_t, pybind11::ssize_t> point_to_voxel_indices(
    const std::array<double, 3>& point,
    const std::vector<std::pair<double, double>>& voxel_grid_extent,
    const std::array<pybind11::ssize_t, 3>& voxel_grid_size)
{
    double x_min = voxel_grid_extent[0].first;
    double x_max = voxel_grid_extent[0].second;
    double y_min = voxel_grid_extent[1].first;
    double y_max = voxel_grid_extent[1].second;
    double z_min = voxel_grid_extent[2].first;
    double z_max = voxel_grid_extent[2].second;

    double x = point[0];
    double y = point[1];
    double z = point[2];

    // Check if the point is inside the voxel grid bounds
    if (x_min <= x && x <= x_max && y_min <= y && y <= y_max && z_min <= z && z <= z_max)
    {
        pybind11::ssize_t nx = voxel_grid_size[0];
        pybind11::ssize_t ny = voxel_grid_size[1];
        pybind11::ssize_t nz = voxel_grid_size[2];

        // Compute normalized position within the grid
        double x_norm = (x - x_min) / (x_max - x_min);
        double y_norm = (y - y_min) / (y_max - y_min);
        double z_norm = (z - z_min) / (z_max - z_min);

        // Convert normalized coordinates to voxel indices
        pybind11::ssize_t x_idx = static_cast<pybind11::ssize_t>(x_norm * nx);
        pybind11::ssize_t y_idx = static_cast<pybind11::ssize_t>(y_norm * ny);
        pybind11::ssize_t z_idx = static_cast<pybind11::ssize_t>(z_norm * nz);

        // Clamp the indices to valid range
        x_idx = std::min(std::max(x_idx, pybind11::ssize_t(0)), nx - 1);
        y_idx = std::min(std::max(y_idx, pybind11::ssize_t(0)), ny - 1);
        z_idx = std::min(std::max(z_idx, pybind11::ssize_t(0)), nz - 1);

        return std::make_tuple(x_idx, y_idx, z_idx);
    }
    else
    {
        // The point is outside the voxel grid
        return std::make_tuple(-1, -1, -1);
    }
}

/**
 * Process an image and update the voxel grid and celestial sphere texture.
 *
 * This function:
 * 1. Computes the direction of each pixel in the image.
 * 2. Maps that direction to RA/Dec to find the corresponding brightness on the celestial sphere.
 * 3. Optionally subtracts the background (celestial sphere brightness) from the image brightness.
 * 4. If updating the celestial sphere, accumulates brightness values into the celestial_sphere_texture.
 * 5. If a voxel grid is provided, casts rays into the grid and updates voxel brightness accordingly.
 */
void process_image_cpp(
    py::array_t<double> image,
    std::array<double, 3> earth_position,
    std::array<double, 3> pointing_direction,
    double fov,
    pybind11::ssize_t image_width,
    pybind11::ssize_t image_height,
    py::array_t<double> voxel_grid,
    std::vector<std::pair<double, double>> voxel_grid_extent,
    double max_distance,
    int num_steps,
    py::array_t<double> celestial_sphere_texture,
    double center_ra_rad,
    double center_dec_rad,
    double angular_width_rad,
    double angular_height_rad,
    bool update_celestial_sphere,
    bool perform_background_subtraction
)
{
    // Access the image and celestial sphere texture arrays
    auto image_unchecked = image.unchecked<2>();
    auto texture_mutable = celestial_sphere_texture.mutable_unchecked<2>();
    pybind11::ssize_t texture_height = celestial_sphere_texture.shape(0);
    pybind11::ssize_t texture_width = celestial_sphere_texture.shape(1);

    // Check if voxel_grid is provided and non-empty
    bool voxel_grid_provided = voxel_grid && voxel_grid.size() > 0;

    // Variables for voxel grid (only if voxel_grid_provided)
    std::array<pybind11::ssize_t, 3> voxel_grid_size = {0, 0, 0};
    double x_min = 0, x_max = 0;
    double y_min = 0, y_max = 0;
    double z_min = 0, z_max = 0;

    // We only declare voxel_grid_mutable inside the if block if voxel_grid is provided
    // This avoids the need for a default constructor for unchecked_mutable_reference.
    py::detail::unchecked_mutable_reference<double, 3>* voxel_grid_mutable_ptr = nullptr;

    if (voxel_grid_provided)
    {
        // Get a mutable reference to the voxel grid
        auto voxel_grid_mutable = voxel_grid.mutable_unchecked<3>();
        voxel_grid_mutable_ptr = &voxel_grid_mutable;

        // Extract voxel grid dimensions and extents
        voxel_grid_size = {
            voxel_grid.shape(0),
            voxel_grid.shape(1),
            voxel_grid.shape(2)
        };

        x_min = voxel_grid_extent[0].first;
        x_max = voxel_grid_extent[0].second;
        y_min = voxel_grid_extent[1].first;
        y_max = voxel_grid_extent[1].second;
        z_min = voxel_grid_extent[2].first;
        z_max = voxel_grid_extent[2].second;
    }

    // Compute focal length from the field of view
    double focal_length = (image_width / 2.0) / std::tan(fov / 2.0);

    // Principal point (optical center)
    double cx = image_width / 2.0;
    double cy = image_height / 2.0;

    // pointing_direction is the z-axis of the camera frame
    // Normalize pointing_direction to ensure it's a unit vector
    double z_norm = std::sqrt(pointing_direction[0]*pointing_direction[0] +
                              pointing_direction[1]*pointing_direction[1] +
                              pointing_direction[2]*pointing_direction[2]);
    pointing_direction[0] /= z_norm;
    pointing_direction[1] /= z_norm;
    pointing_direction[2] /= z_norm;

    // Define an 'up' vector to avoid singularities
    std::array<double, 3> up = {0.0, 0.0, 1.0};
    if ((std::abs(pointing_direction[0] - up[0]) < 1e-8 &&
         std::abs(pointing_direction[1] - up[1]) < 1e-8 &&
         std::abs(pointing_direction[2] - up[2]) < 1e-8) ||
        (std::abs(pointing_direction[0] + up[0]) < 1e-8 &&
         std::abs(pointing_direction[1] + up[1]) < 1e-8 &&
         std::abs(pointing_direction[2] + up[2]) < 1e-8))
    {
        up = {0.0, 1.0, 0.0};
    }

    // Compute orthonormal basis: x_axis, y_axis, z_axis (z_axis = pointing_direction)
    std::array<double, 3> z_axis = pointing_direction;
    std::array<double, 3> x_axis;
    x_axis[0] = up[1]*z_axis[2] - up[2]*z_axis[1];
    x_axis[1] = up[2]*z_axis[0] - up[0]*z_axis[2];
    x_axis[2] = up[0]*z_axis[1] - up[1]*z_axis[0];

    double x_norm = std::sqrt(x_axis[0]*x_axis[0] + x_axis[1]*x_axis[1] + x_axis[2]*x_axis[2]);
    x_axis[0] /= x_norm;
    x_axis[1] /= x_norm;
    x_axis[2] /= x_norm;

    std::array<double, 3> y_axis;
    y_axis[0] = z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1];
    y_axis[1] = z_axis[2]*x_axis[0] - z_axis[0]*x_axis[2];
    y_axis[2] = z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0];

    // Iterate over each pixel in the image
    #pragma omp parallel for
    for (pybind11::ssize_t i = 0; i < image_height; ++i)
    {
        for (pybind11::ssize_t j = 0; j < image_width; ++j)
        {
            double brightness = image_unchecked(i, j);

            if (brightness > 0)
            {
                // Compute the direction in camera coordinates
                double x_cam = (j - cx);
                double y_cam = (i - cy);
                double z_cam = focal_length;

                double norm = std::sqrt(x_cam*x_cam + y_cam*y_cam + z_cam*z_cam);
                double direction_camera[3] = { x_cam/norm, y_cam/norm, z_cam/norm };

                // Transform direction_camera to world coordinates
                double direction_world[3];
                direction_world[0] = x_axis[0]*direction_camera[0] + y_axis[0]*direction_camera[1] + z_axis[0]*direction_camera[2];
                direction_world[1] = x_axis[1]*direction_camera[0] + y_axis[1]*direction_camera[1] + z_axis[1]*direction_camera[2];
                direction_world[2] = x_axis[2]*direction_camera[0] + y_axis[2]*direction_camera[1] + z_axis[2]*direction_camera[2];

                // Normalize direction_world (should already be unit, but just in case)
                double dir_norm = std::sqrt(direction_world[0]*direction_world[0] +
                                            direction_world[1]*direction_world[1] +
                                            direction_world[2]*direction_world[2]);
                direction_world[0] /= dir_norm;
                direction_world[1] /= dir_norm;
                direction_world[2] /= dir_norm;

                // Compute RA/Dec for direction_world
                double dx = direction_world[0];
                double dy = direction_world[1];
                double dz = direction_world[2];

                double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                double dec = std::asin(dz / r);
                double ra = std::atan2(dy, dx);
                if (ra < 0) ra += 2 * M_PI;

                // Compute offsets from the center of the sky patch
                double ra_offset = ra - center_ra_rad;
                double dec_offset = dec - center_dec_rad;

                // Adjust RA offset for wrapping
                if (ra_offset > M_PI) ra_offset -= 2 * M_PI;
                if (ra_offset < -M_PI) ra_offset += 2 * M_PI;

                // Check if within the defined sky patch
                bool within_sky_patch = (std::abs(ra_offset) <= angular_width_rad / 2) &&
                                        (std::abs(dec_offset) <= angular_height_rad / 2);

                // Map RA/Dec to texture coordinates
                double u = (ra_offset + angular_width_rad / 2) / angular_width_rad * texture_width;
                double v = (dec_offset + angular_height_rad / 2) / angular_height_rad * texture_height;

                pybind11::ssize_t u_idx = static_cast<pybind11::ssize_t>(u);
                pybind11::ssize_t v_idx = static_cast<pybind11::ssize_t>(v);

                // Clamp texture coordinates
                u_idx = std::min(std::max(u_idx, pybind11::ssize_t(0)), texture_width - 1);
                v_idx = std::min(std::max(v_idx, pybind11::ssize_t(0)), texture_height - 1);

                // Background subtraction
                double background_brightness = 0.0;
                if (within_sky_patch)
                {
                    background_brightness = texture_mutable(v_idx, u_idx);
                }

                if (perform_background_subtraction)
                {
                    brightness -= background_brightness;
                    if (brightness <= 0)
                        continue;  // Skip if adjusted brightness is zero or negative
                }

                // Update celestial sphere texture if needed
                if (update_celestial_sphere && within_sky_patch)
                {
                    #pragma omp atomic
                    texture_mutable(v_idx, u_idx) += brightness;
                }

                // If we have a voxel grid, cast rays into it and update voxel brightness
                if (voxel_grid_provided)
                {
                    // Safe to use voxel_grid_mutable_ptr now because voxel_grid_provided is true
                    auto &voxel_grid_mutable = *voxel_grid_mutable_ptr; // Reference to the voxel grid

                    // Ray casting into voxel grid
                    double step_size = max_distance / num_steps;

                    Vec3 ray_origin = {(float)earth_position[0], (float)earth_position[1], (float)earth_position[2] };
                    Vec3 ray_direction = {(float)direction_world[0], (float)direction_world[1], (float)direction_world[2] };

                    Vec3 box_min = {(float)x_min, (float)y_min, (float)z_min };
                    Vec3 box_max = {(float)x_max, (float)y_max, (float)z_max };
                    float t_entry, t_exit;

                    if (ray_aabb_intersection(ray_origin, ray_direction, box_min, box_max, t_entry, t_exit))
                    {
                        t_entry = std::max(t_entry, 0.0f);
                        t_exit = std::min(t_exit, (float)max_distance);

                        if (t_entry <= t_exit)
                        {
                            int s_entry = static_cast<int>(t_entry / step_size);
                            int s_exit = static_cast<int>(t_exit / step_size);

                            for (int s = s_entry; s <= s_exit; ++s)
                            {
                                double d = s * step_size;
                                double px = ray_origin.x + d * ray_direction.x;
                                double py = ray_origin.y + d * ray_direction.y;
                                double pz = ray_origin.z + d * ray_direction.z;

                                auto indices = point_to_voxel_indices({ px, py, pz }, voxel_grid_extent, voxel_grid_size);
                                pybind11::ssize_t x_idx = std::get<0>(indices);
                                pybind11::ssize_t y_idx = std::get<1>(indices);
                                pybind11::ssize_t z_idx = std::get<2>(indices);

                                if (x_idx >= 0)
                                {
                                    #pragma omp atomic
                                    voxel_grid_mutable(x_idx, y_idx, z_idx) += brightness;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//----------------------------------------------
// 6) Voxel DDA
//----------------------------------------------
struct RayStep {
    int ix, iy, iz;
    int step_count;
    float distance;
};

static inline float safe_div(float num, float den) {
    float eps = 1e-12f;
    if(std::fabs(den) < eps) {
        return std::numeric_limits<float>::infinity();
    }
    return num / den;
}

std::vector<RayStep> cast_ray_into_grid(
    const Vec3 &camera_pos, 
    const Vec3 &dir_normalized, 
    int N, 
    float voxel_size, 
    const Vec3 &grid_center)
{
    std::vector<RayStep> steps;
    steps.reserve(64);

    float half_size = 0.5f * (N * voxel_size);
    Vec3 grid_min = { grid_center.x - half_size,
                      grid_center.y - half_size,
                      grid_center.z - half_size };
    Vec3 grid_max = { grid_center.x + half_size,
                      grid_center.y + half_size,
                      grid_center.z + half_size };

    float t_min = 0.f;
    float t_max = std::numeric_limits<float>::infinity();

    // 1) Ray-box intersection
    for(int i=0; i<3; i++){
        float origin = (i==0)? camera_pos.x : ((i==1)? camera_pos.y : camera_pos.z);
        float d      = (i==0)? dir_normalized.x : ((i==1)? dir_normalized.y : dir_normalized.z);
        float mn     = (i==0)? grid_min.x : ((i==1)? grid_min.y : grid_min.z);
        float mx     = (i==0)? grid_max.x : ((i==1)? grid_max.y : grid_max.z);

        if(std::fabs(d) < 1e-12f){
            if(origin < mn || origin > mx){
                return steps; // no intersection
            }
        } else {
            float t1 = (mn - origin)/d;
            float t2 = (mx - origin)/d;
            float t_near = std::fmin(t1, t2);
            float t_far  = std::fmax(t1, t2);
            if(t_near > t_min) t_min = t_near;
            if(t_far  < t_max) t_max = t_far;
            if(t_min > t_max){
                return steps;
            }
        }
    }

    if(t_min < 0.f) t_min = 0.f;

    // 2) Start voxel
    Vec3 start_world = { camera_pos.x + t_min*dir_normalized.x,
                         camera_pos.y + t_min*dir_normalized.y,
                         camera_pos.z + t_min*dir_normalized.z };
    float fx = (start_world.x - grid_min.x)/voxel_size;
    float fy = (start_world.y - grid_min.y)/voxel_size;
    float fz = (start_world.z - grid_min.z)/voxel_size;

    int ix = int(fx);
    int iy = int(fy);
    int iz = int(fz);
    if(ix<0 || ix>=N || iy<0 || iy>=N || iz<0 || iz>=N) {
        return steps;
    }

    // 3) Step direction
    int step_x = (dir_normalized.x >= 0.f)? 1 : -1;
    int step_y = (dir_normalized.y >= 0.f)? 1 : -1;
    int step_z = (dir_normalized.z >= 0.f)? 1 : -1;

    auto boundary_in_world_x = [&](int i_x){ return grid_min.x + i_x*voxel_size; };
    auto boundary_in_world_y = [&](int i_y){ return grid_min.y + i_y*voxel_size; };
    auto boundary_in_world_z = [&](int i_z){ return grid_min.z + i_z*voxel_size; };

    int nx_x = ix + (step_x>0?1:0);
    int nx_y = iy + (step_y>0?1:0);
    int nx_z = iz + (step_z>0?1:0);

    float next_bx = boundary_in_world_x(nx_x);
    float next_by = boundary_in_world_y(nx_y);
    float next_bz = boundary_in_world_z(nx_z);

    float t_max_x = safe_div(next_bx - camera_pos.x, dir_normalized.x);
    float t_max_y = safe_div(next_by - camera_pos.y, dir_normalized.y);
    float t_max_z = safe_div(next_bz - camera_pos.z, dir_normalized.z);

    float t_delta_x = safe_div(voxel_size, std::fabs(dir_normalized.x));
    float t_delta_y = safe_div(voxel_size, std::fabs(dir_normalized.y));
    float t_delta_z = safe_div(voxel_size, std::fabs(dir_normalized.z));

    float t_current = t_min;
    int step_count = 0;

    // 4) Walk
    while(t_current <= t_max){
        RayStep rs;
        rs.ix = ix; 
        rs.iy = iy; 
        rs.iz = iz;
        rs.step_count = step_count;
        rs.distance = t_current;

        steps.push_back(rs);

        if(t_max_x < t_max_y && t_max_x < t_max_z){
            ix += step_x;
            t_current = t_max_x;
            t_max_x += t_delta_x;
        } else if(t_max_y < t_max_z){
            iy += step_y;
            t_current = t_max_y;
            t_max_y += t_delta_y;
        } else {
            iz += step_z;
            t_current = t_max_z;
            t_max_z += t_delta_z;
        }
        step_count++;
        if(ix<0 || ix>=N || iy<0 || iy>=N || iz<0 || iz>=N){
            break;
        }
    }

    return steps;
}

void process_motion(
    const std::string &metadata_path,
    const std::string &images_folder,
    const std::string &output_bin,
    int N,
    float voxel_size,
    const Vec3 &grid_center,
    float motion_threshold,
    float alpha
)
{
    //------------------------------------------
    // 7.1) Load metadata
    //------------------------------------------
    std::vector<FrameInfo> frames = load_metadata(metadata_path);
    if(frames.empty()) {
        std::cerr << "No frames loaded.\n";
        return;
    }
    // Group by camera_index
    // map< camera_index, vector<FrameInfo> >
    std::map<int, std::vector<FrameInfo>> frames_by_cam;
    for(const auto &f : frames) {
        frames_by_cam[f.camera_index].push_back(f);
    }
    // Sort each by frame_index
    for(auto &kv : frames_by_cam) {
        auto &v = kv.second;
        std::sort(v.begin(), v.end(), [](auto &a, auto &b){
            return a.frame_index < b.frame_index;
        });
    }

    //------------------------------------------
    // 7.2) Create a 3D voxel grid
    //------------------------------------------
    std::vector<float> voxel_grid(N*N*N, 0.f);

    //------------------------------------------
    // 7.3) For each camera, load consecutive frames, detect motion,
    //      and cast rays for changed pixels
    //------------------------------------------
    for(auto &kv : frames_by_cam) {
        auto &cam_frames = kv.second;

        if(cam_frames.size() < 2) {
            // Need at least two frames to see motion
            continue;
        }

        // We'll keep the previous image to compare
        ImageGray prev_img;
        bool prev_valid = false;

        for(size_t i=0; i<cam_frames.size(); i++){
            // Load current frame
            FrameInfo curr_info = cam_frames[i];
            std::string img_path = images_folder + "/" + curr_info.image_file;

            ImageGray curr_img;
            if(!load_image_gray(img_path, curr_img)) {
                std::cerr << "Skipping frame due to load error.\n";
                continue;
            }

            if(!prev_valid) {
                // Just store it, and wait for next
                prev_img = curr_img;
                prev_valid = true;
                continue;
            }

            // Now we have prev + curr => detect motion
            MotionMask mm = detect_motion(prev_img, curr_img, motion_threshold);

            // Use the "current" frame's camera info for ray-casting
            Vec3 cam_pos    = curr_info.camera_position;
            Mat3 cam_rot    = rotation_matrix_yaw_pitch_roll(curr_info.yaw, curr_info.pitch, curr_info.roll);
            float fov_rad   = deg2rad(curr_info.fov_degrees);
            float focal_len = (mm.width*0.5f) / std::tan(fov_rad*0.5f);

            // For each changed pixel, accumulate into the voxel grid
            for(int v = 0; v < mm.height; v++){
                for(int u = 0; u < mm.width; u++){
                    if(!mm.changed[v*mm.width + u]){
                        continue; // skip if no motion
                    }
                    // Pixel brightness from current or use mm.diff
                    float pix_val = mm.diff[v*mm.width + u];
                    if(pix_val < 1e-3f) {
                        continue;
                    }

                    // Build local camera direction
                    float x = (float(u) - 0.5f*mm.width);
                    float y = - (float(v) - 0.5f*mm.height);
                    float z = -focal_len;

                    Vec3 ray_cam = {x,y,z};
                    ray_cam = normalize(ray_cam);

                    // transform to world
                    Vec3 ray_world = mat3_mul_vec3(cam_rot, ray_cam);
                    ray_world = normalize(ray_world);

                    // DDA
                    std::vector<RayStep> steps = cast_ray_into_grid(
                        cam_pos, ray_world, N, voxel_size, grid_center
                    );

                    // Accumulate
                    for(const auto &rs : steps) {
                        float dist = rs.distance;
                        float attenuation = 1.f/(1.f + alpha*dist);
                        float val = pix_val * attenuation;
                        int idx = rs.ix*N*N + rs.iy*N + rs.iz;
                        voxel_grid[idx] += val;
                    }
                }
            }

            // Move current -> previous
            prev_img = curr_img;
        }
    }

    //------------------------------------------
    // 7.4) Save the voxel grid to .bin
    //------------------------------------------
    {
        std::ofstream ofs(output_bin, std::ios::binary);
        if(!ofs) {
            std::cerr << "Cannot open output file: " << output_bin << "\n";
            return;
        }
        // Write metadata (N, voxel_size)
        ofs.write(reinterpret_cast<const char*>(&N), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&voxel_size), sizeof(float));
        // Write the data
        ofs.write(reinterpret_cast<const char*>(voxel_grid.data()), voxel_grid.size()*sizeof(float));
        ofs.close();
        std::cout << "Saved voxel grid to " << output_bin << "\n";
    }
}

// Expose the function to Python
PYBIND11_MODULE(process_image_cpp, m) {
    m.doc() = "C++ implementation of the process_image function";
    m.def("process_image_cpp", &process_image_cpp, "Process image and update voxel grid in C++",
          py::arg("image"),
          py::arg("earth_position"),
          py::arg("pointing_direction"),
          py::arg("fov"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("voxel_grid"),
          py::arg("voxel_grid_extent"),
          py::arg("max_distance"),
          py::arg("num_steps"),
          py::arg("celestial_sphere_texture"),
          py::arg("center_ra_rad"),
          py::arg("center_dec_rad"),
          py::arg("angular_width_rad"),
          py::arg("angular_height_rad"),
          py::arg("update_celestial_sphere"),
          py::arg("perform_background_subtraction")
    );

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

    m.def("process_motion", &process_motion, "Process motion and create voxel grid",
          py::arg("metadata_path"),
          py::arg("images_folder"),
          py::arg("output_bin"),
          py::arg("N"),
          py::arg("voxel_size"),
          py::arg("grid_center"),
          py::arg("motion_threshold"),
          py::arg("alpha")
    );
}
