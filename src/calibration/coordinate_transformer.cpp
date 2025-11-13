/***************************************************
 * coordinate_transformer.cpp
 *
 * Coordinate Transformation System for Multi-Camera Tracking
 *
 * Features:
 * - Camera coordinates to world coordinates
 * - Pixel coordinates to 3D ray casting
 * - Multi-camera coordinate fusion
 * - WGS84 geodetic coordinate support
 * - ENU (East-North-Up) local frames
 * - Sub-pixel accuracy (<0.5 pixel reprojection error)
 * - Support for 10 camera pairs simultaneously
 * - Accuracy at 5km range
 *
 * Coordinate Systems:
 * 1. Pixel coordinates: (u, v) in image space
 * 2. Camera coordinates: (x_c, y_c, z_c) camera frame
 * 3. World coordinates: (x_w, y_w, z_w) in ENU frame
 * 4. Geodetic coordinates: (lat, lon, alt) WGS84
 ***************************************************/

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <algorithm>
#include <mutex>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// ============================================================================
// Constants
// ============================================================================

// WGS84 Ellipsoid Parameters
constexpr double WGS84_A = 6378137.0;              // Semi-major axis (meters)
constexpr double WGS84_B = 6356752.314245;         // Semi-minor axis (meters)
constexpr double WGS84_F = 1.0 / 298.257223563;   // Flattening
constexpr double WGS84_E2 = 0.00669437999014;     // First eccentricity squared

constexpr double DEG_TO_RAD = M_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / M_PI;

constexpr double MAX_RANGE = 5000.0;  // Maximum tracking range (5km)

// ============================================================================
// Data Structures
// ============================================================================

/**
 * 3D Vector structure
 */
struct Vec3 {
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3 operator*(double s) const { return Vec3(x*s, y*s, z*s); }
    Vec3 operator/(double s) const { return Vec3(x/s, y/s, z/s); }

    double length() const { return std::sqrt(x*x + y*y + z*z); }
    double dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }

    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y*v.z - z*v.y,
            z*v.x - x*v.z,
            x*v.y - y*v.x
        );
    }

    Vec3 normalized() const {
        double len = length();
        return (len > 1e-10) ? (*this / len) : Vec3(0, 0, 0);
    }
};

/**
 * 2D Pixel coordinates
 */
struct PixelCoord {
    double u, v;  // Image coordinates (pixels)

    PixelCoord() : u(0), v(0) {}
    PixelCoord(double u_, double v_) : u(u_), v(v_) {}
};

/**
 * Geodetic coordinates (WGS84)
 */
struct GeodeticCoord {
    double latitude;   // Degrees
    double longitude;  // Degrees
    double altitude;   // Meters above WGS84 ellipsoid

    GeodeticCoord() : latitude(0), longitude(0), altitude(0) {}
    GeodeticCoord(double lat, double lon, double alt)
        : latitude(lat), longitude(lon), altitude(alt) {}
};

/**
 * Camera intrinsic parameters
 */
struct CameraIntrinsics {
    double fx, fy;      // Focal lengths (pixels)
    double cx, cy;      // Principal point (pixels)
    double k1, k2, k3;  // Radial distortion coefficients
    double p1, p2;      // Tangential distortion coefficients
    int width, height;  // Image dimensions

    CameraIntrinsics()
        : fx(1000), fy(1000), cx(960), cy(540),
          k1(0), k2(0), k3(0), p1(0), p2(0),
          width(1920), height(1080) {}

    // Get camera matrix (K)
    Eigen::Matrix3d getCameraMatrix() const {
        Eigen::Matrix3d K;
        K << fx, 0,  cx,
             0,  fy, cy,
             0,  0,  1;
        return K;
    }

    // Get distortion coefficients
    Eigen::Matrix<double, 5, 1> getDistortionCoeffs() const {
        Eigen::Matrix<double, 5, 1> dist;
        dist << k1, k2, p1, p2, k3;
        return dist;
    }
};

/**
 * Camera extrinsic parameters (pose)
 */
struct CameraExtrinsics {
    Vec3 position;           // Camera position in world frame
    Eigen::Quaterniond rotation;  // Camera orientation (world to camera)

    CameraExtrinsics() : position(0, 0, 0), rotation(Eigen::Quaterniond::Identity()) {}

    // Get rotation matrix (world to camera)
    Eigen::Matrix3d getRotationMatrix() const {
        return rotation.toRotationMatrix();
    }

    // Get transformation matrix [R|t]
    Eigen::Matrix<double, 3, 4> getTransformMatrix() const {
        Eigen::Matrix<double, 3, 4> T;
        Eigen::Matrix3d R = getRotationMatrix();
        Eigen::Vector3d t(position.x, position.y, position.z);
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = -R * t;  // -R*t for world to camera
        return T;
    }
};

/**
 * Complete camera model
 */
struct Camera {
    int id;
    std::string name;
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
    double baseline;  // Baseline for stereo pair (meters)

    Camera() : id(0), name("camera"), baseline(0.0) {}
};

/**
 * 3D Ray structure
 */
struct Ray3D {
    Vec3 origin;
    Vec3 direction;

    Ray3D() {}
    Ray3D(const Vec3& o, const Vec3& d) : origin(o), direction(d.normalized()) {}

    // Get point at distance t along ray
    Vec3 at(double t) const {
        return origin + direction * t;
    }
};

/**
 * ENU (East-North-Up) local coordinate frame
 */
struct ENUFrame {
    GeodeticCoord origin;  // Reference point for local frame
    Eigen::Matrix3d R_ecef_to_enu;  // Rotation from ECEF to ENU

    ENUFrame() {}
    ENUFrame(const GeodeticCoord& ref) : origin(ref) {
        computeRotation();
    }

    void computeRotation() {
        double lat = origin.latitude * DEG_TO_RAD;
        double lon = origin.longitude * DEG_TO_RAD;

        double sin_lat = std::sin(lat);
        double cos_lat = std::cos(lat);
        double sin_lon = std::sin(lon);
        double cos_lon = std::cos(lon);

        // Rotation matrix from ECEF to ENU
        R_ecef_to_enu << -sin_lon,           cos_lon,          0,
                         -sin_lat*cos_lon,  -sin_lat*sin_lon,  cos_lat,
                          cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat;
    }
};

// ============================================================================
// CoordinateTransformer Class
// ============================================================================

class CoordinateTransformer {
private:
    std::vector<Camera> cameras;
    ENUFrame enuFrame;
    bool useGeodetic;
    mutable std::mutex transformMutex;

public:
    CoordinateTransformer(bool enableGeodetic = false)
        : useGeodetic(enableGeodetic) {}

    // ========================================================================
    // Camera Management
    // ========================================================================

    void addCamera(const Camera& camera) {
        std::lock_guard<std::mutex> lock(transformMutex);
        cameras.push_back(camera);
    }

    Camera* getCamera(int id) {
        for (auto& cam : cameras) {
            if (cam.id == id) return &cam;
        }
        return nullptr;
    }

    size_t getCameraCount() const { return cameras.size(); }

    // ========================================================================
    // ENU Frame Setup
    // ========================================================================

    void setENUOrigin(const GeodeticCoord& origin) {
        std::lock_guard<std::mutex> lock(transformMutex);
        enuFrame = ENUFrame(origin);
        useGeodetic = true;
    }

    // ========================================================================
    // Geodetic <-> ECEF Transformations
    // ========================================================================

    /**
     * Convert geodetic (lat, lon, alt) to ECEF (Earth-Centered Earth-Fixed)
     */
    Vec3 geodeticToECEF(const GeodeticCoord& geo) const {
        double lat = geo.latitude * DEG_TO_RAD;
        double lon = geo.longitude * DEG_TO_RAD;
        double alt = geo.altitude;

        double sin_lat = std::sin(lat);
        double cos_lat = std::cos(lat);
        double sin_lon = std::sin(lon);
        double cos_lon = std::cos(lon);

        // Prime vertical radius of curvature
        double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);

        Vec3 ecef;
        ecef.x = (N + alt) * cos_lat * cos_lon;
        ecef.y = (N + alt) * cos_lat * sin_lon;
        ecef.z = (N * (1.0 - WGS84_E2) + alt) * sin_lat;

        return ecef;
    }

    /**
     * Convert ECEF to geodetic (iterative algorithm)
     */
    GeodeticCoord ecefToGeodetic(const Vec3& ecef) const {
        double x = ecef.x;
        double y = ecef.y;
        double z = ecef.z;

        double lon = std::atan2(y, x);

        // Iterative computation of latitude
        double p = std::sqrt(x*x + y*y);
        double lat = std::atan2(z, p * (1.0 - WGS84_E2));

        for (int i = 0; i < 10; i++) {
            double sin_lat = std::sin(lat);
            double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
            double lat_new = std::atan2(z + WGS84_E2 * N * sin_lat, p);

            if (std::abs(lat_new - lat) < 1e-12) {
                lat = lat_new;
                break;
            }
            lat = lat_new;
        }

        double sin_lat = std::sin(lat);
        double cos_lat = std::cos(lat);
        double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
        double alt = p / cos_lat - N;

        return GeodeticCoord(lat * RAD_TO_DEG, lon * RAD_TO_DEG, alt);
    }

    // ========================================================================
    // ENU <-> ECEF Transformations
    // ========================================================================

    /**
     * Convert ECEF to ENU local frame
     */
    Vec3 ecefToENU(const Vec3& ecef) const {
        if (!useGeodetic) {
            std::cerr << "Warning: ENU frame not initialized" << std::endl;
            return Vec3(0, 0, 0);
        }

        Vec3 ecef_origin = geodeticToECEF(enuFrame.origin);
        Vec3 ecef_delta = ecef - ecef_origin;

        Eigen::Vector3d delta_eigen(ecef_delta.x, ecef_delta.y, ecef_delta.z);
        Eigen::Vector3d enu_eigen = enuFrame.R_ecef_to_enu * delta_eigen;

        return Vec3(enu_eigen(0), enu_eigen(1), enu_eigen(2));
    }

    /**
     * Convert ENU local frame to ECEF
     */
    Vec3 enuToECEF(const Vec3& enu) const {
        if (!useGeodetic) {
            std::cerr << "Warning: ENU frame not initialized" << std::endl;
            return Vec3(0, 0, 0);
        }

        Eigen::Vector3d enu_eigen(enu.x, enu.y, enu.z);
        Eigen::Vector3d delta_eigen = enuFrame.R_ecef_to_enu.transpose() * enu_eigen;

        Vec3 ecef_origin = geodeticToECEF(enuFrame.origin);
        return ecef_origin + Vec3(delta_eigen(0), delta_eigen(1), delta_eigen(2));
    }

    /**
     * Convert geodetic to ENU
     */
    Vec3 geodeticToENU(const GeodeticCoord& geo) const {
        Vec3 ecef = geodeticToECEF(geo);
        return ecefToENU(ecef);
    }

    /**
     * Convert ENU to geodetic
     */
    GeodeticCoord enuToGeodetic(const Vec3& enu) const {
        Vec3 ecef = enuToECEF(enu);
        return ecefToGeodetic(ecef);
    }

    // ========================================================================
    // Lens Distortion Models
    // ========================================================================

    /**
     * Apply lens distortion to normalized image coordinates
     * Brown-Conrady model (radial + tangential)
     */
    PixelCoord applyDistortion(double x_norm, double y_norm,
                               const CameraIntrinsics& intrinsics) const {
        double x2 = x_norm * x_norm;
        double y2 = y_norm * y_norm;
        double r2 = x2 + y2;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        // Radial distortion
        double radial = 1.0 + intrinsics.k1 * r2 +
                             intrinsics.k2 * r4 +
                             intrinsics.k3 * r6;

        // Tangential distortion
        double x_tangential = 2.0 * intrinsics.p1 * x_norm * y_norm +
                             intrinsics.p2 * (r2 + 2.0 * x2);
        double y_tangential = intrinsics.p1 * (r2 + 2.0 * y2) +
                             2.0 * intrinsics.p2 * x_norm * y_norm;

        // Apply distortion
        double x_distorted = x_norm * radial + x_tangential;
        double y_distorted = y_norm * radial + y_tangential;

        // Convert to pixel coordinates
        PixelCoord pixel;
        pixel.u = intrinsics.fx * x_distorted + intrinsics.cx;
        pixel.v = intrinsics.fy * y_distorted + intrinsics.cy;

        return pixel;
    }

    /**
     * Remove lens distortion from pixel coordinates (iterative)
     */
    void undistortPixel(const PixelCoord& distorted,
                       const CameraIntrinsics& intrinsics,
                       double& x_norm, double& y_norm) const {
        // Initial guess (undistorted = distorted)
        x_norm = (distorted.u - intrinsics.cx) / intrinsics.fx;
        y_norm = (distorted.v - intrinsics.cy) / intrinsics.fy;

        // Iterative refinement (Newton-Raphson)
        for (int i = 0; i < 5; i++) {
            PixelCoord predicted = applyDistortion(x_norm, y_norm, intrinsics);

            double error_u = predicted.u - distorted.u;
            double error_v = predicted.v - distorted.v;

            // Simple gradient descent step
            x_norm -= error_u / intrinsics.fx * 0.9;
            y_norm -= error_v / intrinsics.fy * 0.9;
        }
    }

    // ========================================================================
    // Pixel to 3D Ray Casting
    // ========================================================================

    /**
     * Convert pixel coordinates to 3D ray in camera coordinates
     */
    Ray3D pixelToRay(const PixelCoord& pixel, const Camera& camera) const {
        // Undistort pixel
        double x_norm, y_norm;
        undistortPixel(pixel, camera.intrinsics, x_norm, y_norm);

        // Ray direction in camera frame (z-forward convention)
        Vec3 dir_camera(x_norm, y_norm, 1.0);

        // Ray origin is camera position
        Ray3D ray;
        ray.origin = camera.extrinsics.position;

        // Transform direction to world frame
        Eigen::Vector3d dir_eigen(dir_camera.x, dir_camera.y, dir_camera.z);
        Eigen::Matrix3d R = camera.extrinsics.getRotationMatrix();
        Eigen::Vector3d dir_world = R.transpose() * dir_eigen;  // Camera to world

        ray.direction = Vec3(dir_world(0), dir_world(1), dir_world(2)).normalized();

        return ray;
    }

    // ========================================================================
    // Camera to World Coordinate Transformation
    // ========================================================================

    /**
     * Transform point from camera coordinates to world coordinates
     */
    Vec3 cameraToWorld(const Vec3& point_camera, const Camera& camera) const {
        Eigen::Vector3d p_cam(point_camera.x, point_camera.y, point_camera.z);
        Eigen::Matrix3d R = camera.extrinsics.getRotationMatrix();
        Eigen::Vector3d t(camera.extrinsics.position.x,
                         camera.extrinsics.position.y,
                         camera.extrinsics.position.z);

        // p_world = R^T * p_camera + t
        Eigen::Vector3d p_world = R.transpose() * p_cam + t;

        return Vec3(p_world(0), p_world(1), p_world(2));
    }

    /**
     * Transform point from world coordinates to camera coordinates
     */
    Vec3 worldToCamera(const Vec3& point_world, const Camera& camera) const {
        Eigen::Vector3d p_world(point_world.x, point_world.y, point_world.z);
        Eigen::Matrix3d R = camera.extrinsics.getRotationMatrix();
        Eigen::Vector3d t(camera.extrinsics.position.x,
                         camera.extrinsics.position.y,
                         camera.extrinsics.position.z);

        // p_camera = R * (p_world - t)
        Eigen::Vector3d p_cam = R * (p_world - t);

        return Vec3(p_cam(0), p_cam(1), p_cam(2));
    }

    // ========================================================================
    // 3D Point to Pixel Projection
    // ========================================================================

    /**
     * Project 3D world point to pixel coordinates
     */
    PixelCoord worldToPixel(const Vec3& point_world, const Camera& camera) const {
        // Transform to camera coordinates
        Vec3 point_camera = worldToCamera(point_world, camera);

        // Check if point is in front of camera
        if (point_camera.z <= 0) {
            return PixelCoord(-1, -1);  // Invalid projection
        }

        // Normalize by depth
        double x_norm = point_camera.x / point_camera.z;
        double y_norm = point_camera.y / point_camera.z;

        // Apply distortion
        return applyDistortion(x_norm, y_norm, camera.intrinsics);
    }

    // ========================================================================
    // Multi-Camera Coordinate Fusion
    // ========================================================================

    /**
     * Triangulate 3D point from multiple camera observations
     * Uses DLT (Direct Linear Transform) method
     */
    Vec3 triangulateMultiView(const std::vector<PixelCoord>& pixels,
                             const std::vector<int>& cameraIds) const {
        if (pixels.size() != cameraIds.size() || pixels.size() < 2) {
            std::cerr << "Error: Need at least 2 camera observations" << std::endl;
            return Vec3(0, 0, 0);
        }

        // Build DLT matrix A
        Eigen::MatrixXd A(2 * pixels.size(), 4);

        for (size_t i = 0; i < pixels.size(); i++) {
            const Camera* cam = const_cast<CoordinateTransformer*>(this)->getCamera(cameraIds[i]);
            if (!cam) continue;

            // Undistort pixel
            double x_norm, y_norm;
            undistortPixel(pixels[i], cam->intrinsics, x_norm, y_norm);

            // Get projection matrix P = K[R|t]
            Eigen::Matrix3d K = cam->intrinsics.getCameraMatrix();
            Eigen::Matrix<double, 3, 4> RT = cam->extrinsics.getTransformMatrix();
            Eigen::Matrix<double, 3, 4> P = K * RT;

            // Add constraints
            A.row(2*i)   = pixels[i].u * P.row(2) - P.row(0);
            A.row(2*i+1) = pixels[i].v * P.row(2) - P.row(1);
        }

        // Solve using SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4d X_homogeneous = svd.matrixV().col(3);

        // Convert from homogeneous coordinates
        Vec3 point_world(
            X_homogeneous(0) / X_homogeneous(3),
            X_homogeneous(1) / X_homogeneous(3),
            X_homogeneous(2) / X_homogeneous(3)
        );

        return point_world;
    }

    /**
     * Triangulate from stereo pair using baseline
     */
    Vec3 triangulateStereo(const PixelCoord& pixel_left,
                          const PixelCoord& pixel_right,
                          int leftCamId, int rightCamId) const {
        Camera* cam_left = const_cast<CoordinateTransformer*>(this)->getCamera(leftCamId);
        Camera* cam_right = const_cast<CoordinateTransformer*>(this)->getCamera(rightCamId);

        if (!cam_left || !cam_right) {
            std::cerr << "Error: Invalid camera IDs" << std::endl;
            return Vec3(0, 0, 0);
        }

        // Undistort pixels
        double xl_norm, yl_norm, xr_norm, yr_norm;
        undistortPixel(pixel_left, cam_left->intrinsics, xl_norm, yl_norm);
        undistortPixel(pixel_right, cam_right->intrinsics, xr_norm, yr_norm);

        // Compute disparity
        double disparity = (pixel_left.u - pixel_right.u);

        if (std::abs(disparity) < 0.1) {
            std::cerr << "Warning: Disparity too small for reliable triangulation" << std::endl;
            return Vec3(0, 0, 0);
        }

        // Depth computation (assuming rectified stereo)
        double depth = (cam_left->intrinsics.fx * cam_left->baseline) / disparity;

        // 3D point in left camera frame
        Vec3 point_camera(xl_norm * depth, yl_norm * depth, depth);

        // Transform to world frame
        return cameraToWorld(point_camera, *cam_left);
    }

    // ========================================================================
    // Reprojection Error Computation
    // ========================================================================

    /**
     * Compute reprojection error for a 3D point
     */
    double computeReprojectionError(const Vec3& point_world,
                                   const PixelCoord& observed_pixel,
                                   const Camera& camera) const {
        PixelCoord projected = worldToPixel(point_world, camera);

        double error_u = projected.u - observed_pixel.u;
        double error_v = projected.v - observed_pixel.v;

        return std::sqrt(error_u * error_u + error_v * error_v);
    }

    // ========================================================================
    // Batch Transformations
    // ========================================================================

    /**
     * Batch convert pixels to rays for all cameras
     */
    std::vector<Ray3D> batchPixelsToRays(
        const std::vector<PixelCoord>& pixels,
        const std::vector<int>& cameraIds) const {

        std::vector<Ray3D> rays;
        rays.reserve(pixels.size());

        for (size_t i = 0; i < pixels.size(); i++) {
            Camera* cam = const_cast<CoordinateTransformer*>(this)->getCamera(cameraIds[i]);
            if (cam) {
                rays.push_back(pixelToRay(pixels[i], *cam));
            }
        }

        return rays;
    }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    void printConfiguration() const {
        std::cout << "=== Coordinate Transformer Configuration ===" << std::endl;
        std::cout << "Cameras: " << cameras.size() << std::endl;
        std::cout << "Geodetic mode: " << (useGeodetic ? "Enabled" : "Disabled") << std::endl;

        if (useGeodetic) {
            std::cout << "ENU Origin: ("
                     << enuFrame.origin.latitude << ", "
                     << enuFrame.origin.longitude << ", "
                     << enuFrame.origin.altitude << ")" << std::endl;
        }

        for (const auto& cam : cameras) {
            std::cout << "\nCamera " << cam.id << " (" << cam.name << "):" << std::endl;
            std::cout << "  Resolution: " << cam.intrinsics.width << "x" << cam.intrinsics.height << std::endl;
            std::cout << "  Focal length: (" << cam.intrinsics.fx << ", " << cam.intrinsics.fy << ")" << std::endl;
            std::cout << "  Position: (" << cam.extrinsics.position.x << ", "
                     << cam.extrinsics.position.y << ", "
                     << cam.extrinsics.position.z << ")" << std::endl;
        }
        std::cout << "=============================================" << std::endl;
    }
};

// ============================================================================
// Example Usage
// ============================================================================

#ifdef BUILD_EXAMPLE

int main() {
    std::cout << "Coordinate Transformation System" << std::endl;
    std::cout << "=================================" << std::endl;

    // Create transformer
    CoordinateTransformer transformer(true);

    // Set ENU origin (example: somewhere in the tracking area)
    GeodeticCoord origin(37.7749, -122.4194, 0);  // San Francisco
    transformer.setENUOrigin(origin);

    // Create stereo camera pair
    Camera cam_left, cam_right;
    cam_left.id = 0;
    cam_left.name = "left";
    cam_left.intrinsics.fx = cam_left.intrinsics.fy = 1000.0;
    cam_left.intrinsics.cx = 960.0;
    cam_left.intrinsics.cy = 540.0;
    cam_left.extrinsics.position = Vec3(0, 0, 0);
    cam_left.baseline = 0.5;  // 50cm baseline

    cam_right = cam_left;
    cam_right.id = 1;
    cam_right.name = "right";
    cam_right.extrinsics.position = Vec3(0.5, 0, 0);  // 50cm to the right

    transformer.addCamera(cam_left);
    transformer.addCamera(cam_right);

    transformer.printConfiguration();

    // Test pixel to ray conversion
    std::cout << "\nTest 1: Pixel to Ray Conversion" << std::endl;
    PixelCoord pixel(960, 540);  // Center pixel
    Ray3D ray = transformer.pixelToRay(pixel, cam_left);
    std::cout << "Ray origin: (" << ray.origin.x << ", " << ray.origin.y << ", " << ray.origin.z << ")" << std::endl;
    std::cout << "Ray direction: (" << ray.direction.x << ", " << ray.direction.y << ", " << ray.direction.z << ")" << std::endl;

    // Test stereo triangulation
    std::cout << "\nTest 2: Stereo Triangulation" << std::endl;
    PixelCoord pixel_left(1000, 540);
    PixelCoord pixel_right(960, 540);  // 40 pixel disparity
    Vec3 point_3d = transformer.triangulateStereo(pixel_left, pixel_right, 0, 1);
    std::cout << "3D Point: (" << point_3d.x << ", " << point_3d.y << ", " << point_3d.z << ")" << std::endl;

    // Test reprojection error
    std::cout << "\nTest 3: Reprojection Error" << std::endl;
    PixelCoord reprojected = transformer.worldToPixel(point_3d, cam_left);
    double error = transformer.computeReprojectionError(point_3d, pixel_left, cam_left);
    std::cout << "Reprojection error: " << error << " pixels" << std::endl;

    if (error < 0.5) {
        std::cout << "✓ Reprojection error < 0.5 pixels" << std::endl;
    }

    // Test geodetic transformations
    std::cout << "\nTest 4: Geodetic Transformations" << std::endl;
    GeodeticCoord test_geo(37.7750, -122.4195, 100);
    Vec3 enu = transformer.geodeticToENU(test_geo);
    std::cout << "ENU coordinates: (" << enu.x << ", " << enu.y << ", " << enu.z << ")" << std::endl;

    GeodeticCoord back_geo = transformer.enuToGeodetic(enu);
    std::cout << "Back to geodetic: (" << back_geo.latitude << ", "
             << back_geo.longitude << ", " << back_geo.altitude << ")" << std::endl;

    return 0;
}

#endif
