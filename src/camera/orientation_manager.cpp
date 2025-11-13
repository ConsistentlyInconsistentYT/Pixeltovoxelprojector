/**
 * Orientation Manager - High-frequency quaternion-based orientation tracking
 *
 * Features:
 * - Quaternion-based orientation (gimbal lock prevention)
 * - 1000Hz IMU integration
 * - Smooth orientation interpolation (SLERP)
 * - Complementary filter for IMU fusion
 * - Sub-millisecond timestamp synchronization
 *
 * Author: Pixeltovoxelprojector Team
 */

#include <array>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

// High-precision timestamp type (nanoseconds)
using Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using NanoSeconds = std::chrono::nanoseconds;

// Forward declarations
class Quaternion;
class OrientationManager;

/**
 * Quaternion class for rotation representation
 * Format: [w, x, y, z] (scalar-first convention)
 */
class Quaternion {
public:
    double w, x, y, z;

    // Constructors
    Quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) {}
    Quaternion(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}

    /**
     * Create quaternion from Euler angles (roll, pitch, yaw) in radians
     */
    static Quaternion fromEuler(double roll, double pitch, double yaw) {
        double cy = std::cos(yaw * 0.5);
        double sy = std::sin(yaw * 0.5);
        double cp = std::cos(pitch * 0.5);
        double sp = std::sin(pitch * 0.5);
        double cr = std::cos(roll * 0.5);
        double sr = std::sin(roll * 0.5);

        Quaternion q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;

        return q.normalized();
    }

    /**
     * Create quaternion from axis-angle representation
     */
    static Quaternion fromAxisAngle(const std::array<double, 3>& axis, double angle) {
        double half_angle = angle * 0.5;
        double s = std::sin(half_angle);
        double norm = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);

        if (norm < 1e-10) {
            return Quaternion(1.0, 0.0, 0.0, 0.0);
        }

        Quaternion q;
        q.w = std::cos(half_angle);
        q.x = axis[0] / norm * s;
        q.y = axis[1] / norm * s;
        q.z = axis[2] / norm * s;

        return q;
    }

    /**
     * Convert quaternion to Euler angles (roll, pitch, yaw) in radians
     */
    std::array<double, 3> toEuler() const {
        std::array<double, 3> euler;

        // Roll (x-axis rotation)
        double sinr_cosp = 2.0 * (w * x + y * z);
        double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        euler[0] = std::atan2(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        double sinp = 2.0 * (w * y - z * x);
        if (std::abs(sinp) >= 1.0)
            euler[1] = std::copysign(M_PI / 2.0, sinp); // Use 90 degrees if out of range
        else
            euler[1] = std::asin(sinp);

        // Yaw (z-axis rotation)
        double siny_cosp = 2.0 * (w * z + x * y);
        double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        euler[2] = std::atan2(siny_cosp, cosy_cosp);

        return euler;
    }

    /**
     * Quaternion multiplication
     */
    Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        );
    }

    /**
     * Quaternion conjugate
     */
    Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }

    /**
     * Quaternion norm
     */
    double norm() const {
        return std::sqrt(w*w + x*x + y*y + z*z);
    }

    /**
     * Normalize quaternion
     */
    Quaternion normalized() const {
        double n = norm();
        if (n < 1e-10) {
            return Quaternion(1.0, 0.0, 0.0, 0.0);
        }
        return Quaternion(w/n, x/n, y/n, z/n);
    }

    /**
     * Spherical Linear Interpolation (SLERP)
     */
    static Quaternion slerp(const Quaternion& q0, const Quaternion& q1, double t) {
        Quaternion q0_norm = q0.normalized();
        Quaternion q1_norm = q1.normalized();

        // Compute dot product
        double dot = q0_norm.w * q1_norm.w + q0_norm.x * q1_norm.x +
                     q0_norm.y * q1_norm.y + q0_norm.z * q1_norm.z;

        // If quaternions are close, use linear interpolation
        if (dot > 0.9995) {
            Quaternion result(
                q0_norm.w + t * (q1_norm.w - q0_norm.w),
                q0_norm.x + t * (q1_norm.x - q0_norm.x),
                q0_norm.y + t * (q1_norm.y - q0_norm.y),
                q0_norm.z + t * (q1_norm.z - q0_norm.z)
            );
            return result.normalized();
        }

        // Ensure shortest path
        Quaternion q1_adjusted = q1_norm;
        if (dot < 0.0) {
            q1_adjusted.w = -q1_adjusted.w;
            q1_adjusted.x = -q1_adjusted.x;
            q1_adjusted.y = -q1_adjusted.y;
            q1_adjusted.z = -q1_adjusted.z;
            dot = -dot;
        }

        // Clamp dot product
        dot = std::max(-1.0, std::min(1.0, dot));

        double theta_0 = std::acos(dot);
        double theta = theta_0 * t;

        // Compute orthogonal quaternion
        Quaternion q2(
            q1_adjusted.w - q0_norm.w * dot,
            q1_adjusted.x - q0_norm.x * dot,
            q1_adjusted.y - q0_norm.y * dot,
            q1_adjusted.z - q0_norm.z * dot
        );
        q2 = q2.normalized();

        // Final interpolation
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);

        return Quaternion(
            q0_norm.w * cos_theta + q2.w * sin_theta,
            q0_norm.x * cos_theta + q2.x * sin_theta,
            q0_norm.y * cos_theta + q2.y * sin_theta,
            q0_norm.z * cos_theta + q2.z * sin_theta
        ).normalized();
    }

    /**
     * Rotate vector by quaternion
     */
    std::array<double, 3> rotateVector(const std::array<double, 3>& v) const {
        Quaternion v_quat(0.0, v[0], v[1], v[2]);
        Quaternion result = (*this) * v_quat * this->conjugate();
        return {result.x, result.y, result.z};
    }

    /**
     * Get rotation matrix (3x3) representation
     */
    std::array<std::array<double, 3>, 3> toRotationMatrix() const {
        std::array<std::array<double, 3>, 3> R;

        double xx = x * x;
        double yy = y * y;
        double zz = z * z;
        double xy = x * y;
        double xz = x * z;
        double yz = y * z;
        double wx = w * x;
        double wy = w * y;
        double wz = w * z;

        R[0][0] = 1.0 - 2.0 * (yy + zz);
        R[0][1] = 2.0 * (xy - wz);
        R[0][2] = 2.0 * (xz + wy);

        R[1][0] = 2.0 * (xy + wz);
        R[1][1] = 1.0 - 2.0 * (xx + zz);
        R[1][2] = 2.0 * (yz - wx);

        R[2][0] = 2.0 * (xz - wy);
        R[2][1] = 2.0 * (yz + wx);
        R[2][2] = 1.0 - 2.0 * (xx + yy);

        return R;
    }
};

/**
 * IMU measurement structure
 */
struct IMUMeasurement {
    uint64_t timestamp_ns;  // Nanosecond timestamp
    std::array<double, 3> angular_velocity;  // rad/s [wx, wy, wz]
    std::array<double, 3> linear_acceleration;  // m/s^2 [ax, ay, az]
    int camera_id;

    IMUMeasurement() : timestamp_ns(0), camera_id(0) {
        angular_velocity.fill(0.0);
        linear_acceleration.fill(0.0);
    }
};

/**
 * Orientation state for a single camera
 */
struct OrientationState {
    Quaternion orientation;
    std::array<double, 3> angular_velocity;
    std::array<double, 3> gyro_bias;
    uint64_t timestamp_ns;
    double accuracy_deg;  // Estimated orientation accuracy in degrees

    OrientationState() : timestamp_ns(0), accuracy_deg(1.0) {
        orientation = Quaternion(1.0, 0.0, 0.0, 0.0);
        angular_velocity.fill(0.0);
        gyro_bias.fill(0.0);
    }
};

/**
 * Complementary filter for IMU fusion
 * Combines gyroscope (high-frequency) with accelerometer (low-frequency drift correction)
 */
class ComplementaryFilter {
private:
    double alpha;  // Filter coefficient (0.98 typical)
    std::array<double, 3> gyro_bias;

public:
    ComplementaryFilter(double alpha = 0.98) : alpha(alpha) {
        gyro_bias.fill(0.0);
    }

    /**
     * Update orientation using gyroscope and accelerometer
     */
    Quaternion update(const Quaternion& current_orientation,
                      const std::array<double, 3>& gyro,
                      const std::array<double, 3>& accel,
                      double dt) {

        // Remove bias from gyroscope
        std::array<double, 3> gyro_corrected = {
            gyro[0] - gyro_bias[0],
            gyro[1] - gyro_bias[1],
            gyro[2] - gyro_bias[2]
        };

        // Integrate gyroscope (predict orientation)
        double angle = std::sqrt(gyro_corrected[0]*gyro_corrected[0] +
                                 gyro_corrected[1]*gyro_corrected[1] +
                                 gyro_corrected[2]*gyro_corrected[2]) * dt;

        Quaternion q_gyro = current_orientation;
        if (angle > 1e-10) {
            Quaternion delta_q = Quaternion::fromAxisAngle(gyro_corrected, angle);
            q_gyro = current_orientation * delta_q;
            q_gyro = q_gyro.normalized();
        }

        // Correct drift using accelerometer (measure gravity direction)
        double accel_norm = std::sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]);

        if (accel_norm < 1e-6) {
            return q_gyro;  // No accelerometer correction available
        }

        // Normalized acceleration vector
        std::array<double, 3> accel_normalized = {
            accel[0] / accel_norm,
            accel[1] / accel_norm,
            accel[2] / accel_norm
        };

        // Expected gravity direction in body frame
        std::array<double, 3> gravity_world = {0.0, 0.0, -1.0};
        std::array<double, 3> gravity_body = q_gyro.conjugate().rotateVector(gravity_world);

        // Correction quaternion from accelerometer
        Quaternion q_accel = Quaternion::fromAxisAngle({
            accel_normalized[1] * gravity_body[2] - accel_normalized[2] * gravity_body[1],
            accel_normalized[2] * gravity_body[0] - accel_normalized[0] * gravity_body[2],
            accel_normalized[0] * gravity_body[1] - accel_normalized[1] * gravity_body[0]
        }, 1.0 - alpha);

        // Fuse gyro and accel
        Quaternion q_fused = q_accel * q_gyro;
        return q_fused.normalized();
    }

    void updateGyroBias(const std::array<double, 3>& bias_update) {
        // Slowly adapt gyro bias
        const double bias_alpha = 0.01;
        gyro_bias[0] += bias_alpha * bias_update[0];
        gyro_bias[1] += bias_alpha * bias_update[1];
        gyro_bias[2] += bias_alpha * bias_update[2];
    }

    std::array<double, 3> getGyroBias() const {
        return gyro_bias;
    }
};

/**
 * Orientation Manager - Main class
 *
 * Manages high-frequency orientation tracking for multiple cameras
 */
class OrientationManager {
private:
    static constexpr int MAX_CAMERAS = 20;
    static constexpr int MAX_HISTORY = 1000;  // Keep 1 second of history at 1000Hz

    // Per-camera state
    std::array<OrientationState, MAX_CAMERAS> current_states;
    std::array<std::deque<OrientationState>, MAX_CAMERAS> state_history;
    std::array<ComplementaryFilter, MAX_CAMERAS> filters;
    std::array<std::mutex, MAX_CAMERAS> camera_mutexes;

    // Thread management
    std::atomic<bool> running;
    std::vector<std::thread> processing_threads;

    // Performance monitoring
    std::array<std::atomic<uint64_t>, MAX_CAMERAS> update_counts;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_performance_report;

    int num_cameras;
    double target_rate_hz;

public:
    OrientationManager(int num_cameras = 20, double rate_hz = 1000.0)
        : running(false), num_cameras(num_cameras), target_rate_hz(rate_hz) {

        if (num_cameras > MAX_CAMERAS) {
            throw std::runtime_error("Too many cameras");
        }

        // Initialize filters
        for (int i = 0; i < num_cameras; ++i) {
            filters[i] = ComplementaryFilter(0.98);
            update_counts[i] = 0;
        }

        last_performance_report = std::chrono::high_resolution_clock::now();

        std::cout << "OrientationManager initialized for " << num_cameras
                  << " cameras at " << rate_hz << " Hz" << std::endl;
    }

    ~OrientationManager() {
        stop();
    }

    /**
     * Start orientation tracking
     */
    void start() {
        running = true;

        // Start monitoring thread
        processing_threads.emplace_back(&OrientationManager::performanceMonitor, this);

        std::cout << "OrientationManager started" << std::endl;
    }

    /**
     * Stop orientation tracking
     */
    void stop() {
        running = false;
        for (auto& thread : processing_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        processing_threads.clear();

        std::cout << "OrientationManager stopped" << std::endl;
    }

    /**
     * Process IMU measurement
     *
     * This is the main high-frequency update function called at 1000Hz
     */
    void processIMU(const IMUMeasurement& measurement) {
        int camera_id = measurement.camera_id;

        if (camera_id < 0 || camera_id >= num_cameras) {
            return;
        }

        std::lock_guard<std::mutex> lock(camera_mutexes[camera_id]);

        OrientationState& state = current_states[camera_id];

        // Calculate time delta
        double dt = 0.001;  // Default 1ms for 1000Hz
        if (state.timestamp_ns > 0) {
            dt = (measurement.timestamp_ns - state.timestamp_ns) * 1e-9;

            // Sanity check
            if (dt <= 0.0 || dt > 0.1) {
                dt = 0.001;
            }
        }

        // Update orientation using complementary filter
        state.orientation = filters[camera_id].update(
            state.orientation,
            measurement.angular_velocity,
            measurement.linear_acceleration,
            dt
        );

        // Update angular velocity
        state.angular_velocity = measurement.angular_velocity;
        state.gyro_bias = filters[camera_id].getGyroBias();
        state.timestamp_ns = measurement.timestamp_ns;

        // Estimate accuracy based on accelerometer quality
        double accel_mag = std::sqrt(
            measurement.linear_acceleration[0] * measurement.linear_acceleration[0] +
            measurement.linear_acceleration[1] * measurement.linear_acceleration[1] +
            measurement.linear_acceleration[2] * measurement.linear_acceleration[2]
        );
        double gravity_error = std::abs(accel_mag - 9.81);
        state.accuracy_deg = 0.01 + gravity_error * 0.1;  // Accuracy degrades with poor accel

        // Add to history
        state_history[camera_id].push_back(state);
        if (state_history[camera_id].size() > MAX_HISTORY) {
            state_history[camera_id].pop_front();
        }

        update_counts[camera_id]++;
    }

    /**
     * Get current orientation for a camera
     */
    OrientationState getOrientation(int camera_id) const {
        if (camera_id < 0 || camera_id >= num_cameras) {
            return OrientationState();
        }

        std::lock_guard<std::mutex> lock(camera_mutexes[camera_id]);
        return current_states[camera_id];
    }

    /**
     * Get interpolated orientation at specific timestamp
     */
    OrientationState getOrientationAt(int camera_id, uint64_t timestamp_ns) const {
        if (camera_id < 0 || camera_id >= num_cameras) {
            return OrientationState();
        }

        std::lock_guard<std::mutex> lock(camera_mutexes[camera_id]);

        const auto& history = state_history[camera_id];

        if (history.empty()) {
            return current_states[camera_id];
        }

        // Find bounding states
        const OrientationState* before = nullptr;
        const OrientationState* after = nullptr;

        for (const auto& state : history) {
            if (state.timestamp_ns <= timestamp_ns) {
                before = &state;
            }
            if (state.timestamp_ns >= timestamp_ns && after == nullptr) {
                after = &state;
            }
        }

        // If exact match or only one bound, return it
        if (before == nullptr) {
            return history.front();
        }
        if (after == nullptr) {
            return history.back();
        }
        if (before->timestamp_ns == after->timestamp_ns) {
            return *before;
        }

        // Interpolate using SLERP
        uint64_t dt_total = after->timestamp_ns - before->timestamp_ns;
        double t = static_cast<double>(timestamp_ns - before->timestamp_ns) / dt_total;

        OrientationState result;
        result.timestamp_ns = timestamp_ns;
        result.orientation = Quaternion::slerp(before->orientation, after->orientation, t);

        // Linear interpolation for angular velocity
        for (int i = 0; i < 3; ++i) {
            result.angular_velocity[i] = before->angular_velocity[i] +
                t * (after->angular_velocity[i] - before->angular_velocity[i]);
        }

        result.accuracy_deg = std::max(before->accuracy_deg, after->accuracy_deg);

        return result;
    }

    /**
     * Get all camera orientations at once (parallel processing)
     */
    std::vector<OrientationState> getAllOrientations() const {
        std::vector<OrientationState> states(num_cameras);

        #pragma omp parallel for if(num_cameras > 4)
        for (int i = 0; i < num_cameras; ++i) {
            states[i] = getOrientation(i);
        }

        return states;
    }

    /**
     * Set orientation directly (for initialization or external updates)
     */
    void setOrientation(int camera_id, const Quaternion& orientation, uint64_t timestamp_ns) {
        if (camera_id < 0 || camera_id >= num_cameras) {
            return;
        }

        std::lock_guard<std::mutex> lock(camera_mutexes[camera_id]);
        current_states[camera_id].orientation = orientation.normalized();
        current_states[camera_id].timestamp_ns = timestamp_ns;
    }

private:
    /**
     * Performance monitoring thread
     */
    void performanceMonitor() {
        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(now - last_performance_report).count();

            std::cout << "\n=== Orientation Manager Performance ===" << std::endl;

            for (int i = 0; i < num_cameras; ++i) {
                uint64_t count = update_counts[i].exchange(0);
                double rate = count / elapsed;

                OrientationState state = getOrientation(i);
                auto euler = state.orientation.toEuler();

                std::cout << "Camera " << std::setw(2) << i << ": "
                          << std::fixed << std::setprecision(1) << std::setw(7) << rate << " Hz | "
                          << "Roll: " << std::setw(7) << std::setprecision(2) << (euler[0] * 180.0 / M_PI) << "° | "
                          << "Pitch: " << std::setw(7) << std::setprecision(2) << (euler[1] * 180.0 / M_PI) << "° | "
                          << "Yaw: " << std::setw(7) << std::setprecision(2) << (euler[2] * 180.0 / M_PI) << "° | "
                          << "Accuracy: " << std::setprecision(3) << state.accuracy_deg << "°"
                          << std::endl;
            }

            last_performance_report = now;
        }
    }
};

/**
 * C interface for Python binding (using pybind11)
 */
extern "C" {
    // Factory functions
    OrientationManager* createOrientationManager(int num_cameras, double rate_hz) {
        return new OrientationManager(num_cameras, rate_hz);
    }

    void destroyOrientationManager(OrientationManager* manager) {
        delete manager;
    }

    // Control functions
    void startOrientationManager(OrientationManager* manager) {
        manager->start();
    }

    void stopOrientationManager(OrientationManager* manager) {
        manager->stop();
    }

    // Processing function
    void processIMUMeasurement(OrientationManager* manager,
                               int camera_id,
                               uint64_t timestamp_ns,
                               const double* angular_velocity,
                               const double* linear_acceleration) {
        IMUMeasurement measurement;
        measurement.camera_id = camera_id;
        measurement.timestamp_ns = timestamp_ns;

        for (int i = 0; i < 3; ++i) {
            measurement.angular_velocity[i] = angular_velocity[i];
            measurement.linear_acceleration[i] = linear_acceleration[i];
        }

        manager->processIMU(measurement);
    }

    // Query functions
    void getOrientation(OrientationManager* manager,
                       int camera_id,
                       double* quaternion,  // [w, x, y, z]
                       double* euler,       // [roll, pitch, yaw]
                       double* angular_vel, // [wx, wy, wz]
                       double* accuracy) {
        auto state = manager->getOrientation(camera_id);

        quaternion[0] = state.orientation.w;
        quaternion[1] = state.orientation.x;
        quaternion[2] = state.orientation.y;
        quaternion[3] = state.orientation.z;

        auto euler_angles = state.orientation.toEuler();
        euler[0] = euler_angles[0];
        euler[1] = euler_angles[1];
        euler[2] = euler_angles[2];

        angular_vel[0] = state.angular_velocity[0];
        angular_vel[1] = state.angular_velocity[1];
        angular_vel[2] = state.angular_velocity[2];

        *accuracy = state.accuracy_deg;
    }
}

// Example usage
int main() {
    std::cout << "Orientation Manager Test" << std::endl;
    std::cout << "=========================" << std::endl;

    // Create manager for 20 cameras at 1000Hz
    OrientationManager manager(20, 1000.0);
    manager.start();

    // Simulate IMU data for camera 0
    const int camera_id = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run for 5 seconds
    for (int i = 0; i < 5000; ++i) {
        auto current_time = std::chrono::high_resolution_clock::now();
        uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            current_time.time_since_epoch()).count();

        IMUMeasurement measurement;
        measurement.camera_id = camera_id;
        measurement.timestamp_ns = timestamp_ns;

        // Simulate small rotation
        measurement.angular_velocity = {0.01, 0.02, 0.005};

        // Simulate gravity with small noise
        measurement.linear_acceleration = {0.1, 0.0, 9.81};

        manager.processIMU(measurement);

        // Sleep for 1ms (1000Hz)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Get final orientation
    auto final_state = manager.getOrientation(camera_id);
    auto euler = final_state.orientation.toEuler();

    std::cout << "\n=== Final Orientation ===" << std::endl;
    std::cout << "Roll:  " << (euler[0] * 180.0 / M_PI) << "°" << std::endl;
    std::cout << "Pitch: " << (euler[1] * 180.0 / M_PI) << "°" << std::endl;
    std::cout << "Yaw:   " << (euler[2] * 180.0 / M_PI) << "°" << std::endl;
    std::cout << "Accuracy: " << final_state.accuracy_deg << "°" << std::endl;

    manager.stop();

    return 0;
}
