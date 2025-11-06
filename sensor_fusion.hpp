// sensor_fusion.hpp - Unified Sensor Fusion Engine for AstraVoxel
// Author: AI Architect
// Description: Core multi-sensor data fusion capabilities for real-time 3D reconstruction

#pragma once

#ifndef ASTRAXEL_SENSOR_FUSION_HPP
#define ASTRAXEL_SENSOR_FUSION_HPP

#include <vector>
#include <array>
#include <memory>
#include <string>
#include <cmath>
#include <cstdint>
#include "nlohmann/json.hpp"

// Basic math utilities (replacing Eigen for portability)
namespace astraxel_math {

// 3D Vector class
struct Vector3d {
    double x = 0.0, y = 0.0, z = 0.0;

    Vector3d() = default;
    Vector3d(double x, double y, double z) : x(x), y(y), z(z) {}

    Vector3d operator+(const Vector3d& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vector3d operator-(const Vector3d& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vector3d& operator+=(const Vector3d& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }

    Vector3d operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    double dot(const Vector3d& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vector3d cross(const Vector3d& other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    double norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    Vector3d normalized() const {
        double len = norm();
        if (len < 1e-12) return {0, 0, 0};
        return {x/len, y/len, z/len};
    }
};

// 3x3 Matrix class
struct Matrix3d {
    double m[9] = {0}; // Row-major order

    static Matrix3d identity() {
        Matrix3d mat;
        mat.m[0] = mat.m[4] = mat.m[8] = 1.0;
        return mat;
    }

    static Matrix3d rotation_x(double angle_rad);
    static Matrix3d rotation_y(double angle_rad);
    static Matrix3d rotation_z(double angle_rad);

    Vector3d operator*(const Vector3d& v) const {
        return {
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z
        };
    }

    Matrix3d operator*(const Matrix3d& other) const;
};

// Quaternion class
struct Quaternion {
    double w = 1.0, x = 0.0, y = 0.0, z = 0.0;

    Quaternion() = default;
    Quaternion(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}

    static Quaternion from_euler(double yaw, double pitch, double roll);
    Matrix3d to_rotation_matrix() const;
    Quaternion normalized() const;
};

}

namespace astraxel {

// Forward declarations
class SensorInterface;
class FusionStrategy;
class QualityAssurance;

//==============================================================================
// Core Data Structures
//==============================================================================

/**
 * @brief Sensor calibration data structure
 * Stores intrinsic and extrinsic parameters for each sensor
 */
struct SensorCalibration {
    int sensor_id;
    std::string sensor_type;  // "optical", "thermal", "radar", "lidar"
    std::string coordinate_system;  // "cartesian", "spherical", "ra_dec"

    // Position and orientation
    astraxel_math::Vector3d position;    // 3D world position
    astraxel_math::Quaternion orientation;  // Rotation from sensor to world frame
    astraxel_math::Matrix3d rotation_matrix() const { return orientation.to_rotation_matrix(); }

    // Sensor pose (transformation matrix)
    astraxel_math::Matrix3d pose_rotation;
    astraxel_math::Vector3d pose_translation;

    // sensor-specific parameters
    double field_of_view_rad;   // radians
    double min_range;           // meters for range sensors
    double max_range;           // meters for range sensors
    astraxel_math::Matrix3d intrinsic_matrix;  // camera matrix for optical sensors

    // Quality metrics
    double calibration_uncertainty;  // meters
    double angular_uncertainty;      // radians
    bool is_calibrated = true;

    // Conversion to/from JSON
    nlohmann::json to_json() const;
    static SensorCalibration from_json(const nlohmann::json& j);
};

/**
 * @brief Sensor measurement data structure
 * Contains measurement data from individual sensors
 */
struct SensorMeasurement {
    int sensor_id;
    uint64_t timestamp_ns;
    std::string measurement_type;  // "raw_image", "point_cloud", "range_data"

    // Raw measurement data
    std::vector<uint8_t> raw_data;
    size_t width = 0;      // for image data
    size_t height = 0;     // for image data
    size_t point_count = 0;  // for point cloud data

    // Processing flags
    bool processed = false;
    bool valid = true;
    double processing_time_ms = 0.0;

    // Quality metrics
    double signal_to_noise_ratio = 0.0;
    double confidence_level = 1.0;

    // Motion detection results (if applicable)
    std::vector<astraxel_math::Vector3d> motion_vectors;  // 2D vectors stored as 3D with z=0
    double motion_magnitude = 0.0;
};

/**
 * @brief Fusion result data structure
 * Contains the integrated results from multiple sensors
 */
struct FusionResult {
    uint64_t timestamp_ns;
    std::vector<int> sensor_ids;

    // Fused voxel data
    std::vector<astraxel_math::Vector3d> point_cloud;
    std::vector<float> evidences;    // confidence values
    std::vector<float> intensities;  // brightness/intensity values

    // Statistical properties
    astraxel_math::Vector3d centroid;
    astraxel_math::Matrix3d covariance;
    double volume = 0.0;

    // Detection metrics
    bool object_detected = false;
    astraxel_math::Vector3d object_position;
    astraxel_math::Vector3d object_velocity;
    double object_confidence = 0.0;

    // Quality assessment
    double fusion_confidence = 0.0;
    double noise_level = 0.0;
};

/**
 * @brief Voxel grid for fused sensor data
 * 3D spatial grid for accumulating motion evidence
 */
struct Vector3i {
    int x = 0, y = 0, z = 0;
};

class VoxelGrid {
private:
    astraxel_math::Vector3d origin_;
    astraxel_math::Vector3d resolution_;  // voxel size in each dimension
    Vector3i dimensions_;  // num voxels in each dimension

    std::vector<float> evidence_grid_;
    std::vector<float> intensity_grid_;
    std::vector<int> observation_counts_;

public:
    VoxelGrid(const astraxel_math::Vector3d& origin, const astraxel_math::Vector3d& resolution,
             const Vector3i& dimensions);

    bool accumulate_point(const astraxel_math::Vector3d& point, float evidence, float intensity);
    bool get_voxel_data(int x, int y, int z, float& evidence, float& intensity, int& count) const;
    bool reset();

    // Accessors
    const astraxel_math::Vector3d& origin() const { return origin_; }
    const astraxel_math::Vector3d& resolution() const { return resolution_; }
    const Vector3i& dimensions() const { return dimensions_; }
    size_t total_voxels() const { return evidence_grid_.size(); }

    // Data export
    std::vector<float> get_evidence_data() const { return evidence_grid_; }
    std::vector<float> get_intensity_data() const { return intensity_grid_; }

    // Utility functions
    size_t point_to_voxel_index(const astraxel_math::Vector3d& point) const;
    bool is_valid_voxel(int x, int y, int z) const;

private:
    void initialize_grid();
};

//==============================================================================
// Sensor Interface
//==============================================================================

/**
 * @brief Abstract base class for sensor interfaces
 * Defines the interface that all sensor types must implement
 */
class SensorInterface {
public:
    virtual ~SensorInterface() = default;

    /**
     * @brief Get sensor calibration data
     */
    virtual const SensorCalibration& get_calibration() const = 0;

    /**
     * @brief Capture a measurement from the sensor
     * @return SensorMeasurement containing the captured data
     */
    virtual SensorMeasurement capture_measurement() = 0;

    /**
     * @brief Process raw measurement into usable format
     * @param measurement Raw measurement to process
     * @return Processed measurement data
     */
    virtual SensorMeasurement process_measurement(const SensorMeasurement& measurement) = 0;

    /**
     * @brief Extract motion information from measurement
     * @param current Current measurement
     * @param previous Previous measurement for comparison
     * @return Motion vectors and magnitudes
     */
    virtual std::vector<astraxel_math::Vector3d> extract_motion_vectors(
        const SensorMeasurement& current,
        const SensorMeasurement& previous = SensorMeasurement{}) = 0;

    /**
     * @brief Project motion data into 3D space
     * @param measurements Set of measurements to project
     * @param grid Voxel grid to accumulate projections
     * @return Success status
     */
    virtual bool project_to_voxels(const std::vector<SensorMeasurement>& measurements,
                                 VoxelGrid& grid) = 0;

    /**
     * @brief Check sensor connectivity and status
     */
    virtual bool is_connected() const = 0;

    /**
     * @brief Get sensor health metrics
     */
    virtual double get_health_score() const = 0;

    /**
     * @brief Reinitialize sensor
     */
    virtual bool reinitialize() = 0;
};

//==============================================================================
// Fusion Strategies
//==============================================================================

/**
 * @brief Abstract base class for fusion strategies
 * Defines different methods for combining sensor data
 */
class FusionStrategy {
public:
    virtual ~FusionStrategy() = default;

    /**
     * @brief Fuse multiple sensor measurements
     * @param measurements Set of measurements from different sensors
     * @param calibrations Corresponding sensor calibrations
     * @return Fused result
     */
    virtual FusionResult fuse_measurements(
        const std::vector<SensorMeasurement>& measurements,
        const std::vector<SensorCalibration>& calibrations) = 0;

    /**
     * @brief Update strategy parameters based on performance feedback
     */
    virtual void update_parameters(double performance_score) = 0;

    /**
     * @brief Get strategy parameters for serialization
     */
    virtual nlohmann::json get_parameters() const = 0;
};

/**
 * @brief Bayesian fusion strategy
 * Uses probabilistic models for sensor data integration
 */
class BayesianFusion : public FusionStrategy {
private:
    double prior_weight_ = 0.3;
    double likelihood_threshold_ = 0.8;
    bool use_correlation_ = true;

public:
    FusionResult fuse_measurements(
        const std::vector<SensorMeasurement>& measurements,
        const std::vector<SensorCalibration>& calibrations) override;

    void update_parameters(double performance_score) override;
    nlohmann::json get_parameters() const override;
};

/**
 * @brief Evidence accumulation fusion strategy
 * Accumulates evidence from multiple sensors with confidence weighting
 */
class EvidenceAccumulationFusion : public FusionStrategy {
private:
    double evidence_threshold_ = 0.1;
    double decay_factor_ = 0.95;
    bool normalize_evidence_ = true;

public:
    FusionResult fuse_measurements(
        const std::vector<SensorMeasurement>& measurements,
        const std::vector<SensorCalibration>& calibrations) override;

    void update_parameters(double performance_score) override;
    nlohmann::json get_parameters() const override;
};

//==============================================================================
// Quality Assurance
//==============================================================================

/**
 * @brief Quality assessment system for sensor fusion
 */
class QualityAssurance {
public:
    struct QualityMetrics {
        double data_completeness = 0.0;      // 0-1 coverage ratio
        double temporal_consistency = 0.0;   // 0-1 smoothness
        double spatial_consistency = 0.0;    // 0-1 agreement
        double sensor_diversity = 0.0;       // 0-1 complementary coverage
        double confidence_score = 0.0;       // 0-1 overall quality

        nlohmann::json to_json() const;
    };

    static QualityMetrics assess_fusion_quality(
        const FusionResult& result,
        const std::vector<SensorMeasurement>& measurements,
        const std::vector<SensorCalibration>& calibrations);

    static std::vector<std::string> generate_quality_report(
        const QualityMetrics& metrics,
        const std::vector<SensorMeasurement>& measurements);
};

//==============================================================================
// Main Sensor Fusion Engine
//==============================================================================

/**
 * @brief Main sensor fusion engine class
 * Manages multiple sensors, handles data fusion, and provides unified interface
 */
class AstraVoxelFusionEngine {
private:
    std::vector<std::unique_ptr<SensorInterface>> sensors_;
    std::unique_ptr<FusionStrategy> fusion_strategy_;
    std::unique_ptr<VoxelGrid> voxel_grid_;
    QualityAssurance quality_assessor_;

    // Configuration
    double fusion_rate_hz_ = 10.0;  // Fusion frequency
    bool real_time_mode_ = true;
    std::string output_directory_ = "./fusion_output";

    // State management
    bool initialized_ = false;
    uint64_t last_fusion_timestamp_ = 0;
    std::vector<FusionResult> recent_results_;

    // Performance monitoring
    double average_processing_time_ms_ = 0.0;
    size_t successful_fusions_ = 0;
    size_t failed_fusions_ = 0;

public:
    AstraVoxelFusionEngine();
    ~AstraVoxelFusionEngine();

    /**
     * @brief Initialize the fusion engine
     * @param config Configuration parameters
     * @return Success status
     */
    bool initialize(const nlohmann::json& config = nlohmann::json());

    /**
     * @brief Add a sensor to the fusion system
     * @param sensor Sensor interface to add
     * @return Assigned sensor ID
     */
    int add_sensor(std::unique_ptr<SensorInterface> sensor);

    /**
     * @brief Remove a sensor from the fusion system
     * @param sensor_id ID of sensor to remove
     * @return Success status
     */
    bool remove_sensor(int sensor_id);

    /**
     * @brief Start the fusion process
     * @return Success status
     */
    bool start_fusion();

    /**
     * @brief Stop the fusion process
     */
    void stop_fusion();

    /**
     * @brief Perform one fusion step
     * @return Latest fusion result
     */
    FusionResult fuse_data();

    /**
     * @brief Get current system status
     */
    nlohmann::json get_system_status() const;

    /**
     * @brief Export fusion results
     * @param format Output format ("json", "binary", "npy")
     * @return Success status
     */
    bool export_results(const std::string& format = "json");

    /**
     * @brief Configure fusion parameters
     */
    void configure_fusion_parameters(const nlohmann::json& params);

private:
    bool validate_sensor_configurations() const;
    void update_performance_metrics(double processing_time_ms);
    FusionResult create_empty_result() const;
    void cleanup_old_results();
};

} // namespace astraxel

#endif // SENSOR_FUSION_HPP