// sensor_fusion.cpp - Implementation of Unified Sensor Fusion Engine for AstraVoxel

#include "sensor_fusion.hpp"
#include <algorithm>
#include <numeric>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace astraxel {

//==============================================================================
// Math Utilities Implementation
//==============================================================================

namespace astraxel_math {

Matrix3d Matrix3d::rotation_x(double angle_rad) {
    Matrix3d mat;
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    mat.m[0] = 1.0; mat.m[1] = 0.0; mat.m[2] = 0.0;
    mat.m[3] = 0.0; mat.m[4] = c;   mat.m[5] = -s;
    mat.m[6] = 0.0; mat.m[7] = s;   mat.m[8] = c;
    return mat;
}

Matrix3d Matrix3d::rotation_y(double angle_rad) {
    Matrix3d mat;
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    mat.m[0] = c;   mat.m[1] = 0.0; mat.m[2] = s;
    mat.m[3] = 0.0; mat.m[4] = 1.0; mat.m[5] = 0.0;
    mat.m[6] = -s;  mat.m[7] = 0.0; mat.m[8] = c;
    return mat;
}

Matrix3d Matrix3d::rotation_z(double angle_rad) {
    Matrix3d mat;
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    mat.m[0] = c;   mat.m[1] = -s;  mat.m[2] = 0.0;
    mat.m[3] = s;   mat.m[4] = c;   mat.m[5] = 0.0;
    mat.m[6] = 0.0; mat.m[7] = 0.0; mat.m[8] = 1.0;
    return mat;
}

Matrix3d Matrix3d::operator*(const Matrix3d& other) const {
    Matrix3d result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result.m[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; ++k) {
                result.m[i * 3 + j] += m[i * 3 + k] * other.m[k * 3 + j];
            }
        }
    }
    return result;
}

Quaternion Quaternion::from_euler(double yaw, double pitch, double roll) {
    // Convert Euler angles to quaternion
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
    return q;
}

Matrix3d Quaternion::to_rotation_matrix() const {
    Matrix3d mat;
    double xx = x * x, xy = x * y, xz = x * z, xw = x * w;
    double yy = y * y, yz = y * z, yw = y * w;
    double zz = z * z, zw = z * w;

    mat.m[0] = 1.0 - 2.0 * (yy + zz);
    mat.m[1] = 2.0 * (xy - zw);
    mat.m[2] = 2.0 * (xz + yw);

    mat.m[3] = 2.0 * (xy + zw);
    mat.m[4] = 1.0 - 2.0 * (xx + zz);
    mat.m[5] = 2.0 * (yz - xw);

    mat.m[6] = 2.0 * (xz - yw);
    mat.m[7] = 2.0 * (yz + xw);
    mat.m[8] = 1.0 - 2.0 * (xx + yy);

    return mat;
}

Quaternion Quaternion::normalized() const {
    double norm = std::sqrt(w*w + x*x + y*y + z*z);
    if (norm < 1e-12) return {1.0, 0.0, 0.0, 0.0};
    return {w/norm, x/norm, y/norm, z/norm};
}

} // namespace astraxel_math

//==============================================================================
// SensorCalibration Implementation
//==============================================================================

nlohmann::json SensorCalibration::to_json() const {
    nlohmann::json j;
    j["sensor_id"] = sensor_id;
    j["sensor_type"] = sensor_type;
    j["coordinate_system"] = coordinate_system;
    j["position"] = {position.x, position.y, position.z};
    j["orientation"] = {orientation.w, orientation.x, orientation.y, orientation.z};
    j["intrinsic_matrix"] = {{
        intrinsic_matrix.m[0], intrinsic_matrix.m[1], intrinsic_matrix.m[2],
        intrinsic_matrix.m[3], intrinsic_matrix.m[4], intrinsic_matrix.m[5],
        intrinsic_matrix.m[6], intrinsic_matrix.m[7], intrinsic_matrix.m[8]
    }};
    j["field_of_view_rad"] = field_of_view_rad;
    j["min_range"] = min_range;
    j["max_range"] = max_range;
    j["calibration_uncertainty"] = calibration_uncertainty;
    j["angular_uncertainty"] = angular_uncertainty;
    j["is_calibrated"] = is_calibrated;
    return j;
}

SensorCalibration SensorCalibration::from_json(const nlohmann::json& j) {
    SensorCalibration cal;
    cal.sensor_id = j.value("sensor_id", 0);
    cal.sensor_type = j.value("sensor_type", "unknown");
    cal.coordinate_system = j.value("coordinate_system", "cartesian");

    auto pos = j.value("position", std::vector<double>{0,0,0});
    if (pos.size() >= 3) {
        cal.position = {pos[0], pos[1], pos[2]};
    }

    auto ori = j.value("orientation", std::vector<double>{1,0,0,0});
    if (ori.size() >= 4) {
        cal.orientation = {ori[0], ori[1], ori[2], ori[3]};
    }

    auto intrinsic = j.value("intrinsic_matrix", std::vector<std::vector<double>>{
        {1,0,0}, {0,1,0}, {0,0,1}
    });
    if (intrinsic.size() >= 3 && intrinsic[0].size() >= 3) {
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k) {
                cal.intrinsic_matrix.m[i*3 + k] = intrinsic[i][k];
            }
        }
    }

    cal.field_of_view_rad = j.value("field_of_view_rad", 1.0);
    cal.min_range = j.value("min_range", 0.0);
    cal.max_range = j.value("max_range", 1000.0);
    cal.calibration_uncertainty = j.value("calibration_uncertainty", 0.1);
    cal.angular_uncertainty = j.value("angular_uncertainty", 0.01);
    cal.is_calibrated = j.value("is_calibrated", false);

    return cal;
}

//==============================================================================
// VoxelGrid Implementation
//==============================================================================

VoxelGrid::VoxelGrid(const astraxel_math::Vector3d& origin,
                    const astraxel_math::Vector3d& resolution,
                    const Vector3i& dimensions)
    : origin_(origin), resolution_(resolution), dimensions_(dimensions) {
    initialize_grid();
}

void VoxelGrid::initialize_grid() {
    size_t total_size = static_cast<size_t>(dimensions_.x) *
                       static_cast<size_t>(dimensions_.y) *
                       static_cast<size_t>(dimensions_.z);
    evidence_grid_.resize(total_size, 0.0f);
    intensity_grid_.resize(total_size, 0.0f);
    observation_counts_.resize(total_size, 0);
}

bool VoxelGrid::accumulate_point(const astraxel_math::Vector3d& point,
                                float evidence, float intensity) {
    // Convert world point to voxel index
    size_t index = point_to_voxel_index(point);
    if (index >= evidence_grid_.size()) {
        return false;
    }

    // Accumulate evidence and intensity
    evidence_grid_[index] += evidence;
    intensity_grid_[index] += intensity;
    observation_counts_[index]++;

    return true;
}

bool VoxelGrid::get_voxel_data(int x, int y, int z, float& evidence,
                              float& intensity, int& count) const {
    if (!is_valid_voxel(x, y, z)) {
        return false;
    }

    size_t index = static_cast<size_t>(x) +
                   static_cast<size_t>(y) * dimensions_.x +
                   static_cast<size_t>(z) * dimensions_.x * dimensions_.y;

    evidence = evidence_grid_[index];
    intensity = intensity_grid_[index];
    count = observation_counts_[index];

    return true;
}

bool VoxelGrid::reset() {
    std::fill(evidence_grid_.begin(), evidence_grid_.end(), 0.0f);
    std::fill(intensity_grid_.begin(), intensity_grid_.end(), 0.0f);
    std::fill(observation_counts_.begin(), observation_counts_.end(), 0);
    return true;
}

size_t VoxelGrid::point_to_voxel_index(const astraxel_math::Vector3d& point) const {
    // Convert world coordinates to voxel indices
    astraxel_math::Vector3d relative_point = point - origin_;

    int x_idx = static_cast<int>(relative_point.x / resolution_.x);
    int y_idx = static_cast<int>(relative_point.y / resolution_.y);
    int z_idx = static_cast<int>(relative_point.z / resolution_.z);

    // Check bounds
    if (x_idx < 0 || x_idx >= dimensions_.x ||
        y_idx < 0 || y_idx >= dimensions_.y ||
        z_idx < 0 || z_idx >= dimensions_.z) {
        return evidence_grid_.size(); // Return invalid index
    }

    return static_cast<size_t>(x_idx) +
           static_cast<size_t>(y_idx) * dimensions_.x +
           static_cast<size_t>(z_idx) * dimensions_.x * dimensions_.y;
}

bool VoxelGrid::is_valid_voxel(int x, int y, int z) const {
    return x >= 0 && x < dimensions_.x &&
           y >= 0 && y < dimensions_.y &&
           z >= 0 && z < dimensions_.z;
}

//==============================================================================
// BayesianFusion Implementation
//==============================================================================

FusionResult BayesianFusion::fuse_measurements(
    const std::vector<SensorMeasurement>& measurements,
    const std::vector<SensorCalibration>& calibrations) {

    FusionResult result;
    if (measurements.empty()) {
        return result;
    }

    result.timestamp_ns = measurements[0].timestamp_ns;
    result.sensor_ids.reserve(measurements.size());
    for (const auto& measurement : measurements) {
        result.sensor_ids.push_back(measurement.sensor_id);
    }

    // Simple evidence accumulation for now
    // In a full implementation, this would use proper Bayesian inference
    std::vector<astraxel_math::Vector3d> fused_points;
    std::vector<float> weights;

    for (size_t i = 0; i < measurements.size(); ++i) {
        const auto& measurement = measurements[i];
        const auto& calibration = calibrations[i];

        // For simplicity, assume each measurement has a weight based on confidence
        float weight = measurement.confidence_level;

        // Mock point cloud generation (would be sensor-specific in real implementation)
        astraxel_math::Vector3d mock_point = {
            static_cast<double>(measurement.sensor_id * 10.0),
            static_cast<double>(i * 5.0),
            static_cast<double>(measurement.timestamp_ns % 1000)
        };

        fused_points.push_back(mock_point);
        weights.push_back(weight);
    }

    // Compute weighted centroid
    astraxel_math::Vector3d centroid = {0, 0, 0};
    float total_weight = 0.0f;

    for (size_t i = 0; i < fused_points.size(); ++i) {
        centroid = centroid + (fused_points[i] * weights[i]);
        total_weight += weights[i];
    }

    if (total_weight > 0.0f) {
        centroid = centroid * (1.0f / total_weight);
        result.object_detected = true;
        result.centroid = centroid;
        result.object_position = centroid;
        result.point_cloud = fused_points;
        result.fusion_confidence = std::min(1.0f, total_weight / measurements.size());
    }

    return result;
}

void BayesianFusion::update_parameters(double performance_score) {
    // Adaptive parameter adjustment based on performance
    if (performance_score > 0.8) {
        prior_weight_ = std::min(0.9, prior_weight_ + 0.01);
    } else if (performance_score < 0.6) {
        prior_weight_ = std::max(0.1, prior_weight_ - 0.01);
    }
}

nlohmann::json BayesianFusion::get_parameters() const {
    return {
        {"fusion_type", "bayesian"},
        {"prior_weight", prior_weight_},
        {"likelihood_threshold", likelihood_threshold_},
        {"correlation_enabled", use_correlation_}
    };
}

//==============================================================================
// EvidenceAccumulationFusion Implementation
//==============================================================================

FusionResult EvidenceAccumulationFusion::fuse_measurements(
    const std::vector<SensorMeasurement>& measurements,
    const std::vector<SensorCalibration>& calibrations) {

    FusionResult result;
    if (measurements.empty()) {
        return result;
    }

    result.timestamp_ns = measurements[0].timestamp_ns;
    result.sensor_ids.reserve(measurements.size());
    for (const auto& measurement : measurements) {
        result.sensor_ids.push_back(measurement.sensor_id);
    }

    // Evidence accumulation with decay
    std::map<astraxel_math::Vector3d, float> evidence_map;

    for (size_t i = 0; i < measurements.size(); ++i) {
        const auto& measurement = measurements[i];

        // Use sensor confidence for evidence weight
        float evidence_weight = measurement.confidence_level * evidence_threshold_;

        // Mock spatial points (would be extracted from sensor data)
        astraxel_math::Vector3d evidence_point = {
            static_cast<double>(measurement.sensor_id * 5.0),
            static_cast<double>(i * 2.0),
            static_cast<double>(measurement.timestamp_ns % 500)
        };

        // Apply decay to existing evidence
        for (auto& [point, evidence] : evidence_map) {
            evidence *= decay_factor_;
        }

        // Add new evidence
        evidence_map[evidence_point] += evidence_weight;
    }

    // Extract strongest evidence points
    std::vector<astraxel_math::Vector3d> strong_points;
    float max_evidence = 0.0f;

    for (const auto& [point, evidence] : evidence_map) {
        if (evidence > evidence_threshold_) {
            strong_points.push_back(point);
            result.evidences.push_back(evidence);
            max_evidence = std::max(max_evidence, evidence);
        }
    }

    result.point_cloud = strong_points;
    result.fusion_confidence = normalize_evidence_ ? max_evidence : max_evidence;

    if (!strong_points.empty()) {
        // Compute centroid of strong evidence points
        result.centroid = {0, 0, 0};
        for (const auto& point : strong_points) {
            result.centroid = result.centroid + point;
        }
        result.centroid = result.centroid * (1.0 / strong_points.size());
        result.object_detected = true;
        result.object_position = result.centroid;
    }

    return result;
}

void EvidenceAccumulationFusion::update_parameters(double performance_score) {
    // Adjust accumulation parameters based on performance
    if (performance_score > 0.7) {
        evidence_threshold_ = std::max(0.05f, evidence_threshold_ - 0.01f);
    } else {
        evidence_threshold_ = std::min(0.5f, evidence_threshold_ + 0.01f);
    }
}

nlohmann::json EvidenceAccumulationFusion::get_parameters() const {
    return {
        {"fusion_type", "evidence_accumulation"},
        {"evidence_threshold", evidence_threshold_},
        {"decay_factor", decay_factor_},
        {"normalize_evidence", normalize_evidence_}
    };
}

//==============================================================================
// QualityAssurance Implementation
//==============================================================================

QualityAssurance::QualityMetrics QualityAssurance::assess_fusion_quality(
    const FusionResult& result,
    const std::vector<SensorMeasurement>& measurements,
    const std::vector<SensorCalibration>& calibrations) {

    QualityMetrics metrics;

    // Data completeness based on coverage
    metrics.data_completeness = std::min(1.0,
        static_cast<double>(measurements.size()) / 4.0); // Assume 4 sensors ideal

    // Temporal consistency (variation in timestamps)
    if (measurements.size() > 1) {
        std::vector<uint64_t> timestamps;
        for (const auto& m : measurements) timestamps.push_back(m.timestamp_ns);

        uint64_t mean_ts = std::accumulate(timestamps.begin(), timestamps.end(), uint64_t(0)) / timestamps.size();
        double variance = 0.0;
        for (uint64_t ts : timestamps) {
            double diff = static_cast<double>(ts) - mean_ts;
            variance += diff * diff;
        }
        variance /= timestamps.size();
        metrics.temporal_consistency = std::exp(-variance / 1e12); // Time consistency metric
    } else {
        metrics.temporal_consistency = 0.5;
    }

    // Spatial consistency (agreement between sensors)
    if (measurements.size() >= 2) {
        double avg_confidence = 0.0;
        for (const auto& m : measurements) avg_confidence += m.confidence_level;
        avg_confidence /= measurements.size();
        metrics.spatial_consistency = avg_confidence;
    } else {
        metrics.spatial_consistency = 0.5;
    }

    // Sensor diversity
    std::set<int> sensor_types;
    for (const auto& cal : calibrations) {
        if (cal.is_calibrated) sensor_types.insert(cal.sensor_id);
    }
    metrics.sensor_diversity = std::min(1.0, static_cast<double>(sensor_types.size()) / 3.0);

    // Overall confidence score
    metrics.confidence_score = (metrics.data_completeness * 0.3 +
                               metrics.temporal_consistency * 0.2 +
                               metrics.spatial_consistency * 0.3 +
                               metrics.sensor_diversity * 0.2);

    return metrics;
}

std::vector<std::string> QualityAssurance::generate_quality_report(
    const QualityMetrics& metrics,
    const std::vector<SensorMeasurement>& measurements) {

    std::vector<std::string> report;

    char buffer[256];

    sprintf(buffer, "Fusion Quality Report:");
    report.push_back(buffer);

    sprintf(buffer, "  Overall Confidence: %.2f", metrics.confidence_score);
    report.push_back(buffer);

    sprintf(buffer, "  Data Completeness: %.2f", metrics.data_completeness);
    report.push_back(buffer);

    sprintf(buffer, "  Temporal Consistency: %.2f", metrics.temporal_consistency);
    report.push_back(buffer);

    sprintf(buffer, "  Spatial Consistency: %.2f", metrics.spatial_consistency);
    report.push_back(buffer);

    sprintf(buffer, "  Sensor Diversity: %.2f", metrics.sensor_diversity);
    report.push_back(buffer);

    sprintf(buffer, "  Measurements Processed: %zu", measurements.size());
    report.push_back(buffer);

    // Generate recommendations
    if (metrics.confidence_score < 0.6) {
        report.push_back("  WARNING: Low confidence - consider sensor calibration");
    }
    if (metrics.data_completeness < 0.8) {
        report.push_back("  RECOMMENDATION: Add more sensor coverage");
    }
    if (metrics.temporal_consistency < 0.7) {
        report.push_back("  ISSUE: High timestamp variation detected");
    }

    return report;
}

//==============================================================================
// AstraVoxelFusionEngine Implementation
//==============================================================================

AstraVoxelFusionEngine::AstraVoxelFusionEngine()
    : fusion_strategy_(std::make_unique<BayesianFusion>()),
      voxel_grid_(nullptr),
      initialized_(false) {
}

AstraVoxelFusionEngine::~AstraVoxelFusionEngine() {
    stop_fusion();
}

bool AstraVoxelFusionEngine::initialize(const nlohmann::json& config) {
    if (config.contains("fusion_rate_hz")) {
        fusion_rate_hz_ = config.value("fusion_rate_hz", 10.0);
    }

    if (config.contains("real_time_mode")) {
        real_time_mode_ = config.value("real_time_mode", true);
    }

    // Create voxel grid with default parameters
    astraxel_math::Vector3d grid_origin(-100.0, -100.0, -100.0);
    astraxel_math::Vector3d grid_resolution(1.0, 1.0, 1.0);
    Vector3i grid_dimensions{200, 200, 200};

    if (config.contains("voxel_grid")) {
        auto& grid_config = config["voxel_grid"];
        if (grid_config.contains("origin")) {
            auto origin = grid_config["origin"];
            grid_origin = {origin[0], origin[1], origin[2]};
        }
        if (grid_config.contains("resolution")) {
            auto res = grid_config["resolution"];
            grid_resolution = {res[0], res[1], res[2]};
        }
        if (grid_config.contains("dimensions")) {
            auto dims = grid_config["dimensions"];
            grid_dimensions = {dims[0], dims[1], dims[2]};
        }
    }

    voxel_grid_ = std::make_unique<VoxelGrid>(grid_origin, grid_resolution, grid_dimensions);
    initialized_ = true;

    return true;
}

int AstraVoxelFusionEngine::add_sensor(std::unique_ptr<SensorInterface> sensor) {
    if (!sensor) return -1;

    int sensor_id = static_cast<int>(sensors_.size());
    sensor->get_calibration().sensor_id = sensor_id;
    sensors_.push_back(std::move(sensor));

    return sensor_id;
}

bool AstraVoxelFusionEngine::remove_sensor(int sensor_id) {
    if (sensor_id < 0 || sensor_id >= static_cast<int>(sensors_.size())) {
        return false;
    }

    sensors_.erase(sensors_.begin() + sensor_id);
    return true;
}

bool AstraVoxelFusionEngine::start_fusion() {
    if (!initialized_ || sensors_.empty()) {
        return false;
    }

    // Implementation would start background fusion thread
    // For now, just mark as initialized
    return true;
}

void AstraVoxelFusionEngine::stop_fusion() {
    // Implementation would stop background fusion thread
    // For now, just cleanup
    if (voxel_grid_) {
        voxel_grid_->reset();
    }
}

FusionResult AstraVoxelFusionEngine::fuse_data() {
    if (!initialized_ || sensors_.empty()) {
        return create_empty_result();
    }

    std::vector<SensorMeasurement> measurements;
    std::vector<SensorCalibration> calibrations;

    // Collect measurements from all sensors
    for (auto& sensor : sensors_) {
        if (sensor->is_connected()) {
            auto measurement = sensor->capture_measurement();
            if (measurement.valid) {
                measurements.push_back(measurement);
                calibrations.push_back(sensor->get_calibration());
            }
        }
    }

    if (measurements.empty()) {
        return create_empty_result();
    }

    // Perform fusion
    FusionResult result = fusion_strategy_->fuse_measurements(measurements, calibrations);

    // Update performance tracking
    successful_fusions_++;
    result.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Store recent results
    recent_results_.push_back(result);
    if (recent_results_.size() > 100) { // Keep last 100 results
        recent_results_.erase(recent_results_.begin());
    }

    return result;
}

nlohmann::json AstraVoxelFusionEngine::get_system_status() const {
    return {
        {"initialized", initialized_},
        {"sensor_count", sensors_.size()},
        {"fusion_rate_hz", fusion_rate_hz_},
        {"real_time_mode", real_time_mode_},
        {"successful_fusions", successful_fusions_},
        {"failed_fusions", failed_fusions_},
        {"voxel_grid_size", voxel_grid_ ? voxel_grid_->total_voxels() : 0}
    };
}

bool AstraVoxelFusionEngine::export_results(const std::string& format) {
    // Mock implementation - would export actual results
    return true;
}

void AstraVoxelFusionEngine::configure_fusion_parameters(const nlohmann::json& params) {
    if (params.contains("strategy")) {
        std::string strategy = params["strategy"];
        if (strategy == "evidence_accumulation") {
            fusion_strategy_ = std::make_unique<EvidenceAccumulationFusion>();
        } else {
            fusion_strategy_ = std::make_unique<BayesianFusion>();
        }
    }

    fusion_strategy_->update_parameters(params.value("performance_feedback", 0.8));
}

bool AstraVoxelFusionEngine::validate_sensor_configurations() const {
    for (const auto& sensor : sensors_) {
        const auto& cal = sensor->get_calibration();
        if (!cal.is_calibrated) {
            return false;
        }
        // Add more validation as needed
    }
    return true;
}

void AstraVoxelFusionEngine::update_performance_metrics(double processing_time_ms) {
    average_processing_time_ms_ = 0.9 * average_processing_time_ms_ + 0.1 * processing_time_ms;
}

FusionResult AstraVoxelFusionEngine::create_empty_result() const {
    FusionResult result;
    result.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    result.object_detected = false;
    result.fusion_confidence = 0.0;
    return result;
}

void AstraVoxelFusionEngine::cleanup_old_results() {
    auto now = std::chrono::high_resolution_clock::now();
    auto cutoff = now - std::chrono::seconds(300); // Keep last 5 minutes

    // Remove old results
    recent_results_.erase(
        std::remove_if(recent_results_.begin(), recent_results_.end(),
            [cutoff](const FusionResult& result) {
                return result.timestamp_ns < cutoff.time_since_epoch().count();
            }),
        recent_results_.end());
}

} // namespace astraxel