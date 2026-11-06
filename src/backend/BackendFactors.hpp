#pragma once
#include "BackendTypes.hpp"

namespace cgmli { namespace backend {
// Reference priors and velocity ZUPT are supplied here for initialization/calibration/testing.
// Sensor-specific IMU, inverse-depth S^2 and point-time LiDAR factories belong in src/factor.
/// Construct r=S*boxminus(x,reference). S has tangentSize columns and may be rectangular
/// to constrain only selected directions; S^T*S is information, not covariance.
/// Tag calibration regularization as Calibration to exclude it from the default
/// calibration-information diagnostic. This is an explicit live cost, not graph.prior.
FactorRecord makeParameterPrior(const ParameterBlock& reference, const Eigen::MatrixXd& sqrtInformation,
    FactorKind kind = FactorKind::Other);
/// Construct a whitened zero-world-velocity observation on the first three SB entries.
/// The caller must first confirm stationarity. This factor does not constrain gyro/
/// accelerometer readings or their biases; static IMU residuals are separate factors.
FactorRecord makeVelocityZupt(StateId id, const Eigen::Matrix3d& sqrtInformation,
    double nisThreshold = std::numeric_limits<double>::infinity());
} }
