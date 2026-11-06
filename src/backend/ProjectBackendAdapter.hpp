#pragma once
#include "BackendTypes.hpp"

namespace cgmli {
class MultiFrame;
namespace cameras { class NCameraSystem; class CameraBase; }
namespace backend {

/// Camera-0-referenced rigid rig plus one LiDAR-to-body transform. This value is an
/// initialization/geometry description; online calibration values subsequently live
/// in the backend graph/snapshots, not in this original copy.
struct RigCalibration {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    kinematics::Transformation T_BR, T_BL;
    // Fixed factory baselines. Online optimization uses ONE shared T_BR, never four independent T_BC.
    std::vector<kinematics::Transformation> T_RC;
    /// Build four global, initially frozen blocks. Offsets are seconds with the convention
    /// t_IMU=t_sensor+td; fixed per-camera baselines are deliberately not optimized blocks.
    std::vector<ParameterBlock> parameterBlocks(double tdCamera = 0.0, double tdLidar = 0.0) const;
};

// Existing project S is the IMU sensor. Default B=S; supply T_BS explicitly if body differs.
// T_AB maps coordinates from B to A. Thus T_BR=T_BS*T_SC0 and
// T_RCi=T_SC0^-1*T_SCi. Supports any nonempty calibrated camera count/model.
RigCalibration makeRigCalibration(const cameras::NCameraSystem& cameras,
    const kinematics::Transformation& T_SL,
    const kinematics::Transformation& T_BS = kinematics::Transformation());

// Prediction is T_WB and [v_W,bg,ba], NOT a PoseLib hypothesis residual.
// Copies frame identity/time and the supplied prediction. This adapter does not rotate
// velocities/biases between frames or create visual residuals from the image data.
StateEstimate makeEpoch(const MultiFrame& frame, const kinematics::Transformation& T_WB,
    const SpeedAndBias& speedBias, SensorHealth visual, SensorHealth lidar, bool keyframe);

// Frontend adapter only: backend visual factors use the resulting unit S^2 bearing, never x/z.
// Rejects invalid pixels, masked pixels, failed back-projection, and zero/nonfinite rays.
// Output is assigned only on success; negative camera-z directions remain valid bearings.
bool pixelToBearing(const cameras::CameraBase& camera, const Eigen::Vector2d& pixel, Eigen::Vector3d& bearing);
} }
