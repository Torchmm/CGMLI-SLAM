#include "ProjectBackendAdapter.hpp"
#include <cameras/NCameraSystem.hpp>
#include <cameras/CameraBase.hpp>
#include <visual/MultiFrame.hpp>
#include <cmath>
#include <stdexcept>

namespace cgmli { namespace backend {
std::vector<ParameterBlock> RigCalibration::parameterBlocks(double tdCamera, double tdLidar) const {
    // Only shared rig/LiDAR extrinsics and scalar clock offsets belong in the graph.
    // Sensor factor factories capture T_RC by value and combine it with the live T_BR.
    if (T_RC.empty() || !std::isfinite(tdCamera) || !std::isfinite(tdLidar)) throw std::invalid_argument("invalid rig/time calibration");
    std::vector<ParameterBlock> blocks;
    blocks.push_back(ParameterBlock::pose({BlockKind::CameraExtrinsic,0},T_BR));
    blocks.push_back(ParameterBlock::euclidean({BlockKind::CameraTimeOffset,0},Eigen::VectorXd::Constant(1,tdCamera)));
    blocks.push_back(ParameterBlock::pose({BlockKind::LidarExtrinsic,0},T_BL));
    blocks.push_back(ParameterBlock::euclidean({BlockKind::LidarTimeOffset,0},Eigen::VectorXd::Constant(1,tdLidar)));
    for (auto& b:blocks) b.constant=true;
    return blocks;
}
RigCalibration makeRigCalibration(const cameras::NCameraSystem& cameras,
    const kinematics::Transformation& T_SL, const kinematics::Transformation& T_BS) {
    if (!cameras.numCameras() || !cameras.T_SC(0)) throw std::invalid_argument("camera rig is empty/missing reference extrinsic");
    RigCalibration out;
    const auto T_SC0=*cameras.T_SC(0);
    // Choose camera 0 as rig R. Factoring all T_BCi through this common transform
    // preserves calibrated inter-camera baselines during online extrinsic optimization.
    out.T_BR=T_BS*T_SC0; out.T_BL=T_BS*T_SL;
    for (size_t i=0;i<cameras.numCameras();++i) {
        if (!cameras.T_SC(i) || !cameras.cameraGeometry(i)) throw std::invalid_argument("missing camera calibration");
        out.T_RC.push_back(T_SC0.inverse()*(*cameras.T_SC(i)));
    }
    return out;
}
StateEstimate makeEpoch(const MultiFrame& frame, const kinematics::Transformation& T_WB,
    const SpeedAndBias& sb, SensorHealth visual, SensorHealth lidar, bool keyframe) {
    // Use physical frame sampling time, not packet arrival. Time offset corrections
    // belong inside sensor factors and must not retime an already-created graph node.
    StateEstimate out;
    out.id=StateId(frame.id()); out.timestamp=frame.timestamp(); out.T_WB=T_WB; out.speedBias=sb;
    out.visual=visual; out.lidar=lidar; out.keyframe=keyframe; return out;
}
bool pixelToBearing(const cameras::CameraBase& camera, const Eigen::Vector2d& p, Eigen::Vector3d& out) {
    // Keep camera-specific projection and masking at the boundary. Downstream S^2
    // factors receive one model-independent unit direction, including fisheye rear rays.
    if (!p.allFinite() || p.x()<0 || p.y()<0 || p.x()>=camera.imageWidth() || p.y()>=camera.imageHeight()) return false;
    Eigen::Vector3d b;
    if (camera.hasMask() && camera.mask().at<unsigned char>(static_cast<int>(p.y()),static_cast<int>(p.x()))==0) return false;
    if (!camera.backProject(p,&b) || !b.allFinite()) return false;
    const double n=b.stableNorm();
    if (!std::isfinite(n) || n<1e-12) return false;
    out=b/n; return true;
}
} }
