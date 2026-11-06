#pragma once
#include <cstdint>
#include <common/FrameTypedefs.hpp>
#include <visual/pose_solver/PoseSolverTypes.hpp>
#include <string>
#include <map>

namespace cgmli {
class MultiFrame;
struct EstimatorParameters;
namespace frontend_pose {
using Extrinsics=std::vector<kinematics::Transformation>;
struct Observation {
    KeypointIdentifier key;
    LandmarkId landmark;
};
struct AbsoluteInput {
    pose_solver::AbsolutePoseCorrespondences data;
    std::vector<Observation> observations; // Same order as data; never inferred from map iteration later.
};
struct RelativeObservation {
    KeypointIdentifier first,second;
    LandmarkId landmark;
};
struct RelativeInput {
    pose_solver::RelativePoseCorrespondences data;
    std::vector<RelativeObservation> observations;
};
struct Options {
    pose_solver::PoseSolverOptions solver;
    std::size_t minimumAbsoluteInliers=10,minimumRelativeInliers=11,minimumLoopLandmarks=8;
    double absoluteInlierRatio=0.7,relativeInlierRatio=0.8,rotationDominanceRatio=0.8;
    double loopPixelSigma=1.0; // Original-image pixels at octave zero, not ORB patch diameter.
    // Only the minimum sampling floor changes. PoseLib confidence stopping, the
    // full maximum budget and all frontend acceptance gates remain in force.
    bool adaptiveAbsoluteBudget=true;
    std::size_t trackingMinIterations=20,fullRansacPeriod=30,minimumBudgetCorrespondences=60;
    double budgetPredictionInlierRatio=0.9,budgetPredictionRmsFraction=0.5;
    bool valid() const;
};
/// Exact half-ray/angular checks, independent of a solver result. Used only to
/// choose a sampling budget, never as a visual acceptance certificate.
bool absolutePredictionHealthy(const pose_solver::AbsolutePoseCorrespondences&,
    const kinematics::Transformation& T_WB,const Options&);
/// Extrinsics map each camera into the solver reference B. The solver's S labels
/// are generic reference-frame labels: feeding T_BC makes its outputs T_WB/T_B1B2.
bool makeRay(const MultiFrame& frame,const KeypointIdentifier& key,const Extrinsics& extrinsics,
    pose_solver::GeneralizedRay& ray);
AbsoluteInput absoluteInput(const MultiFrame& frame,const MapPoints& landmarks,const Extrinsics& extrinsics);
/// Initialized finite points already evaluated by this proposal; preserve observation order.
AbsoluteInput absoluteInput(const MultiFrame& frame,const std::map<LandmarkId,Eigen::Vector3d>& points,const Extrinsics& extrinsics);
/// One independent pair per landmark; balance available camera pairs across the
/// rig, including cross-camera tracks, rather than running separate central solves.
RelativeInput relativeInput(const MultiFrame& first,const MultiFrame& second,
    const Extrinsics& firstExtrinsics,const Extrinsics& secondExtrinsics);
/// Compose T_WB2 = T_WB1*T_B1B2 only for a metric full-pose result. Otherwise keep
/// the predicted world position and use the estimated rotation; unit central-camera
/// baseline never becomes a fabricated one-metre body displacement.
bool relativeWorldSeed(const kinematics::Transformation& firstPose,
    const kinematics::Transformation& predicted,const pose_solver::PoseSolverResult& result,
    bool rotationOnly,kinematics::Transformation& seed);
/// Compare sensor-origin poses against an actual IMU prediction. Bounds have
/// physical units and grow with elapsed time; they do not claim a covariance test.
bool inertialPoseConsistent(const kinematics::Transformation& prediction,
    const kinematics::Transformation& candidate,double elapsedSeconds,
    const EstimatorParameters&,std::string* reason=nullptr);
/// Conditional fixed-landmark information in LEFT [rho,theta] tangent of T_reference_body.
/// Pixel covariance is propagated through each camera's back-projection Jacobian.
/// Failure does not overwrite information; no damping is fabricated to fill null space.
bool absoluteInformation(const MultiFrame& frame,const Extrinsics& extrinsics,const AbsoluteInput& input,
    const pose_solver::PoseSolverResult& result,double pixelSigma,double pyramidScale,
    Eigen::Matrix<double,6,6>& information);
} }
