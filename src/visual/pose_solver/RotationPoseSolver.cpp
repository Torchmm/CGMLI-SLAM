#include <visual/pose_solver/RotationPoseSolver.hpp>
#include "PoseEstimatorAdapter.hpp"
#include <PoseLib/robust/ransac_impl.h>
#include <Eigen/LU>

namespace cgmli { namespace pose_solver {
namespace {
class RotationEstimator final : public detail::PoseEstimatorAdapter {
public:
    RotationEstimator(const RelativePoseCorrespondences& data,const PoseSolverOptions& options)
        : PoseEstimatorAdapter(data.size(),3,3,options),data_(data) {}
    void generate_models(std::vector<poselib::CameraPose>* models) {
        detail::ScopedPoseTimer timer(timing_.generationMs);
        const auto& sample=drawMinimalSample();
        Eigen::Matrix3d covariance=Eigen::Matrix3d::Zero();
        for(auto i:sample) covariance+=data_[i].ray_S1.direction*data_[i].ray_S2.direction.transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance,Eigen::ComputeFullU|Eigen::ComputeFullV);
        if(svd.singularValues()[0]<=1e-12 || svd.singularValues()[1]<=options_.rankTolerance*svd.singularValues()[0]) return;
        Eigen::Matrix3d sign=Eigen::Matrix3d::Identity();
        sign(2,2)=(svd.matrixU()*svd.matrixV().transpose()).determinant()<0?-1.0:1.0;
        const Eigen::Matrix3d rotation=svd.matrixU()*sign*svd.matrixV().transpose();
        const detail::Pose pose(Eigen::Vector3d::Zero(),Eigen::Quaterniond(rotation));
        const auto model=detail::toPoseLib(pose);
        if(detail::validModel(model)) { models->push_back(model); ++generatedModels; }
    }
protected:
    double angularError(const detail::Pose& pose,std::size_t i) const override {
        return detail::angle(pose.C()*data_[i].ray_S2.direction,data_[i].ray_S1.direction);
    }
    bool residualForMatch(const detail::Pose& pose,std::size_t i,Eigen::VectorXd& residual) const override {
        Eigen::Vector3d value;
        if(!detail::sphericalResidual(pose.C()*data_[i].ray_S2.direction,data_[i].ray_S1.direction,value)) return false;
        residual=value; return true;
    }
    bool hasAnalyticLinearization() const override { return options_.analyticRotationJacobians; }
    bool linearizeMatch(const detail::Pose& pose,std::size_t i,Eigen::VectorXd& residual,Eigen::MatrixXd& J) const override {
        const Eigen::Vector3d predicted=pose.C()*data_[i].ray_S2.direction;
        Eigen::Vector3d value; Eigen::Matrix3d D;
        if(!detail::sphericalLinearization(predicted,data_[i].ray_S1.direction,value,D)) return false;
        residual=value; J.resize(3,6); J.leftCols<3>().setZero();
        // Pose::oplus left-multiplies rotation. Translation is constrained to zero,
        // so its three columns vanish exactly; observable rank is still three.
        J.rightCols<3>()=-D*detail::crossMatrix(predicted); return J.allFinite();
    }
    bool constrain(detail::Pose& pose) const override {
        const Eigen::Quaterniond rotation=pose.q(); pose.set(Eigen::Vector3d::Zero(),rotation);
        return detail::validPose(pose);
    }
private:
    const RelativePoseCorrespondences& data_;
};
}
PoseSolverResult RotationPoseSolver::solve(const RelativePoseCorrespondences& correspondences,
    const kinematics::Transformation* initialT_S1S2) const {
    PoseSolverResult result;
    if(!detail::validOptions(options_) || (initialT_S1S2 && !detail::validPose(*initialT_S1S2))) return result;
    auto data=correspondences;
    for(auto& item:data) if(!detail::normalizeRay(item.ray_S1) || !detail::normalizeRay(item.ray_S2)) return result;
    if(data.size()<std::max<std::size_t>(3,options_.minInliers)) {
        result.status=SolverStatus::InsufficientCorrespondences; return result;
    }
    RotationEstimator estimator(data,options_); poselib::CameraPose best;
    auto ransac=options_.ransac; ransac.score_initial_model=initialT_S1S2!=nullptr;
    if(initialT_S1S2) best=detail::toPoseLib(detail::Pose(Eigen::Vector3d::Zero(),initialT_S1S2->q()));
    const auto statistics=poselib::ransac<RotationEstimator>(estimator,ransac,&best);
    return estimator.finish(best,statistics,3,false);
}
} }
