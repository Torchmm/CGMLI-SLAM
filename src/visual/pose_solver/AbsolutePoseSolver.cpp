#include <visual/pose_solver/AbsolutePoseSolver.hpp>
#include "PoseEstimatorAdapter.hpp"
#include <PoseLib/solvers/gp3p.h>
#include <PoseLib/solvers/p3p.h>
#include <PoseLib/robust/ransac_impl.h>

namespace cgmli {
    namespace pose_solver {
        namespace {
            class AbsolutePoseEstimator final : public detail::PoseEstimatorAdapter {
            public:
                AbsolutePoseEstimator(const AbsolutePoseCorrespondences& data, const PoseSolverOptions& options)
                    : PoseEstimatorAdapter(data.size(), 3, 3, options), data_(data),tangentThreshold_(std::tan(options.angularThreshold)) {
                }

                void generate_models(std::vector<poselib::CameraPose>* models) {
                    detail::ScopedPoseTimer timer(timing_.generationMs);
                    const auto& sample = drawMinimalSample();
                    auto& origins=origins_; auto& bearings=bearings_; auto& points=points_;
                    Eigen::Vector3d pointMean = Eigen::Vector3d::Zero(), originMean = Eigen::Vector3d::Zero();
                    for (std::size_t k = 0; k < 3; ++k) {
                        const auto& m = data_[sample[k]];
                        origins[k] = m.ray_S.origin; bearings[k] = m.ray_S.direction; points[k] = m.point_W;
                        pointMean += points[k] / 3.0; originMean += origins[k] / 3.0;
                    }
                    const Eigen::Vector3d edge1 = points[1] - points[0], edge2 = points[2] - points[0];
                    const double length1 = edge1.stableNorm(), length2 = edge2.stableNorm();
                    if (!std::isfinite(length1) || !std::isfinite(length2) || length1 < 1e-12 || length2 < 1e-12 ||
                        (edge1 / length1).cross(edge2 / length2).norm() <= options_.rankTolerance) return;
                    bool central = true;
                    for (std::size_t k = 0; k < 3; ++k) {
                        central = central && (origins[k] - originMean).norm() <= options_.originTolerance;
                        origins[k] -= originMean; points[k] -= pointMean;
                    }
                    auto& candidates=candidates_; candidates.clear();
                    // A same-camera minimal sample is central even in a noncentral rig. P3P
                    // avoids feeding a degenerate generalized polynomial system to GP3P.
                    if (central) poselib::p3p(bearings, points, &candidates);
                    else poselib::gp3p(origins, bearings, points, &candidates);
                    for (auto candidate : candidates) {
                        if (!detail::validModel(candidate)) continue;
                        candidate.t += originMean - candidate.rotate(pointMean);
                        if (!detail::validModel(candidate)) continue;
                        models->push_back(candidate); ++generatedModels;
                    }
                }

            protected:
                double angularError(const detail::Pose& pose, std::size_t index) const override {
                    const auto& m = data_[index];
                    const Eigen::Vector3d predicted = pose.C() * m.point_W + pose.r() - m.ray_S.origin;
                    if (!predicted.allFinite() || predicted.dot(m.ray_S.direction) <= options_.minRayDepth)
                        return std::numeric_limits<double>::infinity();
                    // atan2(|p x b|,p.b) is homogeneous in |p|. Reject capped
                    // MSAC outliers without normalization or atan2; retain the
                    // exact angular score for every admitted correspondence.
                    const double dot=predicted.dot(m.ray_S.direction);
                    const double sine=predicted.cross(m.ray_S.direction).stableNorm();
                    if(!std::isfinite(sine) || sine>=dot*tangentThreshold_)
                        return std::numeric_limits<double>::infinity();
                    return std::atan2(sine,dot);
                }
                bool hasAnalyticLinearization() const override { return options_.analyticAbsoluteJacobians; }
                bool linearizeMatch(const detail::Pose& pose,std::size_t index,Eigen::VectorXd& r,Eigen::MatrixXd& j) const override {
                    const auto& m=data_[index]; const Eigen::Vector3d rotated=pose.C()*m.point_W;
                    const Eigen::Vector3d predicted=rotated+pose.r()-m.ray_S.origin;
                    if(predicted.dot(m.ray_S.direction)<=options_.minRayDepth) return false;
                    Eigen::Vector3d angular; Eigen::Matrix3d D;
                    if(!detail::sphericalLinearization(predicted,m.ray_S.direction,angular,D)) return false;
                    r=angular; j.resize(3,6); j.leftCols<3>()=D;
                    // Transformation::oplus adds t directly; it does NOT rotate t.
                    // This chart differs from the backend's full left-SE3 manifold.
                    j.rightCols<3>()=-D*detail::crossMatrix(rotated); return j.allFinite();
                }
                bool residualForMatch(const detail::Pose& pose, std::size_t index, Eigen::VectorXd& residual) const override {
                    const auto& m = data_[index];
                    const Eigen::Vector3d predicted = pose.C() * m.point_W + pose.r() - m.ray_S.origin;
                    if (predicted.dot(m.ray_S.direction) <= options_.minRayDepth) return false;
                    Eigen::Vector3d angular;
                    if (!detail::sphericalResidual(predicted, m.ray_S.direction, angular)) return false;
                    residual = angular; return true;
                }
            private:
                const AbsolutePoseCorrespondences& data_;
                const double tangentThreshold_;
                std::vector<Eigen::Vector3d> origins_=std::vector<Eigen::Vector3d>(3),
                    bearings_=std::vector<Eigen::Vector3d>(3),points_=std::vector<Eigen::Vector3d>(3);
                std::vector<poselib::CameraPose> candidates_;
            };
        }

        PoseSolverResult AbsolutePoseSolver::solve(const AbsolutePoseCorrespondences& correspondences,
            const kinematics::Transformation* initialT_WS) const {
            PoseSolverResult result;
            if (!detail::validOptions(options_) || (initialT_WS && !detail::validPose(*initialT_WS))) return result;
            auto data = correspondences;
            for (auto& m : data)
                if (!m.point_W.allFinite() || !detail::normalizeRay(m.ray_S)) return result;
            if (data.size() < std::max<std::size_t>(3, options_.minInliers)) {
                result.status = SolverStatus::InsufficientCorrespondences; return result;
            }
            AbsolutePoseEstimator estimator(data, options_);
            poselib::CameraPose best;
            auto ransacOptions = options_.ransac;
            ransacOptions.score_initial_model = initialT_WS != nullptr;
            // GP3P uses p_S = R_SW * p_W + t_SW. The project API returns T_WS.
            if (initialT_WS) best = detail::toPoseLib(initialT_WS->inverse());
            const auto stats = poselib::ransac<AbsolutePoseEstimator>(estimator, ransacOptions, &best);
            result = estimator.finish(best, stats, 6, true);
            if (result.success()) result.pose = result.pose.inverse();
            return result;
        }
    }
}
