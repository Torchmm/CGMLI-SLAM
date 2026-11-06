#include <visual/pose_solver/RelativePoseSolver.hpp>
#include "PoseEstimatorAdapter.hpp"
#include <PoseLib/solvers/gen_relpose_6pt.h>
#include <PoseLib/solvers/relpose_5pt.h>
#include <PoseLib/robust/ransac_impl.h>

namespace cgmli {
    namespace pose_solver {
        namespace {
            bool singleCentrePair(const RelativePoseCorrespondences& data, double tolerance) {
                if (data.empty()) return false;
                for (const auto& m : data)
                    if ((m.ray_S1.origin - data.front().ray_S1.origin).norm() > tolerance ||
                        (m.ray_S2.origin - data.front().ray_S2.origin).norm() > tolerance) return false;
                return true;
            }
            struct RayIntersection {
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                    Eigen::Vector3d predicted1, predicted2;
            };
            class RelativePoseEstimator final : public detail::PoseEstimatorAdapter {
            public:
                RelativePoseEstimator(const RelativePoseCorrespondences& data, const PoseSolverOptions& options, bool central)
                    : PoseEstimatorAdapter(data.size(), central ? 5 : 6, 6, options), data_(data), central_(central),
                    centre1_(data.front().ray_S1.origin), centre2_(data.front().ray_S2.origin) {
                }

                void generate_models(std::vector<poselib::CameraPose>* models) {
                    detail::ScopedPoseTimer timer(timing_.generationMs);
                    const auto& sample = drawMinimalSample();
                    std::vector<Eigen::Vector3d> origins1(sample_sz), origins2(sample_sz),
                        bearings1(sample_sz), bearings2(sample_sz);
                    bool sampleCentral = true;
                    for (std::size_t k = 0; k < sample_sz; ++k) {
                        const auto& m = data_[sample[k]];
                        origins1[k] = m.ray_S1.origin; origins2[k] = m.ray_S2.origin;
                        bearings1[k] = m.ray_S1.direction; bearings2[k] = m.ray_S2.direction;
                        sampleCentral = sampleCentral && (origins1[k] - origins1[0]).norm() <= options_.originTolerance &&
                            (origins2[k] - origins2[0]).norm() <= options_.originTolerance;
                    }
                    std::vector<poselib::CameraPose> candidates;
                    if (central_) {
                        poselib::relpose_5pt(bearings2, bearings1, &candidates);
                    }
                    else {
                        // A single camera pair cannot determine metric scale. Skip this minimal
                        // sample; PoseLib still counts the attempt in its own stopping schedule.
                        if (sampleCentral) return;
                        // PoseLib solves input 1 -> input 2. Swap the arrays to obtain T_S1S2.
                        poselib::gen_relpose_6pt(origins2, bearings2, origins1, bearings1, &candidates);
                    }
                    for (auto candidate : candidates) {
                        if (!detail::validModel(candidate)) continue;
                        if (central_) {
                            const double norm = candidate.t.stableNorm();
                            if (!std::isfinite(norm) || norm < 1e-12) continue;
                            const Eigen::Vector3d baseline = candidate.t / norm;
                            candidate.t = centre1_ - candidate.rotate(centre2_) + baseline;
                            if (!detail::validModel(candidate)) continue;
                        }
                        const auto pose = detail::fromPoseLib(candidate);
                        bool cheiral = true;
                        for (auto i : sample) {
                            RayIntersection intersection;
                            if (!intersect(pose, i, intersection)) { cheiral = false; break; }
                        }
                        if (!cheiral) continue;
                        models->push_back(candidate); ++generatedModels;
                    }
                }

                bool normalizeCentralPose(detail::Pose& pose) const { return constrain(pose); }

            protected:
                bool intersect(const detail::Pose& pose, std::size_t index, RayIntersection& output) const {
                    const auto& m = data_[index];
                    const Eigen::Vector3d a = m.ray_S1.direction, b = pose.C() * m.ray_S2.direction;
                    const Eigen::Vector3d baseline = pose.C() * m.ray_S2.origin + pose.r() - m.ray_S1.origin;
                    if (!baseline.allFinite()) return false;
                    const double cosine = std::clamp(a.dot(b), -1.0, 1.0);
                    const double denominator = a.cross(b).squaredNorm();
                    if (!std::isfinite(denominator) || denominator < std::pow(std::sin(options_.minParallax), 2)) return false;
                    const double depth1 = (a.dot(baseline) - cosine * b.dot(baseline)) / denominator;
                    const double depth2 = (cosine * a.dot(baseline) - b.dot(baseline)) / denominator;
                    if (!std::isfinite(depth1) || !std::isfinite(depth2) ||
                        depth1 <= options_.minRayDepth || depth2 <= options_.minRayDepth) return false;
                    const Eigen::Vector3d midpoint = 0.5 * (depth1 * a + baseline + depth2 * b);
                    output.predicted1 = midpoint;
                    output.predicted2 = pose.C().transpose() * (midpoint - baseline);
                    return output.predicted1.allFinite() && output.predicted2.allFinite() &&
                        output.predicted1.stableNorm() > 1e-12 && output.predicted2.stableNorm() > 1e-12;
                }
                double angularError(const detail::Pose& pose, std::size_t index) const override {
                    RayIntersection intersection;
                    if (!intersect(pose, index, intersection)) return std::numeric_limits<double>::infinity();
                    const auto& m = data_[index];
                    // Both observations must satisfy the threshold, including wide-angle rays.
                    return std::max(detail::angle(intersection.predicted1.normalized(), m.ray_S1.direction),
                        detail::angle(intersection.predicted2.normalized(), m.ray_S2.direction));
                }
                bool residualForMatch(const detail::Pose& pose, std::size_t index, Eigen::VectorXd& residual) const override {
                    RayIntersection intersection;
                    if (!intersect(pose, index, intersection)) return false;
                    Eigen::Vector3d r1, r2;
                    if (!detail::sphericalResidual(intersection.predicted1, data_[index].ray_S1.direction, r1) ||
                        !detail::sphericalResidual(intersection.predicted2, data_[index].ray_S2.direction, r2)) return false;
                    residual.resize(6); residual.head<3>() = r1; residual.tail<3>() = r2;
                    return true;
                }
                bool constrain(detail::Pose& pose) const override {
                    if (!central_) return true;
                    const Eigen::Vector3d baseline = pose.r() + pose.C() * centre2_ - centre1_;
                    const double norm = baseline.stableNorm();
                    if (!baseline.allFinite() || !std::isfinite(norm) || norm < 1e-12) return false;
                    const Eigen::Vector3d translation = centre1_ - pose.C() * centre2_ + baseline / norm;
                    const Eigen::Quaterniond rotation(pose.q());
                    pose.set(translation, rotation); return true;
                }
            private:
                const RelativePoseCorrespondences& data_;
                bool central_;
                Eigen::Vector3d centre1_, centre2_;
            };
        }

        PoseSolverResult RelativePoseSolver::solve(const RelativePoseCorrespondences& correspondences,
            const kinematics::Transformation* initialT_S1S2) const {
            PoseSolverResult result;
            if (!detail::validOptions(options_) || (initialT_S1S2 && !detail::validPose(*initialT_S1S2))) return result;
            auto data = correspondences;
            for (auto& m : data)
                if (!detail::normalizeRay(m.ray_S1) || !detail::normalizeRay(m.ray_S2)) return result;
            const bool central = singleCentrePair(data, options_.originTolerance);
            const std::size_t sampleSize = central ? 5 : 6;
            if (data.size() < std::max(sampleSize, options_.minInliers)) {
                result.status = SolverStatus::InsufficientCorrespondences; return result;
            }
            RelativePoseEstimator estimator(data, options_, central);
            poselib::CameraPose best;
            auto ransacOptions = options_.ransac;
            ransacOptions.score_initial_model = initialT_S1S2 != nullptr;
            if (initialT_S1S2) {
                detail::Pose prior(*initialT_S1S2);
                if (!estimator.normalizeCentralPose(prior)) {
                    // A zero-baseline IMU prior cannot seed central translation, but RANSAC
                    // can still find a valid five-point hypothesis without that prior.
                    ransacOptions.score_initial_model = false;
                }
                else best = detail::toPoseLib(prior);
            }
            const auto stats = poselib::ransac<RelativePoseEstimator>(estimator, ransacOptions, &best);
            return estimator.finish(best, stats, central ? 5 : 6, !central);
        }
    }
}
