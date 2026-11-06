#pragma once

#include <visual/pose_solver/PoseSolverTypes.hpp>
#include <PoseLib/camera_pose.h>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <chrono>

namespace cgmli {
    namespace pose_solver {
        namespace detail {
            using Pose = kinematics::Transformation;
            using Indices = std::vector<std::size_t>;
            using Vector6 = Eigen::Matrix<double, 6, 1>;
            using Matrix6 = Eigen::Matrix<double, 6, 6>;
            using ResidualFunction = std::function<bool(const Pose&, std::size_t, Eigen::VectorXd&)>;
            using ConstraintFunction = std::function<bool(Pose&)>;
            using LinearizationFunction = std::function<bool(const Pose&, std::size_t, Eigen::VectorXd&, Eigen::MatrixXd&)>;
            struct ScopedPoseTimer {
                double& total;
                std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
                explicit ScopedPoseTimer(double& value):total(value) {}
                ~ScopedPoseTimer() { total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); }
            };
            inline Eigen::Matrix3d crossMatrix(const Eigen::Vector3d& p) {
                Eigen::Matrix3d s; s<<0,-p.z(),p.y(),p.z(),0,-p.x(),-p.y(),p.x(),0; return s;
            }
            /// Derivative of the SAME S2 log used below, including normalization.
            /// The antipode has no unique logarithm and remains invalid.
            inline bool sphericalLinearization(const Eigen::Vector3d& predicted,const Eigen::Vector3d& observed,
                Eigen::Vector3d& residual,Eigen::Matrix3d& derivative) {
                const double length=predicted.stableNorm();
                if(!predicted.allFinite() || !std::isfinite(length) || length<1e-12) return false;
                const Eigen::Vector3d n=predicted/length;
                const double c=std::clamp(observed.dot(n),-1.0,1.0);
                const Eigen::Vector3d u=n-c*observed; const double s=u.norm();
                double f=1,g=0;
                if(s<1e-12) { if(c<0) return false; }
                else {
                    const double theta=std::atan2(s,c),d=1-c;
                    f=theta/s;
                    g=c>0 && s<1e-4?(-1.0/3.0-4*d/15.0-6*d*d/35.0):(theta*c-s)/(s*s*s);
                }
                residual=f*u;
                const Eigen::Matrix3d projected=f*(Eigen::Matrix3d::Identity()-observed*observed.transpose())+g*u*observed.transpose();
                derivative=(projected-(projected*n)*n.transpose())/length;
                return residual.allFinite() && derivative.allFinite();
            }

            inline bool normalizeRay(GeneralizedRay& ray) {
                if (!ray.origin.allFinite() || !ray.direction.allFinite()) return false;
                const double norm = ray.direction.stableNorm();
                if (!std::isfinite(norm) || norm < 1e-12) return false;
                ray.direction /= norm;
                return true;
            }
            inline bool validPose(const Pose& pose) {
                return pose.T().allFinite() && std::abs(pose.q().squaredNorm() - 1.0) < 1e-6;
            }
            inline bool validModel(const poselib::CameraPose& model) {
                return model.q.allFinite() && model.t.allFinite() &&
                    std::abs(model.q.squaredNorm() - 1.0) < 1e-6;
            }
            inline Pose fromPoseLib(const poselib::CameraPose& model) {
                // PoseLib uses [w,x,y,z]; Eigen's coefficients are [x,y,z,w].
                return Pose(model.t, Eigen::Quaterniond(model.q[0], model.q[1], model.q[2], model.q[3]));
            }
            inline poselib::CameraPose toPoseLib(const Pose& pose) {
                Eigen::Vector4d q;
                q << pose.q().w(), pose.q().x(), pose.q().y(), pose.q().z();
                return poselib::CameraPose(q, pose.r());
            }
            inline bool validOptions(const PoseSolverOptions& o) {
                const auto& r = o.ransac;
                return std::isfinite(o.angularThreshold) && o.angularThreshold > 0 && o.angularThreshold < 0.5 &&
                    r.max_iterations > 0 && r.min_iterations <= r.max_iterations &&
                    std::isfinite(r.success_prob) && r.success_prob > 0 && r.success_prob < 1 &&
                    std::isfinite(r.dyn_num_trials_mult) && r.dyn_num_trials_mult > 0 &&
                    (!r.progressive_sampling || (r.max_prosac_iterations > 0 &&
                        r.max_prosac_iterations <= static_cast<std::size_t>(std::numeric_limits<int>::max()))) &&
                    std::isfinite(o.refinementRelativeCostTolerance) && o.refinementRelativeCostTolerance>=0 &&
                    o.refinementIterations >= 0 && o.localOptimizationRounds >= 0 &&
                    std::isfinite(o.rankTolerance) && o.rankTolerance > 0 && o.rankTolerance < 1 &&
                    std::isfinite(o.originTolerance) && o.originTolerance >= 0 &&
                    std::isfinite(o.minRayDepth) && o.minRayDepth >= 0 &&
                    std::isfinite(o.minParallax) && o.minParallax > 0 && o.minParallax < 0.5;
            }
            inline double angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return std::atan2(a.cross(b).stableNorm(), a.dot(b));
            }
            /// Spherical log at the observed bearing; its norm is exactly the angle in radians.
            inline bool sphericalResidual(const Eigen::Vector3d& predicted, const Eigen::Vector3d& observed,
                Eigen::Vector3d& residual) {
                const double norm = predicted.stableNorm();
                if (!predicted.allFinite() || !std::isfinite(norm) || norm < 1e-12) return false;
                const Eigen::Vector3d unit = predicted / norm;
                const double cosine = std::clamp(observed.dot(unit), -1.0, 1.0);
                const Eigen::Vector3d tangent = unit - cosine * observed;
                const double sine = tangent.norm();
                if (sine < 1e-12) {
                    if (cosine < 0) return false; // Undefined log at the antipodal bearing.
                    residual = tangent;
                }
                else residual = tangent * (std::atan2(sine, cosine) / sine);
                return residual.allFinite();
            }
            inline bool evaluate(const Pose& pose, const Indices& ids, int rowsPerMatch,
                const ResidualFunction& residual, Eigen::VectorXd& values) {
                values.resize(static_cast<Eigen::Index>(ids.size()) * rowsPerMatch);
                Eigen::VectorXd matchResidual;
                for (std::size_t k = 0; k < ids.size(); ++k) {
                    if (!residual(pose, ids[k], matchResidual) || matchResidual.size() != rowsPerMatch ||
                        !matchResidual.allFinite()) return false;
                    values.segment(static_cast<Eigen::Index>(k) * rowsPerMatch, rowsPerMatch) = matchResidual;
                }
                return values.allFinite();
            }
            inline bool numericalJacobian(const Pose& pose, const Indices& ids, int rowsPerMatch,
                const ResidualFunction& residual, const ConstraintFunction& constrain,
                const Eigen::VectorXd& values, Eigen::MatrixXd& jacobian) {
                jacobian.resize(values.size(), 6);
                for (int k = 0; k < 6; ++k) {
                    const double h = k < 3 ? 1e-6 * std::max(1.0, pose.r().stableNorm()) : 1e-6;
                    Vector6 delta = Vector6::Zero(); delta[k] = h;
                    Pose plus(pose), minus(pose);
                    plus.oplus(delta); minus.oplus(-delta);
                    Eigen::VectorXd vp, vm;
                    const bool plusValid = constrain(plus) && evaluate(plus, ids, rowsPerMatch, residual, vp);
                    const bool minusValid = constrain(minus) && evaluate(minus, ids, rowsPerMatch, residual, vm);
                    if (plusValid && minusValid) jacobian.col(k) = (vp - vm) / (2.0 * h);
                    else if (plusValid) jacobian.col(k) = (vp - values) / h;
                    else if (minusValid) jacobian.col(k) = (values - vm) / h;
                    else return false;
                }
                return jacobian.allFinite();
            }
            inline double huberCost(const Eigen::VectorXd& values, int rowsPerMatch, double cutoff) {
                double cost = 0;
                for (Eigen::Index k = 0; k < values.size(); k += rowsPerMatch) {
                    const double norm = values.segment(k, rowsPerMatch).norm();
                    cost += norm <= cutoff ? 0.5 * norm * norm : cutoff * (norm - 0.5 * cutoff);
                }
                return cost;
            }
            /// Multi-inlier Huber LM in the project's [translation,left-rotation] tangent.
            /// The outer LO schedule and stopping policy remain exclusively in PoseLib.
            inline bool refine(Pose& pose, const Indices& ids, int rowsPerMatch, const PoseSolverOptions& o,
                const ResidualFunction& residual, const ConstraintFunction& constrain,
                const LinearizationFunction& linearize = {},PoseSolverTiming* timing=nullptr) {
                if (ids.empty() || !constrain(pose)) return false;
                Eigen::VectorXd values;
                if (!evaluate(pose, ids, rowsPerMatch, residual, values)) return false;
                double damping = 1e-3;
                const double cutoff = o.angularThreshold;
                double cost = huberCost(values, rowsPerMatch, cutoff);
                Matrix6 normal; Vector6 gradient; bool linearizationValid=false;
                Eigen::VectorXd matchResidual; Eigen::MatrixXd matchJacobian;
                for (int iteration = 0; iteration < o.refinementIterations; ++iteration) {
                    if(timing) ++timing->loIterations;
                    // A rejected damping trial leaves the linearization point and
                    // robust weights unchanged. Reuse H/g until a step is accepted.
                    if(!linearizationValid) {
                        normal.setZero(); gradient.setZero();
                        if(linearize) {
                            // Accumulate fixed-size normal equations without a 3N x 6
                            // Jacobian or twelve perturbed inlier-set evaluations.
                            for(auto id:ids) {
                                auto& r=matchResidual; auto& j=matchJacobian;
                                if(!linearize(pose,id,r,j) || r.size()!=rowsPerMatch || j.rows()!=rowsPerMatch || j.cols()!=6
                                    || !r.allFinite() || !j.allFinite()) return false;
                                const double w=std::min(1.0,cutoff/std::max(r.norm(),1e-15));
                                normal.noalias()+=w*j.transpose()*j; gradient.noalias()+=w*j.transpose()*r;
                            }
                        } else {
                            Eigen::MatrixXd j;
                            if(!numericalJacobian(pose,ids,rowsPerMatch,residual,constrain,values,j)) return false;
                            Eigen::VectorXd weighted=values;
                            for(Eigen::Index k=0;k<values.size();k+=rowsPerMatch) {
                                const double norm=values.segment(k,rowsPerMatch).norm();
                                const double weight=std::sqrt(std::min(1.0,cutoff/std::max(norm,1e-15)));
                                j.middleRows(k,rowsPerMatch)*=weight; weighted.segment(k,rowsPerMatch)*=weight;
                            }
                            normal.noalias()=j.transpose()*j; gradient.noalias()=j.transpose()*weighted;
                        }
                        linearizationValid=true;
                    }
                    if (gradient.lpNorm<Eigen::Infinity>() < 1e-12) break;
                    Matrix6 damped = normal;
                    damped.diagonal().array() += damping * (normal.diagonal().array() + 1e-12);
                    const Vector6 delta = damped.ldlt().solve(-gradient);
                    if (!delta.allFinite()) return false;
                    // A vanishing damped step cannot improve representable cost;
                    // terminate before spending another complete inlier traversal.
                    if(delta.norm()<1e-9) { if(timing) ++timing->smallStepStops; break; }
                    Pose candidate(pose); candidate.oplus(delta);
                    Eigen::VectorXd trial;
                    const bool valid = validPose(candidate) && constrain(candidate) &&
                        evaluate(candidate, ids, rowsPerMatch, residual, trial);
                    const double trialCost = valid ? huberCost(trial, rowsPerMatch, cutoff)
                        : std::numeric_limits<double>::infinity();
                    if (trialCost < cost) {
                        if(timing) ++timing->acceptedLoSteps;
                        const double improvement=cost-trialCost,previousCost=cost;
                        pose = candidate; values = std::move(trial); cost = trialCost; linearizationValid=false;
                        damping = std::max(1e-12, damping * 0.3);
                        if(o.refinementRelativeCostTolerance>0 && improvement<=o.refinementRelativeCostTolerance*std::max(previousCost,1e-12)) {
                            if(timing) ++timing->relativeCostStops; break;
                        }
                    }
                    else {
                        if(timing) ++timing->rejectedLoSteps;
                        damping *= 10.0;
                        if (damping > 1e12) break;
                    }
                }
                return validPose(pose);
            }
            inline bool observable(const Pose& pose, const Indices& ids, int rowsPerMatch, int rank,
                const PoseSolverOptions& o, const ResidualFunction& residual,
                const ConstraintFunction& constrain,const LinearizationFunction& linearize = {}) {
                Eigen::VectorXd values;
                Eigen::MatrixXd j;
                if(linearize) {
                    j.resize(static_cast<Eigen::Index>(ids.size())*rowsPerMatch,6);
                    Eigen::VectorXd r; Eigen::MatrixXd block;
                    for(std::size_t i=0;i<ids.size();++i) {
                        if(!linearize(pose,ids[i],r,block) || block.rows()!=rowsPerMatch || block.cols()!=6) return false;
                        j.middleRows(static_cast<Eigen::Index>(i)*rowsPerMatch,rowsPerMatch)=block;
                    }
                } else if (!evaluate(pose, ids, rowsPerMatch, residual, values) ||
                    !numericalJacobian(pose, ids, rowsPerMatch, residual, constrain, values, j)) return false;
                if(!j.allFinite() || j.rows()<rank) return false;
                // Keep metre/radian units from arbitrarily changing the rank decision. Tiny
                // columns are not magnified: this preserves a truly unobservable direction.
                const double largest = j.colwise().norm().maxCoeff();
                if (!std::isfinite(largest) || largest < 1e-15) return false;
                for (int k = 0; k < 6; ++k) {
                    const double norm = j.col(k).norm();
                    if (norm > largest * o.rankTolerance) j.col(k) /= norm;
                    else j.col(k).setZero();
                }
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(j);
                return svd.singularValues().size() >= rank &&
                    svd.singularValues()[rank - 1] > o.rankTolerance * svd.singularValues()[0];
            }
        }
    }
}
