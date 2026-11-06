#pragma once

#include "PoseSolverInternal.hpp"
#include <PoseLib/robust/sampling.h>

namespace cgmli {
    namespace pose_solver {
        namespace detail {
            /// Common adapter contract for PoseLib 2.0.5's ransac_impl.h.
            /// Derived estimators only add generate_models(); there is no project RANSAC loop.
            class PoseEstimatorAdapter {
            public:
                PoseEstimatorAdapter(std::size_t count, std::size_t sampleSize, int residualSize,
                    const PoseSolverOptions& options)
                    : num_data(checkedDataSize(count, sampleSize)), sample_sz(sampleSize), options_(options), residualSize_(residualSize),
                    sampler_(num_data, sample_sz, options.ransac.seed, options.ransac.progressive_sampling,
                        static_cast<int>(std::min(options.ransac.max_prosac_iterations,
                            static_cast<std::size_t>(std::numeric_limits<int>::max())))), sample_(sample_sz) {
                }
                virtual ~PoseEstimatorAdapter() = default;

                // Required public fields used by PoseLib for sample size and stopping bounds.
                const std::size_t num_data, sample_sz;
                std::size_t generatedModels = 0;

                /// MSAC score: sum min(angle^2, threshold^2). Invalid geometry costs a full cap.
                double score_model(const poselib::CameraPose& model, std::size_t* inlierCount) const {
                    ScopedPoseTimer timer(timing_.scoringMs); ++timing_.scoringPasses;
                    *inlierCount = 0;
                    const double cap = options_.angularThreshold * options_.angularThreshold;
                    if (!validModel(model)) return cap * static_cast<double>(num_data);
                    const Pose pose = fromPoseLib(model);
                    double score = 0.0;
                    for (std::size_t i = 0; i < num_data; ++i) {
                        const double e = angularError(pose, i);
                        if (std::isfinite(e) && e < options_.angularThreshold) {
                            ++*inlierCount; score += e * e;
                        }
                        else score += cap;
                    }
                    return score;
                }

                /// Called by PoseLib's LO scheduler and its final refinement.
                void refine_model(poselib::CameraPose* model) const {
                    ScopedPoseTimer timer(timing_.refinementMs);
                    if (!validModel(*model) || options_.refinementIterations == 0) return;
                    Pose pose = fromPoseLib(*model);
                    const ResidualFunction residual = [this](const Pose& p, std::size_t i, Eigen::VectorXd& r) {
                        ++timing_.residualEvaluations;
                        return residualForMatch(p, i, r);
                        };
                    const ConstraintFunction constraint = [this](Pose& p) { return constrain(p); };
                    const auto linearize=linearization();
                    auto consensus=classify(pose);
                    double beforeScore=consensus.score;
                    for (int round = 0; round < options_.localOptimizationRounds; ++round) {
                        const auto& ids=consensus.inliers;
                        if (ids.size() < sample_sz) break;
                        Pose candidate(pose);
                        if (!refine(candidate, ids, residualSize_, options_, residual, constraint,linearize,&timing_)) break;
                        const auto candidateModel = toPoseLib(candidate);
                        auto next=classify(candidate);
                        const double afterScore=next.score;
                        if (afterScore >= beforeScore) break;
                        *model = candidateModel; pose = candidate; beforeScore = afterScore;
                        consensus=std::move(next);
                    }
                }

                Indices inlierIndices(const Pose& pose) const {
                    Indices ids; ids.reserve(num_data);
                    for (std::size_t i = 0; i < num_data; ++i)
                        if (angularError(pose, i) < options_.angularThreshold) ids.push_back(i);
                    return ids;
                }

                /// Recompute statistics after PoseLib's final LO: v2.0.5 does not update all
                /// RansacStats fields there. Never infer success from its default identity pose.
                PoseSolverResult finish(const poselib::CameraPose& model, const poselib::RansacStats& stats,
                    int requiredRank, bool scaleObservable) const {
                    PoseSolverResult result;
                    result.iterations = stats.iterations; result.refinements = stats.refinements;
                    result.generatedModels = generatedModels;
                    result.timing=timing_;
                    if (stats.model_score == std::numeric_limits<double>::max() || !validModel(model)) {
                        result.status = generatedModels == 0 ? SolverStatus::DegenerateGeometry : SolverStatus::NoConsensus;
                        return result;
                    }
                    const Pose pose = fromPoseLib(model);
                    auto consensus=classify(pose);
                    const auto& ids=consensus.inliers;
                    result.timing=timing_;
                    const std::size_t required = std::max(sample_sz, options_.minInliers);
                    if (ids.size() < required) { result.status = SolverStatus::NoConsensus; return result; }
                    const ResidualFunction residual = [this](const Pose& p, std::size_t i, Eigen::VectorXd& r) {
                        ++timing_.residualEvaluations;
                        return residualForMatch(p, i, r);
                        };
                    const ConstraintFunction constraint = [this](Pose& p) { return constrain(p); };
                    bool fullRank=false;
                    { ScopedPoseTimer timer(timing_.observabilityMs);
                      fullRank=observable(pose, ids, residualSize_, requiredRank, options_, residual, constraint,linearization()); }
                    result.timing=timing_;
                    if (!fullRank) {
                        result.status = SolverStatus::DegenerateGeometry; return result;
                    }
                    result.msacScore = consensus.score;
                    result.pose = pose;
                    result.rmsAngularError = std::sqrt(consensus.squaredInlierError / static_cast<double>(ids.size()));
                    result.inlierRatio = static_cast<double>(ids.size()) / static_cast<double>(num_data);
                    result.inliers=std::move(consensus.inliers);
                    result.translationScaleObservable = scaleObservable;
                    result.status = SolverStatus::Success;
                    result.timing=timing_;
                    return result;
                }

            protected:
                struct Consensus { Indices inliers; double score=0,squaredInlierError=0; };
                Consensus classify(const Pose& pose) const {
                    ScopedPoseTimer timer(timing_.scoringMs); ++timing_.scoringPasses;
                    Consensus result; result.inliers.reserve(num_data);
                    const double cap=options_.angularThreshold*options_.angularThreshold;
                    for(std::size_t i=0;i<num_data;++i) {
                        const double e=angularError(pose,i);
                        if(std::isfinite(e) && e<options_.angularThreshold) {
                            result.inliers.push_back(i); result.squaredInlierError+=e*e; result.score+=e*e;
                        } else result.score+=cap;
                    }
                    return result;
                }
                const Indices& drawMinimalSample() {
                    // PoseLib 2.0.5 writes sample[i] without resizing the vector.
                    // Its caller must provide SIZE sample_sz, not just capacity.
                    // Keep one allocated buffer for both uniform and PROSAC draws.
                    sampler_.generate_sample(&sample_);
                    return sample_;
                }
                virtual double angularError(const Pose& pose, std::size_t index) const = 0;
                virtual bool residualForMatch(const Pose& pose, std::size_t index, Eigen::VectorXd& residual) const = 0;
                virtual bool hasAnalyticLinearization() const { return false; }
                virtual bool linearizeMatch(const Pose&,std::size_t,Eigen::VectorXd&,Eigen::MatrixXd&) const { return false; }
                LinearizationFunction linearization() const {
                    if(!hasAnalyticLinearization()) return {};
                    return [this](const Pose& p,std::size_t i,Eigen::VectorXd& r,Eigen::MatrixXd& j) {
                        ++timing_.jacobianEvaluations; ++timing_.residualEvaluations;
                        return linearizeMatch(p,i,r,j);
                    };
                }
                virtual bool constrain(Pose&) const { return true; }
                const PoseSolverOptions& options_;
                int residualSize_;
                mutable PoseSolverTiming timing_;
            private:
                static std::size_t checkedDataSize(std::size_t count, std::size_t sampleSize) {
                    // Validate before the sampler's PROSAC constructor runs; an
                    // impossible sample can divide by zero or never terminate.
                    if (sampleSize == 0 || count < sampleSize)
                        throw std::invalid_argument("Pose estimator requires 0 < sample size <= correspondence count");
                    return count;
                }
                poselib::RandomSampler sampler_;
                Indices sample_;
            };
        }
    }
}
