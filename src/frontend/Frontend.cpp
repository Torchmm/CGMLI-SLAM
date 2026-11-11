
#include "utils/assert_macros.hpp"
#include <opencv2/imgcodecs.hpp>
#include <limits>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <opencv2/imgproc/imgproc.hpp>
#include <glog/logging.h>

// DBoW2
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <DBoW2/DBoW2.h>
#pragma GCC diagnostic pop

#include <frontend/Frontend.hpp>
#include <visual/pose_solver/AbsolutePoseSolver.hpp>
#include <visual/pose_solver/RelativePoseSolver.hpp>
#include <visual/pose_solver/RotationPoseSolver.hpp>
#include <factor/BodyImuFactor.hpp>

/// \brief cgmli Main namespace of this package.
namespace cgmli {



    using OrbFeatures = features::OrbFrontendFeatures;
    static constexpr int kOrbDescriptorBytes = OrbFeatures::kDescriptorBytes;
    static constexpr int kOrbDescriptorBits = OrbFeatures::kDescriptorBits;

    // FORB uses cv::Mat rows, not the old FBrisk vector<uchar> representation.
    // An unloaded vocabulary is represented by a null dBow_; feature tracking does not
    // require a place-recognition vocabulary and must never query an unconfigured DB.
    class Frontend::DBoW {
    public:
        OrbDatabase database;
        std::vector<uint64_t> poseIds;
        std::string descriptorProfile;
    };

    namespace {
        const char* solverStatusName(pose_solver::SolverStatus status) {
            switch(status) {
            case pose_solver::SolverStatus::Success: return "success";
            case pose_solver::SolverStatus::InvalidInput: return "invalid_input";
            case pose_solver::SolverStatus::InsufficientCorrespondences: return "insufficient_correspondences";
            case pose_solver::SolverStatus::NoConsensus: return "no_consensus";
            case pose_solver::SolverStatus::DegenerateGeometry: return "degenerate_geometry";
            default: return "unknown";
            }
        }
        frontend_pose::Extrinsics poseExtrinsics(const Estimator& estimator, const MultiFrame& frame) {
            frontend_pose::Extrinsics result; result.reserve(frame.numFrames());
            for (size_t camera = 0; camera < frame.numFrames(); ++camera)
                result.push_back(estimator.extrinsics(StateId(frame.id()), camera));
            return result; // Live T_BC, not the original factory T_SC or an assumed body=IMU convention.
        }
        bool inlierMask(const pose_solver::PoseSolverResult& result, size_t count, std::vector<bool>& mask) {
            mask.assign(count, false);
            for (auto index : result.inliers) { if (index >= count || mask[index]) return false; mask[index] = true; }
            return true;
        }
    }
    void Frontend::setPoseSolverOptions(const frontend_pose::Options& options) {
        if (!options.valid()) throw std::invalid_argument("Invalid frontend pose-solver acceptance/noise options");
        poseOptions_ = options;
    }
    Frontend::Frontend(size_t numCameras, const features::OrbExtractorOptions& options)
        : orbFeatures_(numCameras, options), isInitialized_(false), numCameras_(numCameras) {
        trackingLost_ = false;
    }

    void Frontend::setOrbOptions(const features::OrbExtractorOptions& options) {
        orbFeatures_.setOptions(options);
        // Configuration is only legal before extraction or after clear(). A vocabulary
        // belongs to a sampling profile, so discard any previous optional DB as well.
        dBow_.reset();
    }
    void Frontend::prepareOrb(const cameras::NCameraSystem& rig,bool parallel) {
        const MultiFrame calibration(rig,Time(0,0));
        orbFeatures_.prepare(calibration,parallel);
    }
    void Frontend::setOrbMatchingThreshold(int threshold) {
        if (threshold<0 || threshold>kOrbDescriptorBits)
            throw std::invalid_argument("ORB Hamming threshold must be in [0,256]");
        orbMatchingThreshold_ = threshold;
    }
    bool Frontend::loadOrbVocabulary(const std::string& filename, const std::string& expectedProfile,
        std::string& error) {
        try {
            cv::FileStorage file(filename, cv::FileStorage::READ);
            if (!file.isOpened() || expectedProfile.empty()
                || static_cast<std::string>(file["descriptor_profile"]) != expectedProfile) {
                error = "ORB vocabulary requires matching descriptor_profile metadata"; return false;
            }
            const auto vocabularyNode = file["vocabulary"];
            if (vocabularyNode.empty() || vocabularyNode["nodes"].empty() || vocabularyNode["words"].empty()) {
                error = "ORB vocabulary has no nodes/words"; return false;
            }
            // FORB::fromString assumes 32 values. Validate before loading; accepting a BRISK
            // vocabulary merely because its filename says ORB would leave incompatible data.
            for (const auto& node : vocabularyNode["nodes"]) {
                std::istringstream row(static_cast<std::string>(node["descriptor"]));
                int value = 0, count = 0;
                while (row >> value) { if (value < 0 || value>255) { error = "Invalid ORB vocabulary byte"; return false; } ++count; }
                if (count != kOrbDescriptorBytes || !row.eof()) { error = "Vocabulary nodes must contain exactly 32 bytes"; return false; }
            }
            OrbVocabulary vocabulary; vocabulary.load(file);
            if (vocabulary.empty()) { error = "Empty ORB vocabulary"; return false; }
            auto candidate = std::make_unique<DBoW>();
            candidate->database.setVocabulary(vocabulary, false, 0);
            candidate->descriptorProfile = expectedProfile;
            dBow_ = std::move(candidate); return true;
        }
        catch (const std::exception& exception) { error = exception.what(); return false; }
    }
    Frontend::~Frontend() = default;

    bool Frontend::loadComponent(std::string filename,
        const ImuParameters& imuParameters,
        const cameras::NCameraSystem& nCameraSystem,
        bool componentFixed)
    {
        // Legacy full-graph import has not been migrated to the managed window backend.
        // In particular, imported BRISK map descriptors cannot enter this ORB frontend.
        return false;
    }

    // Per-camera locking, LUT binding and gravity updates live in the feature-only
    // component. No call touches another camera's mutable extractor during extraction.
    bool Frontend::detectAndDescribe(size_t cameraIndex, std::shared_ptr<MultiFrame> frameOut,
        const kinematics::Transformation& T_WC, const std::vector<cv::KeyPoint>* keypoints) {
        if (!frameOut) throw std::invalid_argument("Null frontend frame");
        return orbFeatures_.detectAndDescribe(cameraIndex, *frameOut, T_WC, keypoints);
    }


    bool Frontend::verifyRecognisedPlace(const Estimator& estimator, const ViParameters& params,
        const std::shared_ptr<const MultiFrame> framesInOut, const std::shared_ptr<const MultiFrame> oldFrame,
        kinematics::Transformation& T_Sold_Snew, Eigen::Matrix<double, 6, 6>& H, int minInliers) {
        if (!framesInOut || !oldFrame || minInliers < 3 || !poseOptions_.valid()) return false;
        try {
            orbFeatures_.validateFrame(*framesInOut); orbFeatures_.validateFrame(*oldFrame);
            // The old frame exports points in W. Transform them explicitly to old BODY
            // coordinates so AbsolutePoseSolver returns T_Bold_Bnew directly. The historic
            // argument name T_Sold_Snew is kept for source compatibility; B=S by default.
            const auto T_old_W = estimator.pose(StateId(oldFrame->id())).inverse();
            const auto extrinsics = poseExtrinsics(estimator, *framesInOut);
            std::map<LandmarkId, Eigen::Vector3d> points;
            std::map<LandmarkId, std::vector<const uchar*>> descriptors;
            for (size_t camera = 0; camera < oldFrame->numFrames(); ++camera)
                for (size_t keypoint = 0; keypoint < oldFrame->numKeypoints(camera); ++keypoint) {
                    const LandmarkId id(oldFrame->landmarkId(camera, keypoint)); if (!id.isInitialised()) continue;
                    Eigen::Vector4d homogeneous; bool initialized = false;
                    if (!oldFrame->getLandmark(camera, keypoint, homogeneous, initialized) || !initialized
                        || !homogeneous.allFinite() || std::abs(homogeneous[3]) < 1e-12) continue;
                    (void)params;
                    const Eigen::Vector3d world = homogeneous.head<3>() / homogeneous[3];
                    const Eigen::Vector3d point = T_old_W * world; if (!point.allFinite()) continue;
                    points.emplace(id, point);
                    descriptors[id].push_back(oldFrame->descriptors(camera).ptr<uchar>(static_cast<int>(keypoint)));
                }
            if (points.size() < poseOptions_.minimumLoopLandmarks) return false;
            struct Candidate { KeypointIdentifier key; LandmarkId landmark; int distance; };
            std::vector<Candidate> candidates;
            for (const auto& point : points) for (size_t camera = 0; camera < framesInOut->numFrames(); ++camera) {
                int best = orbMatchingThreshold_; size_t chosen = 0;
                for (const auto* descriptor : descriptors.at(point.first))
                    for (size_t keypoint = 0; keypoint < framesInOut->numKeypoints(camera); ++keypoint) {
                        const int distance = OrbFeatures::hammingDistance(
                            framesInOut->descriptors(camera).ptr<uchar>(static_cast<int>(keypoint)), descriptor);
                        if (distance < best) { best = distance; chosen = keypoint; }
                    }
                if (best < orbMatchingThreshold_) candidates.push_back({ {framesInOut->id(),camera,chosen},point.first,best });
            }
            // One current keypoint cannot vote for several different 3D points. Preserve
            // deterministic descriptor-distance ordering when resolving those conflicts.
            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return std::tie(a.distance, a.key, a.landmark) < std::tie(b.distance, b.key, b.landmark);
                });
            frontend_pose::AbsoluteInput input; std::set<KeypointIdentifier> used;
            for (const auto& candidate : candidates) {
                if (used.count(candidate.key)) continue;
                pose_solver::AbsolutePoseCorrespondence match; match.point_W = points.at(candidate.landmark);
                if (!frontend_pose::makeRay(*framesInOut, candidate.key, extrinsics, match.ray_S)) continue;
                input.data.push_back(match); input.observations.push_back({ candidate.key,candidate.landmark }); used.insert(candidate.key);
            }
            auto options = poseOptions_.solver;
            options.minInliers = std::max(options.minInliers, static_cast<size_t>(minInliers));
            // This initial hypothesis is scored by PoseLib, not imposed as an IMU constraint.
            const auto initial = T_old_W * estimator.pose(StateId(framesInOut->id()));
            const auto result = pose_solver::AbsolutePoseSolver(options).solve(input.data, &initial);
            if (!result.success() || result.inliers.size() < static_cast<size_t>(minInliers)
                || result.inlierRatio < poseOptions_.absoluteInlierRatio) return false;
            std::set<LandmarkId> uniquePoints;
            Eigen::Matrix<float, Eigen::Dynamic, kOrbDescriptorBits> bits(result.inliers.size(), kOrbDescriptorBits);
            size_t row = 0;
            for (auto index : result.inliers) {
                if (index >= input.observations.size()) return false;
                const auto& observation = input.observations[index]; uniquePoints.insert(observation.landmark);
                const auto* descriptor = framesInOut->descriptors(observation.key.cameraIndex).ptr<uchar>(static_cast<int>(observation.key.keypointIndex));
                for (int bit = 0; bit < kOrbDescriptorBits; ++bit) bits(static_cast<Eigen::Index>(row), bit) = (descriptor[bit / 8] >> (bit % 8)) & 1;
                ++row;
            }
            if (uniquePoints.size() < poseOptions_.minimumLoopLandmarks || row < 2) return false;
            const auto variance = ((bits.rowwise() - bits.colwise().mean()).colwise().squaredNorm() / static_cast<float>(row - 1)).eval();
            const float diversity = variance.cwiseSqrt().mean();
            if (diversity < (182.0f / 384.0f) && row < 20) return false;
            Eigen::Matrix<double, 6, 6> information;
            if (!frontend_pose::absoluteInformation(*framesInOut, extrinsics, input, result, poseOptions_.loopPixelSigma,
                orbOptions().scaleFactor, information)) return false;
            // PoseLib's adapter already performed all-inlier spherical LM. No second solver
            // with old pixel-only camera templates or incompatible manifolds is constructed.
            T_Sold_Snew = result.pose; H = information;
            return true;
        }
        catch (const std::exception& error) {
            LOG(WARNING) << "Loop pose verification failed: " << error.what(); return false;
        }
    }
    int Frontend::getFilteredDBoWResult(const std::unique_ptr<DBoW>& dBow,
        const std::vector<cv::Mat>& features, std::vector<std::pair<StateId, double>>& stateIds) const {
        stateIds.clear();
        if (!dBow || features.empty()) return 0;
        DBoW2::QueryResults result; dBow->database.query(features, result, -1);
        // Query results need not contain every database entry. Entry IDs are identities,
        // never vector positions in the (sparse, score-sorted) result array.
        std::set<unsigned int> suppressed;
        for (const auto& candidate : result) {
            if (candidate.Id >= dBow->poseIds.size() || suppressed.count(candidate.Id)) continue;
            stateIds.emplace_back(StateId(dBow->poseIds[candidate.Id]), candidate.Score);
            for (const auto& other : result)
                if (std::abs(static_cast<long long>(other.Id) - candidate.Id) <= 1) suppressed.insert(other.Id);
        }
        return static_cast<int>(stateIds.size());
    }
    void Frontend::setAssociationOptions(const frontend_association::Options& options) {
        options.validate(); associationOptions_ = options;
        orbMatchingThreshold_ = options.matcher.maxHammingDistance;
    }

    bool Frontend::dataAssociationAndInitialization(Estimator& estimator, const ViParameters& params,
        std::shared_ptr<MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) {
        lastReport_ = FrontendAssociationReport();
        if (!framesInOut || !asKeyframe) { lastReport_.message = "Null association input"; return false; }
        const auto started=std::chrono::steady_clock::now();
        try {
            orbFeatures_.validateFrame(*framesInOut);
            const StateId id(framesInOut->id());
            if (framesInOut->numFrames() != numCameras_ || params.nCameraSystem.numCameras() != numCameras_
                || estimator.currentStateId() != id || lastAssociatedState_ == id)
                throw std::invalid_argument("Association requires one new, current rig frame");
            const auto frozen = estimator.window().frames().measurements(id);
            if (!frozen || !frontend_association::sameMeasurements(*framesInOut, *frozen))
                throw std::invalid_argument("Extract before addStates; caller features differ from frozen backend measurements");
            if (!estimator.window().landmarks().inFrame(id).empty())
                throw std::invalid_argument("Current frame already has managed associations");
            const auto published = estimator.snapshot();
            if (published) for (const auto& state : published->states) if (state.id == id)
                throw std::invalid_argument("Frontend association must precede optimize/publication");

            if (!framesInOut->numKeypoints()) throw std::invalid_argument("No usable ORB keypoints");
            ++absoluteBudgetEpochs_; // Count input epochs, not numeric-ID spacing or solver retries.
            auto options = associationOptions_;
            if(params.frontend.max_promotions_per_frame<0 || params.frontend.max_active_initialized<0 || params.frontend.max_tracking_seeds<0)
                throw std::invalid_argument("Negative landmark promotion budget");
            options.maxPromotionsPerFrame=size_t(params.frontend.max_promotions_per_frame);
            options.maxActiveInitialized=size_t(params.frontend.max_active_initialized);
            options.maxTrackingSeeds=size_t(params.frontend.max_tracking_seeds);
            options.seedMaxUntrackedSeconds=params.frontend.seed_max_untracked_seconds;
            options.requireKeyframeAnchor=params.frontend.require_keyframe_anchor_for_promotion;
            options.estimateCameraTimeOffset=params.camera.online_calibration.estimate_td_camera;
            options.matcher.maxHammingDistance = orbMatchingThreshold_;
            options.collectDebugMatches=params.output.display_matches;
            options.matchingThreads=size_t(std::max(1,params.frontend.num_matching_threads));
            options.useCudaMatcher=params.frontend.orb_cuda_matcher;
            options.pyramidScale = orbOptions().scaleFactor;
            options.stagedMatching=options.stagedMatching && params.frontend.orb_staged_matching;
            options.replenishSeedGaps=options.replenishSeedGaps && isInitialized_ && !trackingLost_;
            if(isInitialized_ && !trackingLost_ && !kfPrior && options.stagedMatching) {
                const auto prediction=estimator.pose(id);
                for(auto it=estimator.window().order().rbegin();it!=estimator.window().order().rend();++it) {
                    if(*it==id || !estimator.isKeyframe(*it)) continue;
                    const auto reference=estimator.pose(*it);
                    options.deferHealthyStereo=(prediction.r()-reference.r()).norm()<0.5*params.frontend.keyframe_translation &&
                        prediction.q().angularDistance(reference.q())<0.5*params.frontend.keyframe_rotation;
                    break;
                }
            }
            auto plan = frontend_association::propose(estimator, options,framesInOut.get());
            const auto poseStarted=std::chrono::steady_clock::now();
            const auto milliseconds=[](const auto& start) { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); };
            // Validate older payloads against this frontend's actual sampling profile, not
            // merely the camera enum. Backend views are const and never edited in place.
            for (const auto& item : plan.sources) orbFeatures_.validateFrame(*item.second);
            frontend_association::reject(plan, {});
            bool inertialMotion=false;
            if(params.imu.use) for(const auto& item:estimator.window().graph().factors) {
                const auto& factor=item.second;
                if(factor.kind==backend::FactorKind::Imu && factor.participates()
                    && std::find(factor.epochs.begin(),factor.epochs.end(),id)!=factor.epochs.end()) inertialMotion=true;
            }
            const auto predicted = estimator.pose(id);
            const double elapsed = estimator.numFrames()>1
                ?(plan.current->timestamp()-estimator.window().estimate(estimator.stateIdByAge(1))->timestamp).toSec():0;
            const auto consistent = [&](const PoseDecision& value) {
                if (!inertialMotion) return true;
                std::string reason;
                // Sensor-origin translation removes the fixed B/S lever-arm ambiguity.
                const bool ok=frontend_pose::inertialPoseConsistent(predicted*params.imu.T_BS,
                    value.T_WB*params.imu.T_BS,elapsed,params.estimator,&reason);
                if(!ok) LOG(WARNING)<<reason;
                return ok;
            };
            PoseDecision decision; decision.T_WB = predicted;
            const auto solveProposal=[&]() {
                decision=PoseDecision{}; decision.T_WB=predicted;
                if(runRansac3d2d(estimator,plan,decision)) {
                    if(consistent(decision)) return true;
                    if(decision.reducedBudget && runRansac3d2d(estimator,plan,decision,true) && consistent(decision)) return true;
                }
                decision=PoseDecision{}; decision.T_WB=predicted;
                return runRansac2d2d(estimator,plan,decision) && consistent(decision);
            };
            double retryPreparationMs=0;
            const auto retryProposal=[&](const auto& retryOptions) {
                const auto retryStarted=std::chrono::steady_clock::now();
                auto retry=frontend_association::propose(estimator,retryOptions,framesInOut.get());
                for(const auto& item:retry.sources) orbFeatures_.validateFrame(*item.second);
                retryPreparationMs+=milliseconds(retryStarted);
                return retry;
            };
            const auto addAttemptTiming=[&](const frontend_association::Plan& previous) {
                plan.preparationMs+=previous.preparationMs; plan.matchingMs+=previous.matchingMs;
                plan.assignmentMs+=previous.assignmentMs; plan.matchingJobs+=previous.matchingJobs;
                plan.cudaMatchingJobs+=previous.cudaMatchingJobs;
                plan.residentDescriptorSets+=previous.residentDescriptorSets;
                plan.framePreparationMs+=previous.framePreparationMs; plan.mapPreparationMs+=previous.mapPreparationMs;
                plan.rayPreparationMs+=previous.rayPreparationMs; plan.matchingSetupMs+=previous.matchingSetupMs;
                plan.matchingTraversalMs+=previous.matchingTraversalMs; plan.matchingSelectionMs+=previous.matchingSelectionMs;
                plan.matchingGuidanceMs+=previous.matchingGuidanceMs; plan.candidatePairs+=previous.candidatePairs;
                plan.descriptorComparisons+=previous.descriptorComparisons; plan.mapJobs+=previous.mapJobs;
                plan.seedJobs+=previous.seedJobs; plan.predictedQueries+=previous.predictedQueries;
                plan.unpredictedQueries+=previous.unpredictedQueries;
                plan.ownedQueries+=previous.ownedQueries; plan.ownedMatches+=previous.ownedMatches;
                plan.prioritySuppressed+=previous.prioritySuppressed;
                plan.stagedUsed=plan.stagedUsed || previous.stagedUsed;
                plan.broadFallback=plan.broadFallback || previous.broadFallback;
                plan.expansionJobs+=previous.expansionJobs; plan.broadJobs+=previous.broadJobs;
                plan.deferredStereoJobs+=previous.deferredStereoJobs;
            };
            bool accepted=solveProposal();
            if(!accepted && plan.stagedUsed) {
                // Rebuild from the same frozen input before declaring visual failure.
                auto previous=std::move(plan);
                auto complete=options; complete.stagedMatching=false;
                plan=retryProposal(complete);
                addAttemptTiming(previous);
                plan.poseFallback=true;
                plan.stageFallbackReason="pose_consensus";
                accepted=solveProposal();
            }
            double filteringMs=0;
            const auto filterAndClassify=[&]() {
                const auto started=std::chrono::steady_clock::now();
                if(accepted) frontend_association::reject(plan,decision.rejected);
                else {
                    decision=PoseDecision{}; decision.T_WB=predicted;
                    std::set<KeypointIdentifier> rejected;
                    for(const auto& observation:plan.batch.observations)
                        if(estimator.isLandmarkInitialised(observation.landmark)) rejected.insert(observation.key);
                    frontend_association::reject(plan,rejected);
                }
                frontend_association::filterMap(estimator,plan,decision.T_WB,options.maximumAngularError);
                auto classification=frontend_association::decideKeyframe(estimator,*plan.current,decision.T_WB,params.frontend,accepted);
                filteringMs+=milliseconds(started);
                return classification;
            };
            auto keyframeDecision=filterAndClassify();
            const bool stereoKeyframeRetry=accepted && keyframeDecision.keyframe && plan.deferredStereoJobs && !plan.poseFallback;
            if(stereoKeyframeRetry) {
                // A visual keyframe can disagree with the motion prediction. Restore
                // all stereo pairs before publishing it. Keep the verified solution
                // if additional matches fail consensus.
                auto previous=std::move(plan);
                auto previousDecision=decision;
                auto fullStereo=options; fullStereo.deferHealthyStereo=false;
                plan=retryProposal(fullStereo);
                if(solveProposal()) {
                    addAttemptTiming(previous);
                    keyframeDecision=filterAndClassify();
                } else {
                    auto failed=std::move(plan);
                    plan=std::move(previous); decision=std::move(previousDecision);
                    addAttemptTiming(failed);
                }
            }
            const double poseMs=std::max(0.0,milliseconds(poseStarted)-retryPreparationMs-filteringMs);
            if(kfPrior) { keyframeDecision.keyframe=true; keyframeDecision.reason="explicit_prior"; }
            const bool keyframe=keyframeDecision.keyframe;
            // Every accepted frame may keep fresh seed identities. Only promotion
            // to an optimized inverse range is gated by stable anchors and budgets.
            const auto triangulationStarted=std::chrono::steady_clock::now();
            // Rig stereo has a known metric baseline. Temporal depth uses only accepted
            // metric geometry; central five-point unit translation never supplies range.
            // An actual metric IMU edge permits temporal range initialization from
            // its predicted translation. A central five-point unit baseline does not.
            // Recovery still permits metric rig stereo when visual consensus is
            // unavailable. Stable-anchor deferral applies to healthy tracking.
            if(!isInitialized_ || trackingLost_ || !accepted || !decision.metric || plan.poseFallback) {
                options.maxPromotionsPerFrame=0; options.maxActiveInitialized=0;
            }
            if (options.estimateCameraTimeOffset && accepted)
                frontend_association::prepareBearingVelocities(estimator,plan);
            frontend_association::triangulate(estimator, plan, options, decision.T_WB, accepted && (decision.metric || inertialMotion),keyframe || !accepted);
            std::set<LandmarkId> committedSeedExtensions;
            for(const auto& observation:plan.batch.observations) if(plan.sourceSeedIds.count(observation.landmark))
                committedSeedExtensions.insert(observation.landmark);
            const double triangulationMs=milliseconds(triangulationStarted); const auto commitStarted=std::chrono::steady_clock::now();
            const auto health = accepted ? backend::SensorHealth::Valid : backend::SensorHealth::Degraded;
            const std::optional<kinematics::Transformation> seed = accepted
                ? std::optional<kinematics::Transformation>(decision.T_WB) : std::nullopt;
            std::string error;
            if (!estimator.applyFrontendBatch(id, plan.batch, seed, health, keyframe, error)) {
                lastReport_.message = "Association transaction rejected: " + error;
                LOG(WARNING)<<"[Frontend] Commit rejected; frame="<<id.value()<<" pose_consensus="<<accepted
                    <<" pose_inliers="<<decision.inliers<<" new_tracks="<<plan.batch.landmarks.size()
                    <<" observations="<<plan.batch.observations.size()<<" reason="<<error;
                return false;
            }
            lastReport_.committed = true;
            lastReport_.keyframe=keyframeDecision;
            const size_t newSeedCount=size_t(std::count_if(plan.batch.landmarks.begin(),plan.batch.landmarks.end(),[](const auto& point) { return !point.inverseDepth; }));
            size_t activeSeedCount=0;
            for(const auto& point:estimator.window().landmarks().landmarks()) if(!point.second.initialized) ++activeSeedCount;
            LOG(INFO)<<"[Keyframe] frame="<<id.value()<<" selected="<<keyframe<<" reason="<<keyframeDecision.reason<<" deferred_promotions="<<plan.deferredPromotions
                <<" new_seed_tracks="<<newSeedCount<<" active_seeds="<<activeSeedCount<<" seed_anchor_updates="<<plan.batch.seedAnchors.size()
                <<" seed_pool_before="<<plan.trackingSeeds<<" seed_budget_skipped="<<plan.seedBudgetSkipped
                <<" expired_seeds="<<plan.expiredSeeds<<" new_seed_cells="<<plan.newSeedCells
                <<" seed_coverage_skipped="<<plan.seedCoverageSkipped
                <<" deferred_promotion_budget="<<plan.deferredPromotionBudget
                <<" deferred_eviction_support="<<plan.deferredEvictionSupport
                <<" reference="<<keyframeDecision.reference.value()<<" shared_tracks="<<keyframeDecision.sharedTracks
                <<" parallax_rad="<<keyframeDecision.parallax<<" translation_m="<<keyframeDecision.translation
                <<" rotation_rad="<<keyframeDecision.rotation<<" previous="<<keyframeDecision.previous.value()
                <<" previous_shared="<<keyframeDecision.previousSharedTracks<<" previous_parallax_rad="<<keyframeDecision.previousParallax
                <<" previous_translation_m="<<keyframeDecision.previousTranslation<<" previous_rotation_rad="<<keyframeDecision.previousRotation;
            if(VLOG_IS_ON(1) || id.value()%30==0 || plan.poseFallback)
                LOG(INFO)<<"[SeedContinuity] frame="<<id.value()<<" source_seed_ids="<<plan.sourceSeedIds.size()
                    <<" matched_seed_ids="<<plan.matchedSeedIds.size()<<" proposed_extensions="<<plan.extendedSeedIds.size()
                    <<" committed_extensions="<<committedSeedExtensions.size()
                    <<" filtered_extensions="<<plan.extendedSeedIds.size()-committedSeedExtensions.size()
                    <<" slot_conflict_attempts="<<plan.seedSlotConflicts<<" identity_conflicts="<<plan.identityConflicts
                    <<" new_seed_tracks="<<newSeedCount<<" coverage_skipped="<<plan.seedCoverageSkipped
                    <<" owned_query_rows="<<plan.ownedQueries<<" owned_matches="<<plan.ownedMatches
                    <<" priority_suppressed="<<plan.prioritySuppressed;
            const double commitMs=milliseconds(commitStarted);
            try {
                for(auto& match:plan.debugMatches) {
                    const auto& first=match.first.frameId==plan.current->id()?*plan.current:*plan.sources.at(StateId(match.first.frameId));
                    const auto& second=match.second.frameId==plan.current->id()?*plan.current:*plan.sources.at(StateId(match.second.frameId));
                    const auto a=first.landmarkId(match.first.cameraIndex,match.first.keypointIndex);
                    match.retained=a!=0 && a==second.landmarkId(match.second.cameraIndex,match.second.keypointIndex);
                }
                lastReport_.debugMatches=std::move(plan.debugMatches);
            } catch(const std::exception& error) { LOG(WARNING) << "[Debug] Match diagnostics skipped: " << error.what(); }
            // Only an accepted backend transaction may update the caller's associations.
            // Features are unchanged; no writable alias to a managed frame is exposed.
            lastAssociatedState_ = id; *asKeyframe = keyframe;
            trackingLost_ = !plan.sources.empty() && !accepted;
            std::set<LandmarkId> finiteTracks;
            for (const auto& key : estimator.window().landmarks().inFrame(id)) {
                const auto observation = estimator.window().landmarks().find(key);
                if (observation && estimator.isLandmarkInitialised(observation->landmark)) finiteTracks.insert(observation->landmark);
            }
            isInitialized_ = isInitialized_ || (accepted && (decision.metric || inertialMotion) && finiteTracks.size() >= poseOptions_.minimumAbsoluteInliers);
            lastReport_.trackingAccepted = accepted;
            lastReport_.metricPose = accepted && decision.metric; lastReport_.inliers = decision.inliers;
            lastReport_.newTracks = plan.batch.landmarks.size(); lastReport_.newObservations = plan.batch.observations.size();
            lastReport_.initializedTracks = plan.batch.initializations.size();
            for (const auto& landmark : plan.batch.landmarks) if (landmark.inverseDepth) ++lastReport_.initializedTracks;
            lastReport_.message = accepted ? "Visual consensus committed; awaiting backend optimize"
                : "Bootstrap/degraded seed tracks committed; no accepted visual pose";
            VLOG(1) << "[Frontend] Staged frame=" << id.value() << " t_ns=" << framesInOut->timestamp().toNSec()
                << " source_frames=" << plan.sources.size() << " keypoints=" << framesInOut->numKeypoints()
                << " visual_consensus=" << accepted << " metric_visual_pose=" << lastReport_.metricPose
                << " inliers=" << lastReport_.inliers << " new_tracks=" << lastReport_.newTracks
                << " new_observations=" << lastReport_.newObservations << " initialized_tracks=" << lastReport_.initializedTracks
                << " finite_tracks=" << finiteTracks.size() << " keyframe=" << keyframe << " elapsed_ms="
                << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            estimator.synchronizeFrame(id,*framesInOut);
            const double totalMs=milliseconds(started);
            if(VLOG_IS_ON(1) || id.value()%30==0 || totalMs>200 || plan.sources.empty() || plan.poseFallback)
                LOG(INFO)<<"[Timing][Frontend] frame="<<id.value()<<" prepare_ms="<<plan.preparationMs
                    <<" prepare_frames_ms="<<plan.framePreparationMs<<" prepare_map_ms="<<plan.mapPreparationMs
                    <<" prepare_rays_ms="<<plan.rayPreparationMs
                    <<" match_ms="<<plan.matchingMs<<" assign_ms="<<plan.assignmentMs<<" pose_ms="<<poseMs
                    <<" filter_ms="<<filteringMs<<" triangulate_ms="<<triangulationMs<<" commit_ms="<<commitMs
                    <<" jobs="<<plan.matchingJobs<<" cuda_jobs="<<plan.cudaMatchingJobs<<" threads="<<options.matchingThreads
                    <<" device_descriptor_sets="<<plan.residentDescriptorSets
                    <<" candidate_tests="<<plan.candidatePairs<<" hamming_tests="<<plan.descriptorComparisons
                    <<" match_setup_sum_ms="<<plan.matchingSetupMs<<" match_traverse_sum_ms="<<plan.matchingTraversalMs
                    <<" match_select_sum_ms="<<plan.matchingSelectionMs
                    <<" match_guidance_sum_ms="<<plan.matchingGuidanceMs
                    <<" predicted_queries="<<plan.predictedQueries<<" unpredicted_queries="<<plan.unpredictedQueries
                    <<" new_track_limit="<<plan.admittedTrackLimit<<" capacity_skipped="<<plan.capacitySkipped
                    <<" retired_seeds="<<plan.retiredSeeds
                    <<" staged="<<plan.stagedUsed<<" broad_fallback="<<plan.broadFallback<<" pose_fallback="<<plan.poseFallback
                    <<" map_jobs="<<plan.mapJobs<<" seed_jobs="<<plan.seedJobs
                    <<" deferred_stereo_jobs="<<plan.deferredStereoJobs<<" stereo_keyframe_retry="<<stereoKeyframeRetry
                    <<" expansion_jobs="<<plan.expansionJobs<<" broad_jobs="<<plan.broadJobs
                    <<" fallback_reason="<<plan.stageFallbackReason
                    <<" total_ms="<<totalMs;
        }
        catch (const std::exception& error) {
            if (lastReport_.committed) {
                lastReport_.message = std::string("Association committed; caller view refresh failed: ") + error.what();
                LOG(WARNING) << lastReport_.message; return true;
            }
            lastReport_.message = error.what(); LOG(WARNING) << "Frontend association failed: " << error.what(); return false;
        }
        // Place recognition is an optional diagnostic after ingestion. Its failure cannot
        // turn an already committed input transaction into a reported ingestion failure.
        try { queryLocalPlaces(estimator, params); }
        catch (const std::exception& error) { LOG(WARNING) << "Local place query skipped: " << error.what(); }
        return true;
    }

    bool Frontend::runRansac3d2d(const Estimator& estimator, const frontend_association::Plan& plan,
        PoseDecision& decision,bool forceFullBudget) const {
        const auto& currentFrame = plan.current;
        const auto prepareStarted=std::chrono::steady_clock::now();
        const auto input = frontend_pose::absoluteInput(*currentFrame,plan.finitePoints,poseExtrinsics(estimator,*currentFrame));
        std::set<std::pair<KeypointIdentifier,LandmarkId>> proposedObservations;
        for(const auto& observation:plan.batch.observations) proposedObservations.emplace(observation.key,observation.landmark);
        for (const auto& observation : input.observations) {
            const auto metadata=estimator.window().landmarks().find(observation.landmark);
            if(!metadata || !metadata->initialized || metadata->anchor.frameId==currentFrame->id()) return false;
            const auto managed = estimator.window().landmarks().find(observation.key);
            if (managed && managed->landmark != observation.landmark) return false;
            const bool proposed=proposedObservations.count({observation.key,observation.landmark})!=0;
            if (!managed && !proposed) return false;
        }
        auto options = poseOptions_.solver;
        options.minInliers = std::max(options.minInliers, poseOptions_.minimumAbsoluteInliers);
        if (input.data.size() < options.minInliers) {
            VLOG(2) << "[Pose] Absolute skipped; frame=" << currentFrame->id()
                << " correspondences=" << input.data.size() << " required=" << options.minInliers;
            return false;
        }
        const auto initial = estimator.pose(StateId(currentFrame->id()));
        const bool reducedBudget=!forceFullBudget && poseOptions_.adaptiveAbsoluteBudget && isInitialized_ && !trackingLost_
            && absoluteBudgetEpochs_%poseOptions_.fullRansacPeriod!=0
            && frontend_pose::absolutePredictionHealthy(input.data,initial,poseOptions_)
            && poseOptions_.trackingMinIterations<options.ransac.min_iterations;
        const auto fullMinimum=options.ransac.min_iterations;
        if(reducedBudget) options.ransac.min_iterations=poseOptions_.trackingMinIterations;
        const double prepareMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prepareStarted).count();
        const auto solveStarted=std::chrono::steady_clock::now();
        auto result = pose_solver::AbsolutePoseSolver(options).solve(input.data, &initial);
        size_t attemptedIterations=result.iterations,attemptedModels=result.generatedModels;
        bool budgetFallback=false;
        if(reducedBudget && (!result.success() || result.inliers.size()<options.minInliers
            || result.inlierRatio<poseOptions_.budgetPredictionInlierRatio
            || result.rmsAngularError>poseOptions_.budgetPredictionRmsFraction*options.angularThreshold
            || (result.pose.r()-initial.r()).norm()>0.1
            || Eigen::AngleAxisd(result.pose.q()*initial.q().conjugate()).angle()>0.03)) {
            // This is a work-budget guard, not an IMU constraint. A full solve
            // may still recover a valid visual pose far from its prediction.
            options.ransac.min_iterations=fullMinimum; budgetFallback=true;
            const auto first=result.timing;
            result=pose_solver::AbsolutePoseSolver(options).solve(input.data,&initial);
            attemptedIterations+=result.iterations; attemptedModels+=result.generatedModels;
            result.timing.generationMs+=first.generationMs; result.timing.scoringMs+=first.scoringMs;
            result.timing.refinementMs+=first.refinementMs; result.timing.observabilityMs+=first.observabilityMs;
            result.timing.acceptedLoSteps+=first.acceptedLoSteps; result.timing.rejectedLoSteps+=first.rejectedLoSteps;
            result.timing.smallStepStops+=first.smallStepStops; result.timing.relativeCostStops+=first.relativeCostStops;
            result.timing.scoringPasses+=first.scoringPasses; result.timing.residualEvaluations+=first.residualEvaluations;
            result.timing.jacobianEvaluations+=first.jacobianEvaluations; result.timing.loIterations+=first.loIterations;
        }
        const double solveMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-solveStarted).count();
        if(VLOG_IS_ON(1) || prepareMs+solveMs>30 || currentFrame->id()%30==0) LOG(INFO)<<"[Timing][Pose] frame="<<currentFrame->id()
            <<" kind=absolute prepare_ms="<<prepareMs<<" solve_ms="<<solveMs<<" correspondences="<<input.data.size()
            <<" inliers="<<result.inliers.size()<<" iterations="<<result.iterations<<" refinements="<<result.refinements
            <<" generated_models="<<result.generatedModels<<" reduced_budget="<<reducedBudget<<" budget_fallback="<<budgetFallback
            <<" forced_full_budget="<<forceFullBudget<<" attempted_iterations="<<attemptedIterations<<" attempted_models="<<attemptedModels
            <<" budget_epoch="<<absoluteBudgetEpochs_
            <<" generation_ms="<<result.timing.generationMs<<" scoring_ms="<<result.timing.scoringMs
            <<" lo_ms="<<result.timing.refinementMs<<" observability_ms="<<result.timing.observabilityMs
            <<" scoring_passes="<<result.timing.scoringPasses<<" residual_point_evals="<<result.timing.residualEvaluations
            <<" jacobian_point_evals="<<result.timing.jacobianEvaluations<<" lo_iterations="<<result.timing.loIterations
            <<" lo_accepted="<<result.timing.acceptedLoSteps<<" lo_rejected="<<result.timing.rejectedLoSteps
            <<" lo_small_step_stops="<<result.timing.smallStepStops<<" lo_relative_cost_stops="<<result.timing.relativeCostStops;
        VLOG(2) << "[Pose] Absolute frame=" << currentFrame->id() << " correspondences=" << input.data.size()
            << " status=" << solverStatusName(result.status) << " inliers=" << result.inliers.size()
            << " ratio=" << result.inlierRatio << " rms_angle_rad=" << result.rmsAngularError
            << " RANSAC_iterations=" << result.iterations << " generated_models=" << result.generatedModels;
        if (!result.success() || result.inliers.size() < options.minInliers
            || result.inlierRatio < poseOptions_.absoluteInlierRatio) return false;
        std::vector<bool> mask; if (!inlierMask(result, input.data.size(), mask)) return false;
        std::set<LandmarkId> distinct;
        for (auto index : result.inliers) distinct.insert(input.observations[index].landmark);
        if (distinct.size() < poseOptions_.minimumAbsoluteInliers) return false;
        PoseDecision candidate; candidate.T_WB = result.pose; candidate.metric = true; candidate.inliers = result.inliers.size();
        candidate.reducedBudget=reducedBudget && !budgetFallback;
        for (size_t i = 0; i < mask.size(); ++i) if (!mask[i]) candidate.rejected.insert(input.observations[i].key);
        decision = std::move(candidate); return true;
    }

    bool Frontend::runRansac2d2d(const Estimator& estimator, const frontend_association::Plan& plan,
        PoseDecision& decision) const {
        if (plan.sources.empty()) { VLOG(2) << "[Pose] Relative skipped; no reference frames"; return false; }
        // Select the retained view with most independent common tracks, newest on ties.
        // Numeric ID adjacency is invalid after SECOND_NEW; no frameId-1 lookup is used.
        std::shared_ptr<MultiFrame> oldFrame;
        const auto currentFrame = plan.current;
        frontend_pose::RelativeInput input;
        for (auto it = plan.sources.rbegin(); it != plan.sources.rend(); ++it) {
            auto candidate = frontend_pose::relativeInput(*it->second, *currentFrame,
                poseExtrinsics(estimator, *it->second), poseExtrinsics(estimator, *currentFrame));
            if (!oldFrame || candidate.data.size() > input.data.size()) { oldFrame = it->second; input = std::move(candidate); }
        }
        auto options = poseOptions_.solver;
        options.minInliers = std::max(options.minInliers, poseOptions_.minimumRelativeInliers);
        if (input.data.size() < options.minInliers) {
            VLOG(2) << "[Pose] Relative skipped; frame=" << currentFrame->id()
                << " correspondences=" << input.data.size() << " required=" << options.minInliers;
            return false;
        }
        const auto oldPose = estimator.pose(StateId(oldFrame->id()));
        const auto predicted = estimator.pose(StateId(currentFrame->id()));
        const auto initial = oldPose.inverse() * predicted;
        const auto accepted = [&](const pose_solver::PoseSolverResult& result) {
            return result.success() && result.inliers.size() >= options.minInliers
                && result.inlierRatio >= poseOptions_.relativeInlierRatio;
        };
        const auto rotationStarted=std::chrono::steady_clock::now();
        const auto rotation = pose_solver::RotationPoseSolver(options).solve(input.data, &initial);
        const double rotationMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-rotationStarted).count();
        // This is an exact short circuit of the selection rule below: once this
        // accepted rotation exceeds rotationDominanceRatio, a relative solution
        // could NEVER be selected. Preserve IMU translation and do not invent scale.
        const bool rotationDominant=accepted(rotation) && rotation.inlierRatio>poseOptions_.rotationDominanceRatio;
        const auto relativeStarted=std::chrono::steady_clock::now();
        pose_solver::PoseSolverResult relative;
        if(!rotationDominant) relative=pose_solver::RelativePoseSolver(options).solve(input.data, &initial);
        const double relativeMs=rotationDominant?0:std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-relativeStarted).count();
        if(VLOG_IS_ON(1) || rotationDominant || rotationMs+relativeMs>30 || currentFrame->id()%30==0)
            LOG(INFO)<<"[Timing][Pose] frame="<<currentFrame->id()
            <<" kind=fallback rotation_ms="<<rotationMs<<" relative_ms="<<relativeMs<<" correspondences="<<input.data.size()
            <<" rotation_inliers="<<rotation.inliers.size()<<" rotation_ratio="<<rotation.inlierRatio
            <<" rotation_iterations="<<rotation.iterations<<" relative_attempted="<<!rotationDominant
            <<" relative_skip="<<(rotationDominant?"rotation_dominance":"none")
            <<" relative_iterations="<<relative.iterations<<" relative_models="<<relative.generatedModels
            <<" relative_generation_ms="<<relative.timing.generationMs<<" relative_scoring_ms="<<relative.timing.scoringMs
            <<" relative_lo_ms="<<relative.timing.refinementMs<<" relative_observability_ms="<<relative.timing.observabilityMs;
        VLOG(2) << "[Pose] Relative frame=" << currentFrame->id() << " reference=" << oldFrame->id()
            << " correspondences=" << input.data.size() << " status=" << solverStatusName(relative.status)
            << " inliers=" << relative.inliers.size() << " ratio=" << relative.inlierRatio
            << " rms_angle_rad=" << relative.rmsAngularError << " RANSAC_iterations=" << relative.iterations
            << " generated_models=" << relative.generatedModels << " metric=" << relative.translationScaleObservable
            << " rotation_status=" << solverStatusName(rotation.status) << " rotation_inliers=" << rotation.inliers.size();
        const bool rotationAccepted = accepted(rotation), relativeAccepted = accepted(relative);
        const bool useRotation = rotationAccepted && (!relativeAccepted || rotation.inliers.size() >= relative.inliers.size()
            || rotation.inlierRatio > poseOptions_.rotationDominanceRatio);
        if (!useRotation && !relativeAccepted) return false;
        const auto& result = useRotation ? rotation : relative;
        const bool metric = !useRotation && result.translationScaleObservable;
        VLOG(2) << "[Pose] Selected " << (useRotation?"rotation":"relative") << " frame=" << currentFrame->id()
            << " metric=" << metric << " inliers=" << result.inliers.size();
        std::vector<bool> mask; if (!inlierMask(result, input.data.size(), mask)) return false;
        PoseDecision candidate; candidate.metric = metric; candidate.inliers = result.inliers.size();
        if (!frontend_pose::relativeWorldSeed(oldPose, predicted, result, useRotation, candidate.T_WB)) return false;
        for (size_t i = 0; i < mask.size(); ++i) if (!mask[i]) {
            candidate.rejected.insert(input.observations[i].second);
            // A rejected track pair is not rescued by another current camera view of the
            // same tentative track. Its proposed current anchor must be rejected as well.
            const auto landmark = input.observations[i].landmark;
            for (const auto& observation : plan.batch.observations)
                if (observation.landmark == landmark) candidate.rejected.insert(observation.key);
            for (const auto& point : plan.batch.landmarks)
                if (point.id == landmark && point.anchor.frameId == currentFrame->id()) candidate.rejected.insert(point.anchor);
        }
        decision = std::move(candidate); return true;
    }

    void Frontend::queryLocalPlaces(const Estimator& estimator, const ViParameters& params) {
        if (!dBow_ || !params.estimator.do_loop_closures) return;
        const auto framesInOut = estimator.multiFrame(estimator.currentStateId());
        if (!framesInOut || dBow_->descriptorProfile != framesInOut->descriptorProfile(0)) return;
        // The window owns no global frame archive. Rebuild only retained keyframes so
        // entries never refer to evicted states and memory does not grow with trajectory.
        dBow_->database.clear(); dBow_->poseIds.clear();
        for (auto id : estimator.window().order()) {
            if (id == estimator.currentStateId() || !estimator.isKeyframe(id)) continue;
            const auto frame = estimator.multiFrame(id); if (!frame) continue;
            orbFeatures_.validateFrame(*frame);
            std::vector<cv::Mat> features;
            for (size_t c = 0; c < frame->numFrames(); ++c) for (int i = 0; i < frame->descriptors(c).rows; ++i)
                features.push_back(frame->descriptors(c).row(i));
            if (features.empty()) continue;
            const auto entry = dBow_->database.add(features);
            if (entry != dBow_->poseIds.size()) throw std::logic_error("Unexpected DBoW entry identity");
            dBow_->poseIds.push_back(id.value());
        }
        std::vector<cv::Mat> features;
        for (size_t c = 0; c < framesInOut->numFrames(); ++c) for (int i = 0; i < framesInOut->descriptors(c).rows; ++i)
            features.push_back(framesInOut->descriptors(c).row(i));
        std::vector<std::pair<StateId, double>> results; getFilteredDBoWResult(dBow_, features, results);
        for (const auto& result : results) {
            if (!std::isfinite(params.estimator.p_dbow) || result.second < params.estimator.p_dbow) continue;
            FrontendLoopCandidate candidate; candidate.oldState = result.first; candidate.currentState = estimator.currentStateId();
            if (verifyRecognisedPlace(estimator, params, framesInOut, estimator.multiFrame(result.first),
                candidate.T_Bold_Bnew, candidate.information, static_cast<int>(poseOptions_.minimumAbsoluteInliers)))
                lastReport_.localLoopCandidates.push_back(std::move(candidate));
        }
        // These candidates are correlated with local visual evidence. Do not add them
        // as independent loop costs to the same window or claim global loop closure.
    }

    bool Frontend::propagation(const ImuMeasurementDeque& measurements, const ImuParameters& parameters,
        kinematics::Transformation& pose, SpeedAndBias& speedBias, const Time& start, const Time& end,
        Eigen::Matrix<double, 15, 15>* covariance, Eigen::Matrix<double, 15, 15>* jacobian) const {
        if (measurements.size() < 2 || end.toNSec() <= start.toNSec()) return false;
        auto propagated = pose; auto sb = speedBias;
        Eigen::Matrix<double, 15, 15> cov = Eigen::Matrix<double, 15, 15>::Constant(std::numeric_limits<double>::quiet_NaN());
        Eigen::Matrix<double, 15, 15> jac = cov;
        try {
            const auto integrate=imuPropagator_?imuPropagator_:ImuPropagator(factor::propagateImu);
            if (!integrate(measurements, parameters, propagated, sb, start, end, covariance ? &cov : nullptr, jacobian ? &jac : nullptr)
                || !backend::ParameterBlock::pose({ backend::BlockKind::CameraExtrinsic,0 }, propagated).valid()
                || !sb.allFinite() || (covariance && !cov.allFinite()) || (jacobian && !jac.allFinite())) return false;
        }
        catch (const std::exception& error) { LOG(WARNING) << "IMU prediction failed: " << error.what(); return false; }
        pose = propagated; speedBias = sb;
        if (covariance) *covariance = cov;
        if (jacobian) *jacobian = jac;
        return true;
    }
    void Frontend::resetTrackingState() {
        absoluteBudgetEpochs_=0;
        isInitialized_=false; trackingLost_=true;
        lastAssociatedState_=StateId(); lastReport_=FrontendAssociationReport();
    }
    void Frontend::clear() {
        absoluteBudgetEpochs_=0;
        orbFeatures_.clear(); dBow_.reset(); isInitialized_ = false; trackingLost_ = false;
        lastAssociatedState_ = StateId(); lastReport_ = FrontendAssociationReport();
    }
} // namespace cgmli
