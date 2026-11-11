#pragma once

#include <backend/ViSlamBackend.hpp>
#include <frontend/FrontendPoseAdapter.hpp>
#include <frontend/OrbFrontendFeatures.hpp>
#include <common/ViFrontendInterface.hpp>
#include <visual/features/OrbMatcher.hpp>
#include <string>
#include <set>
#include <vector>

namespace cgmli { namespace frontend_association {

/// Single-writer association settings, independent of camera distortion enums.
struct Options {
    size_t maxSourceFrames=4, maxNewLandmarks=200;
    size_t matchingThreads=1;
    bool useCudaMatcher=false;
    size_t maxPromotionsPerFrame=40,maxActiveInitialized=1000; ///< Zero disables each healthy-tracking budget.
    size_t maxTrackingSeeds=1000; ///< Admission ceiling for unoptimized identities; zero disables it.
    double seedMaxUntrackedSeconds=0.5; ///< Age of the last real observation; zero disables expiry.
    bool requireKeyframeAnchor=true;
    bool replenishSeedGaps=true; ///< Healthy map targets replenish half-density 4x4 cells, not every free keypoint.
    bool estimateCameraTimeOffset=false;
    float searchRadiusPixels=30.0f;
    double pixelSigma=1.0, pyramidScale=1.2; // Octave-zero pixel noise; not ORB patch diameter.
    double minimumParallax=0.015, maximumAngularError=0.01; // Radians.
    double minimumDepth=0.05, maximumDepth=10000.0;
    features::OrbMatcherOptions matcher;
    bool collectDebugMatches=false; // Preserve actual candidate pairs only when visualization is enabled.
    bool stagedMatching=true; ///< Map first, recent/stereo seeds, target-local expansion, then full pose fallback.
    bool deferHealthyStereo=false; ///< Check recent temporal and map support before launching stereo.
    size_t mapDescriptorsPerLandmark=3, minimumStageMatchesPerCamera=40;
    void validate() const;
};

/// A private, discardable proposal. Working frames own cloned measurements; their
/// associations are NOT backend state. Only batch is submitted after consensus.
struct Plan {
    backend::VisualUpdateBatch batch;
    std::shared_ptr<MultiFrame> current;
    std::map<StateId,std::shared_ptr<MultiFrame>> sources;
    std::vector<FeatureMatchDebug> debugMatches;
    double preparationMs=0, matchingMs=0, assignmentMs=0;
    uint64_t candidatePairs=0, descriptorComparisons=0;
    size_t matchingJobs=0,cudaMatchingJobs=0,residentDescriptorSets=0;
    std::map<LandmarkId,Eigen::Vector3d> finitePoints; ///< One coordinate evaluation per proposal.
    size_t admittedTrackLimit=0, capacitySkipped=0, retiredSeeds=0;
    size_t trackingSeeds=0,seedBudgetSkipped=0,expiredSeeds=0,newSeedCells=0;
    size_t seedCoverageSkipped=0,identityConflicts=0,seedSlotConflicts=0;
    std::set<LandmarkId> sourceSeedIds,matchedSeedIds,extendedSeedIds;
    uint64_t ownedQueries=0,ownedMatches=0,prioritySuppressed=0;
    double matchingSetupMs=0, matchingTraversalMs=0, matchingSelectionMs=0;
    double framePreparationMs=0, mapPreparationMs=0, rayPreparationMs=0, matchingGuidanceMs=0;
    uint64_t predictedQueries=0, unpredictedQueries=0;
    size_t mapJobs=0,seedJobs=0;
    size_t expansionJobs=0,broadJobs=0;
    size_t deferredStereoJobs=0;
    double stageRotationRadians=0;
    std::vector<size_t> stageMapMatches,stagePrimaryMatches,stageExpandedMatches;
    std::vector<size_t> stagePrimaryExistingMatches,stageExpandedExistingMatches;
    std::vector<unsigned> stagePrimarySectors;
    std::string stageFallbackReason="disabled";
    bool stagedUsed=false,broadFallback=false,poseFallback=false;
    size_t deferredPromotions=0,deferredEvictionSupport=0,deferredPromotionBudget=0;
};
struct KeyframeDecision {
    bool keyframe=false;
    StateId reference,previous;
    size_t sharedTracks=0,previousSharedTracks=0;
    double parallax=0,previousParallax=0,translation=0,rotation=0,previousTranslation=0,previousRotation=0;
    std::string reason="unavailable";
};
/// VINS-style low-parallax redundancy decision on full-sphere bearings. It uses
/// independent SAME-camera tracks, with rotation compensation, and accumulated
/// reference motion so slow travel is not mistaken for indefinite stationarity.
KeyframeDecision decideKeyframe(const Estimator& estimator,const MultiFrame& current,
    const kinematics::Transformation& currentPose,const FrontendParameters& parameters,bool trackingAccepted);

/// The caller must extract before addStates(): frozen backend indices are immutable.
bool sameMeasurements(const MultiFrame& input,const MultiFrame& frozen);
/// No mutations, no optimization, no triangulation from a unit-baseline pose.
// descriptorSource is the transient current frame after sameMeasurements()
// verified that its descriptor row order matches the frozen backend copy.
Plan propose(const Estimator& estimator,const Options& options,const MultiFrame* descriptorSource=nullptr);
/// Reject only proposed current observations; prune orphaned newly created tracks.
/// Existing backend observations are never deleted or relabelled by this operation.
void reject(Plan& plan,const std::set<KeypointIdentifier>& keys);
/// Check every proposed view of a mature map point at the accepted/predicted pose.
void filterMap(const Estimator& estimator,Plan& plan,const kinematics::Transformation& T_WB,
    double maximumAngularError);
/// Populate only new measurements, using previous observations of the same camera and track.
void prepareBearingVelocities(const Estimator& estimator,Plan& plan);
/// Initialize inverse RANGE along the ORIGINAL anchor ray, checking ALL retained
/// measurements. Temporal triangulation requires metric motion; rig stereo does not.
void triangulate(const Estimator& estimator,Plan& plan,const Options& options,
    const kinematics::Transformation& T_WB,bool allowTemporal,bool currentKeyframe=true);
/// Copy only authoritative association outputs back into the caller's feature frame.
void synchronize(const MultiFrame& authoritative,MultiFrame& output);

} }
