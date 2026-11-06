#pragma once

#include <cstdint>
#include <common/FrameTypedefs.hpp>
#include <Eigen/Core>
#include <optional>
#include <vector>

namespace cgmli { namespace backend {

/// The observation table owns measurements; frame/landmark indices contain keys only.
/// Absorbed measurements may help appearance association, but must never be resubmitted
/// as independent evidence after their original factor has entered a marginal prior.
enum class ObservationUse { TrackingOnly, LiveFactor, Absorbed };
struct ObservationOptions {
    double pixelSigma = 1.0; // Standard deviation in the original image, not ORB patch diameter.
    Eigen::Vector3d bearingVelocity = Eigen::Vector3d::Zero();
    std::optional<double> referenceTimeOffset; // Default: camera td frozen when this epoch was appended.
};
struct WindowObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeypointIdentifier key;
    LandmarkId landmark;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    Eigen::Vector3d bearing = Eigen::Vector3d::UnitZ();
    Eigen::Matrix2d sqrtInformation = Eigen::Matrix2d::Identity();
    // Tangent bearing derivative per second, expressed at bearing. The correction
    // aligns the camera measurement to the immutable epoch time using td_ref - td.
    Eigen::Vector3d bearingVelocity = Eigen::Vector3d::Zero();
    double referenceTimeOffset = 0.0;
    ObservationUse use = ObservationUse::TrackingOnly;
    uint64_t factorId = 0; // Absorbed implicit anchors have no standalone residual ID.
};

/// Geometric identity/metadata only. Optimized rho lives exclusively in BackendGraph.
/// No independently writable world point is stored here.
struct WindowLandmark {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LandmarkId id;
    KeypointIdentifier anchor;
    Eigen::Vector3d anchorBearing = Eigen::Vector3d::UnitZ();
    bool initialized = false;
    double quality = 0.0;
    int classification = -1;
    // Bounded per-identity lifecycle counters travel with transaction checkpoints.
    // They distinguish repeated resets of ONE track from many new physical points.
    uint64_t createdFrame=0,lastTransitionFrame=0,lastResetFrame=0,lastInitializedFrame=0;
    uint32_t initializationCount=0,resetCount=0,demotionCount=0;
};

/// Immutable exported estimate computed from one accepted graph/calibration revision.
struct LandmarkEstimate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    WindowLandmark metadata;
    std::optional<double> inverseDepth;
    std::optional<Eigen::Vector3d> pointWorld;
    std::vector<WindowObservation> observations;
};

struct LandmarkInsertion {
    LandmarkId id;
    KeypointIdentifier anchor;
    std::optional<double> inverseDepth;
    ObservationOptions options;
};
struct LandmarkInitialization { LandmarkId id; double inverseDepth = 0.0; };
/// Change only a seed's coordinate reference before it has an optimization variable.
struct SeedAnchorUpdate { LandmarkId id; KeypointIdentifier anchor; };
struct ObservationInsertion {
    LandmarkId landmark;
    KeypointIdentifier key;
    ObservationOptions options;
};
/// Production ingestion path: one checkpoint and one consistency validation for all
/// associations in an image, rather than copying the full window per feature.
/// Operations run in order: remove observations, retire points, create seed tracks,
/// add observations, select seed anchors, initialize depths. Failure rolls back all.
struct VisualUpdateBatch {
    std::vector<KeypointIdentifier> removedObservations;
    std::vector<LandmarkId> retiredLandmarks;
    std::vector<LandmarkInsertion> landmarks;
    std::vector<LandmarkInitialization> initializations;
    std::vector<ObservationInsertion> observations;
    std::vector<SeedAnchorUpdate> seedAnchors;
};

} }
