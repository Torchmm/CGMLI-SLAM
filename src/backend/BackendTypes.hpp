#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include "WindowDataTypes.hpp"
#include "LeftSE3Manifold.hpp"
#include <common/FrameTypedefs.hpp>
#include <common/Measurements.hpp>
#include <utils/Time.h>
#include <ceres/cost_function.h>
#include <ceres/loss_function.h>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <array>
#include <cmath>

namespace cgmli { struct ImuParameters; namespace backend {

// BlockKind and an immutable numeric ID form a logical address. Epoch pose/velocity
// blocks use StateId; landmarks use track IDs; each global calibration kind uses ID 0.
enum class BlockKind { Pose, SpeedBias, CameraExtrinsic, CameraTimeOffset,
    LidarExtrinsic, LidarTimeOffset, Landmark, Auxiliary };
struct BlockKey {
    BlockKind kind = BlockKind::Auxiliary;
    uint64_t id = 0;
    bool operator<(const BlockKey& b) const { return std::tie(kind,id) < std::tie(b.kind,b.id); }
    bool operator==(const BlockKey& b) const { return kind == b.kind && id == b.id; }
    bool operator!=(const BlockKey& b) const { return !(*this == b); }
    static BlockKey pose(StateId id) { return {BlockKind::Pose,id.value()}; }
    static BlockKey speedBias(StateId id) { return {BlockKind::SpeedBias,id.value()}; }
    bool isState() const { return kind == BlockKind::Pose || kind == BlockKind::SpeedBias; }
    bool isCalibration() const { return kind >= BlockKind::CameraExtrinsic && kind <= BlockKind::LidarTimeOffset; }
};

/**
 * Owning storage for one optimization variable.
 * value.size() is the ambient size seen by CostFunction; tangentSize() is the number
 * of local coordinates used by Ceres, normal equations, and marginalization priors.
 * Copies duplicate the value buffer but share an immutable manifold implementation.
 * Do not resize/replace buffers while a Ceres Problem borrows their data pointers.
 */
struct ParameterBlock {
    BlockKey key;
    Eigen::VectorXd value;
    std::shared_ptr<::ceres::Manifold> manifold;
    // A block with an owner is removed with that epoch. Calibration MUST have no owner.
    // An inverse-depth track's owner is its anchor epoch, until a landmark adapter reanchors it.
    std::optional<StateId> owner;
    bool constant = false; // Temporary optimization freeze; marginalization still retains its Jacobian.
    Eigen::VectorXd lower, upper; // Optional ambient bounds. SE(3) uses step checks, not quaternion bounds.
    int tangentSize() const { return manifold ? manifold->TangentSize() : static_cast<int>(value.size()); }
    /// Check layout, finiteness, quaternion convention, ownership, and ambient bounds.
    bool valid(std::string* why = nullptr) const;
    /// Copy a project Transformation into independent [p,xyzw] storage with left SE(3).
    static ParameterBlock pose(BlockKey key, const kinematics::Transformation& T,
        std::optional<StateId> owner = std::nullopt);
    static ParameterBlock euclidean(BlockKey key, const Eigen::VectorXd& value,
        std::optional<StateId> owner = std::nullopt);
};

// Invalid measurements are excluded; Degraded may still constrain state estimation.
// Calibration activation is stricter and additionally requires a Valid modality.
enum class SensorHealth { Valid, Degraded, Invalid };
enum class FactorKind { Imu, Visual, Lidar, Zupt, Calibration, Other, Marginalization };
// Pending: dependencies missing. Active: dependencies available; participates()
// additionally checks solveEnabled before gating/optimization/marginalization.
// Rejected: discarded evidence. Marginalized: evidence represented by the frozen prior.
enum class FactorLifecycle { Pending, Active, Rejected, Marginalized };
enum class MarginalizationPolicy { MARGIN_OLD, MARGIN_SECOND_NEW };
enum class BackendLinearSolver { SparseNormalCholesky, DenseQr, DenseSchur };
enum class VisualBatchMode { Reference, Host, Cuda, Audit, Hybrid, Schur };
using FactorId = uint64_t;

/// Immutable replay material for native IMU interval concatenation. It is shared
/// across graph checkpoints; merged intervals receive a new payload and cost.
/// An opaque external IMU cost can omit this and use the Schur bridge fallback.
struct ImuFactorReplay {
    ImuMeasurementDeque samples;
    std::shared_ptr<const ImuParameters> parameters;
    Time start,end;
    bool completeSamples=true;
    std::string integrationMode="full";
};

/**
 * Measurement metadata plus shared ownership of its Ceres implementation.
 * Costs must return covariance-whitened residuals and ambient Jacobians; the manager
 * cannot infer a sensor covariance, bearing model, or preintegration from this record.
 * After submission, cost/loss objects must remain immutable and safe for concurrent
 * Evaluate calls when numThreads > 1. Only participates() records enter the problem.
 */
struct FactorRecord {
    FactorId id = 0; // Assigned by SlidingWindowManager, never reused.
    FactorKind kind = FactorKind::Other;
    FactorLifecycle lifecycle = FactorLifecycle::Pending;
    // Retained interval metadata is separate from eligibility for numerical use.
    // Suppressed native IMU records may be concatenated, never optimized/absorbed.
    bool solveEnabled = true;
    bool participates() const { return lifecycle==FactorLifecycle::Active && solveEnabled; }
    std::vector<BlockKey> keys; // Exactly CostFunction::parameter_block_sizes() order, no duplicate key.
    std::vector<StateId> epochs; // ALL measurement/anchor epochs, including constant-data dependencies.
    StateId sensorEpoch; // Required for Visual/Lidar: epoch whose sensor health applies to this measurement.
    std::shared_ptr<::ceres::CostFunction> cost;
    std::shared_ptr<::ceres::LossFunction> loss;
    std::shared_ptr<const ImuFactorReplay> imuReplay;
    // CostFunction must ALREADY return whitened residuals/Jacobians. Gate BEFORE robust loss.
    // Units are squared Mahalanobis distance; infinity disables the statistical gate.
    double nisThreshold = std::numeric_limits<double>::infinity();
    std::string reason;
};

/// Sensor-independent graph node metadata. Timestamp is physical IMU-clock time;
/// packet arrival and optimized time offsets must not change epoch identity/topology.
struct EpochNode {
    StateId id;
    Time timestamp;
    SensorHealth visual = SensorHealth::Invalid;
    SensorHealth lidar = SensorHealth::Invalid;
    bool keyframe = false;
};

/// Value-type input prediction/output estimate, never a view into Ceres memory.
/// T_WB maps Body B into world W; velocity is that of the Body origin in W.
/// The project IMU factor keeps bg/ba in sensor S and applies the fixed T_BS,
/// including the velocity lever arm. B=S is the identity-extrinsic special case.
struct StateEstimate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StateId id;
    Time timestamp;
    kinematics::Transformation T_WB;
    SpeedAndBias speedBias = SpeedAndBias::Zero(); // [v_W, bg, ba]
    SensorHealth visual = SensorHealth::Invalid, lidar = SensorHealth::Invalid;
    bool keyframe = false;
};

class MarginalizationPrior;
/// Copyable transaction state. Map keys identify blocks independently of memory addresses.
/// A graph copy duplicates numeric values and factor metadata, but shares immutable costs
/// and the prior. The previous prior is stored separately to avoid user factor-ID collisions.
struct BackendGraph {
    std::map<BlockKey,ParameterBlock> blocks;
    std::map<FactorId,FactorRecord> factors; // Pending and active only; terminal records go to bounded history.
    std::shared_ptr<MarginalizationPrior> prior;
    bool valid(std::string* why = nullptr) const;
};

/// Numeric limits and storage budgets for the common CPU backend core.
/// These defaults are not a measured end-to-end realtime guarantee; dense Schur/eigen
/// work, factor construction, and application callbacks need separate profiling.
struct BackendOptions {
    size_t windowSize = 10; // Retained frame states; one extra insertion/optimization slot.
    size_t maxPendingFactors = 4096; // Upper bound for costs awaiting future dependencies.
    size_t historySize = 2048; // Lightweight lifecycle events, not retained sensor data.
    size_t maxLandmarks = 3000;
    size_t maxObservations = 30000;
    size_t maxImuSamplesPerEpoch = 10000;
    bool retainFrameImages = false; // Features/descriptors are owned; raw images are optional.
    bool exportVisualHistory = false; // Accepted pre-Schur measurements for a separate global archive.
    // Opt-in chart candidates must pass world-point derivative AND H/b equivalence
    // checks. A fixed-bearing scalar chart generally cannot pass a baseline-changing
    // reanchor; such tracks are Schur-retired. Full prior transport is not implemented.
    bool autoReanchorLandmarks = false;
    // OLD may discard its outgoing anchor measurement and rebuild a prior-free
    // track on surviving measured rays, as SECOND_NEW does. This is a controlled
    // model reset, not exact transport. Only used when preserveOldestVisualInformation
    // is false; strict autoReanchorLandmarks takes priority over that legacy reset.
    bool resetOldestLandmarks = true;
    bool preserveOldestVisualInformation = true;
    bool preserveSchurTrackIds = true; // New depth requires NEW views after Schur consumption.
    double maxImuIntervalSeconds = 10.0;
    bool incrementalImu = true;
    bool landmarkFirstMarginalization = true;
    bool parallelNormalEquations = true;
    double minInverseDepth = 1e-8;
    double minReanchorParallax = 1e-3; // radians
    double maxReanchorAngularError = 0.03; // radians
    double visualNisThreshold = 9.210340371976184; // Squared whitened 2D residual; pipeline exposes configuration.
    double visualHuberScale = 1.0; // Whitened units; zero disables the robust loss.
    int maxIterations = 5; // Total LM iteration budget across outlier refits.
    int numThreads = 1;
    int maxOutlierRefits = 3; // Bounded internal recovery; all passes still share maxIterations/time.
    bool preferLandmarkSchur = true; // Only when EVERY residual satisfies landmark independence.
    bool analyticVisualJacobians = true; // False retains the central-difference reference.
    bool cacheVisualPairs = true; // False retains per-pose caching but recomputes pair chains per factor.
    VisualBatchMode visualBatchMode = VisualBatchMode::Reference;
    BackendLinearSolver linearSolver = BackendLinearSolver::SparseNormalCholesky;
    double maxSolverTimeSeconds = 0.04; // Ceres calls only, shared across refits; excludes gate/build/marginalization.
    // Rank cutoff = max(absolute threshold, relative threshold * largest nonnegative
    // eigenvalue). Weak/null directions are discarded rather than artificially constrained.
    double relativeEigenThreshold = 1e-9;
    double absoluteEigenThreshold = 1e-12;
    double maxRelativeCostIncrease = 1e-8;
    void validate() const;
    static BackendOptions realtime() { BackendOptions o; o.maxIterations = 4; return o; }
};

/// Solver diagnostic; parameter values are committed only when accepted is true.
/// When an outlier refit changes the factor set, initialCost and finalCost may refer
/// to different sets. Acceptance checks compare each pass against its own initial cost.
struct ResidualStatistics {
    size_t factors=0,components=0,overGate=0;
    double squaredNorm=0,robustCost=0,medianNis=0,p95Nis=0,maxNis=0;
    FactorId worstFactor=0;
    double componentRms() const { return components?std::sqrt(squaredNorm/double(components)):0; }
};
using ResidualStatisticsArray=std::array<ResidualStatistics,5>; // Visual, IMU, LiDAR, prior, other.
enum class SolveTermination { Failure, SolverFinished, TimeBudget, IterationBudget, RefitBudget };
const char* solveTerminationName(SolveTermination reason);
struct SolveReport {
    struct ImuEdgeNis {
        FactorId factor=0;
        double before=std::numeric_limits<double>::quiet_NaN();
        double after=std::numeric_limits<double>::quiet_NaN();
    };
    bool accepted = false;
    SolveTermination termination=SolveTermination::Failure;
    // Live outliers are already removed from the graph, but this accepted state
    // has not been refitted without these observations because a budget ended.
    size_t unrefittedOutliers=0;
    bool retryable = true; // Failure retry policy; a budget-limited accepted result needs no retry.
    std::vector<BlockKey> rejectedCalibration; // Freeze these blocks on a calibration-specific retry.
    double initialCost = 0.0, finalCost = 0.0;
    int iterations = 0;
    size_t preRejected = 0, postRejected = 0;
    bool preNisSkipped=false;
    double preNisMs=0, postNisMs=0;
    double gatingMs=0,problemBuildMs=0,ceresMs=0;
    double ceresTimeLimitMs=0; // Zero reports an effectively unlimited Ceres allowance.
    bool ceresTimeBudgetReached=false; // May exceed the allowance at an iteration boundary or minimum.
    double gateSetupMs=0,gateVisualPrepareMs=0;
    double visualPosePrepareMs=0,visualCameraPrepareMs=0,visualPairPrepareMs=0;
    double visualTimedAnchorPrepareMs=0;
    size_t visualSharedTimedAnchors=0;
    size_t visualCachedCameras=0;
    double validationMs=0,checkpointMs=0;
    double linearSolverMs=0,jacobianMs=0,residualMs=0;
    // Ceres evaluator calls summed over all refits, NOT per-factor Evaluate calls.
    int jacobianEvaluations=0,residualEvaluations=0;
    // Line-search diagnostics overlap the Ceres/evaluation timers; do not sum them.
    double lineSearchCostMs=0,lineSearchGradientMs=0,lineSearchMs=0;
    std::string lineSearchInterpolation="none";
    int passes=0,parameterBlocks=0,residualBlocks=0,residualDimension=0,effectiveParameters=0;
    size_t activeLandmarks=0;
    // Current-frame observations of variable landmark blocks in the last Ceres pass.
    std::vector<size_t> activeLandmarksByCamera;
    int problemBuilds=0,problemReuses=0;
    bool analyticVisualJacobians=true;
    size_t refreshedImuFactors=0;
    double imuRefreshMs=0;
    bool residualStatisticsValid=false;
    double residualStatisticsMs=0;
    ResidualStatisticsArray preGateResiduals{},preSolveResiduals{},postSolveResiduals{};
    ImuEdgeNis lightweightOldImu,lightweightNewImu;
    int priorRows=0,priorColumns=0;
    bool schurIndependent=false;
    std::string linearSolverUsed="none",sparseBackend="none";
    // Visual, IMU, LiDAR, historical prior, other. Summed per-call worker
    // elapsed time, not wall time; sampled every 30 epochs or with VLOG(1).
    bool factorProfiled=false;
    size_t visualCachedPairs=0,visualCacheRefreshes=0;
    std::array<double,5> factorJacobianSumMs{},factorResidualSumMs{};
    std::array<uint64_t,5> factorJacobianCalls{},factorResidualCalls{};
    std::string message;
};

/// Numerical audit of the local Schur quadratic, not extra information in the
/// prior. Robust cost and the squared norm returned by Ceres are kept separate;
/// their constants need not agree under a robust loss.
struct PriorFormationStatistics {
    bool evaluated=false;
    size_t visualFactors=0,imuFactors=0,otherFactors=0;
    double robustCost=0,linearizedSquaredNorm=0;
    double incomingReferenceNorm=0,incomingCurrentNorm=0;
    double eliminatedReduction=0,newReferenceNorm=0,quadraticConstantOffset=0;
    double retainedGradientNorm=0,droppedGradientNorm=0,eigenCutoff=0;
};
struct LinearizationTiming {
    double setupMs=0, evaluateMs=0, normalMs=0;
    int normalThreads=1;
};

/// Topology-change receipt. Dimensions count participating tangent coordinates;
/// unobserved retired blocks need not contribute columns to the linearized system.
struct MarginalizationReport {
    bool success = false;
    StateId removed;
    MarginalizationPolicy policy = MarginalizationPolicy::MARGIN_OLD;
    size_t consumedFactors = 0;
    int eliminatedDimension = 0, retainedDimension = 0, priorRank = 0;
    std::vector<BlockKey> retiredBlocks;
    std::vector<LandmarkId> retiredLandmarks, reanchoredLandmarks;
    struct Retirement {
        LandmarkId id; KeypointIdentifier anchor;
        bool initialized=false;
        size_t survivingLive=0,survivingAbsorbed=0,survivingTracking=0;
        size_t observedEpochs=0;
        std::string reason;
    };
    std::vector<Retirement> retirements;
    size_t landmarksBefore=0,landmarksAfter=0,retiredMature=0,retiredSeeds=0,retiredMatureWithLiveViews=0;
    bool removedKeyframe=false;
    size_t droppedVisualFactors=0,replacedImuFactors=0,movedSeeds=0;
    std::vector<LandmarkId> resetLandmarks; // VINS-style measurement-chart reset, not exact prior transport.
    std::vector<LandmarkId> demotedLandmarks; // ID/measurements retained; insufficient views for a finite optimized point.
    struct TrackTransition {
        LandmarkId id;
        KeypointIdentifier oldAnchor,newAnchor;
        std::string action,reason;
        uint32_t initializations=0,resets=0,demotions=0;
        uint64_t previousTransitionFrame=0;
        size_t survivingViews=0;
        double pointShift=0,maxRetainedNis=0;
    };
    std::vector<TrackTransition> trackTransitions;
    size_t repeatedResets=0,repeatedDemotions=0;
    // Checks the OLD frozen prior during chart preparation, before a NEW Schur
    // prior is formed. Does not claim discarded image information is preserved.
    bool resetPriorUnchanged=false;
    size_t resetPriorProtectedBlocks=0;
    std::string imuMode="none";
    bool priorReused=false;
    size_t priorLandmarks=0,activeImuFactors=0;
    size_t suppressedImuFactors=0,incrementalImuMerges=0,fullImuMerges=0;
    size_t scalarSchurLandmarks=0;
    size_t ignoredAbsorbedIncidentObservations=0,liveIncidentRanges=0,sameFrameRangeEliminations=0;
    PriorFormationStatistics formation;
    std::vector<LandmarkId> schurPreservedTracks;
    std::vector<KeypointIdentifier> removedObservations;
    std::string message;
    double linearizeMs=0,schurMs=0,priorMs=0;
    LinearizationTiming linearization;
    double topologyCopyMs=0,topologyValidationMs=0;
};

struct FactorEvent {
    FactorId id = 0;
    FactorKind kind = FactorKind::Other;
    FactorLifecycle lifecycle = FactorLifecycle::Pending;
    std::string reason;
};

/// Immutable published values for consumers outside the single writer thread.
/// Includes surviving states and the calibration from the same accepted transaction.
/// imuOnly describes the newest epoch's modality availability, not an observability proof.
struct BackendSnapshot {
    uint64_t revision = 0;
    std::vector<StateEstimate> states;
    std::map<BlockKey,Eigen::VectorXd> calibration;
    std::vector<LandmarkEstimate> landmarks;
    bool imuOnly = true;
};

/// End-to-end transaction result. On failure snapshot still points to the last accepted
/// publication; an optional marginalization report may describe the failed attempt.
struct StepReport {
    bool accepted = false;
    SolveReport solve;
    double marginalizationMs=0,snapshotMs=0;
    double inputValidationMs=0,checkpointMs=0,maintenanceMs=0,outgoingMs=0,receiptMs=0;
    std::optional<MarginalizationReport> marginalization;
    // Differences from the last published snapshot, including optimizer outlier removal.
    std::vector<LandmarkId> retiredLandmarks;
    std::vector<KeypointIdentifier> removedObservations;
    // A candidate only, emitted AFTER marginalization. Map owner must apply all geometry/quality gates.
    // SECOND_NEW never emits a map candidate in this implementation.
    std::optional<StateEstimate> mapCandidate;
    // Final accepted outgoing value for BOTH eviction policies. Trajectory output
    // needs it even when no map commit is authorized by the application's gates.
    std::optional<StateEstimate> retiredState;
    // Historical visual export only, not Stable Map insertion authorization.
    // Coordinates are taken after optimization and before Schur retirement.
    MapPointVector retiredVisualPoints;
    // Optional history receipt for an independent global graph/BA. This contains
    // accepted raw observations, never an extra local factor or Stable Map grant.
    std::vector<LandmarkEstimate> optimizedVisualHistory;
    std::vector<KeypointIdentifier> rejectedVisualObservations;
    std::shared_ptr<const BackendSnapshot> snapshot;
    std::string message;
};

/// Per-call controls, copied into the worker task. Minimum iterations affect the
/// soft time limit; numeric convergence may still terminate a solve earlier.
/// timeSeconds counts nonlinear solver time, shared across refits.
/// It excludes frontend work, NIS gates, Problem construction and marginalization.
struct SolveBudget {
    std::optional<double> timeSeconds;
    std::optional<int> maxIterations,minIterations,numThreads;
    bool collectCameraLandmarkCounts=false;
    bool cameraRotationOnly=false; ///< Keep the physical rig translation fixed during this solve.
    bool skipPreNis=false; ///< The post-solve gate still runs before accepting or marginalizing.
    bool latestTwoPoseVelocityOnly=false; ///< Refine the two newest poses and velocities before evicting the penultimate state.
    bool gateLightweightNewImuNis=false; ///< Apply the newest-edge NIS limit in a two-frame solve.
    bool allowOnlineSchur=true; ///< Retries can force the original Ceres solve.
    // Return a reason for a candidate calibration block outside its physical range.
    std::function<std::string(const ParameterBlock&)> calibrationBlockViolation;
};

/// Convert owned pose coefficients back into the project's cached Transformation type.
/// Construct a value rather than exposing writable cached Transformation coefficients.
kinematics::Transformation blockPose(const ParameterBlock& p);

} }
