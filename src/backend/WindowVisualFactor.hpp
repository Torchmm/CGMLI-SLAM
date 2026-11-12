#pragma once
#include "FrameStore.hpp"
#include "LandmarkStore.hpp"
#include "VisualCudaBatch.hpp"
#include <ceres/evaluation_callback.h>
#include <array>
#include <cstdint>
#include <tuple>
#include <map>

namespace cgmli { namespace backend {

/// Deterministic orthonormal tangent basis, valid over the full unit sphere.
Eigen::Matrix<double,3,2> sphereBasis(const Eigen::Vector3d& bearing);
Eigen::Vector3d timeBearing(const Eigen::Vector3d& bearing,const Eigen::Vector3d& velocity,double dt);
struct VisualPoseCacheValue {
    Eigen::Matrix3d rotation=Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation=Eigen::Vector3d::Zero();
    Eigen::Matrix<double,3,4> quaternionLift=Eigen::Matrix<double,3,4>::Zero();
    bool valid=false,unitQuaternion=false;
};
struct VisualCameraCacheValue {
    const VisualPoseCacheValue *body=nullptr,*extrinsic=nullptr;
    Eigen::Matrix3d rigRotation,worldRotation,worldInverse;
    Eigen::Vector3d rigTranslation,positionOffset;
    bool valid=false;
};
struct VisualPairCacheValue {
    const VisualPoseCacheValue *anchor=nullptr,*observer=nullptr,*extrinsic=nullptr;
    const VisualCameraCacheValue *anchorCamera=nullptr,*observerCamera=nullptr;
    Eigen::Matrix3d Rca,Rcb,W,cameraRotation;
    Eigen::Vector3d tca,tcb,offset;
    bool valid=false;
};
struct VisualTimedBearing {
    Eigen::Vector3d direction=Eigen::Vector3d::Zero(),derivative=Eigen::Vector3d::Zero();
    double td=0;
    bool ready=false,derivativeReady=false;
};
using VisualProjection=Eigen::Matrix<double,2,3>;
struct VisualTimedObserver {
    VisualTimedBearing bearing;
    VisualProjection projection=VisualProjection::Zero();
    bool valid=false;
};
/// Solver-local cache. Ceres updates user parameter buffers before this callback;
/// costs receive stable bound entries, never assume their x pointers equal user
/// pointers. Trial points/refits refresh the same immutable worker read view.
class VisualEvaluationCache final : public ::ceres::EvaluationCallback {
public:
    VisualEvaluationCache();
    ~VisualEvaluationCache() override;
    void configureBatch(VisualBatchMode,const std::shared_ptr<detail::VisualCudaWorkspace>& workspace={});
    void setBatchMode(VisualBatchMode);
    bool hasBatch() const { return bool(batch_); }
    std::size_t bindBatch(detail::VisualBatchInput,const std::vector<const double*>&);
    const detail::VisualBatchResidual* batchResult(std::size_t,const double** jacobians) const;
    std::size_t batchFactors() const;
    bool auditDirectNormal(const std::vector<detail::DirectVisualLayout>&,const std::vector<double>& plus,
        const std::vector<int>& retained,const std::vector<int>& eliminated,detail::VisualNormalSystem&) const;
    detail::VisualCudaBatch* linearizeSchur(const std::vector<detail::DirectVisualLayout>&,
        const std::vector<double>& plus,int retained,int ranges,bool newTopology);
    int bindSchurParameter(const double*,int size);
    bool schurTrialCost(double&);
    int batchEvaluations=0,batchFailures=0,batchInvalid=0,batchCpuTrials=0;
    std::size_t batchJacobianBytes=0,batchDenseJacobianBytes=0;
    double batchMs=0;
    const VisualPoseCacheValue* bind(const double* parameter);
    const VisualPairCacheValue* bindPair(const double* anchor,const double* observer,const double* extrinsic,
        const kinematics::Transformation& T_RCa,const kinematics::Transformation& T_RCb);
    const VisualTimedBearing* bindTimedAnchor(const double* td,const Eigen::Vector3d& bearing,
        const WindowObservation& observation,bool fixedTime);
    void PrepareForEvaluation(bool evaluateJacobians,bool newPoint) override;
    double posePrepareMs=0,cameraPrepareMs=0,pairPrepareMs=0,timedAnchorPrepareMs=0;
    std::size_t sharedTimedAnchorCount() const { return sharedTimedAnchors_.size(); }
    std::size_t cameraCount() const { return cameras_.size(); }
    std::size_t pairCount() const { return pairs_.size(); }
    std::size_t refreshCount() const { return refreshCount_; }
private:
    struct Batch;
    std::unique_ptr<Batch> batch_;
    std::map<const double*,VisualPoseCacheValue> poses_;
    // Include fixed per-frame rig transforms, not just camera indices: calibration
    // snapshots may differ. Entries remain stable while costs bind their addresses.
    using PairKey=std::tuple<std::uintptr_t,std::uintptr_t,std::uintptr_t,std::array<double,24>>;
    std::map<PairKey,VisualPairCacheValue> pairs_;
    using CameraKey=std::tuple<std::uintptr_t,std::uintptr_t,std::array<double,12>>;
    std::map<CameraKey,VisualCameraCacheValue> cameras_;
    using TimedAnchorKey=std::pair<std::uintptr_t,std::array<double,7>>;
    struct TimedAnchorBinding {
        const double* td=nullptr;
        Eigen::Vector3d bearing,velocity;
        double reference=0;
        unsigned uses=0;
        VisualTimedBearing value;
    };
    std::map<TimedAnchorKey,TimedAnchorBinding> timedAnchors_;
    std::vector<TimedAnchorBinding*> sharedTimedAnchors_;
    const VisualCameraCacheValue* bindCamera(const double*,const double*,const kinematics::Transformation&);
    bool ready_=false,derivativesReady_=false;
    std::size_t refreshCount_=0;
};
std::unique_ptr<::ceres::CostFunction> cacheWindowVisualCost(const ::ceres::CostFunction&,
    const std::vector<const double*>& parameters,VisualEvaluationCache&,bool cachePairs=true,
    bool fixedTime=false,unsigned derivativeMask=~0u);

/// CPU reference S^2 factor used by the managed observation API. Captures measurement
/// values and fixed rig baselines, never mutable Frame/Store pointers. Same-epoch
/// cameras share one pose block. Analytic tangent derivatives are lifted to Ceres
/// ambient coordinates with the project's left SE(3) convention.
/// BackendOptions::analyticVisualJacobians=false retains central differences for comparison.
bool makeWindowVisualFactor(const WindowLandmark& landmark,const WindowObservation& anchor,
    const WindowObservation& observation,const FrameStore& frames,const BackendGraph& graph,
    const BackendOptions& options,FactorRecord& factor,std::string& error);

/// Local information audit, not an implementation of prior/chart transport. Both
/// systems use the union of tangent keys (including frozen calibration and prior).
/// delta_old = oldFromNew * delta_new. A nominally identical point is insufficient:
/// its derivative, H and b must agree in that SAME physical perturbation chart.
struct ReanchorInformationReport {
    bool equivalent=false;
    double pointError=0,pointDerivativeError=0,informationError=0,gradientError=0;
    std::vector<BlockKey> keys;
    std::vector<int> offsets,sizes;
    Eigen::MatrixXd oldPointJacobian,newPointJacobian,oldFromNew,oldInformation,newInformation;
    Eigen::VectorXd oldGradient,newGradient;
};
/// Returns false on an invalid audit/evaluation, true with equivalent=false for a
/// successfully measured mismatch. Does not mutate graphs, frames or observations.
/// Includes incident active measurements and each graph's historical prior ONCE,
/// with the same whitening/loss convention as actual Schur marginalization.
bool checkReanchorInformation(const WindowLandmark& oldLandmark,const WindowObservation& oldAnchor,
    const WindowLandmark& newLandmark,const WindowObservation& newAnchor,const FrameStore& frames,
    const BackendGraph& before,const BackendGraph& after,const BackendOptions& options,
    ReanchorInformationReport& report,std::string& error);

} }
