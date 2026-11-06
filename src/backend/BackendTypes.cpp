#include "BackendTypes.hpp"
#include "MarginalizationPrior.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace cgmli { namespace backend {
const char* solveTerminationName(SolveTermination reason) {
    switch(reason) {
    case SolveTermination::SolverFinished: return "solver_finished";
    case SolveTermination::TimeBudget: return "time_budget";
    case SolveTermination::IterationBudget: return "iteration_budget";
    case SolveTermination::RefitBudget: return "refit_budget";
    default: return "failure";
    }
}
namespace {
bool fail(std::string* why, const char* text) { if (why) *why = text; return false; }
}
bool ParameterBlock::valid(std::string* why) const {
    // Validate before handing raw buffers to Ceres, whose parameter-size/duplicate-block
    // checks can be fatal rather than recoverable errors. Layout and ownership are part
    // of the graph contract, not merely optimizer implementation details.
    if (value.size() <= 0 || !value.allFinite() || tangentSize() <= 0)
        return fail(why,"empty/nonfinite parameter block");
    if (manifold && (manifold->AmbientSize() != value.size() || manifold->TangentSize() > value.size()))
        return fail(why,"manifold size mismatch");
    if (key.isCalibration() && (owner || key.id != 0)) return fail(why,"calibration must be global, id zero");
    if (owner && !owner->isInitialised()) return fail(why,"uninitialized owner");
    const bool poseType = key.kind == BlockKind::Pose || key.kind == BlockKind::CameraExtrinsic || key.kind == BlockKind::LidarExtrinsic;
    // All project poses/extrinsics share one perturbation convention. A generic product
    // manifold with matching dimensions would still have different translation semantics.
    if (poseType && (value.size() != 7 || !dynamic_cast<LeftSE3Manifold*>(manifold.get())
        || std::abs(value.tail<4>().squaredNorm()-1.0) > 1e-6)) return fail(why,"pose must use unit xyzw quaternion and LeftSE3Manifold");
    if (key.isState() && (!owner || owner->value() != key.id)) return fail(why,"state owner mismatch");
    if (key.kind == BlockKind::SpeedBias && (value.size() != 9 || manifold)) return fail(why,"speed/bias must be Euclidean 9D");
    if (dynamic_cast<LeftSO3Manifold*>(manifold.get()) && std::abs(value.squaredNorm()-1.0)>1e-6)
        return fail(why,"SO3 block must contain a unit xyzw quaternion");
    if (key.kind == BlockKind::Landmark && !key.id) return fail(why,"landmark ID must be initialized");
    if ((key.kind == BlockKind::CameraTimeOffset || key.kind == BlockKind::LidarTimeOffset)
        && (value.size() != 1 || manifold)) return fail(why,"time offset must be Euclidean 1D");
    if ((lower.size() && (lower.size()!=value.size() || !lower.allFinite()))
        || (upper.size() && (upper.size()!=value.size() || !upper.allFinite()))) return fail(why,"invalid parameter bounds");
    if ((lower.size() && (value.array()<lower.array()).any())
        || (upper.size() && (value.array()>upper.array()).any())) return fail(why,"value outside bounds");
    // Bounding quaternion coefficients does not bound a physical rotation step. Pose
    // calibration limits are checked after solving using translation and rotation angle.
    if (poseType && (lower.size() || upper.size())) return fail(why,"pose ambient bounds are unsupported; use a physical pose validator");
    return true;
}
ParameterBlock ParameterBlock::pose(BlockKey k, const kinematics::Transformation& T, std::optional<StateId> owner) {
    // Never optimize the cached Transformation object itself: copy its coefficients.
    ParameterBlock p; p.key=k; p.value=T.coeffs(); p.owner=owner;
    p.manifold=std::make_shared<LeftSE3Manifold>(); return p;
}
ParameterBlock ParameterBlock::euclidean(BlockKey k, const Eigen::VectorXd& v, std::optional<StateId> owner) {
    ParameterBlock p; p.key=k; p.value=v; p.owner=owner; return p;
}
kinematics::Transformation blockPose(const ParameterBlock& p) {
    // Eigen's Quaternion constructor takes (w,x,y,z), whereas the buffer stores (x,y,z,w).
    if (p.value.size()!=7) throw std::invalid_argument("pose block size must be seven");
    return kinematics::Transformation(p.value.head<3>(), Eigen::Quaterniond(p.value[6],p.value[3],p.value[4],p.value[5]));
}
bool BackendGraph::valid(std::string* why) const {
    // Check shared invariants independently of a particular window selection. Pending
    // factors may lack buffers; active factors must have every ordered dependency.
    for (const auto& item : blocks) {
        if (item.first != item.second.key || !item.second.valid(why)) return fail(why,"invalid graph parameter block");
    }
    for (const auto& item : factors) {
        const auto& f=item.second;
        if (!f.id || item.first!=f.id || !f.cost || f.cost->num_residuals()<=0
            || f.keys.size()!=f.cost->parameter_block_sizes().size()
            || !(f.nisThreshold>0.0) || f.kind==FactorKind::Marginalization)
            return fail(why,"invalid factor metadata");
        if(f.imuReplay && (f.kind!=FactorKind::Imu || !f.imuReplay->parameters || f.imuReplay->samples.size()<2
            || f.imuReplay->end<=f.imuReplay->start
            || f.cost->num_residuals()!=15 || f.cost->parameter_block_sizes()!=std::vector<int32_t>{7,9,7,9}))
            return fail(why,"invalid native IMU replay metadata");
        if(!f.solveEnabled && !f.imuReplay) return fail(why,"only native IMU intervals may be suppressed");
        if(f.imuReplay && !f.imuReplay->completeSamples && f.solveEnabled)
            return fail(why,"incomplete IMU interval may not enter numerical computation");
        for (size_t i=0;i<f.keys.size();++i) {
            // Sensor costs contain only a handful of blocks. Avoid allocating a
            // tree node per key during every whole-window validation.
            if (std::find(f.keys.begin(),f.keys.begin()+i,f.keys[i])!=f.keys.begin()+i)
                return fail(why,"duplicate parameter in a residual; coalesce it in factor factory");
            auto b=blocks.find(f.keys[i]);
            if (b==blocks.end()) { if (f.lifecycle==FactorLifecycle::Active) return fail(why,"active factor has a missing parameter"); }
            else if (b->second.value.size()!=f.cost->parameter_block_sizes()[i]) return fail(why,"factor parameter size mismatch");
        }
        // Optimizer may leave terminal records until the window owner archives them after acceptance.
    }
    if (prior) {
        // A frozen prior's column meanings and dimensions cannot be changed by resizing
        // a surviving block or recycling a key after an epoch has left the window.
        for (size_t i=0;i<prior->keys().size();++i) {
            auto p=blocks.find(prior->keys()[i]);
            if (p==blocks.end() || p->second.value.size()!=prior->parameter_block_sizes()[i]
                || p->second.tangentSize()!=prior->tangentSizes()[i]) return fail(why,"prior parameter missing or resized");
        }
    }
    return true;
}
void BackendOptions::validate() const {
    // Invalid thresholds/budgets would otherwise surface inside eigensolvers or Ceres.
    if (windowSize<2 || !maxPendingFactors || !historySize || !maxLandmarks || !maxObservations || maxImuSamplesPerEpoch<4
        || !std::isfinite(minInverseDepth) || minInverseDepth<=0
        || !std::isfinite(maxImuIntervalSeconds) || maxImuIntervalSeconds<=0
        || !std::isfinite(minReanchorParallax) || minReanchorParallax<=0
        || !std::isfinite(maxReanchorAngularError) || maxReanchorAngularError<=0 || maxReanchorAngularError>=1
        || !(visualNisThreshold>0) || !std::isfinite(visualHuberScale) || visualHuberScale<0
        || maxIterations<1 || numThreads<1 || maxOutlierRefits<0
        || !std::isfinite(maxSolverTimeSeconds) || maxSolverTimeSeconds<=0
        || !std::isfinite(relativeEigenThreshold) || relativeEigenThreshold<=0 || relativeEigenThreshold>=1
        || !std::isfinite(absoluteEigenThreshold) || absoluteEigenThreshold<=0
        || !std::isfinite(maxRelativeCostIncrease) || maxRelativeCostIncrease<0
        || (linearSolver!=BackendLinearSolver::SparseNormalCholesky && linearSolver!=BackendLinearSolver::DenseQr
            && linearSolver!=BackendLinearSolver::DenseSchur))
        throw std::invalid_argument("invalid backend options");
}
} }
