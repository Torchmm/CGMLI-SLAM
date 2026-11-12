#include "WindowVisualFactor.hpp"
#include "VisualSphereBatchMath.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace cgmli { namespace backend {
struct VisualEvaluationCache::Batch {
    VisualBatchMode mode=VisualBatchMode::Reference;
    std::vector<detail::VisualBatchInput> inputs;
    std::map<const double*,int> offsets;
    std::vector<std::pair<const double*,int>> parameters;
    std::vector<double> values;
    std::vector<detail::VisualBatchResidual> residuals;
    std::vector<double> jacobians;
    std::size_t jacobianSize=0;
#ifdef CGMLI_ENABLE_CUDA_ORB
    std::shared_ptr<detail::VisualCudaWorkspace> workspace;
    std::unique_lock<std::mutex> lease;
    std::unique_ptr<detail::VisualCudaBatch> localGpu;
    detail::VisualCudaBatch* gpu=nullptr;
    detail::VisualBatchView gpuResults;
#endif
    bool topologyChanged=true,ready=false,derivatives=false;
};

VisualEvaluationCache::VisualEvaluationCache()=default;
VisualEvaluationCache::~VisualEvaluationCache()=default;
void VisualEvaluationCache::configureBatch(VisualBatchMode mode,const std::shared_ptr<detail::VisualCudaWorkspace>& workspace) {
    if(mode==VisualBatchMode::Reference) return;
    batch_=std::make_unique<Batch>();
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(workspace) {
        batch_->workspace=workspace;
        batch_->lease=std::unique_lock<std::mutex>(workspace->mutex,std::try_to_lock);
    }
#else
    (void)workspace;
#endif
    setBatchMode(mode==VisualBatchMode::Audit?VisualBatchMode::Reference:mode);
}
void VisualEvaluationCache::setBatchMode(VisualBatchMode mode) {
    if(batch_ && batch_->mode!=mode) { batch_->mode=mode; batch_->ready=false; }
}
std::size_t VisualEvaluationCache::bindBatch(detail::VisualBatchInput input,const std::vector<const double*>& parameters) {
    if(!batch_ || parameters.size()!=std::size_t(input.blocks)) return std::size_t(-1);
    input.jacobianOffset=int(batch_->jacobianSize);
    for(int i=0;i<input.blocks;++i) {
        const auto inserted=batch_->offsets.try_emplace(parameters[i],int(batch_->values.size()));
        if(inserted.second) {
            batch_->parameters.emplace_back(parameters[i],input.sizes[i]);
            batch_->values.resize(batch_->values.size()+input.sizes[i]);
        }
        input.offsets[i]=inserted.first->second;
        if(input.derivativeMask & (1u<<i)) input.jacobianSize+=2*input.sizes[i];
    }
    batch_->jacobianSize+=input.jacobianSize;
    const auto index=batch_->inputs.size();
    batch_->inputs.push_back(input);
    batch_->topologyChanged=true; batch_->ready=false;
    return index;
}
const detail::VisualBatchResidual* VisualEvaluationCache::batchResult(std::size_t index,
    const double** jacobians) const {
    if(!batch_ || !batch_->ready || index>=batch_->inputs.size()) return nullptr;
    if(jacobians && !batch_->derivatives) return nullptr;
    const auto* rows=batch_->residuals.data();
    const auto* values=batch_->jacobians.data();
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(batch_->mode!=VisualBatchMode::Host) {
        rows=batch_->gpuResults.residuals;
        values=batch_->gpuResults.jacobians;
    }
#endif
    if(jacobians) *jacobians=batch_->inputs[index].jacobianSize?
        values+batch_->inputs[index].jacobianOffset:nullptr;
    return &rows[index];
}
std::size_t VisualEvaluationCache::batchFactors() const { return batch_?batch_->inputs.size():0; }
bool VisualEvaluationCache::auditDirectNormal(const std::vector<detail::DirectVisualLayout>& layout,
    const std::vector<double>& plus,const std::vector<int>& retained,const std::vector<int>& eliminated,
    detail::VisualNormalSystem& out) const {
#ifdef CGMLI_ENABLE_CUDA_ORB
    return batch_ && batch_->mode==VisualBatchMode::Cuda && batch_->ready && batch_->derivatives && batch_->gpu &&
        batch_->gpu->directNormal(batch_->inputs,layout,plus,retained,eliminated,out);
#else
    (void)layout; (void)plus; (void)retained; (void)eliminated; (void)out;
    return false;
#endif
}

detail::VisualCudaBatch* VisualEvaluationCache::linearizeSchur(const std::vector<detail::DirectVisualLayout>& layout,
    const std::vector<double>& plus,int retained,int ranges,bool newTopology) {
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(!batch_ || batch_->inputs.size()!=layout.size()) return nullptr;
    auto& batch=*batch_;
    if(!batch.gpu) {
        auto& storage=batch.lease.owns_lock()?batch.workspace->batch:batch.localGpu;
        if(!storage) storage=std::make_unique<detail::VisualCudaBatch>();
        batch.gpu=storage.get();
    }
    std::size_t offset=0;
    for(const auto& p:batch.parameters) {
        std::copy_n(p.first,p.second,batch.values.data()+offset);
        offset+=p.second;
    }
    // Direct evaluation owns its device data. Force Ceres to repack if the
    // online solver falls back, including after an incomplete CUDA upload.
    batch.topologyChanged=true;
    batch.ready=false;
    return batch.gpu->linearizeSchur(batch.inputs,batch.values,batch.jacobianSize,layout,plus,retained,
        ranges,newTopology)?batch.gpu:nullptr;
#else
    (void)layout; (void)plus; (void)retained; (void)ranges; (void)newTopology;
    return nullptr;
#endif
}

int VisualEvaluationCache::bindSchurParameter(const double* value,int size) {
    const auto slot=batch_->offsets.try_emplace(value,int(batch_->values.size()));
    if(slot.second) {
        batch_->parameters.emplace_back(value,size);
        batch_->values.resize(batch_->values.size()+size);
    }
    return slot.first->second;
}

bool VisualEvaluationCache::schurTrialCost(double& cost) {
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(!batch_ || !batch_->gpu) return false;
    auto& batch=*batch_;
    std::size_t offset=0;
    for(const auto& parameter:batch.parameters) {
        std::copy_n(parameter.first,parameter.second,batch.values.data()+offset);
        offset+=parameter.second;
    }
    return batch.gpu->trialCost(batch.values,cost);
#else
    (void)cost;
    return false;
#endif
}

namespace {
VisualTimedBearing evaluateTimedBearing(const Eigen::Vector3d& bearing,const Eigen::Vector3d& velocity,
    double reference,double td,bool derivative) {
    VisualTimedBearing result;
    result.td=td;
    result.ready=true;
    result.derivativeReady=derivative;
    const double dt=reference-td;
    if (!derivative) {
        result.direction=timeBearing(bearing,velocity,dt);
        return result;
    }
    const double speed=velocity.norm(),angle=speed*dt;
    if (std::abs(angle)<1e-8) {
        const Eigen::Vector3d raw=bearing+velocity*dt;
        const double length=raw.norm();
        result.direction=raw/length;
        result.derivative=-(Eigen::Matrix3d::Identity()-result.direction*result.direction.transpose())*velocity/length;
    } else {
        const double sine=std::sin(angle),cosine=std::cos(angle);
        result.direction=cosine*bearing+(sine/speed)*velocity;
        result.derivative=speed*sine*bearing-cosine*velocity;
    }
    return result;
}
}
const VisualTimedBearing* VisualEvaluationCache::bindTimedAnchor(const double* td,const Eigen::Vector3d& bearing,
    const WindowObservation& observation,bool fixedTime) {
    const auto& v=observation.bearingVelocity;
    if (!bearing.allFinite() || !v.allFinite() || !std::isfinite(observation.referenceTimeOffset)) return nullptr;
    const TimedAnchorKey key{reinterpret_cast<std::uintptr_t>(td),
        {bearing.x(),bearing.y(),bearing.z(),v.x(),v.y(),v.z(),observation.referenceTimeOffset}};
    auto& entry=timedAnchors_[key];
    if (++entry.uses==1) {
        entry.td=td; entry.bearing=bearing; entry.velocity=v; entry.reference=observation.referenceTimeOffset;
        if (fixedTime) entry.value=evaluateTimedBearing(bearing,v,entry.reference,*td,false);
    }
    if (entry.uses==2 && !fixedTime) sharedTimedAnchors_.push_back(&entry);
    return &entry.value;
}
const VisualPoseCacheValue* VisualEvaluationCache::bind(const double* parameter) {
    ready_=false; return &poses_.try_emplace(parameter).first->second;
}
const VisualCameraCacheValue* VisualEvaluationCache::bindCamera(const double* body,const double* extrinsic,
    const kinematics::Transformation& rig) {
    std::array<double,12> values; size_t k=0;
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) values[k++]=rig.C()(r,c);
    for(int r=0;r<3;++r) values[k++]=rig.r()[r];
    const CameraKey key{reinterpret_cast<std::uintptr_t>(body),reinterpret_cast<std::uintptr_t>(extrinsic),values};
    auto& value=cameras_.try_emplace(key).first->second;
    value.body=bind(body); value.extrinsic=bind(extrinsic);
    value.rigRotation=rig.C(); value.rigTranslation=rig.r(); return &value;
}
const VisualPairCacheValue* VisualEvaluationCache::bindPair(const double* a,const double* b,const double* e,
    const kinematics::Transformation& ca,const kinematics::Transformation& cb) {
    std::array<double,24> rig; std::size_t k=0;
    for(const auto* camera:{&ca,&cb}) {
        for(int row=0;row<3;++row) for(int col=0;col<3;++col) rig[k++]=camera->C()(row,col);
        for(int row=0;row<3;++row) rig[k++]=camera->r()[row];
    }
    const PairKey key{reinterpret_cast<std::uintptr_t>(a),reinterpret_cast<std::uintptr_t>(b),reinterpret_cast<std::uintptr_t>(e),rig};
    auto& pair=pairs_.try_emplace(key).first->second;
    pair.anchor=bind(a); pair.observer=bind(b); pair.extrinsic=bind(e);
    pair.anchorCamera=bindCamera(a,e,ca); pair.observerCamera=bindCamera(b,e,cb);
    pair.Rca=ca.C(); pair.Rcb=cb.C(); pair.tca=ca.r(); pair.tcb=cb.r();
    return &pair;
}
void VisualEvaluationCache::PrepareForEvaluation(bool evaluateJacobians,bool newPoint) {
    const auto ms=[](auto start) { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); };
    if(!ready_ || newPoint) {
        ++refreshCount_; derivativesReady_=false;
        auto start=std::chrono::steady_clock::now();
        for(auto& entry:poses_) {
            auto& pose=entry.second; const double* x=entry.first;
            const Eigen::Map<const Eigen::Quaterniond> q(x+3);
            pose.translation=Eigen::Map<const Eigen::Vector3d>(x);
            pose.valid=pose.translation.allFinite() && q.coeffs().allFinite() && q.squaredNorm()>1e-16;
            pose.unitQuaternion=pose.valid && std::abs(q.squaredNorm()-1)<=1e-6;
            if(pose.valid) pose.rotation=q.normalized().toRotationMatrix();
        }
        posePrepareMs+=ms(start); start=std::chrono::steady_clock::now();
        // Share each state/camera chain across all anchor-observer pairs.
        for(auto& entry:cameras_) {
            auto& c=entry.second; const auto& body=*c.body; const auto& e=*c.extrinsic;
            c.valid=body.valid && e.valid;
            if(!c.valid) continue;
            c.worldRotation=body.rotation*e.rotation*c.rigRotation;
            c.worldInverse=c.worldRotation.transpose();
            c.positionOffset=body.rotation*(e.rotation*c.rigTranslation+e.translation);
        }
        cameraPrepareMs+=ms(start); start=std::chrono::steady_clock::now();
        for(auto& entry:pairs_) {
            auto& p=entry.second; const auto& a=*p.anchorCamera; const auto& b=*p.observerCamera;
            p.valid=a.valid && b.valid; if(!p.valid) continue;
            p.W=b.worldInverse;
            p.cameraRotation=p.W*a.worldRotation;
            const Eigen::Vector3d separation=p.anchor->translation-p.observer->translation;
            p.offset=p.W*(a.positionOffset-b.positionOffset+separation);
        }
        pairPrepareMs+=ms(start); ready_=true;
    }
    // Ceres can request residuals first, then Jacobians at the SAME point.
    if(evaluateJacobians && !derivativesReady_) {
        const auto start=std::chrono::steady_clock::now();
        for(auto& entry:poses_) {
            auto& pose=entry.second; if(!pose.valid) continue;
            const Eigen::Map<const Eigen::Quaterniond> q(entry.first+3);
            pose.quaternionLift.leftCols<3>()=2*(q.w()*Eigen::Matrix3d::Identity()+skew(q.vec()));
            pose.quaternionLift.rightCols<1>()=-2*q.vec();
        }
        posePrepareMs+=ms(start); derivativesReady_=true;
    }
    if (!sharedTimedAnchors_.empty()) {
        const auto start=std::chrono::steady_clock::now();
        for (auto* entry:sharedTimedAnchors_) {
            const double td=*entry->td;
            const auto& value=entry->value;
            // Changes in pose alone leave time compensation valid. A request
            // for derivatives at the same point may still need to promote it.
            if (!value.ready || value.td!=td || (evaluateJacobians && !value.derivativeReady))
                entry->value=evaluateTimedBearing(entry->bearing,entry->velocity,entry->reference,td,evaluateJacobians);
        }
        timedAnchorPrepareMs+=ms(start);
    }
    if(!batch_ || batch_->inputs.empty() || batch_->mode==VisualBatchMode::Reference) return;
    auto& batch=*batch_;
    if(batch.ready && !newPoint && (!evaluateJacobians || batch.derivatives)) return;
    if(!evaluateJacobians && (batch.mode==VisualBatchMode::Hybrid || batch.mode==VisualBatchMode::Schur)) {
        // A trial point only needs the cost. Let Ceres evaluate its cached
        // CPU factors in parallel and invalidate any rows for the old point.
        batch.ready=false;
        ++batchCpuTrials;
        return;
    }
    const auto start=std::chrono::steady_clock::now();
    std::size_t offset=0;
    for(const auto& p:batch.parameters) {
        std::copy_n(p.first,p.second,batch.values.data()+offset);
        offset+=p.second;
    }
    batch.derivatives=evaluateJacobians;
    batch.ready=false;
    const detail::VisualBatchResidual* rows=nullptr;
    ++batchEvaluations;
    if(batch.mode==VisualBatchMode::Host) {
        batch.residuals.resize(batch.inputs.size());
        if(evaluateJacobians) batch.jacobians.resize(batch.jacobianSize);
        for(std::size_t i=0;i<batch.inputs.size();++i) {
            auto& r=batch.residuals[i];
            const auto& in=batch.inputs[i];
            r.valid=detail::sphere_batch::evaluate(in,batch.values.data(),r.value,
                evaluateJacobians && in.jacobianSize?batch.jacobians.data()+in.jacobianOffset:nullptr);
        }
        batch.ready=true;
        rows=batch.residuals.data();
    }
#ifdef CGMLI_ENABLE_CUDA_ORB
    else {
        if(!batch.gpu) {
            auto& storage=batch.lease.owns_lock()?batch.workspace->batch:batch.localGpu;
            if(!storage) storage=std::make_unique<detail::VisualCudaBatch>();
            batch.gpu=storage.get();
        }
        batch.ready=batch.gpu->evaluate(batch.inputs,batch.values,batch.topologyChanged,evaluateJacobians,batch.jacobianSize,
            batch.gpuResults);
        if(batch.ready) {
            rows=batch.gpuResults.residuals;
            batch.topologyChanged=false;
            if(evaluateJacobians) {
                batchJacobianBytes+=batch.jacobianSize*sizeof(double);
                batchDenseJacobianBytes+=batch.inputs.size()*46*sizeof(double);
            }
        }
    }
#endif
    if(!batch.ready) { ++batchFailures; batch.mode=VisualBatchMode::Reference; }
    else batchInvalid+=int(std::count_if(rows,rows+batch.inputs.size(),
        [](const auto& r){return !r.valid;}));
    batchMs+=ms(start);
}
Eigen::Matrix<double,3,2> sphereBasis(const Eigen::Vector3d& b) {
    const Eigen::Vector3d axis=std::abs(b.x())<0.8?Eigen::Vector3d::UnitX():Eigen::Vector3d::UnitY();
    Eigen::Matrix<double,3,2> basis;
    basis.col(0)=(axis-b*b.dot(axis)).normalized(); basis.col(1)=b.cross(basis.col(0)); return basis;
}
Eigen::Vector3d timeBearing(const Eigen::Vector3d& b,const Eigen::Vector3d& velocity,double dt) {
    const Eigen::Vector3d u=velocity*dt; const double a=u.norm();
    if(a<1e-8) return (b+u).normalized();
    return std::cos(a)*b+(std::sin(a)/a)*u;
}
namespace {
kinematics::Transformation pose(const double* x) {
    return kinematics::Transformation(Eigen::Map<const Eigen::Vector3d>(x),Eigen::Map<const Eigen::Quaterniond>(x+3));
}
class CacheableSphereCost : public ::ceres::CostFunction {
public:
    virtual bool packBatch(detail::VisualBatchInput&,unsigned) const=0;
    virtual bool EvaluateCached(double const* const*,double*,double**,
        const std::vector<const VisualPoseCacheValue*>*,const VisualPairCacheValue* = nullptr,
        const VisualTimedBearing* = nullptr,const VisualTimedObserver* = nullptr) const=0;
    virtual const VisualPairCacheValue* bindPair(const std::vector<const double*>&,VisualEvaluationCache&) const=0;
    virtual const VisualTimedBearing* bindTimedAnchor(const std::vector<const double*>&,VisualEvaluationCache&,bool) const=0;
    virtual std::optional<VisualTimedObserver> fixedObserver(const std::vector<const double*>&) const=0;
};
class InverseDepthSphereCost final : public CacheableSphereCost {
public:
    bool packBatch(detail::VisualBatchInput& in,unsigned derivativeMask) const override {
        if(!analytic_) return false;
        in.blocks=int(parameter_block_sizes().size()); in.observer=observerPose_;
        in.extrinsic=extrinsic_; in.time=timeOffset_; in.range=inverseDepth_;
        in.derivativeMask=derivativeMask;
        for(int i=0;i<in.blocks;++i) in.sizes[i]=parameter_block_sizes()[i];
        const auto vector=[](const Eigen::Vector3d& value,double* output) {
            Eigen::Map<Eigen::Vector3d> mapped(output);
            mapped=value;
        };
        vector(anchorBearing_,in.anchor); vector(observation_.bearing,in.measured);
        vector(anchorStatic_,in.anchorStatic); vector(observationStatic_,in.measuredStatic);
        vector(anchor_.bearingVelocity,in.anchorVelocity); vector(observation_.bearingVelocity,in.measuredVelocity);
        vector(tca_,in.anchorTranslation); vector(tcb_,in.measuredTranslation);
        Eigen::Map<detail::sphere_batch::R3> ca(in.anchorRotation),cb(in.measuredRotation);
        Eigen::Map<detail::sphere_batch::R23> projection(in.measurement);
        ca=Rca_; cb=Rcb_; projection=measurement_;
        in.anchorTime=anchor_.referenceTimeOffset; in.measuredTime=observation_.referenceTimeOffset;
        in.movingAnchor=movingAnchor_; in.movingMeasured=movingObservation_;
        return true;
    }
    const VisualPairCacheValue* bindPair(const std::vector<const double*>& p,VisualEvaluationCache& cache) const override {
        return observerPose_==0?nullptr:cache.bindPair(p[0],p[observerPose_],p[extrinsic_],T_RCa_,T_RCb_);
    }
    const VisualTimedBearing* bindTimedAnchor(const std::vector<const double*>& p,
        VisualEvaluationCache& cache,bool fixedTime) const override {
        return analytic_ && movingAnchor_ && timeOffset_>=0
            ? cache.bindTimedAnchor(p[timeOffset_],anchorBearing_,anchor_,fixedTime) : nullptr;
    }
    std::optional<VisualTimedObserver> fixedObserver(const std::vector<const double*>& p) const override {
        if (!analytic_ || !movingObservation_) return std::nullopt;
        const double td=timeOffset_<0?anchor_.referenceTimeOffset:p[timeOffset_][0];
        VisualTimedObserver value;
        value.bearing=evaluateTimedBearing(observation_.bearing,observation_.bearingVelocity,observation_.referenceTimeOffset,td,false);
        const auto& b=value.bearing.direction;
        const double denominator=1+observation_.bearing.dot(b);
        if (b.allFinite() && denominator>=1e-8) {
            const Eigen::Matrix3d K=skew(observation_.bearing.cross(b));
            const Eigen::Matrix3d transport=Eigen::Matrix3d::Identity()+K+K*K/denominator;
            value.projection=measurement_*transport.transpose();
            value.valid=value.projection.allFinite();
        }
        return value;
    }
    InverseDepthSphereCost(const WindowLandmark& landmark,const WindowObservation& anchor,
        const WindowObservation& observation,const kinematics::Transformation& T_RCa,
        const kinematics::Transformation& T_RCb,bool sameEpoch,bool withTd,bool analytic)
        : anchorBearing_(landmark.anchorBearing),anchor_(anchor),observation_(observation),
          T_RCa_(T_RCa),T_RCb_(T_RCb),basis_(sphereBasis(observation.bearing)),analytic_(analytic) {
        mutable_parameter_block_sizes()->push_back(7);
        observerPose_=sameEpoch?0:1;
        if(!sameEpoch) mutable_parameter_block_sizes()->push_back(7);
        extrinsic_=static_cast<int>(parameter_block_sizes().size()); mutable_parameter_block_sizes()->push_back(7);
        if(withTd) { timeOffset_=static_cast<int>(parameter_block_sizes().size()); mutable_parameter_block_sizes()->push_back(1); }
        inverseDepth_=static_cast<int>(parameter_block_sizes().size()); mutable_parameter_block_sizes()->push_back(1);
        Rca_=T_RCa.C(); Rcb_=T_RCb.C(); tca_=T_RCa.r(); tcb_=T_RCb.r();
        anchorStatic_=anchorBearing_.normalized(); observationStatic_=observation_.bearing.normalized();
        measurement_=observation_.sqrtInformation*basis_.transpose();
        movingAnchor_=anchor_.bearingVelocity.squaredNorm()>0; movingObservation_=observation_.bearingVelocity.squaredNorm()>0;
        set_num_residuals(2);
    }
    bool residual(double const* const* x,Eigen::Vector2d& out) const {
        const double rho=x[inverseDepth_][0],td=timeOffset_<0?anchor_.referenceTimeOffset:x[timeOffset_][0];
        if(!std::isfinite(rho) || rho<=0 || !std::isfinite(td)) return false;
        const Eigen::Vector3d ba=timeBearing(anchorBearing_,anchor_.bearingVelocity,anchor_.referenceTimeOffset-td);
        const Eigen::Vector3d bb=timeBearing(observation_.bearing,observation_.bearingVelocity,observation_.referenceTimeOffset-td);
        const auto T_WCa=pose(x[0])*pose(x[extrinsic_])*T_RCa_;
        const auto T_WCb=pose(x[observerPose_])*pose(x[extrinsic_])*T_RCb_;
        const Eigen::Vector3d pW=T_WCa.C()*(ba/rho)+T_WCa.r();
        const Eigen::Vector3d pC=T_WCb.C().transpose()*(pW-T_WCb.r());
        if(!pC.allFinite() || pC.norm()<1e-10 || !bb.allFinite()) return false;
        const Eigen::Vector3d predicted=pC.normalized();
        const double c=std::clamp(bb.dot(predicted),-1.0,1.0),s=bb.cross(predicted).norm();
        if(c<-1.0+1e-8 || observation_.bearing.dot(bb)<-1.0+1e-8) return false;
        const Eigen::Vector3d logarithm=s<1e-8?Eigen::Vector3d(predicted-c*bb)
            :Eigen::Vector3d((std::atan2(s,c)/s)*(predicted-c*bb));
        // Parallel-transport the raw measurement's basis to its time-corrected bearing.
        const Eigen::Matrix3d transport=Eigen::Quaterniond::FromTwoVectors(observation_.bearing,bb).toRotationMatrix();
        out=observation_.sqrtInformation*(transport*basis_).transpose()*logarithm;
        return out.allFinite();
    }
    bool Evaluate(double const* const* x,double* residuals,double** jacobians) const override {
        return EvaluateCached(x,residuals,jacobians,nullptr);
    }
    bool EvaluateCached(double const* const* x,double* residuals,double** jacobians,
        const std::vector<const VisualPoseCacheValue*>* cache,const VisualPairCacheValue* pair=nullptr,
        const VisualTimedBearing* anchorTime=nullptr,const VisualTimedObserver* observerTime=nullptr) const override {
        if(!analytic_) return evaluateNumerical(x,residuals,jacobians);
        if(cache) for(const auto* pose:*cache) if(pose && !pose->valid) return false;
        if(pair && !pair->valid) return false;
        bool derivatives=false;
        if(jacobians) for(size_t i=0;i<parameter_block_sizes().size();++i) derivatives=derivatives || jacobians[i]!=nullptr;
        // VINS' chain rule is useful, but Exp(delta)*T here also rotates the
        // current position. Derive in the LEFT tangent and lift to ambient 7D.
        const double rho=x[inverseDepth_][0],td=timeOffset_<0?anchor_.referenceTimeOffset:x[timeOffset_][0];
        if(!std::isfinite(rho) || rho<=0 || !std::isfinite(td)) return false;
        const bool timeDerivative=timeOffset_>=0 && jacobians && jacobians[timeOffset_];
        const bool cachedObserver=observerTime && observerTime->bearing.td==td && !timeDerivative;
        Eigen::Vector3d ba=anchorStatic_,bb=observationStatic_;
        Eigen::Vector3d da=Eigen::Vector3d::Zero(),db=Eigen::Vector3d::Zero();
        if (movingAnchor_) {
            const auto value=anchorTime && anchorTime->ready && anchorTime->td==td && (!timeDerivative || anchorTime->derivativeReady)
                ? *anchorTime : evaluateTimedBearing(anchorBearing_,anchor_.bearingVelocity,anchor_.referenceTimeOffset,td,timeDerivative);
            ba=value.direction; da=value.derivative;
        }
        if (movingObservation_) {
            if (cachedObserver) {
                if (!observerTime->valid) return false;
                bb=observerTime->bearing.direction;
            } else {
                const auto value=evaluateTimedBearing(observation_.bearing,observation_.bearingVelocity,
                    observation_.referenceTimeOffset,td,timeDerivative);
                bb=value.direction; db=value.derivative;
            }
        }
        if(!ba.allFinite() || !bb.allFinite()) return false;
        const Eigen::Vector3d local=Rca_*(ba/rho)+tca_;
        Eigen::Matrix3d Ra,Rb,Re,W,cameraRotation;
        Eigen::Vector3d body,observer,rotatedBody,relativeWorld,te,pc;
        if(observerPose_==0) {
            // Rig-only stereo: common T_WB and T_BR cancel algebraically. Avoid
            // constructing seven normalized/cached SE3 objects per observation.
            pc=Rcb_.transpose()*(local-tcb_);
            if(derivatives) cameraRotation=Rcb_.transpose()*Rca_;
        } else {
            // Residual-only requests need no per-landmark world/body chain.
            // Pair entries are immutable for the complete Ceres worker phase.
            const bool poseDerivatives=derivatives && (jacobians[0] || jacobians[observerPose_] || jacobians[extrinsic_]);
            if(!pair || poseDerivatives) {
                Ra=cache?(*cache)[0]->rotation:Eigen::Map<const Eigen::Quaterniond>(x[0]+3).normalized().toRotationMatrix();
                Rb=cache?(*cache)[observerPose_]->rotation:Eigen::Map<const Eigen::Quaterniond>(x[observerPose_]+3).normalized().toRotationMatrix();
                Re=cache?(*cache)[extrinsic_]->rotation:Eigen::Map<const Eigen::Quaterniond>(x[extrinsic_]+3).normalized().toRotationMatrix();
                te=cache?(*cache)[extrinsic_]->translation:Eigen::Vector3d(Eigen::Map<const Eigen::Vector3d>(x[extrinsic_]));
                body=Re*local+te; rotatedBody=Ra*body;
                const Eigen::Vector3d separation=Eigen::Map<const Eigen::Vector3d>(x[0])-Eigen::Map<const Eigen::Vector3d>(x[observerPose_]);
                relativeWorld=rotatedBody+separation;
                observer=Rb.transpose()*relativeWorld;
            }
            if(pair) pc=pair->cameraRotation*(ba/rho)+pair->offset;
            else pc=Rcb_.transpose()*(Re.transpose()*(observer-te)-tcb_);
            if(derivatives) {
                if(pair) { W=pair->W; cameraRotation=pair->cameraRotation; }
                else { W=Rcb_.transpose()*Re.transpose()*Rb.transpose(); cameraRotation=W*Ra*Re*Rca_; }
            }
        }
        const double range=pc.norm();
        if(!pc.allFinite() || !std::isfinite(range) || range<1e-10) return false;
        const Eigen::Vector3d n=pc/range;
        const double c=std::clamp(bb.dot(n),-1.0,1.0),s=bb.cross(n).norm();
        const double denominator=movingObservation_ && !cachedObserver?1+observation_.bearing.dot(bb):2;
        if(c<-1+1e-8 || denominator<1e-8) return false;
        const Eigen::Vector3d u=n-c*bb;
        const double d=1-c;
        const bool useSmallAngleSeries=c>0 && s<1e-3;
        // theta/sin(theta) and d()/dc share a consistent series. At this
        // cutoff the omitted d^4 term is below double rounding precision.
        const double theta=useSmallAngleSeries?0:std::atan2(s,c);
        const double f=s<1e-8?1.0:(useSmallAngleSeries?1+d/3+2*d*d/15+2*d*d*d/35:theta/s);
        const Eigen::Vector3d logarithm=f*u;
        // Shortest-arc transport R(b0 -> b). The raw measurement basis is
        // fixed, but its transport MUST also be differentiated for time offset.
        Eigen::Matrix3d K=Eigen::Matrix3d::Zero();
        VisualProjection P=cachedObserver?observerTime->projection:measurement_;
        if(movingObservation_ && !cachedObserver) {
            K=skew(observation_.bearing.cross(bb));
            const Eigen::Matrix3d transport=Eigen::Matrix3d::Identity()+K+K*K/denominator;
            P=measurement_*transport.transpose();
        }
        Eigen::Map<Eigen::Vector2d> output(residuals); output=P*logarithm;
        if(!output.allFinite()) return false;
        if(!derivatives) return true;
        // d(theta/s)/dc on unit spheres. Series prevents cancellation at zero;
        // below the ORIGINAL residual cutoff f=1 and its derivative is zero.
        double g=0;
        if(s>=1e-8) {
            g=useSmallAngleSeries?(-1.0/3.0-4*d/15.0-6*d*d/35.0):(theta*c-s)/(s*s*s);
        }
        // Project to the two measured tangent components BEFORE applying the
        // normalization derivative. Two rank-one updates replace chained 3x3
        // products; retain P*bb explicitly for moving bearings and roundoff.
        const Eigen::Matrix<double,2,3> projected=f*P+(g*(P*u)-f*(P*bb))*bb.transpose();
        const Eigen::Matrix<double,2,3> reduce=(projected-(projected*n)*n.transpose())/range;
        const auto writePose=[&](int block,const Eigen::Matrix<double,2,3>& translation,
            const Eigen::Matrix<double,2,3>& rotation) {
            if(!jacobians[block]) return true;
            Eigen::Matrix<double,3,4> qm;
            if(cache) {
                if(!(*cache)[block]->unitQuaternion) return false;
                qm=(*cache)[block]->quaternionLift;
            } else {
                const Eigen::Map<const Eigen::Quaterniond> q(x[block]+3);
                if(!q.coeffs().allFinite() || std::abs(q.squaredNorm()-1)>1e-6) return false;
                qm.leftCols<3>()=2*(q.w()*Eigen::Matrix3d::Identity()+skew(q.vec())); qm.rightCols<1>()=-2*q.vec();
            }
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> J(jacobians[block]);
            J.leftCols<3>()=translation; J.rightCols<4>()=rotation*qm; return J.allFinite();
        };
        if(observerPose_==0) {
            // Same-epoch cameras share both T_WB and T_BR, which cancel exactly.
            for(int block:{0,extrinsic_}) if(jacobians[block])
                Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>>(jacobians[block]).setZero();
        } else {
            // Cancel the left-SE3 ambient lift symbolically, before multiplication.
            // This avoids world-position cancellation and repeated 2x6 * 6x7 work.
            if(jacobians[0] || jacobians[observerPose_] || jacobians[extrinsic_]) {
                const Eigen::Matrix<double,2,3> D=reduce*W;
                if(!writePose(0,D,-D*skew(rotatedBody)) || !writePose(observerPose_,-D,D*skew(relativeWorld))) return false;
                if(jacobians[extrinsic_] && !writePose(extrinsic_,D*(Ra-Rb),D*(-Ra*skew(body-te)+Rb*skew(observer-te)))) return false;
            }
        }
        if(jacobians[inverseDepth_]) {
            Eigen::Map<Eigen::Vector2d> J(jacobians[inverseDepth_]); J=reduce*cameraRotation*(-ba/(rho*rho));
            if(!J.allFinite()) return false;
        }
        if(timeDerivative) {
            if(!movingAnchor_ && !movingObservation_) {
                Eigen::Map<Eigen::Vector2d>(jacobians[timeOffset_]).setZero(); return true;
            }
            const Eigen::Matrix3d Jb=f*(-bb*n.transpose()-c*Eigen::Matrix3d::Identity())+g*u*n.transpose();
            const Eigen::Matrix3d dK=skew(observation_.bearing.cross(db));
            const Eigen::Matrix3d dR=dK+(dK*K+K*dK)/denominator
                -(K*K)*(observation_.bearing.dot(db)/(denominator*denominator));
            Eigen::Map<Eigen::Vector2d> J(jacobians[timeOffset_]);
            J=reduce*cameraRotation*(da/rho)+P*Jb*db+measurement_*dR.transpose()*logarithm;
            if(!J.allFinite()) return false;
        }
        return true;
    }
    bool evaluateNumerical(double const* const* x,double* residuals,double** jacobians) const {
        Eigen::Vector2d base; if(!residual(x,base)) return false;
        Eigen::Map<Eigen::Vector2d> output(residuals); output=base;
        if(!jacobians) return true;
        std::vector<const double*> parameters(x,x+parameter_block_sizes().size());
        const LeftSE3Manifold manifold;
        for(size_t block=0;block<parameter_block_sizes().size();++block) {
            if(!jacobians[block]) continue;
            const int ambient=parameter_block_sizes()[block],tangent=ambient==7?6:1;
            Eigen::MatrixXd local(2,tangent);
            Eigen::VectorXd plus(ambient),minus(ambient),step=Eigen::VectorXd::Zero(tangent);
            for(int column=0;column<tangent;++column) {
                double h=ambient==7?(column<3?1e-6:1e-7):1e-7;
                if(static_cast<int>(block)==inverseDepth_) h=std::min(0.01*x[block][0],std::max(1e-12,1e-6*x[block][0]));
                if(ambient==7) {
                    step[column]=h; if(!manifold.Plus(x[block],step.data(),plus.data())) return false;
                    step[column]=-h; if(!manifold.Plus(x[block],step.data(),minus.data())) return false;
                    step[column]=0;
                } else { plus[0]=x[block][0]+h; minus[0]=x[block][0]-h; }
                Eigen::Vector2d rp,rm;
                parameters[block]=plus.data(); if(!residual(parameters.data(),rp)) return false;
                parameters[block]=minus.data(); if(!residual(parameters.data(),rm)) return false;
                parameters[block]=x[block]; local.col(column)=(rp-rm)/(2*h);
            }
            Eigen::Map<Eigen::Matrix<double,2,Eigen::Dynamic,Eigen::RowMajor>> J(jacobians[block],2,ambient);
            if(ambient==7) {
                Eigen::Matrix<double,6,7,Eigen::RowMajor> lift;
                if(!manifold.MinusJacobian(x[block],lift.data())) return false;
                J=local*lift;
            } else J=local;
            if(!J.allFinite()) return false;
        }
        return true;
    }
private:
    Eigen::Vector3d anchorBearing_;
    WindowObservation anchor_,observation_;
    kinematics::Transformation T_RCa_,T_RCb_;
    Eigen::Matrix<double,3,2> basis_;
    Eigen::Matrix3d Rca_,Rcb_;
    Eigen::Vector3d tca_,tcb_,anchorStatic_,observationStatic_;
    Eigen::Matrix<double,2,3> measurement_;
    bool movingAnchor_=false,movingObservation_=false;
    int observerPose_=0,extrinsic_=0,timeOffset_=-1,inverseDepth_=0;
    bool analytic_=true;
};
class CachedSphereAdapter final : public ::ceres::CostFunction {
public:
    CachedSphereAdapter(const CacheableSphereCost& cost,const std::vector<const double*>& parameters,
        VisualEvaluationCache& cache,bool cachePairs,bool fixedTime,unsigned derivativeMask)
        :cost_(cost),cache_(cache),derivativeMask_(derivativeMask) {
        *mutable_parameter_block_sizes()=cost.parameter_block_sizes(); set_num_residuals(cost.num_residuals());
        poses_.resize(parameters.size(),nullptr);
        for(size_t i=0;i<parameters.size();++i) if(parameter_block_sizes()[i]==7) poses_[i]=cache.bind(parameters[i]);
        if(cachePairs) pair_=cost.bindPair(parameters,cache);
        anchorTime_=cost.bindTimedAnchor(parameters,cache,fixedTime);
        if(fixedTime) observerTime_=cost.fixedObserver(parameters);
        if(cache.hasBatch()) {
            detail::VisualBatchInput input;
            if(cost.packBatch(input,derivativeMask)) batchIndex_=cache.bindBatch(input,parameters);
        }
    }
    bool Evaluate(double const* const* p,double* r,double** J) const override {
        const double* jacobians=nullptr;
        const auto* result=batchIndex_==std::size_t(-1)?nullptr:cache_.batchResult(batchIndex_,J?&jacobians:nullptr);
        if(result && result->valid) {
            int offset=0;
            if(J) for(std::size_t i=0;i<parameter_block_sizes().size();++i) {
                if(!(derivativeMask_&(1u<<i))) {
                    if(J[i]) return cost_.EvaluateCached(p,r,J,&poses_,pair_,anchorTime_,observerTime_?&*observerTime_:nullptr);
                    continue;
                }
                const int width=2*parameter_block_sizes()[i];
                if(J[i]) std::copy_n(jacobians+offset,width,J[i]);
                offset+=width;
            }
            std::copy_n(result->value,2,r);
            return true;
        }
        return cost_.EvaluateCached(p,r,J,&poses_,pair_,anchorTime_,observerTime_?&*observerTime_:nullptr);
    }
private:
    const CacheableSphereCost& cost_; std::vector<const VisualPoseCacheValue*> poses_;
    const VisualEvaluationCache& cache_;
    std::size_t batchIndex_=std::size_t(-1);
    unsigned derivativeMask_;
    const VisualPairCacheValue* pair_=nullptr;
    const VisualTimedBearing* anchorTime_=nullptr;
    std::optional<VisualTimedObserver> observerTime_;
};
}
std::unique_ptr<::ceres::CostFunction> cacheWindowVisualCost(const ::ceres::CostFunction& cost,
    const std::vector<const double*>& parameters,VisualEvaluationCache& cache,bool cachePairs,
    bool fixedTime,unsigned derivativeMask) {
    const auto* sphere=dynamic_cast<const CacheableSphereCost*>(&cost);
    if(!sphere || parameters.size()!=cost.parameter_block_sizes().size()) return {};
    return std::make_unique<CachedSphereAdapter>(*sphere,parameters,cache,cachePairs,fixedTime,derivativeMask);
}
bool makeWindowVisualFactor(const WindowLandmark& landmark,const WindowObservation& anchor,
    const WindowObservation& observation,const FrameStore& frames,const BackendGraph& graph,
    const BackendOptions& options,FactorRecord& out,std::string& error) {
    const StateId a(landmark.anchor.frameId),b(observation.key.frameId);
    const auto fa=frames.find(a),fb=frames.find(b);
    const BlockKey rho{BlockKind::Landmark,landmark.id.value()},extrinsic{BlockKind::CameraExtrinsic,0},td{BlockKind::CameraTimeOffset,0};
    if(!landmark.initialized || anchor.landmark!=landmark.id || observation.landmark!=landmark.id
        || !fa || !fb || landmark.anchor.cameraIndex>=fa->T_RC.size() || observation.key.cameraIndex>=fb->T_RC.size()
        || !graph.blocks.count(rho) || !graph.blocks.count(extrinsic)) {
        error="visual factor requires initialized landmark, valid cameras and global T_BR"; return false;
    }
    FactorRecord f; f.kind=FactorKind::Visual; f.sensorEpoch=b; f.epochs={a,b};
    f.keys={BlockKey::pose(a)}; if(a!=b) f.keys.push_back(BlockKey::pose(b));
    f.keys.push_back(extrinsic); if(graph.blocks.count(td)) f.keys.push_back(td); f.keys.push_back(rho);
    f.cost=std::make_shared<InverseDepthSphereCost>(landmark,anchor,observation,fa->T_RC[landmark.anchor.cameraIndex],
        fb->T_RC[observation.key.cameraIndex],a==b,graph.blocks.count(td)!=0,options.analyticVisualJacobians);
    f.nisThreshold=options.visualNisThreshold;
    if(options.visualHuberScale>0) f.loss=std::make_shared<::ceres::HuberLoss>(options.visualHuberScale);
    out=std::move(f); return true;
}
} }
