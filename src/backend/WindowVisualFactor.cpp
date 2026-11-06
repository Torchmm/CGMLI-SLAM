#include "WindowVisualFactor.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace cgmli { namespace backend {
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
    virtual bool EvaluateCached(double const* const*,double*,double**,
        const std::vector<const VisualPoseCacheValue*>*,const VisualPairCacheValue* = nullptr) const=0;
    virtual const VisualPairCacheValue* bindPair(const std::vector<const double*>&,VisualEvaluationCache&) const=0;
};
class InverseDepthSphereCost final : public CacheableSphereCost {
public:
    const VisualPairCacheValue* bindPair(const std::vector<const double*>& p,VisualEvaluationCache& cache) const override {
        return observerPose_==0?nullptr:cache.bindPair(p[0],p[observerPose_],p[extrinsic_],T_RCa_,T_RCb_);
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
        const std::vector<const VisualPoseCacheValue*>* cache,const VisualPairCacheValue* pair=nullptr) const override {
        if(!analytic_) return evaluateNumerical(x,residuals,jacobians);
        if(cache) for(const auto* pose:*cache) if(pose && !pose->valid) return false;
        if(pair && !pair->valid) return false;
        bool derivatives=false;
        if(jacobians) for(size_t i=0;i<parameter_block_sizes().size();++i) derivatives=derivatives || jacobians[i]!=nullptr;
        // VINS' chain rule is useful, but Exp(delta)*T here also rotates the
        // current position. Derive in the LEFT tangent and lift to ambient 7D.
        const double rho=x[inverseDepth_][0],td=timeOffset_<0?anchor_.referenceTimeOffset:x[timeOffset_][0];
        if(!std::isfinite(rho) || rho<=0 || !std::isfinite(td)) return false;
        const auto bearing=[](const Eigen::Vector3d& b,const Eigen::Vector3d& velocity,double dt,Eigen::Vector3d& derivative) {
            const double speed=velocity.norm(),angle=speed*dt;
            Eigen::Vector3d value;
            if(speed==0) { derivative.setZero(); return Eigen::Vector3d(b.normalized()); }
            if(std::abs(angle)<1e-8) {
                const Eigen::Vector3d raw=b+velocity*dt; const double length=raw.norm();
                value=raw/length;
                derivative=-(Eigen::Matrix3d::Identity()-value*value.transpose())*velocity/length;
            } else {
                value=std::cos(angle)*b+(std::sin(angle)/speed)*velocity;
                derivative=speed*std::sin(angle)*b-std::cos(angle)*velocity;
            }
            return value;
        };
        Eigen::Vector3d da,db;
        da.setZero(); db.setZero();
        const Eigen::Vector3d ba=!movingAnchor_?anchorStatic_:(derivatives
            ?bearing(anchorBearing_,anchor_.bearingVelocity,anchor_.referenceTimeOffset-td,da)
            :timeBearing(anchorBearing_,anchor_.bearingVelocity,anchor_.referenceTimeOffset-td));
        const Eigen::Vector3d bb=!movingObservation_?observationStatic_:(derivatives
            ?bearing(observation_.bearing,observation_.bearingVelocity,observation_.referenceTimeOffset-td,db)
            :timeBearing(observation_.bearing,observation_.bearingVelocity,observation_.referenceTimeOffset-td));
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
        const double denominator=movingObservation_?1+observation_.bearing.dot(bb):2;
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
        Eigen::Matrix<double,2,3> P=measurement_;
        if(movingObservation_) {
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
        if(timeOffset_>=0 && jacobians[timeOffset_]) {
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
    CachedSphereAdapter(const CacheableSphereCost& cost,const std::vector<const double*>& parameters,VisualEvaluationCache& cache,bool cachePairs)
        :cost_(cost) {
        *mutable_parameter_block_sizes()=cost.parameter_block_sizes(); set_num_residuals(cost.num_residuals());
        poses_.resize(parameters.size(),nullptr);
        for(size_t i=0;i<parameters.size();++i) if(parameter_block_sizes()[i]==7) poses_[i]=cache.bind(parameters[i]);
        if(cachePairs) pair_=cost.bindPair(parameters,cache);
    }
    bool Evaluate(double const* const* p,double* r,double** J) const override { return cost_.EvaluateCached(p,r,J,&poses_,pair_); }
private:
    const CacheableSphereCost& cost_; std::vector<const VisualPoseCacheValue*> poses_;
    const VisualPairCacheValue* pair_=nullptr;
};
}
std::unique_ptr<::ceres::CostFunction> cacheWindowVisualCost(const ::ceres::CostFunction& cost,
    const std::vector<const double*>& parameters,VisualEvaluationCache& cache,bool cachePairs) {
    const auto* sphere=dynamic_cast<const CacheableSphereCost*>(&cost);
    if(!sphere || parameters.size()!=cost.parameter_block_sizes().size()) return {};
    return std::make_unique<CachedSphereAdapter>(*sphere,parameters,cache,cachePairs);
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
