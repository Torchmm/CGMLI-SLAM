#include "SlidingWindowManager.hpp"
#include "WindowVisualFactor.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <glog/logging.h>

namespace cgmli { namespace backend {
namespace {
bool evaluate(const FactorRecord& factor,const BackendGraph& graph,BlockKey changed,double value,
    Eigen::VectorXd& residual) {
    std::vector<const double*> parameters;
    for(auto key:factor.keys) parameters.push_back(key==changed?&value:graph.blocks.at(key).value.data());
    residual.resize(factor.cost->num_residuals());
    return factor.cost->Evaluate(parameters.data(),residual.data(),nullptr) && residual.allFinite();
}
}
bool checkReanchorInformation(const WindowLandmark& oldLandmark,const WindowObservation& oldAnchor,
    const WindowLandmark& newLandmark,const WindowObservation& newAnchor,const FrameStore& frames,
    const BackendGraph& before,const BackendGraph& after,const BackendOptions& options,
    ReanchorInformationReport& report,std::string& error) {
    report=ReanchorInformationReport(); error.clear();
    if(oldLandmark.id!=newLandmark.id || !oldLandmark.initialized || !newLandmark.initialized
        || !(oldAnchor.key==oldLandmark.anchor) || !(newAnchor.key==newLandmark.anchor)
        || oldAnchor.landmark!=oldLandmark.id || newAnchor.landmark!=oldLandmark.id) {
        error="reanchor audit requires the same initialized physical landmark"; return false;
    }
    const BlockKey rho{BlockKind::Landmark,oldLandmark.id.value()};
    const auto incident=[&](const BackendGraph& graph) {
        std::vector<FactorId> ids;
        for(const auto& item:graph.factors) if(item.second.lifecycle==FactorLifecycle::Active
            && std::find(item.second.keys.begin(),item.second.keys.end(),rho)!=item.second.keys.end()) ids.push_back(item.first);
        return ids;
    };
    auto oldGraph=before,newGraph=after;
    Marginalizer marginalizer(options); LinearizedSystem oldSystem,newSystem;
    if(!marginalizer.linearize(oldGraph,incident(oldGraph),true,true,oldSystem,error)
        || !marginalizer.linearize(newGraph,incident(newGraph),true,true,newSystem,error)) return false;
    std::set<BlockKey> keys(oldSystem.keys.begin(),oldSystem.keys.end());
    keys.insert(newSystem.keys.begin(),newSystem.keys.end()); keys.insert(rho);
    keys.insert(BlockKey::pose(StateId(oldLandmark.anchor.frameId)));
    keys.insert(BlockKey::pose(StateId(newLandmark.anchor.frameId)));
    keys.insert(BlockKey{BlockKind::CameraExtrinsic,0});
    if(before.blocks.count({BlockKind::CameraTimeOffset,0})) keys.insert(BlockKey{BlockKind::CameraTimeOffset,0});
    std::map<BlockKey,int> offsets; int columns=0;
    for(auto key:keys) {
        const auto a=before.blocks.find(key),b=after.blocks.find(key);
        if(a==before.blocks.end() || b==after.blocks.end() || a->second.value.size()!=b->second.value.size()
            || a->second.tangentSize()!=b->second.tangentSize() || a->second.manifold!=b->second.manifold) {
            error="reanchor audit parameter layouts differ"; return false;
        }
        if(key!=rho && (a->second.owner!=b->second.owner || !(a->second.value.array()==b->second.value.array()).all())) {
            error="reanchor audit may not change another physical state"; return false;
        }
        offsets[key]=columns; report.keys.push_back(key); report.offsets.push_back(columns);
        report.sizes.push_back(a->second.tangentSize()); columns+=a->second.tangentSize();
    }
    const auto embed=[&](const LinearizedSystem& system,Eigen::MatrixXd& H,Eigen::VectorXd& b) {
        H=Eigen::MatrixXd::Zero(columns,columns); b=Eigen::VectorXd::Zero(columns);
        for(size_t i=0;i<system.keys.size();++i) {
            b.segment(offsets.at(system.keys[i]),system.sizes[i])=system.b.segment(system.offsets[i],system.sizes[i]);
            for(size_t j=0;j<system.keys.size();++j)
                H.block(offsets.at(system.keys[i]),offsets.at(system.keys[j]),system.sizes[i],system.sizes[j])
                    =system.H.block(system.offsets[i],system.offsets[j],system.sizes[i],system.sizes[j]);
        }
    };
    embed(oldSystem,report.oldInformation,report.oldGradient); embed(newSystem,report.newInformation,report.newGradient);
    const auto point=[&](const WindowLandmark& landmark,const WindowObservation& anchor,
        const BackendGraph& graph,Eigen::Vector3d& value) {
        const auto block=graph.blocks.find(rho);
        if(block==graph.blocks.end() || block->second.value.size()!=1 || block->second.value[0]<=0) return false;
        kinematics::Transformation camera;
        if(!frames.cameraPose(graph,StateId(landmark.anchor.frameId),landmark.anchor.cameraIndex,camera,error)) return false;
        const auto td=graph.blocks.find({BlockKind::CameraTimeOffset,0});
        const double dt=anchor.referenceTimeOffset-(td==graph.blocks.end()?anchor.referenceTimeOffset:td->second.value[0]);
        const Eigen::Vector3d u=anchor.bearingVelocity*dt; const double a=u.norm();
        const Eigen::Vector3d direction=a<1e-8?(landmark.anchorBearing+u).normalized()
            :Eigen::Vector3d(std::cos(a)*landmark.anchorBearing+(std::sin(a)/a)*u);
        value=camera.C()*(direction/block->second.value[0])+camera.r(); return value.allFinite();
    };
    Eigen::Vector3d oldPoint,newPoint;
    if(!point(oldLandmark,oldAnchor,before,oldPoint) || !point(newLandmark,newAnchor,after,newPoint)) {
        error="reanchor audit cannot evaluate world geometry"; return false;
    }
    report.pointError=(oldPoint-newPoint).norm()/std::max(1.0,oldPoint.norm());
    const auto derivative=[&](const WindowLandmark& landmark,const WindowObservation& anchor,
        BackendGraph& graph,Eigen::MatrixXd& J) {
        J=Eigen::MatrixXd::Zero(3,columns);
        for(size_t k=0;k<report.keys.size();++k) {
            const auto key=report.keys[k];
            if(key!=rho && key!=BlockKey::pose(StateId(landmark.anchor.frameId))
                && key!=BlockKey{BlockKind::CameraExtrinsic,0} && key!=BlockKey{BlockKind::CameraTimeOffset,0}) continue;
            auto& block=graph.blocks.at(key); const Eigen::VectorXd saved=block.value;
            const int tangent=block.tangentSize();
            for(int c=0;c<tangent;++c) {
                const double h=key==rho?std::min(1e-6,saved[0]*1e-3):1e-6;
                Eigen::VectorXd step=Eigen::VectorXd::Zero(tangent),plus=saved,minus=saved;
                if(block.manifold) {
                    step[c]=h;
                    if(!block.manifold->Plus(saved.data(),step.data(),plus.data())) return false;
                    step[c]=-h;
                    if(!block.manifold->Plus(saved.data(),step.data(),minus.data())) return false;
                } else { plus[c]+=h; minus[c]-=h; }
                Eigen::Vector3d p,m;
                block.value=plus; const bool pOk=point(landmark,anchor,graph,p);
                block.value=minus; const bool mOk=point(landmark,anchor,graph,m);
                block.value=saved;
                if(!pOk || !mOk) return false;
                J.col(report.offsets[k]+c)=(p-m)/(2*h);
            }
        }
        return J.allFinite();
    };
    if(!derivative(oldLandmark,oldAnchor,oldGraph,report.oldPointJacobian)
        || !derivative(newLandmark,newAnchor,newGraph,report.newPointJacobian)) {
        error="reanchor audit cannot differentiate world geometry"; return false;
    }
    const int depthColumn=offsets.at(rho);
    const Eigen::Vector3d depthDirection=report.oldPointJacobian.col(depthColumn);
    if(depthDirection.squaredNorm()<1e-16) { error="singular inverse-range chart"; return false; }
    // Other states retain their physical tangent coordinates. The best possible
    // scalar rho transport removes only the component along the OLD anchor ray.
    // Any remaining component proves that a fixed-bearing 1D chart swap is not
    // a local coordinate change, even though its nominal point happens to agree.
    report.oldFromNew=Eigen::MatrixXd::Identity(columns,columns);
    report.oldFromNew.row(depthColumn)+=depthDirection.transpose()
        *(report.newPointJacobian-report.oldPointJacobian)/depthDirection.squaredNorm();
    const Eigen::MatrixXd transportedH=report.oldFromNew.transpose()*report.oldInformation*report.oldFromNew;
    const Eigen::VectorXd transportedB=report.oldFromNew.transpose()*report.oldGradient;
    report.pointDerivativeError=(report.oldPointJacobian*report.oldFromNew-report.newPointJacobian).norm()
        /std::max(1.0,report.newPointJacobian.norm());
    // Check blocks separately so a strong unrelated prior cannot hide a weak
    // direction's lost information in a single whole-matrix relative norm.
    for(size_t i=0;i<report.keys.size();++i) {
        const int oi=report.offsets[i],si=report.sizes[i];
        report.gradientError=std::max(report.gradientError,
            (transportedB.segment(oi,si)-report.newGradient.segment(oi,si)).norm()
            /std::max(1.0,transportedB.segment(oi,si).norm()));
        for(size_t j=0;j<report.keys.size();++j) {
            const int oj=report.offsets[j],sj=report.sizes[j];
            report.informationError=std::max(report.informationError,
                (transportedH.block(oi,oj,si,sj)-report.newInformation.block(oi,oj,si,sj)).norm()
                /std::max(1.0,transportedH.block(oi,oj,si,sj).norm()));
        }
    }
    report.equivalent=report.oldFromNew.allFinite() && std::abs(report.oldFromNew(depthColumn,depthColumn))>1e-10
        && report.pointError<=1e-8 && report.pointDerivativeError<=1e-6
        && report.informationError<=1e-6 && report.gradientError<=1e-6;
    return true;
}
bool SlidingWindowManager::reanchorOutgoing(StateId outgoing,BackendGraph& graph,LandmarkStore& store,
    std::vector<LandmarkId>& changed,std::string& error,std::map<LandmarkId,std::string>* reasons) {
    // Opt-in candidates must also pass the physical tangent / normal-equation
    // audit below. A failed chart change falls back to ordinary Schur retirement.
    if(!options_.autoReanchorLandmarks || frames_.epochs_.at(outgoing).visual==SensorHealth::Invalid) return true;
    std::set<BlockKey> priorKeys;
    if(graph.prior) priorKeys.insert(graph.prior->keys().begin(),graph.prior->keys().end());
    for(auto& item:store.landmarks_) {
        const auto original=item.second;
        if(!original.initialized || original.anchor.frameId!=outgoing.value()) continue;
        const BlockKey rho{BlockKind::Landmark,item.first.value()};
        const auto oldAnchor=store.find(original.anchor);
        const auto why=[&](const char* reason) { if(reasons) (*reasons)[item.first]=reason; };
        if(priorKeys.count(rho)) { why("historical_prior_coupled"); continue; }
        if(!oldAnchor || oldAnchor->bearingVelocity.norm()>1e-12) { why("missing_or_time_varying_anchor"); continue; }
        bool eligible=true;
        for(const auto& key:store.ofLandmark(item.first)) {
            const auto observation=store.find(key);
            if(observation->use==ObservationUse::Absorbed || observation->bearingVelocity.norm()>1e-12) eligible=false;
        }
        // An arbitrary non-visual cost may encode the old inverse-depth chart. Without
        // its explicit transport, changing rho would silently change that measurement.
        for(const auto& factor:graph.factors) if(std::find(factor.second.keys.begin(),factor.second.keys.end(),rho)!=factor.second.keys.end()
            && !store.byFactor_.count(factor.first)) eligible=false;
        if(!eligible) { why("absorbed_motion_or_external_factor"); continue; }
        Eigen::Vector3d point; kinematics::Transformation oldCamera;
        if(!pointWorld(graph,store,item.first,point,error)
            || !frames_.cameraPose(graph,outgoing,original.anchor.cameraIndex,oldCamera,error)) return false;
        const Eigen::Vector3d oldRay=(point-oldCamera.r()).normalized();
        std::optional<KeypointIdentifier> best;
        Eigen::Vector3d newBearing=Eigen::Vector3d::UnitZ(); double newRho=0;
        std::tuple<bool,double,uint64_t> bestScore{false,-1.0,0};
        for(const auto& key:store.ofLandmark(item.first)) {
            const auto observation=store.find(key); const StateId observer(key.frameId);
            if(observer==outgoing || observation->use!=ObservationUse::LiveFactor
                || frames_.epochs_.at(observer).visual==SensorHealth::Invalid) continue;
            kinematics::Transformation camera;
            if(!frames_.cameraPose(graph,observer,key.cameraIndex,camera,error)) return false;
            const Eigen::Vector3d ray=point-camera.r(); const double depth=ray.norm();
            if(depth<1e-6 || 1.0/depth<options_.minInverseDepth) continue;
            const Eigen::Vector3d predicted=camera.C().transpose()*(ray/depth);
            const double angularError=std::atan2(predicted.cross(observation->bearing).norm(),predicted.dot(observation->bearing));
            const double parallax=std::atan2(oldRay.cross(ray/depth).norm(),oldRay.dot(ray/depth));
            if(angularError>options_.maxReanchorAngularError || parallax<options_.minReanchorParallax) continue;
            const auto score=std::make_tuple(frames_.epochs_.at(observer).keyframe,parallax,
                std::numeric_limits<uint64_t>::max()-key.frameId);
            if(!best || score>bestScore) { best=key; bestScore=score; newBearing=predicted; newRho=1.0/depth; }
        }
        if(!best) { why("no_safe_surviving_anchor"); continue; }
        WindowLandmark replacement=original; replacement.anchor=*best; replacement.anchorBearing=newBearing;
        const auto newAnchor=*store.find(*best);
        std::map<FactorId,FactorRecord> replacements;
        for(const auto& key:store.ofLandmark(item.first)) {
            const auto observation=store.find(key);
            if(observation->use!=ObservationUse::LiveFactor) continue;
            auto previous=graph.factors.find(observation->factorId);
            if(previous==graph.factors.end() || previous->second.lifecycle!=FactorLifecycle::Active) { eligible=false; break; }
            FactorRecord factor;
            if(!makeWindowVisualFactor(replacement,newAnchor,*observation,frames_,graph,options_,factor,error)) return false;
            factor.id=previous->first; factor.lifecycle=previous->second.lifecycle;
            factor.loss=previous->second.loss; factor.nisThreshold=previous->second.nisThreshold;
            Eigen::VectorXd before,after;
            if(!evaluate(previous->second,graph,rho,graph.blocks.at(rho).value[0],before)
                || !evaluate(factor,graph,rho,newRho,after) || (before-after).norm()>1e-6*(1.0+before.norm())) {
                eligible=false; break;
            }
            replacements.emplace(factor.id,std::move(factor));
        }
        if(!eligible) { why("candidate_residual_or_lifecycle_changed"); continue; }
        // The original anchor was implicit in the old chart. After moving the chart,
        // turn that image measurement into an explicit residual before removing its
        // pose, otherwise its bearing information would disappear from the Schur step.
        std::optional<FactorRecord> oldAnchorCost;
        if(oldAnchor->use==ObservationUse::TrackingOnly) {
            FactorRecord factor;
            if(!makeWindowVisualFactor(replacement,newAnchor,*oldAnchor,frames_,graph,options_,factor,error)) return false;
            Eigen::VectorXd residual;
            if(!evaluate(factor,graph,rho,newRho,residual) || residual.norm()>1e-6) { why("implicit_anchor_residual_changed"); continue; }
            if(!nextFactorId_) { error="factor ID exhausted during reanchor"; return false; }
            factor.id=nextFactorId_; factor.lifecycle=FactorLifecycle::Active; oldAnchorCost=std::move(factor);
        }
        BackendGraph proposed=graph;
        proposed.blocks.at(rho).owner=StateId(best->frameId); proposed.blocks.at(rho).value[0]=newRho;
        for(const auto& factor:replacements) proposed.factors.at(factor.first)=factor.second;
        if(oldAnchorCost) proposed.factors.emplace(oldAnchorCost->id,*oldAnchorCost);
        ReanchorInformationReport audit;
        if(!checkReanchorInformation(original,*oldAnchor,replacement,newAnchor,frames_,graph,proposed,options_,audit,error)) return false;
        if(!audit.equivalent) {
            why("information_mismatch");
            VLOG(1)<<"[Reanchor] rejected non-equivalent chart; landmark="<<item.first.value()
                <<" point_error="<<audit.pointError<<" tangent_error="<<audit.pointDerivativeError
                <<" information_error="<<audit.informationError<<" gradient_error="<<audit.gradientError;
            continue;
        }
        item.second=replacement;
        graph.blocks.at(rho).owner=StateId(best->frameId); graph.blocks.at(rho).value[0]=newRho;
        for(auto& factor:replacements) graph.factors.at(factor.first)=std::move(factor.second);
        if(oldAnchorCost) {
            ++nextFactorId_;
            const auto id=oldAnchorCost->id; graph.factors.emplace(id,std::move(*oldAnchorCost)); store.bind(original.anchor,id);
        }
        Eigen::Vector3d reconstructed;
        if(!pointWorld(graph,store,item.first,reconstructed,error) || (reconstructed-point).norm()>1e-7*(1.0+point.norm())) {
            error="reanchor changed world point"; return false;
        }
        changed.push_back(item.first);
    }
    return true;
}
} }
