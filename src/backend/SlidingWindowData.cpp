#include "SlidingWindowManager.hpp"
#include "WindowVisualFactor.hpp"
#include <visual/MultiFrame.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cgmli { namespace backend {
bool SlidingWindowManager::applyVisualUpdates(const VisualUpdateBatch& batch,std::string& error) {
    // Copy metadata/numeric buffers once; immutable feature payloads and costs remain
    // shared. All helpers below mutate only this private candidate, so a bad index or
    // failed Schur retirement cannot leave half an image's associations installed.
    auto staged=*this;
    if(!staged.applyVisualUpdatesImpl(batch,error) || !staged.valid(error)) return false;
    *this=std::move(staged); return true;
}
bool SlidingWindowManager::applyVisualUpdatesImpl(const VisualUpdateBatch& batch,std::string& error) {
    // Restore on every return/exception; the owner still discards this private
    // checkpoint on failure. Promotion must not rescan the full graph per seed.
    struct PendingGuard {
        bool& flag; bool previous;
        explicit PendingGuard(bool& value):flag(value),previous(value) { flag=true; }
        ~PendingGuard() { flag=previous; }
    };
    bool success=false;
    {
        PendingGuard guard(deferPendingActivation_);
        success=[&]() {
            for(const auto& key:batch.removedObservations) if(!removeObservationImpl(key,error)) return false;
            if(!retireLandmarksImpl(batch.retiredLandmarks,error)) return false;
            for(const auto& point:batch.landmarks) if(!addLandmarkImpl(point.id,point.anchor,std::nullopt,error,point.options)) return false;
            for(const auto& observation:batch.observations) if(!addObservationImpl(observation.landmark,observation.key,error,observation.options)) return false;
            std::set<LandmarkId> changedAnchors;
            for(const auto& point:batch.seedAnchors) {
                if(!changedAnchors.insert(point.id).second) { error="duplicate seed reference update"; return false; }
                if(!reanchorSeedImpl(point.id,point.anchor,error)) return false;
            }
            for(const auto& point:batch.landmarks) if(point.inverseDepth && !initializeLandmarkImpl(point.id,*point.inverseDepth,error)) return false;
            for(const auto& point:batch.initializations) if(!initializeLandmarkImpl(point.id,point.inverseDepth,error)) return false;
            return true;
        }();
    }
    if(success && !deferPendingActivation_) activatePending();
    return success;
}
bool SlidingWindowManager::append(const StateEstimate& prediction,const MultiFrame& frame,std::string& error) {
    auto staged=*this;
    if(!staged.append(prediction,error) || !staged.attachFrame(frame,error)) return false;
    *this=std::move(staged); return true;
}
bool SlidingWindowManager::attachFrame(const MultiFrame& frame,std::string& error) {
    return frames_.attach(frame,options_.retainFrameImages,error);
}
bool SlidingWindowManager::attachImu(StateId id,const ImuMeasurementDeque& samples,std::string& error) {
    return frames_.attachImu(id,samples,options_.maxImuSamplesPerEpoch,error);
}
LandmarkId SlidingWindowManager::nextLandmarkId() const {
    if(landmarks_.largestId_==std::numeric_limits<uint64_t>::max()) throw std::overflow_error("landmark ID exhausted");
    return LandmarkId(landmarks_.largestId_+1);
}
bool SlidingWindowManager::observationInput(const KeypointIdentifier& key,const ObservationOptions& options,
    WindowObservation& observation,std::string& error) const {
    if(!frames_.observation(key,options.pixelSigma,observation,error)) return false;
    observation.referenceTimeOffset=options.referenceTimeOffset.value_or(frames_.find(StateId(key.frameId))->referenceCameraTimeOffset);
    observation.bearingVelocity=options.bearingVelocity;
    return true; // LandmarkStore::insert validates velocity tangency and finiteness.
}
bool SlidingWindowManager::addLandmark(LandmarkId id,const KeypointIdentifier& anchor,
    std::optional<double> inverseDepth,std::string& error,const ObservationOptions& options) {
    VisualUpdateBatch batch; batch.landmarks.push_back({id,anchor,inverseDepth,options});
    return applyVisualUpdates(batch,error);
}
bool SlidingWindowManager::addLandmarkImpl(LandmarkId id,const KeypointIdentifier& anchor,
    std::optional<double> inverseDepth,std::string& error,const ObservationOptions& options) {
    if(landmarks_.size()>=options_.maxLandmarks || landmarks_.observations_.size()+landmarks_.absorbedKeys_.size()>=options_.maxObservations) {
        error="landmark/observation budget exceeded"; return false;
    }
    WindowObservation measurement;
    if(!observationInput(anchor,options,measurement,error)) return false;
    auto epoch=frames_.epochs_.find(StateId(anchor.frameId));
    if(epoch==frames_.epochs_.end() || epoch->second.visual==SensorHealth::Invalid) { error="anchor camera epoch is invalid"; return false; }
    WindowLandmark landmark; landmark.id=id; landmark.anchor=anchor; landmark.anchorBearing=measurement.bearing;
    landmark.createdFrame=order_.empty()?anchor.frameId:order_.back().value();
    measurement.landmark=id;
    if(!landmarks_.insert(landmark,error) || !landmarks_.insert(measurement,error)) return false;
    if(inverseDepth && !initializeLandmarkImpl(id,*inverseDepth,error)) return false;
    return true;
}
bool SlidingWindowManager::reanchorSeedImpl(LandmarkId id,const KeypointIdentifier& anchor,std::string& error) {
    auto point=landmarks_.landmarks_.find(id);
    const BlockKey rho{BlockKind::Landmark,id.value()};
    const auto view=landmarks_.find(anchor);
    const auto epoch=frames_.epochs_.find(StateId(anchor.frameId));
    const bool priorCoupled=graph_.prior && std::find(graph_.prior->keys().begin(),graph_.prior->keys().end(),rho)!=graph_.prior->keys().end();
    if(point==landmarks_.landmarks_.end() || point->second.initialized || graph_.blocks.count(rho) || priorCoupled
        || epoch==frames_.epochs_.end() || epoch->second.visual==SensorHealth::Invalid
        || !view || view->landmark!=id || view->use!=ObservationUse::TrackingOnly || view->factorId
        || landmarks_.consumed(anchor)) {
        error="seed reference requires a fresh owned observation and no optimized/prior coordinate"; return false;
    }
    // No rho or linearized information is transported. Absorbed observations keep
    // their tombstones and can NEVER become fresh factors through this operation.
    point->second.anchor=anchor; point->second.anchorBearing=view->bearing;
    return true;
}
bool SlidingWindowManager::initializeLandmarkImpl(LandmarkId id,double rho,std::string& error) {
    // Promotion adds an optimization coordinate, not a synthetic depth observation.
    // Depth/gauge observability still has to come from real multi-view measurements
    // and initialization constraints supplied by the caller.
    auto point=landmarks_.landmarks_.find(id);
    if(point==landmarks_.landmarks_.end() || point->second.initialized || !std::isfinite(rho) || rho<options_.minInverseDepth
        || !graph_.blocks.count({BlockKind::CameraExtrinsic,0})) {
        error="invalid seed/rho or global rig calibration missing"; return false;
    }
    const BlockKey key{BlockKind::Landmark,id.value()};
    if(graph_.blocks.count(key)) { error="landmark parameter already exists"; return false; }
    const auto anchor=landmarks_.find(point->second.anchor);
    if(!anchor || anchor->use==ObservationUse::Absorbed || landmarks_.consumed(anchor->key)) {
        error="depth initialization requires a fresh, unconsumed anchor"; return false;
    }
    auto block=ParameterBlock::euclidean(key,Eigen::VectorXd::Constant(1,rho),StateId(point->second.anchor.frameId));
    block.lower=Eigen::VectorXd::Constant(1,options_.minInverseDepth);
    graph_.blocks.emplace(key,std::move(block)); point->second.initialized=true;
    ++point->second.initializationCount;
    point->second.lastInitializedFrame=order_.back().value();
    point->second.lastTransitionFrame=order_.back().value();
    // The original anchor ray is the inverse-depth coordinate reference. Other
    // tracked measurements become real S^2 residuals in the same transaction.
    const auto& keys=landmarks_.ofLandmark(id); // Binding costs does not change the landmark's key set.
    for(const auto& observation:keys) if(!(observation==point->second.anchor)
        && landmarks_.find(observation)->use==ObservationUse::TrackingOnly && !landmarks_.consumed(observation)
        && !attachVisualCost(observation,error)) return false;
    if(!deferPendingActivation_) activatePending(); return true;
}
bool SlidingWindowManager::initializeLandmark(LandmarkId id,double rho,std::string& error) {
    VisualUpdateBatch batch; batch.initializations.push_back({id,rho}); return applyVisualUpdates(batch,error);
}
bool SlidingWindowManager::attachVisualCost(const KeypointIdentifier& key,std::string& error) {
    auto& observation=landmarks_.observations_.at(key);
    const auto& point=landmarks_.landmarks_.at(observation.landmark);
    if(observation.use!=ObservationUse::TrackingOnly || landmarks_.consumed(key)) {
        error="measurement is already bound or absorbed into a prior"; return false;
    }
    const auto anchor=landmarks_.find(point.anchor);
    if(!anchor) { error="landmark anchor observation is missing"; return false; }
    FactorRecord factor;
    if(!makeWindowVisualFactor(point,*anchor,observation,frames_,graph_,options_,factor,error)) return false;
    const auto submitted=submitFactor(std::move(factor),true);
    if(submitted.lifecycle!=FactorLifecycle::Active) { error=submitted.message; return false; }
    landmarks_.bind(key,submitted.id); return true;
}
bool SlidingWindowManager::addObservation(LandmarkId id,const KeypointIdentifier& key,std::string& error,
    const ObservationOptions& options) {
    VisualUpdateBatch batch; batch.observations.push_back({id,key,options}); return applyVisualUpdates(batch,error);
}
bool SlidingWindowManager::addObservationImpl(LandmarkId id,const KeypointIdentifier& key,std::string& error,
    const ObservationOptions& options) {
    if(!landmarks_.find(id) || landmarks_.observations_.size()+landmarks_.absorbedKeys_.size()>=options_.maxObservations) {
        error="unknown landmark or observation budget exceeded"; return false;
    }
    WindowObservation observation;
    if(!observationInput(key,options,observation,error)) return false;
    auto epoch=frames_.epochs_.find(StateId(key.frameId));
    if(epoch==frames_.epochs_.end() || epoch->second.visual==SensorHealth::Invalid) { error="observer epoch is invalid"; return false; }
    observation.landmark=id;
    if(!landmarks_.insert(observation,error)) return false;
    auto& point=landmarks_.landmarks_.at(id);
    if(!point.initialized && landmarks_.consumed(point.anchor)) {
        point.anchor=key; point.anchorBearing=observation.bearing;
    }
    if(landmarks_.find(id)->initialized && !attachVisualCost(key,error)) return false;
    return true;
}
bool SlidingWindowManager::removeObservation(const KeypointIdentifier& key,std::string& error) {
    VisualUpdateBatch batch; batch.removedObservations.push_back(key); return applyVisualUpdates(batch,error);
}
bool SlidingWindowManager::removeObservationImpl(const KeypointIdentifier& key,std::string& error) {
    const auto observation=landmarks_.find(key);
    if(!observation) { error="observation not found"; return false; }
    if(landmarks_.find(observation->landmark)->anchor==key) {
        error="cannot erase geometric anchor; retire or reanchor the landmark"; return false;
    }
    if(observation->use==ObservationUse::LiveFactor && !rejectFactor(observation->factorId,"observation removed")) {
        error="observation factor is missing"; return false;
    }
    landmarks_.eraseObservation(key); return true;
}
bool SlidingWindowManager::retireLandmark(LandmarkId id,std::string& error) {
    VisualUpdateBatch batch; batch.retiredLandmarks.push_back(id); return applyVisualUpdates(batch,error);
}
bool SlidingWindowManager::retireLandmarksImpl(const std::vector<LandmarkId>& ids,std::string& error) {
    // Eliminate the requested rho blocks jointly: a previous marginal prior may
    // couple them, so independently deleting residuals or summing separate priors
    // would lose cross information or count historical measurements more than once.
    std::set<BlockKey> eliminated;
    std::set<LandmarkId> unique;
    for(auto id:ids) {
        const auto point=landmarks_.find(id);
        if(!point || !unique.insert(id).second) { error="unknown/duplicate landmark retirement"; return false; }
        if(point->initialized) eliminated.insert({BlockKind::Landmark,id.value()});
    }
    if(!eliminated.empty()) {
        auto result=marginalizer_.compute(graph_,eliminated);
        if(!result.success) { error=result.message; return false; }
        graph_.prior=result.prior;
        for(auto factor:result.consumed) {
            auto event=graph_.factors.at(factor); event.lifecycle=FactorLifecycle::Marginalized;
            event.reason="landmark Schur retirement"; record(event);
            landmarks_.factorRemoved(factor,true); graph_.factors.erase(factor);
        }
        for(auto it=graph_.factors.begin();it!=graph_.factors.end();) {
            bool touches=false; for(auto key:it->second.keys) touches=touches || eliminated.count(key)!=0;
            if(touches) {
                auto event=it->second; event.lifecycle=FactorLifecycle::Rejected; event.reason="landmark retired before activation";
                record(event); it=graph_.factors.erase(it);
            } else ++it;
        }
        for(auto key:eliminated) graph_.blocks.erase(key);
    }
    for(auto id:ids) landmarks_.eraseLandmark(id,landmarks_.find(id)->initialized);
    return true;
}
bool SlidingWindowManager::setLandmarkQuality(LandmarkId id,double quality,int classification) {
    auto point=landmarks_.landmarks_.find(id);
    if(point==landmarks_.landmarks_.end() || !std::isfinite(quality) || quality<0) return false;
    point->second.quality=quality; point->second.classification=classification; return true;
}
bool SlidingWindowManager::pointWorld(const BackendGraph& graph,const LandmarkStore& store,LandmarkId id,
    Eigen::Vector3d& point,std::string& error) const {
    const auto landmark=store.find(id);
    auto block=graph.blocks.find({BlockKind::Landmark,id.value()});
    if(!landmark || !landmark->initialized || block==graph.blocks.end() || block->second.value.size()!=1
        || block->second.value[0]<=0) { error="landmark is not initialized"; return false; }
    kinematics::Transformation T_WC;
    if(!frames_.cameraPose(graph,StateId(landmark->anchor.frameId),landmark->anchor.cameraIndex,T_WC,error)) return false;
    const auto anchor=store.find(landmark->anchor);
    if(!anchor) { error="anchor observation missing"; return false; }
    auto td=graph.blocks.find({BlockKind::CameraTimeOffset,0});
    const double dt=anchor->referenceTimeOffset-(td==graph.blocks.end()?anchor->referenceTimeOffset:td->second.value[0]);
    const Eigen::Vector3d direction=timeBearing(landmark->anchorBearing,anchor->bearingVelocity,dt);
    point=T_WC.C()*(direction/block->second.value[0])+T_WC.r(); return point.allFinite();
}
std::optional<LandmarkEstimate> SlidingWindowManager::landmark(LandmarkId id) const {
    const auto record=landmarks_.find(id); if(!record) return std::nullopt;
    LandmarkEstimate result; result.metadata=*record;
    if(record->initialized) {
        Eigen::Vector3d point; std::string error;
        if(!pointWorld(graph_,landmarks_,id,point,error)) return std::nullopt;
        result.pointWorld=point; result.inverseDepth=graph_.blocks.at({BlockKind::Landmark,id.value()}).value[0];
    }
    for(const auto& key:landmarks_.ofLandmark(id)) result.observations.push_back(*landmarks_.find(key));
    return result;
}
bool SlidingWindowManager::getLandmark(LandmarkId id,MapPoint2& point) const {
    auto estimate=landmark(id); if(!estimate) return false;
    MapPoint2 result{}; result.id=id; result.point=Eigen::Vector4d::Zero();
    result.isInitialised=estimate->metadata.initialized; result.quality=estimate->metadata.quality;
    result.classification=estimate->metadata.classification;
    if(estimate->pointWorld) { result.point.head<3>()=*estimate->pointWorld; result.point[3]=1; }
    for(const auto& observation:estimate->observations) result.observations.insert(observation.key);
    point=result; return true;
}
std::shared_ptr<const MultiFrame> SlidingWindowManager::exportFrame(StateId id,const kinematics::Transformation& T_SB) const {
    return copyFrame(id,T_SB,options_.retainFrameImages);
}
std::shared_ptr<MultiFrame> SlidingWindowManager::associationFrame(StateId id,const kinematics::Transformation& T_SB) const {
    return copyFrame(id,T_SB,false);
}
std::shared_ptr<MultiFrame> SlidingWindowManager::copyFrame(StateId id,const kinematics::Transformation& T_SB,bool keepImages) const {
    auto source=frames_.measurements(id); if(!source) return nullptr;
    auto result=FrameStore::copyMeasurements(*source,keepImages);
    auto rig=graph_.blocks.find({BlockKind::CameraExtrinsic,0});
    if(rig!=graph_.blocks.end()) {
        // Online T_BR is shared by the rigid assembly. Express the original
        // camera extrinsics in the updated sensor frame with one common change;
        // per-camera setters would invalidate unchanged image-space FOV masks.
        const auto T_new_old=T_SB*blockPose(rig->second)*source->T_SC(0)->inverse();
        result->transformCameraSystemReferenceFrame(T_new_old);
    }
    synchronizeFrame(id,*result);
    return result;
}
void SlidingWindowManager::synchronizeFrame(StateId id,MultiFrame& target) const {
    const auto source=frames_.measurements(id);
    if(!source || target.id()!=id.value() || target.timestamp()!=source->timestamp()
        || target.numFrames()!=source->numFrames()) throw std::invalid_argument("association target frame identity differs");
    for(size_t c=0;c<target.numFrames();++c)
        if(target.numKeypoints(c)!=source->numKeypoints(c) || target.geometry(c).get()!=source->geometry(c).get())
            throw std::invalid_argument("association target feature layout differs");
    for(size_t c=0;c<target.numFrames();++c) for(size_t k=0;k<target.numKeypoints(c);++k) {
        target.setLandmarkId(c,k,0); target.setLandmark(c,k,Eigen::Vector4d::Zero(),false);
    }
    std::map<LandmarkId,MapPoint2> exportedPoints;
    for(const auto& key:landmarks_.inFrame(id)) {
        const auto observation=landmarks_.find(key); if(!observation) continue;
        auto cached=exportedPoints.find(observation->landmark);
        if(cached==exportedPoints.end()) {
            // Frame export needs coordinates/identity, not a copy of every other
            // observation of this point. Keep the full getLandmark() API unchanged.
            const auto metadata=landmarks_.find(observation->landmark); if(!metadata) continue;
            MapPoint2 point{}; point.id=metadata->id; point.point.setZero();
            point.isInitialised=metadata->initialized;
            if(metadata->initialized) {
                Eigen::Vector3d world; std::string error;
                if(!pointWorld(graph_,landmarks_,metadata->id,world,error)) continue;
                point.point.head<3>()=world; point.point[3]=1;
            }
            cached=exportedPoints.emplace(observation->landmark,std::move(point)).first;
        }
        const auto& point=cached->second;
        target.setLandmarkId(key.cameraIndex,key.keypointIndex,point.id.value());
        target.setLandmark(key.cameraIndex,key.keypointIndex,point.point,point.isInitialised);
    }
}
bool SlidingWindowManager::valid(std::string& error) const {
    if(!graph_.valid(&error) || !landmarks_.valid(error)) return false;
    if(order_.size()!=frames_.epochs_.size() || order_.size()!=frames_.records_.size()) {
        error="window frame cardinality mismatch"; return false;
    }
    for(auto id:order_) if(!frames_.epochs_.count(id) || !frames_.find(id)
        || !graph_.blocks.count(BlockKey::pose(id)) || !graph_.blocks.count(BlockKey::speedBias(id))) {
        error="frame metadata/state blocks disagree"; return false;
    }
    for(const auto& item:graph_.blocks) {
        if(item.first.isState() && !frames_.epochs_.count(StateId(item.first.id))) { error="state block has no frame"; return false; }
        if(item.first.kind==BlockKind::Landmark) {
            auto record=landmarks_.find(LandmarkId(item.first.id));
            if(!record || !record->initialized || !item.second.owner || item.second.owner->value()!=record->anchor.frameId
                || item.second.value.size()!=1 || item.second.manifold || item.second.value[0]<options_.minInverseDepth) {
                error="landmark metadata/rho/anchor owner mismatch"; return false;
            }
        }
    }
    for(const auto& item:landmarks_.landmarks_) {
        const auto& point=item.second; const auto anchor=landmarks_.find(point.anchor);
        if(!anchor || anchor->landmark!=point.id || !frames_.contains(point.anchor)
            || !point.anchorBearing.allFinite() || std::abs(point.anchorBearing.norm()-1.0)>1e-6
            || point.initialized!=(graph_.blocks.count({BlockKind::Landmark,point.id.value()})!=0)) {
            error="landmark anchor/parameter is missing"; return false;
        }
    }
    // Observation keys are ordered by frame. Resolve the immutable feature layout
    // once per frame instead of repeating record/epoch tree lookups for every ray.
    uint64_t checkedFrame=0;
    const MultiFrame* checkedMeasurements=nullptr;
    for(const auto& item:landmarks_.observations_) {
        const auto& observation=item.second;
        if(item.first.frameId!=checkedFrame) {
            checkedFrame=item.first.frameId;
            const auto record=frames_.find(StateId(checkedFrame));
            checkedMeasurements=record?record->measurements.get():nullptr;
        }
        if(!checkedMeasurements || item.first.cameraIndex>=checkedMeasurements->numFrames()
            || item.first.keypointIndex>=checkedMeasurements->numKeypoints(item.first.cameraIndex)) {
            error="observation key is outside attached frame"; return false;
        }
        if(observation.use==ObservationUse::LiveFactor) {
            auto factor=graph_.factors.find(observation.factorId);
            const auto& point=landmarks_.landmarks_.at(observation.landmark);
            unsigned dependencies=0;
            if(factor!=graph_.factors.end()) for(auto key:factor->second.keys) {
                if(key==BlockKey::pose(StateId(point.anchor.frameId))) dependencies|=1;
                if(key==BlockKey::pose(StateId(item.first.frameId))) dependencies|=2;
                if(key==BlockKey{BlockKind::CameraExtrinsic,0}) dependencies|=4;
                if(key==BlockKey{BlockKind::Landmark,observation.landmark.value()}) dependencies|=8;
            }
            if(factor==graph_.factors.end() || factor->second.kind!=FactorKind::Visual
                || factor->second.cost->num_residuals()!=2
                || factor->second.sensorEpoch.value()!=item.first.frameId
                || (factor->second.lifecycle!=FactorLifecycle::Active && factor->second.lifecycle!=FactorLifecycle::Pending)
                || dependencies!=15) {
                error="observation/factor binding mismatch"; return false;
            }
        }
    }
    for(const auto& item:graph_.factors) if(item.second.kind==FactorKind::Visual
        && (item.second.lifecycle==FactorLifecycle::Active || item.second.lifecycle==FactorLifecycle::Pending)) {
        for(auto key:item.second.keys) if(key.kind==BlockKind::Landmark && !landmarks_.byFactor_.count(item.first)) {
            error="visual landmark factor has no managed observation"; return false;
        }
    }
    for(const auto& key:landmarks_.absorbedKeys_) if(!frames_.epochs_.count(StateId(key.frameId))) {
        error="consumed-observation tombstone outlived its frame"; return false;
    }
    return true;
}
} }
