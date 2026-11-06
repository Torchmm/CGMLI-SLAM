#include "LandmarkStore.hpp"
#include <Eigen/LU>
#include <cmath>
#include <algorithm>

namespace cgmli { namespace backend {
namespace {
const LandmarkStore::ObservationKeys emptyKeys;
void insertKey(LandmarkStore::ObservationKeys& keys,const KeypointIdentifier& key) {
    keys.insert(std::lower_bound(keys.begin(),keys.end(),key),key);
}
void eraseKey(LandmarkStore::ObservationKeys& keys,const KeypointIdentifier& key) {
    const auto it=std::lower_bound(keys.begin(),keys.end(),key);
    if(it!=keys.end() && *it==key) keys.erase(it);
}
}
const WindowLandmark* LandmarkStore::find(LandmarkId id) const {
    auto it=landmarks_.find(id); return it==landmarks_.end()?nullptr:&it->second;
}
const WindowObservation* LandmarkStore::find(const KeypointIdentifier& key) const {
    auto it=observations_.find(key); return it==observations_.end()?nullptr:&it->second;
}
const LandmarkStore::ObservationKeys& LandmarkStore::inFrame(StateId id) const {
    auto it=byFrame_.find(id); return it==byFrame_.end()?emptyKeys:it->second;
}
const LandmarkStore::ObservationKeys& LandmarkStore::ofLandmark(LandmarkId id) const {
    auto it=byLandmark_.find(id); return it==byLandmark_.end()?emptyKeys:it->second;
}
const WindowObservation* LandmarkStore::inCamera(LandmarkId id,StateId frame,size_t camera) const {
    const auto& keys=ofLandmark(id);
    const auto it=std::lower_bound(keys.begin(),keys.end(),KeypointIdentifier(frame.value(),camera,0));
    return it!=keys.end() && it->frameId==frame.value() && it->cameraIndex==camera?find(*it):nullptr;
}
bool LandmarkStore::insert(WindowLandmark landmark,std::string& error) {
    if(!landmark.id.isInitialised() || landmark.id.value()<=largestId_ || !landmark.anchor.frameId
        || !landmark.anchorBearing.allFinite() || std::abs(landmark.anchorBearing.norm()-1.0)>1e-6
        || !std::isfinite(landmark.quality) || landmark.quality<0) {
        error="invalid landmark metadata or non-increasing/reused landmark ID"; return false;
    }
    largestId_=landmark.id.value(); landmarks_.emplace(landmark.id,std::move(landmark)); return true;
}
bool LandmarkStore::insert(WindowObservation observation,std::string& error) {
    if(!landmarks_.count(observation.landmark) || observations_.count(observation.key) || consumed(observation.key)
        || !observation.bearing.allFinite() || std::abs(observation.bearing.norm()-1.0)>1e-6
        || !observation.sqrtInformation.allFinite() || std::abs(observation.sqrtInformation.determinant())<1e-20
        || !observation.bearingVelocity.allFinite() || !std::isfinite(observation.referenceTimeOffset)
        || std::abs(observation.bearing.dot(observation.bearingVelocity))>1e-7*(1.0+observation.bearingVelocity.norm())) {
        error="invalid/duplicate observation or non-tangent bearing velocity"; return false;
    }
    // A point may appear in several cameras, but at most once per camera and epoch.
    if(inCamera(observation.landmark,StateId(observation.key.frameId),observation.key.cameraIndex)) {
        error="landmark already observed in this camera/epoch"; return false;
    }
    insertKey(byFrame_[StateId(observation.key.frameId)],observation.key);
    insertKey(byLandmark_[observation.landmark],observation.key);
    observations_.emplace(observation.key,std::move(observation)); return true;
}
void LandmarkStore::bind(const KeypointIdentifier& key,uint64_t factor) {
    auto& observation=observations_.at(key); observation.factorId=factor;
    observation.use=ObservationUse::LiveFactor; byFactor_[factor]=key;
}
void LandmarkStore::eraseObservation(const KeypointIdentifier& key) {
    auto it=observations_.find(key); if(it==observations_.end()) return;
    auto frame=byFrame_.find(StateId(key.frameId));
    if(frame!=byFrame_.end()) { eraseKey(frame->second,key); if(frame->second.empty()) byFrame_.erase(frame); }
    auto landmark=byLandmark_.find(it->second.landmark);
    if(landmark!=byLandmark_.end()) { eraseKey(landmark->second,key); if(landmark->second.empty()) byLandmark_.erase(landmark); }
    if(it->second.use==ObservationUse::LiveFactor) byFactor_.erase(it->second.factorId);
    observations_.erase(it);
}
void LandmarkStore::eraseLandmark(LandmarkId id,bool consumed) {
    const auto keys=ofLandmark(id);
    if(consumed) absorbedKeys_.insert(keys.begin(),keys.end());
    for(const auto& key:keys) eraseObservation(key);
    landmarks_.erase(id);
}
void LandmarkStore::eraseFrame(StateId id) {
    const auto keys=inFrame(id);
    // Remove the frame bucket once. Per-observation removal only edits each small
    // landmark bucket; it must not shift a large frame vector once per key.
    byFrame_.erase(id);
    for(const auto& key:keys) eraseObservation(key);
    // Tombstones use the same frame-major ordering as observations. Erase only
    // this frame's range, rather than scanning every retained frame's history.
    auto first=absorbedKeys_.lower_bound(KeypointIdentifier(id.value(),0,0)),last=first;
    while(last!=absorbedKeys_.end() && last->frameId==id.value()) ++last;
    absorbedKeys_.erase(first,last);
}
void LandmarkStore::factorRemoved(uint64_t factor,bool absorbed) {
    auto it=byFactor_.find(factor); if(it==byFactor_.end()) return;
    const auto key=it->second;
    if(!absorbed) {
        const auto& obs=observations_.at(key);
        if(landmarks_.at(obs.landmark).anchor==key) {
            // A reanchored coordinate ray can remain a reference after its optional
            // image residual is rejected. Do not leave the geometric anchor dangling.
            auto& reference=observations_.at(key); reference.use=ObservationUse::TrackingOnly;
            reference.factorId=0; byFactor_.erase(it); return;
        }
        eraseObservation(key); return;
    }
    absorbedKeys_.insert(key);
    observations_.at(key).use=ObservationUse::Absorbed; byFactor_.erase(it);
}
bool LandmarkStore::valid(std::string& error) const {
    size_t frameCount=0,landmarkCount=0,liveCount=0;
    auto orderedObservation=observations_.begin();
    for(const auto& item:byFrame_) for(const auto& key:item.second) {
        // KeypointIdentifier is ordered by frame/camera/key. The concatenated
        // frame index therefore admits one linear merge against the primary map.
        if(orderedObservation==observations_.end() || !(orderedObservation->first==key)
            || key.frameId!=item.first.value()) { error="invalid frame observation index"; return false; }
        ++orderedObservation; ++frameCount;
    }
    auto orderedLandmark=landmarks_.begin();
    for(const auto& item:byLandmark_) {
      if(item.second.empty() || orderedLandmark==landmarks_.end() || orderedLandmark->first!=item.first) {
          error="observation has missing landmark or empty index"; return false;
      }
      ++orderedLandmark;
      for(size_t i=1;i<item.second.size();++i) {
          const auto& a=item.second[i-1]; const auto& b=item.second[i];
          if(!(a<b) || (a.frameId==b.frameId && a.cameraIndex==b.cameraIndex)) {
              error="unordered/duplicate landmark camera index"; return false;
          }
      }
      for(const auto& key:item.second) {
        auto obs=find(key); if(!obs || obs->landmark!=item.first) { error="invalid landmark observation index"; return false; } ++landmarkCount;
      }
    }
    if(orderedLandmark!=landmarks_.end()) { error="landmark has no observations"; return false; }
    for(const auto& item:observations_) {
        const auto& obs=item.second;
        if(!(obs.key==item.first)) { error="observation payload key differs from its table key"; return false; }
        // Both reverse indexes above already prove injective membership: a key
        // can match exactly one frame and exactly one landmark. Their checked
        // cardinalities below prove complete coverage, without four more tree
        // lookups per observation. Landmark existence was checked by the sorted
        // outer-index merge above, so there is no third lookup for every view.
        if(obs.use==ObservationUse::LiveFactor) {
            auto f=byFactor_.find(obs.factorId);
            if(!obs.factorId || f==byFactor_.end() || !(f->second==item.first)) { error="observation factor index mismatch"; return false; }
            ++liveCount;
        }
        if((obs.use==ObservationUse::TrackingOnly && obs.factorId)
            || (obs.use==ObservationUse::Absorbed && !consumed(item.first))
            || (obs.use==ObservationUse::LiveFactor && consumed(item.first))) {
            error="observation lifecycle/consumption mismatch"; return false;
        }
    }
    if(frameCount!=observations_.size() || landmarkCount!=observations_.size() || liveCount!=byFactor_.size()) {
        error="observation index cardinality mismatch"; return false;
    }
    return true;
}
} }
