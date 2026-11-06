#pragma once
#include "WindowDataTypes.hpp"
#include <map>
#include <set>
#include <string>

namespace cgmli { namespace backend {
class SlidingWindowManager;

/// Authoritative geometric identities and observation table. No optimized coordinate
/// values are duplicated here; graph rho and live camera poses produce world estimates.
class LandmarkStore {
public:
    /// Sorted unique keys. Small per-track vectors keep window checkpoints contiguous
    /// instead of allocating a red-black-tree node for every reverse-index entry.
    using ObservationKeys=std::vector<KeypointIdentifier>;
    const std::map<LandmarkId,WindowLandmark>& landmarks() const { return landmarks_; }
    const std::map<KeypointIdentifier,WindowObservation>& observations() const { return observations_; }
    const WindowLandmark* find(LandmarkId id) const;
    const WindowObservation* find(const KeypointIdentifier& key) const;
    const ObservationKeys& inFrame(StateId frame) const;
    const ObservationKeys& ofLandmark(LandmarkId id) const;
    const WindowObservation* inCamera(LandmarkId id,StateId frame,size_t camera) const;
    size_t size() const { return landmarks_.size(); }
    bool consumed(const KeypointIdentifier& key) const { return absorbedKeys_.count(key)!=0; }
    bool valid(std::string& error) const;
private:
    friend class SlidingWindowManager;
    bool insert(WindowLandmark landmark,std::string& error);
    bool insert(WindowObservation observation,std::string& error);
    void eraseObservation(const KeypointIdentifier& key);
    void eraseLandmark(LandmarkId id,bool consumed = false);
    void eraseFrame(StateId id);
    void bind(const KeypointIdentifier& key,uint64_t factorId);
    void factorRemoved(uint64_t factorId,bool absorbed);
    std::map<LandmarkId,WindowLandmark> landmarks_;
    std::map<KeypointIdentifier,WindowObservation> observations_;
    std::map<StateId,ObservationKeys> byFrame_;
    std::map<LandmarkId,ObservationKeys> byLandmark_;
    std::map<uint64_t,KeypointIdentifier> byFactor_;
    // Tombstones are scoped to active frames, not an unbounded history. They prevent
    // an erased historical observation from being recycled into another independent cost.
    std::set<KeypointIdentifier> absorbedKeys_;
    uint64_t largestId_=0;
};
} }
