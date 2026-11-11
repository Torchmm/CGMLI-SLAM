#include "SlidingWindowManager.hpp"
#include "WindowVisualFactor.hpp"
#include <factor/BodyImuFactor.hpp>
#include <visual/MultiFrame.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>

namespace cgmli {
    namespace backend {
        SlidingWindowManager::SlidingWindowManager(const BackendOptions& o) : options_(o), marginalizer_(o) { options_.validate(); }
        SlidingWindowManager::SlidingWindowManager(const SlidingWindowManager& source,BackendGraph&& graph,LandmarkStore&& landmarks,FrameStore&& frames)
            :options_(source.options_),marginalizer_(source.marginalizer_),graph_(std::move(graph)),order_(source.order_),
             frames_(std::move(frames)),landmarks_(std::move(landmarks)),history_(source.history_),largestEpochId_(source.largestEpochId_),
             largestTimestampNs_(source.largestTimestampNs_),nextFactorId_(source.nextFactorId_),deferPendingActivation_(source.deferPendingActivation_) {}
        void SlidingWindowManager::recordEvent(FactorEvent event) {
            history_.push_back(std::move(event));
            while(history_.size()>options_.historySize) history_.pop_front();
        }
        void SlidingWindowManager::record(const FactorRecord& f) {
            // Do not retain cost objects or sensor payloads in history. Its bounded event
            // queue is diagnostic only and is never replayed as optimization evidence.
            recordEvent({f.id,f.kind,f.lifecycle,f.reason});
        }
        void SlidingWindowManager::collectTerminalFactors(std::vector<KeypointIdentifier>* rejectedVisual) {
            for (auto it = graph_.factors.begin(); it != graph_.factors.end();) {
                if (it->second.lifecycle == FactorLifecycle::Rejected || it->second.lifecycle == FactorLifecycle::Marginalized) {
                    if(rejectedVisual && it->second.lifecycle==FactorLifecycle::Rejected) {
                        const auto observation=landmarks_.byFactor_.find(it->first);
                        if(observation!=landmarks_.byFactor_.end()) rejectedVisual->push_back(observation->second);
                    }
                    landmarks_.factorRemoved(it->first,it->second.lifecycle==FactorLifecycle::Marginalized);
                    record(it->second); it = graph_.factors.erase(it);
                }
                else ++it;
            }
            if(rejectedVisual) std::sort(rejectedVisual->begin(),rejectedVisual->end());
        }
        bool SlidingWindowManager::ready(const FactorRecord& f) const {
            // Both epoch metadata and parameter buffers must exist. Checking only keys
            // would miss explicit observation/anchor dependencies carried by the factor.
            for (auto id : f.epochs) if (!frames_.epochs_.count(id)) return false;
            for (auto key : f.keys) if (!graph_.blocks.count(key)) return false;
            return true;
        }
        bool SlidingWindowManager::healthy(const FactorRecord& f) const {
            // sensorEpoch identifies the observer/measurement epoch, not necessarily an
            // inverse-depth anchor. Degraded measurements remain eligible for optimization;
            // the facade independently applies the stricter calibration-unlock policy.
            if (f.kind != FactorKind::Visual && f.kind != FactorKind::Lidar) return true;
            auto e = frames_.epochs_.find(f.sensorEpoch);
            if (e == frames_.epochs_.end()) return false;
            return (f.kind == FactorKind::Visual ? e->second.visual : e->second.lidar) != SensorHealth::Invalid;
        }
        void SlidingWindowManager::activatePending() {
            // A cost may arrive before its future state or landmark. Once dependencies
            // exist, validate their actual dimensions before passing pointers to Ceres.
            for (auto& item : graph_.factors) {
                auto& f = item.second;
                if (f.lifecycle != FactorLifecycle::Pending || !ready(f)) continue;
                bool sizes = true;
                std::set<StateId> deps(f.epochs.begin(), f.epochs.end());
                for (size_t i = 0; i < f.keys.size(); ++i) {
                    sizes = sizes && graph_.blocks.at(f.keys[i]).value.size() == f.cost->parameter_block_sizes()[i];
                    const auto& p = graph_.blocks.at(f.keys[i]);
                    if (p.owner) deps.insert(*p.owner);
                }
                f.epochs.assign(deps.begin(), deps.end());
                if (!sizes || !healthy(f)) {
                    f.lifecycle = FactorLifecycle::Rejected; f.reason = sizes ? "sensor invalid" : "pending parameter size mismatch";
                }
                else {
                    if(f.imuReplay) factor::configureBodyImuDuration(f,options_.maxImuIntervalSeconds,
                        *estimate(StateId(f.keys[0].id)),*estimate(StateId(f.keys[2].id)));
                    f.lifecycle = FactorLifecycle::Active; record(f);
                }
            }
            collectTerminalFactors();
        }
        bool SlidingWindowManager::append(const StateEstimate& s, std::string& error) {
            // Never reuse an evicted slot's ID or reinterpret its timestamp after a td update.
            // The one extra slot is needed to optimize the new state before choosing a victim.
            if (!s.id.isInitialised() || s.id.value() <= largestEpochId_ || (largestEpochId_ && s.timestamp.toNSec() <= largestTimestampNs_)) {
                error = "FusionEpoch IDs and physical timestamps must strictly increase"; return false;
            }
            if (size() >= options_.windowSize + 1) { error = "window insertion slot occupied; solve/marginalize before appending"; return false; }
            auto p = ParameterBlock::pose(BlockKey::pose(s.id), s.T_WB, s.id);
            auto v = ParameterBlock::euclidean(BlockKey::speedBias(s.id), s.speedBias, s.id);
            if (!p.valid(&error) || !v.valid(&error)) return false;
            graph_.blocks.emplace(p.key, std::move(p)); graph_.blocks.emplace(v.key, std::move(v));
            // Sensor arrival is independent: a state can be IMU-only, but cannot become a
            // normal visual/LiDAR keyframe while both exteroceptive modalities are invalid.
            const bool keyframe = s.keyframe && (s.visual != SensorHealth::Invalid || s.lidar != SensorHealth::Invalid);
            frames_.insert(EpochNode{ s.id,s.timestamp,s.visual,s.lidar,keyframe }); order_.push_back(s.id);
            const auto td=graph_.blocks.find({BlockKind::CameraTimeOffset,0});
            frames_.records_.at(s.id).referenceCameraTimeOffset=td==graph_.blocks.end()?0.0:td->second.value[0];
            largestEpochId_ = s.id.value(); largestTimestampNs_ = s.timestamp.toNSec();
            activatePending(); return true;
        }
        bool SlidingWindowManager::addParameter(ParameterBlock p, std::string& error) {
            if(p.key.kind==BlockKind::Landmark) { error="use addLandmark/initializeLandmark to keep geometry and graph consistent"; return false; }
            if (!p.valid(&error)) return false;
            if (p.key.isState() || graph_.blocks.count(p.key)) { error = "state insertion/replacing existing parameter is forbidden"; return false; }
            if (p.owner && !frames_.epochs_.count(*p.owner)) { error = "local parameter owner is outside active window"; return false; }
            if (p.key.kind == BlockKind::Landmark && !p.owner) { error = "landmark must declare its anchor owner"; return false; }
            graph_.blocks.emplace(p.key, std::move(p)); activatePending(); return true;
        }
        bool SlidingWindowManager::configureParameter(BlockKey k, bool constant, const Eigen::VectorXd& lower,
            const Eigen::VectorXd& upper, std::string& error) {
            // Validate a copy first so invalid bounds cannot partially alter a live block.
            auto p = graph_.blocks.find(k);
            if (p == graph_.blocks.end()) { error = "parameter not found"; return false; }
            auto candidate = p->second; candidate.constant = constant; candidate.lower = lower; candidate.upper = upper;
            if (!candidate.valid(&error)) return false;
            p->second = std::move(candidate); return true;
        }
        FactorSubmission SlidingWindowManager::addFactor(FactorRecord f) { return submitFactor(std::move(f)); }
        FactorSubmission SlidingWindowManager::submitFactor(FactorRecord f,bool managedObservation) {
            // Even rejected submissions consume an ID, keeping event logs unambiguous.
            FactorSubmission out;
            if (!nextFactorId_) throw std::overflow_error("factor ID exhausted");
            f.id = nextFactorId_++; out.id = f.id;
            const auto reject = [&](const char* text) {
                f.lifecycle = FactorLifecycle::Rejected; f.reason = text; record(f); out.message = text; return out;
                };
            if(f.kind==FactorKind::Visual && !managedObservation) for(auto key:f.keys)
                if(key.kind==BlockKind::Landmark) return reject("use addObservation for visual landmark factors");
            for(auto key:f.keys) if(key.kind==BlockKind::Landmark) {
                const auto point=landmarks_.find(LandmarkId(key.id));
                if(!point || !point->initialized) return reject("landmark factor requires a managed initialized point");
            }
            if (!f.cost || f.cost->num_residuals() <= 0 || f.keys.empty()
                || f.keys.size() != f.cost->parameter_block_sizes().size() || !(f.nisThreshold > 0)
                || f.kind == FactorKind::Marginalization) return reject("invalid factor contract");
            if(f.imuReplay && (f.kind!=FactorKind::Imu || f.keys.size()!=4 || f.cost->num_residuals()!=15
                || f.keys[0].kind!=BlockKind::Pose || f.keys[1].kind!=BlockKind::SpeedBias
                || f.keys[2].kind!=BlockKind::Pose || f.keys[3].kind!=BlockKind::SpeedBias
                || f.keys[0].id!=f.keys[1].id || f.keys[2].id!=f.keys[3].id
                || !f.imuReplay->parameters || f.imuReplay->samples.size()<2
                || f.imuReplay->end<=f.imuReplay->start))
                return reject("invalid native IMU layout");
            if(!f.solveEnabled && !f.imuReplay) return reject("only native IMU intervals may be suppressed");
            std::set<BlockKey> unique;
            // Infer state and anchor dependencies from keys/owners in addition to the
            // caller's declared epochs. This prevents a local variable outliving its anchor.
            std::set<StateId> deps(f.epochs.begin(), f.epochs.end());
            for (size_t i = 0; i < f.keys.size(); ++i) {
                auto k = f.keys[i];
                // One Ceres residual cannot list the same parameter block twice. Same-epoch
                // cross-camera factors must combine those derivatives in their factory.
                if (!unique.insert(k).second) return reject("duplicate parameter block in one factor");
                if (k.isState()) deps.insert(StateId(k.id));
                auto p = graph_.blocks.find(k);
                if (p != graph_.blocks.end()) {
                    if (p->second.owner) deps.insert(*p->second.owner);
                    if (p->second.value.size() != f.cost->parameter_block_sizes()[i]) return reject("factor size mismatch");
                }
            }
            if (f.kind == FactorKind::Visual || f.kind == FactorKind::Lidar) {
                if (!f.sensorEpoch.isInitialised()) return reject("sensor factor requires sensorEpoch");
                deps.insert(f.sensorEpoch);
            }
            for (auto id : deps) {
                // An absent ID beyond the high-water mark can still arrive; an absent
                // past ID cannot. Marginalization priors are never reopened for late data.
                if (!id.isInitialised()) return reject("zero epoch dependency");
                if (!frames_.epochs_.count(id) && id.value() <= largestEpochId_) return reject("late factor targets an evicted/skipped epoch");
                // An epoch dependency must actually occur in the factor graph, not be baked as a fixed pose.
                bool connected = false;
                for (auto k : f.keys) {
                    auto p = graph_.blocks.find(k);
                    connected = connected || (k.isState() && k.id == id.value())
                        || (p != graph_.blocks.end() && p->second.owner && *p->second.owner == id);
                }
                if (!connected) return reject("epoch dependency has no connected parameter");
            }
            f.epochs.assign(deps.begin(), deps.end());
            const bool active = ready(f);
            if (active && !healthy(f)) return reject("sensor invalid");
            if (!active) {
                // Bound storage even when a producer repeatedly references dependencies
                // that never arrive. Pending costs do not contribute to the current solve.
                size_t pending = 0;
                for (const auto& item : graph_.factors) if (item.second.lifecycle == FactorLifecycle::Pending) ++pending;
                if (pending >= options_.maxPendingFactors) return reject("pending factor budget exceeded");
            }
            f.lifecycle = active ? FactorLifecycle::Active : FactorLifecycle::Pending;
            if(active && f.imuReplay) factor::configureBodyImuDuration(f,options_.maxImuIntervalSeconds,
                *estimate(StateId(f.keys[0].id)),*estimate(StateId(f.keys[2].id)));
            out.lifecycle = f.lifecycle; record(f); graph_.factors.emplace(f.id, std::move(f)); return out;
        }
        bool SlidingWindowManager::rejectFactor(FactorId id, const std::string& reason) {
            auto f = graph_.factors.find(id); if (f == graph_.factors.end()) return false;
            f->second.lifecycle = FactorLifecycle::Rejected; f->second.reason = reason; collectTerminalFactors(); return true;
        }
        bool SlidingWindowManager::setHealth(StateId id, SensorHealth v, SensorHealth l) {
            auto e = frames_.epochs_.find(id); if (e == frames_.epochs_.end()) return false;
            const bool invalidated=(v==SensorHealth::Invalid && e->second.visual!=SensorHealth::Invalid)
                || (l==SensorHealth::Invalid && e->second.lidar!=SensorHealth::Invalid);
            e->second.visual = v; e->second.lidar = l;
            if (v == SensorHealth::Invalid && l == SensorHealth::Invalid) e->second.keyframe = false;
            // Valid/degraded updates cannot invalidate a live measurement. In
            // particular the unused LiDAR remaining Invalid needs no graph scan.
            if(invalidated) for (auto& item : graph_.factors) {
                auto& f = item.second;
                if (f.sensorEpoch == id && !healthy(f) && f.lifecycle == FactorLifecycle::Active) {
                    f.lifecycle = FactorLifecycle::Rejected; f.reason = "sensor became invalid";
                }
            }
            if(!deferPendingActivation_) { collectTerminalFactors(); activatePending(); } return true;
        }
        bool SlidingWindowManager::setKeyframe(StateId id, bool keyframe) {
            auto e = frames_.epochs_.find(id); if (e == frames_.epochs_.end()) return false;
            if (keyframe && e->second.visual == SensorHealth::Invalid && e->second.lidar == SensorHealth::Invalid) return false;
            e->second.keyframe = keyframe; return true;
        }
        MarginalizationPolicy SlidingWindowManager::choosePolicy() const {
            // The frontend decides whether the penultimate epoch deserves long-term
            // retention. A non-keyframe is removed while the newest prediction survives.
            if (size() < 2 || frames_.epochs_.at(order_[size() - 2]).keyframe) return MarginalizationPolicy::MARGIN_OLD;
            return MarginalizationPolicy::MARGIN_SECOND_NEW;
        }
        bool SlidingWindowManager::canOptimizeLatestTwo() const {
            if(!needsMarginalization() || size()<3 || choosePolicy()!=MarginalizationPolicy::MARGIN_SECOND_NEW) return false;
            const auto left=order_[size()-3],middle=order_[size()-2],right=order_.back();
            const auto& latest=frames_.epochs_.at(right);
            if(latest.keyframe || latest.visual!=SensorHealth::Valid || latest.lidar!=SensorHealth::Invalid) return false;
            const auto& leftEpoch=frames_.epochs_.at(left);
            const auto& middleEpoch=frames_.epochs_.at(middle);
            if((latest.timestamp-leftEpoch.timestamp).toSec()>options_.maxImuIntervalSeconds) return false;
            const auto middlePose=BlockKey::pose(middle),middleSpeed=BlockKey::speedBias(middle);
            if(graph_.prior && std::find(graph_.prior->keys().begin(),graph_.prior->keys().end(),middleSpeed)
                !=graph_.prior->keys().end()) return false;
            const auto interval=[](const FactorRecord& factor,const EpochNode& from,const EpochNode& to) {
                return factor.solveEnabled && factor.imuReplay && factor.imuReplay->completeSamples
                    && factor.imuReplay->start==from.timestamp && factor.imuReplay->end==to.timestamp
                    && factor.keys.size()==4 && factor.keys.front()==BlockKey::pose(from.id)
                    && factor.keys[2]==BlockKey::pose(to.id);
            };
            bool incoming=false,outgoing=false;
            size_t imuEdges=0;
            for(const auto& item:graph_.factors) {
                const auto& f=item.second;
                if(f.lifecycle!=FactorLifecycle::Active || f.kind==FactorKind::Visual) continue;
                if(std::none_of(f.keys.begin(),f.keys.end(),[&](BlockKey k) {return k==middlePose || k==middleSpeed;}))
                    continue;
                if(f.kind!=FactorKind::Imu) return false;
                ++imuEdges;
                incoming=incoming || interval(f,leftEpoch,middleEpoch);
                outgoing=outgoing || interval(f,middleEpoch,latest);
            }
            return imuEdges==2 && incoming && outgoing;
        }
        std::optional<StateEstimate> SlidingWindowManager::estimate(StateId id) const {
            auto e = frames_.epochs_.find(id); if (e == frames_.epochs_.end()) return std::nullopt;
            StateEstimate s;
            s.id = id; s.timestamp = e->second.timestamp; s.visual = e->second.visual; s.lidar = e->second.lidar; s.keyframe = e->second.keyframe;
            s.T_WB = blockPose(graph_.blocks.at(BlockKey::pose(id)));
            s.speedBias = graph_.blocks.at(BlockKey::speedBias(id)).value;
            return s;
        }
        bool SlidingWindowManager::prepareEviction(MarginalizationReport& out,BackendGraph& graph,
            LandmarkStore& store,FrameStore& frames,std::vector<FactorEvent>& discarded,
            std::map<LandmarkId,std::string>& reasons,std::string& error) {
            const bool redundant=out.policy==MarginalizationPolicy::MARGIN_SECOND_NEW;
            // Chart resets may replace ONLY live measurement costs. Freeze every
            // block used by the incoming FEJ prior until the later Schur step.
            const auto incomingPrior=graph.prior;
            std::vector<Eigen::VectorXd> protectedValues;
            if(incomingPrior) for(auto key:incomingPrior->keys()) protectedValues.push_back(graph.blocks.at(key).value);
            out.resetPriorProtectedBlocks=protectedValues.size();
            std::set<BlockKey> priorKeys;
            if(graph.prior) priorKeys.insert(graph.prior->keys().begin(),graph.prior->keys().end());
            if(redundant && order_.size()>=3) {
                const auto left=order_[order_.size()-3],right=order_.back();
                std::vector<FactorId> incident;
                bool externalStateConstraint=false;
                for(const auto& item:graph.factors) if(item.second.lifecycle==FactorLifecycle::Active
                    && item.second.kind!=FactorKind::Imu && item.second.kind!=FactorKind::Visual)
                    for(auto key:item.second.keys) if(key.isState() && key.id==out.removed.value()) externalStateConstraint=true;
                for(const auto& item:graph.factors) if(item.second.kind==FactorKind::Imu
                    && item.second.lifecycle==FactorLifecycle::Active
                    && std::find(item.second.keys.begin(),item.second.keys.end(),BlockKey::pose(out.removed))!=item.second.keys.end())
                    incident.push_back(item.first);
                // A native second-newest SB must not already carry an absorbed IMU
                // bridge. Mixing that bridge with replay would double-count samples.
                // External/non-replayable factors retain the exact Schur fallback.
                out.imuMode=incident.empty()?"none":"prior_bridge";
                if(incident.size()==2 && !externalStateConstraint && !priorKeys.count(BlockKey::speedBias(out.removed))) {
                    auto a=incident[0],b=incident[1];
                    if(graph.factors.at(a).keys.front()!=BlockKey::pose(left)) std::swap(a,b);
                    const auto& first=graph.factors.at(a); const auto& second=graph.factors.at(b);
                    if(first.imuReplay && second.imuReplay) {
                        FactorRecord merged;
                        if(!factor::mergeBodyImuRecords(first,second,*estimate(left),*estimate(right),
                            options_.maxImuSamplesPerEpoch,merged,error,options_.maxImuIntervalSeconds,options_.incrementalImu)) return false;
                        if(!nextFactorId_) { error="IMU merge factor ID exhausted"; return false; }
                        merged.id=nextFactorId_++; merged.lifecycle=FactorLifecycle::Active;
                        frames.records_.at(right).imuSamples=std::make_shared<const ImuMeasurementDeque>(merged.imuReplay->samples);
                        discarded.push_back({a,FactorKind::Imu,FactorLifecycle::Rejected,"replaced by concatenated IMU interval"});
                        discarded.push_back({b,FactorKind::Imu,FactorLifecycle::Rejected,"replaced by concatenated IMU interval"});
                        graph.factors.erase(a); graph.factors.erase(b);
                        const bool enabled=merged.solveEnabled;
                        const bool appended=merged.imuReplay->integrationMode=="incremental";
                        if(enabled) { if(appended) ++out.incrementalImuMerges; else ++out.fullImuMerges; }
                        graph.factors.emplace(merged.id,std::move(merged));
                        out.replacedImuFactors=2; out.imuMode=!enabled?"suppressed_duration":appended?"incremental":"reintegrated";
                    }
                }
            }
            std::set<LandmarkId> externalLandmarks;
            for(const auto& item:graph.factors) if(!store.byFactor_.count(item.first)
                || item.second.lifecycle!=FactorLifecycle::Active)
                for(auto key:item.second.keys) if(key.kind==BlockKind::Landmark) externalLandmarks.insert(LandmarkId(key.id));
            for(auto& item:store.landmarks_) {
                auto& landmark=item.second;
                if(landmark.anchor.frameId!=out.removed.value()) continue;
                const BlockKey rho{BlockKind::Landmark,item.first.value()};
                const auto oldAnchor=store.find(landmark.anchor);
                std::vector<KeypointIdentifier> surviving;
                bool eligible=oldAnchor && oldAnchor->bearingVelocity.squaredNorm()==0;
                for(const auto& key:store.ofLandmark(item.first)) {
                    const auto& observation=*store.find(key);
                    eligible=eligible && observation.use!=ObservationUse::Absorbed && observation.bearingVelocity.squaredNorm()==0;
                    if(key.frameId!=out.removed.value()
                        && frames.epochs_.at(StateId(key.frameId)).visual!=SensorHealth::Invalid) surviving.push_back(key);
                }
                if(surviving.empty()) continue;
                const auto preferAnchor=[&](const auto& a,const auto& b) {
                    const bool ka=frames.epochs_.at(StateId(a.frameId)).keyframe,kb=frames.epochs_.at(StateId(b.frameId)).keyframe;
                    if(ka!=kb) return ka>kb;
                    if(a.frameId!=b.frameId) return a.frameId>b.frameId;
                    return a<b;
                };
                std::sort(surviving.begin(),surviving.end(),preferAnchor);
                if(!landmark.initialized) {
                    // Absorbed views may retain an appearance identity, never a
                    // new depth constraint. A future fresh view replaces this
                    // dormant seed anchor before depth can be initialized.
                    landmark.anchor=surviving.front(); landmark.anchorBearing=store.find(landmark.anchor)->bearing;
                    ++out.movedSeeds; continue;
                }
                if(!redundant && options_.preserveOldestVisualInformation) continue;
                if(priorKeys.count(rho)) { reasons[item.first]="historical_prior_coupled"; continue; }
                if(!eligible) { reasons[item.first]="absorbed_or_time_varying_anchor"; continue; }
                if(!redundant && (!options_.resetOldestLandmarks || options_.autoReanchorLandmarks)) continue;
                if(graph.blocks.at(rho).constant) { reasons[item.first]="constant_landmark_chart"; continue; }
                // This is a measurement-chart RESET after discarding an outgoing
                // anchor observation, not an invertible scalar reparameterization.
                // Never apply it to prior-coupled, moving-bearing or external costs.
                if(externalLandmarks.count(item.first)) { reasons[item.first]="external_or_pending_landmark_factor"; continue; }
                if(surviving.size()==1) {
                    MarginalizationReport::TrackTransition transition;
                    transition.id=item.first; transition.oldAnchor=landmark.anchor; transition.newAnchor=surviving.front();
                    transition.action="demote"; transition.reason="one_surviving_ray"; transition.survivingViews=1;
                    transition.previousTransitionFrame=landmark.lastTransitionFrame;
                    // A single retained ray cannot constrain range after the old
                    // anchor image is discarded. Keep the track as a seed rather
                    // than publishing an unconstrained finite point or retiring its
                    // ID. Future real multi-view support can initialize it again.
                    const auto keys=store.ofLandmark(item.first);
                    for(const auto& key:keys) {
                        auto& observation=store.observations_.at(key);
                        if(observation.factorId) {
                            const auto id=observation.factorId; graph.factors.erase(id); store.byFactor_.erase(id);
                            discarded.push_back({id,FactorKind::Visual,FactorLifecycle::Rejected,"outgoing anchor lost multi-view range support"});
                            ++out.droppedVisualFactors;
                        }
                        observation.factorId=0; observation.use=ObservationUse::TrackingOnly;
                    }
                    graph.blocks.erase(rho); landmark.initialized=false;
                    landmark.anchor=surviving.front(); landmark.anchorBearing=store.find(landmark.anchor)->bearing;
                    ++landmark.demotionCount; landmark.lastTransitionFrame=order_.back().value();
                    transition.initializations=landmark.initializationCount; transition.resets=landmark.resetCount;
                    transition.demotions=landmark.demotionCount;
                    if(landmark.demotionCount>1) ++out.repeatedDemotions;
                    out.trackTransitions.push_back(std::move(transition));
                    out.demotedLandmarks.push_back(item.first); continue;
                }
                Eigen::Vector3d world;
                if(!pointWorld(graph,store,item.first,world,error)) return false;
                WindowLandmark replacement=landmark;
                std::optional<KeypointIdentifier> chosen; double newRho=0,maxRetainedNis=0;
                std::map<FactorId,FactorRecord> rebuilt;
                const double oldRho=graph.blocks.at(rho).value[0];
                bool hadGeometricCandidate=false;
                for(const auto& key:surviving) {
                    kinematics::Transformation camera;
                    if(!frames.cameraPose(graph,StateId(key.frameId),key.cameraIndex,camera,error)) return false;
                    const Eigen::Vector3d delta=camera.C().transpose()*(world-camera.r());
                    const auto& ray=store.find(key)->bearing;
                    const double range=delta.dot(ray);
                    if(range<=0 || !std::isfinite(range) || 1/range<options_.minInverseDepth
                        || std::atan2(delta.cross(ray).norm(),range)>options_.maxReanchorAngularError) continue;
                    const auto& coordinate=graph.blocks.at(rho);
                    if((coordinate.lower.size() && 1/range<coordinate.lower[0])
                        || (coordinate.upper.size() && 1/range>coordinate.upper[0])) continue;
                    hadGeometricCandidate=true;
                    replacement=landmark; replacement.anchor=key; replacement.anchorBearing=ray;
                    newRho=1/range; graph.blocks.at(rho).value[0]=newRho;
                    rebuilt.clear(); bool valid=true; maxRetainedNis=0;
                    for(const auto& retained:surviving) if(!(retained==key)) {
                        const auto& observation=*store.find(retained); FactorRecord f;
                        if(!observation.factorId || !makeWindowVisualFactor(replacement,*store.find(key),observation,frames,graph,options_,f,error)) {
                            valid=false; break;
                        }
                        const auto& previous=graph.factors.at(observation.factorId);
                        f.id=previous.id; f.lifecycle=previous.lifecycle; f.loss=previous.loss; f.nisThreshold=previous.nisThreshold;
                        std::vector<const double*> parameters;
                        for(auto k:f.keys) parameters.push_back(graph.blocks.at(k).value.data());
                        Eigen::Vector2d residual;
                        if(!f.cost->Evaluate(parameters.data(),residual.data(),nullptr) || !residual.allFinite()
                            || residual.squaredNorm()>f.nisThreshold) { valid=false; break; }
                        maxRetainedNis=std::max(maxRetainedNis,residual.squaredNorm());
                        rebuilt.emplace(f.id,std::move(f));
                    }
                    graph.blocks.at(rho).value[0]=oldRho;
                    // A newer ray can fail the existing NIS gate. Try another
                    // measured anchor before retiring an otherwise valid track.
                    if(valid) { chosen=key; break; }
                    error.clear();
                }
                graph.blocks.at(rho).value[0]=oldRho;
                if(!chosen) {
                    reasons[item.first]=hadGeometricCandidate?"reset_residual_gate_failed":"surviving_anchor_geometry_invalid";
                    error.clear(); continue;
                }
                MarginalizationReport::TrackTransition transition;
                transition.id=item.first; transition.oldAnchor=landmark.anchor; transition.newAnchor=*chosen;
                transition.action="reset"; transition.reason="anchor_evicted_prior_free";
                transition.previousTransitionFrame=landmark.lastTransitionFrame; transition.survivingViews=surviving.size();
                kinematics::Transformation newCamera;
                if(!frames.cameraPose(graph,StateId(chosen->frameId),chosen->cameraIndex,newCamera,error)) return false;
                transition.pointShift=(newCamera.C()*(replacement.anchorBearing/newRho)+newCamera.r()-world).norm();
                transition.maxRetainedNis=maxRetainedNis;
                ++replacement.resetCount; replacement.lastResetFrame=order_.back().value();
                replacement.lastTransitionFrame=order_.back().value();
                transition.initializations=replacement.initializationCount; transition.resets=replacement.resetCount;
                transition.demotions=replacement.demotionCount;
                if(replacement.resetCount>1) ++out.repeatedResets;
                out.trackTransitions.push_back(std::move(transition));
                const auto keys=store.ofLandmark(item.first);
                for(const auto& key:keys) {
                    auto& observation=store.observations_.at(key);
                    if(!observation.factorId) continue;
                    if(rebuilt.count(observation.factorId)) graph.factors.at(observation.factorId)=std::move(rebuilt.at(observation.factorId));
                    else {
                        const auto id=observation.factorId; graph.factors.erase(id); store.byFactor_.erase(id);
                        observation.factorId=0; observation.use=ObservationUse::TrackingOnly;
                        discarded.push_back({id,FactorKind::Visual,FactorLifecycle::Rejected,"outgoing anchor/chart observation removed"});
                        ++out.droppedVisualFactors;
                    }
                }
                landmark=replacement; graph.blocks.at(rho).owner=StateId(chosen->frameId); graph.blocks.at(rho).value[0]=newRho;
                out.resetLandmarks.push_back(item.first);
            }
            // Redundant frames (and the optional legacy OLD reset) discard their
            // image measurements. Information-preserving OLD leaves them intact
            // for joint elimination of the outgoing state and incident ranges.
            const auto outgoing=store.inFrame(out.removed);
            for(const auto& key:outgoing) {
                if(!redundant && options_.preserveOldestVisualInformation) continue;
                const auto& observation=*store.find(key);
                if(store.landmarks_.at(observation.landmark).anchor.frameId==out.removed.value()) continue;
                if(observation.use==ObservationUse::LiveFactor) {
                    discarded.push_back({observation.factorId,FactorKind::Visual,FactorLifecycle::Rejected,"outgoing image observation removed"});
                    graph.factors.erase(observation.factorId); ++out.droppedVisualFactors;
                }
                store.eraseObservation(key);
            }
            if(graph.prior!=incomingPrior) { error="chart preparation replaced the frozen prior"; return false; }
            if(incomingPrior) for(size_t i=0;i<incomingPrior->keys().size();++i) {
                const auto block=graph.blocks.find(incomingPrior->keys()[i]);
                if(block==graph.blocks.end() || block->second.value.size()!=protectedValues[i].size()
                    || !(block->second.value.array()==protectedValues[i].array()).all()) {
                    error="chart preparation changed a prior-coupled coordinate"; return false;
                }
            }
            out.resetPriorUnchanged=true;
            return true;
        }
        bool SlidingWindowManager::promotionWouldBeMarginalized(const KeypointIdentifier& anchor,
            const std::vector<KeypointIdentifier>& support) const {
            if(!needsMarginalization()) return false;
            const auto policy=choosePolicy();
            const auto outgoing=policy==MarginalizationPolicy::MARGIN_OLD?order_.front():order_[size()-2];
            if(anchor.frameId==outgoing.value()) return true;
            if(policy!=MarginalizationPolicy::MARGIN_OLD || !options_.preserveOldestVisualInformation) return false;
            for(const auto& key:support) if(key.frameId==outgoing.value() && !landmarks_.consumed(key)) return true;
            return false;
        }
        MarginalizationReport SlidingWindowManager::marginalize(MarginalizationPolicy policy, const LandmarkReanchorHook& reanchor) {
            // Phase 1: select the victim without editing the live graph. OLD is the
            // chronological front; SECOND_NEW is index size()-2, never the latest state.
            MarginalizationReport out; out.policy = policy;
            if (!needsMarginalization()) { out.message = "window has no overflow"; return out; }
            struct FactorIdGuard {
                FactorId& value; FactorId saved; bool committed=false;
                explicit FactorIdGuard(FactorId& id):value(id),saved(id) {}
                ~FactorIdGuard() { if(!committed) value=saved; }
            } idGuard(nextFactorId_);
            out.removed = policy == MarginalizationPolicy::MARGIN_OLD ? order_.front() : order_[size() - 2];
            out.landmarksBefore=landmarks_.size(); out.removedKeyframe=frames_.epochs_.at(out.removed).keyframe;
            const auto stageMs=[](auto start) { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); };
            const auto copyStarted=std::chrono::steady_clock::now();
            BackendGraph candidate = graph_;
            LandmarkStore candidateLandmarks=landmarks_;
            FrameStore candidateFrames=frames_;
            out.topologyCopyMs=stageMs(copyStarted);
            // Parameter vectors are copied, while immutable cost/manifold/prior objects
            // are shared. Hooks must replace costs rather than mutate shared objects.
            if (reanchor) {
                try {
                    if (!reanchor(out.removed, candidate, out.message)) return out;
                }
                catch (const std::exception& e) { out.message = e.what(); return out; }
                if (candidate.blocks.size() != graph_.blocks.size() || candidate.factors.size() != graph_.factors.size()
                    || candidate.prior != graph_.prior) {
                    out.message = "reanchor may not add/drop blocks/factors or replace FEJ prior"; return out;
                }
                std::set<BlockKey> priorKeys;
                // Changing a prior-coupled rho would change the meaning of a frozen
                // Jacobian column. Until explicit coordinate transport exists, reject it.
                if (graph_.prior) priorKeys.insert(graph_.prior->keys().begin(), graph_.prior->keys().end());
                for (const auto& item : graph_.blocks) 
                {
                    const auto found = candidate.blocks.find(item.first);
                    if (found == candidate.blocks.end()) { out.message = "reanchor changed parameter identities"; return out; }
                    const auto& a = item.second; const auto& b = found->second;
                    if (a.key != b.key || a.value.size() != b.value.size() || a.manifold != b.manifold || a.constant != b.constant
                        || a.lower.size() != b.lower.size() || a.upper.size() != b.upper.size()
                        || (a.lower.size() && !(a.lower.array() == b.lower.array()).all())
                        || (a.upper.size() && !(a.upper.array() == b.upper.array()).all())) {
                        out.message = "reanchor changed parameter layout/configuration"; return out;
                    }
                    const bool changed = a.owner != b.owner || !(a.value.array() == b.value.array()).all();
                    if(changed && landmarks_.find(LandmarkId(a.key.id)) && a.key.kind==BlockKind::Landmark) {
                        out.message="legacy graph-only hook cannot change managed landmark geometry"; return out;
                    }
                    if (changed && (a.key.kind != BlockKind::Landmark || priorKeys.count(a.key)
                        || !a.owner || *a.owner != out.removed || !b.owner || *b.owner == out.removed || !frames_.epochs_.count(*b.owner))) {
                        out.message = "unsafe reanchor: only outgoing, non-prior landmarks may change anchor"; return out;
                    }
                }
                for (const auto& item : graph_.factors) {
                    auto f = candidate.factors.find(item.first);
                    // Reanchoring changes representation, not the underlying evidence.
                    // It cannot silently drop a measurement, change its loss, or reset NIS.
                    if (f == candidate.factors.end() || f->second.id != item.second.id || f->second.kind != item.second.kind
                        || f->second.loss != item.second.loss || f->second.nisThreshold != item.second.nisThreshold
                        || f->second.lifecycle != item.second.lifecycle || f->second.solveEnabled != item.second.solveEnabled
                        || f->second.sensorEpoch != item.second.sensorEpoch) {
                        out.message = "reanchor changed measurement identity/lifecycle/weight"; return out;
                    }
                    // Recompute owner dependencies after geometry layer replaces the visual residual's anchor.
                    std::set<StateId> deps;
                    if (f->second.sensorEpoch.isInitialised()) deps.insert(f->second.sensorEpoch);
                    for (auto k : f->second.keys) {
                        if (k.isState()) deps.insert(StateId(k.id));
                        auto p = candidate.blocks.find(k);
                        if (p != candidate.blocks.end() && p->second.owner) deps.insert(*p->second.owner);
                    }
                    f->second.epochs.assign(deps.begin(), deps.end());
                }
                if (!candidate.valid(&out.message)) return out;
                // Verify nominal residual equivalence at the accepted solution. This is
                // a safety check, not a proof of the geometry layer's reanchor Jacobians.
                for (const auto& item : graph_.factors) {
                    const auto& a = item.second; const auto& b = candidate.factors.at(item.first);
                    if (a.cost->num_residuals() != b.cost->num_residuals()) { out.message = "reanchor changed residual dimension"; return out; }
                    if (!item.second.participates()) continue;
                    std::vector<const double*> xa, xb;
                    for (auto k : a.keys) xa.push_back(graph_.blocks.at(k).value.data());
                    for (auto k : b.keys) xb.push_back(candidate.blocks.at(k).value.data());
                    Eigen::VectorXd ra(a.cost->num_residuals()), rb(b.cost->num_residuals());
                    if (!a.cost->Evaluate(xa.data(), ra.data(), nullptr) || !b.cost->Evaluate(xb.data(), rb.data(), nullptr)
                        || !ra.allFinite() || !rb.allFinite() || (ra - rb).norm() > 1e-7 * (1.0 + ra.norm())) {
                        out.message = "reanchor did not preserve the measurement residual at the accepted state"; return out;
                    }
                }
            }
            std::map<LandmarkId,std::string> reanchorReasons;
            if(policy==MarginalizationPolicy::MARGIN_OLD
                && !reanchorOutgoing(out.removed,candidate,candidateLandmarks,out.reanchoredLandmarks,out.message,&reanchorReasons)) return out;
            std::vector<FactorEvent> discarded;
            if(!prepareEviction(out,candidate,candidateLandmarks,candidateFrames,discarded,reanchorReasons,out.message)) return out;
            std::set<LandmarkId> outgoingTracks;
            for(const auto& key:candidateLandmarks.inFrame(out.removed)) {
                const auto* view=candidateLandmarks.find(key);
                if(view) outgoingTracks.insert(view->landmark);
            }
            // Capture live support BEFORE Schur marks it absorbed. Otherwise the
            // retirement log would falsely report no surviving measurements.
            for(auto track:outgoingTracks) {
                const auto found=candidateLandmarks.landmarks_.find(track);
                if(found==candidateLandmarks.landmarks_.end() || found->second.anchor.frameId!=out.removed.value()) continue;
                const auto& item=*found;
                MarginalizationReport::Retirement detail; detail.id=item.first; detail.anchor=item.second.anchor;
                detail.initialized=item.second.initialized;
                uint64_t lastObservedEpoch=0;
                for(const auto& key:candidateLandmarks.ofLandmark(item.first)) {
                    if(key.frameId!=lastObservedEpoch) { ++detail.observedEpochs; lastObservedEpoch=key.frameId; }
                    if(key.frameId==out.removed.value()) continue;
                    const auto observation=candidateLandmarks.find(key);
                    if(observation->use==ObservationUse::Absorbed) ++detail.survivingAbsorbed;
                    else if(observation->use==ObservationUse::TrackingOnly) ++detail.survivingTracking;
                    else {
                        const auto factor=candidate.factors.find(observation->factorId);
                        if(factor!=candidate.factors.end() && factor->second.lifecycle==FactorLifecycle::Active) ++detail.survivingLive;
                    }
                }
                if(!detail.initialized) { ++out.retiredSeeds; detail.reason="uninitialized_anchor_evicted"; }
                else {
                    ++out.retiredMature; if(detail.survivingLive) ++out.retiredMatureWithLiveViews;
                    const auto why=reanchorReasons.find(item.first);
                    detail.reason=why!=reanchorReasons.end()?why->second:!options_.autoReanchorLandmarks
                        ?"anchor_evicted_reanchor_disabled":"anchor_evicted_sensor_invalid";
                    if(!detail.survivingLive && !detail.survivingAbsorbed && !detail.survivingTracking) detail.reason="no_surviving_observation";
                }
                out.retirements.push_back(std::move(detail));
            }
            std::vector<FactorEvent> consumedEvents,pendingEvents;
            std::set<BlockKey> eliminate;
            // Phase 2: ownership selects Pose, SpeedBias, and any tracks still anchored
            // to the victim. Global T_BR/T_BL/td blocks have no owner and survive.
            for (const auto& p : candidate.blocks) if (p.second.owner && *p.second.owner == out.removed) eliminate.insert(p.first);
            if(policy==MarginalizationPolicy::MARGIN_OLD && options_.preserveOldestVisualInformation) {
                // Keep ALL outgoing visual evidence. Eliminate incident ranges
                // too, so retained ranges do not become mutually coupled through
                // the separator prior. SECOND_NEW still discards redundant views.
                for(const auto& key:candidateLandmarks.inFrame(out.removed)) {
                    const auto observation=candidateLandmarks.find(key);
                    if(observation->use==ObservationUse::Absorbed || candidateLandmarks.consumed(key)) {
                        ++out.ignoredAbsorbedIncidentObservations; continue;
                    }
                    // Metadata incidence is not measurement incidence. An old,
                    // consumed ray belongs to the prior, not a freshly created rho.
                    // Only an actually participating bound visual cost may cause
                    // this additional range elimination; owners were handled above.
                    if(observation->use!=ObservationUse::LiveFactor) continue;
                    const auto factor=candidate.factors.find(observation->factorId);
                    if(factor==candidate.factors.end() || !factor->second.participates()) continue;
                    const auto point=candidateLandmarks.find(observation->landmark);
                    const BlockKey rho{BlockKind::Landmark,point->id.value()};
                    if(point->initialized && factor->second.kind==FactorKind::Visual
                        && std::find(factor->second.keys.begin(),factor->second.keys.end(),rho)!=factor->second.keys.end()
                        && std::find(factor->second.keys.begin(),factor->second.keys.end(),BlockKey::pose(out.removed))!=factor->second.keys.end()) {
                        if(eliminate.insert(rho).second) ++out.liveIncidentRanges;
                    }
                }
            }
            for(const auto& key:eliminate) if(key.kind==BlockKind::Landmark) {
                const auto point=candidateLandmarks.find(LandmarkId(key.id));
                if(point && point->lastInitializedFrame==order_.back().value()) ++out.sameFrameRangeEliminations;
            }
            // Unreanchored tracks are retired together with ALL live measurements, preserving pose/calibration
            // information through Schur, without reusing the same observations in a fresh track.
            auto result = marginalizer_.compute(candidate, eliminate);
            if (!result.success) { out.message = result.message; return out; }
            consumedEvents.reserve(result.consumed.size());
            for(auto id:result.consumed) {
                const auto& factor=candidate.factors.at(id);
                consumedEvents.push_back({id,factor.kind,FactorLifecycle::Marginalized,"absorbed into fixed-linearization prior"});
            }
            candidate.prior = result.prior;
            out.priorReused=result.priorReused;
            if(candidate.prior) for(auto key:candidate.prior->keys()) if(key.kind==BlockKind::Landmark) ++out.priorLandmarks;
            // Phase 3: remove exactly the costs absorbed into the new prior. Leaving them
            // active would count the same measurements twice on the next solve.
            for (auto id : result.consumed) {
                candidateLandmarks.factorRemoved(id,true); candidate.factors.erase(id);
            }
            for (auto it = candidate.factors.begin(); it != candidate.factors.end();) {
                bool touches = std::find(it->second.epochs.begin(), it->second.epochs.end(), out.removed) != it->second.epochs.end();
                for (auto k : it->second.keys) touches = touches || eliminate.count(k) != 0;
                if (touches && it->second.participates()) {
                    // An active metadata dependency without an eliminated graph connection
                    // indicates an inconsistent factor contract; fail rather than lose it.
                    out.message = "active dependency was not consumed by marginalizer"; return out;
                }
                if (touches) {
                    if(it->second.lifecycle==FactorLifecycle::Pending)
                        pendingEvents.push_back({it->first,it->second.kind,FactorLifecycle::Rejected,"dependency evicted before activation"});
                    it = candidate.factors.erase(it);
                } else ++it;
            }
            for (auto k : eliminate) candidate.blocks.erase(k);
            if(policy==MarginalizationPolicy::MARGIN_OLD && options_.preserveOldestVisualInformation && options_.preserveSchurTrackIds) {
                for(auto key:eliminate) if(key.kind==BlockKind::Landmark) {
                    auto point=candidateLandmarks.landmarks_.find(LandmarkId(key.id));
                    if(point==candidateLandmarks.landmarks_.end()) continue;
                    std::vector<KeypointIdentifier> surviving;
                    for(const auto& view:candidateLandmarks.ofLandmark(point->first))
                        if(view.frameId!=out.removed.value()) surviving.push_back(view);
                    if(surviving.empty()) continue;
                    const auto previous=point->second.anchor;
                    const auto newest=*std::max_element(surviving.begin(),surviving.end());
                    auto& p=point->second;
                    MarginalizationReport::TrackTransition event;
                    event.id=p.id; event.oldAnchor=previous; event.newAnchor=newest;
                    event.action="schur_to_seed"; event.reason="historical_visual_information_absorbed";
                    event.previousTransitionFrame=p.lastTransitionFrame; event.survivingViews=surviving.size();
                    p.initialized=false; p.anchor=newest; p.anchorBearing=candidateLandmarks.find(newest)->bearing;
                    for(const auto& view:candidateLandmarks.ofLandmark(p.id)) {
                        // The old anchor ray was implicit in every consumed
                        // reprojection factor, even though it had no own residual.
                        candidateLandmarks.absorbedKeys_.insert(view);
                        candidateLandmarks.observations_.at(view).use=ObservationUse::Absorbed;
                    }
                    p.lastTransitionFrame=order_.back().value(); ++p.demotionCount;
                    event.initializations=p.initializationCount; event.resets=p.resetCount; event.demotions=p.demotionCount;
                    out.trackTransitions.push_back(std::move(event)); out.schurPreservedTracks.push_back(p.id);
                    // This identity survives; its OLD optimization coordinate did
                    // not. Exclude it from true retirement statistics/receipts.
                    out.retirements.erase(std::remove_if(out.retirements.begin(),out.retirements.end(),[&](const auto& detail) {
                        if(detail.id!=p.id) return false;
                        if(detail.initialized) { --out.retiredMature; if(detail.survivingLive) --out.retiredMatureWithLiveViews; }
                        else --out.retiredSeeds;
                        return true;
                    }),out.retirements.end());
                }
            }
            for(const auto& item:candidate.factors) if(item.second.kind==FactorKind::Imu
                && item.second.lifecycle==FactorLifecycle::Active) {
                if(item.second.solveEnabled) ++out.activeImuFactors; else ++out.suppressedImuFactors;
            }
            // Geometry, observations, frame payloads, and graph commit as one value.
            // This includes uninitialized tracks, which have no optimization block.
            std::vector<LandmarkId> retired;
            std::set<LandmarkId> retirementCandidates=outgoingTracks;
            for(auto key:eliminate) if(key.kind==BlockKind::Landmark) retirementCandidates.insert(LandmarkId(key.id));
            for(auto track:retirementCandidates) {
                const auto found=candidateLandmarks.landmarks_.find(track);
                if(found==candidateLandmarks.landmarks_.end()) continue;
                const auto& item=*found;
                const bool removedRange=item.second.initialized && eliminate.count({BlockKind::Landmark,item.first.value()});
                if(item.second.anchor.frameId!=out.removed.value() && !removedRange) continue;
                retired.push_back(item.first);
                if(removedRange && item.second.anchor.frameId!=out.removed.value()) {
                    // Strict retirement mode also removes incident ranges whose
                    // implicit anchor survived. They must not retain a missing rho.
                    MarginalizationReport::Retirement detail; detail.id=item.first;
                    detail.anchor=item.second.anchor; detail.initialized=true;
                    detail.reason="incident_range_schur_retired"; ++out.retiredMature;
                    out.retirements.push_back(std::move(detail));
                }
            }
            for(auto id:retired) {
                candidateLandmarks.eraseLandmark(id,candidateLandmarks.landmarks_.at(id).initialized);
                out.retiredLandmarks.push_back(id);
            }
            candidateLandmarks.eraseFrame(out.removed);
            out.landmarksAfter=candidateLandmarks.size();
            for(const auto& item:landmarks_.observations_) if(!candidateLandmarks.find(item.first)) out.removedObservations.push_back(item.first);
            SlidingWindowManager staged(*this,std::move(candidate),std::move(candidateLandmarks),std::move(candidateFrames));
            staged.frames_.erase(out.removed);
            staged.order_.erase(std::find(staged.order_.begin(),staged.order_.end(),out.removed));
            const auto validationStarted=std::chrono::steady_clock::now();
            if(!staged.valid(out.message)) return out;
            out.topologyValidationMs=stageMs(validationStarted);
            // Phase 4: publish terminal events, then replace topology and chronological
            // membership. No Ceres Problem may still reference the replaced value buffers.
            // Replayed native edges were removed BEFORE Schur selection. Opaque
            // external edges use the prior bridge instead; never do both.
            for(auto& event:discarded) staged.recordEvent(std::move(event));
            for(auto& event:consumedEvents) staged.recordEvent(std::move(event));
            for(auto& event:pendingEvents) staged.recordEvent(std::move(event));
            *this=std::move(staged);
            idGuard.committed=true;
            out.consumedFactors = result.consumed.size(); out.eliminatedDimension = result.eliminatedDimension;
            out.retainedDimension = result.retainedDimension; out.priorRank = result.priorRank;
            out.linearizeMs=result.linearizeMs; out.schurMs=result.schurMs; out.priorMs=result.priorMs;
            out.linearization=result.linearization;
            out.scalarSchurLandmarks=result.scalarSchurLandmarks;
            out.formation=result.formation;
            out.retiredBlocks.assign(eliminate.begin(), eliminate.end()); out.success = true; 
            return out;
        }
    }
}
