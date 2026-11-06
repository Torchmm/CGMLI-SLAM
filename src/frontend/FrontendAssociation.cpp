#include "FrontendAssociation.hpp"
#include <backend/FrameStore.hpp>
#include <backend/WindowVisualFactor.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <atomic>
#include <chrono>
#include <thread>
#include <exception>
#include <cstring>
#include <sstream>
#include <queue>
#include <functional>

namespace cgmli {
    namespace frontend_association {
        namespace {
            using Transform = kinematics::Transformation;
            using Slot = std::tuple<LandmarkId, uint64_t, size_t>;
            using Candidate = FeatureMatchDebug;

            bool usablePixel(const cameras::CameraBase& camera, const Eigen::Vector2d& pixel) {
                // CameraBase::isMasked is protected for camera implementations. Frontend
                // clients validate through the public dimensions/mask API, including finite
                // coordinates before any float-to-integer conversion or mask indexing.
                if (!pixel.allFinite() || pixel.x() < 0 || pixel.y() < 0
                    || pixel.x() >= camera.imageWidth() || pixel.y() >= camera.imageHeight()) return false;
                return !camera.hasMask() || camera.mask().at<uchar>(int(pixel.y()), int(pixel.x())) != 0;
            }

            bool bearing(const MultiFrame& frame, const KeypointIdentifier& key, Eigen::Vector3d& out) {
                if (key.frameId != frame.id() || key.cameraIndex >= frame.numFrames()
                    || key.keypointIndex >= frame.numKeypoints(key.cameraIndex)) return false;
                const auto& point = frame.keypoints(key.cameraIndex)[key.keypointIndex].pt;
                const Eigen::Vector2d pixel(point.x, point.y); const auto camera = frame.geometry(key.cameraIndex);
                if (!camera || !usablePixel(*camera, pixel)
                    || (!frame.getBackProjection(key.cameraIndex, key.keypointIndex, out) && !camera->backProject(pixel, &out))
                    || !out.allFinite() || out.norm() < 1e-12) return false;
                out.normalize(); return true;
            }
            bool fits(const MultiFrame& frame, const KeypointIdentifier& key, const Transform& T_WC,
                const Eigen::Vector3d& point, double limit) {
                Eigen::Vector3d ray;
                if (!bearing(frame, key, ray)) return false;
                const Eigen::Vector3d delta = T_WC.C().transpose() * (point - T_WC.r());
                // Positive distance along a half-ray, NOT positive optical z. EUCM/fisheye
                // observations in the valid rear hemisphere must remain usable.
                if (!delta.allFinite() || delta.norm() < 1e-10 || delta.dot(ray) <= 0.0) return false;
                return std::atan2(delta.cross(ray).norm(), delta.dot(ray)) <= limit;
            }
            Transform cameraPose(const Estimator& estimator, const KeypointIdentifier& key,
                StateId current, const Transform& T_WB) {
                return (StateId(key.frameId) == current ? T_WB : estimator.pose(StateId(key.frameId)))
                    * estimator.extrinsics(StateId(key.frameId), key.cameraIndex);
            }
        }

        void Options::validate() const {
            if (!maxSourceFrames || !maxNewLandmarks || !mapDescriptorsPerLandmark || mapDescriptorsPerLandmark > 8
                || !minimumStageMatchesPerCamera || !matchingThreads || matchingThreads > 64 || !std::isfinite(searchRadiusPixels) || searchRadiusPixels <= 0
                || !std::isfinite(pixelSigma) || pixelSigma <= 0 || !std::isfinite(pyramidScale) || pyramidScale < 1
                || !std::isfinite(minimumParallax)
                || minimumParallax <= 0 || minimumParallax >= 1.0 || !std::isfinite(maximumAngularError)
                || maximumAngularError <= 0 || maximumAngularError >= 1.0 || !std::isfinite(minimumDepth)
                || minimumDepth <= 0 || !std::isfinite(maximumDepth) || maximumDepth <= minimumDepth
                || !std::isfinite(seedMaxUntrackedSeconds) || seedMaxUntrackedSeconds<0)
                throw std::invalid_argument("Invalid frontend association options");
            features::OrbMatcher validateMatcher(matcher);
        }
        bool sameMeasurements(const MultiFrame& input, const MultiFrame& frozen) {
            if (input.id() != frozen.id() || input.timestamp().toNSec() != frozen.timestamp().toNSec()
                || input.numFrames() != frozen.numFrames()) return false;
            for (size_t c = 0; c < input.numFrames(); ++c) {
                const auto& inputTiming = input.imageTiming(c); const auto& frozenTiming = frozen.imageTiming(c);
                if (inputTiming.has_value() != frozenTiming.has_value() || (inputTiming
                    && (inputTiming->originalTimestamp.toNSec() != frozenTiming->originalTimestamp.toNSec()
                        || inputTiming->offsetNs != frozenTiming->offsetNs))) return false;
                if (input.geometry(c).get() != frozen.geometry(c).get()
                    || input.descriptorProfile(c) != frozen.descriptorProfile(c)
                    || input.numKeypoints(c) != frozen.numKeypoints(c)) return false;
                const auto& a = input.descriptors(c); const auto& b = frozen.descriptors(c);
                // OpenCV clone() may collapse an empty 0x32 Mat to a default empty Mat.
                // Empty cameras are legal in a rig; the zero keypoint count already agrees.
                if (a.empty() != b.empty() || (!a.empty()
                    && (a.size() != b.size() || a.type() != b.type() || cv::norm(a, b, cv::NORM_INF) != 0))) return false;
                for (size_t i = 0; i < input.numKeypoints(c); ++i) {
                    const auto& x = input.keypoints(c)[i]; const auto& y = frozen.keypoints(c)[i];
                    if (x.pt != y.pt || x.size != y.size || x.angle != y.angle || x.octave != y.octave
                        || x.response != y.response || x.class_id != y.class_id) return false;
                }
            }
            return true;
        }
        void synchronize(const MultiFrame& authoritative, MultiFrame& output) {
            if (authoritative.id() != output.id() || authoritative.numFrames() != output.numFrames())
                throw std::invalid_argument("Association output frame identity mismatch");
            for (size_t c = 0; c < output.numFrames(); ++c) {
                if (authoritative.numKeypoints(c) != output.numKeypoints(c)) throw std::invalid_argument("Association output index mismatch");
                for (size_t i = 0; i < output.numKeypoints(c); ++i) {
                    Eigen::Vector4d point = Eigen::Vector4d::Zero(); bool initialized = false;
                    authoritative.getLandmark(c, i, point, initialized);
                    output.setLandmarkId(c, i, authoritative.landmarkId(c, i));
                    output.setLandmark(c, i, point, initialized);
                }
            }
        }
        Plan propose(const Estimator& estimator, const Options& options) {
            const auto started = std::chrono::steady_clock::now();
            const auto milliseconds = [](const auto& beginning) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - beginning).count(); };
            options.validate(); Plan plan; const auto currentId = estimator.currentStateId();
            const auto current = estimator.associationFrame(currentId);
            if (!current) throw std::invalid_argument("Append extracted features with addStates before association");
            if (current->numFrames() > 1 && !current->cameraSystem().overlapComputationValid())
                throw std::logic_error("Frontend cross-camera matching requires a valid rig FOV overlap graph");
            plan.current = current;
            for (auto it = estimator.window().order().rbegin(); it != estimator.window().order().rend(); ++it) {
                if (*it == currentId) continue;
                if (estimator.window().epochs().at(*it).visual == backend::SensorHealth::Invalid) continue;
                const auto frame = estimator.associationFrame(*it); if (!frame) continue; // IMU/LiDAR-only epochs are legal.
                if (frame->numFrames() != current->numFrames()) throw std::invalid_argument("Association rig size changed");
                for (size_t c = 0; c < frame->numFrames(); ++c) {
                    features::OrbFrontendFeatures::validateDescriptors(*frame, c);
                    if (frame->descriptorProfile(c) != current->descriptorProfile(c)
                        || frame->geometry(c).get() != current->geometry(c).get())
                        throw std::invalid_argument("Historical descriptor profile/camera differs from current rig");
                }
                plan.sources.emplace(*it, frame);
                if (plan.sources.size() >= options.maxSourceFrames) break;
            }
            plan.framePreparationMs = milliseconds(started); const auto mapStarted = std::chrono::steady_clock::now();
            features::OrbMatcher matcher(options.matcher);
            for (const auto& item : estimator.window().landmarks().landmarks()) if (item.second.initialized) {
                Eigen::Vector3d world;
                if (!estimator.window().landmarkPoint(item.first, world)) throw std::runtime_error("Cannot prepare initialized landmark coordinates");
                plan.finitePoints.emplace(item.first, world);
            }
            const auto& points = plan.finitePoints;
            // Seeds absent from every selected source are no longer useful for this
            // association horizon. Reclaim only tracking metadata, never optimized evidence.
            size_t observationSlots = estimator.window().remainingObservationCapacity();
            size_t landmarkSlots = estimator.window().remainingLandmarkCapacity();
            std::set<LandmarkId> retiredSeedIds;
            for (const auto& item : estimator.window().landmarks().landmarks()) if (!item.second.initialized) {
                ++plan.trackingSeeds;
                bool recent = false, trackingOnly = true;
                uint64_t lastObservationNs=0;
                const auto& observations = estimator.window().landmarks().ofLandmark(item.first);
                for (const auto& key : observations) {
                    recent = recent || key.frameId == currentId.value() || plan.sources.count(StateId(key.frameId));
                    lastObservationNs=std::max(lastObservationNs,estimator.window().epochs().at(StateId(key.frameId)).timestamp.toNSec());
                    const auto observation = estimator.window().landmarks().find(key);
                    trackingOnly = trackingOnly && observation && observation->use == backend::ObservationUse::TrackingOnly
                        && !observation->factorId && !estimator.isObservationConsumed(key);
                }
                const double untrackedSeconds=double(current->timestamp().toNSec()-lastObservationNs)*1e-9;
                const bool expired=options.seedMaxUntrackedSeconds>0 && untrackedSeconds>options.seedMaxUntrackedSeconds;
                if ((!recent || expired) && trackingOnly) {
                    plan.batch.retiredLandmarks.push_back(item.first); ++landmarkSlots;
                    retiredSeedIds.insert(item.first); --plan.trackingSeeds;
                    if(expired) ++plan.expiredSeeds;
                    observationSlots += observations.size();
                    // A timed-out seed can still occur in an old source image.
                    // Clear only this private proposal, never the authoritative store.
                    for(const auto& key:observations) {
                        auto source=plan.sources.find(StateId(key.frameId));
                        if(source!=plan.sources.end()) {
                            source->second->setLandmarkId(key.cameraIndex,key.keypointIndex,0);
                            source->second->setLandmark(key.cameraIndex,key.keypointIndex,Eigen::Vector4d::Zero(),false);
                        }
                    }
                }
            }
            plan.retiredSeeds = plan.batch.retiredLandmarks.size();
            plan.admittedTrackLimit = std::min(options.maxNewLandmarks, landmarkSlots);
            if(options.maxTrackingSeeds) plan.admittedTrackLimit=std::min(plan.admittedTrackLimit,
                plan.trackingSeeds>=options.maxTrackingSeeds?size_t(0):options.maxTrackingSeeds-plan.trackingSeeds);
            plan.mapPreparationMs = milliseconds(mapStarted); const auto raysStarted = std::chrono::steady_clock::now();
            struct PreparedCamera {
                Transform T_WC;
                features::OrbBearings bearings;
                features::OrbWordRows words;
                features::OrbTrainIndex trainIndex;
                std::vector<uint8_t> queryEnabled, trainEnabled,queryPriorities;
            };
            std::map<uint64_t, std::vector<PreparedCamera>> prepared;
            const auto prepare = [&](const MultiFrame& frame) {
                auto& cameras = prepared[frame.id()]; cameras.resize(frame.numFrames());
                const auto pose = estimator.pose(StateId(frame.id()));
                for (size_t c = 0; c < frame.numFrames(); ++c) {
                    auto& value = cameras[c]; value.T_WC = pose * estimator.extrinsics(StateId(frame.id()), c);
                    value.bearings.resize(frame.numKeypoints(c)); value.queryEnabled.resize(frame.numKeypoints(c));
                    value.queryPriorities.resize(frame.numKeypoints(c));
                    value.words.resize(frame.numKeypoints(c));
                    if (frame.id() == current->id()) value.trainIndex = features::OrbMatcher::prepareTrainIndex(frame.keypoints(c), frame.geometry(c).get(), options.searchRadiusPixels);
                    value.trainEnabled.resize(frame.numKeypoints(c));
                    for (size_t k = 0; k < frame.numKeypoints(c); ++k) {
                        const KeypointIdentifier key(frame.id(), c, k);
                        std::memcpy(value.words[k].data(), frame.descriptors(c).ptr(int(k)), 32);
                        const bool observed = estimator.isObserved(key), consumed = estimator.isObservationConsumed(key);
                        value.queryEnabled[k] = !consumed || observed;
                        const LandmarkId identity(frame.landmarkId(c,k));
                        value.queryPriorities[k]=identity.isInitialised()?1:0;
                        const auto metadata=estimator.window().landmarks().find(identity);
                        if(frame.id()!=current->id() && metadata && !metadata->initialized && !retiredSeedIds.count(identity))
                            plan.sourceSeedIds.insert(identity);
                        value.trainEnabled[k] = !consumed && !observed;
                        if (!bearing(frame, key, value.bearings[k])) value.bearings[k].setConstant(std::numeric_limits<double>::quiet_NaN());
                    }
                }
                };
            prepare(*current); for (const auto& source : plan.sources) prepare(*source.second);
            plan.rayPreparationMs = milliseconds(raysStarted);
            const auto& immutable = prepared;
            struct Job { const MultiFrame* source; size_t q, t; bool stereo; };
            struct Result { std::vector<Candidate> matches; features::OrbMatchStatistics statistics; std::exception_ptr error; double guidanceMs = 0; };
            std::vector<Job> jobs;
            for (const auto& source : plan.sources) for (size_t q = 0; q < current->numFrames(); ++q)
                for (size_t t = 0; t < current->numFrames(); ++t) jobs.push_back({ source.second.get(),q,t,false });
            for (size_t q = 0; q < current->numFrames(); ++q) for (size_t t = q + 1; t < current->numFrames(); ++t)
                if (current->hasOverlap(t, q) && current->hasOverlap(q, t)) jobs.push_back({ current.get(),q,t,true });
            std::vector<Result> results(jobs.size());
            plan.preparationMs = milliseconds(started); const auto matchingStarted = std::chrono::steady_clock::now();
            std::vector<Candidate> candidates;
            plan.stagedUsed = options.stagedMatching && !plan.sources.empty() && !points.empty();
            std::vector<std::vector<uint8_t>> mapClaimed(current->numFrames());
            for (size_t c = 0; c < current->numFrames(); ++c) mapClaimed[c].resize(current->numKeypoints(c), 0);
            const auto accumulate = [&](const Result& result) {
                if (result.error) std::rethrow_exception(result.error);
                plan.candidatePairs += result.statistics.candidatePairs; plan.descriptorComparisons += result.statistics.descriptorComparisons;
                plan.matchingSetupMs += result.statistics.setupMs; plan.matchingTraversalMs += result.statistics.traversalMs;
                plan.matchingSelectionMs += result.statistics.selectionMs; plan.matchingGuidanceMs += result.guidanceMs;
                plan.predictedQueries += result.statistics.predictedQueries; plan.unpredictedQueries += result.statistics.unpredictedQueries;
                plan.ownedQueries+=result.statistics.ownedQueries; plan.ownedMatches+=result.statistics.ownedMatches;
                plan.prioritySuppressed+=result.statistics.prioritySuppressed;
                for(const auto& match:result.matches) if(match.first.frameId!=current->id()) {
                    const LandmarkId id(plan.sources.at(StateId(match.first.frameId))->landmarkId(match.first.cameraIndex,match.first.keypointIndex));
                    if(plan.sourceSeedIds.count(id)) plan.matchedSeedIds.insert(id);
                }
                candidates.insert(candidates.end(), result.matches.begin(), result.matches.end());
                };
            const auto parallelFor = [&](size_t count, const auto& operation) {
                std::atomic<size_t> next{ 0 }; std::vector<std::thread> workers;
                const auto work = [&]() { for (;;) { const size_t i = next.fetch_add(1, std::memory_order_relaxed); if (i >= count) break; operation(i); } };
                try { for (size_t t = 1; t < std::min(options.matchingThreads, count); ++t) workers.emplace_back(work); work(); }
                catch (...) { for (auto& worker : workers) worker.join(); throw; }
                for (auto& worker : workers) worker.join();
                };
            if (plan.stagedUsed) {
                struct Representative { KeypointIdentifier key; double score; };
                std::map<LandmarkId, std::vector<KeypointIdentifier>> observations;
                for (const auto& source : plan.sources) for (size_t q = 0; q < source.second->numFrames(); ++q)
                    for (size_t k = 0; k < source.second->numKeypoints(q); ++k) {
                        const LandmarkId id(source.second->landmarkId(q, k));
                        if (points.count(id) && immutable.at(source.second->id()).at(q).queryEnabled[k])
                            observations[id].push_back({ source.second->id(),q,k });
                    }
                std::vector<Result> mapResults(current->numFrames());
                parallelFor(current->numFrames(), [&](size_t t) { try {
                    auto& result = mapResults[t]; const auto guidanceStarted = std::chrono::steady_clock::now();
                    const auto& trainCache = immutable.at(current->id()).at(t); const auto T_CW = trainCache.T_WC.inverse();
                    cv::Mat descriptors; std::vector<cv::KeyPoint> keys; features::OrbBearings rays;
                    features::OrbWordRows words; std::vector<uint64_t> identities; std::vector<KeypointIdentifier> owners;
                    for (const auto& point : observations) {
                        const auto& world = points.at(point.first); const Eigen::Vector3d pc = T_CW.C() * world + T_CW.r();
                        Eigen::Vector2d projection;
                        if (!pc.allFinite() || pc.norm() < 1e-10
                            || current->geometry(t)->project(pc, &projection) != cameras::ProjectionStatus::Successful
                            || !usablePixel(*current->geometry(t), projection)) continue;
                        const Eigen::Vector3d currentView = (world - trainCache.T_WC.r()).normalized();
                        const Eigen::Vector3d expectedBearing = pc.normalized();
                        const double distance = (world - trainCache.T_WC.r()).norm();
                        std::vector<Representative> representatives;
                        for (const auto& key : point.second) {
                            const auto& pose = immutable.at(key.frameId).at(key.cameraIndex).T_WC;
                            const Eigen::Vector3d delta = world - pose.r(); const double range = delta.norm();
                            if (range < 1e-10) continue;
                            const double score = 1 - std::clamp(currentView.dot(delta / range), -1.0, 1.0) + std::abs(std::log(distance / range));
                            representatives.push_back({ key,score });
                        }
                        std::sort(representatives.begin(), representatives.end(), [](const auto& a, const auto& b) {
                            return a.score != b.score ? a.score < b.score : a.key < b.key;
                            });
                        for (size_t r = 0; r < std::min(options.mapDescriptorsPerLandmark, representatives.size()); ++r) {
                            const auto& key = representatives[r].key; const auto& frame = *plan.sources.at(StateId(key.frameId));
                            auto kp = frame.keypoints(key.cameraIndex)[key.keypointIndex];
                            kp.pt = cv::Point2f(float(projection.x()), float(projection.y())); keys.push_back(kp);
                            words.push_back(immutable.at(key.frameId).at(key.cameraIndex).words[key.keypointIndex]);
                            descriptors.push_back(frame.descriptors(key.cameraIndex).row(int(key.keypointIndex)));
                            rays.push_back(expectedBearing); identities.push_back(point.first.value()); owners.push_back(key);
                        }
                    }
                    if (keys.empty()) return;
                    features::OrbMatchGuidance guidance;
                    // Use the projection neighborhood for descriptor ambiguity tests.
                    // Applying the final tight angular gate here would often leave only
                    // one neighbor and make strict ratio tests impossible. Validate the
                    // winning finite-point bearing before claiming it below.
                    guidance.useSphericalGeometry = false;
                    guidance.queryCamera = current->geometry(t); guidance.trainCamera = current->geometry(t);
                    guidance.queryBearings = &rays; guidance.trainBearings = &trainCache.bearings;
                    guidance.queryWords = &words; guidance.trainWords = &trainCache.words; guidance.trainIndex = &trainCache.trainIndex;
                    guidance.queryIdentities = &identities; guidance.trainEnabled = trainCache.trainEnabled;
                    guidance.searchRadiusPixels = options.searchRadiusPixels;
                    for (const auto& key : keys) guidance.predictedTrainPixels.push_back(key.pt);
                    auto mapOptions = options.matcher; mapOptions.rotationThresholdRadians = std::min(options.maximumAngularError, 0.49);
                    features::OrbMatcher mapMatcher(mapOptions); std::vector<cv::DMatch> matches;
                    result.guidanceMs = milliseconds(guidanceStarted);
                    mapMatcher.matchGuided(descriptors, current->descriptors(t), keys, current->keypoints(t), guidance, matches, &result.statistics);
                    for (const auto& match : matches) {
                        const KeypointIdentifier key{ current->id(),t,size_t(match.trainIdx) };
                        if (fits(*current, key, trainCache.T_WC, points.at(LandmarkId(identities[size_t(match.queryIdx)])), options.maximumAngularError))
                            result.matches.push_back({ owners[size_t(match.queryIdx)],key,match.distance });
                    }
                }
                catch (...) { mapResults[t].error = std::current_exception(); } });
                for (const auto& result : mapResults) {
                    accumulate(result); if (result.statistics.predictedQueries) ++plan.mapJobs;
                    for (const auto& match : result.matches) mapClaimed[match.second.cameraIndex][match.second.keypointIndex] = 1;
                }
            }
            // Reliability is target-camera local. One weak camera must not force every
            // other camera to repeat all historical cross-camera searches.
            std::vector<uint8_t> mapReady(current->numFrames(), 0);
            plan.stageMapMatches.resize(current->numFrames(), 0);
            for (size_t c = 0; c < mapClaimed.size(); ++c) if (current->numKeypoints(c)) {
                const auto count = size_t(std::count(mapClaimed[c].begin(), mapClaimed[c].end(), uint8_t(1)));
                plan.stageMapMatches[c] = count;
                mapReady[c] = plan.stagedUsed && count >= std::min<size_t>(10, options.minimumStageMatchesPerCamera);
            }
            const auto collect = [&](const MultiFrame& source, size_t q, size_t t, bool stereo, Result& result) {
                const auto guidanceStarted = std::chrono::steady_clock::now();
                if (source.descriptors(q).empty() || current->descriptors(t).empty()) return;
                features::OrbMatchGuidance guidance;
                guidance.queryCamera = source.geometry(q); guidance.trainCamera = current->geometry(t);
                const auto& queryCache = immutable.at(source.id()).at(q);
                const auto& trainCache = immutable.at(current->id()).at(t);
                const auto& T_WQ = queryCache.T_WC;
                const auto& T_WT = trainCache.T_WC;
                guidance.T_train_query = T_WT.inverse() * T_WQ;
                if (stereo) {
                    if (!current->hasOverlap(t, q) || !current->hasOverlap(q, t)) {
                        VLOG(2) << "[Match] Stereo pair skipped; frame=" << current->id() << " cameras=" << q << ',' << t << " reason=no_FOV_overlap";
                        return;
                    }
                    guidance.queryRegionMask = current->overlap(t, q); guidance.trainRegionMask = current->overlap(q, t);
                }
                guidance.queryEnabled = queryCache.queryEnabled; guidance.trainEnabled = trainCache.trainEnabled;
                guidance.queryBearings = &queryCache.bearings; guidance.trainBearings = &trainCache.bearings;
                guidance.queryWords = &queryCache.words; guidance.trainWords = &trainCache.words;
                guidance.queryPriorities=&queryCache.queryPriorities;
                guidance.trainIndex = &trainCache.trainIndex;
                if (mapReady[t]) {
                    for (size_t k = 0; k < guidance.trainEnabled.size(); ++k) if (mapClaimed[t][k]) guidance.trainEnabled[k] = 0;
                    if (!stereo) for (size_t k = 0; k < guidance.queryEnabled.size(); ++k)
                        if (points.count(LandmarkId(source.landmarkId(q, k)))) guidance.queryEnabled[k] = 0;
                }
                guidance.allowUnpredictedQueries = true; guidance.searchRadiusPixels = options.searchRadiusPixels;
                const float unavailable = std::numeric_limits<float>::quiet_NaN();
                guidance.predictedTrainPixels.assign(source.numKeypoints(q), cv::Point2f(unavailable, unavailable));
                const auto T_CW = T_WT.inverse();
                for (size_t i = 0; i < source.numKeypoints(q); ++i) {
                    if (!guidance.queryEnabled[i]) continue;
                    const LandmarkId id(source.landmarkId(q, i)); const auto point = points.find(id);
                    Eigen::Vector2d projected;
                    if (point != points.end()) {
                        const Eigen::Vector3d& world = point->second;
                        // Vector3 multiplication rotates directions only in this project.
                        // A finite map point requires the complete affine transformation.
                        const Eigen::Vector3d pc = T_CW.C() * world + T_CW.r();
                        if (current->geometry(t)->project(pc, &projected) != cameras::ProjectionStatus::Successful
                            || !usablePixel(*current->geometry(t), projected)) {
                            guidance.queryEnabled[i] = 0; continue;
                        }
                        guidance.predictedTrainPixels[i] = cv::Point2f(float(projected.x()), float(projected.y()));
                    }
                }
                std::vector<cv::DMatch> matches;
                result.guidanceMs = milliseconds(guidanceStarted);
                matcher.matchGuided(source.descriptors(q), current->descriptors(t), source.keypoints(q), current->keypoints(t), guidance, matches, &result.statistics);
                VLOG(2) << "[Match] " << (stereo ? "stereo" : "temporal") << " query=" << source.id() << ':' << q
                    << " train=" << current->id() << ':' << t << " guided_matches=" << matches.size();
                for (const auto& match : matches) result.matches.push_back({ {source.id(),q,size_t(match.queryIdx)},
                    {current->id(),t,size_t(match.trainIdx)},match.distance });
                };
            // Workers read owned snapshots only; no worker reads or mutates the estimator.
            // Merge in job order, then sort by the original distance/key ordering so thread
            // scheduling cannot change track ownership or tie resolution.
            const auto runJobs = [&](const std::vector<size_t>& indices) {
                parallelFor(indices.size(), [&](size_t item) {
                    const auto index = indices[item];
                    try { const auto& job = jobs[index]; collect(*job.source, job.q, job.t, job.stereo, results[index]); }
                    catch (...) { results[index].error = std::current_exception(); }
                    });
                for (auto index : indices) accumulate(results[index]);
                plan.seedJobs += indices.size();
                };
            std::vector<size_t> primary, remaining;
            const auto latestSource = plan.sources.empty() ? StateId() : plan.sources.rbegin()->first;
            for (size_t i = 0; i < jobs.size(); ++i) {
                const auto& job = jobs[i];
                if (!plan.stagedUsed || job.stereo || (StateId(job.source->id()) == latestSource && job.q == job.t)) primary.push_back(i);
                else remaining.push_back(i);
            }
            runJobs(primary);
            if (plan.stagedUsed) {
                const auto weakCoverage = [&](std::vector<size_t>& counts, std::vector<unsigned>& sectors,std::vector<size_t>& existingCounts) {
                    std::vector<std::set<size_t>> covered(current->numFrames()),existingCovered(current->numFrames());
                    sectors.assign(current->numFrames(), 0); counts.assign(current->numFrames(), 0);
                    existingCounts.assign(current->numFrames(),0);
                    const auto cover = [&](const KeypointIdentifier& key) {
                        if (key.frameId != current->id()) return;
                        const auto& point = current->keypoints(key.cameraIndex)[key.keypointIndex].pt;
                        covered[key.cameraIndex].insert(key.keypointIndex);
                        const int sector = int(point.x >= current->geometry(key.cameraIndex)->imageWidth() * 0.5)
                            + 2 * int(point.y >= current->geometry(key.cameraIndex)->imageHeight() * 0.5);
                        sectors[key.cameraIndex] |= 1u << sector;
                        };
                    for (const auto& match : candidates) {
                        cover(match.first); cover(match.second);
                        if(match.first.frameId!=current->id()
                            && plan.sources.at(StateId(match.first.frameId))->landmarkId(match.first.cameraIndex,match.first.keypointIndex))
                            existingCovered[match.second.cameraIndex].insert(match.second.keypointIndex);
                    }
                    std::vector<uint8_t> weak(current->numFrames(), 0);
                    for (size_t c = 0; c < current->numFrames(); ++c) if (current->numKeypoints(c)) {
                        counts[c] = covered[c].size(); const unsigned mask = sectors[c];
                        existingCounts[c]=existingCovered[c].size();
                        const bool twoSectors = mask && (mask & (mask - 1));
                        // New texture candidates do not certify continuation of the
                        // map/seed identities. Retain a small independent existing-ID
                        // floor before suppressing historical recovery for this camera.
                        weak[c] = counts[c] < options.minimumStageMatchesPerCamera || !twoSectors
                            || existingCounts[c]<std::min<size_t>(10,options.minimumStageMatchesPerCamera);
                    }
                    return weak;
                    };
                auto weak = weakCoverage(plan.stagePrimaryMatches, plan.stagePrimarySectors,plan.stagePrimaryExistingMatches);
                const auto& oldPose = immutable.at(latestSource.value()).at(0).T_WC;
                const auto& newPose = immutable.at(current->id()).at(0).T_WC;
                const double rotation = std::acos(std::clamp(0.5 * ((oldPose.C().transpose() * newPose.C()).trace() - 1), -1.0, 1.0));
                plan.stageRotationRadians = rotation;
                plan.stageFallbackReason = "sufficient_support";
                if (rotation > 0.10) {
                    // IMU-predicted turns can move features between any camera pair.
                    plan.stageFallbackReason = "rotation"; plan.broadJobs = remaining.size();
                    plan.broadFallback = !remaining.empty(); runJobs(remaining);
                }
                else {
                    std::vector<size_t> expansion, deferred;
                    for (auto i : remaining) {
                        const auto& job = jobs[i];
                        if (weak[job.t] && (job.q == job.t || StateId(job.source->id()) == latestSource)) expansion.push_back(i);
                        else deferred.push_back(i);
                    }
                    // First recover same-camera history and newest cross-camera handoffs
                    // for deficient targets. Do not cull temporal pairs by FOV overlap.
                    plan.expansionJobs = expansion.size(); runJobs(expansion);
                    std::vector<unsigned> expandedSectors;
                    weak = weakCoverage(plan.stageExpandedMatches, expandedSectors,plan.stageExpandedExistingMatches);
                    std::vector<size_t> broad;
                    for (auto i : deferred) if (weak[jobs[i].t]) broad.push_back(i);
                    plan.broadJobs = broad.size(); plan.broadFallback = !broad.empty();
                    if (!broad.empty()) plan.stageFallbackReason = "camera_coverage";
                    else if (!expansion.empty()) plan.stageFallbackReason = "camera_expansion";
                    runJobs(broad);
                }
            }
            if (plan.stagedUsed && (VLOG_IS_ON(1) || current->id() % 30 == 0 || plan.broadFallback)) {
                std::ostringstream support,existingSupport;
                for (size_t c = 0; c < current->numFrames(); ++c) {
                    if (c) support << ';';
                    support << c << ':' << plan.stageMapMatches[c] << '/' << int(mapReady[c]) << '/'
                        << plan.stagePrimaryMatches[c] << '/' << plan.stagePrimarySectors[c] << '/';
                    if (plan.stageExpandedMatches.empty()) support << '-'; else support << plan.stageExpandedMatches[c];
                    if(c) existingSupport<<';';
                    existingSupport<<c<<':'<<plan.stagePrimaryExistingMatches[c]<<'/';
                    if(plan.stageExpandedExistingMatches.empty()) existingSupport<<'-'; else existingSupport<<plan.stageExpandedExistingMatches[c];
                }
                LOG(INFO) << "[Match][Stages] frame=" << current->id() << " reason=" << plan.stageFallbackReason
                    << " rotation_rad=" << plan.stageRotationRadians << " primary_jobs=" << primary.size()
                    << " expansion_jobs=" << plan.expansionJobs << " broad_jobs=" << plan.broadJobs
                    << " camera_map_ready_primary_sectors_expanded=" << support.str()
                    << " camera_existing_primary_expanded="<<existingSupport.str();
            }
            plan.matchingJobs = plan.mapJobs + plan.seedJobs; plan.matchingMs = milliseconds(matchingStarted);
            const auto assignmentStarted = std::chrono::steady_clock::now();
            VLOG(1) << "[Match] frame=" << current->id() << " source_frames=" << plan.sources.size()
                << " guided_candidate_pairs=" << candidates.size();
            std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                return std::tie(a.distance, a.first, a.second) < std::tie(b.distance, b.first, b.second);
                });
            std::map<KeypointIdentifier, LandmarkId> assigned;
            std::set<Slot> occupied;
            // Only selected feature frames can be candidate endpoints. Avoid
            // rebuilding two trees from every observation in the whole window.
            const auto indexFrame=[&](StateId id) {
                for(const auto& key:estimator.window().landmarks().inFrame(id)) {
                    const auto& observation=*estimator.window().landmarks().find(key);
                    if(retiredSeedIds.count(observation.landmark)) continue;
                    assigned.emplace(key,observation.landmark);
                    // attach() can add only CURRENT observations. Historical keys
                    // already have immutable ownership in assigned; duplicating all
                    // their camera slots into a second tree adds no admission check.
                    if(key.frameId==current->id()) occupied.emplace(observation.landmark,key.frameId,key.cameraIndex);
                }
            };
            indexFrame(currentId); for(const auto& source:plan.sources) indexFrame(source.first);
            const size_t cellCount=current->numFrames()*16;
            std::vector<size_t> load(cellCount,0),cellFeatures(cellCount,0),cursor(cellCount,0);
            const auto cellFor=[&](const KeypointIdentifier& key) {
                const auto pixel=current->keypoints(key.cameraIndex)[key.keypointIndex].pt;
                const auto camera=current->geometry(key.cameraIndex);
                return key.cameraIndex*16+size_t(std::clamp(int(pixel.y*4/camera->imageHeight()),0,3)*4
                    +std::clamp(int(pixel.x*4/camera->imageWidth()),0,3));
            };
            for(size_t c=0;c<current->numFrames();++c) for(size_t k=0;k<current->numKeypoints(c);++k)
                ++cellFeatures[cellFor({current->id(),c,k})];
            for(const auto& item:assigned) if(item.first.frameId==current->id()) ++load[cellFor(item.first)];
            auto next = estimator.nextLandmarkId().value();
            const auto frame = [&](const KeypointIdentifier& key)->MultiFrame& {
                return key.frameId == current->id() ? *plan.current : *plan.sources.at(StateId(key.frameId));
                };
            const auto noiseAt = [&](const KeypointIdentifier& key) {
                backend::ObservationOptions noise;
                noise.pixelSigma = options.pixelSigma * std::pow(options.pyramidScale,
                    frame(key).keypoints(key.cameraIndex)[key.keypointIndex].octave);
                return noise;
                };
            const auto attach = [&](LandmarkId id, const KeypointIdentifier& key) {
                if(key.frameId==current->id() && occupied.count({id,key.frameId,key.cameraIndex}) && plan.sourceSeedIds.count(id))
                    ++plan.seedSlotConflicts;
                if (!observationSlots || key.frameId != current->id() || assigned.count(key)
                    || occupied.count({ id,key.frameId,key.cameraIndex })) return false;
                const auto point = points.find(id);
                if (point != points.end()) {
                    if (!fits(frame(key), key, immutable.at(current->id()).at(key.cameraIndex).T_WC,
                        point->second, options.maximumAngularError)) return false;
                }
                plan.batch.observations.push_back({ id,key,noiseAt(key) }); assigned[key] = id;
                --observationSlots;
                occupied.emplace(id, key.frameId, key.cameraIndex);
                ++load[cellFor(key)];
                if(plan.sourceSeedIds.count(id)) plan.extendedSeedIds.insert(id);
                frame(key).setLandmarkId(key.cameraIndex, key.keypointIndex, id.value()); return true;
                };
            // Existing tracks receive their current measurements before new seeds can
            // consume any observation budget. The distance ordering within each pass stays deterministic.
            for (const auto& candidate : candidates) {
                const auto a = assigned.find(candidate.first), b = assigned.find(candidate.second);
                if (a != assigned.end() && b == assigned.end()) attach(a->second, candidate.second);
                else if (b != assigned.end() && a == assigned.end()) attach(b->second, candidate.first);
            }
            const auto admit=[&](const Candidate& candidate) {
                auto a = assigned.find(candidate.first), b = assigned.find(candidate.second);
                if (a != assigned.end() && b != assigned.end()) {
                    if(a->second!=b->second) ++plan.identityConflicts;
                    return false; // Never merge distinct identities silently.
                }
                if (a != assigned.end()) { attach(a->second, candidate.second); return false; }
                if (b != assigned.end()) { attach(b->second, candidate.first); return false; }
                // Once a camera has reliable finite-map support, preserve existing
                // IDs first and replenish only spatial gaps. Weak/bootstrap/recovery
                // targets retain the full admission path. Never gate extensions here.
                const auto covered=[&](const KeypointIdentifier& key) {
                    if(key.frameId!=current->id() || !options.replenishSeedGaps || !mapReady[key.cameraIndex]) return false;
                    const auto cell=cellFor(key);
                    return load[cell]>=(cellFeatures[cell]+1)/2;
                };
                if(covered(candidate.second) && (candidate.first.frameId!=current->id() || covered(candidate.first))) {
                    ++plan.seedCoverageSkipped; return false;
                }
                if (plan.batch.landmarks.size() >= plan.admittedTrackLimit || observationSlots < 2 || !next
                    || next == std::numeric_limits<uint64_t>::max()) {
                    ++plan.capacitySkipped;
                    if(options.maxTrackingSeeds && plan.trackingSeeds+plan.batch.landmarks.size()>=options.maxTrackingSeeds) ++plan.seedBudgetSkipped;
                    return false;
                }
                const LandmarkId id(next++);
                plan.batch.landmarks.push_back({ id,candidate.first,std::nullopt,noiseAt(candidate.first) });
                --observationSlots; // The backend also inserts the anchor measurement.
                assigned[candidate.first] = id; occupied.emplace(id, candidate.first.frameId, candidate.first.cameraIndex);
                frame(candidate.first).setLandmarkId(candidate.first.cameraIndex, candidate.first.keypointIndex, id.value());
                if(candidate.first.frameId==current->id()) ++load[cellFor(candidate.first)];
                if (!attach(id, candidate.second)) {
                    assigned.erase(candidate.first); occupied.erase({ id,candidate.first.frameId,candidate.first.cameraIndex });
                    frame(candidate.first).setLandmarkId(candidate.first.cameraIndex, candidate.first.keypointIndex, 0);
                    if(candidate.first.frameId==current->id()) --load[cellFor(candidate.first)];
                    plan.batch.landmarks.pop_back(); ++observationSlots;
                    return false;
                }
                return true;
            };
            // Existing identities were extended first. Balance only new identities
            // across per-camera 4x4 cells; distances still order each cell's queue.
            std::vector<std::vector<size_t>> cells(cellCount);
            for(size_t i=0;i<candidates.size();++i) cells[cellFor(candidates[i].second)].push_back(i);
            using CellEntry=std::tuple<size_t,float,size_t>;
            std::priority_queue<CellEntry,std::vector<CellEntry>,std::greater<CellEntry>> queue;
            std::vector<uint8_t> admittedCells(cellCount,0);
            for(size_t cell=0;cell<cellCount;++cell) if(!cells[cell].empty())
                queue.emplace(load[cell],candidates[cells[cell].front()].distance,cell);
            while(!queue.empty()) {
                const size_t cell=std::get<2>(queue.top()); queue.pop();
                if(admit(candidates[cells[cell][cursor[cell]]])) admittedCells[cell]=1;
                if(++cursor[cell]<cells[cell].size()) queue.emplace(load[cell],candidates[cells[cell][cursor[cell]]].distance,cell);
            }
            plan.newSeedCells=size_t(std::count(admittedCells.begin(),admittedCells.end(),uint8_t(1)));
            if (options.collectDebugMatches) plan.debugMatches = std::move(candidates);
            plan.assignmentMs = milliseconds(assignmentStarted);
            return plan;
        }
        void reject(Plan& plan, const std::set<KeypointIdentifier>& keys) {
            auto& observations = plan.batch.observations;
            observations.erase(std::remove_if(observations.begin(), observations.end(), [&](const auto& observation) {
                if (!keys.count(observation.key)) return false;
                plan.current->setLandmarkId(observation.key.cameraIndex, observation.key.keypointIndex, 0); return true;
                }), observations.end());
            // Rejecting a proposed anchor invalidates that entire new track. Removing the
            // last non-anchor view also discards it, avoiding one-view seed accumulation.
            std::set<LandmarkId> supported, orphaned;
            for (const auto& observation : observations) supported.insert(observation.landmark);
            const auto clear = [&](const KeypointIdentifier& key) {
                auto& f = key.frameId == plan.current->id() ? *plan.current : *plan.sources.at(StateId(key.frameId));
                f.setLandmarkId(key.cameraIndex, key.keypointIndex, 0);
                };
            auto& landmarks = plan.batch.landmarks;
            for (const auto& landmark : landmarks) if (keys.count(landmark.anchor) || !supported.count(landmark.id)) {
                orphaned.insert(landmark.id); clear(landmark.anchor);
            }
            observations.erase(std::remove_if(observations.begin(), observations.end(), [&](const auto& observation) {
                if (!orphaned.count(observation.landmark)) return false;
                clear(observation.key); return true;
                }), observations.end());
            landmarks.erase(std::remove_if(landmarks.begin(), landmarks.end(), [&](const auto& landmark) {
                return orphaned.count(landmark.id) != 0;
                }), landmarks.end());
        }
        void filterMap(const Estimator& estimator, Plan& plan, const Transform& T_WB, double limit) {
            std::set<KeypointIdentifier> rejected;
            std::vector<Transform> poses;
            for (size_t c = 0; c < plan.current->numFrames(); ++c) poses.push_back(T_WB * estimator.extrinsics(StateId(plan.current->id()), c));
            for (const auto& observation : plan.batch.observations) {
                const auto point = plan.finitePoints.find(observation.landmark);
                if (point != plan.finitePoints.end() && !fits(*plan.current, observation.key,
                    poses.at(observation.key.cameraIndex), point->second, limit))
                    rejected.insert(observation.key);
            }
            reject(plan, rejected);
        }
        void prepareBearingVelocities(const Estimator& estimator,Plan& plan) {
            const auto& store=estimator.window().landmarks();
            std::map<LandmarkId,backend::LandmarkInsertion*> inserted;
            for (auto& point : plan.batch.landmarks) inserted.emplace(point.id,&point);
            for (auto& view : plan.batch.observations) {
                const auto& key=view.key;
                if (key.frameId!=plan.current->id()) continue;
                KeypointIdentifier previous;
                const auto& historyKeys=store.ofLandmark(view.landmark);
                for (auto it=historyKeys.rbegin(); it!=historyKeys.rend(); ++it) {
                    if (it->frameId<key.frameId && it->cameraIndex==key.cameraIndex && !store.consumed(*it)
                        && estimator.window().epochs().at(StateId(it->frameId)).visual==backend::SensorHealth::Valid) {
                        previous=*it;
                        break;
                    }
                }
                const auto newPoint=inserted.find(view.landmark);
                if (!previous.frameId && newPoint!=inserted.end()) {
                    const auto& anchor=newPoint->second->anchor;
                    if (anchor.frameId<key.frameId && anchor.cameraIndex==key.cameraIndex
                        && estimator.window().epochs().at(StateId(anchor.frameId)).visual==backend::SensorHealth::Valid) previous=anchor;
                }
                if (!previous.frameId) continue;
                const auto history=estimator.window().frames().measurements(StateId(previous.frameId));
                if (!history) continue;
                const auto& before=history->imageTiming(key.cameraIndex);
                const auto& now=plan.current->imageTiming(key.cameraIndex);
                const uint64_t t0=(before && now?before->originalTimestamp:history->timestamp()).toNSec();
                const uint64_t t1=(before && now?now->originalTimestamp:plan.current->timestamp()).toNSec();
                if (t1<=t0) continue;
                const double dt=double(t1-t0)*1e-9;
                if (dt<0.001 || dt>0.5) continue;
                Eigen::Vector3d a,b;
                if (!bearing(*history,previous,a) || !bearing(*plan.current,key,b)) continue;
                const double cosine=std::clamp(a.dot(b),-1.0,1.0);
                const double sine=a.cross(b).norm();
                const double angle=std::atan2(sine,cosine);
                if (angle>0.5 || angle/dt>8.0) continue;
                const double scale=(sine>1e-8?angle/sine:1.0)/dt;
                // Extrapolate forward along the measured track. The residual uses
                // (referenceTimeOffset - td), so reference is tied to the fixed
                // bundle time, not to the estimated td at observation creation.
                view.options.bearingVelocity=scale*(cosine*b-a);
                view.options.referenceTimeOffset=now?-double(now->offsetNs)*1e-9:0.0;
                if (newPoint!=inserted.end() && newPoint->second->anchor==previous) {
                    newPoint->second->options.bearingVelocity=scale*(b-cosine*a);
                    newPoint->second->options.referenceTimeOffset=before?-double(before->offsetNs)*1e-9:0.0;
                }
            }
        }
        void triangulate(const Estimator& estimator, Plan& plan, const Options& options, const Transform& T_WB, bool allowTemporal, bool currentKeyframe) {
            const StateId currentId(plan.current->id());
            std::optional<StateId> victim;
            if (estimator.window().needsMarginalization()) {
                const auto& order = estimator.window().order();
                victim = estimator.window().choosePolicy() == backend::MarginalizationPolicy::MARGIN_OLD
                    ? order.front() : order[order.size() - 2];
            }
            std::map<LandmarkId, KeypointIdentifier> anchors;
            std::map<LandmarkId, std::vector<KeypointIdentifier>> views;
            std::map<StateId, std::shared_ptr<const MultiFrame>> cache;
            cache.emplace(currentId, plan.current);
            for (const auto& source : plan.sources) cache.emplace(source.first, source.second);
            for (const auto& observation : plan.batch.observations) {
                views[observation.landmark].push_back(observation.key);
                const auto metadata = estimator.window().landmarks().find(observation.landmark);
                if (metadata && !metadata->initialized) anchors.emplace(metadata->id, metadata->anchor);
            }
            for (const auto& landmark : plan.batch.landmarks) anchors.emplace(landmark.id, landmark.anchor);
            std::map<KeypointIdentifier,const backend::ObservationOptions*> timing;
            double td=0.0;
            if (options.estimateCameraTimeOffset) {
                const auto& blocks=estimator.window().graph().blocks;
                const auto offset=blocks.find({backend::BlockKind::CameraTimeOffset,0});
                if (offset!=blocks.end()) td=offset->second.value[0];
                for (const auto& observation : plan.batch.observations) timing.emplace(observation.key,&observation.options);
                for (const auto& point : plan.batch.landmarks) timing.emplace(point.anchor,&point.options);
            }
            struct WorldRay { Eigen::Vector3d origin, direction; bool valid = false; };
            std::map<KeypointIdentifier, WorldRay> worldRays;
            std::map<std::pair<StateId, size_t>, Transform> poses;
            const auto ray = [&](const KeypointIdentifier& key)->const WorldRay& {
                auto inserted = worldRays.emplace(key, WorldRay{});
                if (!inserted.second) return inserted.first->second;
                auto& result = inserted.first->second;
                auto it = cache.find(StateId(key.frameId));
                if (it == cache.end()) it = cache.emplace(StateId(key.frameId), estimator.window().frames().measurements(StateId(key.frameId))).first;
                if (!it->second || !bearing(*it->second, key, result.direction)) return result;
                if (options.estimateCameraTimeOffset) {
                    const auto fresh=timing.find(key);
                    if (fresh!=timing.end()) {
                        const auto& measurement=*fresh->second;
                        result.direction=backend::timeBearing(result.direction,measurement.bearingVelocity,
                            measurement.referenceTimeOffset.value_or(td)-td);
                    } else if (const auto* measurement=estimator.window().landmarks().find(key)) {
                        result.direction=backend::timeBearing(measurement->bearing,measurement->bearingVelocity,
                            measurement->referenceTimeOffset-td);
                    }
                }
                const auto index = std::make_pair(StateId(key.frameId), key.cameraIndex);
                auto pose = poses.find(index);
                if (pose == poses.end()) pose = poses.emplace(index, cameraPose(estimator, key, currentId, T_WB)).first;
                result.origin = pose->second.r(); result.direction = pose->second.C() * result.direction;
                result.valid = result.origin.allFinite() && result.direction.allFinite(); return result;
                };
            struct Promotion { LandmarkId id; double rho,parallax; size_t bin; };
            std::vector<Promotion> promotions;
            for (const auto& item : anchors) {
                auto& all = views[item.first];
                if(!estimator.isObservationConsumed(item.second)) all.push_back(item.second);
                for (const auto& key : estimator.window().landmarks().ofLandmark(item.first))
                    if (!estimator.isObservationConsumed(key)) all.push_back(key);
                std::sort(all.begin(), all.end()); all.erase(std::unique(all.begin(), all.end()), all.end());
                const auto stable=[&](const KeypointIdentifier& key) {
                    if(victim && key.frameId==victim->value()) return false;
                    if(estimator.isObservationConsumed(key)) return false;
                    if(key.frameId==currentId.value()) return !options.requireKeyframeAnchor || currentKeyframe;
                    const auto epoch=estimator.window().epochs().find(StateId(key.frameId));
                    return epoch!=estimator.window().epochs().end() && epoch->second.visual!=backend::SensorHealth::Invalid
                        && (!options.requireKeyframeAnchor || epoch->second.keyframe);
                };
                auto anchorKey=item.second;
                if(!stable(anchorKey)) {
                    const KeypointIdentifier* replacement=nullptr;
                    for(const auto& key:all) if(stable(key)
                        && (!replacement || key.frameId>replacement->frameId || (key.frameId==replacement->frameId && key<*replacement))) replacement=&key;
                    if(replacement) {
                        anchorKey=*replacement;
                        // Seed reference selection transports no optimized rho or prior.
                        // The backend applies it atomically after these raw views exist.
                        auto update=std::find_if(plan.batch.seedAnchors.begin(),plan.batch.seedAnchors.end(),[&](const auto& value) { return value.id==item.first; });
                        if(update==plan.batch.seedAnchors.end()) plan.batch.seedAnchors.push_back({item.first,anchorKey});
                        else update->anchor=anchorKey;
                    }
                }
                const bool transient=!stable(anchorKey);
                if (transient) { ++plan.deferredPromotions; continue; }
                if (estimator.window().promotionWouldBeMarginalized(anchorKey, all)) {
                    // A retained anchor alone is insufficient: a fresh outgoing view
                    // would attach a live factor and make OLD consume this new rho.
                    // Keep the ID and raw views as a seed until that view has left.
                    ++plan.deferredPromotions; ++plan.deferredEvictionSupport; continue;
                }
                // Without accepted metric temporal motion, promote only a track whose
                // complete support is one capture's known rig baseline. Historical stereo
                // must not smuggle an unverified current observation into a map factor.
                if (!allowTemporal && std::any_of(all.begin(), all.end(), [&](const auto& key) {
                    return key.frameId != anchorKey.frameId;
                    })) continue;
                const auto& anchorRay = ray(anchorKey); if (!anchorRay.valid) continue;
                const auto& origin = anchorRay.origin; const auto& direction = anchorRay.direction;
                std::vector<const WorldRay*> retainedRays; retainedRays.reserve(all.size());
                bool allValid = true;
                for (const auto& view : all) { const auto& value = ray(view); if (!value.valid) { allValid = false; break; } retainedRays.push_back(&value); }
                if (!allValid) continue;
                if (victim) {
                    // The impending eviction must leave actual range support, not just
                    // two copies of one ray. Otherwise promotion immediately followed by
                    // demotion spends work and injects a short-lived measurement model.
                    bool supported = false;
                    for (size_t i = 0; i < all.size() && !supported; ++i) if (all[i].frameId != victim->value())
                        for (size_t j = i + 1; j < all.size(); ++j) if (all[j].frameId != victim->value()) {
                            const auto& a = *retainedRays[i]; const auto& b = *retainedRays[j];
                            const double angle = std::atan2(a.direction.cross(b.direction).norm(), a.direction.dot(b.direction));
                            if ((a.origin - b.origin).norm() > 1e-8 && angle >= options.minimumParallax) { supported = true; break; }
                        }
                    if (!supported) { ++plan.deferredPromotions; continue; }
                }
                double bestParallax = 0.0, bestDepth = 0.0;
                for (const auto& key : all) {
                    if (key == anchorKey || (!allowTemporal && key.frameId != anchorKey.frameId)) continue;
                    const auto& other = ray(key); if (!other.valid) continue;
                    const auto& otherOrigin = other.origin; const auto& otherDirection = other.direction;
                    const double cosine = std::clamp(direction.dot(otherDirection), -1.0, 1.0);
                    const double parallax = std::acos(cosine), denominator = 1.0 - cosine * cosine;
                    if (parallax < options.minimumParallax || denominator < 1e-10 || parallax <= bestParallax) continue;
                    const Eigen::Vector3d baseline = otherOrigin - origin;
                    const double depth = (direction.dot(baseline) - cosine * otherDirection.dot(baseline)) / denominator;
                    const double otherDepth = (cosine * direction.dot(baseline) - otherDirection.dot(baseline)) / denominator;
                    if (depth<options.minimumDepth || depth>options.maximumDepth || otherDepth < options.minimumDepth) continue;
                    // The inverse-range chart constrains the point to the original anchor
                    // bearing; a midpoint of skew rays would violate this representation.
                    const Eigen::Vector3d point = origin + depth * direction; bool valid = true;
                    for (const auto* view : retainedRays) {
                        // Equivalent all-view half-ray/angular test in world coordinates.
                        // No repeated camera inversion/back-projection for each depth hypothesis.
                        const Eigen::Vector3d delta = point - view->origin; const double dot = delta.dot(view->direction);
                        if (!delta.allFinite() || delta.norm() < 1e-10 || dot <= 0
                            || std::atan2(delta.cross(view->direction).norm(), dot) > options.maximumAngularError) {
                            valid = false; break;
                        }
                    }
                    if (valid) { bestDepth = depth; bestParallax = parallax; }
                }
                if (bestDepth <= 0.0) continue;
                const auto& bearing=anchorRay.direction;
                const int azimuth=std::clamp(int((std::atan2(bearing.y(),bearing.x())+3.14159265358979323846)*8/(2*3.14159265358979323846)),0,7);
                const int z=std::clamp(int((bearing.z()+1)*2),0,3);
                promotions.push_back({item.first,1.0/bestDepth,bestParallax,size_t(z*8+azimuth)});
            }
            size_t limit=promotions.size(),active=0;
            for(const auto& entry:estimator.window().landmarks().landmarks()) if(entry.second.initialized) ++active;
            // Bootstrap/recovery retains its original support. Only bound new rho
            // creation during established metric tracking; existing factors survive.
            if(allowTemporal && plan.finitePoints.size()>=60) {
                if(options.maxPromotionsPerFrame) limit=std::min(limit,options.maxPromotionsPerFrame);
                if(options.maxActiveInitialized) limit=std::min(limit,active>=options.maxActiveInitialized?size_t(0):options.maxActiveInitialized-active);
            }
            std::array<size_t,32> occupancy{};
            for(size_t accepted=0;accepted<limit;++accepted) {
                auto best=limit==promotions.size()?promotions.begin()+accepted:std::min_element(promotions.begin()+accepted,promotions.end(),[&](const auto& a,const auto& b) {
                    if(occupancy[a.bin]!=occupancy[b.bin]) return occupancy[a.bin]<occupancy[b.bin];
                    if(a.parallax!=b.parallax) return a.parallax>b.parallax;
                    return a.id<b.id;
                });
                std::iter_swap(promotions.begin()+accepted,best); const auto& chosen=promotions[accepted]; ++occupancy[chosen.bin];
                auto inserted=std::find_if(plan.batch.landmarks.begin(),plan.batch.landmarks.end(),[&](const auto& x) { return x.id==chosen.id; });
                if(inserted!=plan.batch.landmarks.end()) inserted->inverseDepth=chosen.rho;
                else {
                    auto previous=std::find_if(plan.batch.initializations.begin(),plan.batch.initializations.end(),[&](const auto& value) { return value.id==chosen.id; });
                    if(previous==plan.batch.initializations.end()) plan.batch.initializations.push_back({chosen.id,chosen.rho});
                    else previous->inverseDepth=chosen.rho;
                }
            }
            plan.deferredPromotionBudget+=promotions.size()-limit;

        }

        KeyframeDecision decideKeyframe(const Estimator& estimator, const MultiFrame& current,
            const kinematics::Transformation& currentPose, const FrontendParameters& settings, bool trackingAccepted)
        {
            const auto positive = [](double x) { return std::isfinite(x) && x > 0; };
            if (!positive(settings.keyframe_parallax) || !positive(settings.keyframe_translation)
                || !positive(settings.keyframe_rotation) || !positive(settings.keyframe_low_motion_translation)
                || !positive(settings.keyframe_low_motion_rotation) || settings.keyframe_min_shared_tracks < 1)
                throw std::invalid_argument("Invalid keyframe motion/parallax thresholds");
            KeyframeDecision result;
            for (auto it = estimator.window().order().rbegin(); it != estimator.window().order().rend(); ++it) {
                if (it->value() == current.id() || !estimator.window().frames().measurements(*it)) continue;
                if (!result.previous.isInitialised()) result.previous = *it;
                if (estimator.isKeyframe(*it)) { result.reference = *it; break; }
            }
            if (estimator.numFrames() < 3 || !result.previous.isInitialised()) {
                result.keyframe = true; result.reason = "bootstrap"; return result;
            }
            if (!result.reference.isInitialised()) result.reference = result.previous;
            const auto measure = [&](StateId id, size_t& count, double& parallax, double& translation, double& rotation) {
                const auto old = estimator.window().frames().measurements(id); const auto oldPose = estimator.pose(id);
                translation = (currentPose.r() - oldPose.r()).norm();
                rotation = backend::logSO3(currentPose.q() * oldPose.q().conjugate()).norm();
                std::map<std::pair<LandmarkId, size_t>, size_t> oldKeys;
                for (const auto& key : estimator.window().landmarks().inFrame(id)) {
                    const auto observation = estimator.window().landmarks().find(key);
                    if (observation) oldKeys.emplace(std::make_pair(observation->landmark, key.cameraIndex), key.keypointIndex);
                }
                std::map<LandmarkId, double> independent;
                for (size_t c = 0; c < current.numFrames(); ++c) {
                    const Eigen::Matrix3d R = (currentPose * estimator.extrinsics(StateId(current.id()), c)).C().transpose()
                        * (oldPose * estimator.extrinsics(id, c)).C();
                    for (size_t k = 0; k < current.numKeypoints(c); ++k) {
                        const LandmarkId point(current.landmarkId(c, k)); const auto found = oldKeys.find({ point,c });
                        if (!point.isInitialised() || found == oldKeys.end()) continue;
                        Eigen::Vector3d a, b;
                        if (!bearing(*old, { id.value(),c,found->second }, a) || !bearing(current, { current.id(),c,k }, b)) continue;
                        const Eigen::Vector3d rotated = R * a;
                        const double angle = std::atan2(rotated.cross(b).norm(), rotated.dot(b));
                        auto inserted = independent.emplace(point, angle);
                        if (!inserted.second) inserted.first->second = std::min(inserted.first->second, angle);
                    }
                }
                std::vector<double> values; for (const auto& item : independent) values.push_back(item.second);
                count = values.size();
                if (!values.empty()) { auto middle = values.begin() + values.size() / 2; std::nth_element(values.begin(), middle, values.end()); parallax = *middle; }
                };
            measure(result.reference, result.sharedTracks, result.parallax, result.translation, result.rotation);
            if (result.previous == result.reference) {
                result.previousSharedTracks = result.sharedTracks; result.previousParallax = result.parallax;
                result.previousTranslation = result.translation; result.previousRotation = result.rotation;
            }
            else measure(result.previous, result.previousSharedTracks, result.previousParallax, result.previousTranslation, result.previousRotation);
            const size_t minimum = size_t(settings.keyframe_min_shared_tracks);
            if (!trackingAccepted) { result.reason = "visual_unverified"; return result; }
            if (result.translation >= settings.keyframe_translation || result.rotation >= settings.keyframe_rotation) {
                result.keyframe = true; result.reason = "accumulated_motion";
            }
            else if (result.sharedTracks >= minimum && result.parallax >= settings.keyframe_parallax) {
                result.keyframe = true; result.reason = "angular_parallax";
            }
            else if (result.previousSharedTracks >= minimum && result.previousTranslation < settings.keyframe_low_motion_translation
                && result.previousRotation < settings.keyframe_low_motion_rotation && result.previousParallax < 0.25 * settings.keyframe_parallax) {
                result.reason = "low_motion";
            }
            else if (result.sharedTracks >= minimum) result.reason = "low_parallax";
            else { result.keyframe = true; result.reason = "insufficient_shared_tracks"; }
            return result;
        }
    }
}
