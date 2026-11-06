#include "ViSlamBackend.hpp"
#include <visual/MultiFrame.hpp>
#include <cameras/NCameraSystem.hpp>
#include <stdexcept>

namespace cgmli {
    using namespace backend;
    ViSlamBackend::ViSlamBackend(const BackendOptions& options) : backend_(options) {}
    bool ViSlamBackend::initializeRig(const cameras::NCameraSystem& cameras, const kinematics::Transformation& T_SL,
        std::string& error, double tdCamera, double tdLidar, const kinematics::Transformation& T_BS) {
        RigCalibration rig;
        try { rig = makeRigCalibration(cameras, T_SL, T_BS); }
        catch (const std::exception& e) { error = e.what(); return false; }
        auto models = std::vector<std::shared_ptr<const cameras::CameraBase>>();
        for (size_t c = 0; c < cameras.numCameras(); ++c) models.push_back(cameras.cameraGeometry(c));
        auto baselines = rig.T_RC;
        if (!backend_.initializeCalibration(rig.parameterBlocks(tdCamera, tdLidar), error)) return false;
        cameraModels_ = std::move(models); rigBaselines_ = std::move(baselines); T_BS_ = T_BS; return true;
    }
    bool ViSlamBackend::addStates(const MultiFrame& frame, const StateEstimate& prediction,
        const ImuMeasurementDeque& imu, std::string& error) {
        if (frame.numFrames() != rigBaselines_.size() || cameraModels_.empty()) { error = "initialize a matching rig before adding visual frames"; return false; }
        if (!frame.T_SC(0)) { error = "missing reference extrinsic"; return false; }
        for (size_t c = 0; c < frame.numFrames(); ++c) if (!frame.T_SC(c) || frame.geometry(c).get() != cameraModels_[c].get()
            || (rigBaselines_[c].T() - (frame.T_SC(0)->inverse() * (*frame.T_SC(c))).T()).norm() > 1e-8) {
            error = "visual frame uses a different camera rig"; return false;
        }
        return backend_.append(prediction, frame, imu, error);
    }
    bool ViSlamBackend::addState(const StateEstimate& prediction, std::string& error) { return backend_.append(prediction, error); }
    FactorSubmission ViSlamBackend::addImuFactor(StateId from, StateId to, std::shared_ptr<::ceres::CostFunction> cost) {
        if (!cost || cost->num_residuals() != 15 || cost->parameter_block_sizes() != std::vector<int32_t>{7, 9, 7, 9}) {
            FactorSubmission rejected; rejected.message = "IMU cost must have 15 residuals and block sizes [7,9,7,9]"; return rejected;
        }
        FactorRecord factor; factor.kind = FactorKind::Imu; factor.epochs = { from,to };
        factor.keys = { BlockKey::pose(from),BlockKey::speedBias(from),BlockKey::pose(to),BlockKey::speedBias(to) };
        factor.cost = std::move(cost); return backend_.addFactor(std::move(factor));
    }
    bool ViSlamBackend::addLandmark(LandmarkId id, const KeypointIdentifier& anchor, std::optional<double> rho,
        std::string& error, const ObservationOptions& options) {
        return backend_.addLandmark(id, anchor, rho, error, options);
    }
    bool ViSlamBackend::initializeLandmark(LandmarkId id, double rho, std::string& error) { return backend_.initializeLandmark(id, rho, error); }
    bool ViSlamBackend::addObservation(LandmarkId id, StateId frame, size_t camera, size_t keypoint, std::string& error,
        const ObservationOptions& options) {
        return backend_.addObservation(id, { frame.value(),camera,keypoint }, error, options);
    }
    bool ViSlamBackend::removeObservation(StateId frame, size_t camera, size_t keypoint, std::string& error) {
        return backend_.removeObservation({ frame.value(),camera,keypoint }, error);
    }
    bool ViSlamBackend::setLandmarkClassification(LandmarkId id, int classification) {
        const auto point = window().landmarks().find(id);
        return point && backend_.setLandmarkQuality(id, point->quality, classification);
    }
    bool ViSlamBackend::isObserved(const KeypointIdentifier& key) const { return window().landmarks().find(key) != nullptr; }
    bool ViSlamBackend::isObservationConsumed(const KeypointIdentifier& key) const { return window().landmarks().consumed(key); }
    bool ViSlamBackend::isLandmarkAdded(LandmarkId id) const { return window().landmarks().find(id) != nullptr; }
    bool ViSlamBackend::isLandmarkInitialised(LandmarkId id) const {
        const auto point = window().landmarks().find(id); return point && point->initialized;
    }
    bool ViSlamBackend::getLandmark(LandmarkId id, MapPoint2& point) const { return window().getLandmark(id, point); }
    size_t ViSlamBackend::getLandmarks(MapPoints& points) const {
        points.clear();
        for (const auto& item : window().landmarks().landmarks()) {
            MapPoint2 point; if (window().getLandmark(item.first, point)) points.emplace(item.first, std::move(point));
        }
        return points.size();
    }
    std::shared_ptr<const MultiFrame> ViSlamBackend::multiFrame(StateId id) const { return window().exportFrame(id, T_BS_.inverse()); }
    kinematics::Transformation ViSlamBackend::pose(StateId id) const {
        auto estimate = window().estimate(id); if (!estimate) throw std::out_of_range("unknown state"); return estimate->T_WB;
    }
    SpeedAndBias ViSlamBackend::speedAndBias(StateId id) const {
        auto estimate = window().estimate(id); if (!estimate) throw std::out_of_range("unknown state"); return estimate->speedBias;
    }
    kinematics::Transformation ViSlamBackend::extrinsics(StateId id, size_t camera) const {
        kinematics::Transformation T_WC; std::string error;
        if (!window().frames().cameraPose(window().graph(), id, camera, T_WC, error)) throw std::out_of_range(error);
        return pose(id).inverse() * T_WC;
    }
    bool ViSlamBackend::isKeyframe(StateId id) const {
        auto epoch = window().epochs().find(id); return epoch != window().epochs().end() && epoch->second.keyframe;
    }
    StateId ViSlamBackend::currentStateId() const { return window().order().empty() ? StateId() : window().order().back(); }
    StateId ViSlamBackend::stateIdByAge(size_t age) const {
        const auto& order = window().order(); if (age >= order.size()) throw std::out_of_range("frame age outside window");
        return order[order.size() - 1 - age];
    }
    StateId ViSlamBackend::currentKeyframeStateId() const {
        for (auto it = window().order().rbegin(); it != window().order().rend(); ++it) if (isKeyframe(*it)) return *it;
        return StateId();
    }
    bool ViSlamBackend::getObservedIds(StateId id, std::set<StateId>& ids) const {
        ids.clear(); if (!window().epochs().count(id)) return false;
        for (const auto& key : window().landmarks().inFrame(id)) {
            const auto observation = window().landmarks().find(key);
            for (const auto& other : window().landmarks().ofLandmark(observation->landmark)) if (other.frameId != id.value()) ids.insert(StateId(other.frameId));
        }
        return true;
    }
    StateId ViSlamBackend::mostOverlappedStateId(StateId id) const {
        std::map<StateId, std::set<LandmarkId>> shared;
        for (const auto& key : window().landmarks().inFrame(id)) {
            const auto observation = window().landmarks().find(key);
            for (const auto& other : window().landmarks().ofLandmark(observation->landmark)) if (other.frameId != id.value())
                shared[StateId(other.frameId)].insert(observation->landmark);
        }
        StateId best; size_t count = 0;
        for (const auto& item : shared) if (item.second.size() > count) { best = item.first; count = item.second.size(); }
        return best;
    }
    StepReport ViSlamBackend::optimize(std::optional<MarginalizationPolicy> policy, const CalibrationUpdatePolicy& calibration,
        const SolutionValidator& validator, const MapCommitGates& gates, const SolveBudget& budget) {
        return backend_.step(policy, calibration, validator, gates, {}, budget);
    }

}
