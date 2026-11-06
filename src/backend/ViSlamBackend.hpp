#pragma once

#include "FixedLagBackend.hpp"
#include "ProjectBackendAdapter.hpp"
#include <ceres/solver.h>
#include <common/Parameters.hpp>
#include <visual/MultiFrame.hpp>

namespace cgmli {

    /**
     * Project-facing backend API. Owns exactly one active window through FixedLagBackend;
     * there is no second multiFrames_/realtimeGraph_ state to keep in sync manually.
     *
     * Inputs are explicit IMU/body predictions and feature frames. Raw IMU samples are
     * retained for factor construction, NOT silently converted into a pretend IMU factor.
     * The IMU preintegrator supplies its whitened cost through addImuFactor().
     * All mutations/queries below run on the backend thread; snapshot() is cross-thread.
     * The pipeline uses the native transaction directly. No old dual-graph
     * declarations, duplicate sensor configuration or empty compatibility methods
     * are required for local estimation.
     */
    class ViSlamBackend 
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            explicit ViSlamBackend(const backend::BackendOptions& options = {});
        bool initializeRig(const cameras::NCameraSystem& cameras, const kinematics::Transformation& T_SL,
            std::string& error, double tdCamera = 0.0, double tdLidar = 0.0,
            const kinematics::Transformation& T_BS = kinematics::Transformation());
        bool addStates(const MultiFrame& frame, const backend::StateEstimate& prediction,
            const ImuMeasurementDeque& imu, std::string& error);
        bool addState(const backend::StateEstimate& prediction, std::string& error);
        bool applyFrontendUpdate(StateId id,const std::optional<kinematics::Transformation>& T_WB,
            const std::vector<KeypointIdentifier>& rejected,std::string& error) {
            return backend_.applyFrontendUpdate(id,T_WB,rejected,error);
        }
        /// Opaque cost compatibility path: no raw replay is inferred. Native
        /// production uses addFactor(factor::makeBodyImuRecord(...)) so SECOND_NEW
        /// can concatenate intervals; opaque costs use the Schur bridge fallback.
        backend::FactorSubmission addImuFactor(StateId from, StateId to, std::shared_ptr<::ceres::CostFunction> whitenedCost);
        backend::FactorSubmission addFactor(backend::FactorRecord factor) { return backend_.addFactor(std::move(factor)); }
        /// Non-landmark auxiliary variables owned by an external module.
        /// Existing state/managed landmark identities still cannot be bypassed.
        bool addParameter(backend::ParameterBlock block,std::string& error) {
            return backend_.addParameter(std::move(block),error);
        }
        bool setCameraTimeOffsetBounds(double lower, double upper, std::string& error) {
            return backend_.configureParameter({backend::BlockKind::CameraTimeOffset, 0}, true,
                Eigen::VectorXd::Constant(1, lower), Eigen::VectorXd::Constant(1, upper), error);
        }
        bool addLandmark(LandmarkId id, const KeypointIdentifier& anchor, std::optional<double> inverseDepth,
            std::string& error, const backend::ObservationOptions& options = {});
        bool initializeLandmark(LandmarkId id, double inverseDepth, std::string& error);
        bool applyVisualUpdates(const backend::VisualUpdateBatch& batch, std::string& error) { return backend_.applyVisualUpdates(batch, error); }
        bool applyFrontendBatch(StateId id,const backend::VisualUpdateBatch& batch,
            const std::optional<kinematics::Transformation>& T_WB,backend::SensorHealth visual,
            bool keyframe,std::string& error) {
            return backend_.applyFrontendBatch(id,batch,T_WB,visual,keyframe,error);
        }
        bool addObservation(LandmarkId id, StateId frame, size_t camera, size_t keypoint, std::string& error,
            const backend::ObservationOptions& options = {});
        bool removeObservation(StateId frame, size_t camera, size_t keypoint, std::string& error);
        bool removeLandmark(LandmarkId id, std::string& error) { return backend_.retireLandmark(id, error); }
        bool setLandmarkClassification(LandmarkId id, int classification);
        bool setLandmarkQuality(LandmarkId id, double quality, int classification) {
            return backend_.setLandmarkQuality(id, quality, classification);
        }
        bool isObserved(const KeypointIdentifier& key) const;
        bool isObservationConsumed(const KeypointIdentifier& key) const;
        bool isLandmarkAdded(LandmarkId id) const;
        bool isLandmarkInitialised(LandmarkId id) const;
        bool getLandmark(LandmarkId id, MapPoint2& point) const;
        size_t getLandmarks(MapPoints& points) const;
        LandmarkId nextLandmarkId() const { return backend_.window().nextLandmarkId(); }
        // Detached, read-only feature+association snapshot. Editing an old MultiFrame cannot
        // mutate backend data; addObservation()/removeObservation() are the authoritative APIs.
        std::shared_ptr<const MultiFrame> multiFrame(StateId id) const;
        std::shared_ptr<MultiFrame> associationFrame(StateId id) const { return window().associationFrame(id,T_BS_.inverse()); }
        void synchronizeFrame(StateId id,MultiFrame& target) const { window().synchronizeFrame(id,target); }
        kinematics::Transformation pose(StateId id) const;
        SpeedAndBias speedAndBias(StateId id) const;
        kinematics::Transformation extrinsics(StateId id, size_t camera) const;
        bool isKeyframe(StateId id) const;
        bool setKeyframe(StateId id, bool keyframe) { return backend_.setKeyframe(id, keyframe); }
        bool setHealth(StateId id, backend::SensorHealth visual, backend::SensorHealth lidar) {
            return backend_.setHealth(id, visual, lidar);
        }
        StateId currentStateId() const;
        StateId stateIdByAge(size_t age) const;
        StateId currentKeyframeStateId() const;
        size_t numFrames() const { return backend_.window().size(); }
        size_t numLandmarks() const { return backend_.window().landmarks().size(); }
        bool getObservedIds(StateId id, std::set<StateId>& ids) const;
        StateId mostOverlappedStateId(StateId id) const;
        backend::StepReport optimize(std::optional<backend::MarginalizationPolicy> policy = std::nullopt,
            const backend::CalibrationUpdatePolicy& calibration = {}, const backend::SolutionValidator& validator = {},
            const backend::MapCommitGates& mapGates = {}, const backend::SolveBudget& budget = {});
        backend::CalibrationInformation calibrationInformation(bool includeHistoricalPrior = false) {
            return backend_.calibrationInformation(includeHistoricalPrior);
        }
        const backend::SlidingWindowManager& window() const { return backend_.window(); }
        std::shared_ptr<const backend::BackendSnapshot> snapshot() const { return backend_.snapshot(); }

    private:
        backend::FixedLagBackend backend_;
        std::vector<kinematics::Transformation> rigBaselines_; // Immutable geometry only, never optimized T_BR.
        std::vector<std::shared_ptr<const cameras::CameraBase>> cameraModels_;
        kinematics::Transformation T_BS_; // Fixed body/sensor convention, not an estimated extrinsic.
    };
}
