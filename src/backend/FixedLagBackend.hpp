#pragma once
#include "BackendOptimizer.hpp"
#include "SlidingWindowManager.hpp"
#include <atomic>

namespace cgmli {
    namespace backend {
        /// Application-provided evidence required before emitting a marginalized map candidate.
        /// Defaults deny publication. A true result authorizes a candidate, not map mutation;
        /// geometry storage and Stable Map insertion remain the map module's responsibility.
        struct MapCommitGates {
            bool outlierCheckPassed = false, trackingGood = false, calibrationValid = false;
            bool poseInformationSufficient = false, geometryValid = false, noSevereDirectionalDegeneracy = false;
            bool passed() const {
                return outlierCheckPassed && trackingGood && calibrationValid
                    && poseInformationSufficient && geometryValid && noSevereDirectionalDegeneracy;
            }
        };
        // Called for each global calibration block on EVERY step. Return true only if the external
        // excitation/conditional-information controller permits this update. Empty policy freezes all.
        using CalibrationUpdatePolicy = std::function<bool(BlockKey, const BackendGraph&)>;

        /**
         * Public single-writer facade for the backend transaction.
         * Owns one window manager and one optimizer, keeping topology decisions separate
         * from numerical solving. append()/addFactor() stage input; step() accepts or rolls
         * back an optimization-and-marginalization attempt on that staged input.
         * Only snapshot() supports concurrent readers; other methods run on the backend thread.
         */
        class FixedLagBackend {
        public:
            explicit FixedLagBackend(const BackendOptions& options = BackendOptions());
            // Initialization is atomic and allowed only before the first epoch.
            // Blocks must be global calibration variables. All start frozen regardless of
            // their incoming flags; a per-step observability policy is needed to unlock them.
            bool initializeCalibration(const std::vector<ParameterBlock>& blocks, std::string& error);
            /// Forward staging operations without implicitly solving or publishing predictions.
            bool append(const StateEstimate& prediction, std::string& error) { return window_.append(prediction, error); }
            bool append(const StateEstimate& prediction, const MultiFrame& frame, const ImuMeasurementDeque& imu, std::string& error);
            /// Atomically reseed the newest unpublished pose and reject its live
            /// observations after geometric consensus. Does not solve or publish.
            /// Published poses, absorbed evidence and geometric anchors cannot be
            /// silently overwritten/removed by this frontend initialization path.
            bool applyFrontendUpdate(StateId id,const std::optional<kinematics::Transformation>& T_WB,
                const std::vector<KeypointIdentifier>& rejected,std::string& error);
            /// Commit a complete frontend plan after consensus, before optimization.
            /// All extra observations belong to the newest unpublished epoch. New
            /// tracks may anchor in a retained frame; historical evidence is never
            /// removed or merged. Only obsolete uninitialized TrackingOnly seeds may
            /// be retired for admission; bound/absorbed evidence remains protected.
            /// Pose, tracks, health and keyframe flag commit together.
            bool applyFrontendBatch(StateId id,const VisualUpdateBatch& batch,
                const std::optional<kinematics::Transformation>& T_WB,SensorHealth visual,
                bool keyframe,std::string& error);
            bool attachFrame(const MultiFrame& frame, std::string& error) { return window_.attachFrame(frame, error); }
            bool attachImu(StateId id, const ImuMeasurementDeque& imu, std::string& error) { return window_.attachImu(id, imu, error); }
            bool addLandmark(LandmarkId id, const KeypointIdentifier& anchor, std::optional<double> rho,
                std::string& error, const ObservationOptions& options = {}) {
                return window_.addLandmark(id, anchor, rho, error, options);
            }
            bool initializeLandmark(LandmarkId id, double rho, std::string& error) { return window_.initializeLandmark(id, rho, error); }
            bool applyVisualUpdates(const VisualUpdateBatch& batch, std::string& error) { return window_.applyVisualUpdates(batch, error); }
            bool addObservation(LandmarkId id, const KeypointIdentifier& key, std::string& error,
                const ObservationOptions& options = {}) {
                return window_.addObservation(id, key, error, options);
            }
            bool removeObservation(const KeypointIdentifier& key, std::string& error) { return window_.removeObservation(key, error); }
            bool retireLandmark(LandmarkId id, std::string& error) { return window_.retireLandmark(id, error); }
            bool setLandmarkQuality(LandmarkId id, double quality, int classification) { return window_.setLandmarkQuality(id, quality, classification); }
            bool addParameter(ParameterBlock block, std::string& error) { return window_.addParameter(std::move(block), error); }
            bool configureParameter(BlockKey key, bool constant, const Eigen::VectorXd& lower,
                const Eigen::VectorXd& upper, std::string& error) {
                return window_.configureParameter(key, constant, lower, upper, error);
            }
            FactorSubmission addFactor(FactorRecord factor) { return window_.addFactor(std::move(factor)); }
            bool rejectFactor(FactorId id, const std::string& reason) { return window_.rejectFactor(id, reason); }
            bool setHealth(StateId id, SensorHealth v, SensorHealth l) { return window_.setHealth(id, v, l); }
            bool setKeyframe(StateId id, bool value) { return window_.setKeyframe(id, value); }
            // Atomic solve + marginalization + snapshot publication; on failure retain inserted measurements for retry.
            // nullopt chooses from the penultimate keyframe flag. Empty calibration freezes
            // all globals; validator adds application quality checks; reanchor executes only
            // when eviction is needed. Callbacks are synchronous and must not reenter this object.
            StepReport step(std::optional<MarginalizationPolicy> policy = std::nullopt,
                const CalibrationUpdatePolicy& calibration = {}, const SolutionValidator& validator = {},
                const MapCommitGates& mapGates = {}, const LandmarkReanchorHook& reanchor = {}, const SolveBudget& budget = {});
            /// Synchronous diagnostic on the staged graph, not an automatic calibration controller.
            CalibrationInformation calibrationInformation(bool includeHistoricalPrior = false);
            const SlidingWindowManager& window() const { return window_; }
            /// Thread-safe value publication. Null until the first successful step; returned
            /// snapshots remain valid and unchanged even after later window replacement.
            std::shared_ptr<const BackendSnapshot> snapshot() const {
                return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
            }
        private:
            std::shared_ptr<const BackendSnapshot> makeSnapshot(uint64_t revision) const;
            BackendOptions options_;
            SlidingWindowManager window_;
            BackendOptimizer optimizer_;
            uint64_t revision_ = 0;
            std::shared_ptr<const BackendSnapshot> snapshot_;
        };
    }
}
