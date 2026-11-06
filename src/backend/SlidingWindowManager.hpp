#pragma once
#include "Marginalizer.hpp"
#include "FrameStore.hpp"
#include "LandmarkStore.hpp"

namespace cgmli {
    namespace backend {
        /**
         * Geometry-specific preparation immediately before removing an epoch.
         * Arguments: outgoing epoch ID, private staging graph, failure explanation.
         * Return false to abort the entire marginalization transaction.
         *
         * The hook may move an outgoing landmark to a surviving anchor and replace its
         * observation costs. It must preserve the world point, measurement identities,
         * whitening, and residual values at the accepted solution. Replace cost objects;
         * do not mutate objects shared with the original graph/checkpoint.
         *
         * State/calibration values, graph identities, and the existing FEJ prior are
         * protected by the manager. A landmark already referenced by that prior cannot
         * be reanchored here: leave it unchanged so Schur elimination can retire it.
         * General prior-coordinate transport is not implemented by this interface.
         */
        using LandmarkReanchorHook = std::function<bool(StateId, BackendGraph&, std::string&)>;
        /// Submission receipt. Pending is accepted storage, not permission to optimize yet.
        struct FactorSubmission {
            FactorId id = 0;
            FactorLifecycle lifecycle = FactorLifecycle::Rejected;
            std::string message;
            bool accepted() const { return lifecycle == FactorLifecycle::Pending || lifecycle == FactorLifecycle::Active; }
        };

        /**
         * Owns the chronological sliding window and its factor graph.
         *
         * Typical sequence: append prediction -> add measurements -> optimize through
         * FixedLagBackend::step() -> marginalize if the insertion slot is occupied.
         * Retains windowSize epochs plus one temporary insertion/optimization slot.
         *
         * order_ determines oldest/second-newest selection; blocks are addressed by
         * immutable BlockKey values rather than recycled array indices. Calibration
         * is global and never belongs to a removable epoch. Local block ownership
         * determines which additional variables leave with the selected epoch.
         *
         * All methods and graph references require the backend's single writer thread.
         * Ceres problems are temporary views and must be destroyed before graph storage
         * is replaced. Readers on other threads use FixedLagBackend::snapshot().
         */
        class SlidingWindowManager {
        public:
            explicit SlidingWindowManager(const BackendOptions& options = BackendOptions());
            /// Insert Pose[7], SpeedBias[9], and immutable epoch ID/time metadata.
            /// Reject non-increasing IDs/times or a second unprocessed overflow slot.
            /// This only inserts a prediction; it does not solve or remove an epoch.
            bool append(const StateEstimate& prediction, std::string& error);
            /// Atomically insert state plus an owned feature payload. The caller may
            /// reuse its MultiFrame afterwards; descriptor/image buffers are frozen copies.
            bool append(const StateEstimate& prediction,const MultiFrame& frame,std::string& error);
            bool attachFrame(const MultiFrame& frame,std::string& error);
            bool attachImu(StateId id,const ImuMeasurementDeque& samples,std::string& error);
            /// Create an uninitialized track, or initialize rho immediately. IDs increase
            /// monotonically and cannot be reused after retirement. Anchor observations
            /// and both reverse indices are inserted by this operation, not by the caller.
            bool addLandmark(LandmarkId id,const KeypointIdentifier& anchor,
                std::optional<double> inverseDepth,std::string& error,const ObservationOptions& options = {});
            bool initializeLandmark(LandmarkId id,double inverseDepth,std::string& error);
            bool applyVisualUpdates(const VisualUpdateBatch& batch,std::string& error);
            bool addObservation(LandmarkId id,const KeypointIdentifier& key,std::string& error,
                const ObservationOptions& options = {});
            bool removeObservation(const KeypointIdentifier& key,std::string& error);
            /// Schur-retire a landmark, preserving its historical information on other
            /// variables. This is not an attempt to subtract bad data from an old prior.
            bool retireLandmark(LandmarkId id,std::string& error);
            bool setLandmarkQuality(LandmarkId id,double quality,int classification);
            std::optional<LandmarkEstimate> landmark(LandmarkId id) const;
            bool getLandmark(LandmarkId id,MapPoint2& point) const;
            /// Coordinates only; avoids materializing a track's observation list.
            bool landmarkPoint(LandmarkId id,Eigen::Vector3d& point) const {
                std::string error; return pointWorld(graph_,landmarks_,id,point,error);
            }
            std::shared_ptr<const MultiFrame> exportFrame(StateId id,
                const kinematics::Transformation& T_SB = kinematics::Transformation()) const;
            /// Detached writable association proposal: clone descriptors once, omit
            /// images, and materialize authoritative IDs. Never aliases stored features.
            std::shared_ptr<MultiFrame> associationFrame(StateId id,
                const kinematics::Transformation& T_SB = kinematics::Transformation()) const;
            /// Refresh only caller-owned associations. Feature/image buffers are
            /// untouched; avoids a detached descriptor copy after every commit.
            void synchronizeFrame(StateId id,MultiFrame& target) const;
            LandmarkId nextLandmarkId() const;
            const FrameStore& frames() const { return frames_; }
            const LandmarkStore& landmarks() const { return landmarks_; }
            /// Admission budgets include tentative tracks and consumed-observation tombstones.
            size_t remainingLandmarkCapacity() const {
                return landmarks_.size()<options_.maxLandmarks?options_.maxLandmarks-landmarks_.size():0;
            }
            size_t remainingObservationCapacity() const {
                const auto used=landmarks_.observations_.size()+landmarks_.absorbedKeys_.size();
                return used<options_.maxObservations?options_.maxObservations-used:0;
            }
            /// Cross-check frame/landmark/observation indices against the authoritative graph.
            bool valid(std::string& error) const;
            /// Add a new landmark/auxiliary/global block without replacing an existing key.
            /// State blocks are inserted by append(); landmark owner is its anchor epoch.
            bool addParameter(ParameterBlock block, std::string& error);
            /// Change temporary optimization freezing and optional Euclidean bounds.
            /// Empty bound vectors disable that bound. Existing values must remain valid.
            /// This is not a coordinate-change or landmark-reanchoring API.
            bool configureParameter(BlockKey key, bool constant, const Eigen::VectorXd& lower,
                const Eigen::VectorXd& upper, std::string& error);
            /// Assign an ID, validate ordered parameter dependencies, then activate or queue.
            /// A dependency already evicted/skipped is rejected instead of reopening history.
            FactorSubmission addFactor(FactorRecord factor);
            /// Remove a live measurement and retain only its bounded diagnostic event.
            bool rejectFactor(FactorId id, const std::string& reason);
            /// Update measurement health for one active epoch; retire its invalid sensor costs.
            /// Other epochs keep their own health. IMU-only epochs lose normal keyframe status.
            bool setHealth(StateId id, SensorHealth visual, SensorHealth lidar);
            /// Apply the frontend's keyframe decision; no parallax test is performed here.
            bool setKeyframe(StateId id, bool keyframe);
            /// Inspect the penultimate epoch: keyframe -> OLD, otherwise SECOND_NEW.
            /// The newest epoch is retained in either policy.
            MarginalizationPolicy choosePolicy() const;
            /// Predict whether a proposed seed range would be consumed by the next
            /// eviction. Support contains real measured/planned views, not synthetic
            /// constraints. Already absorbed rays cannot trigger a new range's removal.
            bool promotionWouldBeMarginalized(const KeypointIdentifier& anchor,
                const std::vector<KeypointIdentifier>& support) const;
            /// Prepare reanchoring, compute a replacement prior, then atomically replace the graph.
            /// Call after accepting an optimized solution; FixedLagBackend enforces that ordering.
            /// Failure leaves the live graph/window unchanged under the hook's immutability contract.
            MarginalizationReport marginalize(MarginalizationPolicy policy, const LandmarkReanchorHook& reanchor = {});
            size_t size() const { return order_.size(); }
            bool needsMarginalization() const { return size() > options_.windowSize; }
            /// Borrowed views for the backend thread only; do not retain them across mutation.
            const BackendGraph& graph() const { return graph_; }
            const std::deque<StateId>& order() const { return order_; }
            const std::map<StateId, EpochNode>& epochs() const { return frames_.epochs(); }
            const std::deque<FactorEvent>& history() const { return history_; }
            /// Copy current values into project types; an evicted/unknown ID returns nullopt.
            std::optional<StateEstimate> estimate(StateId id) const;
        private:
            friend class FixedLagBackend;
            /// Adopt already-private topology while copying only the small frame/
            /// order/history shell. Avoid cloning the original graph a second time.
            SlidingWindowManager(const SlidingWindowManager& source,BackendGraph&& graph,LandmarkStore&& landmarks,FrameStore&& frames);
            std::shared_ptr<MultiFrame> copyFrame(StateId id,const kinematics::Transformation& T_SB,bool keepImages) const;
            void recordEvent(FactorEvent event);
            /// Mutates an already private checkpoint; caller validates and commits once.
            bool applyVisualUpdatesImpl(const VisualUpdateBatch& batch,std::string& error);
            /// Retry queued factors after inserting dependencies or updating health.
            void activatePending();
            /// Release terminal cost/loss ownership after recording a lightweight event.
            void collectTerminalFactors();
            void record(const FactorRecord& f);
            bool ready(const FactorRecord& f) const;
            bool healthy(const FactorRecord& f) const;
            FactorSubmission submitFactor(FactorRecord factor,bool managedObservation = false);
            bool initializeLandmarkImpl(LandmarkId id,double rho,std::string& error);
            bool addLandmarkImpl(LandmarkId id,const KeypointIdentifier& anchor,std::optional<double> rho,
                std::string& error,const ObservationOptions& options);
            bool addObservationImpl(LandmarkId id,const KeypointIdentifier& key,std::string& error,const ObservationOptions& options);
            bool removeObservationImpl(const KeypointIdentifier& key,std::string& error);
            bool retireLandmarksImpl(const std::vector<LandmarkId>& ids,std::string& error);
            bool observationInput(const KeypointIdentifier& key,const ObservationOptions& options,
                WindowObservation& observation,std::string& error) const;
            bool attachVisualCost(const KeypointIdentifier& key,std::string& error);
            /// Metadata-only reference selection; rejects rho/prior-coupled or consumed charts.
            bool reanchorSeedImpl(LandmarkId id,const KeypointIdentifier& anchor,std::string& error);
            bool pointWorld(const BackendGraph& graph,const LandmarkStore& landmarks,LandmarkId id,
                Eigen::Vector3d& point,std::string& error) const;
            bool reanchorOutgoing(StateId outgoing,BackendGraph& graph,LandmarkStore& landmarks,
                std::vector<LandmarkId>& changed,std::string& error,std::map<LandmarkId,std::string>* reasons=nullptr);
            /// Prepare redundancy removal on private graph/store copies. Native IMU
            /// edges are replaced, never both replayed and absorbed into the prior.
            bool prepareEviction(MarginalizationReport& report,BackendGraph& graph,LandmarkStore& landmarks,
                FrameStore& frames,std::vector<FactorEvent>& discarded,
                std::map<LandmarkId,std::string>& reasons,std::string& error);
            BackendOptions options_;
            Marginalizer marginalizer_;
            BackendGraph graph_; // Owns parameter buffers, live measurements, and the current prior.
            // Chronological membership and metadata are separate from optimization buffers.
            std::deque<StateId> order_;
            FrameStore frames_;
            LandmarkStore landmarks_;
            std::deque<FactorEvent> history_;
            // High-water marks survive eviction, distinguishing future dependencies from late data
            // without an unbounded set of every historical epoch ID.
            uint64_t largestEpochId_ = 0, largestTimestampNs_ = 0;
            FactorId nextFactorId_ = 1;
            bool deferPendingActivation_ = false; ///< Private ingestion defers the global pending scan until batch end.
        };
    }
}
