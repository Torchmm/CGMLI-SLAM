#pragma once
#include "BackendTypes.hpp"
#include <common/Measurements.hpp>

namespace cgmli {
    class MultiFrame;
    namespace cameras { class CameraBase; }
    namespace backend {
        class SlidingWindowManager;

        struct WindowFrameData
        {
            // Owned, frozen feature payload. State and observation associations are NOT copied
            // into this object; exportFrame() materializes those from the authoritative stores.
            std::shared_ptr<const MultiFrame> measurements;
            std::vector<kinematics::Transformation> T_RC;
            std::shared_ptr<const ImuMeasurementDeque> imuSamples;
            double referenceCameraTimeOffset = 0.0; // Frozen when the epoch is created, not when a late match is added.
        };

        /// Stores immutable frame payloads independently of the graph's only mutable state copy.
        /// Mutation is private to SlidingWindowManager so an epoch cannot disappear in one
        /// container while its graph variables/observations remain alive in another.
        class FrameStore {
        public:
            const std::map<StateId, EpochNode>& epochs() const { return epochs_; }
            const WindowFrameData* find(StateId id) const;
            std::shared_ptr<const MultiFrame> measurements(StateId id) const;
            bool contains(const KeypointIdentifier& key) const;
            bool cameraPose(const BackendGraph& graph, StateId id, size_t camera,
                kinematics::Transformation& T_WC, std::string& error) const;
            bool observation(const KeypointIdentifier& key, double pixelSigma,
                WindowObservation& observation, std::string& error) const;
            // Explicit feature reconstruction is required: Frame's existing copy constructor
            // does not copy feature arrays. Never use MultiFrame's implicit copy for this job.
            static std::shared_ptr<MultiFrame> copyMeasurements(const MultiFrame& source, bool keepImages);

        private:
            friend class SlidingWindowManager;
            bool insert(const EpochNode& epoch);
            bool attach(const MultiFrame& frame, bool keepImages, std::string& error);
            bool attachImu(StateId id, const ImuMeasurementDeque& samples, size_t limit, std::string& error);
            void erase(StateId id);
            std::map<StateId, EpochNode> epochs_;
            std::map<StateId, WindowFrameData> records_;
            // Preserve rig identity even during a run of IMU-only epochs that evicts every image.
            std::vector<std::shared_ptr<const cameras::CameraBase>> rigModels_;
            std::vector<kinematics::Transformation> rigBaselines_;
        };
    }
}
