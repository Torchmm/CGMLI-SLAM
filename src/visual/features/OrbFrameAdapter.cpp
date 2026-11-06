#include <visual/features/OrbFrameAdapter.hpp>
#include <visual/Frame.hpp>
#include <visual/MultiFrame.hpp>
#include <cmath>
#include <stdexcept>

namespace cgmli {
    namespace features {
        std::shared_ptr<OrbFeatureExtractor> configureOrb(Frame& frame, const OrbExtractorOptions& options) {
            auto orb = std::make_shared<OrbFeatureExtractor>(frame.geometry(), options);
            frame.setDetector(orb); frame.setExtractor(orb);
            return orb;
        }
        std::vector<std::shared_ptr<OrbFeatureExtractor>> configureOrb(MultiFrame& frame,
            const OrbExtractorOptions& options) {
            std::vector<std::shared_ptr<OrbFeatureExtractor>> extractors;
            extractors.reserve(frame.numFrames());
            // Build first: an invalid calibration must not leave a half-configured rig.
            for (std::size_t camera = 0; camera < frame.numFrames(); ++camera)
                extractors.push_back(std::make_shared<OrbFeatureExtractor>(frame.geometry(camera), options));
            for (std::size_t camera = 0; camera < frame.numFrames(); ++camera) {
                frame.setDetector(camera, extractors[camera]); frame.setExtractor(camera, extractors[camera]);
            }
            return extractors;
        }
        void bindOrb(Frame& frame, const std::shared_ptr<OrbFeatureExtractor>& extractor) {
            if (!extractor || extractor->camera().get() != frame.geometry().get())
                throw std::invalid_argument("ORB extractor calibration does not match this frame");
            frame.setDetector(extractor); frame.setExtractor(extractor);
        }
        void bindOrb(MultiFrame& frame, const std::vector<std::shared_ptr<OrbFeatureExtractor>>& extractors) {
            if (extractors.size() != frame.numFrames())
                throw std::invalid_argument("ORB extractor count must match the camera rig");
            for (std::size_t camera = 0; camera < extractors.size(); ++camera)
                if (!extractors[camera] || extractors[camera]->camera().get() != frame.geometry(camera).get())
                    throw std::invalid_argument("ORB extractor calibration does not match this rig camera");
            for (std::size_t camera = 0; camera < extractors.size(); ++camera) {
                frame.setDetector(camera, extractors[camera]); frame.setExtractor(camera, extractors[camera]);
            }
        }
        void updateOrbGravity(OrbFeatureExtractor& extractor, const kinematics::Transformation& T_WC,
            const Eigen::Vector3d& gravityWorld) {
            const Eigen::Vector3d gravityCamera = T_WC.C().transpose() * gravityWorld;
            extractor.setExtractionDirection(gravityCamera);
        }
        void updateOrbGravity(const MultiFrame& frame,
            const std::vector<std::shared_ptr<OrbFeatureExtractor>>& extractors,
            const kinematics::Transformation& T_WS, const Eigen::Vector3d& gravityWorld) {
            if (extractors.size() != frame.numFrames())
                throw std::invalid_argument("ORB extractor count must match the camera rig");
            std::vector<Eigen::Vector3d> directions; directions.reserve(extractors.size());
            for (std::size_t camera = 0; camera < extractors.size(); ++camera) {
                if (!extractors[camera] || extractors[camera]->camera().get() != frame.geometry(camera).get() ||
                    !frame.T_SC(camera))
                    throw std::invalid_argument("ORB gravity requires matching calibration and T_SC extrinsics");
                const Eigen::Vector3d direction = frame.T_SC(camera)->C().transpose() * T_WS.C().transpose() * gravityWorld;
                const double norm = direction.stableNorm();
                if (!direction.allFinite() || !std::isfinite(norm) || norm < 1e-12)
                    throw std::invalid_argument("Invalid world gravity or camera/rig rotation");
                directions.push_back(direction / norm);
            }
            // Validate the complete rig before changing any per-camera extraction state.
            for (std::size_t camera = 0; camera < extractors.size(); ++camera)
                extractors[camera]->setExtractionDirection(directions[camera]);
        }
    }
}
