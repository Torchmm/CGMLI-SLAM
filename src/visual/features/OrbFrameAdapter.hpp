#pragma once
#include <visual/features/OrbFeatureExtractor.hpp>
#include <kinematics/Transformation.hpp>

namespace cgmli {
    class Frame;
    class MultiFrame;
    namespace features {
        /// Freeze camera calibration/mask before binding. Rebind after either changes.
        /// The same shared Feature2D fills Frame's detector and descriptor extractor.
        std::shared_ptr<OrbFeatureExtractor> configureOrb(Frame& frame, const OrbExtractorOptions& options = {});
        /// One independent extractor per camera; safe to schedule cameras separately.
        std::vector<std::shared_ptr<OrbFeatureExtractor>> configureOrb(MultiFrame& frame,
            const OrbExtractorOptions& options = {});
        /// Attach cached per-camera extractors to a new frame; calibration pointers must match.
        void bindOrb(Frame& frame, const std::shared_ptr<OrbFeatureExtractor>& extractor);
        void bindOrb(MultiFrame& frame, const std::vector<std::shared_ptr<OrbFeatureExtractor>>& extractors);
        /// T_WC maps camera to world. g_W is gravity in that SAME world convention.
        void updateOrbGravity(OrbFeatureExtractor& extractor, const kinematics::Transformation& T_WC,
            const Eigen::Vector3d& gravityWorld = Eigen::Vector3d(0, 0, -1));
        /// T_WS maps rig/body S to world. Uses this frame's T_SC for every camera:
        /// g_C = R_SC.transpose() * R_WS.transpose() * g_W.
        /// Call once per frame before detect()/describe(), even when reusing extractors.
        void updateOrbGravity(const MultiFrame& frame,
            const std::vector<std::shared_ptr<OrbFeatureExtractor>>& extractors,
            const kinematics::Transformation& T_WS,
            const Eigen::Vector3d& gravityWorld = Eigen::Vector3d(0, 0, -1));
    }
}
