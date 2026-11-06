#include <visual/pose_solver/PoseSolverTypes.hpp>
#include "PoseSolverInternal.hpp"
#include <cameras/NCameraSystem.hpp>
#include <visual/MultiFrame.hpp>

namespace cgmli {
    namespace pose_solver {
        bool makeGeneralizedRay(const kinematics::Transformation& T_SC,
            const Eigen::Vector3d& bearing_C, GeneralizedRay& ray) {
            if (!detail::validPose(T_SC)) return false;
            GeneralizedRay candidate;
            candidate.origin = T_SC.r();
            candidate.direction = bearing_C;
            if (!detail::normalizeRay(candidate)) return false;
            candidate.direction = T_SC.C() * candidate.direction;
            if (!detail::normalizeRay(candidate)) return false;
            ray = candidate;
            return true;
        }
        bool makeGeneralizedRay(const cameras::NCameraSystem& rig, std::size_t cameraIndex,
            const Eigen::Vector2d& pixel, GeneralizedRay& ray) {
            if (cameraIndex >= rig.numCameras() || !pixel.allFinite()) return false;
            const auto camera = rig.cameraGeometry(cameraIndex);
            const auto extrinsics = rig.T_SC(cameraIndex);
            Eigen::Vector3d bearing;
            if (!camera || !extrinsics || !camera->backProject(pixel, &bearing)) return false;
            return makeGeneralizedRay(*extrinsics, bearing, ray);
        }
        bool makeGeneralizedRay(const MultiFrame& frame, std::size_t cameraIndex,
            std::size_t keypointIndex, GeneralizedRay& ray) {
            if (cameraIndex >= frame.numFrames() || keypointIndex >= frame.numKeypoints(cameraIndex))
                return false;
            Eigen::Vector2d pixel;
            if (!frame.getKeypoint(cameraIndex, keypointIndex, pixel)) return false;
            return makeGeneralizedRay(frame.cameraSystem(), cameraIndex, pixel, ray);
        }
    }
}
