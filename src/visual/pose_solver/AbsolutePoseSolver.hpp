#pragma once
#include <visual/pose_solver/PoseSolverTypes.hpp>

namespace cgmli {
    namespace pose_solver {
        /// PoseLib GP3P + generic LO-RANSAC + spherical multi-inlier LM refinement.
        class AbsolutePoseSolver {
        public:
            explicit AbsolutePoseSolver(const PoseSolverOptions& options = {}) : options_(options) {}
            /// At least 3 non-collinear world points. Planar points are supported by GP3P.
            /// initialT_WS is a scored hypothesis, NOT an IMU gate or optimization constraint.
            /// Invalid/non-finite observations reject the request; no hidden data reordering.
            PoseSolverResult solve(const AbsolutePoseCorrespondences& correspondences,
                const kinematics::Transformation* initialT_WS = nullptr) const;
        private:
            PoseSolverOptions options_;
        };
    }
}
