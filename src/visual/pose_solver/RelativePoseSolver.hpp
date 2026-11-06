#pragma once
#include <visual/pose_solver/PoseSolverTypes.hpp>

namespace cgmli {
    namespace pose_solver {
        /// PoseLib generalized six-point + generic LO-RANSAC + spherical LM refinement.
        class RelativePoseSolver {
        public:
            explicit RelativePoseSolver(const PoseSolverOptions& options = {}) : options_(options) {}
            /// At least 6 matches for a noncentral rig. If BOTH sets have a single centre,
            /// PoseLib's five-point solver is used and the camera-centre baseline is unit length.
            /// Returns T_S1S2 (frame 2 -> frame 1). Metric scale degeneracy returns failure.
            /// initialT_S1S2 is only a scored hypothesis. Zero-parallax observations and
            /// points at infinity are rejected; dispatch those cases in the frontend.
            PoseSolverResult solve(const RelativePoseCorrespondences& correspondences,
                const kinematics::Transformation* initialT_S1S2 = nullptr) const;
        private:
            PoseSolverOptions options_;
        };
    }
}
