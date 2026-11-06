#pragma once
#include <visual/pose_solver/PoseSolverTypes.hpp>

namespace cgmli { namespace pose_solver {
/// Direction-only consensus for low-parallax/infinite-depth tracks. Ray origins are
/// deliberately ignored: success establishes a rotation, never metric translation.
/// Uses Wahba SVD hypotheses and the same PoseLib LO-RANSAC/LM adapter as other solvers.
class RotationPoseSolver {
public:
    explicit RotationPoseSolver(const PoseSolverOptions& options = {}) : options_(options) {}
    /// Returns R_S1S2 in result.pose with zero translation. Original input indices
    /// are preserved. At least three rays and two nonparallel directions are needed.
    PoseSolverResult solve(const RelativePoseCorrespondences& correspondences,
        const kinematics::Transformation* initialT_S1S2 = nullptr) const;
private:
    PoseSolverOptions options_;
};
} }
