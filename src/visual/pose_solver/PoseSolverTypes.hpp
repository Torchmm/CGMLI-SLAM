#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <kinematics/Transformation.hpp>
#include <PoseLib/types.h>
#include <Eigen/StdVector>

namespace cgmli {
    namespace cameras { class NCameraSystem; }
    class MultiFrame;
    namespace pose_solver {

        /// Half-ray in the sensor frame: origin + depth * direction, depth > 0.
        /// Solvers normalize directions and support the entire unit sphere (no z division).
        struct GeneralizedRay {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                Eigen::Vector3d origin = Eigen::Vector3d::Zero();
            Eigen::Vector3d direction = Eigen::Vector3d::UnitZ();
        };
        struct AbsolutePoseCorrespondence {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                Eigen::Vector3d point_W = Eigen::Vector3d::Zero();
            GeneralizedRay ray_S;
        };
        struct RelativePoseCorrespondence {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                GeneralizedRay ray_S1, ray_S2;
        };
        using AbsolutePoseCorrespondences = std::vector<AbsolutePoseCorrespondence,
            Eigen::aligned_allocator<AbsolutePoseCorrespondence>>;
        using RelativePoseCorrespondences = std::vector<RelativePoseCorrespondence,
            Eigen::aligned_allocator<RelativePoseCorrespondence>>;

        enum class SolverStatus {
            Success, InvalidInput, InsufficientCorrespondences,
            NoConsensus, DegenerateGeometry
        };
        struct PoseSolverOptions {
            /// Passed to PoseLib's generic LO-RANSAC. PROSAC requires quality-sorted input.
            /// max_reproj_error/max_epipolar_error are NOT used: the adapter scores radians.
            poselib::RansacOptions ransac;
            double angularThreshold = 0.01; ///< Radians; squared internally for MSAC.
            std::size_t minInliers = 0; ///< 0: minimal sample size. Set a SLAM acceptance count upstairs.
            int refinementIterations = 30;
            double refinementRelativeCostTolerance = 1e-10; ///< Zero disables relative-cost termination.
            int localOptimizationRounds = 2; ///< Reclassify inliers between LM fits.
            bool analyticAbsoluteJacobians = true; ///< False retains the central-difference reference.
            bool analyticRotationJacobians = true; ///< Rotation-only LO and observability use the same S^2 derivative.
            double rankTolerance = 1e-8; ///< Relative singular-value cutoff after column scaling.
            double originTolerance = 1e-9; ///< Metres: classify a single pair of camera centres.
            double minRayDepth = 1e-6; ///< Metres for metric poses; unit-baseline units for central relative pose.
            double minParallax = 1e-4; ///< Radians; finite-depth relative geometry only.

            PoseSolverOptions() {
                ransac.max_iterations = 1000;
                ransac.min_iterations = 50;
                ransac.success_prob = 0.99;
                ransac.seed = 42;
            }
        };
        struct PoseSolverTiming {
            // Scoring includes scores requested inside LO; refinement includes those
            // scores too. These inclusive times must NOT be added together.
            double generationMs=0,scoringMs=0,refinementMs=0,observabilityMs=0;
            std::size_t acceptedLoSteps=0,rejectedLoSteps=0,smallStepStops=0,relativeCostStops=0;
            std::size_t scoringPasses=0,residualEvaluations=0,jacobianEvaluations=0,loIterations=0;
        };
        struct PoseSolverResult {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                SolverStatus status = SolverStatus::InvalidInput;
            /// Absolute: T_WS. Relative: T_S1S2. p_A = C_AB * p_B + r_AB.
            /// Only valid on Success; failure results never expose an unvalidated candidate.
            kinematics::Transformation pose;
            std::vector<std::size_t> inliers; ///< Indices into the original supplied correspondence vector.
            double rmsAngularError = std::numeric_limits<double>::infinity();
            double msacScore = std::numeric_limits<double>::infinity(); ///< Sum of truncated squared radians.
            double inlierRatio = 0.0;
            std::size_t iterations = 0, refinements = 0, generatedModels = 0;
            PoseSolverTiming timing;
            bool translationScaleObservable = false;
            bool success() const { return status == SolverStatus::Success; }
        };

        /// Atomic on failure: ray is only overwritten on success.
        /// Uses T_SC.r() and T_SC.C()*bearing_C; a direction must never be translated.
        bool makeGeneralizedRay(const kinematics::Transformation& T_SC,
            const Eigen::Vector3d& bearing_C, GeneralizedRay& ray);
        bool makeGeneralizedRay(const cameras::NCameraSystem& rig, std::size_t cameraIndex,
            const Eigen::Vector2d& pixel, GeneralizedRay& ray);
        /// Back-projects the keypoint directly; does not require Frame's ray cache.
        bool makeGeneralizedRay(const MultiFrame& frame, std::size_t cameraIndex,
            std::size_t keypointIndex, GeneralizedRay& ray);

    } // namespace pose_solver
} // namespace cgmli
