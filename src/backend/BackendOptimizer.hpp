#pragma once
#include "Marginalizer.hpp"

namespace cgmli {
    namespace backend {
        class LandmarkStore;
        /// Application quality gate evaluated after numeric/NIS/calibration checks. Return false
        /// with a reason to reject the candidate. The graph is a synchronous, borrowed view;
        /// do not store it, mutate shared costs, or recursively enter the backend from the callback.
        using SolutionValidator = std::function<bool(const BackendGraph&, std::string&)>;
        /// Conditional information, not a covariance or an automatic calibration unlock decision.
        /// Eigenvalues are ascending; absent parameter directions have zero information.
        struct CalibrationInformation {
            bool success = false;
            // [theta_BR(3), p_BR(3), td_camera, theta_BL(3), p_BL(3), td_lidar].
            Eigen::Matrix<double, 14, 14> conditional = Eigen::Matrix<double, 14, 14>::Zero();
            Eigen::Matrix<double, 14, 1> eigenvalues = Eigen::Matrix<double, 14, 1>::Zero();
            bool includesHistoricalPrior = false;
            std::string message;
        };
        /// Read-only gradient balance in common CURRENT tangent coordinates.
        /// Categories: visual, IMU, LiDAR, frozen prior, other. Gradients include
        /// rho'(r^T r) for live robust costs; a prior receives no second loss.
        /// Factor counts cover the UNION of requested keys. Gradient segments
        /// follow keys/offsets/sizes and include only that block's contribution.
        struct ConstraintGradientReport {
            bool success=false;
            std::vector<BlockKey> keys;
            std::vector<int> offsets,sizes;
            std::array<Eigen::VectorXd,5> gradients;
            std::array<size_t,5> factors{};
            std::string message;
        };

        /**
         * Solves the current graph without choosing a window victim or owning sensor history.
         * A temporary Ceres Problem borrows graph buffers; shared cost/loss/manifold ownership
         * remains outside Ceres. Accepted parameter updates are left in the supplied graph.
         * Rejected factor records remain available for the window owner to archive.
         */
        class BackendOptimizer {
        public:
            explicit BackendOptimizer(const BackendOptions& options = BackendOptions());
            // Cost functions are whitened. Apply NIS gates, Ceres LM and outlier refits.
            // A failed calibration range check identifies the affected blocks for a constrained retry.
            // On any failure restore ALL parameter values and factor lifecycle states.
            // Refit after removing new outliers while budget remains. At a solve limit,
            // retain the last usable state only if the cleaned graph passes all quality
            // gates; unrefittedOutliers explicitly reports its incomplete refinement.
            // All passes share iteration and Ceres-only time budgets.
            SolveReport optimize(BackendGraph& graph, const SolutionValidator& validator = {}, const SolveBudget& budget = {}) const;
            /// Compute Hcc-Hcx*Hxx^dagger*Hxc from whitened, pre-robust Jacobians, including
            /// derivatives of frozen blocks. Explicit calibration priors are excluded; callers
            /// may optionally include the historical marginal prior and inspect its flag.
            /// Other active constraints (including state priors tagged Other) still contribute.
            CalibrationInformation calibrationInformation(BackendGraph& graph, bool includeHistoricalPrior = false) const;
            ConstraintGradientReport constraintGradients(const BackendGraph& graph,const std::vector<BlockKey>& keys) const;
        private:
            friend class FixedLagBackend;
            SolveReport optimizeImpl(BackendGraph&,const SolutionValidator&,const SolveBudget&,bool validateInput,
                const LandmarkStore* observations=nullptr,StateId currentFrame=StateId()) const;
            struct GateEntry {
                FactorRecord* factor=nullptr;
                const ::ceres::CostFunction* evaluationCost=nullptr;
                std::vector<const double*> parameters; // Valid only during one optimize() transaction.
            };
            bool statisticalGate(const std::vector<GateEntry>& entries,size_t& rejected,std::string& error,int threads,
                ResidualStatisticsArray* evaluated=nullptr,ResidualStatisticsArray* admitted=nullptr,double* statisticsMs=nullptr) const;
            BackendOptions options_;
            Marginalizer marginalizer_;
        };
    }
}
