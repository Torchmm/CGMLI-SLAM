#pragma once
#include "MarginalizationPrior.hpp"

namespace cgmli { namespace backend {
/// Tangent-coordinate normal equations for 0.5 * ||r + J*delta||^2:
/// H = J^T J, b = J^T r, hence the Gauss-Newton step solves H*delta = -b.
/// offsets[i]/sizes[i] locate keys[i] within both H and b, not within ambient buffers.
struct LinearizedSystem {
    std::vector<BlockKey> keys;
    std::vector<int> offsets, sizes;
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    double robustCost=0,residualSquaredNorm=0;
    LinearizationTiming timing;
};
/// Proposed replacement prior and the exact active factor IDs it has consumed.
/// The window owner applies this result; computing it does not commit any graph changes.
struct MarginalizationResult {
    bool success = false;
    std::shared_ptr<MarginalizationPrior> prior;
    std::vector<FactorId> consumed;
    int eliminatedDimension = 0, retainedDimension = 0, priorRank = 0;
    bool priorReused=false;
    size_t scalarSchurLandmarks=0;
    PriorFormationStatistics formation;
    std::string message;
    double linearizeMs=0,schurMs=0,priorMs=0;
    LinearizationTiming linearization;
};

/**
 * Stateless numerical service used by both window eviction policies.
 * The caller supplies a set of variables to eliminate; the mathematics does not
 * depend on whether they belong to the oldest or penultimate epoch.
 * Unaffected priors are reused without evaluation/relinearization. Other cases
 * keep J sparse and perform rank-aware Schur/eigen operations on the separator.
 */
class Marginalizer {
public:
    explicit Marginalizer(const BackendOptions& options = BackendOptions());
    // Consume ONLY factors incident to eliminated blocks, and the previous prior exactly once.
    // Other measurements stay nonlinear in the graph. Does not mutate graph topology or values.
    MarginalizationResult compute(BackendGraph& graph, const std::set<BlockKey>& eliminate) const;
    // Includes Jacobians of temporarily frozen blocks, so online calibration may be unfrozen later.
    // includePrior selects the separate historical prior; applyLoss asks Ceres to apply
    // its robust residual/Jacobian transformation after sensor whitening. Returned columns
    // already include manifold PlusJacobian. The non-const graph argument supplies Ceres
    // mutable pointer types; this Evaluate path does not intentionally modify its values.
    bool linearize(BackendGraph& graph, const std::vector<FactorId>& factors,
        bool includePrior, bool applyLoss, LinearizedSystem& result, std::string& error) const;
    // Rank-aware PSD solve; never adds diagonal damping to the historical prior/gauge nullspace.
    // Solves all RHS columns using the same eigensystem. Null directions use zero
    // reciprocal eigenvalue; significantly negative information is an error.
    bool psdSolve(const Eigen::MatrixXd& H, const Eigen::MatrixXd& rhs,
        Eigen::MatrixXd& solution, std::string& error) const;
private:
    BackendOptions options_;
};
} }
