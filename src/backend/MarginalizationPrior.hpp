#pragma once
#include "BackendTypes.hpp"

namespace cgmli { namespace backend {

// Immutable square-root prior. References are snapshots, never pointers into the sliding window.
// A_new is only constructed when another marginalization consumes this prior.
// A acts on concatenated tangent coordinates; CostFunction still exposes ambient
// parameter sizes to Ceres. Keeping this distinction is essential for quaternion poses.
class MarginalizationPrior final : public ::ceres::CostFunction {
public:
    struct BlockDiagnostic {
        BlockKey key;
        double contributionNorm=0,referenceGradientNorm=0;
        double translationDelta=0,rotationDelta=0,velocityDelta=0,gyroBiasDelta=0,accelBiasDelta=0;
    };
    bool diagnose(double const* const* parameters,std::vector<BlockDiagnostic>& blocks,
        double& frozenResidualNorm,double& totalResidualNorm) const;
    /// Capture block references in A-column order. Rows are supported information
    /// directions; a rank-zero result is represented by no prior, not this constructor.
    MarginalizationPrior(const std::vector<ParameterBlock>& reference,
        const Eigen::MatrixXd& A, const Eigen::VectorXd& r0);
    /// Evaluate current displacement from frozen references. Historical measurements
    /// are not reevaluated; only boxminus and its current-chart derivative can change.
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override;
    const std::vector<BlockKey>& keys() const { return keys_; }
    const std::vector<int>& tangentSizes() const { return tangentSizes_; }
    const std::vector<Eigen::VectorXd>& references() const { return references_; }
    const ::ceres::Manifold* referenceManifold(size_t index) const { return manifolds_.at(index).get(); }
    const Eigen::MatrixXd& squareRootInformation() const { return A_; }
    const Eigen::VectorXd& residualAtReference() const { return r0_; }
private:
    // offsets_ indexes tangent columns of A_, while references_ stores ambient values.
    std::vector<BlockKey> keys_;
    std::vector<int> tangentSizes_, offsets_;
    std::vector<Eigen::VectorXd> references_;
    std::vector<std::shared_ptr<::ceres::Manifold>> manifolds_;
    Eigen::MatrixXd A_; // A^T*A is the supported Schur information matrix.
    Eigen::VectorXd r0_; // A^T*r0 is its normal-equation gradient at the reference.
};
} }
