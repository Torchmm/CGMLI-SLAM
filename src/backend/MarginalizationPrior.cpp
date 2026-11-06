#include "MarginalizationPrior.hpp"
#include <stdexcept>

namespace cgmli { namespace backend {
MarginalizationPrior::MarginalizationPrior(const std::vector<ParameterBlock>& reference,
    const Eigen::MatrixXd& A, const Eigen::VectorXd& r0) : A_(A), r0_(r0) {
    int n=0;
    // Copy numeric references rather than retaining pointers to slots that the window
    // will erase. Shared manifold implementations are expected to be immutable.
    for (const auto& p : reference) {
        if (!p.valid()) throw std::invalid_argument("invalid marginalization reference");
        keys_.push_back(p.key); references_.push_back(p.value); manifolds_.push_back(p.manifold);
        tangentSizes_.push_back(p.tangentSize()); offsets_.push_back(n); n+=p.tangentSize();
        mutable_parameter_block_sizes()->push_back(static_cast<int32_t>(p.value.size()));
    }
    if (A.rows()<=0 || A.cols()!=n || A.rows()!=r0.size() || !A.allFinite() || !r0.allFinite())
        throw std::invalid_argument("invalid square-root marginalization prior");
    set_num_residuals(static_cast<int>(A.rows()));
}
bool MarginalizationPrior::Evaluate(double const* const* x, double* residuals, double** jacobians) const {
    // Displacement is geometric: poses use Log(T_current*T_reference^-1), not raw
    // subtraction of translation/quaternion coefficients. Euclidean blocks subtract.
    Eigen::VectorXd delta(A_.cols());
    for (size_t i=0;i<keys_.size();++i) {
        const int a=parameter_block_sizes()[i], t=tangentSizes_[i], o=offsets_[i];
        if (!Eigen::Map<const Eigen::VectorXd>(x[i],a).allFinite()) return false;
        if (manifolds_[i]) {
            if (!manifolds_[i]->Minus(x[i], references_[i].data(), delta.data()+o)) return false;
        } else delta.segment(o,t)=Eigen::Map<const Eigen::VectorXd>(x[i],a)-references_[i];
    }
    Eigen::Map<Eigen::VectorXd> r(residuals,num_residuals());
    r = r0_+A_*delta;
    if (!r.allFinite()) return false;
    if (!jacobians) return true;
    for (size_t i=0;i<keys_.size();++i) {
        if (!jacobians[i]) continue;
        const int a=parameter_block_sizes()[i], t=tangentSizes_[i], o=offsets_[i];
        Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> j(jacobians[i],num_residuals(),a);
        // For Euclidean blocks the chart derivative is identity; selecting A columns
        // gives the exact ambient derivative without finite differences.
        if (!manifolds_[i]) { j=A_.middleCols(o,t); continue; }
        // J_prior is frozen. Only differentiate the current chart's boxminus, NOT historical factors.
        // Central tangent differences also handle SE(3)'s translation/rotation coupling away from x_ref.
        Eigen::MatrixXd D(t,t);
        Eigen::VectorXd step=Eigen::VectorXd::Zero(t), yp(a),ym(a),dp(t),dm(t);
        constexpr double h=1e-7;
        for (int c=0;c<t;++c) {
            step[c]=h;
            if (!manifolds_[i]->Plus(x[i],step.data(),yp.data())) return false;
            step[c]=-h;
            if (!manifolds_[i]->Plus(x[i],step.data(),ym.data())) return false;
            step[c]=0;
            if (!manifolds_[i]->Minus(yp.data(),references_[i].data(),dp.data())
                || !manifolds_[i]->Minus(ym.data(),references_[i].data(),dm.data())) return false;
            D.col(c)=(dp-dm)/(2*h);
        }
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> lift(t,a);
        // D differentiates boxminus along current tangent increments. Ceres expects
        // ambient Jacobians, so lift back using MinusJacobian(current). Its subsequent
        // multiplication by PlusJacobian(current) recovers A_block*D, not A_block twice.
        if (!manifolds_[i]->MinusJacobian(x[i],lift.data())) return false;
        j = A_.middleCols(o,t)*D*lift;
        if (!j.allFinite()) return false;
    }
    return true;
}
bool MarginalizationPrior::diagnose(double const* const* x,std::vector<BlockDiagnostic>& blocks,
    double& frozenResidualNorm,double& totalResidualNorm) const {
    blocks.clear(); Eigen::VectorXd delta(A_.cols());
    for(size_t i=0;i<keys_.size();++i) {
        const int a=parameter_block_sizes()[i],t=tangentSizes_[i],o=offsets_[i];
        if(!Eigen::Map<const Eigen::VectorXd>(x[i],a).allFinite()) return false;
        if(manifolds_[i]) {
            if(!manifolds_[i]->Minus(x[i],references_[i].data(),delta.data()+o)) return false;
        } else delta.segment(o,t)=Eigen::Map<const Eigen::VectorXd>(x[i],a)-references_[i];
    }
    const Eigen::VectorXd residual=r0_+A_*delta;
    if(!residual.allFinite()) return false;
    frozenResidualNorm=r0_.norm(); totalResidualNorm=residual.norm();
    for(size_t i=0;i<keys_.size();++i) {
        BlockDiagnostic d; d.key=keys_[i]; const int o=offsets_[i],t=tangentSizes_[i];
        d.contributionNorm=(A_.middleCols(o,t)*delta.segment(o,t)).norm();
        d.referenceGradientNorm=(A_.middleCols(o,t).transpose()*residual).norm();
        if(t==6 && manifolds_[i]) { d.translationDelta=delta.segment<3>(o).norm(); d.rotationDelta=delta.segment<3>(o+3).norm(); }
        if(d.key.kind==BlockKind::SpeedBias && t==9) {
            d.velocityDelta=delta.segment<3>(o).norm(); d.gyroBiasDelta=delta.segment<3>(o+3).norm(); d.accelBiasDelta=delta.segment<3>(o+6).norm();
        }
        blocks.push_back(d);
    }
    return true;
}
} }
