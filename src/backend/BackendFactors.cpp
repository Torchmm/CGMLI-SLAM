#include "BackendFactors.hpp"
#include "MarginalizationPrior.hpp"
#include <ceres/sized_cost_function.h>
#include <stdexcept>

namespace cgmli { namespace backend {
namespace {
class VelocityZupt final : public ::ceres::SizedCostFunction<3,9> {
public:
    explicit VelocityZupt(const Eigen::Matrix3d& S) : S_(S) {}
    bool Evaluate(double const* const* x, double* r, double** j) const override {
        // SB ordering is [v_W(3), bg(3), ba(3)]. The measurement is v_W=0;
        // whitening is applied here, before the manager's NIS gate and any robust loss.
        Eigen::Map<Eigen::Vector3d> out(r); out=S_*Eigen::Map<const Eigen::Vector3d>(x[0]);
        if (j && j[0]) {
            // Ambient and tangent sizes coincide for Euclidean SB. Bias columns are zero.
            Eigen::Map<Eigen::Matrix<double,3,9,Eigen::RowMajor>> J(j[0]);
            J.setZero(); J.leftCols<3>()=S_;
        }
        return out.allFinite();
    }
private:
    Eigen::Matrix3d S_;
};
}
FactorRecord makeParameterPrior(const ParameterBlock& ref, const Eigen::MatrixXd& S, FactorKind kind) {
    // Reuse the frozen-reference residual implementation, but submit it as a normal
    // explicit constraint. It is only absorbed into graph.prior when incident to eviction.
    if (!ref.valid() || S.cols()!=ref.tangentSize() || S.rows()<=0 || !S.allFinite()
        || kind==FactorKind::Marginalization || kind==FactorKind::Visual || kind==FactorKind::Lidar)
        throw std::invalid_argument("invalid parameter prior");
    FactorRecord f; f.kind=kind; f.keys={ref.key};
    f.cost=std::make_shared<MarginalizationPrior>(std::vector<ParameterBlock>{ref},S,Eigen::VectorXd::Zero(S.rows()));
    return f;
}
FactorRecord makeVelocityZupt(StateId id, const Eigen::Matrix3d& S, double threshold) {
    // Keep threshold configurable in squared Mahalanobis units; infinity explicitly
    // disables statistical rejection rather than selecting a hardcoded confidence level.
    if (!id.isInitialised() || !S.allFinite() || !(threshold>0)) throw std::invalid_argument("invalid velocity ZUPT");
    FactorRecord f; f.kind=FactorKind::Zupt; f.keys={BlockKey::speedBias(id)}; f.epochs={id};
    f.nisThreshold=threshold; f.cost=std::make_shared<VelocityZupt>(S); return f;
}
} }
