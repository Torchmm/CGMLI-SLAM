#include "VisualBatchAudit.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Cholesky>
#include <ceres/crs_matrix.h>
#include <ceres/loss_function.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace cgmli::backend::detail {
namespace {
using Sparse=Eigen::SparseMatrix<double,Eigen::RowMajor,int>;
struct Linearization {
    double cost=0;
    std::vector<double> residuals;
    ::ceres::CRSMatrix J;
    Eigen::Map<const Sparse> jacobian() const {
        return {J.num_rows,J.num_cols,int(J.values.size()),J.rows.data(),J.cols.data(),J.values.data()};
    }
};
template<class A,class B> double relative(const A& a,const B& b) {
    return (a-b).norm()/std::max(1.0,a.norm());
}
bool compare(const Linearization& a,const Linearization& b,VisualAuditReport& report) {
    if(a.J.num_rows!=b.J.num_rows || a.J.num_cols!=b.J.num_cols || a.residuals.size()!=b.residuals.size()) return false;
    const Eigen::Map<const Eigen::VectorXd> ra(a.residuals.data(),a.residuals.size()),rb(b.residuals.data(),b.residuals.size());
    const Sparse delta=a.jacobian()-b.jacobian();
    report.residual=std::max(report.residual,relative(ra,rb));
    report.jacobian=std::max(report.jacobian,delta.norm()/std::max(1.0,a.jacobian().norm()));
    return report.residual<1e-8 && report.jacobian<1e-8 &&
        std::abs(a.cost-b.cost)<1e-8*std::max(1.0,std::abs(a.cost));
}
#ifdef CGMLI_ENABLE_CUDA_ORB
void compareDirect(::ceres::Problem& problem,VisualEvaluationCache& cache,
    ::ceres::Problem::EvaluateOptions options,const std::vector<int>& retained,const std::vector<int>& eliminated,
    const std::vector<::ceres::ResidualBlockId>& visual,double huberScale,VisualAuditReport& out) {
    if(visual.empty() || visual.size()!=cache.batchFactors()) return;
    struct Block { int column,width,plus; };
    std::map<const double*,Block> offsets;
    std::vector<double> plus;
    int column=0;
    for(const auto* variable:options.parameter_blocks) {
        const int ambient=problem.ParameterBlockSize(variable),width=problem.ParameterBlockTangentSize(variable);
        offsets.emplace(variable,Block{column,width,int(plus.size())});
        const auto start=plus.size();
        plus.resize(start+ambient*width,0.0);
        const auto* manifold=problem.GetManifold(variable);
        if(manifold) {
            if(!manifold->PlusJacobian(variable,plus.data()+start)) return;
        } else for(int j=0;j<width;++j) plus[start+j*width+j]=1.0;
        column+=width;
    }
    std::vector<DirectVisualLayout> layout(visual.size());
    std::vector<double*> blocks;
    for(std::size_t i=0;i<visual.size();++i) {
        const auto* loss=problem.GetLossFunctionForResidualBlock(visual[i]);
        if(loss) {
            if(!dynamic_cast<const ::ceres::HuberLoss*>(loss) || huberScale<=0) return;
            double rho[3];
            loss->Evaluate(4*huberScale*huberScale,rho);
            if(std::abs(rho[1]-0.5)>1e-12) return;
            layout[i].huberScale=huberScale;
        }
        problem.GetParameterBlocksForResidualBlock(visual[i],&blocks);
        if(blocks.size()>5) return;
        for(std::size_t b=0;b<blocks.size();++b) {
            const auto entry=offsets.find(blocks[b]);
            if(entry==offsets.end()) continue;
            const auto& block=entry->second;
            layout[i].column[b]=block.column;
            layout[i].width[b]=block.width;
            layout[i].plus[b]=block.plus;
        }
    }
    VisualNormalSystem direct;
    if(!cache.auditDirectNormal(layout,plus,retained,eliminated,direct)) return;
    // Ceres supplies a separate reference for exactly the factors packed into the GPU batch.
    cache.setBatchMode(VisualBatchMode::Reference);
    options.residual_blocks=visual;
    Linearization reference;
    if(!problem.Evaluate(options,&reference.cost,&reference.residuals,nullptr,&reference.J) ||
        reference.J.num_cols!=column) return;
    const Sparse J=reference.jacobian();
    const Eigen::MatrixXd H=Eigen::MatrixXd(J.transpose()*J);
    const Eigen::Map<const Eigen::VectorXd> r(reference.residuals.data(),reference.residuals.size());
    const Eigen::VectorXd g=J.transpose()*r;
    using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    out.directH=relative(H,Eigen::Map<const RowMatrix>(direct.H.data(),H.rows(),H.cols()));
    out.directG=relative(g,Eigen::Map<const Eigen::VectorXd>(direct.g.data(),g.size()));
    if(direct.reducedH.empty()) return;
    const auto n=Eigen::Index(retained.size());
    const auto damp=[&](int index) { return 1e-3*std::max(H(index,index),1e-6); };
    Eigen::MatrixXd reduced(n,n);
    Eigen::VectorXd rhs(n),cross(n);
    for(Eigen::Index i=0;i<n;++i) {
        rhs[i]=g[retained[i]];
        for(Eigen::Index j=0;j<n;++j) reduced(i,j)=H(retained[i],retained[j]);
        reduced(i,i)+=damp(retained[i]);
    }
    for(int p:eliminated) {
        for(Eigen::Index i=0;i<n;++i) cross[i]=H(retained[i],p);
        const double scale=1.0/(H(p,p)+damp(p));
        reduced.noalias()-=scale*cross*cross.transpose();
        rhs.noalias()-=(scale*g[p])*cross;
    }
    const Eigen::VectorXd expected=reduced.ldlt().solve(-rhs);
    const Eigen::Map<const RowMatrix> deviceH(direct.reducedH.data(),n,n);
    const Eigen::Map<const Eigen::VectorXd> deviceG(direct.reducedG.data(),n);
    const Eigen::VectorXd step=deviceH.ldlt().solve(-deviceG);
    out.directStep=relative(expected,step);
    out.direct=expected.allFinite() && step.allFinite() &&
        out.directH<1e-8 && out.directG<1e-8 && out.directStep<1e-6;
}
#endif
}

VisualAuditReport auditVisualWindow(::ceres::Problem& problem,VisualEvaluationCache& cache,
    const std::vector<double*>& variables,const std::vector<int>& retained,const std::vector<int>& eliminated,
    const std::vector<::ceres::ResidualBlockId>& visual,double huberScale) {
    VisualAuditReport out;
    if(variables.empty() || !cache.batchFactors()) return out;
    ::ceres::Problem::EvaluateOptions options;
    options.parameter_blocks=variables;
    options.apply_loss_function=true;
    const auto evaluate=[&](VisualBatchMode mode,Linearization& linearization) {
        cache.setBatchMode(mode);
        const auto failures=cache.batchFailures;
        const auto invalid=cache.batchInvalid;
        return problem.Evaluate(options,&linearization.cost,&linearization.residuals,nullptr,&linearization.J)
            && cache.batchFailures==failures && cache.batchInvalid==invalid;
    };
    Linearization reference,host,gpu;
    if(!evaluate(VisualBatchMode::Reference,reference)) return out;
    out.host=evaluate(VisualBatchMode::Host,host) && compare(reference,host,out);
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(out.host) out.cuda=evaluate(VisualBatchMode::Cuda,gpu) && compare(reference,gpu,out);
    if(out.cuda) compareDirect(problem,cache,options,retained,eliminated,visual,huberScale,out);
#else
    (void)visual; (void)huberScale;
#endif
    cache.setBatchMode(VisualBatchMode::Reference);
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(!out.cuda) return out;
    const Sparse J=reference.jacobian();
    const Eigen::MatrixXd H=Eigen::MatrixXd(J.transpose()*J);
    const Eigen::Map<const Eigen::VectorXd> r(reference.residuals.data(),reference.residuals.size());
    const Eigen::VectorXd g=J.transpose()*r;
    VisualNormalSystem device;
    if(!cudaVisualNormalSystem(gpu.J.rows,gpu.J.cols,gpu.J.values,gpu.residuals,gpu.J.num_cols,retained,eliminated,device)) return out;
    using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    out.normalH=relative(H,Eigen::Map<const RowMatrix>(device.H.data(),H.rows(),H.cols()));
    out.normalG=relative(g,Eigen::Map<const Eigen::VectorXd>(device.g.data(),g.size()));
    out.normal=out.normalH<1e-8 && out.normalG<1e-8;
    if(device.reducedH.empty()) return out;
    const auto n=Eigen::Index(retained.size());
    Eigen::MatrixXd reducedH(n,n);
    Eigen::VectorXd reducedG(n),column(n);
    const auto damped=[&](int index) { return 1e-3*std::max(H(index,index),1e-6); };
    for(Eigen::Index i=0;i<n;++i) {
        const int row=retained[i];
        reducedG[i]=g[row];
        for(Eigen::Index j=0;j<n;++j) reducedH(i,j)=H(row,retained[j]);
        reducedH(i,i)+=damped(row);
    }
    for(int landmark:eliminated) {
        for(Eigen::Index i=0;i<n;++i) column[i]=H(retained[i],landmark);
        const double denominator=H(landmark,landmark)+damped(landmark);
        reducedH.noalias()-=(column/denominator)*column.transpose();
        reducedG.noalias()-=column*(g[landmark]/denominator);
    }
    out.reducedH=relative(reducedH,Eigen::Map<const RowMatrix>(device.reducedH.data(),n,n));
    out.reducedG=relative(reducedG,Eigen::Map<const Eigen::VectorXd>(device.reducedG.data(),n));
    out.schur=out.reducedH<1e-8 && out.reducedG<1e-8;
#endif
    return out;
}

}
