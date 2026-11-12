#pragma once

#include "VisualCudaBatch.hpp"
#include "MarginalizationPrior.hpp"
#include <factor/BodyImuFactor.hpp>
#include <cmath>

namespace cgmli::backend::detail {

template<class PackBlock>
bool packGpuPrior(const MarginalizationPrior& prior,const PackBlock& packBlock,GpuRetainedFactors& problem) {
    std::vector<GpuPriorBlock> blocks(prior.keys().size());
    for(size_t i=0;i<blocks.size();++i) {
        auto& b=blocks[i];
        if(!packBlock(prior.keys()[i],b.state)) return false;
        const auto* manifold=prior.referenceManifold(i);
        if(b.state.chart==GpuChart::Euclidean ? manifold!=nullptr :
            dynamic_cast<const LeftSE3Manifold*>(manifold)==nullptr) return false;
        b.tangentSize=prior.tangentSizes()[i];
    }
    auto& data=problem.priorData;
    GpuPriorFactor f;
    f.firstBlock=int(problem.priorBlocks.size()); f.blockCount=int(blocks.size());
    f.firstRow=problem.rows; f.rows=prior.num_residuals();
    f.columns=int(prior.squareRootInformation().cols()); f.chartOffset=problem.chartSize;
    for(size_t i=0;i<blocks.size();++i) {
        auto& b=blocks[i]; const auto& ref=prior.references()[i];
        b.reference=int(data.size()); b.chartOffset=problem.chartSize;
        data.insert(data.end(),ref.data(),ref.data()+ref.size());
        problem.chartSize+=b.tangentSize;
    }
    f.matrix=int(data.size());
    const auto& A=prior.squareRootInformation();
    for(int r=0;r<A.rows();++r) for(int c=0;c<A.cols();++c) data.push_back(A(r,c));
    f.residual=int(data.size());
    const auto& r=prior.residualAtReference();
    data.insert(data.end(),r.data(),r.data()+r.size());
    problem.rows+=f.rows;
    problem.priorBlocks.insert(problem.priorBlocks.end(),blocks.begin(),blocks.end());
    problem.priors.push_back(f);
    return true;
}

template<class PackBlock>
bool packGpuRetainedFactor(const FactorRecord& f,bool applyLoss,const PackBlock& packBlock,GpuRetainedFactors& problem) {
    if(applyLoss && f.loss) return false;
    if(const auto* prior=dynamic_cast<const MarginalizationPrior*>(f.cost.get()))
        return prior->keys()==f.keys && packGpuPrior(*prior,packBlock,problem);
    GpuImuFactor imu;
    if(f.kind!=FactorKind::Imu || f.keys.size()!=4 || !factor::exportFrozenImu(*f.cost,imu.imu)) return false;
    for(int i=0;i<4;++i) if(!packBlock(f.keys[i],imu.blocks[i]) ||
        (i%2==0 ? imu.blocks[i].chart!=GpuChart::LeftSE3 :
            imu.blocks[i].chart!=GpuChart::Euclidean || imu.blocks[i].size!=9)) return false;
    imu.firstRow=problem.rows;
    problem.imu.push_back(imu); problem.rows+=15;
    return true;
}

inline bool packGpuVisualLoss(const FactorRecord& f,bool applyLoss,double huberScale,DirectVisualLayout& row) {
    if(!applyLoss || !f.loss) return true;
    if(!dynamic_cast<const ::ceres::HuberLoss*>(f.loss.get()) || huberScale<=0) return false;
    double rho[3];
    f.loss->Evaluate(4*huberScale*huberScale,rho);
    if(std::abs(rho[1]-0.5)>1e-12) return false;
    row.huberScale=huberScale;
    return true;
}

}
