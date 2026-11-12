#include "Marginalizer.hpp"
#include "BackendProblem.hpp"
#include "VisualCudaBatch.hpp"
#include "GpuRetainedPacking.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <system_error>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>

namespace cgmli { namespace backend {
namespace {
// Ceres has already produced robustified tangent-coordinate rows. Accumulate
// their upper-triangular outer products without duplicating a sparse Jacobian.
// Thread-local buffers and a fixed reduction order avoid shared matrix writes.
bool accumulateNormalEquations(const ::ceres::CRSMatrix& jacobian, const Eigen::Map<const Eigen::VectorXd>& residual,
    int threadCount, LinearizedSystem& result) {
    const int n=jacobian.num_cols;
    struct Accumulator { Eigen::MatrixXd H; Eigen::VectorXd b; };
    std::vector<Accumulator> partial(threadCount);
    for (auto& block : partial) {
        block.H=Eigen::MatrixXd::Zero(n,n);
        block.b=Eigen::VectorXd::Zero(n);
    }
    const auto work=[&](int slot) {
        auto& block=partial[slot];
        // Interleave rows so dense prior rows do not all land on one worker.
        for (int row=slot; row<jacobian.num_rows; row+=threadCount) {
            const int end=jacobian.rows[row+1];
            for (int j=jacobian.rows[row]; j<end; ++j) {
                const int a=jacobian.cols[j];
                const double value=jacobian.values[j];
                if (value==0.0) continue;
                block.b[a]+=value*residual[row];
                for (int k=j; k<end; ++k) {
                    const int b=jacobian.cols[k];
                    block.H(std::min(a,b),std::max(a,b))+=value*jacobian.values[k];
                }
            }
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(threadCount-1);
    try {
        for (int slot=1; slot<threadCount; ++slot) workers.emplace_back(work,slot);
    } catch (const std::system_error&) {
        for (auto& worker : workers) worker.join();
        return false;
    }
    work(0);
    for (auto& worker : workers) worker.join();
    result.H=std::move(partial[0].H);
    result.b=std::move(partial[0].b);
    for (int slot=1; slot<threadCount; ++slot) {
        result.H+=partial[slot].H;
        result.b+=partial[slot].b;
    }
    for (int column=0; column<n; ++column)
        for (int row=column+1; row<n; ++row) result.H(row,column)=result.H(column,row);
    result.timing.normalThreads=threadCount;
    return true;
}
#ifdef CGMLI_ENABLE_CUDA_ORB
bool linearizeDevice(const BackendGraph& graph,const std::vector<FactorId>& factors,bool includePrior,bool applyLoss,
    double huberScale,int n,LinearizedSystem& out,const std::shared_ptr<detail::VisualCudaWorkspace>& workspace) {
    if(n<1 || n>4000) return false;
    VisualEvaluationCache cache;
    cache.configureBatch(VisualBatchMode::Schur,workspace);
    std::map<BlockKey,detail::GpuVariable> blocks;
    std::vector<double> plus;
    for(size_t i=0;i<out.keys.size();++i) {
        const auto key=out.keys[i];
        const auto& p=graph.blocks.at(key);
        auto& block=blocks[key];
        auto& state=block.state;
        state.size=int(p.value.size()); state.width=p.tangentSize(); state.column=out.offsets[i];
        state.offset=cache.bindSchurParameter(p.value.data(),state.size);
        block.lift=int(plus.size()); plus.resize(plus.size()+state.size*state.width,0);
        if(p.manifold) {
            if(!dynamic_cast<const LeftSE3Manifold*>(p.manifold.get()) || state.size!=7 || state.width!=6 ||
                !p.manifold->PlusJacobian(p.value.data(),plus.data()+block.lift)) return false;
            state.chart=detail::GpuChart::LeftSE3;
        } else {
            if(state.size>9) return false;
            for(int j=0;j<state.width;++j) plus[block.lift+j*(state.width+1)]=1;
        }
    }
    const auto packBlock=[&](BlockKey key,detail::GpuRetainedBlock& state) {
        state=blocks.at(key).state; return true;
    };
    detail::GpuRetainedFactors retained;
    if(includePrior && graph.prior && !detail::packGpuPrior(*graph.prior,packBlock,retained)) return false;
    std::vector<detail::DirectVisualLayout> layout;
    for(auto id:factors) {
        const auto& f=graph.factors.at(id);
        if(f.kind!=FactorKind::Visual) {
            if(!detail::packGpuRetainedFactor(f,applyLoss,packBlock,retained)) return false;
            continue;
        }
        detail::DirectVisualLayout row;
        if(f.keys.size()>5 || !detail::packGpuVisualLoss(f,applyLoss,huberScale,row)) return false;
        std::vector<const double*> parameters;
        for(size_t j=0;j<f.keys.size();++j) {
            const auto key=f.keys[j];
            const auto& block=blocks.at(key);
            row.column[j]=block.state.column; row.width[j]=block.state.width; row.plus[j]=block.lift;
            parameters.push_back(graph.blocks.at(key).value.data());
        }
        if(!cacheWindowVisualCost(*f.cost,parameters,cache) || cache.batchFactors()!=layout.size()+1) return false;
        layout.push_back(row);
    }
    auto* device=cache.linearizeSchur(layout,plus,n,0,true);
    if(!device || !device->configureRetained(retained) || !device->linearizeRetained()) return false;
    using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    RowMatrix upper(n,n);
    out.b.resize(n);
    if(!device->readNormal(upper.data(),out.b.data(),out.robustCost,out.residualSquaredNorm)) return false;
    out.H=upper.selfadjointView<Eigen::Upper>();
    return out.H.allFinite() && out.b.allFinite();
}
#endif
}
Marginalizer::Marginalizer(const BackendOptions& o) : options_(o) {
    options_.validate();
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(options_.visualBatchMode==VisualBatchMode::Schur) {
        visualCuda_=std::make_shared<detail::VisualCudaWorkspace>();
        marginalCuda_=std::make_shared<detail::MarginalCudaWorkspace>();
    }
#endif
}
bool Marginalizer::linearize(BackendGraph& graph, const std::vector<FactorId>& factors,
    bool includePrior, bool applyLoss, LinearizedSystem& out, std::string& error) const {
    out=LinearizedSystem();
    using Clock=std::chrono::steady_clock;
    const auto setupStarted=Clock::now();
    if (!graph.valid(&error)) return false;
    std::set<BlockKey> keys;
    // Use deterministic key ordering so Jacobian columns, reference snapshots, and
    // the eventual prior share exactly the same block-to-column correspondence.
    std::set<FactorId> unique;
    for (auto id:factors) {
        auto f=graph.factors.find(id);
        if (!unique.insert(id).second || f==graph.factors.end() || !f->second.participates()) {
            error="linearization requested duplicate/missing/inactive factor"; return false;
        }
        keys.insert(f->second.keys.begin(),f->second.keys.end());
    }
    if (includePrior && graph.prior) keys.insert(graph.prior->keys().begin(),graph.prior->keys().end());
    if (keys.empty()) { out.H.resize(0,0); out.b.resize(0); return true; }
    int n=0;
    for (auto k:keys)
    {
        const auto& p=graph.blocks.at(k);
        out.keys.push_back(k); out.offsets.push_back(n); out.sizes.push_back(p.tangentSize()); n+=p.tangentSize();
    }
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(options_.visualBatchMode==VisualBatchMode::Schur) {
        const auto gpuStarted=Clock::now();
        if(linearizeDevice(graph,factors,includePrior,applyLoss,options_.visualHuberScale,n,out,visualCuda_)) {
            out.timing.setupMs=std::chrono::duration<double,std::milli>(gpuStarted-setupStarted).count();
            out.timing.normalMs=std::chrono::duration<double,std::milli>(Clock::now()-gpuStarted).count();
            out.timing.gpuFactors=true; out.timing.gpuNormal=true;
            return true;
        }
    }
#endif
    ::ceres::Problem problem(detail::problemOptions());
    ::ceres::Problem::EvaluateOptions eval;
    eval.apply_loss_function=applyLoss;
    eval.num_threads=options_.numThreads;
    for(auto k:keys) {
        auto& p=graph.blocks.at(k);
        // Marginalization preserves the derivative of temporarily frozen blocks.
        detail::addBlock(problem,p,false);
        eval.parameter_blocks.push_back(p.value.data());
    }

    for (auto id:factors) detail::addFactor(problem,graph,graph.factors.at(id));
    if (includePrior) detail::addPrior(problem,graph);
    ::ceres::CRSMatrix sparse;

    // Ceres returns tangent Jacobians and consistent robustification when applyLoss is
    // enabled. Do not multiply by PlusJacobian or apply the loss a second time.
    std::vector<double> residuals;
    double cost=0;

    const auto evaluateStarted=Clock::now();
    out.timing.setupMs=std::chrono::duration<double,std::milli>(evaluateStarted-setupStarted).count();
    if (!problem.Evaluate(eval,&cost,&residuals,nullptr,&sparse) || !std::isfinite(cost) || sparse.num_cols!=n) {
        error="Ceres tangent linearization failed"; return false;
    }
    const auto normalStarted=Clock::now();
    out.timing.evaluateMs=std::chrono::duration<double,std::milli>(normalStarted-evaluateStarted).count();
    const Eigen::Map<const Eigen::VectorXd> r(residuals.data(),static_cast<int>(residuals.size()));
    out.robustCost=cost; out.residualSquaredNorm=r.squaredNorm();
    bool gpuNormal=false;
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(options_.visualBatchMode==VisualBatchMode::Schur) {
        std::vector<int> retained(n);
        std::iota(retained.begin(),retained.end(),0);
        detail::VisualNormalSystem normal;
        if(detail::cudaVisualNormalSystem(sparse.rows,sparse.cols,sparse.values,residuals,n,retained,{},normal)) {
            using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
            out.H=Eigen::Map<const RowMatrix>(normal.H.data(),n,n);
            out.b=Eigen::Map<const Eigen::VectorXd>(normal.g.data(),n);
            gpuNormal=out.H.allFinite() && out.b.allFinite();
        }
    }
#endif
    // Keep thread creation away from small priors and cap temporary dense storage.
    const int normalThreads=std::min({4,options_.numThreads,sparse.num_rows/128});
    bool useParallel=options_.parallelNormalEquations && normalThreads>1 && n<=768 && sparse.values.size()>=2048;
    if (useParallel) {
        size_t products=0;
        for (int row=0; row<sparse.num_rows; ++row) {
            const size_t width=sparse.rows[row+1]-sparse.rows[row];
            products+=width*(width+1)/2;
        }
        // Avoid clearing and merging several large dense buffers for a sparse,
        // lightly connected system where the existing Eigen product is cheaper.
        useParallel=products>=size_t(n)*n*normalThreads/2;
    }
    if (!gpuNormal && (!useParallel || !accumulateNormalEquations(sparse,r,normalThreads,out))) {
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(sparse.values.size());
        for (int row=0;row<sparse.num_rows;++row)
            for (int k=sparse.rows[row];k<sparse.rows[row+1];++k) entries.emplace_back(row,sparse.cols[k],sparse.values[k]);
        Eigen::SparseMatrix<double> J(sparse.num_rows,n);
        J.setFromTriplets(entries.begin(),entries.end());
        out.H=Eigen::MatrixXd(J.transpose()*J); out.b=J.transpose()*r;
    }
    out.timing.normalMs=std::chrono::duration<double,std::milli>(Clock::now()-normalStarted).count();
    out.timing.gpuNormal=gpuNormal;
    if (!out.H.allFinite() || !out.b.allFinite()) { error="nonfinite normal equations"; return false; }
    return true;
}
bool Marginalizer::psdSolve(const Eigen::MatrixXd& H, const Eigen::MatrixXd& rhs,
    Eigen::MatrixXd& x, std::string& error) const {
    // H can be rank deficient because global gauge or weak geometry is unobservable.
    // Solve in the supported eigenspace instead of adding persistent information to
    // those directions. LM damping is a solve-time device, not part of the prior.
    if (H.rows()!=H.cols() || H.rows()!=rhs.rows() || !H.allFinite() || !rhs.allFinite()) {
        error="invalid PSD system"; return false;
    }
    if (!H.rows()) { x.resize(0,rhs.cols()); return true; }
    // Exact diagonal fast path (including a single inverse range). This uses
    // the SAME eigenvalue cutoff as the general solve and adds no damping.
    if(H.isDiagonal(0.0)) {
        const double cutoff=std::max(options_.absoluteEigenThreshold,
            options_.relativeEigenThreshold*std::max(0.0,H.diagonal().maxCoeff()));
        if(H.diagonal().minCoeff() < -10.0*cutoff) { error="information matrix is indefinite"; return false; }
        x=rhs;
        for(int i=0;i<H.rows();++i) {
            if(H(i,i)>cutoff) x.row(i)/=H(i,i);
            else x.row(i).setZero();
        }
        return x.allFinite();
    }
    const Eigen::MatrixXd symmetric=0.5*(H+H.transpose());
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(options_.visualBatchMode==VisualBatchMode::Schur) {
        x.resize(rhs.rows(),rhs.cols());
        if(marginalCuda_->psdSolve(symmetric.data(),rhs.data(),int(rhs.rows()),int(rhs.cols()),
            options_.absoluteEigenThreshold,options_.relativeEigenThreshold,x.data()) && x.allFinite()) return true;
    }
#endif
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(symmetric);
    if (eig.info()!=Eigen::Success) { error="PSD eigensolve failed"; return false; }
    const double cutoff=std::max(options_.absoluteEigenThreshold,
        options_.relativeEigenThreshold*std::max(0.0,eig.eigenvalues().maxCoeff()));
    if (eig.eigenvalues().minCoeff() < -10.0*cutoff) { error="information matrix is indefinite"; return false; }
    Eigen::MatrixXd projected=eig.eigenvectors().transpose()*rhs;
    // H = U Lambda U^T; apply U Lambda^dagger U^T to RHS without forming H^-1.
    for (int i=0;i<H.rows();++i) {
        if (eig.eigenvalues()[i]>cutoff) projected.row(i)/=eig.eigenvalues()[i];
        else projected.row(i).setZero();
    }
    x=eig.eigenvectors()*projected;
    return x.allFinite();
}
MarginalizationResult Marginalizer::compute(BackendGraph& graph, const std::set<BlockKey>& eliminate) const {
    // Global calibration remains a separator variable even when its sensor is frozen.
    MarginalizationResult out;
    if(!graph.valid(&out.message)) return out;
    for (auto k:eliminate) {
        if (k.isCalibration()) { out.message="global calibration cannot be marginalized with a frame"; return out; }
        if (!graph.blocks.count(k)) { out.message="missing elimination block"; return out; }
    }
    if (eliminate.empty()) { out.message="empty elimination set"; return out; }
    for (const auto& item:graph.factors) {
        // Only incident measurements are consumed. Nonincident costs remain nonlinear
        // and must not also be included in the new prior, which would double-count them.
        if (!item.second.participates()) continue;
        for (auto k:item.second.keys) if (eliminate.count(k)) { out.consumed.push_back(item.first); break; }
    }
    bool priorTouches=false;
    if(graph.prior) for(auto key:graph.prior->keys()) priorTouches=priorTouches || eliminate.count(key)!=0;
    if(out.consumed.empty() && !priorTouches) {
        // The native SECOND_NEW path has replaced its two IMU edges and dropped
        // redundant image measurements. Rebuilding an unrelated frozen prior
        // wastes dense eigensolves and needlessly moves its reference chart.
        out.prior=graph.prior; out.priorReused=true;
        if(graph.prior) {
            out.retainedDimension=static_cast<int>(graph.prior->squareRootInformation().cols());
            out.priorRank=graph.prior->num_residuals();
        }
        out.success=true; return out;
    }
    LinearizedSystem lin;
    const auto linearizeStarted=std::chrono::steady_clock::now();
    // The previous prior is incorporated once, regardless of which victim is selected.
    if (!linearize(graph,out.consumed,true,true,lin,out.message)) return out;
    out.linearization=lin.timing;
    auto& formation=out.formation; formation.evaluated=true;
    formation.robustCost=lin.robustCost; formation.linearizedSquaredNorm=lin.residualSquaredNorm;
    for(auto id:out.consumed) {
        const auto kind=graph.factors.at(id).kind;
        if(kind==FactorKind::Visual) ++formation.visualFactors;
        else if(kind==FactorKind::Imu) ++formation.imuFactors; else ++formation.otherFactors;
    }
    if(graph.prior) {
        formation.incomingReferenceNorm=graph.prior->residualAtReference().norm();
        std::vector<const double*> x; for(auto key:graph.prior->keys()) x.push_back(graph.blocks.at(key).value.data());
        Eigen::VectorXd residual(graph.prior->num_residuals());
        if(!graph.prior->Evaluate(x.data(),residual.data(),nullptr)) { out.message="invalid incoming prior during formation audit"; return out; }
        formation.incomingCurrentNorm=residual.norm();
    }
    const auto schurStarted=std::chrono::steady_clock::now();
    out.linearizeMs=std::chrono::duration<double,std::milli>(schurStarted-linearizeStarted).count();
    std::vector<int> m,r;
    // Partition participating tangent coordinates into marginalized m and retained r.
    // Retained references are value copies, independent of buffers that will be erased.
    std::vector<ParameterBlock> refs;
    for (size_t i=0;i<lin.keys.size();++i) {
        auto& indices=eliminate.count(lin.keys[i])?m:r;
        for (int j=0;j<lin.sizes[i];++j) indices.push_back(lin.offsets[i]+j);
        if (!eliminate.count(lin.keys[i])) refs.push_back(graph.blocks.at(lin.keys[i]));
    }
    const int nm=static_cast<int>(m.size()), nr=static_cast<int>(r.size());
    out.eliminatedDimension=nm; out.retainedDimension=nr;
    if (!nr) { out.success=true; return out; }
    Eigen::MatrixXd Hmm(nm,nm),Hmr(nm,nr),Hrr(nr,nr);
    Eigen::VectorXd bm(nm),br(nr);
    for (int i=0;i<nm;++i) {
        bm[i]=lin.b[m[i]];
        for (int j=0;j<nm;++j) Hmm(i,j)=lin.H(m[i],m[j]);
        for (int j=0;j<nr;++j) Hmr(i,j)=lin.H(m[i],r[j]);
    }
    for (int i=0;i<nr;++i) {
        br[i]=lin.b[r[i]];
        for (int j=0;j<nr;++j) Hrr(i,j)=lin.H(r[i],r[j]);
    }
    const auto subtract=[&](const Eigen::MatrixXd& coupling,const Eigen::MatrixXd& solution,
        Eigen::MatrixXd& H,Eigen::VectorXd& gradient) {
#ifdef CGMLI_ENABLE_CUDA_ORB
        if(options_.visualBatchMode==VisualBatchMode::Schur && coupling.rows()) {
            Eigen::MatrixXd deviceH=H;
            Eigen::VectorXd deviceGradient=gradient;
            if(marginalCuda_->subtract(coupling.data(),solution.data(),int(coupling.rows()),int(coupling.cols()),
                deviceH.data(),deviceGradient.data()) && deviceH.allFinite() && deviceGradient.allFinite()) {
                H.swap(deviceH); gradient.swap(deviceGradient);
                out.linearization.gpuSchur=true;
                return;
            }
        }
#endif
        H.noalias()-=coupling.transpose()*solution.leftCols(coupling.cols());
        gradient.noalias()-=coupling.transpose()*solution.rightCols(1);
    };
    if(options_.landmarkFirstMarginalization) {
        std::vector<int> ranges,states;
        for(size_t k=0;k<lin.keys.size();++k) if(eliminate.count(lin.keys[k])) {
            for(int c=0;c<lin.sizes[k];++c) {
                const int original=lin.offsets[k]+c;
                const int index=static_cast<int>(std::find(m.begin(),m.end(),original)-m.begin());
                if(lin.keys[k].kind==BlockKind::Landmark && lin.sizes[k]==1) ranges.push_back(index); else states.push_back(index);
            }
        }
        const int nl=static_cast<int>(ranges.size()),ns=static_cast<int>(states.size());
        Eigen::MatrixXd Hll(nl,nl);
        for(int i=0;i<nl;++i) for(int j=0;j<nl;++j) Hll(i,j)=Hmm(ranges[i],ranges[j]);
        if(nl && Hll.isDiagonal(0.0)) {
            // OLD now preserves visual evidence. Eliminate independent scalar
            // ranges first, then only the small pose/SB block. Coupled historical
            // range priors keep the general joint PSD path below.
            Eigen::MatrixXd coupling(nl,ns+nr),rhsL(nl,ns+nr+1),solvedL,reduced(ns+nr,ns+nr);
            Eigen::VectorXd gradient(ns+nr);
            for(int i=0;i<nl;++i) {
                for(int j=0;j<ns;++j) coupling(i,j)=Hmm(ranges[i],states[j]);
                coupling.row(i).tail(nr)=Hmr.row(ranges[i]); rhsL(i,ns+nr)=bm[ranges[i]];
            }
            rhsL.leftCols(ns+nr)=coupling;
            if(!psdSolve(Hll,rhsL,solvedL,out.message)) return out;
            for(int i=0;i<nl;++i) formation.eliminatedReduction+=0.5*bm[ranges[i]]*solvedL(i,ns+nr);
            for(int i=0;i<ns;++i) {
                gradient[i]=bm[states[i]];
                for(int j=0;j<ns;++j) reduced(i,j)=Hmm(states[i],states[j]);
                reduced.row(i).tail(nr)=Hmr.row(states[i]);
                reduced.col(i).tail(nr)=Hmr.row(states[i]).transpose();
            }
            reduced.bottomRightCorner(nr,nr)=Hrr; gradient.tail(nr)=br;
            subtract(coupling,solvedL,reduced,gradient);
            Hmm=reduced.topLeftCorner(ns,ns); Hmr=reduced.topRightCorner(ns,nr); Hrr=reduced.bottomRightCorner(nr,nr);
            bm=gradient.head(ns); br=gradient.tail(nr); out.scalarSchurLandmarks=nl;
        }
    }
    Eigen::MatrixXd rhs(Hmm.rows(),nr+1),solved;
    // Jointly solve Hmm * [X,y] = [Hmr,bm], then form the Schur system:
    //   H_prior = Hrr - Hrm*X
    //   b_prior = br  - Hrm*y.
    // Eliminating an interior epoch retains the information bridge between its neighbors.
    rhs.leftCols(nr)=Hmr; rhs.rightCols(1)=bm;
    if (!psdSolve(Hmm,rhs,solved,out.message)) return out;
    formation.eliminatedReduction+=0.5*bm.dot(solved.col(nr));
    // Reuse the partition buffers for the dense separator instead of allocating
    // another nr-by-nr product/subtraction result at the peak working-set size.
    subtract(Hmr,solved,Hrr,br);
    Eigen::MatrixXd H=std::move(Hrr);
    const Eigen::VectorXd b=std::move(br);
    H=(0.5*(H+H.transpose())).eval();
    const auto priorStarted=std::chrono::steady_clock::now();
    out.schurMs=std::chrono::duration<double,std::milli>(priorStarted-schurStarted).count();
    Eigen::VectorXd eigenvalues(nr);
    Eigen::MatrixXd eigenvectors(nr,nr);
    bool gpuEigen=false;
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(options_.visualBatchMode==VisualBatchMode::Schur)
        gpuEigen=marginalCuda_->eigen(H.data(),nr,eigenvalues.data(),eigenvectors.data()) &&
            eigenvalues.allFinite() && eigenvectors.allFinite();
#endif
    out.linearization.gpuEigen=gpuEigen;
    if(!gpuEigen) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
        if(eig.info()!=Eigen::Success) { out.message="Schur eigensolve failed"; return out; }
        eigenvalues=eig.eigenvalues(); eigenvectors=eig.eigenvectors();
    }
    if(!b.allFinite()) { out.message="Schur eigensolve failed"; return out; }
    const double cutoff=std::max(options_.absoluteEigenThreshold,
        options_.relativeEigenThreshold*std::max(0.0,eigenvalues.maxCoeff()));
    formation.eigenCutoff=cutoff; formation.retainedGradientNorm=b.norm();
    if (eigenvalues.minCoeff() < -10.0*cutoff) { out.message="Schur prior is indefinite"; return out; }
    double droppedGradientSquared=0;
    for (int i=0;i<nr;++i) {
        if (eigenvalues[i]>cutoff) ++out.priorRank;
        else { const double v=eigenvectors.col(i).dot(b); droppedGradientSquared+=v*v; }
    }
    formation.droppedGradientNorm=std::sqrt(droppedGradientSquared);
    // Rank zero means no observable retained information, not an identity/diagonal prior.
    if (out.priorRank) {
        Eigen::MatrixXd A(out.priorRank,nr);
        Eigen::VectorXd r0(out.priorRank);
        int row=0;
        // For positive eigenpairs (lambda,u), choose A_row=sqrt(lambda)*u^T and
        // r0_row=(u^T*b)/sqrt(lambda). Then A^T*A=H and A^T*r0=b in the supported
        // subspace. The omitted constant objective offset does not affect optimization.
        for (int i=0;i<nr;++i) if (eigenvalues[i]>cutoff) {
            const double root=std::sqrt(eigenvalues[i]);
            A.row(row)=root*eigenvectors.col(i).transpose();
            r0[row]=eigenvectors.col(i).dot(b)/root; ++row;
        }
        out.prior=std::make_shared<MarginalizationPrior>(refs,A,r0);
        formation.newReferenceNorm=r0.norm();
    }
    // Ceres robust cost is not generally 0.5*||returned residual||^2.
    // This offset belongs to the returned LOCAL quadratic only; the square-root
    // prior reproduces H/b and intentionally omits its constant term.
    formation.quadraticConstantOffset=0.5*lin.residualSquaredNorm-formation.eliminatedReduction
        -0.5*formation.newReferenceNorm*formation.newReferenceNorm;
    out.priorMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-priorStarted).count();
    out.success=true; return out;
}
} }
