#include "BackendOptimizer.hpp"
#include "BackendProblem.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <glog/logging.h>
#include <thread>

namespace cgmli {
    namespace backend {
        BackendOptimizer::BackendOptimizer(const BackendOptions& o) : options_(o), marginalizer_(o) { options_.validate(); }
        ConstraintGradientReport BackendOptimizer::constraintGradients(const BackendGraph& graph,const std::vector<BlockKey>& keys) const {
            ConstraintGradientReport out; int columns=0;
            if(!graph.valid(&out.message)) return out;
            std::map<BlockKey,size_t> indices;
            for(auto key:keys) {
                const auto block=graph.blocks.find(key);
                if(block==graph.blocks.end() || !indices.emplace(key,out.keys.size()).second) {
                    out.message="gradient audit requires distinct existing blocks"; return out;
                }
                out.keys.push_back(key); out.offsets.push_back(columns); out.sizes.push_back(block->second.tangentSize());
                columns+=block->second.tangentSize();
            }
            for(auto& gradient:out.gradients) gradient=Eigen::VectorXd::Zero(columns);
            const auto accumulate=[&](const ::ceres::CostFunction& cost,const std::vector<BlockKey>& costKeys,
                const ::ceres::LossFunction* loss,size_t category) {
                bool incident=false; for(auto key:costKeys) incident=incident || indices.count(key)!=0;
                if(!incident) return true;
                const int rows=cost.num_residuals(); std::vector<const double*> x;
                std::vector<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> J(costKeys.size());
                std::vector<double*> requests(costKeys.size(),nullptr);
                for(size_t i=0;i<costKeys.size();++i) {
                    const auto& block=graph.blocks.at(costKeys[i]); x.push_back(block.value.data());
                    if(indices.count(costKeys[i])) { J[i].resize(rows,block.value.size()); requests[i]=J[i].data(); }
                }
                Eigen::VectorXd r(rows);
                if(!cost.Evaluate(x.data(),r.data(),requests.data()) || !r.allFinite()) return false;
                double rho[3]{r.squaredNorm(),1,0}; if(loss) loss->Evaluate(rho[0],rho);
                if(!std::isfinite(rho[1]) || rho[1]<0) return false;
                for(size_t i=0;i<costKeys.size();++i) if(requests[i]) {
                    if(!J[i].allFinite()) return false;
                    const auto& block=graph.blocks.at(costKeys[i]); const auto index=indices.at(costKeys[i]);
                    Eigen::VectorXd g=J[i].transpose()*r;
                    if(block.manifold) {
                        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> lift(block.value.size(),block.tangentSize());
                        if(!block.manifold->PlusJacobian(block.value.data(),lift.data())) return false;
                        g=(lift.transpose()*g).eval();
                    }
                    out.gradients[category].segment(out.offsets[index],out.sizes[index])+=rho[1]*g;
                }
                ++out.factors[category]; return true;
            };
            for(const auto& item:graph.factors) if(item.second.participates()) {
                const auto& f=item.second;
                const size_t category=f.kind==FactorKind::Visual?0:f.kind==FactorKind::Imu?1:f.kind==FactorKind::Lidar?2:4;
                if(!accumulate(*f.cost,f.keys,f.loss.get(),category)) { out.message="live gradient audit failed"; return out; }
            }
            if(graph.prior && !accumulate(*graph.prior,graph.prior->keys(),nullptr,3)) {
                out.message="prior gradient audit failed"; return out;
            }
            for(const auto& gradient:out.gradients) if(!gradient.allFinite()) { out.message="nonfinite audit gradient"; return out; }
            out.success=true; return out;
        }
        bool BackendOptimizer::statisticalGate(const std::vector<GateEntry>& entries,size_t& rejected,std::string& error,int threads,
            ResidualStatisticsArray* evaluated,ResidualStatisticsArray* admitted,double* statisticsMs) const {
            // Evaluate the sensor's whitened residual directly, before applying Huber/Cauchy.
            // Otherwise robust attenuation could hide a statistically implausible measurement.
            // Evaluation failure/nonfinite residual aborts the solve rather than inventing a cost.
            rejected = 0;
            int residualSize=0;
            for(const auto& entry:entries) residualSize=std::max(residualSize,entry.factor->cost->num_residuals());
            std::vector<uint8_t> decisions(entries.size(),0);
            std::vector<double> energies(entries.size(),0);
            std::atomic<size_t> next{0};
            const size_t count=entries.size()<128?1:std::min<size_t>(size_t(threads),(entries.size()+31)/32);
            std::vector<Eigen::VectorXd> buffers; buffers.reserve(count);
            for(size_t i=0;i<count;++i) buffers.emplace_back(residualSize);
            const auto work=[&](size_t workerIndex) {
                // One buffer per worker, reused across all factors and residual sizes.
                auto& buffer=buffers[workerIndex];
                for(;;) {
                    const auto begin=next.fetch_add(32,std::memory_order_relaxed);
                    if(begin>=entries.size()) break;
                    for(size_t i=begin;i<std::min(begin+32,entries.size());++i) {
                        const auto& entry=entries[i]; const auto& f=*entry.factor;
                        if(!f.participates()) continue;
                        const int n=f.cost->num_residuals();
                        try {
                            if(!(entry.evaluationCost?entry.evaluationCost:f.cost.get())->Evaluate(entry.parameters.data(),buffer.data(),nullptr) || !buffer.head(n).allFinite()) decisions[i]=2;
                            else { energies[i]=buffer.head(n).squaredNorm();
                                if(!std::isfinite(energies[i])) decisions[i]=2;
                                else if(energies[i]>f.nisThreshold) decisions[i]=1;
                            }
                        } catch(...) { decisions[i]=3; }
                    }
                }
            };
            // Cost objects already have the Ceres concurrent-Evaluate contract.
            // Workers only read graph values; apply lifecycle changes after joining,
            // in factor-ID order, so scheduling cannot affect rejection decisions.
            std::vector<std::thread> workers;
            try { for(size_t i=1;i<count;++i) workers.emplace_back(work,i); work(0); }
            catch(...) { for(auto& worker:workers) worker.join(); throw; }
            for(auto& worker:workers) worker.join();
            const auto statisticsStarted=std::chrono::steady_clock::now();
            if(evaluated) *evaluated={}; if(admitted) *admitted={};
            std::array<std::vector<double>,5> allEnergies,admittedEnergies;
            const auto accumulate=[&](ResidualStatisticsArray* target,std::array<std::vector<double>,5>& samples,
                const FactorRecord& factor,double energy,bool over) {
                if(!target) return;
                const size_t category=factor.kind==FactorKind::Visual?0:factor.kind==FactorKind::Imu?1:factor.kind==FactorKind::Lidar?2:4;
                auto& s=(*target)[category]; ++s.factors; s.components+=factor.cost->num_residuals(); s.squaredNorm+=energy;
                if(over) ++s.overGate;
                if(energy>s.maxNis) { s.maxNis=energy; s.worstFactor=factor.id; }
                double rho[3]{energy,1,0}; if(factor.loss) factor.loss->Evaluate(energy,rho);
                s.robustCost+=0.5*rho[0]; samples[category].push_back(energy);
            };
            for(size_t i=0;i<entries.size();++i) {
                if(decisions[i]>=2) {
                    error=decisions[i]==3?"measurement evaluation threw before robust loss":"measurement evaluation failed/nonfinite before robust loss";
                    return false;
                }
                if(!entries[i].factor->participates()) continue;
                accumulate(evaluated,allEnergies,*entries[i].factor,energies[i],decisions[i]==1);
                if(!decisions[i]) accumulate(admitted,admittedEnergies,*entries[i].factor,energies[i],false);
                if(decisions[i]==1) {
                    auto& f=*entries[i].factor;
                    f.lifecycle=FactorLifecycle::Rejected; f.reason="whitened pre-loss NIS gate"; ++rejected;
                }
            }
            const auto quantiles=[](ResidualStatisticsArray* target,std::array<std::vector<double>,5>& samples) {
                if(!target) return;
                for(size_t i=0;i<samples.size();++i) if(!samples[i].empty()) {
                    auto& values=samples[i]; std::sort(values.begin(),values.end());
                    (*target)[i].medianNis=values[values.size()/2];
                    (*target)[i].p95Nis=values[size_t(std::ceil(0.95*values.size()))-1];
                }
            };
            quantiles(evaluated,allEnergies); quantiles(admitted,admittedEnergies);
            if(statisticsMs) *statisticsMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-statisticsStarted).count();
            return true;
        }
        SolveReport BackendOptimizer::optimize(BackendGraph& graph, const SolutionValidator& validator, const SolveBudget& requested) const {
            return optimizeImpl(graph,validator,requested,true);
        }
        SolveReport BackendOptimizer::optimizeImpl(BackendGraph& graph,const SolutionValidator& validator,
            const SolveBudget& requested,bool validateInput,const LandmarkStore* observations,StateId currentFrame) const {
            SolveReport out;
            out.analyticVisualJacobians=options_.analyticVisualJacobians;
            const int maxIterations=requested.maxIterations.value_or(options_.maxIterations);
            const int minIterations=requested.minIterations.value_or(0);
            const int numThreads=requested.numThreads.value_or(options_.numThreads);
            const double timeLimit=requested.timeSeconds.value_or(options_.maxSolverTimeSeconds);
            if(maxIterations<1 || minIterations<0 || minIterations>maxIterations || numThreads<1
                || !std::isfinite(timeLimit) || timeLimit<=0) { out.message="invalid per-step solve budget"; return out; }
            // max<double> is the pipeline's disabled-budget sentinel. Keep its
            // millisecond diagnostic finite; zero means effectively unlimited.
            out.ceresTimeLimitMs=timeLimit<std::numeric_limits<double>::max()/1000.0?timeLimit*1000.0:0.0;
            const auto validationStarted=std::chrono::steady_clock::now();
            if (validateInput && !graph.valid(&out.message)) return out;
            const auto checkpointStarted=std::chrono::steady_clock::now();
            out.validationMs=std::chrono::duration<double,std::milli>(checkpointStarted-validationStarted).count();
            struct NumericCheckpoint {
                struct Block { ParameterBlock* parameter; size_t offset; };
                struct Factor { FactorRecord* record; FactorLifecycle lifecycle; std::string reason; };
                std::vector<Block> blocks; std::vector<double> values; std::vector<Factor> factors;
                bool committed=false,restored=false;
                explicit NumericCheckpoint(BackendGraph& source) {
                    size_t count=0; for(const auto& p:source.blocks) count+=size_t(p.second.value.size());
                    values.reserve(count); blocks.reserve(source.blocks.size()); factors.reserve(source.factors.size());
                    for(auto& p:source.blocks) {
                        blocks.push_back({&p.second,values.size()});
                        values.insert(values.end(),p.second.value.data(),p.second.value.data()+p.second.value.size());
                    }
                    for(auto& f:source.factors) factors.push_back({&f.second,f.second.lifecycle,f.second.reason});
                }
                void restore() {
                    if(restored || committed) return;
                    for(const auto& p:blocks) std::copy(values.begin()+p.offset,
                        values.begin()+p.offset+size_t(p.parameter->value.size()),p.parameter->value.data());
                    for(auto& f:factors) { f.record->lifecycle=f.lifecycle; f.record->reason.swap(f.reason); }
                    restored=true;
                }
                ~NumericCheckpoint() { restore(); }
            } checkpoint(graph);
            std::unique_ptr<detail::SolveWorkspace> workspace;
            out.checkpointMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-checkpointStarted).count();
            // Values and lifecycle flags are copied; cost objects are shared and must remain
            // immutable. Rollback cannot undo arbitrary mutations made inside an external cost.
            const auto rollback = [&](const std::string& why) {
                // Drop every Ceres view BEFORE restoring/resizing graph buffers.
                workspace.reset(); checkpoint.restore(); out.message = why;
                out.accepted=false; out.termination=SolveTermination::Failure; out.unrefittedOutliers=0;
                return out;
            };
            const auto gateSetupStarted=std::chrono::steady_clock::now();
            VisualEvaluationCache gateVisualCache;
            const auto timeBlock=graph.blocks.find({BlockKind::CameraTimeOffset,0});
            const bool fixedCameraTime=timeBlock==graph.blocks.end() || timeBlock->second.constant;
            std::vector<std::unique_ptr<::ceres::CostFunction>> gateAdapters;
            std::vector<GateEntry> gateEntries; gateEntries.reserve(graph.factors.size());
            for(auto& item:graph.factors) if(item.second.participates()) {
                GateEntry entry; entry.factor=&item.second; entry.parameters.reserve(item.second.keys.size());
                for(auto key:item.second.keys) entry.parameters.push_back(graph.blocks.at(key).value.data());
                if(item.second.kind==FactorKind::Visual) {
                    auto cost=cacheWindowVisualCost(*item.second.cost,entry.parameters,gateVisualCache,options_.cacheVisualPairs,
                        fixedCameraTime);
                    if(cost) { entry.evaluationCost=cost.get(); gateAdapters.push_back(std::move(cost)); }
                }
                gateEntries.push_back(std::move(entry));
            }
            // No block buffers or factor nodes are resized/erased by the optimizer.
            // These borrowed pointers expire before any rollback or caller mutation.
            out.gateSetupMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-gateSetupStarted).count();
            const auto priorStatistics=[&](ResidualStatisticsArray& statistics) {
                if(!graph.prior) return true;
                std::vector<const double*> parameters; for(auto key:graph.prior->keys()) parameters.push_back(graph.blocks.at(key).value.data());
                Eigen::VectorXd r(graph.prior->num_residuals());
                if(!graph.prior->Evaluate(parameters.data(),r.data(),nullptr) || !r.allFinite()) return false;
                auto& s=statistics[3]; s.factors=1; s.components=r.size(); s.squaredNorm=r.squaredNorm();
                s.maxNis=s.medianNis=s.p95Nis=s.squaredNorm; s.robustCost=0.5*s.squaredNorm; return std::isfinite(s.squaredNorm);
            };
            const auto gate=[&](size_t& rejected,ResidualStatisticsArray& evaluated,double& phaseMs,ResidualStatisticsArray* admitted=nullptr) {
                const auto beginning=std::chrono::steady_clock::now();
                // Explicit refresh at graph values: the last Ceres trial need not
                // be the accepted state. Never reuse loss-corrected residuals.
                gateVisualCache.PrepareForEvaluation(false,true);
                const bool valid=statisticalGate(gateEntries,rejected,out.message,numThreads,&evaluated,admitted,&out.residualStatisticsMs);
                const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-beginning).count();
                phaseMs+=elapsed; out.gatingMs+=elapsed;
                out.gateVisualPrepareMs=gateVisualCache.posePrepareMs+gateVisualCache.cameraPrepareMs
                    +gateVisualCache.pairPrepareMs+gateVisualCache.timedAnchorPrepareMs;
                return valid;
            };
            out.preNisSkipped=requested.skipPreNis;
            if (!requested.skipPreNis) {
                if (!gate(out.preRejected,out.preGateResiduals,out.preNisMs,&out.preSolveResiduals)) return rollback(out.message);
                const auto statisticsStarted=std::chrono::steady_clock::now();
                if(!priorStatistics(out.preGateResiduals)) return rollback("invalid historical prior residual statistics");
                out.preSolveResiduals[3]=out.preGateResiduals[3]; out.residualStatisticsValid=true;
                out.residualStatisticsMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-statisticsStarted).count();
            }
            int remaining = maxIterations;
            uint64_t newest=0;
            for(const auto& item:graph.blocks) if(item.first.kind==BlockKind::Pose) newest=std::max(newest,item.first.id);
            out.factorProfiled=VLOG_IS_ON(1) || (newest && newest%30==0);
            const auto acceptCandidate=[&](SolveTermination termination,const std::string& message) {
                // A limit is not a numerical failure. Accept only a usable Ceres
                // state whose CLEANED graph still passes every normal quality gate.
                bool constrained=bool(graph.prior);
                for(const auto& item:graph.factors) constrained=constrained || item.second.participates();
                if(!out.passes || !constrained) return rollback("no usable constrained solution at solve limit");
                // Bounds/finiteness were checked after Solve; gating only changes
                // factor lifecycle, so do not repeat full graph validation here.
                try {
                    if (requested.calibrationBlockViolation) {
                        std::string violations;
                        for (const auto& item : graph.blocks) {
                            if (!item.first.isCalibration() || item.second.constant) continue;
                            const auto reason = requested.calibrationBlockViolation(item.second);
                            if (reason.empty()) continue;
                            out.rejectedCalibration.push_back(item.first);
                            if (!violations.empty()) violations += ", ";
                            violations += reason;
                        }
                        if (!out.rejectedCalibration.empty()) return rollback(violations);
                    }
                    if(validator && !validator(graph,out.message))
                        return rollback(out.message.empty()?"application quality gate rejected candidate":out.message);
                } catch(const std::exception& e) { return rollback(std::string("solution validator failed: ")+e.what()); }
                checkpoint.committed=true; out.accepted=true; out.termination=termination; out.message=message;
                return out;
            };
            for (int pass = 0; pass <= options_.maxOutlierRefits; ++pass) {
                // Reserve some of the remaining iteration allowance for possible outlier refits.
                // The initial pass can therefore stop below maxIterations even without outliers.
                // Only completed Ceres calls consume time. Gate evaluation,
                // Problem construction and inter-pass validation do not.
                const double timeRemaining = timeLimit - out.ceresMs*1e-3;
                const int minimumRemaining=std::max(0,minIterations-out.iterations);
                if (remaining <= 0 || (timeRemaining <= 0 && minimumRemaining==0)) {
                    return acceptCandidate(timeRemaining<=0?SolveTermination::TimeBudget:SolveTermination::IterationBudget,
                        "solve allowance exhausted; accepted last usable state after outlier cleanup");
                }
                const int reserve = std::min(options_.maxOutlierRefits - pass,
                    std::max(0,remaining-std::max(1,minimumRemaining)));
                const int budget = remaining - std::max(0, reserve);
                ::ceres::Solver::Summary summary;
                bool hasResidual = false;
                {
                    const auto buildStarted=std::chrono::steady_clock::now();
                    if(!workspace) workspace=std::make_unique<detail::SolveWorkspace>();
                    auto& counters=workspace->counters; auto& adapters=workspace->adapters;
                    auto& problem=*workspace->problem;
                    const auto profiled=[&](::ceres::CostFunction* cost,size_t category)->::ceres::CostFunction* {
                        if(!out.factorProfiled) return nullptr;
                        adapters.emplace_back(std::make_unique<detail::ProfiledCost>(cost,counters[category]));
                        return adapters.back().get();
                    };
                    // Add only used parameters. Do not feed disconnected states to Ceres as unconstrained variables.
                    auto& used=workspace->used;
                    if(!workspace->built) {
                        for (const auto& item : graph.factors) if (item.second.participates())
                            used.insert(item.second.keys.begin(), item.second.keys.end());
                        if (graph.prior) used.insert(graph.prior->keys().begin(), graph.prior->keys().end());
                        // Schur group zero must be an independent set in EVERY residual,
                        // including the historical prior. Never split/approximate a prior
                        // merely to obtain the faster solver. Frozen landmarks are excluded.
                        std::set<BlockKey> eliminate;
                        for(auto key:used) if(key.kind==BlockKind::Landmark && !graph.blocks.at(key).constant) eliminate.insert(key);
                        const auto independent=[&](const auto& keys) {
                            size_t count=0; for(auto key:keys) if(eliminate.count(key) && ++count>1) return false;
                            return true;
                        };
                        bool schur=!eliminate.empty() && (!graph.prior || independent(graph.prior->keys()));
                        bool retainedVariable=false;
                        for(auto key:used) retainedVariable=retainedVariable || (!graph.blocks.at(key).constant && !eliminate.count(key));
                        schur=schur && retainedVariable;
                        for(const auto& item:graph.factors) if(item.second.participates())
                            schur=schur && independent(item.second.keys);
                        out.schurIndependent=schur; out.activeLandmarks=eliminate.size();
                        workspace->schur=schur;
                        if(graph.prior) {
                            out.priorRows=int(graph.prior->squareRootInformation().rows());
                            out.priorColumns=int(graph.prior->squareRootInformation().cols());
                        }
                        for (auto k : used) {
                            auto& block = graph.blocks.at(k);
                            detail::addBlock(problem, block, true);
                            if (requested.cameraRotationOnly && k.kind == BlockKind::CameraExtrinsic && !block.constant) {
                                // Only this Ceres problem uses the restricted tangent. The graph,
                                // marginalizer and FEJ prior keep the canonical six-dimensional tangent.
                                workspace->rigRotationManifold = std::make_unique<detail::SolveWorkspace::RigRotationManifold>(
                                    ::ceres::SubsetManifold(3, {0, 1, 2}), LeftSO3Manifold());
                                problem.SetManifold(block.value.data(), workspace->rigRotationManifold.get());
                            }
                        }
                        for (const auto& item : graph.factors) if (item.second.participates()) {
                            const auto kind=item.second.kind;
                            const size_t category=kind==FactorKind::Visual?0:kind==FactorKind::Imu?1:kind==FactorKind::Lidar?2:4;
                            ::ceres::CostFunction* evaluated=item.second.cost.get();
                            if(kind==FactorKind::Visual) {
                                std::vector<const double*> parameters;
                                for(auto key:item.second.keys) parameters.push_back(graph.blocks.at(key).value.data());
                                auto cached=cacheWindowVisualCost(*evaluated,parameters,workspace->visualCache,options_.cacheVisualPairs,
                                    fixedCameraTime);
                                if(cached) { evaluated=cached.get(); adapters.push_back(std::move(cached)); }
                            }
                            auto* measured=profiled(evaluated,category);
                            workspace->residuals[item.first]=detail::addFactor(problem, graph, item.second,measured?measured:evaluated);
                        }
                        if (graph.prior) { detail::addPrior(problem, graph,profiled(graph.prior.get(),3)); workspace->hasPrior=true; }
                        workspace->built=true; ++out.problemBuilds;
                    } else {
                        // Post-fit NIS only removes measurements; it cannot introduce
                        // a new dependency or break a previously valid Schur ordering.
                        for(auto it=workspace->residuals.begin();it!=workspace->residuals.end();) {
                            if(!graph.factors.at(it->first).participates()) {
                                problem.RemoveResidualBlock(it->second); it=workspace->residuals.erase(it);
                            } else ++it;
                        }
                        std::set<BlockKey> stillUsed;
                        for(const auto& item:graph.factors) if(item.second.participates())
                            stillUsed.insert(item.second.keys.begin(),item.second.keys.end());
                        if(graph.prior) stillUsed.insert(graph.prior->keys().begin(),graph.prior->keys().end());
                        // Removing the last residual of a landmark must not leave an
                        // unused elimination group or an unconstrained Ceres buffer.
                        for(auto key:used) if(!stillUsed.count(key)) problem.RemoveParameterBlock(graph.blocks.at(key).value.data());
                        used=std::move(stillUsed);
                        bool landmarkVariable=false,retainedVariable=false; out.activeLandmarks=0;
                        for(auto key:used) if(!graph.blocks.at(key).constant) {
                            if(key.kind==BlockKind::Landmark) { landmarkVariable=true; ++out.activeLandmarks; }
                            else retainedVariable=true;
                        }
                        workspace->schur=workspace->schur && landmarkVariable && retainedVariable;
                        out.schurIndependent=workspace->schur;
                        ++out.problemReuses;
                    }
                    const bool schur=workspace->schur;
                    hasResidual=workspace->hasPrior || !workspace->residuals.empty();
                    out.problemBuildMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-buildStarted).count();
                    if (hasResidual) {
                        if (observations) {
                            // Count the same variable blocks as activeLandmarks at each pass.
                            // The store guarantees one observation per landmark, frame and camera.
                            auto& counts=out.activeLandmarksByCamera;
                            counts.clear();
                            for (const auto& key : observations->inFrame(currentFrame)) {
                                const auto& observation=*observations->find(key);
                                const BlockKey block{BlockKind::Landmark,observation.landmark.value()};
                                if (!used.count(block) || graph.blocks.at(block).constant
                                    || observation.use==ObservationUse::Absorbed) continue;
                                const auto& landmark=*observations->find(observation.landmark);
                                if (!(key==landmark.anchor)) {
                                    const auto factor=graph.factors.find(observation.factorId);
                                    if (factor==graph.factors.end() || !factor->second.participates()) continue;
                                }
                                if (counts.size()<=key.cameraIndex) counts.resize(key.cameraIndex+1);
                                ++counts[key.cameraIndex];
                            }
                        }
                        ::ceres::Solver::Options solver;
                        solver.trust_region_strategy_type = ::ceres::LEVENBERG_MARQUARDT;
                        // Positive inverse-range bounds trigger Ceres' projected Armijo
                        // line search. Quadratic interpolation reuses the initial slope;
                        // unlike CUBIC, it needs no gradient at each trial point. Keep
                        // the bounds and normal step-acceptance/backtracking checks.
                        solver.line_search_interpolation_type = ::ceres::QUADRATIC;
                        out.lineSearchInterpolation = ::ceres::LineSearchInterpolationTypeToString(solver.line_search_interpolation_type);
                        solver.linear_solver_type = ::ceres::DENSE_QR;
                        // Prefer an available sparse normal-equation backend. Do not blindly use
                        // a Schur landmark ordering: a historical dense prior can couple several
                        // landmark blocks in one residual, violating Schur's independent-set rule.
                        if(options_.linearSolver==BackendLinearSolver::DenseSchur && options_.preferLandmarkSchur && schur) {
                            solver.linear_solver_type=::ceres::DENSE_SCHUR;
                            solver.linear_solver_ordering=std::make_shared<::ceres::ParameterBlockOrdering>();
                            for(auto key:used) if(!graph.blocks.at(key).constant)
                                solver.linear_solver_ordering->AddElementToGroup(graph.blocks.at(key).value.data(),key.kind==BlockKind::Landmark?0:1);
                        } else if (options_.linearSolver == BackendLinearSolver::SparseNormalCholesky || options_.linearSolver==BackendLinearSolver::DenseSchur) {
                            if (::ceres::IsSparseLinearAlgebraLibraryTypeAvailable(::ceres::SUITE_SPARSE)) {
                                solver.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;
                                solver.sparse_linear_algebra_library_type = ::ceres::SUITE_SPARSE;
                            }
                            else if (::ceres::IsSparseLinearAlgebraLibraryTypeAvailable(::ceres::EIGEN_SPARSE)) {
                                solver.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;
                                solver.sparse_linear_algebra_library_type = ::ceres::EIGEN_SPARSE;
                            }
                            if(options_.preferLandmarkSchur && schur && solver.linear_solver_type==::ceres::SPARSE_NORMAL_CHOLESKY) {
                                solver.linear_solver_type=::ceres::SPARSE_SCHUR;
                                solver.linear_solver_ordering=std::make_shared<::ceres::ParameterBlockOrdering>();
                                for(auto key:used) if(!graph.blocks.at(key).constant)
                                    solver.linear_solver_ordering->AddElementToGroup(graph.blocks.at(key).value.data(),key.kind==BlockKind::Landmark?0:1);
                            }
                        }
                        solver.max_num_iterations = budget;
                        // All passes share Ceres-only elapsed time. The clock starts
                        // immediately before Solve, after Problem/options preparation.
                        // The minimum-iteration setting makes this a soft allowance;
                        // convergence can still finish before that minimum is reached.
                        class TimeBudget final : public ::ceres::IterationCallback {
                        public:
                            std::function<double()> elapsed;
                            double limit; int minimum;
                            ::ceres::CallbackReturnType operator()(const ::ceres::IterationSummary& iteration) override {
                                return iteration.iteration>=minimum && elapsed()>=limit
                                    ? ::ceres::SOLVER_TERMINATE_SUCCESSFULLY : ::ceres::SOLVER_CONTINUE;
                            }
                        } timer;
                        std::chrono::steady_clock::time_point solveStarted;
                        timer.elapsed=[&]() {
                            return out.ceresMs*1e-3+std::chrono::duration<double>(
                                std::chrono::steady_clock::now()-solveStarted).count();
                        };
                        timer.limit=timeLimit; timer.minimum=minimumRemaining;
                        solver.max_solver_time_in_seconds = minimumRemaining>0
                            ?std::numeric_limits<double>::max():std::max(1e-9,timeRemaining);
                        if(minimumRemaining>0) solver.callbacks.push_back(&timer);
                        solver.num_threads = numThreads;
                        solver.minimizer_progress_to_stdout = false;
                        solveStarted=std::chrono::steady_clock::now();
                        ::ceres::Solve(solver, &problem, &summary);
                        out.ceresMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-solveStarted).count();
                        out.ceresTimeBudgetReached=out.ceresMs*1e-3>=timeLimit;
                        ++out.passes;
                        out.linearSolverMs+=std::max(0.0,summary.linear_solver_time_in_seconds)*1000;
                        out.jacobianMs+=std::max(0.0,summary.jacobian_evaluation_time_in_seconds)*1000;
                        out.residualMs+=std::max(0.0,summary.residual_evaluation_time_in_seconds)*1000;
                        out.jacobianEvaluations+=std::max(0,summary.num_jacobian_evaluations);
                        out.residualEvaluations+=std::max(0,summary.num_residual_evaluations);
                        out.lineSearchCostMs+=std::max(0.0,summary.line_search_cost_evaluation_time_in_seconds)*1000;
                        out.lineSearchGradientMs+=std::max(0.0,summary.line_search_gradient_evaluation_time_in_seconds)*1000;
                        out.lineSearchMs+=std::max(0.0,summary.line_search_total_time_in_seconds)*1000;
                        out.parameterBlocks=summary.num_parameter_blocks; out.residualBlocks=summary.num_residual_blocks;
                        out.residualDimension=summary.num_residuals; out.effectiveParameters=summary.num_effective_parameters;
                        out.linearSolverUsed=::ceres::LinearSolverTypeToString(summary.linear_solver_type_used);
                        out.sparseBackend=(solver.linear_solver_type==::ceres::DENSE_QR || solver.linear_solver_type==::ceres::DENSE_SCHUR)?"none":
                            ::ceres::SparseLinearAlgebraLibraryTypeToString(solver.sparse_linear_algebra_library_type);
                        out.visualPosePrepareMs=workspace->visualCache.posePrepareMs;
                        out.visualCameraPrepareMs=workspace->visualCache.cameraPrepareMs;
                        out.visualPairPrepareMs=workspace->visualCache.pairPrepareMs;
                        out.visualTimedAnchorPrepareMs=workspace->visualCache.timedAnchorPrepareMs;
                        out.visualSharedTimedAnchors=workspace->visualCache.sharedTimedAnchorCount();
                        out.visualCachedCameras=workspace->visualCache.cameraCount();
                        out.visualCachedPairs=workspace->visualCache.pairCount();
                        out.visualCacheRefreshes=workspace->visualCache.refreshCount();
                        if(out.factorProfiled) for(size_t i=0;i<counters.size();++i) {
                            const auto total=counters[i].totals();
                            out.factorJacobianSumMs[i]=double(total.jacobianNs)*1e-6;
                            out.factorResidualSumMs[i]=double(total.residualNs)*1e-6;
                            out.factorJacobianCalls[i]=total.jacobianCalls;
                            out.factorResidualCalls[i]=total.residualCalls;
                        }
                    }
                } // Workspace persists across refits; rollback explicitly destroys its views.
                if (!hasResidual) return rollback("no active measurements or marginalization prior");
                int usedIterations = 0;
                // Iteration zero describes initialization, not a completed LM update. Charge at
                // least one unit per pass so an immediate convergence cannot fund unlimited refits.
                for (const auto& iteration : summary.iterations) if (iteration.iteration > 0) ++usedIterations;
                remaining -= std::max(1, usedIterations); out.iterations += usedIterations;
                if (!pass) out.initialCost = summary.initial_cost;
                out.finalCost = summary.final_cost;
                // Compare costs within the same pass/factor set. After rejection, a refit's
                // objective contains different evidence, so cross-pass cost ratios are misleading.
                if (!summary.IsSolutionUsable() || !std::isfinite(summary.initial_cost) || !std::isfinite(summary.final_cost)
                    || summary.final_cost > summary.initial_cost + options_.maxRelativeCostIncrease * std::max(1.0, summary.initial_cost))
                    return rollback("Ceres rejected solution or cost increased: " + summary.BriefReport());
                for (const auto& p : graph.blocks) if (!p.second.valid(&out.message)) return rollback(out.message);
                size_t rejected = 0;
                // Remove live outliers even when there is no time to refit. Window
                // maintenance archives them BEFORE any marginalization linearization.
                ResidualStatisticsArray survivingResiduals;
                if (!gate(rejected,out.postSolveResiduals,out.postNisMs,&survivingResiduals)) return rollback(out.message);
                const auto postStatisticsStarted=std::chrono::steady_clock::now();
                if(!priorStatistics(out.postSolveResiduals)) return rollback("invalid optimized historical prior residual statistics");
                out.residualStatisticsValid=true;
                out.residualStatisticsMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-postStatisticsStarted).count();
                out.postRejected += rejected;
                out.unrefittedOutliers=rejected;
                if(VLOG_IS_ON(1) || rejected || out.ceresMs>200)
                    LOG(INFO)<<"[Backend][Refit] pass="<<pass<<" budget="<<budget<<" remaining="<<remaining
                        <<" post_rejected="<<rejected<<" solver="<<out.linearSolverUsed<<" sparse_backend="<<out.sparseBackend
                        <<" initial_cost="<<summary.initial_cost<<" final_cost="<<summary.final_cost;
                const bool iterationsEnded=remaining<=0;
                const bool refitsEnded=pass==options_.maxOutlierRefits;
                if(!rejected || out.ceresTimeBudgetReached || iterationsEnded || refitsEnded) {
                    const auto termination=out.ceresTimeBudgetReached?SolveTermination::TimeBudget:
                        iterationsEnded?SolveTermination::IterationBudget:
                        rejected && refitsEnded?SolveTermination::RefitBudget:SolveTermination::SolverFinished;
                    // Report the surviving factor set at the accepted state. This
                    // is NOT a claim that the cleaned objective has converged.
                    survivingResiduals[3]=out.postSolveResiduals[3];
                    out.postSolveResiduals=survivingResiduals;
                    return acceptCandidate(termination,summary.BriefReport()+(rejected?
                        "; accepted at solve limit after live outlier cleanup without another refit":""));
                }
            }
            return rollback("outlier refit failed");
        }

        CalibrationInformation BackendOptimizer::calibrationInformation(BackendGraph& graph, bool includePrior) const {
            CalibrationInformation out; out.includesHistoricalPrior = includePrior;
            std::vector<FactorId> factors;
            // Exclude explicit calibration priors so their chosen strength cannot manufacture
            // apparent sensor observability. The historical prior is separately opt-in; other
            // active constraints, including state priors tagged Other, remain in this report.
            for (const auto& item : graph.factors) if (item.second.participates()
                && item.second.kind != FactorKind::Calibration) factors.push_back(item.first);
            LinearizedSystem lin;
            if (!marginalizer_.linearize(graph, factors, includePrior, true, lin, out.message)) return out;
            std::vector<int> nuisance, calibration, ranges;
            // Partition every participating tangent coordinate. Even temporarily constant
            // calibration blocks contribute derivatives through Marginalizer::linearize().
            std::vector<int> reportIndices;
            for (size_t k = 0; k < lin.keys.size(); ++k) {
                const auto key = lin.keys[k]; const int o = lin.offsets[k], n = lin.sizes[k];
                if (!key.isCalibration()) {
                    if (key.kind == BlockKind::Landmark && n == 1) ranges.push_back(o);
                    else for (int j = 0; j < n; ++j) nuisance.push_back(o + j);
                    continue;
                }
                const bool pose = key.kind == BlockKind::CameraExtrinsic || key.kind == BlockKind::LidarExtrinsic;
                const int base = (key.kind == BlockKind::LidarExtrinsic || key.kind == BlockKind::LidarTimeOffset) ? 7 : 0;
                if ((pose && n != 6) || (!pose && n != 1)) { out.message = "invalid calibration dimensions"; return out; }
                for (int j = 0; j < n; ++j) {
                    calibration.push_back(o + j);
                    // Reorder [rho,theta] to [theta,rho,td] for camera first, then LiDAR.
                    // The physical translation conversion follows after elimination.
                    reportIndices.push_back(base + (pose ? (j < 3 ? 3 + j : j - 3) : 6));
                }
            }
            // Visual factors each own one inverse range. Eliminate these scalars before
            // the state solve. A coupled historical prior requires joint elimination.
            bool independentRanges = options_.landmarkFirstMarginalization;
            for (size_t i = 0; independentRanges && i < ranges.size(); ++i)
                for (size_t j = 0; j < i; ++j)
                    if (lin.H(ranges[i], ranges[j]) != 0.0) { independentRanges = false; break; }
            if (!independentRanges) nuisance.insert(nuisance.end(), ranges.begin(), ranges.end());
            const int nx = static_cast<int>(nuisance.size()), nc = static_cast<int>(calibration.size());
            if (!nc) { out.message = "no calibration Jacobians in selected measurements"; return out; }
            Eigen::MatrixXd Hxx(nx, nx), Hxc(nx, nc), Hcc(nc, nc), solved;
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < nx; ++j) Hxx(i, j) = lin.H(nuisance[i], nuisance[j]);
                for (int j = 0; j < nc; ++j) Hxc(i, j) = lin.H(nuisance[i], calibration[j]);
            }
            for (int i = 0; i < nc; ++i) for (int j = 0; j < nc; ++j) Hcc(i, j) = lin.H(calibration[i], calibration[j]);
            if (independentRanges && !ranges.empty()) {
                Eigen::VectorXd diagonal(ranges.size());
                for (Eigen::Index i = 0; i < diagonal.size(); ++i) diagonal[i] = lin.H(ranges[i], ranges[i]);
                const double cutoff = std::max(options_.absoluteEigenThreshold,
                    options_.relativeEigenThreshold * diagonal.maxCoeff());
                Eigen::VectorXd x(nx), c(nc);
                for (Eigen::Index i = 0; i < diagonal.size(); ++i) {
                    if (diagonal[i] <= cutoff) continue;
                    const double scale = 1.0 / std::sqrt(diagonal[i]);
                    for (int j = 0; j < nx; ++j) x[j] = scale * lin.H(ranges[i], nuisance[j]);
                    for (int j = 0; j < nc; ++j) c[j] = scale * lin.H(ranges[i], calibration[j]);
                    Hxx.noalias() -= x * x.transpose();
                    Hxc.noalias() -= x * c.transpose();
                    Hcc.noalias() -= c * c.transpose();
                }
            }
            if (!marginalizer_.psdSolve(Hxx, Hxc, solved, out.message)) return out;
            // Removing nuisance-state freedom is essential: Hcc alone overstates information
            // when pose/velocity/landmark changes can explain the same measurement variation.
            const Eigen::MatrixXd conditional = Hcc - Hxc.transpose() * solved;
            for (int i = 0; i < nc; ++i) for (int j = 0; j < nc; ++j) out.conditional(reportIndices[i], reportIndices[j]) = conditional(i, j);
            Eigen::Matrix<double, 14, 14> basis = Eigen::Matrix<double, 14, 14>::Identity();
            for (const auto kind : {BlockKind::CameraExtrinsic, BlockKind::LidarExtrinsic}) {
                const auto pose = graph.blocks.find({kind, 0});
                if (pose == graph.blocks.end()) continue;
                const int base = kind == BlockKind::CameraExtrinsic ? 0 : 7;
                // Under left SE(3), dp = rho - [p]x dtheta.
                basis.block<3, 3>(base + 3, base) = skew(blockPose(pose->second).r());
            }
            out.conditional = (basis.transpose() * out.conditional * basis).eval();
            out.conditional = (0.5 * (out.conditional + out.conditional.transpose())).eval();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 14, 14>> eig(out.conditional);
            if (eig.info() != Eigen::Success || !out.conditional.allFinite()) { out.message = "conditional calibration eigensolve failed"; return out; }
            const double cutoff = std::max(options_.absoluteEigenThreshold,
                options_.relativeEigenThreshold * std::max(0.0, eig.eigenvalues().maxCoeff()));
            if (eig.eigenvalues().minCoeff() < -10.0 * cutoff) { out.message = "conditional calibration information is indefinite"; return out; }
            out.eigenvalues = eig.eigenvalues().cwiseMax(0.0);
            // Clamp only tolerated negative roundoff; significant indefiniteness was rejected.
            out.conditional = eig.eigenvectors() * out.eigenvalues.asDiagonal() * eig.eigenvectors().transpose();
            out.success = true; return out;
        }
    }
}
