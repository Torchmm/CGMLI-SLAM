#pragma once
// Internal Ceres bridge shared by nonlinear optimization and marginalization.
#include "MarginalizationPrior.hpp"
#include "WindowVisualFactor.hpp"
#include <ceres/ceres.h>
#include <ceres/product_manifold.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <stdexcept>

namespace cgmli {
    namespace backend {
        namespace detail {
            struct EvaluationCounters {
                struct Totals { uint64_t jacobianNs=0,residualNs=0,jacobianCalls=0,residualCalls=0; };
                struct alignas(64) Slot {
                    std::atomic<uint64_t> jacobianNs{0},residualNs{0},jacobianCalls{0},residualCalls{0};
                };
                // Workers usually hit different cache lines. The former single
                // category-wide atomic counter serialized thousands of tiny factors.
                std::array<Slot,64> slots;
                void record(bool derivative,uint64_t ns) {
                    thread_local const size_t stripe=std::hash<std::thread::id>{}(std::this_thread::get_id())%64;
                    auto& s=slots[stripe];
                    (derivative?s.jacobianNs:s.residualNs).fetch_add(ns,std::memory_order_relaxed);
                    (derivative?s.jacobianCalls:s.residualCalls).fetch_add(1,std::memory_order_relaxed);
                }
                Totals totals() const {
                    Totals result;
                    for(const auto& s:slots) {
                        result.jacobianNs+=s.jacobianNs.load(std::memory_order_relaxed);
                        result.residualNs+=s.residualNs.load(std::memory_order_relaxed);
                        result.jacobianCalls+=s.jacobianCalls.load(std::memory_order_relaxed);
                        result.residualCalls+=s.residualCalls.load(std::memory_order_relaxed);
                    }
                    return result;
                }
                void reset() {
                    for(auto& s:slots) {
                        s.jacobianNs.store(0,std::memory_order_relaxed);
                        s.residualNs.store(0,std::memory_order_relaxed);
                        s.jacobianCalls.store(0,std::memory_order_relaxed);
                        s.residualCalls.store(0,std::memory_order_relaxed);
                    }
                }
            };
            // Per-solve adapter, never installed in the graph or a checkpoint.
            // Cost ownership/whitening and all Jacobian requests pass through.
            class ProfiledCost final : public ::ceres::CostFunction {
            public:
                ProfiledCost(const ::ceres::CostFunction* cost,EvaluationCounters& counters):cost_(cost),counters_(counters) {
                    set_num_residuals(cost->num_residuals()); *mutable_parameter_block_sizes()=cost->parameter_block_sizes();
                }
                bool Evaluate(double const* const* p,double* r,double** J) const override {
                    bool derivative=false;
                    if(J) for(size_t i=0;i<parameter_block_sizes().size();++i) derivative=derivative || J[i]!=nullptr;
                    const auto start=std::chrono::steady_clock::now(); const bool ok=cost_->Evaluate(p,r,J);
                    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
                    counters_.record(derivative,uint64_t(ns));
                    return ok;
                }
            private:
                const ::ceres::CostFunction* cost_; EvaluationCounters& counters_;
            };
            inline ::ceres::Problem::Options problemOptions() {
                // Graph records/shared_ptrs own these objects across solves and checkpoints. Default
                // Ceres ownership would delete shared costs/manifolds when a temporary Problem dies.
                ::ceres::Problem::Options o;
                o.cost_function_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
                o.loss_function_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
                o.manifold_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
                return o;
            }
            inline void addBlock(::ceres::Problem& problem, ParameterBlock& p, bool honorConstant) {
                // Nonlinear solving honors temporary freezing. Marginalization/observability call
                // with false to retain the complete variable dependence, including calibration.
                // p.value must remain at the same address until this Problem is destroyed.
                problem.AddParameterBlock(p.value.data(), static_cast<int>(p.value.size()), p.manifold.get());
                if (honorConstant && p.constant) problem.SetParameterBlockConstant(p.value.data());
                for (int i = 0; i < p.lower.size(); ++i) problem.SetParameterLowerBound(p.value.data(), i, p.lower[i]);
                for (int i = 0; i < p.upper.size(); ++i) problem.SetParameterUpperBound(p.value.data(), i, p.upper[i]);
            }
            inline ::ceres::ResidualBlockId addFactor(::ceres::Problem& problem, BackendGraph& graph, const FactorRecord& f,::ceres::CostFunction* profiled=nullptr) {
                if(!f.participates()) throw std::invalid_argument("ineligible factor passed to Ceres");
                // Resolve by stable logical keys immediately before constructing the borrowed view.
                // The order is the factor factory's ambient parameter order, not map iteration order.
                std::vector<double*> p;
                for (const auto& k : f.keys) p.push_back(graph.blocks.at(k).value.data());
                return problem.AddResidualBlock(profiled?profiled:f.cost.get(), f.loss.get(), p);
            }
            inline ::ceres::ResidualBlockId addPrior(::ceres::Problem& problem, BackendGraph& graph,::ceres::CostFunction* profiled=nullptr) {
                // This cost already represents historical robustified/whitened information.
                // Applying a new robust loss would alter previously accepted evidence a second time.
                if (!graph.prior) return nullptr;
                std::vector<double*> p;
                for (const auto& k : graph.prior->keys()) p.push_back(graph.blocks.at(k).value.data());
                return problem.AddResidualBlock(profiled?profiled:graph.prior.get(), nullptr, p);
            }
            struct SolveWorkspace {
                std::array<EvaluationCounters,5> counters;
                VisualEvaluationCache visualCache;
                std::vector<std::unique_ptr<::ceres::CostFunction>> adapters;
                using RigRotationManifold = ::ceres::ProductManifold<::ceres::SubsetManifold, LeftSO3Manifold>;
                std::unique_ptr<RigRotationManifold> rigRotationManifold;
                std::unique_ptr<::ceres::SubsetManifold> velocityManifold;
                // Problem dies before its borrowed adapters/counters.
                std::unique_ptr<::ceres::Problem> problem;
                SolveWorkspace() {
                    auto options=problemOptions(); options.evaluation_callback=&visualCache;
                    problem=std::make_unique<::ceres::Problem>(options);
                }
                std::map<FactorId,::ceres::ResidualBlockId> residuals;
                ::ceres::ResidualBlockId priorResidual=nullptr;
                std::set<BlockKey> used;
                bool built=false,schur=false,hasPrior=false;
            };
        }
    }
}
