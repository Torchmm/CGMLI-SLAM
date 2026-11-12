#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>
#include "GpuRetainedFactors.hpp"

namespace cgmli::backend::detail {

// Layout at the device boundary. Parameter offsets refer to one contiguous
// vector of unique solver blocks. Fixed camera transforms are row-major.
struct VisualBatchInput {
    int offsets[5]{}, sizes[5]{};
    int blocks=0, observer=0, extrinsic=0, time=-1, range=0;
    int jacobianOffset=0,jacobianSize=0;
    unsigned derivativeMask=0;
    double anchor[3]{}, measured[3]{}, anchorStatic[3]{}, measuredStatic[3]{};
    double anchorVelocity[3]{}, measuredVelocity[3]{}, anchorTime=0, measuredTime=0;
    double anchorRotation[9]{}, measuredRotation[9]{}, anchorTranslation[3]{}, measuredTranslation[3]{};
    double measurement[6]{};
    bool movingAnchor=false, movingMeasured=false;
};
struct VisualBatchResidual {
    double value[2]{};
    int valid=0;
};
struct VisualBatchView {
    const VisualBatchResidual* residuals=nullptr;
    const double* jacobians=nullptr;
};
struct DirectVisualLayout {
    int column[5]{},width[5]{},plus[5]{};
    double huberScale=0;
};
struct VisualNormalSystem;
struct VisualSchurStep {
    const double* increment=nullptr; ///< State tangents followed by inverse-range increments.
    double cost=0,predictedReduction=0,trialCost=0,maxIncrement=0;
    bool trialValid=false;
};
const char* takeVisualCudaRuntimeError();
class VisualCudaBatch {
public:
    VisualCudaBatch();
    ~VisualCudaBatch();
    VisualCudaBatch(const VisualCudaBatch&)=delete;
    VisualCudaBatch& operator=(const VisualCudaBatch&)=delete;
    bool evaluate(const std::vector<VisualBatchInput>& inputs,const std::vector<double>& parameters,
        bool topologyChanged,bool derivatives,std::size_t jacobianSize,VisualBatchView& result);
    /// Audit-only path from the last device Jacobian to robust tangent normal equations.
    bool directNormal(const std::vector<VisualBatchInput>&,const std::vector<DirectVisualLayout>&,
        const std::vector<double>& plus,const std::vector<int>& retained,
        const std::vector<int>& eliminated,VisualNormalSystem&);
    /// Partition coordinates are ordered [state tangents, scalar inverse ranges].
    /// Keep the visual normal blocks on device across damping trials.
    bool linearizeSchur(const std::vector<VisualBatchInput>&,const std::vector<double>& parameters,
        std::size_t jacobianSize,const std::vector<DirectVisualLayout>&,const std::vector<double>& plus,
        int retained,int ranges,bool newTopology);
    bool solveOnline(double lambda,const double* retainedH,const double* retainedG,
        const std::vector<double>& lower,const std::vector<double>& upper,bool newPoint,VisualSchurStep&);
    bool configureRetained(const GpuRetainedFactors&);
    bool linearizeRetained();
    bool readNormal(double* upperRowMajor,double* gradient,double& cost,double& residualSquaredNorm);
    bool configureResident(const std::vector<GpuVariable>&,const std::vector<double>& lower,const std::vector<double>& upper);
    bool linearizeResident();
    void acceptResident();
    bool readResident(std::vector<double>&);
    bool trialCost(const std::vector<double>& parameters,double& cost);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Backend copies share capacity only. A solve leases it exclusively and binds
// its own factors, parameter offsets and topology.
struct VisualCudaWorkspace {
    std::mutex mutex;
    std::unique_ptr<VisualCudaBatch> batch;
};

struct VisualNormalSystem {
    std::vector<double> H,g,reducedH,reducedG;
};
/// Debug replay of a Ceres-robustified tangent Jacobian. Every listed landmark
/// column is scalar and independent of the other eliminated columns.
bool cudaVisualNormalSystem(const std::vector<int>& rows,const std::vector<int>& columns,
    const std::vector<double>& values,const std::vector<double>& residuals,int dimensions,
    const std::vector<int>& retained,const std::vector<int>& eliminated,VisualNormalSystem&);

// Reuse the stream, handles and dense buffers across short-lived backend workers.
// Dense marginalization uses column-major matrices and no LM damping.
class MarginalCudaWorkspace {
public:
    MarginalCudaWorkspace();
    ~MarginalCudaWorkspace();
    bool eigen(const double* H,int n,double* eigenvalues,double* eigenvectors);
    bool psdSolve(const double* H,const double* rhs,int n,int columns,double absoluteCutoff,double relativeCutoff,double* solution);
    bool subtract(const double* coupling,const double* solution,int eliminated,int retained,double* H,double* g);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::mutex mutex_;
};

}
