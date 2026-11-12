#include "VisualSphereBatchMath.hpp"
#include "GpuRetainedFactorMath.hpp"
#include <common/CudaArray.hpp>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>

namespace cgmli::backend::detail {
namespace {
__global__ void evaluateVisual(const VisualBatchInput* inputs,int count,const double* values,
    VisualBatchResidual* residuals,double* jacobians) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count) return;
    auto& output=residuals[i];
    output.value[0]=output.value[1]=0;
    const auto& factor=inputs[i];
    double* J=jacobians && factor.jacobianSize?jacobians+factor.jacobianOffset:nullptr;
    if(J) for(int k=0;k<factor.jacobianSize;++k) J[k]=0;
    output.valid=sphere_batch::evaluate(factor,values,output.value,J);
}

__global__ void normalRows(const int* rows,const int* columns,const double* values,const double* r,
    int residuals,int retained,int eliminated,double* pose,double* cross,double* diagonal,double* g,int* coupled) {
    const int row=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(row>=residuals) return;
    for(int i=rows[row];i<rows[row+1];++i) {
        const int column=columns[i];
        const double v=values[i];
        if(v==0) continue;
        atomicAdd(g+column,v*r[row]);
        for(int j=rows[row];j<rows[row+1];++j) {
            const int other=columns[j];
            const double w=values[j];
            if(w==0) continue;
            if(column<retained && other<retained) atomicAdd(pose+column*retained+other,v*w);
            else if(column>=retained && other<retained) atomicAdd(cross+(column-retained)*retained+other,v*w);
            else if(column>=retained && other>=retained) {
                if(column==other) atomicAdd(diagonal+column-retained,v*w);
                else if(eliminated>1) atomicExch(coupled,1);
            }
        }
    }
}

__global__ void projectVisualRows(const VisualBatchInput* factors,const DirectVisualLayout* layout,
    const double* plus,const VisualBatchResidual* residuals,const double* ambient,const int* rows,
    int count,double* tangent,double* r) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count) return;
    const auto& residual=residuals[i];
    const auto& factor=factors[i];
    const auto& blocks=layout[i];
    const double squared=residual.value[0]*residual.value[0]+residual.value[1]*residual.value[1];
    // Huber has nonpositive curvature. Ceres scales both rows by sqrt(rho').
    const double weight=blocks.huberScale>0 && squared>blocks.huberScale*blocks.huberScale
        ? sqrt(blocks.huberScale/sqrt(squared)):1.0;
    r[2*i]=weight*residual.value[0];
    r[2*i+1]=weight*residual.value[1];
    int input=factor.jacobianOffset,output=0;
    for(int block=0;block<factor.blocks;++block) {
        if(!(factor.derivativeMask & (1u<<block))) continue;
        const int a=factor.sizes[block],t=blocks.width[block];
        for(int row=0;row<2;++row) for(int col=0;col<t;++col) {
            double value=0;
            for(int k=0;k<a;++k) value+=ambient[input+row*a+k]*plus[blocks.plus[block]+k*t+col];
            tangent[rows[2*i+row]+output+col]=weight*value;
        }
        input+=2*a;
        output+=t;
    }
}

__global__ void accumulateVisualPairs(const int* rows,const int* columns,const double* values,const double* r,
    int count,int nr,double* pose,double* cross,double* diagonal,double* g) {
    const int factor=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(factor>=count) return;
    const int row=2*factor,begin=rows[row],end=rows[row+1],width=end-begin;
    // The two residuals share the same column pattern. Sum both products
    // before touching global memory and store only the upper state triangle.
    for(int i=begin;i<end;++i) {
        const int column=columns[i];
        const double a=values[i],b=values[i+width];
        if(a==0 && b==0) continue;
        atomicAdd(g+column,a*r[row]+b*r[row+1]);
        if(column>=nr) atomicAdd(diagonal+column-nr,a*a+b*b);
        for(int j=begin;j<end;++j) {
            const int other=columns[j];
            if(other>=nr || (column<nr && other<column)) continue;
            const double value=a*values[j]+b*values[j+width];
            if(value==0) continue;
            if(column<nr) atomicAdd(pose+column*nr+other,value);
            else atomicAdd(cross+(column-nr)*nr+other,value);
        }
    }
}

__global__ void reduceSchur(const double* pose,const double* cross,const double* diagonal,const double* g,
    int nr,int ne,double* outputH,double* outputG) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=nr*(nr+1)) return;
    const int j=i%nr;
    const bool gradient=i>=nr*nr;
    const int a=gradient?j:i/nr;
    double result=gradient?g[a]:pose[a*nr+j]+(a==j?1e-3*fmax(pose[a*nr+a],1e-6):0);
    for(int k=0;k<ne;++k) {
        const double denominator=diagonal[k]+1e-3*fmax(diagonal[k],1e-6);
        result-=cross[k*nr+a]*(gradient?g[nr+k]:cross[k*nr+j])/denominator;
    }
    if(gradient) outputG[j]=result; else outputH[i]=result;
}

__device__ void sumBlock(double value,double* total) {
    __shared__ double values[128];
    const int thread=int(threadIdx.x);
    values[thread]=value;
    __syncthreads();
    for(int stride=64;stride>0;stride/=2) {
        if(thread<stride) values[thread]+=values[thread+stride];
        __syncthreads();
    }
    if(thread==0) atomicAdd(total,values[0]);
}

__global__ void visualCost(const VisualBatchResidual* residuals,const DirectVisualLayout* layout,
    int count,double* cost,int* invalid) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    double contribution=0;
    if(i<count) {
        const auto& r=residuals[i];
        const double squared=r.value[0]*r.value[0]+r.value[1]*r.value[1], scale=layout[i].huberScale;
        if(!r.valid || !isfinite(squared)) atomicExch(invalid,1);
        else contribution=scale>0 && squared>scale*scale?scale*(sqrt(squared)-0.5*scale):0.5*squared;
    }
    sumBlock(contribution,cost);
}

__global__ void initializeReduced(const double* pose,const double* gradient,int n,double* out) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n*n) out[i]=pose[i];
    else if(i<n*(n+1)) out[i]=gradient[i-n*n];
    else if(i<n*(n+2)) out[i]=pose[(i-n*(n+1))*(n+1)];
}

__device__ double dampedPoint(double diagonal,double lambda) {
    return diagonal+lambda*fmax(diagonal,1e-6);
}

__global__ void subtractPointSchur(const double* cross,const double* diagonal,const double* g,
    const int* offsets,const int* support,int nr,double lambda,double* reduced) {
    const int point=int(blockIdx.x),begin=offsets[point],width=offsets[point+1]-begin;
    const double inverse=1/dampedPoint(diagonal[point],lambda);
    const auto* c=cross+point*nr;
    for(int i=int(threadIdx.x);i<width*(width+1);i+=int(blockDim.x)) {
        const int a=support[begin+i%width];
        if(i<width*width) {
            const int b=support[begin+i/width];
            if(a<=b) atomicAdd(reduced+a*nr+b,-inverse*c[a]*c[b]);
        } else atomicAdd(reduced+nr*nr+a,-inverse*c[a]*g[nr+point]);
    }
}

__global__ void prepareStateSolve(const double* retainedH,const double* retainedG,int n,double lambda,
    double* reduced,double* delta) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n*n) {
        reduced[i]+=retainedH[i];
        if(i%(n+1)==0) reduced[i]+=lambda*fmax(reduced[n*(n+1)+i/(n+1)]+retainedH[i],1e-6);
    } else if(i<n*(n+1)) {
        delta[i-n*n]=-(reduced[i]+retainedG[i-n*n]);
    }
}

__global__ void clampStateStep(double* delta,int n,const double* lower,const double* upper) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n && isfinite(delta[i])) delta[i]=fmax(lower[i],fmin(upper[i],delta[i]));
}

__global__ void statePrediction(const double* pose,const double* g,const double* retainedH,const double* retainedG,
    const double* delta,int n,double* prediction) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    double contribution=0;
    if(i<n) {
        double hd=0;
        for(int j=0;j<n;++j) hd+=(pose[min(i,j)*n+max(i,j)]+retainedH[i*n+j])*delta[j];
        contribution=-delta[i]*(g[i]+retainedG[i]+0.5*hd);
    }
    sumBlock(contribution,prediction);
}

__global__ void backSubstitute(const double* cross,const double* diagonal,const double* g,
    const int* offsets,const int* support,int nr,int ne,double lambda,const double* lower,const double* upper,
    double* delta,double* prediction) {
    const int k=int(blockIdx.x*blockDim.x+threadIdx.x);
    double contribution=0;
    if(k<ne) {
        double c=0;
        for(int j=offsets[k];j<offsets[k+1];++j) c+=cross[k*nr+support[j]]*delta[support[j]];
        const int i=nr+k;
        const double d=-(g[i]+c)/dampedPoint(diagonal[k],lambda);
        delta[i]=isfinite(d)?fmax(lower[i],fmin(upper[i],d)):d;
        contribution=-delta[i]*(g[i]+c+0.5*diagonal[k]*delta[i]);
    }
    sumBlock(contribution,prediction);
}

__global__ void evaluateImuFactors(const GpuImuFactor* factors,int count,const double* values,int n,
    double* residual,double* jacobian,int* invalid) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count) return;
    const auto& f=factors[i];
    if(!retained_math::evaluateImu(f,values,n,residual+f.firstRow,
        jacobian?jacobian+f.firstRow*n:nullptr)) atomicExch(invalid,1);
}

__global__ void priorCharts(const GpuPriorBlock* blocks,int count,const double* data,const double* values,
    double* chart,double* derivatives) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count) return;
    const auto& b=blocks[i];
    const auto& state=b.state;
    const double* x=values+state.offset;
    const double* ref=data+b.reference;
    double* d=chart+b.chartOffset;
    if(state.chart==GpuChart::Euclidean) {
        for(int c=0;c<b.tangentSize;++c) d[c]=x[c]-ref[c];
    } else retained_math::poseMinus(x,ref,d);
    if(!derivatives || !state.width) return;
    double* D=derivatives+9*b.chartOffset;
    if(state.chart==GpuChart::Euclidean) {
        for(int r=0;r<b.tangentSize;++r) for(int c=0;c<state.width;++c) D[9*r+c]=double(r==c);
        return;
    }
    // Differentiate only the current chart, just as the host FEJ prior does.
    constexpr double h=1e-7;
    double step[6]{},plus[7],minus[7],dp[6],dm[6];
    const bool rotationOnly=state.chart==GpuChart::RotationOnly;
    for(int c=0;c<state.width;++c) {
        step[c]=h; retained_math::posePlus(x,step,plus,rotationOnly);
        step[c]=-h; retained_math::posePlus(x,step,minus,rotationOnly);
        step[c]=0;
        retained_math::poseMinus(plus,ref,dp);
        retained_math::poseMinus(minus,ref,dm);
        for(int r=0;r<6;++r) D[9*r+c]=(dp[r]-dm[r])/(2*h);
    }
}

__global__ void evaluatePriorRows(const GpuPriorFactor* factors,const GpuPriorBlock* blocks,const double* data,
    const double* chart,const double* derivatives,int n,double* residual,double* jacobian) {
    const auto& f=factors[blockIdx.x];
    for(int r=int(threadIdx.x);r<f.rows;r+=int(blockDim.x)) {
        const double* A=data+f.matrix+r*f.columns;
        double value=data[f.residual+r];
        for(int c=0;c<f.columns;++c) value+=A[c]*chart[f.chartOffset+c];
        residual[f.firstRow+r]=value;
        if(!jacobian) continue;
        for(int b=0;b<f.blockCount;++b) {
            const auto& block=blocks[f.firstBlock+b];
            const double* D=derivatives+9*block.chartOffset;
            const double* a=A+block.chartOffset-f.chartOffset;
            for(int c=0;c<block.state.width;++c) {
                double d=0;
                for(int j=0;j<block.tangentSize;++j) d+=a[j]*D[9*j+c];
                jacobian[(f.firstRow+r)*n+block.state.column+c]=d;
            }
        }
    }
}

__global__ void retainedCost(const double* residual,int rows,double* cost,int* invalid) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    double contribution=0;
    if(i<rows) {
        contribution=0.5*residual[i]*residual[i];
        if(!isfinite(contribution)) atomicExch(invalid,1);
    }
    sumBlock(contribution,cost);
}

__global__ void accumulateRetained(const double* J,const double* r,int rows,int n,double* H,double* g) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=n*(n+1)) return;
    const int column=i%n,other=i/n;
    if(other<n && other>column) return;
    double product=0;
    for(int k=0;k<rows;++k) product+=J[k*n+column]*(other==n?r[k]:J[k*n+other]);
    if(other==n) g[column]+=product;
    else H[i]+=product;
}

__global__ void prepareResidentVariables(const GpuVariable* variables,int count,const double* values,
    const double* absoluteLower,const double* absoluteUpper,double* lower,double* upper,double* lifts) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count) return;
    const auto& v=variables[i]; const auto& b=v.state;
    const auto* x=values+b.offset;
    for(int j=0;j<b.width;++j) {
        lower[b.column+j]=absoluteLower[b.column+j]-x[j];
        upper[b.column+j]=absoluteUpper[b.column+j]-x[j];
    }
    auto* J=lifts+v.lift;
    for(int k=0;k<b.width*b.size;++k) J[k]=0;
    if(b.chart==GpuChart::Euclidean) {
        for(int j=0;j<b.width;++j) J[j*b.width+j]=1;
        return;
    }
    const int rotation=b.chart==GpuChart::LeftSE3?3:0;
    if(rotation) {
        const retained_math::M3 cross=-retained_math::skewMatrix(Eigen::Map<const retained_math::V3>(x));
        for(int r=0;r<3;++r) {
            J[r*b.width+r]=1;
            for(int c=0;c<3;++c) J[r*b.width+3+c]=cross(r,c);
        }
    }
    const Eigen::Map<const Eigen::Quaterniond> q(x+3);
    const retained_math::M3 top=0.5*(q.w()*retained_math::M3::Identity()-retained_math::skewMatrix(q.vec()));
    for(int c=0;c<3;++c) {
        for(int r=0;r<3;++r) J[(r+3)*b.width+rotation+c]=top(r,c);
        J[6*b.width+rotation+c]=-0.5*q.vec()[c];
    }
}

__global__ void residentTrial(const GpuVariable* variables,int count,const double* accepted,const double* delta,
    const double* lower,const double* upper,double* candidate,double* maximum,int* invalid) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count) return;
    const auto& b=variables[i].state;
    const auto* x=accepted+b.offset; const auto* d=delta+b.column;
    auto* y=candidate+b.offset;
    double norm=0;
    for(int k=0;k<b.width;++k) {
        if(!isfinite(d[k])) { atomicExch(invalid,1); return; }
        norm=fmax(norm,fabs(d[k]));
    }
    atomicMax(reinterpret_cast<unsigned long long*>(maximum),static_cast<unsigned long long>(__double_as_longlong(norm)));
    if(b.chart==GpuChart::Euclidean) {
        for(int k=0;k<b.width;++k) y[k]=fmax(lower[b.column+k],fmin(upper[b.column+k],x[k]+d[k]));
    } else retained_math::posePlus(x,d,y,b.chart==GpuChart::RotationOnly);
    for(int k=0;k<b.size;++k) if(!isfinite(y[k])) atomicExch(invalid,1);
}
}

struct VisualCudaBatch::Impl {
    cudaStream_t stream=nullptr;
    cuda::Array<VisualBatchInput> inputs;
    cuda::Array<double> parameters,jacobians;
    cuda::Array<VisualBatchResidual> residuals;
    cuda::Array<double,true> hostParameters;
    cuda::Array<double,true> hostJacobians;
    cuda::Array<VisualBatchResidual,true> hostResiduals;
    // Keep the normal equations, reduced system and point cross blocks on the device.
    cusolverDnHandle_t solver=nullptr;
    cuda::Array<DirectVisualLayout> layout;
    cuda::Array<int> rows,columns,offsets,support,invalid;
    cuda::Array<double> lifts,tangent,whiteResiduals,pose,cross,diagonal,gradient,reduced,metric;
    cuda::Array<double> retainedH,retainedG,lower,upper,step,solverWork;
    cuda::Array<GpuImuFactor> imuFactors;
    cuda::Array<GpuPriorFactor> priorFactors;
    cuda::Array<GpuPriorBlock> priorBlocks;
    cuda::Array<double> priorData,retainedResidual,retainedJacobian,chart,chartDerivative;
    cuda::Array<GpuImuFactor,true> hostImuFactors;
    cuda::Array<GpuPriorFactor,true> hostPriorFactors;
    cuda::Array<GpuPriorBlock,true> hostPriorBlocks;
    cuda::Array<double,true> hostPriorData;
    cuda::Array<GpuVariable> variables;
    cuda::Array<GpuVariable,true> hostVariables;
    cuda::Array<double> candidate,absoluteLower,absoluteUpper;
    cuda::Array<int> solverInfo;
    cuda::Array<VisualBatchInput,true> hostInputs;
    cuda::Array<DirectVisualLayout,true> hostLayout;
    cuda::Array<int,true> hostRows,hostColumns,hostOffsets,hostSupport;
    cuda::Array<double,true> hostLifts,hostRetainedH,hostRetainedG,hostLower,hostUpper,hostStep,hostMetrics;
    cuda::Array<int,true> hostInvalid;
    int retained=0,eliminated=0,visualRows=0,solverWorkSize=0,solverDimension=0;
    int imuCount=0,priorCount=0,priorBlockCount=0,retainedRows=0,residentCount=0;
    std::size_t parameterSize=0;
    bool pending=false;
    bool finishPending() {
        if(!pending) return true;
        pending=false;
        return cudaStreamSynchronize(stream)==cudaSuccess;
    }
    bool evaluateRetained(const double* values,bool derivatives,int metricIndex,int invalidIndex);
    bool launchVisualLinearization();
    bool launchTrial(const double* values,int metricIndex);
    Impl() { cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking); }
    ~Impl() {
        if(stream) {
            cudaStreamSynchronize(stream);
            if(solver) cusolverDnDestroy(solver);
            cudaStreamDestroy(stream);
        }
    }
};

VisualCudaBatch::VisualCudaBatch():impl_(std::make_unique<Impl>()) {}
VisualCudaBatch::~VisualCudaBatch()=default;

const char* takeVisualCudaRuntimeError() {
    const auto status=cudaGetLastError();
    return status==cudaSuccess?nullptr:cudaGetErrorString(status);
}

bool VisualCudaBatch::evaluate(const std::vector<VisualBatchInput>& inputs,const std::vector<double>& values,
    bool topologyChanged,bool derivatives,std::size_t jacobianSize,VisualBatchView& result) {
    result={};
    auto& s=*impl_;
    const auto count=inputs.size();
    if(!s.stream || !s.finishPending() || !count || count>1000000 || !s.inputs.reserve(count) ||
        !s.parameters.reserve(values.size()) || !s.hostParameters.reserve(values.size()) ||
        !s.residuals.reserve(count) || !s.hostResiduals.reserve(count) ||
        (derivatives && (!s.jacobians.reserve(jacobianSize) || !s.hostJacobians.reserve(jacobianSize)))) return false;
    const auto copy=[&](void* to,const void* from,std::size_t bytes,cudaMemcpyKind kind) {
        return cudaMemcpyAsync(to,from,bytes,kind,s.stream)==cudaSuccess;
    };
    std::copy(values.begin(),values.end(),s.hostParameters.data());
    bool ok=(!topologyChanged || copy(s.inputs.data(),inputs.data(),count*sizeof(VisualBatchInput),cudaMemcpyHostToDevice))
        && copy(s.parameters.data(),s.hostParameters.data(),values.size()*sizeof(double),cudaMemcpyHostToDevice);
    if(ok) {
        evaluateVisual<<<(count+127)/128,128,0,s.stream>>>(s.inputs.data(),int(count),s.parameters.data(),
            s.residuals.data(),derivatives?s.jacobians.data():nullptr);
        ok=cudaGetLastError()==cudaSuccess &&
            copy(s.hostResiduals.data(),s.residuals.data(),count*sizeof(VisualBatchResidual),cudaMemcpyDeviceToHost);
        if(ok && derivatives && jacobianSize) ok=copy(s.hostJacobians.data(),s.jacobians.data(),
            jacobianSize*sizeof(double),cudaMemcpyDeviceToHost);
    }
    if(cudaStreamSynchronize(s.stream)!=cudaSuccess || !ok) return false;
    // Read the synchronized pinned results in place until the next callback.
    result={s.hostResiduals.data(),derivatives && jacobianSize?s.hostJacobians.data():nullptr};
    return true;
}

bool VisualCudaBatch::linearizeSchur(const std::vector<VisualBatchInput>& factors,const std::vector<double>& values,
    std::size_t jacobianSize,const std::vector<DirectVisualLayout>& layout,const std::vector<double>& plus,
    int nr,int ne,bool newTopology) {
    auto& s=*impl_;
    const auto count=factors.size();
    const int n=nr+ne,rowCount=int(2*count);
    if(!s.stream || !s.finishPending() || count>1000000 || nr<=0 || ne<0 || n>4000 ||
        factors.size()!=layout.size()) return false;
    const auto reserve=[](auto& buffer,std::size_t needed) {
        std::size_t capacity=1;
        while(capacity<needed) capacity*=2;
        return !needed || buffer.reserve(capacity);
    };
    const auto prepare=[&](auto& buffer,auto& pinned,std::size_t size) {
        return reserve(buffer,size) && reserve(pinned,size);
    };
    std::vector<int> rows,columns,offsets,support;
    if(newTopology) {
        s.residentCount=0;
        rows.push_back(0); offsets.push_back(0);
        std::vector<std::vector<int>> pointColumns(ne);
        std::vector<int> factorColumns;
        for(std::size_t i=0;i<count;++i) {
            const auto& factor=factors[i];
            const auto& blocks=layout[i];
            int point=-1;
            factorColumns.clear();
            for(int b=0;b<factor.blocks;++b) {
                const int width=blocks.width[b],column=blocks.column[b];
                if(!(factor.derivativeMask & (1u<<b))) {
                    if(width) return false;
                    continue;
                }
                if(width<1 || column<0 || column+width>n || (column<nr && column+width>nr) ||
                    blocks.plus[b]<0 || std::size_t(blocks.plus[b]+factor.sizes[b]*width)>plus.size()) return false;
                if(column>=nr) {
                    if(width!=1 || point>=0) return false;
                    point=column-nr;
                }
                for(int j=0;j<width;++j) factorColumns.push_back(column+j);
            }
            if(point>=0) for(int column:factorColumns) if(column<nr) pointColumns[point].push_back(column);
            for(int row=0;row<2;++row) {
                columns.insert(columns.end(),factorColumns.begin(),factorColumns.end());
                rows.push_back(int(columns.size()));
            }
        }
        for(auto& columnsForPoint:pointColumns) {
            std::sort(columnsForPoint.begin(),columnsForPoint.end());
            support.insert(support.end(),columnsForPoint.begin(),std::unique(columnsForPoint.begin(),columnsForPoint.end()));
            offsets.push_back(int(support.size()));
        }
        if(!prepare(s.rows,s.hostRows,rows.size()) || !prepare(s.columns,s.hostColumns,columns.size()) ||
            !prepare(s.layout,s.hostLayout,layout.size()) || !prepare(s.offsets,s.hostOffsets,offsets.size()) ||
            !prepare(s.support,s.hostSupport,support.size()) || !prepare(s.inputs,s.hostInputs,count) ||
            !reserve(s.tangent,columns.size())) return false;
        s.retained=nr; s.eliminated=ne; s.visualRows=rowCount;
    } else if(nr!=s.retained || ne!=s.eliminated || rowCount!=s.visualRows) return false;
    if(!prepare(s.parameters,s.hostParameters,values.size()) || !prepare(s.lifts,s.hostLifts,plus.size()) ||
        !reserve(s.jacobians,jacobianSize) || !reserve(s.residuals,count) ||
        !reserve(s.pose,nr*nr) || !reserve(s.cross,nr*ne) || !reserve(s.diagonal,ne) ||
        !reserve(s.gradient,n) || !reserve(s.whiteResiduals,rowCount) || !s.invalid.reserve(2) ||
        !s.metric.reserve(4) || !reserve(s.reduced,nr*(nr+2)) || !s.hostInvalid.reserve(4)) return false;
    const auto upload=[&](auto& buffer,auto& pinned,const auto& data) {
        if(data.empty()) return true;
        std::copy(data.begin(),data.end(),pinned.data());
        return cudaMemcpyAsync(buffer.data(),pinned.data(),data.size()*sizeof(data[0]),cudaMemcpyHostToDevice,s.stream)==cudaSuccess;
    };
    s.pending=true;
    s.parameterSize=values.size();
    // Pinned copies keep topology and values alive through reduceOnline(),
    // without a pageable staging wait before the CPU starts its own factors.
    const bool ok=(!newTopology || (upload(s.rows,s.hostRows,rows) && upload(s.columns,s.hostColumns,columns) &&
        upload(s.layout,s.hostLayout,layout) && upload(s.offsets,s.hostOffsets,offsets) &&
        upload(s.support,s.hostSupport,support) && upload(s.inputs,s.hostInputs,factors))) &&
        upload(s.lifts,s.hostLifts,plus) && upload(s.parameters,s.hostParameters,values) && s.launchVisualLinearization();
    if(!ok) s.finishPending();
    return ok;
}

bool VisualCudaBatch::Impl::launchVisualLinearization() {
    const auto zero=[&](auto& buffer,std::size_t size) {
        return !size || cudaMemsetAsync(buffer.data(),0,size*sizeof(*buffer.data()),stream)==cudaSuccess;
    };
    const int nr=retained,ne=eliminated,count=visualRows/2,blocks=(count+127)/128;
    if(!zero(pose,nr*nr) || !zero(cross,nr*ne) || !zero(diagonal,ne) || !zero(gradient,nr+ne) ||
        !zero(metric,1) || !zero(invalid,1)) return false;
    if(!count) return true;
    evaluateVisual<<<blocks,128,0,stream>>>(inputs.data(),count,parameters.data(),residuals.data(),jacobians.data());
    projectVisualRows<<<blocks,128,0,stream>>>(inputs.data(),layout.data(),lifts.data(),residuals.data(),jacobians.data(),
        rows.data(),count,tangent.data(),whiteResiduals.data());
    accumulateVisualPairs<<<blocks,128,0,stream>>>(rows.data(),columns.data(),tangent.data(),whiteResiduals.data(),count,nr,
        pose.data(),cross.data(),diagonal.data(),gradient.data());
    visualCost<<<blocks,128,0,stream>>>(residuals.data(),layout.data(),count,metric.data(),invalid.data());
    return cudaGetLastError()==cudaSuccess;
}

bool VisualCudaBatch::solveOnline(double lambda,const double* retainedH,const double* retainedG,
    const std::vector<double>& lower,const std::vector<double>& upper,bool newPoint,VisualSchurStep& out) {
    out={};
    auto& s=*impl_;
    const int nr=s.retained,ne=s.eliminated,n=nr+ne,size=nr*(nr+2);
    if(nr<=0 || !std::isfinite(lambda) || lambda<=0 || lower.size()!=std::size_t(n) || upper.size()!=lower.size()) return false;
    if(!s.solver) {
        if(cusolverDnCreate(&s.solver)!=CUSOLVER_STATUS_SUCCESS) return false;
        if(cusolverDnSetStream(s.solver,s.stream)!=CUSOLVER_STATUS_SUCCESS) {
            cusolverDnDestroy(s.solver); s.solver=nullptr; return false;
        }
    }
    if(s.solverDimension!=nr) {
        int work=0;
        if(cusolverDnDpotrf_bufferSize(s.solver,CUBLAS_FILL_MODE_LOWER,nr,s.reduced.data(),nr,&work)!=CUSOLVER_STATUS_SUCCESS ||
            work<1 || !s.solverWork.reserve(work)) return false;
        s.solverDimension=nr; s.solverWorkSize=work;
    }
    if(!s.retainedH.reserve(nr*nr) || !s.retainedG.reserve(nr) || !s.lower.reserve(n) || !s.upper.reserve(n) ||
        !s.step.reserve(n) || !s.solverInfo.reserve(2) || !s.hostRetainedH.reserve(nr*nr) ||
        !s.hostRetainedG.reserve(nr) || !s.hostLower.reserve(n) || !s.hostUpper.reserve(n) ||
        !s.hostStep.reserve(n) || !s.hostMetrics.reserve(4)) return false;
    const auto upload=[&](auto& device,auto& host,const double* data,int count) {
        std::copy_n(data,count,host.data());
        return cudaMemcpyAsync(device.data(),host.data(),count*sizeof(double),cudaMemcpyHostToDevice,s.stream)==cudaSuccess;
    };
    s.pending=true;
    if(newPoint && (!(retainedH?upload(s.retainedH,s.hostRetainedH,retainedH,nr*nr):
            cudaMemsetAsync(s.retainedH.data(),0,nr*nr*sizeof(double),s.stream)==cudaSuccess) ||
        !(retainedG?upload(s.retainedG,s.hostRetainedG,retainedG,nr):
            cudaMemsetAsync(s.retainedG.data(),0,nr*sizeof(double),s.stream)==cudaSuccess) ||
        (!s.residentCount && (!upload(s.lower,s.hostLower,lower.data(),n) || !upload(s.upper,s.hostUpper,upper.data(),n))))) {
        s.finishPending(); return false;
    }
    initializeReduced<<<(size+127)/128,128,0,s.stream>>>(s.pose.data(),s.gradient.data(),nr,s.reduced.data());
    if(ne) subtractPointSchur<<<ne,128,0,s.stream>>>(s.cross.data(),s.diagonal.data(),s.gradient.data(),
        s.offsets.data(),s.support.data(),nr,lambda,s.reduced.data());
    prepareStateSolve<<<(nr*(nr+1)+127)/128,128,0,s.stream>>>(s.retainedH.data(),s.retainedG.data(),nr,lambda,
        s.reduced.data(),s.step.data());
    // Row-major upper triangle is the column-major lower triangle expected by cuSOLVER.
    if(cudaGetLastError()!=cudaSuccess ||
        cusolverDnDpotrf(s.solver,CUBLAS_FILL_MODE_LOWER,nr,s.reduced.data(),nr,s.solverWork.data(),
            s.solverWorkSize,s.solverInfo.data())!=CUSOLVER_STATUS_SUCCESS ||
        cusolverDnDpotrs(s.solver,CUBLAS_FILL_MODE_LOWER,nr,1,s.reduced.data(),nr,s.step.data(),nr,
            s.solverInfo.data()+1)!=CUSOLVER_STATUS_SUCCESS ||
        cudaMemsetAsync(s.metric.data()+1,0,sizeof(double),s.stream)!=cudaSuccess) { s.finishPending(); return false; }
    clampStateStep<<<(nr+127)/128,128,0,s.stream>>>(s.step.data(),nr,s.lower.data(),s.upper.data());
    statePrediction<<<(nr+127)/128,128,0,s.stream>>>(s.pose.data(),s.gradient.data(),s.retainedH.data(),
        s.retainedG.data(),s.step.data(),nr,s.metric.data()+1);
    if(ne) backSubstitute<<<(ne+127)/128,128,0,s.stream>>>(s.cross.data(),s.diagonal.data(),s.gradient.data(),
        s.offsets.data(),s.support.data(),nr,ne,lambda,s.lower.data(),s.upper.data(),s.step.data(),s.metric.data()+1);
    const auto download=[&](auto* host,const auto* device,std::size_t count) {
        return cudaMemcpyAsync(host,device,count*sizeof(*device),cudaMemcpyDeviceToHost,s.stream)==cudaSuccess;
    };
    bool ok=cudaGetLastError()==cudaSuccess;
    if(ok && s.residentCount) {
        ok=cudaMemsetAsync(s.metric.data()+3,0,sizeof(double),s.stream)==cudaSuccess &&
            cudaMemsetAsync(s.invalid.data()+1,0,sizeof(int),s.stream)==cudaSuccess &&
            cudaMemcpyAsync(s.candidate.data(),s.parameters.data(),s.parameterSize*sizeof(double),cudaMemcpyDeviceToDevice,s.stream)==cudaSuccess;
        if(ok) {
            residentTrial<<<(s.residentCount+127)/128,128,0,s.stream>>>(s.variables.data(),s.residentCount,s.parameters.data(),
                s.step.data(),s.absoluteLower.data(),s.absoluteUpper.data(),s.candidate.data(),s.metric.data()+3,s.invalid.data()+1);
            ok=s.launchTrial(s.candidate.data(),2);
        }
    } else if(ok) ok=download(s.hostStep.data(),s.step.data(),n);
    ok=ok && download(s.hostMetrics.data(),s.metric.data(),s.residentCount?4:2) &&
        download(s.hostInvalid.data(),s.invalid.data(),s.residentCount?2:1) && download(s.hostInvalid.data()+2,s.solverInfo.data(),2);
    if(!s.finishPending() || !ok || s.hostInvalid.data()[0] || s.hostInvalid.data()[2] || s.hostInvalid.data()[3]) return false;
    out.cost=s.hostMetrics.data()[0]; out.predictedReduction=s.hostMetrics.data()[1];
    if(s.residentCount) {
        out.maxIncrement=s.hostMetrics.data()[3]; out.trialCost=s.hostMetrics.data()[2];
        out.trialValid=!s.hostInvalid.data()[1] && std::isfinite(out.trialCost);
    } else out.increment=s.hostStep.data();
    return true;
}

bool VisualCudaBatch::configureRetained(const GpuRetainedFactors& problem) {
    auto& s=*impl_;
    s.imuCount=int(problem.imu.size()); s.priorCount=int(problem.priors.size());
    s.priorBlockCount=int(problem.priorBlocks.size()); s.retainedRows=problem.rows;
    const auto upload=[&](auto& device,auto& host,const auto& data) {
        if(data.empty()) return true;
        if(!device.reserve(data.size()) || !host.reserve(data.size())) return false;
        std::copy(data.begin(),data.end(),host.data());
        return cudaMemcpyAsync(device.data(),host.data(),data.size()*sizeof(data[0]),cudaMemcpyHostToDevice,s.stream)==cudaSuccess;
    };
    if(!problem.rows) return true;
    if(!s.retainedResidual.reserve(problem.rows) || !s.retainedJacobian.reserve(std::size_t(problem.rows)*s.retained) ||
        !s.chart.reserve(problem.chartSize) || !s.chartDerivative.reserve(9*problem.chartSize)) return false;
    s.pending=true;
    const bool ok=upload(s.imuFactors,s.hostImuFactors,problem.imu) && upload(s.priorFactors,s.hostPriorFactors,problem.priors) &&
        upload(s.priorBlocks,s.hostPriorBlocks,problem.priorBlocks) && upload(s.priorData,s.hostPriorData,problem.priorData);
    if(!ok) s.finishPending();
    return ok;
}

bool VisualCudaBatch::Impl::evaluateRetained(const double* values,bool derivatives,int metricIndex,int invalidIndex) {
    if(!retainedRows) return true;
    auto* J=derivatives?retainedJacobian.data():nullptr;
    if(J && cudaMemsetAsync(J,0,std::size_t(retainedRows)*retained*sizeof(double),stream)!=cudaSuccess) return false;
    if(imuCount) evaluateImuFactors<<<(imuCount+31)/32,32,0,stream>>>(imuFactors.data(),imuCount,values,retained,
        retainedResidual.data(),J,invalid.data()+invalidIndex);
    if(priorCount) {
        priorCharts<<<(priorBlockCount+31)/32,32,0,stream>>>(priorBlocks.data(),priorBlockCount,priorData.data(),values,
            chart.data(),derivatives?chartDerivative.data():nullptr);
        evaluatePriorRows<<<priorCount,128,0,stream>>>(priorFactors.data(),priorBlocks.data(),priorData.data(),chart.data(),
            chartDerivative.data(),retained,retainedResidual.data(),J);
    }
    retainedCost<<<(retainedRows+127)/128,128,0,stream>>>(retainedResidual.data(),retainedRows,metric.data()+metricIndex,invalid.data()+invalidIndex);
    if(J) accumulateRetained<<<(retained*(retained+1)+127)/128,128,0,stream>>>(J,retainedResidual.data(),retainedRows,retained,
        pose.data(),gradient.data());
    return cudaGetLastError()==cudaSuccess;
}

bool VisualCudaBatch::linearizeRetained() {
    auto& s=*impl_;
    const bool ok=s.evaluateRetained(s.parameters.data(),true,0,0);
    if(!ok) s.finishPending();
    return ok;
}

bool VisualCudaBatch::readNormal(double* H,double* g,double& cost,double& residualSquaredNorm) {
    auto& s=*impl_;
    const int n=s.retained;
    if(n<1 || s.eliminated) return false;
    s.pending=true;
    bool ok=cudaMemsetAsync(s.metric.data()+1,0,sizeof(double),s.stream)==cudaSuccess;
    if(ok && s.visualRows) retainedCost<<<(s.visualRows+127)/128,128,0,s.stream>>>(s.whiteResiduals.data(),
        s.visualRows,s.metric.data()+1,s.invalid.data());
    if(ok && s.retainedRows) retainedCost<<<(s.retainedRows+127)/128,128,0,s.stream>>>(s.retainedResidual.data(),
        s.retainedRows,s.metric.data()+1,s.invalid.data());
    double metrics[2]{};
    int invalid=1;
    const auto download=[&](auto* target,const auto* source,std::size_t count) {
        return cudaMemcpyAsync(target,source,count*sizeof(*source),cudaMemcpyDeviceToHost,s.stream)==cudaSuccess;
    };
    ok=ok && cudaGetLastError()==cudaSuccess && download(H,s.pose.data(),std::size_t(n)*n) && download(g,s.gradient.data(),n)
        && download(metrics,s.metric.data(),2) && download(&invalid,s.invalid.data(),1);
    if(!s.finishPending() || !ok || invalid || !std::isfinite(metrics[0]) || !std::isfinite(metrics[1])) return false;
    cost=metrics[0]; residualSquaredNorm=2*metrics[1];
    return true;
}

bool VisualCudaBatch::configureResident(const std::vector<GpuVariable>& variables,const std::vector<double>& lower,const std::vector<double>& upper) {
    auto& s=*impl_;
    const int n=s.retained+s.eliminated;
    if(variables.empty() || lower.size()!=std::size_t(n) || upper.size()!=lower.size() ||
        !s.variables.reserve(variables.size()) || !s.hostVariables.reserve(variables.size()) || !s.candidate.reserve(s.parameterSize) ||
        !s.absoluteLower.reserve(n) || !s.absoluteUpper.reserve(n) || !s.lower.reserve(n) || !s.upper.reserve(n) ||
        !s.hostLower.reserve(n) || !s.hostUpper.reserve(n)) return false;
    const auto upload=[&](auto& device,auto& host,const auto& source) {
        std::copy(source.begin(),source.end(),host.data());
        return cudaMemcpyAsync(device.data(),host.data(),source.size()*sizeof(source[0]),cudaMemcpyHostToDevice,s.stream)==cudaSuccess;
    };
    s.pending=true;
    if(!upload(s.variables,s.hostVariables,variables) || !upload(s.absoluteLower,s.hostLower,lower) ||
        !upload(s.absoluteUpper,s.hostUpper,upper)) { s.finishPending(); return false; }
    s.residentCount=int(variables.size());
    prepareResidentVariables<<<(s.residentCount+127)/128,128,0,s.stream>>>(s.variables.data(),s.residentCount,s.parameters.data(),
        s.absoluteLower.data(),s.absoluteUpper.data(),s.lower.data(),s.upper.data(),s.lifts.data());
    return cudaGetLastError()==cudaSuccess;
}

bool VisualCudaBatch::linearizeResident() {
    auto& s=*impl_;
    if(!s.residentCount || !s.finishPending()) return false;
    s.pending=true;
    prepareResidentVariables<<<(s.residentCount+127)/128,128,0,s.stream>>>(s.variables.data(),s.residentCount,s.parameters.data(),
        s.absoluteLower.data(),s.absoluteUpper.data(),s.lower.data(),s.upper.data(),s.lifts.data());
    return cudaGetLastError()==cudaSuccess && s.launchVisualLinearization() && linearizeRetained();
}

void VisualCudaBatch::acceptResident() { impl_->parameters.swap(impl_->candidate); }

bool VisualCudaBatch::readResident(std::vector<double>& values) {
    auto& s=*impl_;
    if(!s.residentCount || !s.finishPending()) return false;
    s.pending=true;
    const bool ok=cudaMemcpyAsync(s.hostParameters.data(),s.parameters.data(),s.parameterSize*sizeof(double),
        cudaMemcpyDeviceToHost,s.stream)==cudaSuccess;
    if(!s.finishPending() || !ok) return false;
    values.assign(s.hostParameters.data(),s.hostParameters.data()+s.parameterSize);
    return true;
}

bool VisualCudaBatch::Impl::launchTrial(const double* values,int metricIndex) {
    const int count=visualRows/2,blocks=(count+127)/128;
    if(cudaMemsetAsync(metric.data()+metricIndex,0,sizeof(double),stream)!=cudaSuccess) return false;
    evaluateVisual<<<blocks,128,0,stream>>>(inputs.data(),count,values,residuals.data(),nullptr);
    visualCost<<<blocks,128,0,stream>>>(residuals.data(),layout.data(),count,metric.data()+metricIndex,invalid.data()+1);
    return cudaGetLastError()==cudaSuccess && evaluateRetained(values,false,metricIndex,1);
}

bool VisualCudaBatch::trialCost(const std::vector<double>& values,double& cost) {
    auto& s=*impl_;
    if(!s.finishPending() || s.visualRows<=0 || !s.parameters.reserve(values.size()) ||
        !s.hostParameters.reserve(values.size())) return false;
    std::copy(values.begin(),values.end(),s.hostParameters.data());
    s.pending=true;
    // The accepted normal blocks and their cost survive a rejected trial.
    bool ok=cudaMemcpyAsync(s.parameters.data(),s.hostParameters.data(),values.size()*sizeof(double),
        cudaMemcpyHostToDevice,s.stream)==cudaSuccess &&
        cudaMemsetAsync(s.invalid.data()+1,0,sizeof(int),s.stream)==cudaSuccess;
    if(ok) {
        ok=s.launchTrial(s.parameters.data(),1) &&
            cudaMemcpyAsync(s.hostMetrics.data()+1,s.metric.data()+1,sizeof(double),cudaMemcpyDeviceToHost,s.stream)==cudaSuccess &&
            cudaMemcpyAsync(s.hostInvalid.data(),s.invalid.data()+1,sizeof(int),cudaMemcpyDeviceToHost,s.stream)==cudaSuccess;
    }
    if(!s.finishPending() || !ok || *s.hostInvalid.data()) return false;
    cost=s.hostMetrics.data()[1];
    return std::isfinite(cost);
}

namespace {
bool remapColumns(const std::vector<int>& partition,std::vector<int>& remap) {
    const int dim=int(partition.size());
    remap.assign(dim,-1);
    for(int i=0;i<dim;++i) {
        const int index=partition[i];
        if(index<0 || index>=dim || remap[index]>=0) return false;
        remap[index]=i;
    }
    return true;
}
template<class Buffer,class Values> bool upload(Buffer& buffer,const Values& from) {
    return buffer.reserve(from.size()) && (from.empty() ||
        cudaMemcpy(buffer.data(),from.data(),from.size()*sizeof(from[0]),cudaMemcpyHostToDevice)==cudaSuccess);
}
bool formNormal(const int* rows,const int* columns,const double* jacobians,const double* r,
    int rowCount,const std::vector<int>& partition,int nr,VisualNormalSystem& out) {
    const int dim=int(partition.size()),ne=dim-nr;
    cuda::Array<int> coupled;
    cuda::Array<double> pose,cross,diagonal,g,reducedH,reducedG;
    const auto zero=[](auto& buffer,std::size_t count) {
        return !count || (buffer.reserve(count) && cudaMemset(buffer.data(),0,count*sizeof(*buffer.data()))==cudaSuccess);
    };
    const std::size_t poseSize=std::size_t(nr)*nr,crossSize=std::size_t(nr)*ne;
    if(!zero(pose,poseSize) || !zero(cross,crossSize) || !zero(diagonal,ne) ||
        !zero(g,dim) || !zero(coupled,1)) return false;
    normalRows<<<(rowCount+127)/128,128>>>(rows,columns,jacobians,r,
        rowCount,nr,ne,pose.data(),cross.data(),diagonal.data(),g.data(),coupled.data());
    if(cudaGetLastError()!=cudaSuccess) return false;
    const auto download=[](auto& to,const auto& buffer,std::size_t count) {
        to.resize(count);
        return !count || cudaMemcpy(to.data(),buffer.data(),count*sizeof(to[0]),cudaMemcpyDeviceToHost)==cudaSuccess;
    };
    int hasCoupling=0;
    if(cudaMemcpy(&hasCoupling,coupled.data(),sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess || hasCoupling) return false;
    if(ne) {
        if(!reducedH.reserve(poseSize) || !reducedG.reserve(nr)) return false;
        reduceSchur<<<(nr*(nr+1)+127)/128,128>>>(pose.data(),cross.data(),diagonal.data(),g.data(),nr,ne,
            reducedH.data(),reducedG.data());
        if(cudaGetLastError()!=cudaSuccess || !download(out.reducedH,reducedH,poseSize) ||
            !download(out.reducedG,reducedG,nr)) return false;
    }
    // The audit compares the full normal equations in Ceres' original column order.
    // Only the pose block, point/pose blocks and independent point diagonals were stored on device.
    std::vector<double> p,c,d,gradient;
    if(!download(p,pose,poseSize) || !download(c,cross,crossSize) || !download(d,diagonal,ne) ||
        !download(gradient,g,dim)) return false;
    out.H.assign(std::size_t(dim)*dim,0);
    out.g.resize(dim);
    const auto set=[&](int i,int j,double value) { out.H[std::size_t(partition[i])*dim+partition[j]]=value; };
    for(int i=0;i<dim;++i) out.g[partition[i]]=gradient[i];
    for(int i=0;i<nr;++i) for(int j=0;j<nr;++j) set(i,j,p[i*nr+j]);
    for(int k=0;k<ne;++k) {
        set(nr+k,nr+k,d[k]);
        for(int j=0;j<nr;++j) { const double value=c[k*nr+j]; set(nr+k,j,value); set(j,nr+k,value); }
    }
    return true;
}
}

bool cudaVisualNormalSystem(const std::vector<int>& rows,const std::vector<int>& columns,
    const std::vector<double>& values,const std::vector<double>& residuals,int dim,
    const std::vector<int>& retained,const std::vector<int>& eliminated,VisualNormalSystem& out) {
    if(dim<1 || residuals.empty() || dim>4000 || rows.size()!=residuals.size()+1 ||
        columns.size()!=values.size() || retained.empty() || retained.size()+eliminated.size()!=std::size_t(dim)) return false;
    std::vector<int> partition=retained;
    partition.insert(partition.end(),eliminated.begin(),eliminated.end());
    std::vector<int> remap,compactColumns;
    if(!remapColumns(partition,remap)) return false;
    compactColumns.reserve(columns.size());
    for(int column:columns) {
        if(column<0 || column>=dim) return false;
        compactColumns.push_back(remap[column]);
    }
    cuda::Array<int> rowBuffer,colBuffer;
    cuda::Array<double> jacobians,r;
    return upload(rowBuffer,rows) && upload(colBuffer,compactColumns) && upload(jacobians,values) && upload(r,residuals) &&
        formNormal(rowBuffer.data(),colBuffer.data(),jacobians.data(),r.data(),int(residuals.size()),
            partition,int(retained.size()),out);
}

bool VisualCudaBatch::directNormal(const std::vector<VisualBatchInput>& factors,
    const std::vector<DirectVisualLayout>& layout,const std::vector<double>& plus,
    const std::vector<int>& retained,const std::vector<int>& eliminated,VisualNormalSystem& out) {
    if(factors.empty() || factors.size()!=layout.size() || retained.empty() ||
        retained.size()+eliminated.size()>4000) return false;
    std::vector<int> partition=retained;
    partition.insert(partition.end(),eliminated.begin(),eliminated.end());
    std::vector<int> remap,rows{0},columns;
    if(!remapColumns(partition,remap)) return false;
    for(std::size_t i=0;i<factors.size();++i) {
        const auto& factor=factors[i];
        const auto& blocks=layout[i];
        for(int row=0;row<2;++row) {
            for(int b=0;b<factor.blocks;++b) {
                const int width=blocks.width[b],col=blocks.column[b];
                if(!(factor.derivativeMask & (1u<<b))) {
                    if(width) return false;
                    continue;
                }
                if(width<1 || col<0 || std::size_t(col+width)>remap.size() || blocks.plus[b]<0 ||
                    std::size_t(blocks.plus[b]+factor.sizes[b]*width)>plus.size()) return false;
                for(int j=0;j<width;++j) columns.push_back(remap[col+j]);
            }
            rows.push_back(int(columns.size()));
        }
    }
    cuda::Array<int> rowBuffer,colBuffer;
    cuda::Array<DirectVisualLayout> layouts;
    cuda::Array<double> lifts,jacobians,r;
    if(!upload(rowBuffer,rows) || !upload(colBuffer,columns) || !upload(layouts,layout) || !upload(lifts,plus) ||
        !jacobians.reserve(columns.size()) || !r.reserve(2*factors.size())) return false;
    projectVisualRows<<<(factors.size()+127)/128,128>>>(impl_->inputs.data(),layouts.data(),lifts.data(),
        impl_->residuals.data(),impl_->jacobians.data(),rowBuffer.data(),int(factors.size()),jacobians.data(),r.data());
    return cudaGetLastError()==cudaSuccess &&
        formNormal(rowBuffer.data(),colBuffer.data(),jacobians.data(),r.data(),int(2*factors.size()),
            partition,int(retained.size()),out);
}

struct MarginalCudaWorkspace::Impl {
    cudaStream_t stream=nullptr;
    cusolverDnHandle_t eigen=nullptr;
    cublasHandle_t blas=nullptr;
    cuda::Array<double> matrix,values,work,rhs,projected,output;
    cuda::Array<int> status;
    bool ready=false;
    Impl() {
        ready=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess &&
            cusolverDnCreate(&eigen)==CUSOLVER_STATUS_SUCCESS && cusolverDnSetStream(eigen,stream)==CUSOLVER_STATUS_SUCCESS &&
            cublasCreate(&blas)==CUBLAS_STATUS_SUCCESS && cublasSetStream(blas,stream)==CUBLAS_STATUS_SUCCESS;
    }
    ~Impl() {
        if(stream) cudaStreamSynchronize(stream);
        if(blas) cublasDestroy(blas);
        if(eigen) cusolverDnDestroy(eigen);
        if(stream) cudaStreamDestroy(stream);
    }
    bool copy(double* to,const double* from,std::size_t count,cudaMemcpyKind kind) const {
        return cudaMemcpyAsync(to,from,count*sizeof(double),kind,stream)==cudaSuccess;
    }
    bool finish(bool ok) const { return cudaStreamSynchronize(stream)==cudaSuccess && ok; }
    bool decompose(const double* H,int n) {
        if(!ready || n<1 || n>4000 || !matrix.reserve(std::size_t(n)*n) || !values.reserve(n) || !status.reserve(1)) return false;
        int lwork=0;
        if(cusolverDnDsyevd_bufferSize(eigen,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,matrix.data(),n,
            values.data(),&lwork)!=CUSOLVER_STATUS_SUCCESS || lwork<1 || !work.reserve(lwork)) return false;
        int info=-1;
        const bool ok=copy(matrix.data(),H,std::size_t(n)*n,cudaMemcpyHostToDevice) &&
            cusolverDnDsyevd(eigen,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,matrix.data(),n,
                values.data(),work.data(),lwork,status.data())==CUSOLVER_STATUS_SUCCESS &&
            cudaMemcpyAsync(&info,status.data(),sizeof(info),cudaMemcpyDeviceToHost,stream)==cudaSuccess;
        return finish(ok) && info==0;
    }
};

MarginalCudaWorkspace::MarginalCudaWorkspace()=default;
MarginalCudaWorkspace::~MarginalCudaWorkspace()=default;

namespace {
__global__ void applyEigenCutoff(double* rhs,const double* eigenvalues,int n,int count,double absCut,double relCut,int* status) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count) return;
    const double cutoff=fmax(absCut,relCut*fmax(0.,eigenvalues[n-1])),v=eigenvalues[i%n];
    if(v< -10*cutoff || !isfinite(v)) atomicExch(status,1);
    rhs[i]=v>cutoff?rhs[i]/v:0;
}
}

bool MarginalCudaWorkspace::eigen(const double* H,int n,double* eigenvalues,double* eigenvectors) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if(!impl_) impl_=std::make_unique<Impl>();
    auto& s=*impl_;
    if(!s.decompose(H,n)) return false;
    return s.finish(s.copy(eigenvalues,s.values.data(),n,cudaMemcpyDeviceToHost) &&
        s.copy(eigenvectors,s.matrix.data(),std::size_t(n)*n,cudaMemcpyDeviceToHost));
}

bool MarginalCudaWorkspace::psdSolve(const double* H,const double* rhs,int n,int columns,double absCut,double relCut,double* solution) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if(!impl_) impl_=std::make_unique<Impl>();
    auto& s=*impl_;
    if(columns<1 || columns>4001 || !s.decompose(H,n)) return false;
    const int count=n*columns;
    if(!s.rhs.reserve(count) || !s.projected.reserve(count) || !s.output.reserve(count)) return false;
    const double one=1,zero=0;
    if(!s.copy(s.rhs.data(),rhs,count,cudaMemcpyHostToDevice) ||
        cublasDgemm(s.blas,CUBLAS_OP_T,CUBLAS_OP_N,n,columns,n,&one,s.matrix.data(),n,s.rhs.data(),n,&zero,
            s.projected.data(),n)!=CUBLAS_STATUS_SUCCESS) return s.finish(false);
    applyEigenCutoff<<<(count+127)/128,128,0,s.stream>>>(s.projected.data(),s.values.data(),n,count,absCut,relCut,s.status.data());
    int invalid=1;
    const bool ok=cudaGetLastError()==cudaSuccess && cublasDgemm(s.blas,CUBLAS_OP_N,CUBLAS_OP_N,n,columns,n,&one,
        s.matrix.data(),n,s.projected.data(),n,&zero,s.output.data(),n)==CUBLAS_STATUS_SUCCESS &&
        s.copy(solution,s.output.data(),count,cudaMemcpyDeviceToHost) &&
        cudaMemcpyAsync(&invalid,s.status.data(),sizeof(invalid),cudaMemcpyDeviceToHost,s.stream)==cudaSuccess;
    return s.finish(ok) && !invalid;
}

bool MarginalCudaWorkspace::subtract(const double* coupling,const double* solution,int nm,int nr,double* H,double* g) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if(!impl_) impl_=std::make_unique<Impl>();
    auto& s=*impl_;
    if(!s.ready || nm<1 || nr<1 || nm>4000 || nr>4000) return false;
    if(!s.matrix.reserve(std::size_t(nm)*nr) || !s.rhs.reserve(std::size_t(nm)*(nr+1)) ||
        !s.output.reserve(std::size_t(nr)*(nr+1))) return false;
    const double minus=-1,one=1;
    const bool ok=s.copy(s.matrix.data(),coupling,std::size_t(nm)*nr,cudaMemcpyHostToDevice) &&
        s.copy(s.rhs.data(),solution,std::size_t(nm)*(nr+1),cudaMemcpyHostToDevice) &&
        s.copy(s.output.data(),H,std::size_t(nr)*nr,cudaMemcpyHostToDevice) &&
        s.copy(s.output.data()+nr*nr,g,nr,cudaMemcpyHostToDevice) &&
        cublasDgemm(s.blas,CUBLAS_OP_T,CUBLAS_OP_N,nr,nr+1,nm,&minus,s.matrix.data(),nm,s.rhs.data(),nm,
            &one,s.output.data(),nr)==CUBLAS_STATUS_SUCCESS &&
        s.copy(H,s.output.data(),std::size_t(nr)*nr,cudaMemcpyDeviceToHost) &&
        s.copy(g,s.output.data()+nr*nr,nr,cudaMemcpyDeviceToHost);
    return s.finish(ok);
}
}
