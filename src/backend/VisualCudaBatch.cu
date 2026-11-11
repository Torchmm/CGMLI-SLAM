#include "VisualSphereBatchMath.hpp"
#include <common/CudaArray.hpp>
#include <algorithm>

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

__global__ void packRangeSystem(const double* cross,const double* diagonal,const double* g,
    const int* offsets,const int* support,int nr,int ne,double* packed) {
    const int point=int(blockIdx.x),thread=int(threadIdx.x);
    if(thread==0) {
        packed[point]=diagonal[point];
        packed[ne+point]=g[nr+point];
    }
    for(int i=offsets[point]+thread;i<offsets[point+1];i+=int(blockDim.x))
        packed[2*ne+i]=cross[point*nr+support[i]];
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
    // Online Schur buffers are retained by the backend workspace. Only the
    // reduced state system and compressed range blocks cross to the host.
    cuda::Array<DirectVisualLayout> layout;
    cuda::Array<int> rows,columns,offsets,support,invalid;
    cuda::Array<double> lifts,tangent,whiteResiduals,pose,cross,diagonal,gradient,reduced,rangeSystem,metric;
    cuda::Array<VisualBatchInput,true> hostInputs;
    cuda::Array<DirectVisualLayout,true> hostLayout;
    cuda::Array<int,true> hostRows,hostColumns,hostOffsets,hostSupport;
    cuda::Array<double,true> hostLifts,hostReduced,hostRangeSystem;
    cuda::Array<int,true> hostInvalid;
    int retained=0,eliminated=0,visualRows=0,rangeSystemSize=0;
    bool pending=false;
    bool finishPending() {
        if(!pending) return true;
        pending=false;
        return cudaStreamSynchronize(stream)==cudaSuccess;
    }
    Impl() { cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking); }
    ~Impl() {
        if(stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
};

VisualCudaBatch::VisualCudaBatch():impl_(std::make_unique<Impl>()) {}
VisualCudaBatch::~VisualCudaBatch()=default;

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
    if(!s.stream || !s.finishPending() || !count || count>1000000 || nr<=0 || nr>512 || ne<=0 || n>4000 ||
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
        s.rangeSystemSize=2*ne+int(support.size());
    } else if(nr!=s.retained || ne!=s.eliminated || rowCount!=s.visualRows) return false;
    if(!prepare(s.parameters,s.hostParameters,values.size()) || !prepare(s.lifts,s.hostLifts,plus.size()) ||
        !reserve(s.jacobians,jacobianSize) || !reserve(s.residuals,count) ||
        !reserve(s.pose,nr*nr) || !reserve(s.cross,nr*ne) || !reserve(s.diagonal,ne) ||
        !reserve(s.gradient,n) || !reserve(s.whiteResiduals,rowCount) || !s.invalid.reserve(1) ||
        !s.metric.reserve(1) || !reserve(s.reduced,nr*(nr+2)) ||
        !reserve(s.hostReduced,nr*(nr+2)+1) || !prepare(s.rangeSystem,s.hostRangeSystem,s.rangeSystemSize) ||
        !s.hostInvalid.reserve(1)) return false;
    const auto upload=[&](auto& buffer,auto& pinned,const auto& data) {
        if(data.empty()) return true;
        std::copy(data.begin(),data.end(),pinned.data());
        return cudaMemcpyAsync(buffer.data(),pinned.data(),data.size()*sizeof(data[0]),cudaMemcpyHostToDevice,s.stream)==cudaSuccess;
    };
    const auto zero=[&](auto& buffer,std::size_t size) {
        return cudaMemsetAsync(buffer.data(),0,size*sizeof(*buffer.data()),s.stream)==cudaSuccess;
    };
    s.pending=true;
    // Pinned copies keep topology and values alive through reduceOnline(),
    // without a pageable staging wait before the CPU starts its own factors.
    bool ok=(!newTopology || (upload(s.rows,s.hostRows,rows) && upload(s.columns,s.hostColumns,columns) &&
        upload(s.layout,s.hostLayout,layout) && upload(s.offsets,s.hostOffsets,offsets) &&
        upload(s.support,s.hostSupport,support) && upload(s.inputs,s.hostInputs,factors))) &&
        upload(s.lifts,s.hostLifts,plus) && upload(s.parameters,s.hostParameters,values) &&
        zero(s.pose,nr*nr) && zero(s.cross,nr*ne) && zero(s.diagonal,ne) &&
        zero(s.gradient,n) && zero(s.metric,1) && zero(s.invalid,1);
    if(ok) {
        const auto blocks=(count+127)/128;
        evaluateVisual<<<blocks,128,0,s.stream>>>(s.inputs.data(),int(count),s.parameters.data(),
            s.residuals.data(),s.jacobians.data());
        projectVisualRows<<<blocks,128,0,s.stream>>>(s.inputs.data(),s.layout.data(),s.lifts.data(),
            s.residuals.data(),s.jacobians.data(),s.rows.data(),int(count),s.tangent.data(),s.whiteResiduals.data());
        accumulateVisualPairs<<<blocks,128,0,s.stream>>>(s.rows.data(),s.columns.data(),s.tangent.data(),
            s.whiteResiduals.data(),int(count),nr,s.pose.data(),s.cross.data(),s.diagonal.data(),s.gradient.data());
        packRangeSystem<<<ne,32,0,s.stream>>>(s.cross.data(),s.diagonal.data(),s.gradient.data(),
            s.offsets.data(),s.support.data(),nr,ne,s.rangeSystem.data());
        visualCost<<<blocks,128,0,s.stream>>>(s.residuals.data(),s.layout.data(),int(count),s.metric.data(),s.invalid.data());
        ok=cudaGetLastError()==cudaSuccess;
    }
    if(!ok) s.finishPending();
    return ok;
}

bool VisualCudaBatch::reduceOnline(double lambda,VisualReducedSystem& out) {
    out={};
    auto& s=*impl_;
    const int nr=s.retained,ne=s.eliminated,size=nr*(nr+2);
    initializeReduced<<<(size+127)/128,128,0,s.stream>>>(s.pose.data(),s.gradient.data(),nr,s.reduced.data());
    subtractPointSchur<<<ne,128,0,s.stream>>>(s.cross.data(),s.diagonal.data(),s.gradient.data(),
        s.offsets.data(),s.support.data(),nr,lambda,s.reduced.data());
    const bool ok=cudaGetLastError()==cudaSuccess &&
        cudaMemcpyAsync(s.hostReduced.data(),s.reduced.data(),size*sizeof(double),cudaMemcpyDeviceToHost,s.stream)==cudaSuccess &&
        (!s.pending || cudaMemcpyAsync(s.hostRangeSystem.data(),s.rangeSystem.data(),
            s.rangeSystemSize*sizeof(double),cudaMemcpyDeviceToHost,s.stream)==cudaSuccess) &&
        cudaMemcpyAsync(s.hostReduced.data()+size,s.metric.data(),sizeof(double),cudaMemcpyDeviceToHost,s.stream)==cudaSuccess &&
        cudaMemcpyAsync(s.hostInvalid.data(),s.invalid.data(),sizeof(int),cudaMemcpyDeviceToHost,s.stream)==cudaSuccess;
    s.pending=true;
    if(!s.finishPending() || !ok || *s.hostInvalid.data()) return false;
    out={s.hostReduced.data(),s.hostReduced.data()+nr*nr,s.hostReduced.data()+nr*(nr+1),
        s.hostRangeSystem.data(),s.hostRangeSystem.data()+ne,s.hostRangeSystem.data()+2*ne,
        s.hostOffsets.data(),s.hostSupport.data(),s.hostReduced.data()[size]};
    return true;
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
}
