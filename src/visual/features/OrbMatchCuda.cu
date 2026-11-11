#include "OrbMatchCuda.hpp"
#include <common/CudaArray.hpp>
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>

namespace cgmli::features::detail {
namespace {
constexpr int kThreads=64;
constexpr unsigned kQueryMask=(1u<<20)-1;
constexpr unsigned long long kEmpty=~0ull;
struct Candidate { int group,train,query,distance; };
struct Ranked { unsigned long long first,second; };
struct UploadLayout {
    std::size_t size=0;
    template<class T> std::size_t add(const std::vector<T>& values) {
        const auto start=(size+alignof(T)-1)&~(alignof(T)-1);
        size=start+values.size()*sizeof(T);
        return start;
    }
    template<class T> static const T* stage(std::size_t offset,const std::vector<T>& values,
        unsigned char* host,const unsigned char* device) {
        if(values.empty()) return nullptr;
        std::memcpy(host+offset,values.data(),values.size()*sizeof(T));
        return reinterpret_cast<const T*>(device+offset);
    }
};
struct Work {
    const OrbCudaQuery* queries;
    const OrbCudaTrain* train;
    const unsigned char* queryWords,*trainWords;
    std::size_t queryStride,trainStride;
    const int* queryRows,*trainRows;
    const int* offsets,*grid,*octaves;
    const std::uint8_t* allowed;
    int trainCount,gridColumns,maxOctaveDifference;
    double radiusSquared,sineThreshold,cosineThreshold;
    bool geometry,rotationOnly;
};
__device__ void consider(const Work& w,const OrbCudaQuery& q,int i,int j,
    Candidate* pairs,unsigned capacity,unsigned* count,unsigned& uncertain) {
    if(w.allowed && !w.allowed[std::size_t(i)*w.trainCount+j]) return;
    const auto& p=w.train[j];
    if(w.maxOctaveDifference>=0 &&
        llabs(static_cast<long long>(q.octave)-p.octave)>w.maxOctaveDifference) return;
    if(q.predicted) {
        const float dx=p.x-q.predictedX,dy=p.y-q.predictedY;
        if(double(dx)*dx+double(dy)*dy>w.radiusSquared) return;
    }
    if(w.geometry) {
        const double dot=p.bearing[0]*q.direction[0]+
            (p.bearing[1]*q.direction[1]+p.bearing[2]*q.direction[2]);
        const double limit=w.rotationOnly?w.cosineThreshold:w.sineThreshold*fmin(q.planeNorm,p.planeNorm);
        const double residual=w.rotationOnly?dot:fabs(dot);
        if(!w.rotationOnly && p.planeNorm<1e-12) return;
        // Delegate roundoff at a geometric boundary to the scalar reference.
        if(fabs(residual-limit)<=2e-14) uncertain=1;
        if(w.rotationOnly?residual<limit:residual>limit) return;
    }
    const auto* query=reinterpret_cast<const unsigned long long*>(
        w.queryWords+(w.queryRows?w.queryRows[i]:i)*w.queryStride);
    const auto* train=reinterpret_cast<const unsigned long long*>(
        w.trainWords+(w.trainRows?w.trainRows[j]:j)*w.trainStride);
    int distance=0;
    for(int k=0;k<4;++k) distance+=__popcll(query[k]^train[k]);
    const unsigned slot=atomicAdd(count,1u);
    if(slot<capacity) pairs[slot]={q.group,j,i,distance};
}
struct Counts { unsigned candidates,uncertain; };
struct AddCounts {
    __device__ Counts operator()(const Counts& a,const Counts& b) const {
        return {a.candidates+b.candidates,a.uncertain|b.uncertain};
    }
};
__global__ void candidateDistances(Work w,Candidate* pairs,unsigned capacity,unsigned* totals) {
    const int i=int(blockIdx.x);
    const auto& q=w.queries[i];
    if(q.group<0) return;
    Counts count{};
    if(q.useGrid) {
        for(int y=q.minY;y<=q.maxY;++y) for(int x=q.minX;x<=q.maxX;++x) {
            const int cell=y*w.gridColumns+x;
            for(int k=w.offsets[cell]+int(threadIdx.x);k<w.offsets[cell+1];k+=kThreads) {
                ++count.candidates;
                consider(w,q,i,w.grid[k],pairs,capacity,totals+1,count.uncertain);
            }
        }
    } else {
        for(int k=q.first+int(threadIdx.x);k<q.last;k+=kThreads) {
            ++count.candidates;
            consider(w,q,i,w.octaves[k],pairs,capacity,totals+1,count.uncertain);
        }
    }
    using Reduce=cub::BlockReduce<Counts,kThreads>;
    __shared__ Reduce::TempStorage storage;
    const auto sum=Reduce(storage).Reduce(count,AddCounts{});
    if(threadIdx.x==0) {
        atomicAdd(totals,sum.candidates);
        if(sum.uncertain) atomicOr(totals+2,1u);
    }
}
OrbCudaNeighbors neighbors(Ranked result) {
    OrbCudaNeighbors out;
    const unsigned long long keys[]={result.first,result.second};
    for(int i=0;i<2;++i) if(keys[i]!=kEmpty) {
        out.distance[i]=int(keys[i]>>40);
        out.index[i]=int((keys[i]>>20)&kQueryMask);
    }
    if(out.index[0]>=0) out.representative=int(result.first&kQueryMask);
    return out;
}
template<bool Second>
__device__ void rank(Ranked& row,int distance,int identity,int representative) {
    const auto key=(static_cast<unsigned long long>(distance)<<40) |
        (static_cast<unsigned long long>(identity)<<20) | unsigned(representative);
    if constexpr(Second) {
        if(int((row.first>>20)&kQueryMask)!=identity) atomicMin(&row.second,key);
    } else atomicMin(&row.first,key);
}
// Find the winner before finding the runner-up. Excluding the winner's identity
// lets multiple views of one landmark compete without becoming its second neighbor.
template<bool Second>
__global__ void rankCandidates(const Candidate* pairs,const unsigned* count,unsigned capacity,
    const int* priority,Ranked* forward,Ranked* reverse,Ranked* owned) {
    const unsigned size=min(*count,capacity);
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        const auto pair=pairs[i];
        rank<Second>(forward[pair.group],pair.distance,pair.train,pair.query);
        if(reverse) rank<Second>(reverse[pair.train],pair.distance,pair.group,0);
        if(owned && priority[pair.group]) rank<Second>(owned[pair.train],pair.distance,pair.group,0);
    }
}
struct Workspace {
    cudaStream_t stream=nullptr;
    bool busy=false;
    cuda::Array<unsigned char> input;
    cuda::Array<unsigned char,true> hostInput;
    cuda::Array<Candidate> pairs;
    cuda::Array<unsigned> totals;
    cuda::Array<Ranked> ranked;
    cuda::Array<Ranked,true> hostRanked;
    cuda::Array<unsigned,true> hostTotals;
    Workspace() { cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking); }
    ~Workspace() { if(stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } }
};
}
struct OrbCudaMatcherSession::Impl {
    struct Resident {
        std::vector<CudaOrbWords> words;
        cuda::Array<CudaOrbWords> device;
        cuda::Array<CudaOrbWords,true> staging;
        cudaEvent_t ready=nullptr;
        std::size_t hash=0,generation=0;
        Resident() { cudaEventCreateWithFlags(&ready,cudaEventDisableTiming); }
        ~Resident() { if(ready) cudaEventDestroy(ready); }
    };
    static constexpr std::size_t maxResidents=64;
    std::mutex mutex;
    std::vector<std::unique_ptr<Workspace>> workers;
    std::vector<std::unique_ptr<Resident>> residents;
    std::map<const std::vector<CudaOrbWords>*,Resident*> proposalWords;
    std::size_t generation=0;

    Workspace* acquire() {
        std::lock_guard<std::mutex> lock(mutex);
        for(auto& worker:workers) if(!worker->busy) { worker->busy=true; return worker.get(); }
        workers.emplace_back(std::make_unique<Workspace>());
        workers.back()->busy=true;
        return workers.back().get();
    }
    void release(Workspace* worker) {
        std::lock_guard<std::mutex> lock(mutex);
        worker->busy=false;
    }
    const CudaOrbWords* resident(const std::vector<CudaOrbWords>& words,cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex);
        auto*& slot=proposalWords[&words];
        if(!slot) {
            std::size_t hash=words.size();
            for(const auto& row:words) for(auto word:row) hash=(hash^word)*1099511628211ull;
            Resident* oldest=nullptr;
            for(const auto& entry:residents) {
                if(entry->hash==hash && entry->words==words) { slot=entry.get(); break; }
                if(entry->generation!=generation && (!oldest || entry->generation<oldest->generation)) oldest=entry.get();
            }
            if(!slot) {
                if(residents.size()<maxResidents) {
                    residents.push_back(std::make_unique<Resident>());
                    oldest=residents.back().get();
                }
                if(!oldest || !oldest->ready || !oldest->device.reserve(words.size()) ||
                    !oldest->staging.reserve(words.size())) return nullptr;
                oldest->generation=generation;
                oldest->words=words;
                std::copy(words.begin(),words.end(),oldest->staging.data());
                const auto bytes=words.size()*sizeof(CudaOrbWords);
                if(cudaMemcpyAsync(oldest->device.data(),oldest->staging.data(),bytes,cudaMemcpyHostToDevice,stream)!=cudaSuccess ||
                    cudaEventRecord(oldest->ready,stream)!=cudaSuccess) {
                    // Prevent reusing this staging buffer until the failed job has drained.
                    oldest->words.clear();
                    return nullptr;
                }
                oldest->hash=hash;
                slot=oldest;
            }
            // Entries touched by this proposal cannot be overwritten while parallel jobs read them.
            slot->generation=generation;
        }
        return cudaStreamWaitEvent(stream,slot->ready,0)==cudaSuccess?slot->device.data():nullptr;
    }
};
OrbCudaMatcherSession::OrbCudaMatcherSession():impl_(std::make_unique<Impl>()) {}
OrbCudaMatcherSession::~OrbCudaMatcherSession()=default;
void OrbCudaMatcherSession::beginProposal() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->proposalWords.clear();
    ++impl_->generation;
}

bool OrbCudaMatcherSession::match(const OrbCudaMatchInput& in,OrbCudaMatchOutput& out) {
    if(!in.queryWords || !in.trainWords || in.queries.empty() || in.train.empty() || in.groups<1 ||
        in.queries.size()>=kQueryMask || in.train.size()>=kQueryMask || in.groups>=int(kQueryMask) ||
        in.queries.size()*in.train.size()>64000000) return false;
    auto& state=*impl_;
    auto& w=*state.acquire();
    struct Release {
        Impl& state;
        Workspace& worker;
        bool completed=false;
        ~Release() { if(!completed && worker.stream) cudaStreamSynchronize(worker.stream); state.release(&worker); }
    } release{state,w};
    if(!w.stream) return false;
    int device=-1;
    if((in.deviceQuery || in.deviceTrain) && cudaGetDevice(&device)!=cudaSuccess) return false;
    const auto deviceRows=[&](const auto& view,std::size_t count)->const OrbCudaDescriptorRows* {
        return view && view->data && view->count==count && view->device==device &&
            view->stride>=sizeof(CudaOrbWords) && view->stride%8==0 &&
            reinterpret_cast<std::uintptr_t>(view->data.get())%8==0?view.get():nullptr;
    };
    const auto* qRows=deviceRows(in.deviceQuery,in.queries.size());
    const auto* tRows=deviceRows(in.deviceTrain,in.train.size());
    const CudaOrbWords *q=nullptr,*t=nullptr;
    if(!qRows && in.cacheQuery) q=state.resident(*in.queryWords,w.stream);
    if(!tRows && in.cacheTrain) t=state.resident(*in.trainWords,w.stream);
    const auto capacity=static_cast<unsigned>(std::min({in.queries.size()*in.train.size(),
        std::max<std::size_t>(4096,32*in.queries.size()),std::size_t(262144)}));
    const int train=int(in.train.size());
    const auto rankCount=in.groups+(in.reverse?train:0)+(in.owned?train:0);
    UploadLayout layout;
    const auto queries=layout.add(in.queries), features=layout.add(in.train);
    const auto offsets=layout.add(in.gridOffsets), grid=layout.add(in.gridEntries);
    const auto octaves=layout.add(in.octaveEntries), allowed=layout.add(in.allowed), priorities=layout.add(in.priorities);
    const auto queryWords=(qRows || q)?0:layout.add(*in.queryWords);
    const auto trainWords=(tRows || t)?0:layout.add(*in.trainWords);
    if(!w.input.reserve(layout.size) || !w.hostInput.reserve(layout.size) ||
        !w.pairs.reserve(capacity) || !w.totals.reserve(3) || !w.hostTotals.reserve(3) ||
        !w.ranked.reserve(rankCount) || !w.hostRanked.reserve(rankCount)) return false;
    const auto stage=[&](auto offset,const auto& values) {
        return UploadLayout::stage(offset,values,w.hostInput.data(),w.input.data());
    };
    if(!qRows && !q) q=stage(queryWords,*in.queryWords);
    if(!tRows && !t) t=stage(trainWords,*in.trainWords);
    if((!qRows && !q) || (!tRows && !t)) return false;
    const auto* devicePriorities=stage(priorities,in.priorities);
    const Work work{stage(queries,in.queries),stage(features,in.train),
        qRows?qRows->data.get():reinterpret_cast<const unsigned char*>(q),
        tRows?tRows->data.get():reinterpret_cast<const unsigned char*>(t),
        qRows?qRows->stride:sizeof(CudaOrbWords),tRows?tRows->stride:sizeof(CudaOrbWords),
        qRows?qRows->rowIndices.get():nullptr,tRows?tRows->rowIndices.get():nullptr,
        stage(offsets,in.gridOffsets),stage(grid,in.gridEntries),
        stage(octaves,in.octaveEntries),stage(allowed,in.allowed),train,in.gridColumns,in.maxOctaveDifference,
        in.radiusSquared,in.sineThreshold,in.cosineThreshold,in.geometry,in.rotationOnly};
    if(cudaMemcpyAsync(w.input.data(),w.hostInput.data(),layout.size,cudaMemcpyHostToDevice,w.stream)!=cudaSuccess) return false;
    if(cudaMemsetAsync(w.ranked.data(),0xff,rankCount*sizeof(Ranked),w.stream)!=cudaSuccess ||
        cudaMemsetAsync(w.totals.data(),0,3*sizeof(unsigned),w.stream)!=cudaSuccess) return false;
    candidateDistances<<<static_cast<unsigned>(in.queries.size()),kThreads,0,w.stream>>>(
        work,w.pairs.data(),capacity,w.totals.data());
    auto* reverse=in.reverse?w.ranked.data()+in.groups:nullptr;
    auto* owned=in.owned?w.ranked.data()+in.groups+(in.reverse?train:0):nullptr;
    const unsigned blocks=std::min(128u,(capacity+127)/128);
    rankCandidates<false><<<blocks,128,0,w.stream>>>(w.pairs.data(),w.totals.data()+1,capacity,
        devicePriorities,w.ranked.data(),reverse,owned);
    rankCandidates<true><<<blocks,128,0,w.stream>>>(w.pairs.data(),w.totals.data()+1,capacity,
        devicePriorities,w.ranked.data(),reverse,owned);
    const auto download=[&](const auto& to,const auto& from,std::size_t count) {
        return cudaMemcpyAsync(to.data(),from.data(),count*sizeof(*to.data()),cudaMemcpyDeviceToHost,w.stream)==cudaSuccess;
    };
    if(cudaGetLastError()!=cudaSuccess || !download(w.hostTotals,w.totals,3) ||
        !download(w.hostRanked,w.ranked,rankCount)) return false;
    if(cudaStreamSynchronize(w.stream)!=cudaSuccess) return false;
    release.completed=true;
    if(w.hostTotals.data()[2] || w.hostTotals.data()[1]>capacity) return false;
    out.candidatePairs=w.hostTotals.data()[0];
    out.descriptorComparisons=w.hostTotals.data()[1];
    out.residentDescriptorSets=unsigned(qRows!=nullptr)+unsigned(tRows!=nullptr);
    const auto unpack=[&](int offset,int count,std::vector<OrbCudaNeighbors>& rows) {
        rows.resize(count);
        std::transform(w.hostRanked.data()+offset,w.hostRanked.data()+offset+count,rows.begin(),neighbors);
    };
    unpack(0,in.groups,out.forward);
    if(in.reverse) unpack(in.groups,train,out.reverse);
    if(in.owned) unpack(in.groups+(in.reverse?train:0),train,out.owned);
    return true;
}

}
