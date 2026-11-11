#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace cgmli::features::detail {

using CudaOrbWords=std::array<std::uint64_t,4>;
/// Immutable device rows, with an owner shared by all matching jobs that read them.
struct OrbCudaDescriptorRows {
    std::shared_ptr<const unsigned char> data;
    std::shared_ptr<const int> rowIndices;
    std::size_t count=0,stride=0;
    int device=-1;
};
struct OrbCudaQuery {
    int group=-1, octave=0;
    int minX=0,maxX=-1,minY=0,maxY=-1,first=0,last=0;
    float predictedX=0,predictedY=0;
    double direction[3]{},planeNorm=0;
    bool useGrid=false,predicted=false;
};
struct OrbCudaTrain {
    float x=0,y=0;
    int octave=0;
    double bearing[3]{},planeNorm=0;
};
struct OrbCudaNeighbors {
    int index[2]{-1,-1},distance[2]{257,257};
    int representative=-1;
};
struct OrbCudaMatchInput {
    const std::vector<CudaOrbWords>* queryWords=nullptr,*trainWords=nullptr;
    std::shared_ptr<const OrbCudaDescriptorRows> deviceQuery,deviceTrain;
    bool cacheQuery=false,cacheTrain=false,geometry=false,rotationOnly=false,reverse=false,owned=false;
    int groups=0,gridColumns=0,maxOctaveDifference=0;
    double radiusSquared=0,sineThreshold=0,cosineThreshold=0;
    std::vector<int> gridOffsets,gridEntries,octaveEntries,priorities;
    std::vector<OrbCudaQuery> queries;
    std::vector<OrbCudaTrain> train;
    std::vector<std::uint8_t> allowed;
};
struct OrbCudaMatchOutput {
    std::vector<OrbCudaNeighbors> forward,reverse,owned;
    std::uint64_t candidatePairs=0,descriptorComparisons=0;
    unsigned residentDescriptorSets=0;
};

/// Parallel jobs use private streams and scratch. Immutable descriptor rows
/// can stay in the bounded device cache across consecutive proposals.
class OrbCudaMatcherSession {
public:
    OrbCudaMatcherSession();
    ~OrbCudaMatcherSession();
    OrbCudaMatcherSession(const OrbCudaMatcherSession&)=delete;
    OrbCudaMatcherSession& operator=(const OrbCudaMatcherSession&)=delete;
    void beginProposal();
    bool match(const OrbCudaMatchInput&,OrbCudaMatchOutput&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
