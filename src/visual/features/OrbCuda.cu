#include "OrbCuda.hpp"
#include "OrbMatchCuda.hpp"

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace cgmli::features::detail {
namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw OrbCudaError(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<class T, bool Pinned = false>
class Buffer {
public:
    Buffer() = default;
    ~Buffer() { release(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : data_(other.data_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.capacity_ = 0;
    }
    void reserve(std::size_t size) {
        if (size <= capacity_) return;
        release();
        void* memory = nullptr;
        const auto status = Pinned ? cudaMallocHost(&memory, size * sizeof(T)) : cudaMalloc(&memory, size * sizeof(T));
        check(status, Pinned ? "allocate pinned ORB buffer" : "allocate device ORB buffer");
        data_ = static_cast<T*>(memory);
        capacity_ = size;
    }
    T* get() const { return data_; }
private:
    void release() noexcept {
        if (data_) {
            if constexpr (Pinned) cudaFreeHost(data_);
            else cudaFree(data_);
        }
        data_ = nullptr;
        capacity_ = 0;
    }
    T* data_ = nullptr;
    std::size_t capacity_ = 0;
};

struct Stream {
    Stream() { check(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), "create ORB stream"); }
    ~Stream() { cudaStreamDestroy(handle); }
    cudaStream_t handle = nullptr;
};

struct Event {
    Event() { check(cudaEventCreate(&handle), "create ORB timing event"); }
    ~Event() { cudaEventDestroy(handle); }
    cudaEvent_t handle = nullptr;
};

void record(const Event& event, cudaStream_t stream) {
    check(cudaEventRecord(event.handle, stream), "record ORB timing event");
}

double elapsed(const Event& from, const Event& to) {
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, from.handle, to.handle), "read ORB timing");
    return milliseconds;
}

struct AxisSample { int index, nextWeight; };

// INTER_LINEAR_EXACT uses half-pixel coordinates and 8-bit fixed-point weights
// for unsigned 8-bit images. Keep the full precision until the final rounding.
std::vector<AxisSample> resizeAxis(int input, int output) {
    std::vector<AxisSample> samples(output);
    const double scale = 1.0 / (double(output) / input);
    for (int i = 0; i < output; ++i) {
        const double source = scale * (i + 0.5) - 0.5;
        const int index = int(std::floor(source));
        if (index < 0) samples[i] = {0, 0};
        else if (index >= input - 1) samples[i] = {input - 1, 0};
        else samples[i] = {index, int(std::nearbyint((source - index) * 256.0))};
    }
    return samples;
}

__global__ void resizeExact(const unsigned char* input, int inputWidth, int inputHeight,
    unsigned char* output, int width, int height, const AxisSample* xs, const AxisSample* ys) {
    const int x = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;
    const auto sx = xs[x], sy = ys[y];
    const int nextX = min(sx.index + 1, inputWidth - 1), nextY = min(sy.index + 1, inputHeight - 1);
    const auto* row = input + sy.index * inputWidth;
    const auto* nextRow = input + nextY * inputWidth;
    const int top = row[sx.index] * (256 - sx.nextWeight) + row[nextX] * sx.nextWeight;
    const int bottom = nextRow[sx.index] * (256 - sx.nextWeight) + nextRow[nextX] * sx.nextWeight;
    output[y * width + x] = static_cast<unsigned char>((top * (256 - sy.nextWeight) + bottom * sy.nextWeight + 32768) >> 16);
}

__global__ void fastScores(const unsigned char* image, int width, int height,
    int edge, int threshold, unsigned char* scores) {
    const int x = edge + int(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = edge + int(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width - edge || y >= height - edge) return;
    const auto* center = image + y * width + x;
    const int value = *center;
    const int cardinal[4] = {
        center[3 * width] - value, center[3] - value,
        center[-3 * width] - value, center[-3] - value};
    int bright = 0, dark = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        bright += cardinal[i] > threshold;
        dark += cardinal[i] < -threshold;
    }
    // Any circular arc of length nine covers at least two cardinal samples.
    int score = 0;
    if (bright >= 2 || dark >= 2) {
        const int dx[16] = {0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
        const int dy[16] = {3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3};
        int differences[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) differences[i] = center[dy[i] * width + dx[i]] - value;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int first = differences[i], last = differences[(i + 8) & 15];
            const int endsBright = min(first, last), endsDark = min(-first, -last);
            if (endsBright <= max(score, threshold) && endsDark <= max(score, threshold)) continue;
            int lo = min(first, last), hi = max(first, last);
            #pragma unroll
            for (int j = 1; j < 8; ++j) {
                const int delta = differences[(i + j) & 15];
                lo = min(lo, delta);
                hi = max(hi, delta);
            }
            score = max(score, max(lo, -hi));
        }
    }
    scores[y * width + x] = static_cast<unsigned char>(score > threshold ? score - 1 : 0);
}

__global__ void selectCorners(const unsigned char* scores, int width, int height, int edge,
    int cellSize, OrbCudaCorner* corners, int* count, int capacity) {
    const int x = edge + int(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = edge + int(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width - edge || y >= height - edge) return;
    const int score = scores[y * width + x];
    if (!score) return;
    const int left = x - (x - edge) % cellSize, top = y - (y - edge) % cellSize;
    const int right = min(left + cellSize, width - edge), bottom = min(top + cellSize, height - edge);
    // Interior points use the same neighborhood as layer NMS. At a cell edge
    // the original ROI FAST compares only to centers within that cell.
    for (int yy = max(y - 1, top); yy <= min(y + 1, bottom - 1); ++yy)
        for (int xx = max(x - 1, left); xx <= min(x + 1, right - 1); ++xx)
            if ((xx != x || yy != y) && score <= scores[yy * width + xx]) return;
    const int index = atomicAdd(count, 1);
    if (index < capacity) corners[index].packed = std::uint32_t(x) |
        (std::uint32_t(y) << 12) | (std::uint32_t(score) << 24);
}

constexpr int kScanThreads = 256;
__global__ void integralRows(const unsigned char* image, int width, int* integral) {
    using Scan = cub::BlockScan<int, kScanThreads>;
    __shared__ Scan::TempStorage storage;
    const auto* row = image + blockIdx.x * width;
    auto* output = integral + (blockIdx.x + 1) * (width + 1);
    if (threadIdx.x == 0) output[0] = 0;
    int carry = 0;
    for (int start = 0; start < width; start += kScanThreads) {
        const int x = start + int(threadIdx.x);
        int sum, total;
        Scan(storage).InclusiveSum(x < width ? row[x] : 0, sum, total);
        if (x < width) output[x + 1] = carry + sum;
        carry += total;
        __syncthreads();
    }
}

constexpr int kIntegralTileRows = 64;
constexpr int kIntegralRowsPerThread = 8;
__global__ void integralColumns(int* integral, int width, int height, int* tails) {
    __shared__ int totals[kIntegralTileRows / kIntegralRowsPerThread][32];
    const int x = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int first = int(blockIdx.y) * kIntegralTileRows + int(threadIdx.y) * kIntegralRowsPerThread + 1;
    const int validRows = x <= width ? max(0, min(kIntegralRowsPerThread, height - first + 1)) : 0;
    int partial[kIntegralRowsPerThread];
    int sum = 0;
    #pragma unroll
    for (int row = 0; row < kIntegralRowsPerThread; ++row) {
        if (row < validRows) sum += integral[(first + row) * (width + 1) + x];
        partial[row] = sum;
    }
    totals[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y == 0 && x <= width) {
        int carry = 0;
        #pragma unroll
        for (int y = 0; y < kIntegralTileRows / kIntegralRowsPerThread; ++y) {
            const int chunk = totals[y][threadIdx.x];
            totals[y][threadIdx.x] = carry;
            carry += chunk;
        }
        tails[blockIdx.y * (width + 1) + x] = carry;
    }
    __syncthreads();
    const int carry = totals[threadIdx.y][threadIdx.x];
    #pragma unroll
    for (int row = 0; row < kIntegralRowsPerThread; ++row)
        if (row < validRows) integral[(first + row) * (width + 1) + x] = carry + partial[row];
}

__global__ void addColumnCarries(int* integral, int width, int height, const int* tails) {
    __shared__ int carries[32];
    const int x = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int tile = int(blockIdx.y) + 1;
    if (threadIdx.y == 0 && x <= width) {
        int carry = 0;
        for (int i = 0; i < tile; ++i) carry += tails[i * (width + 1) + x];
        carries[threadIdx.x] = carry;
    }
    __syncthreads();
    if (x > width) return;
    const int first = tile * kIntegralTileRows + 1;
    const int end = min(first + kIntegralTileRows, height + 1);
    for (int y = first + int(threadIdx.y); y < end; y += int(blockDim.y))
        integral[y * (width + 1) + x] += carries[threadIdx.x];
}

struct DeviceLayer { const int* integral; const int* invalid; int width, height; };
struct HalfSize { double x, y; };

__device__ HalfSize halfSize(const OrbCudaPatch& patch) {
    constexpr double scale = 3.4641016151377546;
    return {fmax(0.5, scale * sqrt(patch.a * patch.a + patch.b * patch.b)),
        fmax(0.5, scale * sqrt(patch.c * patch.c + patch.d * patch.d))};
}

__device__ double prefixAt(const DeviceLayer& layer, double x, double y) {
    const int ix = min(int(floor(x)), layer.width - 1), iy = min(int(floor(y)), layer.height - 1);
    const double dx = x - ix, dy = y - iy;
    const auto* a = layer.integral + iy * (layer.width + 1) + ix;
    const auto* b = a + layer.width + 1;
    return (1 - dy) * ((1 - dx) * a[0] + dx * a[1]) + dy * ((1 - dx) * b[0] + dx * b[1]);
}

__device__ bool sample(const DeviceLayer& layer, const OrbCudaPatch& patch, HalfSize half,
    int dx, int dy, float& intensity) {
    const double x = patch.x + (patch.a * dx + patch.b * dy);
    const double y = patch.y + (patch.c * dx + patch.d * dy);
    const double x0 = x + 0.5 - half.x, x1 = x + 0.5 + half.x;
    const double y0 = y + 0.5 - half.y, y1 = y + 0.5 + half.y;
    if (x0 < 0 || y0 < 0 || x1 > layer.width || y1 > layer.height) return false;
    const double sum = prefixAt(layer, x1, y1) - prefixAt(layer, x0, y1) -
        prefixAt(layer, x1, y0) + prefixAt(layer, x0, y0);
    intensity = static_cast<float>(sum / (4 * half.x * half.y));
    return isfinite(intensity);
}

__device__ bool validSupport(const DeviceLayer& layer, const OrbCudaPatch& patch) {
    const auto half = halfSize(patch);
    const double extentX = 15 * (fabs(patch.a) + fabs(patch.b)) + half.x;
    const double extentY = 15 * (fabs(patch.c) + fabs(patch.d)) + half.y;
    const double x0 = patch.x + 0.5 - extentX, x1 = patch.x + 0.5 + extentX;
    const double y0 = patch.y + 0.5 - extentY, y1 = patch.y + 0.5 + extentY;
    if (x0 < 0 || y0 < 0 || x1 > layer.width || y1 > layer.height) return false;
    const int left = int(floor(x0)), right = int(ceil(x1)), stride = layer.width + 1;
    const int top = int(floor(y0)) * stride, bottom = int(ceil(y1)) * stride;
    return layer.invalid[bottom + right] - layer.invalid[top + right] -
        layer.invalid[bottom + left] + layer.invalid[top + left] == 0;
}

__device__ bool orientAtPole(const DeviceLayer& layer, OrbCudaPatch& patch) {
    const auto half = halfSize(patch);
    double mx = 0, my = 0, mass = 0;
    for (int y = -3; y <= 3; ++y) for (int x = -3; x <= 3; ++x) {
        if (x * x + y * y > 9) continue;
        float intensity;
        if (!sample(layer, patch, half, 4 * x, 4 * y, intensity)) return false;
        mx += x * intensity;
        my += y * intensity;
        mass += fabsf(intensity);
    }
    if (hypot(mx, my) <= 1e-6 * fmax(1.0, 3 * mass)) return false;
    constexpr double pi = 3.14159265358979323846;
    const double degrees = atan2(my, mx) * 180.0 / pi;
    const double radians = degrees * pi / 180.0, c = cos(radians), s = sin(radians);
    const double a = patch.a * c + patch.b * s, b = patch.a * -s + patch.b * c;
    const double d = patch.c * -s + patch.d * c;
    patch.c = patch.c * c + patch.d * s;
    patch.a = a;
    patch.b = b;
    patch.d = d;
    return validSupport(layer, patch);
}

constexpr int kPatternSize = 375;
__global__ void describePatches(const DeviceLayer* layers, const OrbCudaPatch* patches,
    const int* points, const int* pairs, OrbCudaDescriptor* results) {
    __shared__ OrbCudaPatch patch;
    __shared__ HalfSize half;
    __shared__ float samples[kPatternSize];
    __shared__ int valid;
    auto& result = results[blockIdx.x];
    const auto& layer = layers[patches[blockIdx.x].level];
    if (threadIdx.x == 0) {
        patch = patches[blockIdx.x];
        valid = patch.aligned || orientAtPole(layer, patch);
        half = halfSize(patch);
        result.valid = false;
    }
    __syncthreads();
    if (!valid) return;
    for (int i = int(threadIdx.x); i < kPatternSize; i += int(blockDim.x))
        if (!sample(layer, patch, half, points[2 * i], points[2 * i + 1], samples[i])) atomicAnd(&valid, 0);
    __syncthreads();
    if (!valid) return;
    if (threadIdx.x < 32) {
        unsigned char bits = 0;
        const int first = int(threadIdx.x) * 8;
        for (int bit = 0; bit < 8; ++bit) {
            const int pair = 2 * (first + bit);
            if (samples[pairs[pair]] < samples[pairs[pair + 1]]) bits |= 1u << bit;
        }
        result.bytes[threadIdx.x] = bits;
    }
    if (threadIdx.x == 0) {
        result.a = static_cast<float>(patch.a);
        result.c = static_cast<float>(patch.c);
        result.valid = true;
    }
}

// The CPU compacts its keypoint rows in this same order. The index lets the
// matcher read the original device result without copying descriptor bytes.
__global__ void compactDescriptorRows(const OrbCudaDescriptor* results,int count,int* rows) {
    using Scan=cub::BlockScan<int,kScanThreads>;
    __shared__ Scan::TempStorage storage;
    int carry=0;
    for(int start=0;start<count;start+=kScanThreads) {
        const int i=start+int(threadIdx.x);
        const int valid=i<count && results[i].valid;
        int index,total;
        Scan(storage).ExclusiveSum(valid,index,total);
        if(valid) rows[carry+index]=i;
        carry+=total;
        __syncthreads();
    }
}

int deviceIndex() {
    int result;
    check(cudaGetDevice(&result), "select CUDA device");
    return result;
}

dim3 grid(int width, int height) { return dim3((width + 31) / 32, (height + 7) / 8); }
const dim3 kTile(32, 8);

struct Layer {
    int width, height, cornerCapacity, downloadCapacity;
    std::size_t cornerOffset;
    Buffer<unsigned char> image, scores;
    Buffer<int> integral, invalid, integralTails;
    Buffer<AxisSample> xs, ys;
};

int nonAdjacentCount(int size, int cellSize) {
    return (size / cellSize) * ((cellSize + 1) / 2) + (size % cellSize + 1) / 2;
}

template<class T>
void upload(Buffer<T>& buffer, const T* values, std::size_t count) {
    buffer.reserve(count);
    check(cudaMemcpy(buffer.get(), values, count * sizeof(T), cudaMemcpyHostToDevice), "upload static ORB data");
}

} // namespace

struct OrbCuda::Impl {
    struct DescriptorSlot {
        Buffer<OrbCudaDescriptor> results;
        Buffer<int> indices;
    };
    const int device = deviceIndex();
    Stream stream;
    std::unique_ptr<Stream> tailStream;
    Event start, uploaded, pyramidDone, fastDone, cornersDone, tailDone, integralStart, integralDone,
        descriptorStart, descriptorDone, end;
    std::vector<Layer> layers;
    Buffer<unsigned char, true> hostImage;
    Buffer<OrbCudaCorner> deviceCorners;
    Buffer<OrbCudaCorner, true> hostCorners;
    Buffer<int> deviceCounts;
    Buffer<int, true> hostCounts;
    Buffer<DeviceLayer> descriptorsLayers;
    Buffer<int> patternPoints, patternPairs;
    Buffer<OrbCudaPatch> devicePatches;
    Buffer<OrbCudaPatch, true> hostPatches;
    Buffer<OrbCudaDescriptor> deviceDescriptors;
    std::shared_ptr<DescriptorSlot> descriptorSlots[2],describedSlot;
    std::size_t describedCount=0;
    Buffer<OrbCudaDescriptor, true> hostDescriptors;
    OrbCudaTiming timing;
    bool integralsQueued=false;
    const int edge, cellSize, minimumThreshold;

    Impl(const std::vector<OrbCudaLevel>& sizes, int edgeThreshold, int cell, int threshold,
        const int* points, const int* pairs, int featureCapacity)
        : edge(edgeThreshold), cellSize(cell), minimumThreshold(threshold) {
        std::vector<DeviceLayer> descriptorLayers;
        layers.reserve(sizes.size());
        std::size_t capacity = 0;
        for (const auto& size : sizes) {
            const int w = size.width, h = size.height;
            if (w > 4096 || h > 4096 || std::size_t(w) * h > std::numeric_limits<int>::max() / 255)
                throw OrbCudaError("image dimensions exceed the exact CUDA ORB workspace");
            const int corners = nonAdjacentCount(w - 2 * edge, cellSize) * nonAdjacentCount(h - 2 * edge, cellSize);
            const int initialDownload=std::min(corners,std::max(512,w*h/32));
            Layer layer{w, h, corners, initialDownload, capacity};
            const auto imageSize = std::size_t(w) * h, integralSize = std::size_t(w + 1) * (h + 1);
            layer.image.reserve(imageSize);
            layer.scores.reserve(imageSize);
            layer.integral.reserve(integralSize);
            layer.integralTails.reserve(std::size_t(w + 1) * ((h + kIntegralTileRows - 1) / kIntegralTileRows));
            upload(layer.invalid, size.invalidIntegral, integralSize);
            if (!layers.empty()) {
                const auto xs = resizeAxis(layers.back().width, w), ys = resizeAxis(layers.back().height, h);
                upload(layer.xs, xs.data(), xs.size());
                upload(layer.ys, ys.data(), ys.size());
            }
            descriptorLayers.push_back({layer.integral.get(), layer.invalid.get(), w, h});
            layers.push_back(std::move(layer));
            capacity += corners;
        }
        upload(descriptorsLayers, descriptorLayers.data(), descriptorLayers.size());
        upload(patternPoints, points, 2 * kPatternSize);
        upload(patternPairs, pairs, 512);
        hostImage.reserve(std::size_t(layers.front().width) * layers.front().height);
        hostCorners.reserve(capacity);
        deviceCorners.reserve(capacity);
        check(cudaMemsetAsync(deviceCorners.get(), 0, capacity * sizeof(OrbCudaCorner), stream.handle), "initialize ORB corner workspace");
        check(cudaStreamSynchronize(stream.handle), "prepare ORB corner workspace");
        hostCounts.reserve(layers.size());
        deviceCounts.reserve(layers.size());
        reserveFeatures(featureCapacity);
    }

    void reserveFeatures(std::size_t size) {
        hostPatches.reserve(size);
        devicePatches.reserve(size);
        hostDescriptors.reserve(size);
        deviceDescriptors.reserve(size);
    }

    void enqueueIntegrals(const bool* needed=nullptr) {
        const auto queue=stream.handle;
        record(integralStart,queue);
        for(std::size_t i=0;i<layers.size();++i) {
            if(needed && !needed[i]) continue;
            const auto& level=layers[i];
            check(cudaMemsetAsync(level.integral.get(),0,(level.width+1)*sizeof(int),queue),"initialize ORB integral");
            integralRows<<<level.height,kScanThreads,0,queue>>>(level.image.get(),level.width,level.integral.get());
            const int tiles=(level.height+kIntegralTileRows-1)/kIntegralTileRows;
            integralColumns<<<dim3((level.width+32)/32,tiles),kTile,0,queue>>>(
                level.integral.get(),level.width,level.height,level.integralTails.get());
            if(tiles>1) addColumnCarries<<<dim3((level.width+32)/32,tiles-1),kTile,0,queue>>>(
                level.integral.get(),level.width,level.height,level.integralTails.get());
        }
        record(integralDone,queue);
        integralsQueued=true;
    }
};

OrbCuda::OrbCuda(const std::vector<OrbCudaLevel>& levels, int edge, int cellSize, int minimumThreshold,
    const int* points, const int* pairs, int featureCapacity)
    : impl_(std::make_unique<Impl>(levels, edge, cellSize, minimumThreshold, points, pairs, featureCapacity)) {}

OrbCuda::~OrbCuda() {
    // The camera worker for this frame may differ from the startup worker.
    cudaSetDevice(impl_->device);
    cudaStreamSynchronize(impl_->stream.handle);
    if(impl_->tailStream) cudaStreamSynchronize(impl_->tailStream->handle);
}

void OrbCuda::begin(const unsigned char* gray, std::size_t rowBytes, bool detect,bool earlyIntegrals) {
    auto& state = *impl_;
    check(cudaSetDevice(state.device), "restore CUDA ORB device");
    auto stream = state.stream.handle;
    state.timing = {};
    state.integralsQueued=false;
    const auto& base = state.layers.front();
    for (int y = 0; y < base.height; ++y)
        std::memcpy(state.hostImage.get() + std::size_t(y) * base.width, gray + y * rowBytes, base.width);
    record(state.start, stream);
    check(cudaMemcpyAsync(base.image.get(), state.hostImage.get(), std::size_t(base.width) * base.height,
        cudaMemcpyHostToDevice, stream), "upload ORB image");
    record(state.uploaded, stream);
    for (std::size_t i = 1; i < state.layers.size(); ++i) {
        const auto& previous = state.layers[i - 1];
        const auto& level = state.layers[i];
        resizeExact<<<grid(level.width, level.height), kTile, 0, stream>>>(previous.image.get(), previous.width,
            previous.height, level.image.get(), level.width, level.height, level.xs.get(), level.ys.get());
    }
    record(state.pyramidDone, stream);
    if (detect) {
        check(cudaMemsetAsync(state.deviceCounts.get(), 0, state.layers.size() * sizeof(int), stream), "clear ORB corner counts");
        for (std::size_t i = 0; i < state.layers.size(); ++i) {
            const auto& level = state.layers[i];
            const auto tiles = grid(level.width - 2 * state.edge, level.height - 2 * state.edge);
            fastScores<<<tiles, kTile, 0, stream>>>(level.image.get(), level.width, level.height,
                state.edge, state.minimumThreshold, level.scores.get());
            selectCorners<<<tiles, kTile, 0, stream>>>(level.scores.get(), level.width, level.height,
                state.edge, state.cellSize, state.deviceCorners.get() + level.cornerOffset,
                state.deviceCounts.get() + i, level.cornerCapacity);
        }
        record(state.fastDone, stream);
        check(cudaMemcpyAsync(state.hostCounts.get(), state.deviceCounts.get(), state.layers.size() * sizeof(int),
            cudaMemcpyDeviceToHost, stream), "download ORB corner counts");
        // A small prefix per level normally holds all compact candidates, so
        // counts and corners can cross the PCIe boundary in one synchronization.
        for (const auto& level : state.layers)
            check(cudaMemcpyAsync(state.hostCorners.get() + level.cornerOffset,
                state.deviceCorners.get() + level.cornerOffset, level.downloadCapacity * sizeof(OrbCudaCorner),
                cudaMemcpyDeviceToHost, stream), "download compact ORB corners");
        record(state.cornersDone, stream);
        // The host only needs the compact candidate prefix. The integrals read
        // the same pyramid and can run while the host filters/balances candidates.
        if(earlyIntegrals) state.enqueueIntegrals();
        check(cudaEventSynchronize(state.cornersDone.handle), "finish ORB FAST");
        bool copiedTail = false;
        for (std::size_t i = 0; i < state.layers.size(); ++i) {
            auto& level = state.layers[i];
            const int count = state.hostCounts.get()[i];
            if (count < 0 || count > level.cornerCapacity) throw OrbCudaError("CUDA ORB corner capacity exceeded");
            if (count <= level.downloadCapacity) continue;
            const auto offset = level.cornerOffset + level.downloadCapacity;
            if(!state.tailStream) state.tailStream=std::make_unique<Stream>();
            check(cudaMemcpyAsync(state.hostCorners.get() + offset, state.deviceCorners.get() + offset,
                (count - level.downloadCapacity) * sizeof(OrbCudaCorner), cudaMemcpyDeviceToHost,state.tailStream->handle),
                "download additional ORB corners");
            level.downloadCapacity = std::min(level.cornerCapacity, count + count / 8);
            copiedTail = true;
        }
        if (copiedTail) {
            record(state.tailDone,state.tailStream->handle);
            check(cudaEventSynchronize(state.tailDone.handle),"finish additional ORB corners");
        }
        state.timing.cornersDownloadMs=elapsed(state.fastDone,copiedTail?state.tailDone:state.cornersDone);
    } else check(cudaStreamSynchronize(stream), "finish CUDA ORB pyramid");
    check(cudaGetLastError(), "launch CUDA ORB pyramid/FAST");
    state.timing.uploadMs = elapsed(state.start, state.uploaded);
    state.timing.pyramidMs = elapsed(state.uploaded, state.pyramidDone);
    if (detect) {
        state.timing.fastMs = elapsed(state.pyramidDone, state.fastDone);
    }
}

const OrbCudaCorner* OrbCuda::corners(std::size_t level) const {
    return impl_->hostCorners.get() + impl_->layers[level].cornerOffset;
}

int OrbCuda::cornerCount(std::size_t level) const { return impl_->hostCounts.get()[level]; }

const OrbCudaDescriptor* OrbCuda::describe(const std::vector<OrbCudaPatch>& patches,bool keepDeviceRows) {
    auto& state = *impl_;
    state.describedSlot.reset();
    if (patches.empty()) return nullptr;
    state.reserveFeatures(patches.size());
    std::shared_ptr<Impl::DescriptorSlot> slot;
    if(keepDeviceRows) {
        for(auto& candidate:state.descriptorSlots) {
            if(candidate && candidate.use_count()!=1) continue;
            if(!candidate) candidate=std::make_shared<Impl::DescriptorSlot>();
            try {
                candidate->results.reserve(patches.size());
                candidate->indices.reserve(patches.size());
                slot=candidate;
            }
            catch(const OrbCudaError&) { /* Host rows remain available. */ }
            break;
        }
    }
    auto* deviceResults=slot?slot->results.get():state.deviceDescriptors.get();
    auto stream = state.stream.handle;
    std::copy(patches.begin(), patches.end(), state.hostPatches.get());
    check(cudaMemcpyAsync(state.devicePatches.get(), state.hostPatches.get(), patches.size() * sizeof(OrbCudaPatch),
        cudaMemcpyHostToDevice, stream), "upload selected ORB patches");
    if(!state.integralsQueued) {
        bool needed[16]{};
        for(const auto& patch:patches) needed[patch.level]=true;
        state.enqueueIntegrals(needed);
    }
    record(state.descriptorStart,stream);
    describePatches<<<static_cast<unsigned int>(patches.size()), 128, 0, stream>>>(state.descriptorsLayers.get(),
        state.devicePatches.get(), state.patternPoints.get(), state.patternPairs.get(), deviceResults);
    if(slot) compactDescriptorRows<<<1,kScanThreads,0,stream>>>(deviceResults,int(patches.size()),slot->indices.get());
    record(state.descriptorDone, stream);
    check(cudaMemcpyAsync(state.hostDescriptors.get(), deviceResults,
        patches.size() * sizeof(OrbCudaDescriptor), cudaMemcpyDeviceToHost, stream), "download ORB descriptors");
    record(state.end, stream);
    check(cudaGetLastError(), "launch CUDA ORB descriptors");
    check(cudaStreamSynchronize(stream), "finish CUDA ORB descriptors");
    state.timing.integralMs = elapsed(state.integralStart, state.integralDone);
    state.timing.descriptorMs = elapsed(state.descriptorStart, state.descriptorDone);
    state.timing.descriptorsDownloadMs = elapsed(state.descriptorDone, state.end);
    state.describedSlot=std::move(slot);
    state.describedCount=patches.size();
    return state.hostDescriptors.get();
}

std::shared_ptr<const OrbCudaDescriptorRows> OrbCuda::descriptorRows(std::size_t accepted) const {
    const auto& slot=impl_->describedSlot;
    if(!slot || !accepted || accepted>impl_->describedCount) return {};
    static_assert(offsetof(OrbCudaDescriptor,bytes)%8==0 && sizeof(OrbCudaDescriptor)%8==0);
    const auto* bytes=reinterpret_cast<const unsigned char*>(slot->results.get())+offsetof(OrbCudaDescriptor,bytes);
    return std::make_shared<OrbCudaDescriptorRows>(OrbCudaDescriptorRows{
        std::shared_ptr<const unsigned char>(slot,bytes),
        accepted<impl_->describedCount?std::shared_ptr<const int>(slot,slot->indices.get()):nullptr,
        accepted,sizeof(OrbCudaDescriptor),impl_->device});
}
const OrbCudaTiming& OrbCuda::timing() const { return impl_->timing; }

} // namespace cgmli::features::detail
