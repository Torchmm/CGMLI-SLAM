#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cgmli::features::detail {

// Plain data boundary between the C++ feature extractor and nvcc.
struct OrbCudaLevel {
    int width;
    int height;
    const int* invalidIntegral;
};

struct OrbCudaCorner {
    std::uint32_t packed;
    int x() const { return int(packed & 4095u); }
    int y() const { return int((packed >> 12) & 4095u); }
    int score() const { return int(packed >> 24); }
};

struct OrbCudaPatch {
    double x, y;
    double a, b, c, d; // Row-major reference-plane warp.
    int level;
    bool aligned;
};

struct alignas(8) OrbCudaDescriptor {
    float a, c; // First column of the final warp for the diagnostic angle.
    unsigned char bytes[32];
    bool valid;
};
struct OrbCudaDescriptorRows;

struct OrbCudaTiming {
    double uploadMs = 0, pyramidMs = 0, fastMs = 0, cornersDownloadMs = 0;
    double integralMs = 0, descriptorMs = 0, descriptorsDownloadMs = 0;
};

class OrbCudaError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// One camera's stream and reusable workspace. Calls are serialized by its owner.
class OrbCuda {
public:
    OrbCuda(const std::vector<OrbCudaLevel>& levels, int edge, int cellSize,
        int minimumThreshold, const int* patternPoints, const int* patternPairs, int featureCapacity);
    ~OrbCuda();
    OrbCuda(const OrbCuda&) = delete;
    OrbCuda& operator=(const OrbCuda&) = delete;

    void begin(const unsigned char* gray, std::size_t rowBytes, bool detect,bool earlyIntegrals=false);
    const OrbCudaCorner* corners(std::size_t level) const;
    int cornerCount(std::size_t level) const;
    const OrbCudaDescriptor* describe(const std::vector<OrbCudaPatch>& patches,bool keepDeviceRows=false);
    std::shared_ptr<const OrbCudaDescriptorRows> descriptorRows(std::size_t accepted) const;
    const OrbCudaTiming& timing() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cgmli::features::detail
