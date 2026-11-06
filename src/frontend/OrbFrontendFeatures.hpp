#pragma once

#include <visual/features/OrbFrameAdapter.hpp>
#include <mutex>
#include <string>

namespace cgmli { namespace features {

/// Feature-only part of Frontend, independent of the legacy graph/loop-closure API.
/// Each camera owns one extractor and one lock. prepare() builds calibration-only
/// caches at startup; lazy creation remains available to standalone Frame callers.
class OrbFrontendFeatures {
public:
    static constexpr int kDescriptorBytes = 32;
    static constexpr int kDescriptorBits = 8 * kDescriptorBytes;
    explicit OrbFrontendFeatures(size_t cameras, const OrbExtractorOptions& options = {});
    /// A geometry-only MultiFrame is sufficient; no image, pose or gravity is used.
    /// Call after setOptions(), before starting sensor/estimator workers.
    void prepare(const MultiFrame& calibration, bool parallel = true);
    bool detectAndDescribe(size_t camera, MultiFrame& frame,
        const kinematics::Transformation& T_WC, const std::vector<cv::KeyPoint>* supplied = nullptr);
    OrbExtractorOptions options() const;
    /// Configure before preparation/extraction, or after clearing frontend AND map.
    /// Existing descriptors cannot be silently mixed with a different sampling profile.
    void setOptions(const OrbExtractorOptions& options);
    void clear();
    /// Validate every row before any legacy pointer-based matcher reads its bytes.
    /// Profile includes mode, smoothing backend, common focal scale and pyramid scale.
    void validateFrame(const MultiFrame& frame) const;
    static void validateDescriptors(const MultiFrame& frame, size_t camera);
    /// Alignment-independent, fixed-length 256-bit Hamming used by legacy guided loops.
    static int hammingDistance(const uchar* first, const uchar* second);
private:
    /// Caller holds the camera lock; returns geometry-construction time (zero if ready).
    double initialiseExtractor(size_t camera, const MultiFrame& frame);
    double referenceFocal(const MultiFrame& frame) const;
    std::string profile(const OrbFeatureExtractor& extractor) const;
    std::vector<std::unique_lock<std::mutex>> lockAll() const;
    OrbExtractorOptions options_;
    std::vector<std::shared_ptr<OrbFeatureExtractor>> extractors_;
    std::vector<unsigned char> extracted_; ///< Separate bytes avoid vector<bool> cross-camera races.
    mutable std::vector<std::unique_ptr<std::mutex>> mutexes_;
};
} }
