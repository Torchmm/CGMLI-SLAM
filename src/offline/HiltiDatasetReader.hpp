#pragma once
#include "HiltiBinReader.hpp"
#include <common/Parameters.hpp>
#include <atomic>
#include <exception>
#include <functional>
#include <thread>

namespace cgmli { namespace offline {
struct HiltiReaderOptions {
    bool decodeLidarPoints=true; // False only for consumers that explicitly need scan metadata/counts.
    HiltiTopics topics;
    int64_t bundleToleranceNs=25000000; // Maximum span of the four exposure midpoints.
    int64_t startupPrerollNs=600000000;
};
struct HiltiReaderStatistics {
    size_t imagePackets=0,imageObservations=0,imuSamples=0,lidarScans=0;
    size_t lidarPoints=0,invalidLidarPoints=0,uncoveredImagePackets=0;
    ImageGroupingStatistics grouping;
};

// Dataset producer, following the callback-based synchronous OKVIS app pattern.
// Only this worker reads binary payloads. The application owns processFrame,
// display, estimator shutdown and output; the reader never calls the estimator.
class HiltiDatasetReader {
public:
    using ImuCallback=std::function<bool(const Time&,const Eigen::Vector3d&,const Eigen::Vector3d&)>;
    // Mean source time, four owned images, and their unmodified source timestamps.
    using ImagesCallback=std::function<bool(const Time&,const std::map<size_t,cv::Mat>&,const std::map<size_t,Time>&)>;
    using LidarCallback=std::function<bool(const LidarScan&)>;
    HiltiDatasetReader(const std::filesystem::path&,const ViParameters&,const HiltiReaderOptions& = {});
    ~HiltiDatasetReader();
    HiltiDatasetReader(const HiltiDatasetReader&)=delete;
    HiltiDatasetReader& operator=(const HiltiDatasetReader&)=delete;
    // Configure callbacks before startStreaming; do not mutate them during replay.
    void setImuCallback(ImuCallback);
    void setImagesCallback(ImagesCallback);
    void setLidarCallback(LidarCallback);
    void startStreaming();
    bool isStreaming() const { return streaming_.load(); }
    double completion() const;
    void requestStop() { stop_=true; }
    // On early exit first requestStop(), then shut down the receiver's blocking
    // input queues, then stopStreaming(). Otherwise a blocked callback cannot join.
    void stopStreaming();
    // A worker error is visible after isStreaming() becomes false. Never silently
    // treat a truncated record, callback failure or IMU gap as successful EOF.
    void rethrowIfFailed() const;
    const HiltiReaderStatistics& statistics() const; // Call after stopStreaming().
private:
    void processing();
    Time corrected(const ImagePacketIndex&) const;
    HiltiBinReader binary_;
    ViParameters parameters_;
    HiltiReaderOptions options_;
    std::vector<ImagePacketIndex> packets_;
    size_t first_=0,end_=0,imuCursor_=0;
    HiltiReaderStatistics stats_;
    ImuCallback imuCallback_;
    ImagesCallback imagesCallback_;
    LidarCallback lidarCallback_;
    std::atomic_bool streaming_{false},stop_{false};
    std::atomic<size_t> delivered_{0};
    bool started_=false;
    std::thread worker_;
    std::exception_ptr failure_;
};
} }
