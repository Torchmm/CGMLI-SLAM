#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace cgmli { namespace offline {

// Exact source timestamps in integer nanoseconds. Bag arrival time is retained
// for diagnostics only; it never substitutes for a missing sensor header time.
enum class RecordKind : uint32_t { Image=1, CompressedImage=2, PointCloud2=3, Imu=4, Ancillary=5 };
struct HiltiTopics {
    // Camera IDs must use this same order in the selected estimator calibration.
    std::array<std::string,4> cameras{{"/oak_cam_front/left/image_raw", "/oak_cam_left/left/image_raw",
        "/oak_cam_rear/left/image_raw", "/oak_cam_right/left/image_raw"}};
    std::string imu="/imu/data", lidar="/rslidar_points";
};
struct ReadLimits {
    uint64_t maximumMetadataBytes=16*1024*1024, maximumPayloadBytes=256*1024*1024;
    size_t maximumRecords=5000000, maximumImagePixels=16000000, maximumScanPoints=2000000;
};
struct RecordIndex {
    RecordKind kind{};
    size_t fileId=0; // Directory mode: per-topic file, selected through summary.json.
    uint64_t offset=0, payloadBytes=0;
    uint32_t metadataBytes=0;
    int64_t sensorNs=0, bagNs=0;
    int camera=-1; // Only selected Image/CompressedImage records have a camera ID.
    uint64_t recordBytes=0; // CSV fast path: checked against the binary header on load.
    bool deferredHeader=false;
};
struct ImageSample {
    int64_t timestampNs=0, bagNs=0;
    size_t camera=0;
    std::string topic, frameId, sourceEncoding;
    cv::Mat image; // Owned MONO8 or BGR8/BGRA8; raw RGB is explicitly converted.
};
struct ImuSample {
    int64_t timestampNs=0, bagNs=0;
    std::string topic, frameId;
    std::array<double,37> original{};
    std::array<double,3> gyro{}, acceleration{};
    // Quaternion/covariance are preserved, never used as a fabricated pose prior.
};
struct CloudField { std::string name; uint32_t offset=0, datatype=0, count=0; };
struct LidarPoint {
    float x=0, y=0, z=0, intensity=0;
    uint16_t ring=0;
    int64_t timestampNs=0; // Absolute point time, not the scan header or a float32 epoch.
};
struct LidarScan {
    int64_t timestampNs=0, bagNs=0, firstPointNs=0, lastPointNs=0;
    std::string topic, frameId;
    uint32_t width=0, height=0, pointStep=0, rowStep=0;
    bool bigEndian=false, dense=false;
    size_t invalidPoints=0;
    std::vector<CloudField> fields;
    std::vector<uint8_t> originalData; // Keep padding and unknown fields for future LiDAR consumers.
    std::vector<LidarPoint> points; // Valid sensor-L points in metres; no deskew/extrinsic applied.
};
struct IndexStatistics {
    bool reusedCsv=false;
    uint64_t totalRecords=0, selectedRecords=0, fileBytes=0;
    std::map<std::string,uint64_t> topicRecords;
};

// Reader for hilti_bag_to_bin.py's HILTBIN1/version-1 format. Directory mode reuses
// the completed converter CSV; missing/stale/malformed CSV falls back to a bounded
// header/JSON scan that seeks over payloads. Accepts either a legacy combined file or
// the converter's per-topic directory plus completed summary.json manifest.
// A single decoded record owns its buffers;
// the complete image/point-cloud sequence is never loaded into memory at once.
class HiltiBinReader {
public:
    HiltiBinReader(const std::filesystem::path&,const HiltiTopics& = {},const ReadLimits& = {});
    const std::vector<RecordIndex>& records() const { return records_; }
    const std::array<std::vector<size_t>,4>& cameras() const { return cameras_; }
    const std::vector<size_t>& imu() const { return imu_; }
    const std::vector<size_t>& lidar() const { return lidar_; }
    const IndexStatistics& statistics() const { return statistics_; }
    ImageSample readImage(size_t record);
    ImuSample readImu(size_t record);
    LidarScan readLidar(size_t record,bool decodePoints=true);
private:
    struct RawRecord;
    RawRecord load(size_t,bool readPayload=true);
    void index();
    bool indexCsv();
    std::filesystem::path path_;
    HiltiTopics topics_;
    ReadLimits limits_;
    std::vector<std::filesystem::path> files_;
    std::vector<std::ifstream> inputs_;
    std::vector<uint64_t> fileSizes_;
    std::vector<std::string> fileTopics_;
    IndexStatistics statistics_;
    std::vector<RecordIndex> records_;
    std::array<std::vector<size_t>,4> cameras_;
    std::vector<size_t> imu_,lidar_;
};

struct ImagePacketIndex {
    int64_t timestampNs=0, firstImageNs=0, lastImageNs=0;
    std::array<int64_t,4> imageTimestampNs{}; // Original exposure midpoints, in camera-ID order.
    std::vector<size_t> records;
};
struct ImageGroupingStatistics {
    size_t unmatchedImages=0; // Images excluded because no complete four-camera bundle was found.
    int64_t maxBundleSpanNs=0,maxCameraOffsetNs=0;
};
// Match one unused image per camera within a bounded midpoint-time span. Incomplete
// groups are discarded; images are never reused. The node time is their integer-ns
// mean, rounded to the nearest ns. This is a synchronous approximation, not a claim
// of equal physical exposure midpoints or verified start-of-exposure trigger IDs.
std::vector<ImagePacketIndex> makeImagePackets(const HiltiBinReader&,int64_t toleranceNs=25000000,
    ImageGroupingStatistics* statistics=nullptr);

} }
