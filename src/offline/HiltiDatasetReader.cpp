#include "HiltiDatasetReader.hpp"
#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <chrono>

namespace cgmli { namespace offline {
namespace {
Time projectTime(int64_t ns) {
    if(ns<0 || uint64_t(ns)/1000000000ULL>std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Sensor timestamp exceeds project Time range");
    Time result; result.fromNSec(uint64_t(ns)); return result;
}
}
HiltiDatasetReader::HiltiDatasetReader(const std::filesystem::path& path,const ViParameters& parameters,
    const HiltiReaderOptions& options):binary_(path,options.topics),parameters_(parameters),options_(options) {
    if(!parameters_.imu.use || parameters_.nCameraSystem.numCameras()!=4)
        throw std::runtime_error("Hilti reader requires enabled IMU and four calibrated cameras in front/left/rear/right LEFT order");
    for(size_t c=0;c<4;++c) if(!parameters_.nCameraSystem.cameraType(c).isUsed)
        throw std::runtime_error("All four Hilti left cameras must be enabled in the estimator config");
    if(!std::isfinite(parameters_.camera.image_delay) || !std::isfinite(parameters_.imu.max_sample_gap)
        || parameters_.imu.max_sample_gap<=0) throw std::runtime_error("Invalid camera timing or IMU gap limit");
    if(options_.startupPrerollNs<20000000 || options_.startupPrerollNs>5000000000LL)
        throw std::runtime_error("IMU preroll must be between 20 ms and 5 seconds");
    packets_=makeImagePackets(binary_,options_.bundleToleranceNs,&stats_.grouping);
    if(packets_.empty()) throw std::runtime_error("No complete four-camera bundle within configured timestamp_tolerance");
    const auto& records=binary_.records(); const auto& imu=binary_.imu();
    while(first_<packets_.size() && corrected(packets_[first_])<projectTime(records[imu.front()].sensorNs)+Duration(0.02)) ++first_;
    end_=packets_.size();
    while(end_>first_ && corrected(packets_[end_-1])+Duration(0.02)>projectTime(records[imu.back()].sensorNs)) --end_;
    stats_.uncoveredImagePackets=first_+packets_.size()-end_;
    if(first_==end_) throw std::runtime_error("No image epoch has IMU prehistory and 20 ms look-ahead");
    const auto start=std::max<int64_t>(0,int64_t(corrected(packets_[first_]).toNSec())-options_.startupPrerollNs);
    imuCursor_=size_t(std::lower_bound(imu.begin(),imu.end(),start,[&](size_t i,int64_t t){return records[i].sensorNs<t;})-imu.begin());
    if(imuCursor_) --imuCursor_; // Retain the left interpolation endpoint.
    LOG(INFO) << "[Dataset] Indexed " << path.u8string() << "; records=" << records.size()
        << " usable_epochs=" << end_-first_ << " uncovered_epochs=" << stats_.uncoveredImagePackets
        << " first_image_t_ns=" << packets_[first_].timestampNs << " last_image_t_ns=" << packets_[end_-1].timestampNs
        << " bundle_tolerance_ns=" << options_.bundleToleranceNs << " IMU_preroll_ns=" << options_.startupPrerollNs;
    for(size_t c=0;c<4;++c) LOG(INFO) << "[Dataset] camera=" << c << " topic=" << options_.topics.cameras[c]
        << " records=" << binary_.cameras()[c].size();
    LOG(INFO) << "[Dataset] imu_topic=" << options_.topics.imu << " samples=" << imu.size()
        << " lidar_topic=" << options_.topics.lidar << " scans=" << binary_.lidar().size();
    LOG(INFO) << "[Dataset] Timing mode=mean-midpoint; complete_bundles=" << packets_.size()
        << " unmatched_images=" << stats_.grouping.unmatchedImages << " max_bundle_span_ns=" << stats_.grouping.maxBundleSpanNs
        << " max_camera_offset_ns=" << stats_.grouping.maxCameraOffsetNs
        << " bearing_time_compensation_enabled=" << parameters_.camera.online_calibration.estimate_td_camera;
    if(stats_.grouping.unmatchedImages) LOG(WARNING) << "[Dataset] Dropped unmatched image records=" << stats_.grouping.unmatchedImages;
    if(stats_.uncoveredImagePackets) LOG(WARNING) << "[Dataset] Trimmed " << stats_.uncoveredImagePackets
        << " image epochs outside IMU coverage; required future support=20 ms";
}
HiltiDatasetReader::~HiltiDatasetReader() { stopStreaming(); }
Time HiltiDatasetReader::corrected(const ImagePacketIndex& packet) const {
    // Coverage only: image input applies image_delay once to the bundle mean.
    return parameters_.camera.correctedImageTime(projectTime(packet.timestampNs));
}
void HiltiDatasetReader::setImuCallback(ImuCallback callback) {
    if(started_) throw std::logic_error("Configure callbacks before streaming");
    imuCallback_=std::move(callback);
}
void HiltiDatasetReader::setImagesCallback(ImagesCallback callback) {
    if(started_) throw std::logic_error("Configure callbacks before streaming");
    imagesCallback_=std::move(callback);
}
void HiltiDatasetReader::setLidarCallback(LidarCallback callback) {
    if(started_) throw std::logic_error("Configure callbacks before streaming");
    lidarCallback_=std::move(callback);
}
void HiltiDatasetReader::startStreaming() {
    if(started_) throw std::logic_error("A dataset reader can stream only once");
    if(!imuCallback_ || !imagesCallback_ || !lidarCallback_)
        throw std::logic_error("Connect image, IMU and LiDAR consumers before streaming");
    LOG(INFO) << "[Dataset] Starting sensor stream";
    started_=true; streaming_=true;
    try { worker_=std::thread(&HiltiDatasetReader::processing,this); }
    catch(...) { streaming_=false; throw; }
}
void HiltiDatasetReader::stopStreaming() {
    requestStop(); if(worker_.joinable()) worker_.join();
}
double HiltiDatasetReader::completion() const { return double(delivered_.load())/double(end_-first_); }
void HiltiDatasetReader::rethrowIfFailed() const {
    // streaming_'s release/acquire ordering publishes failure_ after worker exit.
    if(!isStreaming() && failure_) std::rethrow_exception(failure_);
}
const HiltiReaderStatistics& HiltiDatasetReader::statistics() const {
    if(worker_.joinable()) throw std::logic_error("Join the dataset reader before reading statistics");
    return stats_;
}
void HiltiDatasetReader::processing() {
    try {
        const auto& records=binary_.records(); const auto& imu=binary_.imu(); const auto& lidar=binary_.lidar();
        size_t lidarCursor=0; int64_t lastImu=-1; double lidarReadMs=0;
        const auto milliseconds=[](const auto& beginning) { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-beginning).count(); };
        const auto emitLidar=[&]() {
            const auto beginning=std::chrono::steady_clock::now();
            const auto scan=binary_.readLidar(lidar[lidarCursor++],options_.decodeLidarPoints);
            lidarReadMs+=milliseconds(beginning);
            if(stop_) return;
            if(!lidarCallback_(scan) && !stop_) throw std::runtime_error("LiDAR consumer rejected scan");
            ++stats_.lidarScans; stats_.lidarPoints+=scan.points.size(); stats_.invalidLidarPoints+=scan.invalidPoints;
        };
        for(size_t p=first_;p<end_ && !stop_;++p) {
            const auto packetStarted=std::chrono::steady_clock::now();
            lidarReadMs=0; double imuReadMs=0,imuSubmitMs=0;
            const auto& packet=packets_[p]; std::map<size_t,cv::Mat> images; std::map<size_t,Time> originalTimes;
            for(const auto record:packet.records) {
                auto image=binary_.readImage(record); const auto camera=parameters_.nCameraSystem.cameraGeometry(image.camera);
                if(image.image.cols!=int(camera->imageWidth()) || image.image.rows!=int(camera->imageHeight()))
                    throw std::runtime_error("Image dimensions differ from calibration: "+image.topic);
                images.emplace(image.camera,std::move(image.image));
                originalTimes.emplace(image.camera,projectTime(image.timestampNs));
            }
            if(stop_) break;
            const double imageReadMs=milliseconds(packetStarted); const auto submitStarted=std::chrono::steady_clock::now();
            // Submit the image before its future IMU support. With bounded input
            // queues the control thread must be able to consume this interval.
            if(images.size()!=4 || originalTimes.size()!=4) throw std::runtime_error("Incomplete four-camera bundle");
            if(!imagesCallback_(projectTime(packet.timestampNs),images,originalTimes)) {
                if(stop_) break;
                throw std::runtime_error("Estimator rejected image input");
            }
            const double imageSubmitMs=milliseconds(submitStarted);
            const auto required=corrected(packet)+Duration(0.02);
            while(!stop_ && (lastImu<0 || projectTime(lastImu)<required)) {
                if(imuCursor_>=imu.size()) throw std::runtime_error("Unexpected IMU exhaustion");
                const auto next=records[imu[imuCursor_]].sensorNs;
                while(!stop_ && lidarCursor<lidar.size() && records[lidar[lidarCursor]].sensorNs<=next) emitLidar();
                if(stop_) break;
                const auto imuStarted=std::chrono::steady_clock::now();
                const auto sample=binary_.readImu(imu[imuCursor_++]);
                imuReadMs+=milliseconds(imuStarted);
                if(lastImu>=0 && double(sample.timestampNs-lastImu)*1e-9>parameters_.imu.max_sample_gap+1e-9)
                    throw std::runtime_error("Physical IMU gap exceeds configured max_sample_gap");
                const Eigen::Vector3d acceleration(sample.acceleration[0],sample.acceleration[1],sample.acceleration[2]);
                const Eigen::Vector3d gyro(sample.gyro[0],sample.gyro[1],sample.gyro[2]);
                const auto callbackStarted=std::chrono::steady_clock::now();
                if(!imuCallback_(projectTime(sample.timestampNs),acceleration,gyro)) {
                    if(stop_) break;
                    throw std::runtime_error("Estimator rejected IMU input");
                }
                imuSubmitMs+=milliseconds(callbackStarted);
                lastImu=sample.timestampNs; ++stats_.imuSamples;
            }
            if(!stop_) {
                ++stats_.imagePackets; stats_.imageObservations+=packet.records.size(); ++delivered_;
                const double ioMs=imageReadMs+imuReadMs+lidarReadMs;
                if(VLOG_IS_ON(1) || stats_.imagePackets==1 || stats_.imagePackets%30==0 || ioMs>200 || imageSubmitMs+imuSubmitMs>200)
                    LOG(INFO)<<"[Timing][Input] epoch="<<stats_.imagePackets<<" image_read_ms="<<imageReadMs
                        <<" imu_read_ms="<<imuReadMs<<" lidar_read_ms="<<lidarReadMs
                        <<" submit_block_ms="<<imageSubmitMs+imuSubmitMs<<" total_ms="<<milliseconds(packetStarted)
                        <<" lidar_decode="<<options_.decodeLidarPoints;
                VLOG(2) << "[Dataset] epoch=" << stats_.imagePackets << " image_t_ns=" << packet.timestampNs
                    << " cameras=" << packet.records.size() << " bundle_span_ns=" << packet.lastImageNs-packet.firstImageNs
                    << " original_t_ns=" << packet.imageTimestampNs[0] << ',' << packet.imageTimestampNs[1]
                    << ',' << packet.imageTimestampNs[2] << ',' << packet.imageTimestampNs[3] << " last_imu_t_ns=" << lastImu;
            }
        }
        // Remaining scans are still delivered, without queuing unused IMU after
        // the final image. Cancellation does not continue reading the entire tail.
        while(!stop_ && lidarCursor<lidar.size()) emitLidar();
    } catch(...) { failure_=std::current_exception(); }
    if(failure_) LOG(ERROR) << "[Dataset] Sensor stream failed after epochs=" << stats_.imagePackets
        << "; exception will be rethrown on the control thread";
    else LOG(INFO) << "[Dataset] " << (stop_?"Stopped":"Reached EOF") << "; epochs=" << stats_.imagePackets
        << " images=" << stats_.imageObservations << " IMU=" << stats_.imuSamples
        << " lidar_scans=" << stats_.lidarScans << " lidar_points=" << stats_.lidarPoints
        << " invalid_lidar_points=" << stats_.invalidLidarPoints;
    streaming_=false;
}
} }
