#include "CsvOffline.hpp"
#include <offline/HiltiDatasetReader.hpp>
#include <common/ViParametersReader.hpp>
#include <pipeline/ThreadedSlam.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <glog/logging.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <set>
#include <sstream>
#include <cctype>
#include <thread>
#include <mutex>
#include <memory>
#include <ctime>
#include <string>
#include <vector>

namespace {
namespace fs=std::filesystem;
using namespace cgmli::offline;

fs::path localLogRoot(int argc,char** argv) {
    if(const char* directory=std::getenv("CGMLI_LOG_DIR"))
        if(*directory) return fs::absolute(fs::u8path(directory));
    // Locate the checkout from the supplied estimator file, executable or CWD.
    // This keeps VS launches and command-line launches in the same log folder.
    std::vector<fs::path> starts;
    if(argc>=3 && std::string(argv[1]).rfind("--",0)!=0) starts.push_back(fs::absolute(fs::u8path(argv[1])).parent_path());
    starts.push_back(fs::absolute(fs::u8path(argv[0])).parent_path());
    starts.push_back(fs::current_path());
    for(auto directory:starts) for(;;) {
        if(fs::is_directory(directory/"src"/"frontend") && fs::is_directory(directory/"config")) return directory/"logs";
        const auto parent=directory.parent_path(); if(parent==directory || parent.empty()) break;
        directory=parent;
    }
    return fs::absolute(fs::u8path(argv[0])).parent_path()/"logs"; // Standalone deployment.
}

/// A glog sink rather than stderr redirection: preserve console output and write
/// every severity to one native-path file, including paths containing Chinese.
/// glog calls send() from arbitrary workers while holding its own locks; never
/// call LOG/CHECK or estimator code here. A mutex protects the file and flush clock.
class LocalFileLog final : public google::LogSink {
public:
    LocalFileLog(const fs::path& root,int argc,char** argv) {
        fs::create_directories(root);
        const auto now=std::chrono::system_clock::now();
        const auto stamp=std::chrono::system_clock::to_time_t(now); std::tm calendar{};
#ifdef _WIN32
        localtime_s(&calendar,&stamp);
#else
        localtime_r(&stamp,&calendar);
#endif
        std::ostringstream name; name<<"run_"<<std::put_time(&calendar,"%Y%m%d_%H%M%S")<<'_'
            <<std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()%1000;
        fs::path run;
        for(size_t serial=0;;++serial) {
            run=root/(name.str()+"_"+std::to_string(serial));
            if(fs::create_directory(run)) break; // Exclusive directory allocation across processes.
        }
        const auto file=fs::absolute(run/"slam.log");
        stream_.open(file,std::ios::binary);
        if(!stream_) throw std::runtime_error("Cannot create local log: "+file.u8string());
        // Publish before replay starts so a still-running process can be inspected.
        // UTF-8 manifest; previous run directories are retained unchanged.
        std::ofstream latest(root/"latest.txt",std::ios::binary|std::ios::trunc);
        latest<<file.u8string()<<'\n'; latest.close();
        if(!latest) throw std::runtime_error("Cannot write latest log pointer");
        std::ofstream metadata(run/"arguments.txt",std::ios::binary);
        for(int i=0;i<argc;++i) metadata<<i<<": "<<argv[i]<<'\n';
        metadata.close(); if(!metadata) throw std::runtime_error("Cannot save log arguments");
        lastFlush_=std::chrono::steady_clock::now();
        google::AddLogSink(this);
        LOG(INFO)<<"[Logging] file="<<file.u8string()<<" latest="<<(root/"latest.txt").u8string();
    }
    ~LocalFileLog() override {
        google::RemoveLogSink(this); // Wait for in-flight sink callbacks before closing storage.
        std::lock_guard<std::mutex> lock(mutex_); stream_.flush();
    }
    void send(google::LogSeverity severity,const char*,const char* filename,int line,
        const std::tm* timestamp,const char* message,size_t length) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stream_<<google::LogSink::ToString(severity,filename,line,timestamp,message,length)<<'\n';
        const auto now=std::chrono::steady_clock::now();
        // Batch normal INFO writes to avoid disk flushes on every timing line.
        // Warnings/errors/fatal messages are made visible immediately.
        if(severity>=google::GLOG_WARNING || now-lastFlush_>=std::chrono::seconds(1)) {
            stream_.flush(); lastFlush_=now;
        }
        if(!stream_ && !writeFailureReported_) {
            writeFailureReported_=true;
            std::cerr<<"[Logging] Local file write failed; check disk space and permissions\n";
        }
    }
private:
    std::ofstream stream_;
    std::mutex mutex_;
    std::chrono::steady_clock::time_point lastFlush_;
    bool writeFailureReported_=false;
};

void showDebugViews(const std::map<std::string,cv::Mat>& updates,std::map<std::string,cv::Mat>& latest,std::set<std::string>& windows) {
    for(const auto& view:updates) if(!view.second.empty()) {
        latest[view.first]=view.second;
        if(windows.insert(view.first).second) { cv::namedWindow(view.first,cv::WINDOW_NORMAL); cv::resizeWindow(view.first,960,600); }
        cv::imshow(view.first,view.second);
    }
}
void saveDebugViews(const fs::path& output,const std::map<std::string,cv::Mat>& views,size_t& serial) {
    if(views.empty()) return;
    try {
        fs::path folder;
        do { std::ostringstream name; name<<"snapshot_"<<std::setfill('0')<<std::setw(6)<<++serial; folder=output/"debug"/name.str(); }
        while(fs::exists(folder));
        fs::create_directories(folder);
        for(const auto& view:views) if(!view.second.empty()) {
            std::string name=view.first;
            for(char& c:name) if(!std::isalnum(static_cast<unsigned char>(c)) && c!='-' && c!='_') c='_';
            std::vector<uchar> bytes; if(!cv::imencode(".png",view.second,bytes)) throw std::runtime_error("Debug PNG encoding failed");
            // Native filesystem paths preserve Chinese output directories on Windows.
            std::ofstream file(folder/(name+".png"),std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));
            if(!file) throw std::runtime_error("Debug PNG write failed");
        }
        LOG(INFO) << "[Debug] Saved displayed views to " << folder.u8string();
    } catch(const std::exception& error) { LOG(WARNING) << "[Debug] Snapshot failed: " << error.what(); }
}

int inspect(const fs::path& path) {
    LOG(INFO) << "[Inspect] Reading and validating sensor records; dataset=" << path.u8string();
    HiltiBinReader reader(path); size_t images=0,imus=0,scans=0,points=0;
    for(size_t i=0;i<reader.records().size();++i) {
        const auto& record=reader.records()[i];
        if(record.camera>=0) { reader.readImage(i); ++images; }
        else if(record.kind==RecordKind::Imu) { reader.readImu(i); ++imus; }
        else { const auto scan=reader.readLidar(i); ++scans; points+=scan.points.size(); }
        if((i+1)%10000==0) LOG(INFO) << "[Inspect] Validated records=" << i+1;
    }
    LOG(INFO) << "[Inspect] Complete; images=" << images << " IMU=" << imus << " lidar_scans=" << scans << " valid_points=" << points;
    return 0;
}

int runHilti(const fs::path& config,const fs::path& dataset,const fs::path& output) {
    const auto started=std::chrono::steady_clock::now();
    LOG(INFO) << "[App] Hilti VIO; config=" << config.u8string() << " dataset=" << dataset.u8string()
        << " output=" << output.u8string() << " verbosity=" << FLAGS_v;
    // 1. Read the estimator configuration and construct the dataset producer.
    cgmli::ViParameters parameters{}; cgmli::ViParametersReader parameterReader(config.u8string());
    if(!parameterReader.getParameters(parameters)) throw std::runtime_error("Cannot read estimator configuration");
    std::atomic<size_t> publications{0},visualUpdates{0},lidarScans{0};
    if(!std::isfinite(parameters.camera.timestamp_tolerance) || parameters.camera.timestamp_tolerance<0
        || parameters.camera.timestamp_tolerance>0.05) throw std::runtime_error("Hilti timestamp_tolerance must be between 0 and 0.05 seconds");
    HiltiReaderOptions readerOptions;
    readerOptions.decodeLidarPoints=false; // No LiDAR factors/point consumer in this VIO application.
    readerOptions.bundleToleranceNs=static_cast<int64_t>(std::llround(parameters.camera.timestamp_tolerance*1e9));
    HiltiDatasetReader datasetReader(dataset,parameters,readerOptions);
    fs::create_directories(output);

    // 2. Set up the estimator and its output callback before starting any input.
    cgmli::ThreadedSlam estimator(parameters,""); estimator.setBlocking(true);
    estimator.setFinalTrajectoryCsvFile((output/"trajectory.csv").u8string());
    estimator.setMapCsvFile((output/"map.csv").u8string());
    estimator.setOptimisedGraphCallback([&](const cgmli::State& state,const cgmli::TrackingState& tracking,
        std::shared_ptr<const cgmli::AlignedMap<cgmli::StateId,cgmli::State>>,std::shared_ptr<const cgmli::MapPointVector>) {
        if(!state.T_WS.T().allFinite()) throw std::runtime_error("Non-finite published pose");
        ++publications; if(tracking.trackingQuality!=cgmli::TrackingQuality::Lost) ++visualUpdates;
    });

    // 3. Connect reader callbacks, as in the OKVIS synchronous application.
    datasetReader.setImuCallback([&](const cgmli::Time& t,const Eigen::Vector3d& acceleration,const Eigen::Vector3d& gyro) {
        return estimator.addImuMeasurement(t,acceleration,gyro);
    });
    datasetReader.setImagesCallback([&](const cgmli::Time& t,const std::map<size_t,cv::Mat>& images,const std::map<size_t,cgmli::Time>& originalTimes) {
        return estimator.addImagesWithTimestamps(t,images,originalTimes);
    });
    datasetReader.setLidarCallback([&](const LidarScan&) {
        // Scans are decoded and counted. Native point-cloud fusion is deferred;
        // do not route a scan through the unavailable single-vector LiDAR API.
        ++lidarScans; return true;
    });
    LOG(INFO) << "[App] Four left cameras + IMU VIO; LiDAR scan metadata is counted; unused point payload decoding disabled";
    if(parameters.output.display_matches || parameters.output.display_overhead)
        LOG(INFO) << "[Debug] Views enabled; matches="<<parameters.output.display_matches
            <<" trajectory="<<parameters.output.display_overhead<<"; SPACE=pause/resume, S=save PNGs, ESC=stop";
    std::map<std::string,cv::Mat> latestViews; std::set<std::string> windows;
    bool paused=false,cancelled=false; size_t snapshotSerial=0;

    // 4. Stream on the reader worker; estimation and GUI stay on this thread.
    const auto replayStarted=std::chrono::steady_clock::now();
    LOG(INFO)<<"[Timing][Startup] total_ms="<<std::chrono::duration<double,std::milli>(replayStarted-started).count();
    try {
        datasetReader.startStreaming(); int progress=-1,idleAtEof=0;
        auto lastProgress=std::chrono::steady_clock::now();
        const int drainLimit=2*(parameters.estimator.backend_max_retries+1)+4;
        while(datasetReader.isStreaming() || estimator.hasPendingWork()) {
            datasetReader.rethrowIfFailed();
            const bool processed=!paused && estimator.processFrame();
            if(estimator.hasFatalError()) throw std::runtime_error(estimator.lastError());
            std::map<std::string,cv::Mat> images; estimator.display(images);
            showDebugViews(images,latestViews,windows);
            const int key=latestViews.empty()?-1:(cv::waitKey(5)&0xff);
            if(key==27 || key=='q' || key=='Q') { cancelled=true; LOG(INFO) << "[App] Replay cancelled by user"; break; }
            if(key==' ') { paused=!paused; LOG(INFO) << "[Debug] " << (paused?"Paused":"Resumed"); }
            if(key=='s' || key=='S') saveDebugViews(output,latestViews,snapshotSerial);
            const int next=int(datasetReader.completion()*100.0);
            const auto now=std::chrono::steady_clock::now();
            if(progress<0 || (next>progress && (next==100 || now-lastProgress>=std::chrono::seconds(5)))
                || now-lastProgress>=std::chrono::seconds(10)) {
                progress=next; lastProgress=now;
                LOG(INFO) << "[App] Progress=" << progress << "% accepted_updates=" << publications.load()
                    << " visual_updates=" << visualUpdates.load() << " lidar_scans=" << lidarScans.load()
                    << " elapsed_s=" << std::chrono::duration<double>(now-started).count();
                LOG(INFO)<<"[Timing][Replay] accepted_updates="<<publications.load()
                    <<" replay_elapsed_s="<<std::chrono::duration<double>(now-replayStarted).count();
            }
            // EOF does not mean the last optimization has committed. Retry/join
            // within the backend's configured bound instead of abandoning it.
            if(!paused && !datasetReader.isStreaming() && estimator.hasPendingWork() && !processed) {
                if(++idleAtEof>drainLimit) throw std::runtime_error("Final frame did not commit: "+estimator.lastError());
            } else idleAtEof=0;
            if(!processed) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch(...) {
        // Release blocking input callbacks before joining the dataset worker.
        datasetReader.requestStop(); estimator.stopThreading(); datasetReader.stopStreaming(); throw;
    }
    datasetReader.requestStop(); estimator.stopThreading(); datasetReader.stopStreaming();
    datasetReader.rethrowIfFailed();

    // 5. Export only after input/optimization/publication workers have stopped.
    if(parameters.estimator.do_final_ba) estimator.doFinalBa();
    estimator.writeFinalTrajectoryCsv();
    if(!estimator.saveMap()) throw std::runtime_error("Map export failed");
    const auto& stats=datasetReader.statistics();
    LOG(INFO) << "[App] Replay finished; images=" << stats.imageObservations << " IMU=" << stats.imuSamples
        << " lidar_scans=" << lidarScans.load() << " uncovered_epochs=" << stats.uncoveredImagePackets
        << " unmatched_images=" << stats.grouping.unmatchedImages
        << " accepted_updates=" << publications.load() << " visual_updates=" << visualUpdates.load()
        << " elapsed_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    if(!cancelled) {
        std::map<std::string,cv::Mat> images; estimator.display(images); showDebugViews(images,latestViews,windows);
    }
    if(!windows.empty() && !cancelled) {
        LOG(INFO) << "[Debug] Replay complete; S=save final views, ESC/ENTER=close";
        for(;;) {
            const int key=cv::waitKey(30)&0xff;
            if(key==27 || key==13 || key=='q' || key=='Q') break;
            if(key=='s' || key=='S') saveDebugViews(output,latestViews,snapshotSerial);
            bool visible=false;
            for(const auto& window:windows) try { visible=visible || cv::getWindowProperty(window,cv::WND_PROP_VISIBLE)>=1; } catch(const cv::Exception&) {}
            if(!visible) break;
        }
    }
    if(!windows.empty()) cv::destroyAllWindows();
    if(publications.load()==0) throw std::runtime_error("No accepted states; check stationary initialization and calibration");
    if(visualUpdates.load()==0) throw std::runtime_error("No successful visual tracking; only inertial/bootstrap updates were accepted");
    LOG(INFO) << "[App] Completed successfully; output=" << output.u8string();
    return 0;
}

int application(int argc,char** argv) {
    google::InitGoogleLogging(argv[0]); FLAGS_logtostderr=true;
    struct LoggingShutdown { ~LoggingShutdown() { google::ShutdownGoogleLogging(); } } shutdown;
    std::unique_ptr<LocalFileLog> localLog; // Outlives catch(), including its failure log.
    try {
        localLog=std::make_unique<LocalFileLog>(localLogRoot(argc,argv),argc,argv);
        // Honor the same environment switch with or without a gflags-enabled glog build.
        if(const char* setting=std::getenv("GLOG_v")) {
            const std::string value(setting); size_t used=0;
            const int level=std::stoi(value,&used);
            if(used!=value.size() || level<0) throw std::invalid_argument("GLOG_v must be a non-negative integer");
            FLAGS_v=level;
        }
        if(argc==3 && std::string(argv[1])=="--inspect") return inspect(fs::u8path(argv[2]));
        if(argc>1 && std::string(argv[1])=="--csv") return runCsvOffline(argc-1,argv+1);
        if(argc>=4 && fs::u8path(argv[2]).extension()==".csv") return runCsvOffline(argc,argv);
        if((argc!=3 && argc!=4) || std::string(argv[1]).rfind("--",0)==0) {
            std::cerr<<"Usage: CGMLI-SLAM estimator.yaml dataset_directory [output_directory]\n"
                <<"       CGMLI-SLAM --inspect dataset_directory\n";
            return 2;
        }
        const auto dataset=fs::u8path(argv[2]);
        const auto output=argc==4?fs::u8path(argv[3]):(fs::is_directory(dataset)?dataset:dataset.parent_path())/"results";
        return runHilti(fs::u8path(argv[1]),dataset,output);
    } catch(const std::exception& error) { LOG(ERROR) << "[App] Failed: " << error.what(); return 1; }
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
    std::vector<std::string> storage; storage.reserve(size_t(argc));
    for(int i=0;i<argc;++i) storage.push_back(std::filesystem::path(argv[i]).u8string());
    std::vector<char*> arguments; for(auto& value:storage) arguments.push_back(value.data());
    return application(argc,arguments.data());
}
#else
int main(int argc,char** argv) { return application(argc,argv); }
#endif
