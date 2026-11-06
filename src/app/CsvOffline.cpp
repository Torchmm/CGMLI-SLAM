#include "CsvOffline.hpp"
#include <pipeline/ThreadedSlam.hpp>
#include <common/ViParametersReader.hpp>
#include <opencv2/imgcodecs.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> result; std::string value; bool quoted=false;
    for(size_t i=0;i<line.size();++i) {
        const char c=line[i];
        if(c=='"') { if(quoted && i+1<line.size() && line[i+1]=='"') { value+='"'; ++i; } else quoted=!quoted; }
        else if(c==',' && !quoted) { result.push_back(value); value.clear(); }
        else if(c!='\r') value+=c;
    }
    if(quoted) throw std::invalid_argument("Unterminated CSV quote"); result.push_back(value); return result;
}
bool row(std::ifstream& file,std::vector<std::string>& values) {
    std::string line;
    while(std::getline(file,line)) {
        if(line.size()>=3 && static_cast<unsigned char>(line[0])==0xef
            && static_cast<unsigned char>(line[1])==0xbb && static_cast<unsigned char>(line[2])==0xbf) line.erase(0,3);
        if(line.empty() || line.front()=='#') continue;
        values=fields(line);
        if(values.front()=="timestamp_ns" || values.front()=="timestamp") continue;
        return true;
    }
    if(file.bad()) throw std::runtime_error("CSV input read failed");
    return false;
}
cgmli::Time timestamp(const std::string& text) {
    if(text.empty() || !std::all_of(text.begin(),text.end(),[](unsigned char c){return std::isdigit(c)!=0;}))
        throw std::invalid_argument("CSV timestamps must be unsigned integer nanoseconds");
    const auto ns=std::stoull(text);
    if(ns/1000000000ULL>std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("CSV timestamp exceeds project Time range");
    cgmli::Time time; time.fromNSec(ns); return time;
}
double number(const std::string& text) {
    size_t used=0; const double value=std::stod(text,&used);
    if(used!=text.size() || !std::isfinite(value)) throw std::invalid_argument("Non-finite or invalid IMU number"); return value;
}
cv::Mat readImage(const std::filesystem::path& path) {
    // Native filesystem paths preserve UTF-8 manifest names on Windows; OpenCV
    // decodes owned bytes instead of opening a narrow, code-page-dependent path.
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::runtime_error("Cannot open camera image: "+path.u8string());
    const std::vector<uchar> bytes((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
    if(input.bad() || bytes.empty()) throw std::runtime_error("Cannot read camera image: "+path.u8string());
    return cv::imdecode(bytes,cv::IMREAD_UNCHANGED);
}
}

int runCsvOffline(int argc,char** argv) {
    if(argc<4 || argc>7) {
        std::cerr<<"Usage: CGMLI-SLAM config.yaml images.csv imu.csv|- [trajectory.csv] [map.csv] [orb_vocabulary.yml]\n"
            <<"       CGMLI-SLAM estimator.yaml hilti_directory [output_directory]\n"
            <<"images.csv: timestamp_ns,path_camera0,path_camera1,... (paths relative to manifest)\n"
            <<"imu.csv: timestamp_ns,wx,wy,wz,ax,ay,az (rad/s and m/s^2)\n";
        return 2;
    }
    try {
        LOG(INFO) << "[App] CSV replay; config=" << argv[1] << " images=" << argv[2] << " imu=" << argv[3];
        cgmli::ViParameters parameters{}; cgmli::ViParametersReader reader(argv[1]);
        if(!reader.getParameters(parameters)) throw std::runtime_error("Configuration could not be read");
        std::ifstream images(std::filesystem::u8path(argv[2])); if(!images) throw std::runtime_error("Cannot open image manifest");
        std::ifstream imu; if(parameters.imu.use) { imu.open(std::filesystem::u8path(argv[3])); if(!imu) throw std::runtime_error("Cannot open IMU CSV"); }
        // Captures must outlive Slam, including exceptional unwinding and joins.
        std::atomic<size_t> publications{0},visualUpdates{0};
        cgmli::ThreadedSlam slam(parameters,argc>6?argv[6]:""); slam.setBlocking(true);
        const int maxProcessingAttempts=2*(parameters.estimator.backend_max_retries+1)+4;
        slam.setFinalTrajectoryCsvFile(argc>4?argv[4]:"trajectory.csv"); slam.setMapCsvFile(argc>5?argv[5]:"map.csv");
        slam.setOptimisedGraphCallback([&](const cgmli::State& state,const cgmli::TrackingState& tracking,
            std::shared_ptr<const cgmli::AlignedMap<cgmli::StateId,cgmli::State>>,std::shared_ptr<const cgmli::MapPointVector>) {
            if(!state.T_WS.T().allFinite()) throw std::runtime_error("Non-finite published pose"); ++publications;
            if(tracking.trackingQuality!=cgmli::TrackingQuality::Lost) ++visualUpdates;
        });
        std::vector<std::string> imageRow,imuRow; bool hasImu=parameters.imu.use && row(imu,imuRow);
        cgmli::Time previousImage,previousImu; bool firstImage=true,firstImu=true;
        const auto base=std::filesystem::absolute(std::filesystem::u8path(argv[2])).parent_path();
        while(row(images,imageRow)) {
            if(imageRow.size()!=parameters.nCameraSystem.numCameras()+1) throw std::runtime_error("Image row camera count differs from calibration");
            const auto time=timestamp(imageRow.front());
            if(!firstImage && time<=previousImage) throw std::runtime_error("Image timestamps must strictly increase");
            firstImage=false; previousImage=time; std::map<size_t,cv::Mat> frames;
            for(size_t c=0;c<parameters.nCameraSystem.numCameras();++c) {
                if(imageRow[c+1].empty() || imageRow[c+1]=="-") continue;
                const auto image=readImage(base/std::filesystem::u8path(imageRow[c+1]));
                if(image.empty()) throw std::runtime_error("Cannot decode camera image: "+imageRow[c+1]); frames.emplace(c,image);
            }
            if(!slam.addImages(time,frames)) throw std::runtime_error("Image input rejected");
            const auto required=parameters.camera.correctedImageTime(time)+cgmli::Duration(0.02);
            if(parameters.imu.use) {
                bool covered=false;
                while(hasImu) {
                    if(imuRow.size()!=7) throw std::runtime_error("IMU rows require timestamp plus six measurements");
                    const auto t=timestamp(imuRow.front());
                    if(!firstImu && t<=previousImu) throw std::runtime_error("IMU timestamps must strictly increase");
                    firstImu=false; previousImu=t;
                    if(!slam.addImuMeasurement(t,Eigen::Vector3d(number(imuRow[4]),number(imuRow[5]),number(imuRow[6])),
                        Eigen::Vector3d(number(imuRow[1]),number(imuRow[2]),number(imuRow[3])))) throw std::runtime_error("IMU input rejected");
                    slam.processFrame();
                    if(slam.hasFatalError()) throw std::runtime_error(slam.lastError());
                    hasImu=row(imu,imuRow);
                    if(t>=required) { covered=true; break; }
                }
                if(!covered && previousImu<required) throw std::runtime_error("IMU CSV ends before the last image look-ahead interval");
            }
            // Consume this input while its solve can overlap the next extraction.
            for(int attempts=0;slam.hasPendingImages() && attempts<maxProcessingAttempts;++attempts) slam.processFrame();
            if(slam.hasFatalError()) throw std::runtime_error(slam.lastError());
            if(slam.hasPendingImages()) throw std::runtime_error("Frame input was not consumed: "+slam.lastError());
            std::map<std::string,cv::Mat> display; slam.display(display);
        }
        // EOF drains all accepted work before stopping, final BA and exporting.
        for(int attempts=0;slam.hasPendingWork() && attempts<maxProcessingAttempts;++attempts) slam.processFrame();
        if(slam.hasFatalError()) throw std::runtime_error(slam.lastError());
        if(slam.hasPendingWork()) throw std::runtime_error("Final frame did not commit: "+slam.lastError());
        slam.stopThreading();
        if(parameters.estimator.do_final_ba) slam.doFinalBa();
        slam.writeFinalTrajectoryCsv(); if(!slam.saveMap()) throw std::runtime_error("Map CSV export failed");
        if(publications.load()==0) throw std::runtime_error("No accepted states; inspect initialization and calibration");
        if(visualUpdates.load()==0) throw std::runtime_error("Only inertial/bootstrap states were accepted; no successful visual SLAM tracking");
        LOG(INFO) << "[App] CSV replay completed; accepted_updates=" << publications.load() << " visual_updates=" << visualUpdates.load(); return 0;
    } catch(const std::exception& error) { LOG(ERROR) << "[App] CSV replay failed: " << error.what(); return 1; }
}
