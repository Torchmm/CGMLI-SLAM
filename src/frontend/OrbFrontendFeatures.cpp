#include <frontend/OrbFrontendFeatures.hpp>
#include <visual/MultiFrame.hpp>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <chrono>
#include <glog/logging.h>
#include <stdexcept>
#include <future>
#include <algorithm>
#include <mutex>

namespace cgmli { namespace features {
OrbFrontendFeatures::OrbFrontendFeatures(size_t cameras, const OrbExtractorOptions& options)
    : options_(options), extractors_(cameras), extracted_(cameras,0) {
    if(!cameras) throw std::invalid_argument("Frontend requires at least one camera");
    for(size_t i=0;i<cameras;++i) mutexes_.push_back(std::make_unique<std::mutex>());
}
std::vector<std::unique_lock<std::mutex>> OrbFrontendFeatures::lockAll() const {
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(mutexes_.size());
    // A fixed increasing lock order prevents configuration/clear deadlocks. Extraction
    // acquires only its own camera lock, never another camera's lock or mutable state.
    for(const auto& mutex:mutexes_) locks.emplace_back(*mutex);
    return locks;
}
OrbExtractorOptions OrbFrontendFeatures::options() const {
    std::lock_guard<std::mutex> lock(*mutexes_[0]); return options_;
}
void OrbFrontendFeatures::setOptions(const OrbExtractorOptions& options) {
    auto locks=lockAll();
    for(const auto& extractor:extractors_) if(extractor)
        throw std::logic_error("Clear frontend and map before changing the ORB descriptor profile");
    options_=options;
}
void OrbFrontendFeatures::clear() {
    auto locks=lockAll(); for(auto& extractor:extractors_) extractor.reset();
    std::fill(extracted_.begin(),extracted_.end(),0);
}
double OrbFrontendFeatures::referenceFocal(const MultiFrame& frame) const {
    if(options_.virtualFocalLength>0) return options_.virtualFocalLength;
    // Select ONE rig reference, regardless of which camera thread runs first. Using
    // each camera's fu independently would change angular support across cameras.
    if(!frame.geometry(0)) throw std::invalid_argument("Missing reference camera geometry");
    Eigen::VectorXd intrinsics; frame.geometry(0)->getIntrinsics(intrinsics);
    if(intrinsics.size()==0 || !std::isfinite(intrinsics[0]) || intrinsics[0]<=0)
        throw std::invalid_argument("Configure an explicit ORB virtual focal length for this rig");
    return intrinsics[0];
}
std::string OrbFrontendFeatures::profile(const OrbFeatureExtractor& extractor) const {
    std::ostringstream text; text.imbue(std::locale::classic());
    text<<extractor.getDefaultName()<<";f="<<std::setprecision(std::numeric_limits<double>::max_digits10)
        <<extractor.virtualFocalLength()<<";scale="<<options_.scaleFactor
        <<";pole="<<options_.gravityParallelSinThreshold;
    return text.str();
}
double OrbFrontendFeatures::initialiseExtractor(size_t camera,const MultiFrame& frame) {
    if(frame.numFrames()!=extractors_.size() || camera>=extractors_.size() || !frame.geometry(camera))
        throw std::invalid_argument("Frontend frame does not match configured camera count/geometry");
    auto options=options_;
#ifdef CGMLI_ENABLE_CUDA_ORB
    static const bool useCuda=[] {
        const char* backend=std::getenv("CGMLI_ORB_BACKEND");
        return !backend || std::strcmp(backend,"cpu")!=0;
    }();
    options.useCuda=useCuda;
#endif
    if(!std::isfinite(options.virtualFocalLength) || options.virtualFocalLength<0)
        throw std::invalid_argument("Invalid ORB reference focal length");
    if(options.descriptorMode==OrbExtractorOptions::DescriptorMode::GravityCameraAware)
        options.virtualFocalLength=referenceFocal(frame);
    auto& extractor=extractors_[camera];
    if(extractor && (extractor->camera().get()!=frame.geometry(camera).get()
        || extractor->options().virtualFocalLength!=options.virtualFocalLength))
        throw std::invalid_argument("Camera calibration changed; clear frontend and map before reconfiguration");
    const bool cold=!extractor; const auto initializationStarted=std::chrono::steady_clock::now();
    if(!extractor) extractor=std::make_shared<OrbFeatureExtractor>(frame.geometry(camera),options);
    return cold?std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-initializationStarted).count():0;
}
void OrbFrontendFeatures::prepare(const MultiFrame& calibration,bool parallel) {
    const auto started=std::chrono::steady_clock::now();
    if(calibration.numFrames()!=extractors_.size()) throw std::invalid_argument("ORB preparation rig size mismatch");
    // Validate the whole rig before starting any worker. Each worker then owns
    // only its extractor; futures join before calibration goes out of scope.
    for(size_t camera=0;camera<extractors_.size();++camera)
        if(!calibration.geometry(camera)) throw std::invalid_argument("Missing ORB preparation camera geometry");
    // Warm only OpenCV's dispatch on the startup thread before camera workers
    // compete for first-use library initialization. Never insert synthetic data
    // into a Frame, create descriptors, or consume a state ID.
    static std::once_flag fastWarmup;
    std::call_once(fastWarmup,[]() {
        const auto start=std::chrono::steady_clock::now(); cv::Mat image(128,128,CV_8UC1);
        for(int y=0;y<image.rows;++y) for(int x=0;x<image.cols;++x) image.at<uchar>(y,x)=uchar((x*17)^(y*31));
        std::vector<cv::KeyPoint> scratch;
        cv::FAST(image,scratch,20,false,cv::FastFeatureDetector::TYPE_9_16);
        cv::FAST(image,scratch,20,true,cv::FastFeatureDetector::TYPE_9_16);
        cv::FAST(image,scratch,7,true,cv::FastFeatureDetector::TYPE_9_16);
        LOG(INFO)<<"[Timing][FastWarmup] calls=3 total_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    });
    const auto prepareCamera=[&](size_t camera) {
        const auto cameraStarted=std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(*mutexes_[camera]);
        const double geometryMs=initialiseExtractor(camera,calibration);
        const auto masksStarted=std::chrono::steady_clock::now();
        extractors_[camera]->prepareStaticCaches();
        const double masksMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-masksStarted).count();
        LOG(INFO)<<"[Timing][OrbPrepare] camera="<<camera<<" geometry_ms="<<geometryMs<<" masks_ms="<<masksMs
            <<" backend="<<(extractors_[camera]->cudaReady()?"cuda":"cpu")
            <<" layer_fast="<<options_.layerFast<<" boundary_fast="<<options_.boundaryFast
            <<" adaptive_geometry="<<options_.adaptiveGeometryCandidates<<" grouped_balancing="<<options_.groupedBalancing
            <<" total_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cameraStarted).count();
    };
    if(parallel && extractors_.size()>1) {
        std::vector<std::future<void>> workers; workers.reserve(extractors_.size());
        for(size_t camera=0;camera<extractors_.size();++camera)
            workers.emplace_back(std::async(std::launch::async,prepareCamera,camera));
        // If get() throws, outstanding async futures still join during unwinding.
        for(auto& worker:workers) worker.get();
    } else for(size_t camera=0;camera<extractors_.size();++camera) prepareCamera(camera);
    LOG(INFO)<<"[Timing][OrbPrepareWall] cameras="<<extractors_.size()<<" parallel="<<parallel
        <<" wall_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}
bool OrbFrontendFeatures::detectAndDescribe(size_t camera, MultiFrame& frame,
    const kinematics::Transformation& T_WC, const std::vector<cv::KeyPoint>* supplied) {
    const auto started=std::chrono::steady_clock::now();
    if(camera>=extractors_.size()) throw std::invalid_argument("ORB camera index out of range");
    std::lock_guard<std::mutex> lock(*mutexes_[camera]);
    const bool cold=!extractors_[camera],firstExtraction=!extracted_[camera];
    const double initializationMs=initialiseExtractor(camera,frame);
    auto& extractor=extractors_[camera]; const auto& options=extractor->options();
    // The supplied camera pose already includes this camera's rig extrinsic. Applying
    // T_SC a second time here would rotate gravity incorrectly in non-reference views.
    if(options.descriptorMode!=OrbExtractorOptions::DescriptorMode::Standard)
        updateOrbGravity(*extractor,T_WC);
    std::vector<cv::KeyPoint> points=supplied?*supplied:std::vector<cv::KeyPoint>();
    cv::Mat descriptors;
    // One call shares its pyramid between detection and description. Keypoints may be
    // removed by geometry/support checks, so publish rows/associations only afterwards.
    extractor->detectAndCompute(frame.image(camera),cv::noArray(),points,descriptors,supplied!=nullptr);
    frame.setDetector(camera,extractor); frame.setExtractor(camera,extractor);
    frame.resetKeypoints(camera,points); frame.resetDescriptors(camera,descriptors);
    frame.setCudaDescriptorRows(camera,extractor->cudaDescriptorRows());
    frame.setDescriptorProfile(camera,profile(*extractor));
    frame.computeBackProjections(camera);
    validateDescriptors(frame,camera);
    extracted_[camera]=1;
    const double totalMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    if(VLOG_IS_ON(1) || firstExtraction || cold || totalMs>200 || frame.id()%30==0) {
        const auto& timing=extractor->lastTiming();
        LOG(INFO)<<"[Timing][Extract] frame="<<frame.id()<<" camera="<<camera<<" cold_ms="<<initializationMs
            <<" backend="<<(timing.usedCuda?"cuda":"cpu")
            <<" device_descriptor_rows="<<(frame.cudaDescriptorRows(camera)?descriptors.rows:0)
            <<" pyramid_ms="<<timing.pyramidMs<<" detect_ms="<<timing.detectionMs<<" describe_ms="<<timing.descriptorMs
            <<" gpu_upload_ms="<<timing.gpuUploadMs<<" gpu_pyramid_ms="<<timing.gpuPyramidMs
            <<" gpu_fast_ms="<<timing.gpuFastMs<<" gpu_corners_download_ms="<<timing.gpuCornersDownloadMs
            <<" gpu_integral_ms="<<timing.gpuIntegralMs<<" gpu_descriptor_ms="<<timing.gpuDescriptorMs
            <<" gpu_descriptors_download_ms="<<timing.gpuDescriptorsDownloadMs
            <<" fast_ms="<<timing.fastMs<<" candidate_sort_ms="<<timing.candidateSortMs
            <<" fast_raw_scan_ms="<<timing.fastRawScanMs<<" fast_nms_scan_ms="<<timing.fastNmsScanMs
            <<" fast_boundary_ms="<<timing.fastBoundaryMs<<" fast_bucket_ms="<<timing.fastBucketMs
            <<" fast_fallback_ms="<<timing.fastFallbackMs<<" fast_raw_corners="<<timing.fastRawCorners
            <<" fast_raw_pixels="<<timing.fastRawPixels<<" balance_heap_pushes="<<timing.balanceHeapPushes
            <<" geometry_memo_hits="<<timing.geometryMemoHits<<" geometry_expansion_rounds="<<timing.geometryExpansionRounds
            <<" fast_fallback_calls="<<timing.fastFallbackCalls
            <<" empty_mask_cells="<<timing.emptyMaskCells<<" quota_fallback_cells="<<timing.quotaFallbackCells
            <<" zero_valid_with_candidates_cells="<<timing.geometryFallbackCells
            <<" budget_skipped_cells="<<timing.budgetSkippedCells
            <<" candidate_filter_ms="<<timing.candidateFilterMs<<" balance_ms="<<timing.balanceMs
            <<" mask_cache_hit="<<timing.maskCacheHit<<" fast_candidates="<<timing.fastCandidates
            <<" geometry_evals="<<timing.geometryEvaluations<<" warp_reused="<<timing.geometryCacheHits
            <<" gray_integrals="<<timing.grayIntegralBuilds
            <<" fast_calls="<<timing.fastCalls<<" gray_integrals32="<<timing.grayIntegral32Builds
            <<" keypoints="<<points.size()<<" total_ms="<<totalMs;
    }
    return true;
}
void OrbFrontendFeatures::validateDescriptors(const MultiFrame& frame,size_t camera) {
    const auto& descriptors=frame.descriptors(camera);
    const auto rows=static_cast<int>(frame.numKeypoints(camera));
    if(descriptors.rows!=rows || (rows>0 && (descriptors.empty()
        || descriptors.type()!=CV_8UC1 || descriptors.cols!=kDescriptorBytes)))
        throw std::invalid_argument("Frontend requires one 32-byte ORB row per retained keypoint");
}
void OrbFrontendFeatures::validateFrame(const MultiFrame& frame) const {
    auto locks=lockAll();
    if(frame.numFrames()!=extractors_.size()) throw std::invalid_argument("ORB rig size mismatch");
    for(size_t camera=0;camera<extractors_.size();++camera) {
        validateDescriptors(frame,camera);
        if(!extractors_[camera] || frame.geometry(camera).get()!=extractors_[camera]->camera().get()
            || frame.descriptorProfile(camera)!=profile(*extractors_[camera]))
            throw std::invalid_argument("Missing/incompatible ORB descriptor profile; regenerate frame/map descriptors");
    }
}
int OrbFrontendFeatures::hammingDistance(const uchar* first,const uchar* second) {
    if(!first || !second) throw std::invalid_argument("Null ORB descriptor");
    int result=0;
    for(int offset=0;offset<kDescriptorBytes;offset+=4) {
        // memcpy avoids alignment and strict-aliasing assumptions on cv::Mat rows.
        uint32_t a,b; std::memcpy(&a,first+offset,4); std::memcpy(&b,second+offset,4);
        uint32_t bits=a^b;
        bits-=(bits>>1)&0x55555555u;
        bits=(bits&0x33333333u)+((bits>>2)&0x33333333u);
        bits=(bits+(bits>>4))&0x0f0f0f0fu;
        result+=static_cast<int>((bits*0x01010101u)>>24);
    }
    return result;
}
} }
