#include <pipeline/ThreadedSlam.hpp>
#include <backend/BackendFactors.hpp>
#include <backend/FrameStore.hpp>
#include <factor/BodyImuFactor.hpp>
#include <Eigen/Eigenvalues>
#include <opencv2/imgproc.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <iterator>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cgmli {
    namespace {
        backend::BackendOptions backendOptions(const ViParameters& p) {
            backend::BackendOptions options;
            if(p.estimator.window_size<2) throw std::invalid_argument("window_size must be at least 2");
            options.windowSize = static_cast<size_t>(p.estimator.window_size);
            options.maxIterations = p.estimator.realtime_max_iterations; options.numThreads = p.estimator.realtime_num_threads;
            options.maxSolverTimeSeconds = p.estimator.enforce_realtime ? p.estimator.realtime_time_limit : std::numeric_limits<double>::max();
            options.exportVisualHistory = p.estimator.do_loop_closures || p.estimator.do_final_ba;
            options.visualNisThreshold = p.estimator.visual_nis_threshold;
            options.parallelNormalEquations = p.estimator.parallel_marginalization_accumulation;
            options.maxImuIntervalSeconds=p.estimator.max_imu_interval_seconds;
            options.incrementalImu=p.estimator.incremental_imu;
            options.preserveOldestVisualInformation=p.estimator.preserve_oldest_visual_information;
            options.cacheVisualPairs=p.estimator.cache_visual_pairs;
            const auto& visualBatch=p.estimator.backend_visual_batch;
            if(visualBatch=="cuda") options.visualBatchMode=backend::VisualBatchMode::Cuda;
            else if(visualBatch=="hybrid") options.visualBatchMode=backend::VisualBatchMode::Hybrid;
            else if(visualBatch=="schur") options.visualBatchMode=backend::VisualBatchMode::Schur;
            else if(visualBatch=="audit") options.visualBatchMode=backend::VisualBatchMode::Audit;
            else if(visualBatch!="cpu") throw std::invalid_argument("backend_visual_batch must be cpu, cuda, hybrid, schur, or audit");
            if(p.estimator.backend_dense_schur) options.linearSolver=backend::BackendLinearSolver::DenseSchur;
            options.validate(); return options;
        }
        constexpr double overlap = 0.02;
        const char* descriptorModeName(features::OrbExtractorOptions::DescriptorMode mode) {
            switch(mode) {
            case features::OrbExtractorOptions::DescriptorMode::Standard: return "Standard";
            case features::OrbExtractorOptions::DescriptorMode::GravityAligned: return "GravityAligned";
            case features::OrbExtractorOptions::DescriptorMode::GravityCameraAware: return "GravityCameraAware";
            default: return "unknown";
            }
        }
        const char* trackingName(TrackingQuality quality) {
            switch (quality) {
            case TrackingQuality::Good: return "good";
            case TrackingQuality::Marginal: return "marginal";
            case TrackingQuality::Lost: return "lost";
            default: return "unknown";
            }
        }
        Time before(const Time& time, double seconds) { return time.toSec() < seconds ? Time(0, 0) : time - Duration(seconds); }
    }
    ThreadedSlam::ThreadedSlam(ViParameters& p, std::string vocabulary)
        : parameters_(p), options_(backendOptions(parameters_)), estimator_(options_),
        frontend_(p.nCameraSystem.numCameras()), visualizer_(parameters_), vocabularyPath_(std::move(vocabulary)) {
        initialize();
    }
    ThreadedSlam::~ThreadedSlam() { try { stopThreading(); } catch (const std::exception& e) { LOG(ERROR) << e.what(); } }
    void ThreadedSlam::setError(std::string text) {
        std::lock_guard<std::mutex> lock(errorMutex_);
        if (faulted_) return; // Preserve the first terminal cause while old workers drain.
        const bool changed = text != error_;
        error_ = std::move(text); if (changed && !error_.empty()) LOG(WARNING) << "[VIO] " << error_;
    }
    void ThreadedSlam::setFatalError(std::string text) {
        std::lock_guard<std::mutex> lock(errorMutex_);
        if (faulted_) return;
        error_=std::move(text); faulted_=true; LOG(ERROR)<<"[VIO] Terminal input fault: "<<error_;
    }
    std::string ThreadedSlam::lastError() const 
    { 
        std::lock_guard<std::mutex> lock(errorMutex_);
        return error_;
    }


    void ThreadedSlam::initialize() {
        if (!parameters_.nCameraSystem.numCameras()) throw Exception("A calibrated camera rig is required");
        if (!std::isfinite(parameters_.camera.image_delay)) throw Exception("Image delay must be finite");
        if(parameters_.frontend.max_tracking_seeds<0 || !std::isfinite(parameters_.frontend.seed_max_untracked_seconds)
            || parameters_.frontend.seed_max_untracked_seconds<0) throw Exception("Invalid tracking seed budget/lifetime");
        if (!parameters_.nCameraSystem.overlapComputationValid()) parameters_.nCameraSystem.computeOverlaps();
        const auto& imu = parameters_.imu;
        const auto& health = parameters_.estimator;
        if (health.backend_max_retries<0 || health.backend_max_retries>10
            || !std::isfinite(health.visual_nis_threshold) || health.visual_nis_threshold<=0
            || !std::isfinite(health.max_visual_imu_rotation_error) || health.max_visual_imu_rotation_error<=0 || health.max_visual_imu_rotation_error>=3.141592653589793
            || !std::isfinite(health.max_visual_imu_position_error) || health.max_visual_imu_position_error<=0
            || !std::isfinite(health.max_visual_imu_position_error_rate) || health.max_visual_imu_position_error_rate<0)
            throw Exception("Invalid VIO health/retry parameters");
        if (health.max_consecutive_lightweight_frames<0)
            throw Exception("max_consecutive_lightweight_frames must be nonnegative");
        if (imu.use && (!(imu.g > 0) || !(imu.sigma_g_c > 0) || !(imu.sigma_a_c > 0)
            || !(imu.sigma_gw_c > 0) || !(imu.sigma_aw_c > 0) || !(imu.sigma_bg > 0) || !(imu.sigma_ba > 0)
            || !(imu.g_max > 0) || !(imu.a_max > 0) || !imu.g0.allFinite() || !imu.a0.allFinite()
            || !imu.s_a.allFinite() || (imu.s_a.array() <= 0).any()
            || !std::isfinite(imu.max_sample_gap) || imu.max_sample_gap<=0)) throw Exception("Invalid IMU calibration/noise parameters");
        if (parameters_.estimator.realtime_min_iterations<0 || parameters_.estimator.realtime_min_iterations>options_.maxIterations)
            throw Exception("Invalid minimum solver iteration budget");
        if ((parameters_.estimator.do_loop_closures || parameters_.estimator.do_final_ba)
            && (parameters_.estimator.full_graph_iterations < 1 || parameters_.estimator.full_graph_num_threads < 1))
            throw Exception("Invalid global optimization budget");
        const auto& calibration = parameters_.camera.online_calibration;
        const auto positiveFinite = [](double value) { return std::isfinite(value) && value > 0; };
        if ((calibration.do_extrinsics || calibration.estimate_td_camera) && !imu.use)
            throw Exception("Online camera calibration requires the IMU");
        if (calibration.do_extrinsics && (!positiveFinite(calibration.sigma_r) || !positiveFinite(calibration.sigma_alpha)
            || !positiveFinite(calibration.max_rig_translation_change_m) || !positiveFinite(calibration.max_rig_rotation_change_rad)))
            throw Exception("Invalid online rig calibration prior or limits");
        if (!std::isfinite(calibration.td_camera_initial_s) || (calibration.estimate_td_camera
            && (!positiveFinite(calibration.sigma_td_camera_s) || !positiveFinite(calibration.max_td_change_s))))
            throw Exception("Invalid online camera time offset prior or limits");
        if (parameters_.estimator.do_final_ba && parameters_.camera.online_calibration.do_extrinsics_final_ba
            && (!(parameters_.camera.online_calibration.sigma_r_final_ba > 0) || !(parameters_.camera.online_calibration.sigma_alpha_final_ba > 0)))
            throw Exception("Final rig calibration requires positive calibration prior sigmas");
        if (!imu.use) {
            bool metricBaseline = false;
            for (size_t a = 0; a < parameters_.nCameraSystem.numCameras(); ++a) for (size_t b = a + 1; b < parameters_.nCameraSystem.numCameras(); ++b)
                if (parameters_.nCameraSystem.cameraType(a).isUsed && parameters_.nCameraSystem.cameraType(b).isUsed)
                    metricBaseline = metricBaseline || (parameters_.nCameraSystem.T_SC(a)->r() - parameters_.nCameraSystem.T_SC(b)->r()).norm() > 1e-6;
            if (!metricBaseline) throw Exception("Metric visual SLAM requires a nonzero calibrated rig baseline or enabled IMU");
        }
        features::OrbExtractorOptions orbOptions;
        orbOptions.maxFeatures = parameters_.frontend.max_num_keypoints; orbOptions.pyramidLevels = parameters_.frontend.orb_pyramid_levels;
        orbOptions.scaleFactor = float(parameters_.frontend.orb_scale_factor); orbOptions.initialFastThreshold = parameters_.frontend.orb_fast_threshold;
        orbOptions.layerFast=parameters_.frontend.orb_layer_fast;
        orbOptions.boundaryFast=parameters_.frontend.orb_boundary_fast;
        orbOptions.adaptiveGeometryCandidates=parameters_.frontend.orb_adaptive_geometry;
        orbOptions.batchDescriptorCoordinates=parameters_.frontend.orb_batch_descriptor_coordinates;
        orbOptions.groupedBalancing=parameters_.frontend.orb_grouped_balancing;
        orbOptions.cudaEarlyIntegral=parameters_.frontend.orb_cuda_early_integral;
        orbOptions.keepCudaDescriptorRows=parameters_.frontend.orb_cuda_matcher;
        orbOptions.minimumFastThreshold = parameters_.frontend.orb_min_fast_threshold; orbOptions.virtualFocalLength = parameters_.frontend.orb_virtual_focal_length;
        frontend_.setOrbOptions(orbOptions); 
        frontend_pose::Options poseOptions;
        poseOptions.solver.analyticAbsoluteJacobians=parameters_.frontend.pose_analytic_absolute_jacobians;
        poseOptions.adaptiveAbsoluteBudget=parameters_.frontend.pose_adaptive_ransac;
        frontend_.setPoseSolverOptions(poseOptions);
        LOG(INFO)<<"[LandmarkBudget] promotions_per_frame="<<parameters_.frontend.max_promotions_per_frame
            <<" active_initialized_soft_limit="<<parameters_.frontend.max_active_initialized;
        LOG(INFO)<<"[Pose] analytic_absolute_jacobians="<<poseOptions.solver.analyticAbsoluteJacobians
            <<" adaptive_ransac="<<poseOptions.adaptiveAbsoluteBudget<<" tracking_min_iterations="<<poseOptions.trackingMinIterations
            <<" full_min_iterations="<<poseOptions.solver.ransac.min_iterations<<" full_period="<<poseOptions.fullRansacPeriod;
        frontend_.setOrbMatchingThreshold(parameters_.frontend.orb_matching_threshold);
        std::string error;
        if (!estimator_.initializeRig(parameters_.nCameraSystem, parameters_.lidar ? parameters_.lidar->T_SL : kinematics::Transformation(),
            error, calibration.td_camera_initial_s, 0, imu.T_BS)) throw Exception(error);
        if (calibration.estimate_td_camera && !estimator_.setCameraTimeOffsetBounds(
            calibration.td_camera_initial_s - calibration.max_td_change_s,
            calibration.td_camera_initial_s + calibration.max_td_change_s, error)) throw Exception(error);
        if (calibration.do_extrinsics || calibration.estimate_td_camera)
            LOG(INFO) << "[Calibration] online_rig=" << calibration.do_extrinsics
                << " online_translation=" << (calibration.do_extrinsics && calibration.estimate_camera_rig_translation)
                << " online_td=" << calibration.estimate_td_camera << " td_initial_s=" << calibration.td_camera_initial_s
                << " gate=motion_and_tracks";
        // Move calibration-only ORB work out of the first image. No measurement,
        // gravity estimate, feature extraction or backend state is created here.
        frontend_.prepareOrb(parameters_.nCameraSystem,parameters_.frontend.parallelise_detection);
        if (parameters_.estimator.do_loop_closures || parameters_.estimator.do_final_ba) global_ = std::make_unique<backend::GlobalSlamGraph>(parameters_);
        if (parameters_.lidar) LOG(WARNING) << "LiDAR fusion remains deferred";
        if (!vocabularyPath_.empty() && !std::filesystem::is_regular_file(vocabularyPath_)) {
            LOG(WARNING) << "ORB vocabulary path is not a file; global retrieval uses geometric-verified appearance proposals";
            vocabularyPath_.clear();
        }
        if (parameters_.output.display_matches || parameters_.output.display_overhead)
            visualizationThread_ = std::thread(&ThreadedSlam::visualizationLoop, this);
        try { publishingThread_ = std::thread(&ThreadedSlam::publishingLoop, this); }
        catch (...) { visualizationQueue_.Shutdown(); if (visualizationThread_.joinable()) visualizationThread_.join(); throw; }
        LOG(INFO) << "[VIO] Ready; cameras=" << parameters_.nCameraSystem.numCameras()
            << " imu=" << imu.use << " window_capacity=" << options_.windowSize
            << " solver_iterations=" << options_.maxIterations << " solver_threads=" << options_.numThreads
            << " visual_jacobian="<<(options_.analyticVisualJacobians?"analytic":"numerical")
            << " backend_retries=" << health.backend_max_retries << " image_delay_s=" << parameters_.camera.image_delay
            << " loop_closures=" << parameters_.estimator.do_loop_closures << " final_ba=" << parameters_.estimator.do_final_ba;
        LOG(INFO)<<"[BackendPolicy] max_imu_interval_s="<<options_.maxImuIntervalSeconds
            <<" incremental_imu="<<options_.incrementalImu<<" preserve_oldest_visual_information="<<options_.preserveOldestVisualInformation
            <<" dense_schur="<<parameters_.estimator.backend_dense_schur
            <<" cache_visual_pairs="<<options_.cacheVisualPairs
            <<" backend_visual_batch="<<parameters_.estimator.backend_visual_batch
            <<" lightweight_non_keyframes="<<health.lightweight_non_keyframes
            <<" lightweight_new_imu_nis_gate="<<health.lightweight_new_imu_nis_gate
            <<" max_consecutive_lightweight_frames="<<health.max_consecutive_lightweight_frames
            <<" keyframe_anchor_promotion="<<parameters_.frontend.require_keyframe_anchor_for_promotion
            <<" max_tracking_seeds="<<parameters_.frontend.max_tracking_seeds
            <<" seed_max_untracked_s="<<parameters_.frontend.seed_max_untracked_seconds
            <<" enforce_solver_budget="<<parameters_.estimator.enforce_realtime;
        if(imu.use) LOG(INFO)<<"[ImuCalibration] noise_convention=continuous_density gravity="<<imu.g
            <<" gyro_density="<<imu.sigma_g_c<<" accel_density="<<imu.sigma_a_c
            <<" gyro_random_walk="<<imu.sigma_gw_c<<" accel_random_walk="<<imu.sigma_aw_c
            <<" max_gap_s="<<imu.max_sample_gap<<" T_BS_translation="<<imu.T_BS.r().transpose()
            <<" T_BS_quaternion_xyzw="<<imu.T_BS.q().coeffs().transpose();
        for (size_t c=0; c<parameters_.nCameraSystem.numCameras(); ++c) {
            const auto camera=parameters_.nCameraSystem.cameraGeometry(c);
            LOG(INFO) << "[Rig] camera=" << c << " model=" << camera->type()
                << " resolution=" << camera->imageWidth() << 'x' << camera->imageHeight()
                << " enabled=" << parameters_.nCameraSystem.cameraType(c).isUsed;
        }
        LOG(INFO)<<"[KeyframePolicy] parallax_rad="<<parameters_.frontend.keyframe_parallax
            <<" translation_m="<<parameters_.frontend.keyframe_translation<<" rotation_rad="<<parameters_.frontend.keyframe_rotation
            <<" minimum_shared="<<parameters_.frontend.keyframe_min_shared_tracks
            <<" low_motion_translation_m="<<parameters_.frontend.keyframe_low_motion_translation
            <<" low_motion_rotation_rad="<<parameters_.frontend.keyframe_low_motion_rotation;
        LOG(INFO) << "[ORB] max_features_per_camera=" << orbOptions.maxFeatures
            << " pyramid_levels=" << orbOptions.pyramidLevels << " scale=" << orbOptions.scaleFactor
            << " FAST=" << orbOptions.initialFastThreshold << '/' << orbOptions.minimumFastThreshold
            << " layer_fast=" << orbOptions.layerFast
            << " cuda_matcher=" << parameters_.frontend.orb_cuda_matcher
            << " cuda_early_integral=" << orbOptions.cudaEarlyIntegral
            << " hamming_threshold=" << parameters_.frontend.orb_matching_threshold
            << " descriptor_mode=" << descriptorModeName(orbOptions.descriptorMode)
            << " virtual_focal_config=" << orbOptions.virtualFocalLength;
        if (imu.use) LOG(INFO) << "[Init] Waiting for stationary IMU initialization; max_sample_gap_s=" << imu.max_sample_gap;
    }
    bool ThreadedSlam::addImages(const Time& stamp, const std::map<size_t, cv::Mat>& images, const std::map<size_t, cv::Mat>& depths) {
        return addImagesWithTimestamps(stamp,images,{},depths);
    }
    bool ThreadedSlam::addImagesWithTimestamps(const Time& stamp,const std::map<size_t,cv::Mat>& images,
        const std::map<size_t,Time>& originalTimes,const std::map<size_t,cv::Mat>& depths) {
        if (shutdown_ || stopping_ || faulted_) return false;
        if (!originalTimes.empty() && originalTimes.size()!=images.size()) { setError("Image timestamp count differs from image count"); return false; }
        const Time corrected = parameters_.camera.correctedImageTime(stamp);
        std::vector<CameraMeasurement> packet(parameters_.nCameraSystem.numCameras()); packet.front().timeStamp = corrected; bool used = false;
        try {
            for (const auto& item : images) {
                if (!parameters_.nCameraSystem.isCameraConfigured(item.first) || !parameters_.nCameraSystem.cameraType(item.first).isUsed || item.second.empty()) continue;
                auto& m = packet.at(item.first); m.timeStamp = corrected; m.sensorId = item.first; m.measurement.deliversKeypoints = false;
                const auto source=originalTimes.find(item.first);
                if (!originalTimes.empty() && source==originalTimes.end()) { setError("Missing original timestamp for camera "+std::to_string(item.first)); return false; }
                m.measurement.originalImageTimestamp=source==originalTimes.end()?stamp:source->second;
                m.measurement.imageTimeOffsetNs=static_cast<int64_t>(m.measurement.originalImageTimestamp->toNSec())-static_cast<int64_t>(stamp.toNSec());
                if (item.second.type() == CV_8UC1) m.measurement.image = item.second.clone();
                else if (item.second.type() == CV_8UC3) cv::cvtColor(item.second, m.measurement.image, cv::COLOR_BGR2GRAY);
                else if (item.second.type() == CV_8UC4) cv::cvtColor(item.second, m.measurement.image, cv::COLOR_BGRA2GRAY);
                else { setError("Unsupported image format; expected 8-bit gray/BGR/BGRA"); return false; }
                used = true;
            }
            for (const auto& item : depths) if (parameters_.nCameraSystem.isCameraConfigured(item.first)) packet.at(item.first).measurement.depthImage = item.second.clone();
        }
        catch (const std::exception& e) { setError(e.what()); return false; }
        catch (...) { setError("Unknown input exception"); return false; }
        if (!used) return true;
        if (blocking_) return cameraQueue_.PushBlockingIfFull(packet, 1);
        const bool dropped=cameraQueue_.PushNonBlockingDroppingIfFull(packet, 2);
        if (dropped) {
            const auto count=droppedCameraPackets_.fetch_add(1,std::memory_order_relaxed)+1;
            LOG_IF(WARNING, count==1 || count%30==0) << "[Input] Camera queue full; oldest image packet dropped; total_dropped="
                << count << " incoming_t_ns=" << corrected.toNSec();
        }
        return !dropped;
    }
    bool ThreadedSlam::addImuMeasurement(const Time& stamp, const Eigen::Vector3d& acceleration, const Eigen::Vector3d& gyro) {
        if (shutdown_ || stopping_ || faulted_ || !parameters_.imu.use || !acceleration.allFinite() || !gyro.allFinite()) return false;
        ImuMeasurement m; m.timeStamp = stamp; m.measurement.accelerometers = parameters_.imu.s_a.cwiseProduct(acceleration); m.measurement.gyroscopes = gyro;
        if (blocking_) return imuQueue_.PushBlockingIfFull(m, 5000);
        const bool dropped=imuQueue_.PushNonBlockingDroppingIfFull(m, 5000);
        if (dropped) {
            const auto count=droppedImuSamples_.fetch_add(1,std::memory_order_relaxed)+1;
            LOG_IF(WARNING, count==1 || count%100==0) << "[Input] IMU queue full; oldest sample dropped; total_dropped="
                << count << " incoming_t_ns=" << stamp.toNSec();
        }
        return !dropped;
    }
    bool ThreadedSlam::addLidarMeasurement(const Time&, const Eigen::Vector3d&) { return false; }
    bool ThreadedSlam::addDepthMeasurement(const Time&, const cv::Mat&, const std::optional<cv::Mat>&) { return false; }
    void ThreadedSlam::writeDebugStatisticsCsv(const std::string&) { setError("LiDAR debug statistics are unavailable while submapping is deferred"); }
    bool ThreadedSlam::collectImu(const Time& time) {
        if (!parameters_.imu.use) return true;
        ImuMeasurement m;
        if (imuBuffer_.empty() && (!imuQueue_.getCopyOfFront(&m) || m.timeStamp > before(time, overlap))) return false;
        while (imuBuffer_.empty() || imuBuffer_.back().timeStamp < time + Duration(overlap)) {
            if (!imuQueue_.PopNonBlocking(&m)) return false;
            if (!imuBuffer_.empty() && m.timeStamp <= imuBuffer_.back().timeStamp) { setError("Discarded non-increasing IMU timestamp"); continue; }
            imuBuffer_.push_back(m);
        }
        return true;
    }
    bool ThreadedSlam::predict(const MultiFrame& frame, State& state) {
        state = State{}; state.id = StateId(frame.id()); state.timestamp = frame.timestamp(); state.omega_S.setZero();
        if (!latestLocal_.id.isInitialised()) {
            SpeedAndBias sb = SpeedAndBias::Zero();
            if (parameters_.imu.use) {
                std::string error;
                if (!factor::initializeImu(imuBuffer_, parameters_.imu, frame.timestamp(), state.T_WS, sb, error))
                {
                    setError(error);
                    return false;
                }
                else
                {
                    LOG(INFO) << "IMU initialize with bg: " << sb[3] << " " << sb[4] << " " << sb[5] << " ba: "
                        << sb[6] << " " << sb[7] << " " << sb[8];
                }
            }
            else {
                const kinematics::Transformation T_WC(Eigen::Vector3d::Zero(), Eigen::Quaterniond(-std::sqrt(0.5), std::sqrt(0.5), 0, 0));
                state.T_WS = T_WC * parameters_.nCameraSystem.T_SC(0)->inverse();
            }
            state.v_W = sb.head<3>(); state.b_g = sb.segment<3>(3); state.b_a = sb.tail<3>();
        }
        else {
            state.T_WS = latestLocal_.T_WS; SpeedAndBias sb; sb << latestLocal_.v_W, latestLocal_.b_g, latestLocal_.b_a;
            if (parameters_.imu.use) {
                std::string reason;
                if (!factor::validImuInterval(imuBuffer_,latestLocal_.timestamp,frame.timestamp(),reason,parameters_.imu.max_sample_gap)) {
                    setFatalError(reason+"; stop and initialize a new VIO segment"); return false;
                }
                if (!frontend_.propagation(imuBuffer_, parameters_.imu, state.T_WS, sb, latestLocal_.timestamp, frame.timestamp(), nullptr, nullptr)) {
                    setError("IMU propagation failed"); return false;
                }
            }
            else if (previousLocal_.id.isInitialised()) {
                const double dt = (frame.timestamp() - latestLocal_.timestamp).toSec(), oldDt = (latestLocal_.timestamp - previousLocal_.timestamp).toSec();
                const Eigen::Vector3d angular = backend::logSO3(latestLocal_.T_WS.q() * previousLocal_.T_WS.q().conjugate()) / oldDt;
                const Eigen::Vector3d velocity = (latestLocal_.T_WS.r() - previousLocal_.T_WS.r()) / oldDt;
                state.T_WS.set(latestLocal_.T_WS.r() + dt * velocity, backend::expSO3(angular * dt) * latestLocal_.T_WS.q());
                state.omega_S = state.T_WS.C().transpose() * angular; sb.head<3>() = velocity;
            }
            state.v_W = sb.head<3>(); state.b_g = sb.segment<3>(3); state.b_a = sb.tail<3>();
        }
        if (parameters_.imu.use) state.omega_S = factor::gyroAt(imuBuffer_, frame.timestamp()) - state.b_g;
        return state.T_WS.T().allFinite() && state.v_W.allFinite() && state.b_g.allFinite() && state.b_a.allFinite();
    }
    void ThreadedSlam::extract(const MultiFramePtr& frame, const State& state) {
        const auto started=std::chrono::steady_clock::now();
        // This cache belongs to the control thread. In particular, do not fetch
        // a newer calibration from the running writer with an older pose seed.
        const auto snapshot = extractionSnapshot_;
        const auto live = snapshot ? pipeline::cameraExtrinsics(*snapshot, parameters_.nCameraSystem, parameters_.imu.T_BS) : AlignedVector<kinematics::Transformation>();
        std::vector<std::exception_ptr> errors(frame->numFrames()); std::vector<std::thread> workers;
        const auto run = [&](size_t c) { try {
            const auto extrinsic = snapshot ? live.at(c) : *parameters_.nCameraSystem.T_SC(c);
            frontend_.detectAndDescribe(c, frame, state.T_WS * extrinsic, nullptr);
        }
        catch (...) { errors[c] = std::current_exception(); } };
        try {
            if (parameters_.frontend.parallelise_detection) for (size_t c = 1; c < frame->numFrames(); ++c) workers.emplace_back(run, c);
            else for (size_t c = 1; c < frame->numFrames(); ++c) run(c);
            run(0);
        }
        catch (...) { for (auto& t : workers) if (t.joinable()) t.join(); throw; }
        for (auto& t : workers) t.join(); for (const auto& e : errors) if (e) std::rethrow_exception(e);
        if (parameters_.output.display_matches) try { cacheFeatureDebug(frame); }
        catch (const std::exception& e) { LOG(WARNING) << "[Debug] Feature snapshot skipped: " << e.what(); }
        const double extractionMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        if(VLOG_IS_ON(1) || extractionMs>200 || frame->id()%30==0)
            LOG(INFO)<<"[Timing][ExtractWall] frame="<<frame->id()<<" cameras="<<frame->numFrames()
                <<" keypoints="<<frame->numKeypoints()<<" wall_ms="<<extractionMs;
        if (VLOG_IS_ON(1)) {
            std::ostringstream counts;
            for (size_t c=0; c<frame->numFrames(); ++c) counts << (c?",":"") << c << ':' << frame->numKeypoints(c);
            VLOG(1) << "[ORB] frame=" << frame->id() << " t_ns=" << frame->timestamp().toNSec()
                << " keypoints=" << frame->numKeypoints() << " per_camera={" << counts.str() << "} elapsed_ms="
                << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        }
    }
    void ThreadedSlam::cacheFeatureDebug(const MultiFramePtr& frame) {
        auto captured=ViVisualizer::captureFeatures(*frame);
        featureDebugFrames_[frame->id()]=std::move(captured);
        std::set<uint64_t> retained{frame->id()};
        if (extractionSnapshot_) for (const auto& state:extractionSnapshot_->states) retained.insert(state.id.value());
        if (pendingFrame_) retained.insert(pendingFrame_->id());
        for (auto it=featureDebugFrames_.begin();it!=featureDebugFrames_.end();)
            if (!retained.count(it->first)) it=featureDebugFrames_.erase(it); else ++it;
    }
    void ThreadedSlam::prepareFeatureDebug(const MultiFramePtr& frame,const FrontendAssociationReport& report) {
        pendingFeatureDebug_.reset();
        const auto current=featureDebugFrames_.find(frame->id()); if (current==featureDebugFrames_.end()) return;
        auto debug=std::make_shared<ViVisualizer::FeatureDebugData>(); debug->current=current->second;
        debug->committed=report.committed; debug->visualConsensus=report.trackingAccepted; debug->inliers=report.inliers;
        debug->matches=report.debugMatches;
        for (const auto& match:debug->matches) if (match.first.frameId!=frame->id()) {
            const auto source=featureDebugFrames_.find(match.first.frameId);
            if (source!=featureDebugFrames_.end()) debug->sources.emplace(source->first,source->second);
        }
        pendingFeatureDebug_=debug;
        // Frontend diagnostics are visible even if the following solve is rejected.
        // They are explicitly labelled as staged associations, before backend acceptance.
        auto data=std::make_shared<ViVisualizer::VisualizationData>(); data->featureDebug=std::move(debug);
        visualizationQueue_.PushNonBlockingDroppingIfFull(data,2);
    }
    bool ThreadedSlam::stage(const MultiFramePtr& frame, const State& sensor) {
        const auto state = pipeline::bodyPrediction(sensor, parameters_.imu.T_BS); auto candidate = estimator_; std::string error;
        const bool first = candidate.numFrames() == 0;
        const auto previous = first ? std::optional<backend::StateEstimate>() : candidate.window().estimate(candidate.currentStateId());
        if (!candidate.addStates(*frame, state, imuBuffer_, error)) { setError(error); return false; }
        backend::FactorId currentImu=0;
        const bool sampleImu=VLOG_IS_ON(1) || frame->id()%30==0;
        const auto imuNis=[&]() {
            if(!currentImu || !sampleImu) return -1.0;
            const auto& graph=candidate.window().graph(); const auto& f=graph.factors.at(currentImu);
            std::vector<const double*> x; for(auto key:f.keys) x.push_back(graph.blocks.at(key).value.data());
            Eigen::Matrix<double,15,1> r;
            return f.cost->Evaluate(x.data(),r.data(),nullptr) && r.allFinite()?r.squaredNorm():-1.0;
        };
        if (previous && parameters_.imu.use) {
            const auto added = candidate.addFactor(factor::makeBodyImuRecord(imuBuffer_, parameters_.imu, *previous, state));
            if (!added.accepted()) { setError(added.message); return false; }
            currentImu=added.id;
        }
        const double predictedImuNis=imuNis();
        if (first) {
            const auto& graph = candidate.window().graph();
            const auto submit = [&](backend::FactorRecord prior) { const auto result = candidate.addFactor(std::move(prior)); if (!result.accepted()) throw Exception(result.message); };
            Eigen::Matrix<double, 6, 6> poseInformation = Eigen::Matrix<double, 6, 6>::Identity() * 1e4;
            if (parameters_.imu.use) { poseInformation(3, 3) = 20; poseInformation(4, 4) = 20; }
            submit(backend::makeParameterPrior(graph.blocks.at(backend::BlockKey::pose(state.id)), poseInformation));
            if (parameters_.imu.use) {
                Eigen::Matrix<double, 9, 9> S = Eigen::Matrix<double, 9, 9>::Zero(); S.diagonal().head<3>().setConstant(20);
                S.diagonal().segment<3>(3).setConstant(1 / parameters_.imu.sigma_bg); S.diagonal().tail<3>().setConstant(1 / parameters_.imu.sigma_ba);
                submit(backend::makeParameterPrior(graph.blocks.at(backend::BlockKey::speedBias(state.id)), S));
            }
            if (parameters_.camera.online_calibration.do_extrinsics) {
                Eigen::Matrix<double, 6, 6> S = Eigen::Matrix<double, 6, 6>::Identity();
                const auto& calibration = parameters_.camera.online_calibration;
                const auto& rig = graph.blocks.at({backend::BlockKind::CameraExtrinsic, 0});
                S.topLeftCorner<3, 3>() /= calibration.sigma_r;
                S.topRightCorner<3, 3>() = -backend::skew(backend::blockPose(rig).r()) / calibration.sigma_r;
                S.bottomRightCorner<3, 3>() /= calibration.sigma_alpha;
                submit(backend::makeParameterPrior(rig, S, backend::FactorKind::Calibration));
            }
            if (parameters_.camera.online_calibration.estimate_td_camera) {
                Eigen::Matrix<double, 1, 1> S;
                S(0, 0) = 1 / parameters_.camera.online_calibration.sigma_td_camera_s;
                submit(backend::makeParameterPrior(graph.blocks.at({backend::BlockKind::CameraTimeOffset, 0}),
                    S, backend::FactorKind::Calibration));
            }
        }
        bool keyframe = false;
        const bool associated = frontend_.dataAssociationAndInitialization(candidate, parameters_, frame, false, &keyframe);
        pendingAssociation_ = frontend_.lastAssociationReport();
        if(currentImu && sampleImu) {
            const auto seeded=candidate.window().estimate(state.id);
            LOG(INFO)<<"[ImuSeed] frame="<<frame->id()<<" factor="<<currentImu
                <<" prediction_nis="<<predictedImuNis<<" after_visual_nis="<<imuNis()
                <<" pose_shift_m="<<(seeded->T_WB.r()-state.T_WB.r()).norm()
                <<" rotation_shift_rad="<<backend::logSO3(seeded->T_WB.q()*state.T_WB.q().conjugate()).norm()
                <<" velocity_shift_mps="<<(seeded->speedBias.head<3>()-state.speedBias.head<3>()).norm();
        }
        if (parameters_.output.display_matches) try { prepareFeatureDebug(frame,pendingAssociation_); }
        catch (const std::exception& e) { LOG(WARNING) << "[Debug] Match snapshot skipped: " << e.what(); }
        if (!associated) {
            LOG(WARNING)<<"[VIO] Visual ingestion failed; frame="<<frame->id()
                <<" imu_only="<<parameters_.imu.use<<" reason="<<pendingAssociation_.message;
            if (!parameters_.imu.use) { setError(pendingAssociation_.message); return false; }
            if (!candidate.setHealth(state.id, backend::SensorHealth::Invalid, backend::SensorHealth::Invalid)
                || !candidate.setKeyframe(state.id, false)) {
                setError("Failed to stage an IMU-only state"); return false;
            }
        }
        if (!first && !parameters_.imu.use && !pendingAssociation_.trackingAccepted) { setError("Visual-only pose rejected; frame dropped without publishing prediction"); return false; }
        stagedCheckpoint_.emplace(std::move(estimator_));
        estimator_ = std::move(candidate); pendingFrame_ = frame; pendingImu_ = imuBuffer_; staged_ = true;
        optimizationAttempts_=0; pendingPrediction_=state;
        pendingCalibrationRetry_.reset();
        pendingPredictionSeconds_=previous?(state.timestamp-previous->timestamp).toSec():0;
        std::set<LandmarkId> visual;
        for (const auto& item:estimator_.window().graph().factors) {
            const auto& factor=item.second;
            if (factor.kind==backend::FactorKind::Visual && factor.sensorEpoch==state.id && factor.lifecycle==backend::FactorLifecycle::Active)
                for (auto key:factor.keys) if(key.kind==backend::BlockKind::Landmark) visual.insert(LandmarkId(key.id));
        }
        pendingVisualLandmarks_=visual.size();
        imuByFrame_[state.id] = pendingImu_; gyroByFrame_[state.id] = parameters_.imu.use ? factor::gyroAt(pendingImu_, frame->timestamp()) : sensor.omega_S + sensor.b_g;
        return true;
    }


    bool ThreadedSlam::processFrame() {
        if (shutdown_) return false;
        const auto processStarted=std::chrono::steady_clock::now();
        const auto milliseconds=[](const auto& start) { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); };
        try {
            if (faulted_) {
                // A terminal input fault quarantines future images, but an older
                // valid worker must still be collected and its threads released.
                waitForOptimization();
                if (staged_) rollbackPendingFrame();
                return false;
            }
            // A failed solve owns the sole insertion slot until it succeeds or
            // the retry limit explicitly restores its pre-insertion checkpoint.
            // Keep any already extracted next frame separate while retrying it.
            if (staged_ && !optimizationThread_.joinable()) {
                launchOptimization();
                return true;
            }

            // Bootstrap has no accepted pose/calibration cache yet. Only this
            // transition must finish the first solve before another prediction.
            if (!latestLocal_.id.isInitialised() && !waitForOptimization()) return false;

            if (!preparedFrame_) 
            {
                std::vector<CameraMeasurement> packet;
                if (!cameraQueue_.getCopyOfFront(&packet))
                {
                    // EOF/drain must still collect the last completed estimate.
                    waitForOptimization();
                    return false;
                }
                const auto time = packet.front().timeStamp;
                if (latestLocal_.id.isInitialised() && time <= latestLocal_.timestamp) {
                    cameraQueue_.PopFrontIf(&packet, [&](const auto& p) {return p.front().timeStamp == time; });
                    setError("Dropped non-increasing image timestamp");
                    return true;
                }
                if (parameters_.imu.use && imuBuffer_.empty()) {
                    ImuMeasurement earliest;
                    if (imuQueue_.getCopyOfFront(&earliest) && earliest.timeStamp > before(time, overlap)) {
                        cameraQueue_.PopFrontIf(&packet, [&](const auto& p) {return p.front().timeStamp == time; });
                        ++startupImageDrops_;
                        LOG_IF(WARNING, startupImageDrops_==1 || startupImageDrops_%30==0) << "[Init] Dropping image without IMU prehistory; image_t_ns="
                            << time.toNSec() << " first_imu_t_ns=" << earliest.timeStamp.toNSec() << " total_dropped=" << startupImageDrops_;
                        return true;
                    }
                }
                if (!collectImu(time)) {
                    ++imuCoverageWaits_;
                    VLOG_IF(1, imuCoverageWaits_==1 || imuCoverageWaits_%1000==0) << "[Input] Waiting for IMU coverage; image_t_ns=" << time.toNSec()
                        << " required_tail_ns=" << (time+Duration(overlap)).toNSec() << " buffered_samples=" << imuBuffer_.size();
                    // Let the producer supply the missing IMU tail while the
                    // preceding solve continues. Stop still collects its result
                    // when the final image cannot be covered at end of input.
                    if (stopping_) waitForOptimization();
                    return false;
                }
                if (!cameraQueue_.PopFrontIf(&packet, [&](const auto& p) {return p.front().timeStamp == time; })) return true;

                if (nextId_ == std::numeric_limits<uint64_t>::max()) throw Exception("State IDs exhausted");
                auto frame = std::make_shared<MultiFrame>(parameters_.nCameraSystem, time, nextId_++);
                for (size_t c = 0; c < packet.size(); ++c) {
                    frame->setImage(c, packet[c].measurement.image);
                    frame->setDepthImage(c, packet[c].measurement.depthImage);
                    if (packet[c].measurement.originalImageTimestamp)
                        frame->setImageTiming(c,*packet[c].measurement.originalImageTimestamp,packet[c].measurement.imageTimeOffsetNs);
                }
                State prediction{};
                if (!predict(*frame, prediction)){
                    if (faulted_) return false;
                    // Startup can discard old stationary-test samples. Once a
                    // state exists, keep the IMU interval needed from that state.
                    if (!latestLocal_.id.isInitialised())
                        while (imuBuffer_.size() > 2 && imuBuffer_[1].timeStamp < before(time, 0.6)) imuBuffer_.pop_front();
                    return true;
                }
                // The backend may still be solving frame N-1. Extraction of N
                // touches only its own images, per-camera extractors and cached
                // accepted state/calibration, never mutable estimator contents.
                extract(frame, prediction);
                preparedFrame_ = frame;
                preparedPrediction_ = prediction;
                preparedFromState_ = latestLocal_.id;
            }

            // Writer ownership transfers here, after the expensive extraction.
            // On rejection keep preparedFrame_ and retry the preceding staged
            // packet next time, without re-extracting or duplicating association.
            const auto waitStarted=std::chrono::steady_clock::now();
            if (!waitForOptimization()) return false;
            const double waitMs=milliseconds(waitStarted);
            const auto frame = preparedFrame_;
            if (latestLocal_.id.isInitialised() && frame->timestamp() <= latestLocal_.timestamp) {
                preparedFrame_.reset();
                setError("Dropped non-increasing image timestamp after optimization");
                return true;
            }
            // Matching and IMU insertion use the newest accepted seed. The ORB
            // gravity direction remains the predicted direction used at extraction;
            // descriptor rows are not regenerated when a newer solve completes.
            if (preparedFromState_ != latestLocal_.id) {
                if (!predict(*frame, preparedPrediction_)) return false;
                preparedFromState_ = latestLocal_.id;
            }
            if (!vocabularyLoaded_ && !vocabularyPath_.empty())
            {
                std::string error;
                if (global_ && !global_->loadVocabulary(vocabularyPath_, frame->descriptorProfile(0), error)) throw Exception(error);
                vocabularyLoaded_ = true;
            }
            const auto stageStarted=std::chrono::steady_clock::now();
            if (!stage(frame, preparedPrediction_)) {
                preparedFrame_.reset();
                return true;
            }
            preparedFrame_.reset();
            const double stageMs=milliseconds(stageStarted);

            pendingBudget_.maxIterations = options_.maxIterations;
            pendingBudget_.minIterations = parameters_.estimator.realtime_min_iterations;
            pendingBudget_.numThreads = options_.numThreads;
            pendingBudget_.collectCameraLandmarkCounts = parameters_.output.display_matches;
            // The configured allowance belongs to Ceres, shared across its refits.
            // Frontend staging and backend bookkeeping have separate wall timers.
            pendingBudget_.timeSeconds = options_.maxSolverTimeSeconds;
            launchOptimization(); 
            const double totalMs=milliseconds(processStarted);
            if(VLOG_IS_ON(1) || frame->id()%30==0 || totalMs>200)
                LOG(INFO)<<"[Timing][Pipeline] frame="<<frame->id()<<" wait_backend_ms="<<waitMs
                    <<" stage_ms="<<stageMs<<" process_call_ms="<<totalMs<<" backend=async";
            return true;
        }
        catch (const std::exception& e) { setError(e.what()); return false; }
        catch (...) { setError("Unknown frame-processing exception"); return false; }
    }


    void ThreadedSlam::launchOptimization() {
        if (!staged_ || !pendingFrame_ || !stagedCheckpoint_ || optimizationThread_.joinable()) throw Exception("Invalid native optimization launch order");
        accepted_ = false; optimizationRetryable_=true;
        optimizationError_ = nullptr; 
        ++optimizationAttempts_;
        try { optimizationThread_ = std::thread(&ThreadedSlam::optimizeAndPublish, this); }
        catch (...) {
            // Thread-creation failures must obey the same retry limit as solves.
            optimizationError_=std::current_exception();
            finishOptimization();
            throw;
        }
    }
    bool ThreadedSlam::waitForOptimization() {
        // This is the only normal processing barrier. No worker-written fields
        // (including accepted_, histories or estimator state) are read before join.
        if (!optimizationThread_.joinable()) return true;
        optimizationThread_.join();
        return finishOptimization();
    }
    bool ThreadedSlam::finishOptimization() {
        if (optimizationError_) {
            try { std::rethrow_exception(optimizationError_); }
            catch (const std::exception& e) { setError(e.what()); }
            catch (...) { setError("Unknown optimization exception"); }
            optimizationError_ = nullptr;
        }
        if (!accepted_) {
            if (optimizationRetryable_ && optimizationAttempts_<=parameters_.estimator.backend_max_retries) {
                LOG(WARNING) << "[Backend] Retrying staged frame=" << pendingFrame_->id()
                    << " completed_attempt=" << optimizationAttempts_ << " retry_limit=" << parameters_.estimator.backend_max_retries;
                return false;
            }
            rollbackPendingFrame();
            return true; // The writer slot is free; an extracted next frame survives.
        }
        const auto snapshot = estimator_.snapshot(); const auto id = snapshot->states.back().id;
        if (!latestLocal_.id.isInitialised()) {
            const auto& state=localHistory_.at(id);
            LOG(INFO) << "[Init] First backend state accepted; frame=" << id.value() << " t_ns=" << state.timestamp.toNSec()
                << " position_W=" << state.T_WS.r().transpose() << " velocity_W=" << state.v_W.transpose()
                << " gyro_bias=" << state.b_g.transpose() << " accel_bias=" << state.b_a.transpose()
                << " visual_consensus=" << pendingAssociation_.trackingAccepted;
        }
        latestLocal_ = localHistory_.at(id); previousLocal_ = State{};
        const auto it = localHistory_.lower_bound(id); if (it != localHistory_.begin()) previousLocal_ = std::prev(it)->second;
        extractionSnapshot_ = snapshot;
        std::set<StateId> retained; for (const auto& s : snapshot->states) retained.insert(s.id);
        for (auto it = imuByFrame_.begin(); it != imuByFrame_.end();) if (!retained.count(it->first)) it = imuByFrame_.erase(it); else ++it;
        while (imuBuffer_.size() > 2 && imuBuffer_[1].timeStamp < before(latestLocal_.timestamp, overlap)) imuBuffer_.pop_front();
        pendingFrame_.reset(); pendingImu_.clear(); staged_ = false;
        stagedCheckpoint_.reset(); optimizationAttempts_=0; return true;
    }
    void ThreadedSlam::rollbackPendingFrame() {
        if (optimizationThread_.joinable() || !stagedCheckpoint_ || !pendingFrame_)
            throw Exception("Invalid failed-frame rollback order");
        const StateId failed(pendingFrame_->id());
        const auto cause=lastError();
        estimator_=std::move(*stagedCheckpoint_); stagedCheckpoint_.reset();
        imuByFrame_.erase(failed); gyroByFrame_.erase(failed);
        pendingFrame_.reset(); pendingImu_.clear(); pendingAssociation_=FrontendAssociationReport();
        staged_=false; optimizationAttempts_=0; pendingVisualLandmarks_=0;
        frontend_.resetTrackingState();
        // IMU samples remain available to bridge from the last accepted state to
        // the next image. A rollback never publishes the rejected prediction.
        setError(std::string(optimizationRetryable_?"Backend retries exhausted; rolled back and dropped frame ":
            "Backend failure is not retryable; rolled back and dropped frame ")+std::to_string(failed.value())+": "+cause);
    }
    std::string ThreadedSlam::cameraCalibrationViolation(const backend::ParameterBlock& block) const {
        if (block.key.kind != backend::BlockKind::CameraExtrinsic) return {};
        const auto& calibration = parameters_.camera.online_calibration;
        const auto initial = parameters_.imu.T_BS * (*parameters_.nCameraSystem.T_SC(0));
        const auto rig = backend::blockPose(block);
        const double translation = (rig.r() - initial.r()).norm();
        const double rotation = backend::logSO3(rig.q() * initial.q().conjugate()).norm();
        if (translation <= calibration.max_rig_translation_change_m && rotation <= calibration.max_rig_rotation_change_rad)
            return {};
        return "Camera rig outside absolute range translation_m=" + std::to_string(translation)
            + " rotation_rad=" + std::to_string(rotation);
    }
    bool ThreadedSlam::validateSolution(const backend::BackendGraph& graph,std::string& reason) const {
        const StateId id(pendingFrame_->id());
        const auto& speed=graph.blocks.at(backend::BlockKey::speedBias(id)).value;
        if (parameters_.imu.use && pendingPredictionSeconds_>0) {
            const auto pose=backend::blockPose(graph.blocks.at(backend::BlockKey::pose(id)));
            if (!frontend_pose::inertialPoseConsistent(pendingPrediction_.T_WB*parameters_.imu.T_BS,
                pose*parameters_.imu.T_BS,pendingPredictionSeconds_,parameters_.estimator,&reason)) return false;
            // These conservative per-update envelopes are physical sanity limits,
            // not a substitute for bias covariance or an observability certificate.
            if ((speed.head<3>()-pendingPrediction_.speedBias.head<3>()).norm()>5.0
                || (speed.segment<3>(3)-pendingPrediction_.speedBias.segment<3>(3)).norm()>std::max(0.1,5*parameters_.imu.sigma_bg)
                || (speed.tail<3>()-pendingPrediction_.speedBias.tail<3>()).norm()>std::max(1.0,5*parameters_.imu.sigma_ba)) {
                reason="Optimized velocity/bias change exceeded VIO sanity limits"; return false;
            }
        }
        // Do not apply map-support requirements to legitimate IMU-only or
        // low-parallax bootstrap frames that had no finite visual map at ingestion.
        if (pendingAssociation_.trackingAccepted && pendingVisualLandmarks_>=10) {
            std::set<LandmarkId> surviving;
            for (const auto& item:graph.factors) {
                const auto& factor=item.second;
                if (factor.kind==backend::FactorKind::Visual && factor.sensorEpoch==id && factor.lifecycle==backend::FactorLifecycle::Active)
                    for (auto key:factor.keys) if(key.kind==backend::BlockKind::Landmark) surviving.insert(LandmarkId(key.id));
            }
            if(surviving.size()<10) { reason="Too few independent visual landmarks survived post-solve NIS gating"; return false; }
        }
        return true;
    }
    ThreadedSlam::CameraCalibrationDecision ThreadedSlam::cameraCalibrationDecision() {
        CameraCalibrationDecision decision;
        const auto& config = parameters_.camera.online_calibration;
        if (!config.do_extrinsics && !config.estimate_td_camera) return decision;
        if (optimizationAttempts_ > 1) { decision.reason = "retry_with_fixed_calibration"; return decision; }
        const auto& window = estimator_.window();
        const auto& order = window.order();
        decision.reason = "insufficient_healthy_window";
        if (order.size() < 5 || window.epochs().at(order.back()).visual != backend::SensorHealth::Valid) return decision;
        constexpr size_t minimumCalibrationTracks=10;
        if (pendingVisualLandmarks_ < minimumCalibrationTracks) {
            decision.reason = "insufficient_temporal_tracks";
            return decision;
        }

        // Use recent motion and live tracks to decide which calibration blocks to release.
        // Gyro axes are compared in the IMU frame and accelerations in the world frame.
        struct MotionSamples {
            Eigen::Vector3d sum = Eigen::Vector3d::Zero();
            Eigen::Matrix3d products = Eigen::Matrix3d::Zero();
            int count = 0;
            void add(const Eigen::Vector3d& v) { sum += v; products.noalias() += v * v.transpose(); ++count; }
            Eigen::Matrix3d covariance() const {
                const Eigen::Vector3d mean = sum / count;
                return products / count - mean * mean.transpose();
            }
        } gyro, velocity, acceleration;
        std::optional<backend::StateEstimate> first, previous;
        double maxTurn = 0;
        for (const auto id : order) {
            const auto state = window.estimate(id);
            if (!state) return decision;
            if (!first) first = state;
            const Eigen::Vector3d v = state->speedBias.head<3>();
            velocity.add(v);
            decision.translationSpan = std::max(decision.translationSpan, (state->T_WB.r() - first->T_WB.r()).norm());
            const auto measured = gyroByFrame_.find(id);
            if (measured != gyroByFrame_.end()) gyro.add(measured->second - state->speedBias.segment<3>(3));
            if (previous) {
                const double dt = (state->timestamp - previous->timestamp).toSec();
                if (dt <= 0) return decision;
                const Eigen::Vector3d v0 = previous->speedBias.head<3>();
                acceleration.add((v - v0) / dt);
            }
            const Eigen::Vector3d vFirst = first->speedBias.head<3>();
            if (vFirst.norm() > 0.05 && v.norm() > 0.05)
                maxTurn = std::max(maxTurn, vFirst.normalized().cross(v.normalized()).norm());
            previous = state;
        }
        if (gyro.count < 4 || (previous->timestamp - first->timestamp).toSec() < 0.25) return decision;
        const Eigen::Matrix3d gyroCov = gyro.covariance();
        const Eigen::Matrix3d accelCov = acceleration.covariance();
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> gyroEig(gyroCov, Eigen::EigenvaluesOnly);
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> accelEig(accelCov, Eigen::EigenvaluesOnly);
        if (gyroEig.info() != Eigen::Success || accelEig.info() != Eigen::Success) return decision;
        decision.angularVariation = std::sqrt(std::max(0.0, gyroCov.trace()));
        decision.angularSecondAxis = std::sqrt(std::max(0.0, gyroEig.eigenvalues()[1]));
        decision.velocityVariation = std::sqrt(std::max(0.0, velocity.covariance().trace()));
        decision.accelerationVariation = std::sqrt(std::max(0.0, accelCov.trace()));
        decision.speed = previous->speedBias.head<3>().norm();
        const bool rotationExcited = decision.angularVariation >= 0.03 && decision.angularSecondAxis >= 0.02;
        const bool clockMotion = rotationExcited || (decision.speed >= 0.2 && decision.velocityVariation >= 0.05
            && decision.accelerationVariation >= 0.15);
        const bool translationExcited = decision.translationSpan >= 0.05 && decision.velocityVariation >= 0.05
            && decision.accelerationVariation >= 0.15 && (maxTurn >= 0.10 || accelEig.eigenvalues()[1] >= 0.01);
        const bool wantTranslation = config.do_extrinsics && config.estimate_camera_rig_translation;
        const bool warmedUp = (!config.do_extrinsics || rotationCalibrationUpdates_ >= 3)
            && (!config.estimate_td_camera || timeCalibrationUpdates_ >= 3);
        const bool allowRotation = config.do_extrinsics && rotationExcited;
        const bool considerClock = config.estimate_td_camera && clockMotion;
        decision.reason = "weak_motion_excitation";
        if (!allowRotation && !considerClock) return decision;

        const auto current = order.back();
        const auto& landmarks = window.landmarks();
        const auto& factors = window.graph().factors;
        std::set<LandmarkId> temporal, timed;
        constexpr double minimumBearingSpeed=0.02; // radians per second
        for (const auto& key : landmarks.inFrame(current)) {
            const auto* observation = landmarks.find(key);
            if (!observation || observation->use != backend::ObservationUse::LiveFactor) continue;
            const auto* landmark = landmarks.find(observation->landmark);
            if (!landmark || landmark->anchor.frameId == current.value()) continue;
            const auto factor = factors.find(observation->factorId);
            if (factor == factors.end() || !factor->second.participates()) continue;
            if (temporal.size() < minimumCalibrationTracks) temporal.insert(observation->landmark);
            if (considerClock && observation->bearingVelocity.squaredNorm() >= minimumBearingSpeed*minimumBearingSpeed)
                timed.insert(observation->landmark);
            if (temporal.size() >= minimumCalibrationTracks && (!considerClock || timed.size() >= minimumCalibrationTracks)) break;
        }
        decision.temporalTracks = temporal.size();
        decision.timedTracks = timed.size();
        decision.reason = "insufficient_temporal_tracks";
        if (decision.temporalTracks < minimumCalibrationTracks) return decision;
        decision.rotation = allowRotation;
        decision.timeOffset = considerClock && decision.timedTracks >= minimumCalibrationTracks;
        decision.reason = considerClock && !decision.timeOffset ? "insufficient_timed_tracks" : "motion_and_tracks";
        if (!decision.rotation && !decision.timeOffset) return decision;

        decision.phase = "CALIB_ROT_TD";
        if (wantTranslation && warmedUp && translationExcited && decision.rotation
            && (!config.estimate_td_camera || decision.timeOffset)) {
            decision.translation = true;
            decision.phase = "CALIB_FULL";
        }
        if (trackingCalibrationUpdates_ >= 10 && (!wantTranslation || decision.translation))
            decision.phase = "CALIB_TRACKING";
        return decision;
    }

    ThreadedSlam::CameraCalibrationDecision ThreadedSlam::calibrationRetryDecision(CameraCalibrationDecision decision,
        const std::vector<backend::BlockKey>& rejected) {
        for (const auto key : rejected) {
            if (key.kind == backend::BlockKind::CameraExtrinsic) {
                decision.rotation = false;
                decision.translation = false;
            } else if (key.kind == backend::BlockKind::CameraTimeOffset) {
                decision.timeOffset = false;
            }
        }
        if (!decision.rotation && !decision.timeOffset) decision.phase = "CALIB_FIXED";
        else if (!decision.translation) decision.phase = "CALIB_ROT_TD";
        decision.reason = "retry_with_violating_blocks_fixed";
        return decision;
    }

    void ThreadedSlam::recordCameraCalibration(const CameraCalibrationDecision& decision, const backend::StepReport& step) {
        const auto& config = parameters_.camera.online_calibration;
        if (!config.do_extrinsics && !config.estimate_td_camera) return;
        if (step.accepted) {
            if (decision.rotation && rotationCalibrationUpdates_ < 3) ++rotationCalibrationUpdates_;
            if (decision.timeOffset && timeCalibrationUpdates_ < 3) ++timeCalibrationUpdates_;
            const bool wantTranslation = config.do_extrinsics && config.estimate_camera_rig_translation;
            if ((wantTranslation ? decision.translation : decision.rotation || decision.timeOffset)
                && trackingCalibrationUpdates_ < 10) ++trackingCalibrationUpdates_;
        }
        const bool active = decision.rotation || decision.timeOffset;
        if (optimizationAttempts_ == 1 && cameraCalibrationPhase_ == decision.phase
            && pendingFrame_->id() % 30 != 0 && (step.accepted || !active)) return;
        cameraCalibrationPhase_ = decision.phase;
        const auto& blocks = estimator_.window().graph().blocks;
        const double td = blocks.at({backend::BlockKind::CameraTimeOffset, 0}).value[0];
        const auto rig = backend::blockPose(blocks.at({backend::BlockKind::CameraExtrinsic, 0}));
        LOG(INFO) << "[Calibration] frame=" << pendingFrame_->id() << " phase=" << decision.phase
            << " reason=" << decision.reason << " accepted=" << step.accepted
            << " rotation=" << decision.rotation << " translation=" << decision.translation << " td_variable=" << decision.timeOffset
            << " gyro_rms_rad_s=" << decision.angularVariation << " gyro_second_axis_rad_s=" << decision.angularSecondAxis
            << " accel_rms_m_s2=" << decision.accelerationVariation << " velocity_rms_m_s=" << decision.velocityVariation
            << " translation_span_m=" << decision.translationSpan << " speed_mps=" << decision.speed
            << " temporal_tracks=" << decision.temporalTracks << " timed_tracks=" << decision.timedTracks
            << " gate_ms=" << decision.gateMs << " td_camera_s=" << td
            << " td_camera_total_s=" << (parameters_.camera.image_delay + td)
            << " p_BR_m=" << rig.r().transpose() << " q_BR_xyzw=" << rig.q().coeffs().transpose()
            << " solve_status=" << step.message;
    }

    void ThreadedSlam::optimizeAndPublish() {
        try {
            const auto started=std::chrono::steady_clock::now();
            const bool healthy = optimizationAttempts_ == 1 && latestTracking_.id.isInitialised()
                && latestTracking_.trackingQuality == TrackingQuality::Good
                && pendingAssociation_.trackingAccepted && pendingAssociation_.metricPose
                && pendingAssociation_.inliers >= 20 && pendingVisualLandmarks_ >= 20;
            auto fullBudget=pendingBudget_;
            fullBudget.skipPreNis=parameters_.estimator.skip_pre_nis_when_tracking_healthy && healthy;
            fullBudget.allowOnlineSchur=optimizationAttempts_==1;
            const backend::SolutionValidator quality = [this](const backend::BackendGraph& graph,std::string& reason) {
                return validateSolution(graph,reason);
            };
            CameraCalibrationDecision decision;
            backend::StepReport step;
            bool lightweight=false,lightweightFallback=false;
            double lightweightMs=0;
            const auto& window=estimator_.window();
            const bool canRefineLocally=optimizationAttempts_==1 && latestTracking_.id.isInitialised()
                && latestTracking_.trackingQuality!=TrackingQuality::Lost
                && pendingAssociation_.trackingAccepted && pendingAssociation_.metricPose;
            if(parameters_.estimator.lightweight_non_keyframes && parameters_.imu.use && canRefineLocally
                && consecutiveLightweightFrames_<parameters_.estimator.max_consecutive_lightweight_frames
                && window.canOptimizeLatestTwo()) {
                const auto lightStarted=std::chrono::steady_clock::now();
                auto lightBudget=fullBudget;
                lightBudget.latestTwoPoseVelocityOnly=true;
                lightBudget.gateLightweightNewImuNis=parameters_.estimator.lightweight_new_imu_nis_gate;
                const auto penultimate=window.order()[window.size()-2];
                step=estimator_.optimize(std::nullopt, {}, {}, {}, lightBudget);
                lightweightMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-lightStarted).count();
                lightweight=step.accepted;
                lightweightFallback=!lightweight;
                const auto& oldImu=step.solve.lightweightOldImu;
                const auto& newImu=step.solve.lightweightNewImu;
                LOG(INFO)<<"[Lightweight] frame="<<pendingFrame_->id()<<" penultimate="<<penultimate.value()<<" accepted="<<lightweight
                    <<" step_ms="<<lightweightMs<<" ceres_ms="<<step.solve.ceresMs
                    <<" free_dof="<<step.solve.effectiveParameters<<" residual_blocks="<<step.solve.residualBlocks
                    <<" old_imu_factor="<<oldImu.factor<<" old_imu_nis_pre="<<oldImu.before<<" old_imu_nis_post="<<oldImu.after
                    <<" new_imu_factor="<<newImu.factor<<" new_imu_nis_pre="<<newImu.before<<" new_imu_nis_post="<<newImu.after
                    <<" imu_nis="<<step.solve.postSolveResiduals[1].maxNis<<" reason="<<step.message;
            }
            if(lightweight) {
                ++consecutiveLightweightFrames_;
                decision.reason="lightweight_frozen";
            } else {
                const auto calibrationStarted=std::chrono::steady_clock::now();
                decision=pendingCalibrationRetry_?*pendingCalibrationRetry_:cameraCalibrationDecision();
                decision.gateMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-calibrationStarted).count();
                const backend::CalibrationUpdatePolicy calibration=[decision](backend::BlockKey key,const backend::BackendGraph&) {
                    return (decision.rotation && key.kind==backend::BlockKind::CameraExtrinsic)
                        || (decision.timeOffset && key.kind==backend::BlockKind::CameraTimeOffset);
                };
                fullBudget.cameraRotationOnly=decision.rotation && !decision.translation;
                if(parameters_.camera.online_calibration.do_extrinsics)
                    fullBudget.calibrationBlockViolation=[this](const backend::ParameterBlock& block) {
                        return cameraCalibrationViolation(block);
                    };
                step=estimator_.optimize(std::nullopt,calibration,quality,{},fullBudget);
                if(step.accepted) consecutiveLightweightFrames_=0;
            }
            pendingCalibrationRetry_.reset();
            if (!step.solve.rejectedCalibration.empty())
                pendingCalibrationRetry_ = calibrationRetryDecision(decision, step.solve.rejectedCalibration);
            if(!lightweight || pendingFrame_->id()%30==0) recordCameraCalibration(decision,step);
            optimizationRetryable_=step.solve.retryable;
            if(step.solve.residualStatisticsValid && (VLOG_IS_ON(1) || !step.accepted || pendingFrame_->id()%30==0)) {
                const char* kinds[]{"visual","imu","lidar","prior","other"};
                const char* phases[]{"pre_gate","pre_solve","post_solve"};
                const backend::ResidualStatisticsArray* values[]{&step.solve.preGateResiduals,&step.solve.preSolveResiduals,&step.solve.postSolveResiduals};
                for(size_t phase=0;phase<3;++phase) for(size_t kind=0;kind<5;++kind) {
                    const auto& s=(*values[phase])[kind]; if(!s.factors) continue;
                    LOG(INFO)<<"[Residual] frame="<<pendingFrame_->id()<<" accepted="<<step.accepted<<" phase="<<phases[phase]
                         <<" solve_mode="<<(lightweight?"lightweight":"full")
                        <<" kind="<<kinds[kind]<<" factors="<<s.factors<<" components="<<s.components
                        <<" white_component_rms="<<s.componentRms()<<" nis_p50="<<s.medianNis<<" nis_p95="<<s.p95Nis
                        <<" nis_max="<<s.maxNis<<" over_gate="<<s.overGate<<" robust_cost="<<s.robustCost<<" worst_factor="<<s.worstFactor;
                }
            }
            if(step.solve.factorProfiled) {
                const char* names[]={"visual","imu","lidar","prior","other"};
                for(size_t i=0;i<5;++i)
                    LOG(INFO)<<"[Timing][Factors] frame="<<pendingFrame_->id()<<" kind="<<names[i]
                        <<" jacobian_sum_ms="<<step.solve.factorJacobianSumMs[i]<<" residual_sum_ms="<<step.solve.factorResidualSumMs[i]
                        <<" jacobian_calls="<<step.solve.factorJacobianCalls[i]<<" residual_calls="<<step.solve.factorResidualCalls[i];
            }
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            VLOG(1) << "[Backend] frame=" << pendingFrame_->id() << " attempt=" << optimizationAttempts_
                << " accepted=" << step.accepted << " iterations=" << step.solve.iterations
                << " cost=" << step.solve.initialCost << "->" << step.solve.finalCost
                << " NIS_rejected_pre=" << step.solve.preRejected << " NIS_rejected_post=" << step.solve.postRejected
                << " extrinsics_unlocked=" << decision.rotation << " elapsed_ms=" << elapsed << " reason=" << step.message;
            if(VLOG_IS_ON(1) || !step.accepted || elapsed>200 || pendingFrame_->id()%30==0)
                LOG(INFO)<<"[Timing][Solve] frame="<<pendingFrame_->id()<<" accepted="<<step.accepted
                    <<" solve_mode="<<(lightweight?"lightweight":"full")<<" lightweight_fallback="<<lightweightFallback
                    <<" lightweight_ms="<<lightweightMs
                    <<" visual_jacobian="<<(step.solve.analyticVisualJacobians?"analytic":"numerical")
                    <<" refreshed_imu="<<step.solve.refreshedImuFactors
                    <<" imu_refresh_ms="<<step.solve.imuRefreshMs
                    <<" problem_builds="<<step.solve.problemBuilds<<" problem_reuses="<<step.solve.problemReuses
                    <<" factor_profiled="<<step.solve.factorProfiled
                    <<" retryable="<<step.solve.retryable<<" passes="<<step.solve.passes
                    <<" solver="<<step.solve.linearSolverUsed<<" sparse_backend="<<step.solve.sparseBackend
                    <<" line_search_interpolation="<<step.solve.lineSearchInterpolation
                    <<" schur_independent="<<step.solve.schurIndependent<<" parameter_blocks="<<step.solve.parameterBlocks
                    <<" effective_parameters="<<step.solve.effectiveParameters<<" residual_blocks="<<step.solve.residualBlocks
                    <<" residual_dimension="<<step.solve.residualDimension<<" active_landmarks="<<step.solve.activeLandmarks
                    <<" prior_rows="<<step.solve.priorRows<<" prior_columns="<<step.solve.priorColumns
                    <<" linear_solver_ms="<<step.solve.linearSolverMs<<" jacobian_ms="<<step.solve.jacobianMs
                    <<" jacobian_evaluations="<<step.solve.jacobianEvaluations<<" residual_evaluations="<<step.solve.residualEvaluations
                    <<" visual_pose_prepare_ms="<<step.solve.visualPosePrepareMs<<" visual_camera_prepare_ms="<<step.solve.visualCameraPrepareMs
                    <<" visual_pair_prepare_ms="<<step.solve.visualPairPrepareMs<<" gate_visual_prepare_ms="<<step.solve.gateVisualPrepareMs
                    <<" visual_timed_anchor_prepare_ms="<<step.solve.visualTimedAnchorPrepareMs
                    <<" visual_shared_timed_anchors="<<step.solve.visualSharedTimedAnchors
                    <<" visual_cached_cameras="<<step.solve.visualCachedCameras
                    <<" visual_cached_pairs="<<step.solve.visualCachedPairs<<" visual_cache_refreshes="<<step.solve.visualCacheRefreshes
                    <<" line_search_cost_ms="<<step.solve.lineSearchCostMs<<" line_search_gradient_ms="<<step.solve.lineSearchGradientMs
                    <<" line_search_ms="<<step.solve.lineSearchMs
                    <<" residual_ms="<<step.solve.residualMs<<" ceres_ms="<<step.solve.ceresMs
                    <<" ceres_budget_ms="<<step.solve.ceresTimeLimitMs
                    <<" ceres_budget_reached="<<step.solve.ceresTimeBudgetReached
                    <<" calibration_gate_ms="<<decision.gateMs
                    <<" pre_nis_skipped="<<step.solve.preNisSkipped<<" pre_nis_ms="<<step.solve.preNisMs
                    <<" post_nis_ms="<<step.solve.postNisMs
                    <<" termination="<<backend::solveTerminationName(step.solve.termination)
                    <<" unrefitted_outliers="<<step.solve.unrefittedOutliers
                    <<" post_rejected="<<step.solve.postRejected<<" elapsed_ms="<<elapsed;
            if (!step.accepted) { setError("Backend rejected solve; same staged packet retained: " + step.message); return; }
            if(step.solve.termination!=backend::SolveTermination::SolverFinished && (step.solve.unrefittedOutliers || VLOG_IS_ON(1)))
                LOG(INFO)<<"[Backend][Limited] frame="<<pendingFrame_->id()<<" accepted=1 termination="
                    <<backend::solveTerminationName(step.solve.termination)<<" unrefitted_outliers="<<step.solve.unrefittedOutliers
                    <<" ceres_ms="<<step.solve.ceresMs<<" quality_gate=passed";
            if (step.snapshot && (step.snapshot->revision==1 || step.snapshot->revision%30==0 || elapsed>200))
                LOG(INFO) << "[Backend] frame=" << pendingFrame_->id() << " revision=" << step.snapshot->revision
                    << " window=" << step.snapshot->states.size() << " landmarks=" << step.snapshot->landmarks.size()
                    << " iterations=" << step.solve.iterations << " cost=" << step.solve.initialCost << "->" << step.solve.finalCost
                    << " NIS_rejected=" << step.solve.preRejected+step.solve.postRejected << " elapsed_ms=" << elapsed;
            if (step.marginalization) {
                const auto& marginal=*step.marginalization;
                LOG(INFO) << "[Window] frame=" << pendingFrame_->id() << " policy="
                    << (marginal.policy==backend::MarginalizationPolicy::MARGIN_OLD?"MARGIN_OLD":"MARGIN_SECOND_NEW")
                    << " removed=" << marginal.removed.value() << " consumed_factors=" << marginal.consumedFactors
                    << " eliminated_dim=" << marginal.eliminatedDimension << " retained_dim=" << marginal.retainedDimension
                    << " prior_rank=" << marginal.priorRank << " reanchored=" << marginal.reanchoredLandmarks.size()
                    << " imu_mode="<<marginal.imuMode<<" replaced_imu="<<marginal.replacedImuFactors
                    << " active_imu="<<marginal.activeImuFactors<<" dropped_visual="<<marginal.droppedVisualFactors
                    << " suppressed_imu="<<marginal.suppressedImuFactors
                    << " incremental_imu_merges="<<marginal.incrementalImuMerges<<" full_imu_merges="<<marginal.fullImuMerges
                    << " ignored_absorbed_incident="<<marginal.ignoredAbsorbedIncidentObservations
                    << " live_incident_ranges="<<marginal.liveIncidentRanges
                    << " same_frame_range_eliminations="<<marginal.sameFrameRangeEliminations
                    << " schur_preserved_tracks="<<marginal.schurPreservedTracks.size()<<" scalar_schur_landmarks="<<marginal.scalarSchurLandmarks
                    << " reset_tracks="<<marginal.resetLandmarks.size()<<" moved_seeds="<<marginal.movedSeeds
                    << " demoted_tracks="<<marginal.demotedLandmarks.size()
                    << " repeated_resets="<<marginal.repeatedResets<<" repeated_demotions="<<marginal.repeatedDemotions
                    << " reset_prior_unchanged="<<marginal.resetPriorUnchanged
                    << " reset_prior_blocks="<<marginal.resetPriorProtectedBlocks
                    << " prior_reused="<<marginal.priorReused<<" prior_landmarks="<<marginal.priorLandmarks
                    << " removed_keyframe="<<marginal.removedKeyframe<<" landmarks_before="<<marginal.landmarksBefore
                    << " landmarks_after="<<marginal.landmarksAfter<<" retired_mature="<<marginal.retiredMature
                    << " retired_seeds="<<marginal.retiredSeeds<<" retired_mature_with_live_views="<<marginal.retiredMatureWithLiveViews;
                if(marginal.formation.evaluated) {
                    const auto& d=marginal.formation;
                    LOG(INFO)<<"[PriorFormation] frame="<<pendingFrame_->id()<<" policy="
                        <<(marginal.policy==backend::MarginalizationPolicy::MARGIN_OLD?"MARGIN_OLD":"MARGIN_SECOND_NEW")
                        <<" removed="<<marginal.removed.value()<<" visual_factors="<<d.visualFactors<<" imu_factors="<<d.imuFactors
                        <<" other_factors="<<d.otherFactors<<" incoming_reference_norm="<<d.incomingReferenceNorm
                        <<" incoming_current_norm="<<d.incomingCurrentNorm<<" robust_linearization_cost="<<d.robustCost
                        <<" linearized_squared_norm="<<d.linearizedSquaredNorm<<" eliminated_reduction="<<d.eliminatedReduction
                        <<" new_reference_norm="<<d.newReferenceNorm<<" quadratic_constant_offset="<<d.quadraticConstantOffset
                        <<" retained_gradient_norm="<<d.retainedGradientNorm<<" dropped_gradient_norm="<<d.droppedGradientNorm
                        <<" eigen_cutoff="<<d.eigenCutoff<<" prior_rank="<<marginal.priorRank<<" retained_dim="<<marginal.retainedDimension;
                }
                size_t samples=0;
                for(const auto& event:marginal.trackTransitions) {
                    // Full traces are opt-in. Sample two recurring identities
                    // every 30 frames at normal verbosity to bound log overhead.
                    const bool sampled=pendingFrame_->id()%30==0 && samples<2 && (event.resets>1 || event.demotions>1);
                    if(!VLOG_IS_ON(1) && !sampled) continue;
                    if(sampled) ++samples;
                    LOG(INFO)<<"[LandmarkLifecycle] frame="<<pendingFrame_->id()<<" landmark="<<event.id.value()
                        <<" action="<<event.action<<" reason="<<event.reason
                        <<" old_anchor="<<event.oldAnchor.frameId<<" old_camera="<<event.oldAnchor.cameraIndex
                        <<" new_anchor="<<event.newAnchor.frameId<<" new_camera="<<event.newAnchor.cameraIndex
                        <<" initializations="<<event.initializations<<" resets="<<event.resets<<" demotions="<<event.demotions
                        <<" previous_transition="<<event.previousTransitionFrame<<" surviving_views="<<event.survivingViews
                        <<" point_shift_m="<<event.pointShift<<" max_retained_nis="<<event.maxRetainedNis
                        <<" transaction=accepted";
                }
                std::map<std::string,size_t> reasons;
                size_t retiredSeedCount=0,seedWithoutViews=0,shortSeeds=0,seedEpochSum=0;
                for(const auto& retired:marginal.retirements) {
                    ++reasons[retired.reason];
                    if(!retired.initialized) {
                        ++retiredSeedCount; seedEpochSum+=retired.observedEpochs;
                        if(retired.observedEpochs<=2) ++shortSeeds;
                        if(!retired.survivingLive && !retired.survivingAbsorbed && !retired.survivingTracking) ++seedWithoutViews;
                    }
                    VLOG(1)<<"[LandmarkRetire] frame="<<pendingFrame_->id()<<" landmark="<<retired.id.value()
                        <<" anchor_frame="<<retired.anchor.frameId<<" anchor_camera="<<retired.anchor.cameraIndex
                        <<" initialized="<<retired.initialized<<" surviving_live="<<retired.survivingLive
                        <<" surviving_absorbed="<<retired.survivingAbsorbed<<" surviving_tracking="<<retired.survivingTracking
                        <<" observed_epochs="<<retired.observedEpochs<<" reason="<<retired.reason;
                }
                for(const auto& reason:reasons) LOG(INFO)<<"[LandmarkRetireSummary] frame="<<pendingFrame_->id()<<" reason="<<reason.first<<" count="<<reason.second;
                if(retiredSeedCount && (VLOG_IS_ON(1) || pendingFrame_->id()%30==0))
                    LOG(INFO)<<"[SeedRetire] frame="<<pendingFrame_->id()<<" policy="
                        <<(marginal.policy==backend::MarginalizationPolicy::MARGIN_OLD?"MARGIN_OLD":"MARGIN_SECOND_NEW")
                        <<" retired="<<retiredSeedCount<<" no_surviving_views="<<seedWithoutViews
                        <<" invalid_remaining_views="<<retiredSeedCount-seedWithoutViews
                        <<" at_most_two_epochs="<<shortSeeds<<" mean_observed_epochs="<<double(seedEpochSum)/retiredSeedCount;
            }
            const auto publicationStarted=std::chrono::steady_clock::now();
            accepted_ = true; publishStep(step);
            const double publicationMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-publicationStarted).count();
            if(VLOG_IS_ON(1) || elapsed>200 || publicationMs>200 || step.snapshot->revision==1 || step.snapshot->revision%30==0)
                LOG(INFO)<<"[Timing][Backend] frame="<<pendingFrame_->id()<<" gate_ms="<<step.solve.gatingMs
                    <<" solve_mode="<<(lightweight?"lightweight":"full")<<" lightweight_fallback="<<lightweightFallback
                    <<" pre_nis_skipped="<<step.solve.preNisSkipped<<" pre_nis_ms="<<step.solve.preNisMs
                    <<" post_nis_ms="<<step.solve.postNisMs
                    <<" residual_stats_ms="<<step.solve.residualStatisticsMs
                    <<" gate_setup_ms="<<step.solve.gateSetupMs<<" solve_validation_ms="<<step.solve.validationMs
                    <<" solve_checkpoint_ms="<<step.solve.checkpointMs<<" input_validation_ms="<<step.inputValidationMs
                    <<" window_checkpoint_ms="<<step.checkpointMs<<" maintenance_ms="<<step.maintenanceMs
                    <<" outgoing_ms="<<step.outgoingMs<<" receipt_ms="<<step.receiptMs
                    <<" build_ms="<<step.solve.problemBuildMs<<" ceres_ms="<<step.solve.ceresMs
                    <<" marginalize_linearize_ms="<<(step.marginalization?step.marginalization->linearizeMs:0)
                    <<" marginalize_problem_ms="<<(step.marginalization?step.marginalization->linearization.setupMs:0)
                    <<" marginalize_evaluate_ms="<<(step.marginalization?step.marginalization->linearization.evaluateMs:0)
                    <<" marginalize_normal_ms="<<(step.marginalization?step.marginalization->linearization.normalMs:0)
                    <<" marginalize_normal_threads="<<(step.marginalization?step.marginalization->linearization.normalThreads:0)
                    <<" marginalize_schur_ms="<<(step.marginalization?step.marginalization->schurMs:0)
                    <<" marginalize_prior_ms="<<(step.marginalization?step.marginalization->priorMs:0)
                    <<" marginalize_copy_ms="<<(step.marginalization?step.marginalization->topologyCopyMs:0)
                    <<" marginalize_validate_ms="<<(step.marginalization?step.marginalization->topologyValidationMs:0)
                    <<" marginalize_ms="<<step.marginalizationMs<<" snapshot_ms="<<step.snapshotMs
                    <<" step_ms="<<elapsed<<" publish_ms="<<publicationMs;
        }
        catch (...) { optimizationError_ = std::current_exception(); }
    }
    void ThreadedSlam::publishStep(const backend::StepReport& step) {
        const auto& snapshot = *step.snapshot; const StateId current(pendingFrame_->id());
        const auto convert = [&](const backend::StateEstimate& value) {
            const auto samples = imuByFrame_.find(value.id); const auto gyro = gyroByFrame_.find(value.id);
            return pipeline::sensorState(value, parameters_.imu.T_BS, samples == imuByFrame_.end() ? ImuMeasurementDeque{} : samples->second,
                gyro == gyroByFrame_.end() ? Eigen::Vector3d::Zero() : Eigen::Vector3d(gyro->second - value.speedBias.segment<3>(3)));
            };
        const auto remember = [&](const backend::StateEstimate& value) {
            auto state = convert(value); state.previousImuMeasurements.clear(); localHistory_[state.id] = state;
            };
        if (step.retiredState) remember(*step.retiredState);
        for (const auto& state : snapshot.states) remember(state);
        for (auto id : step.retiredLandmarks) pointHistory_.erase(id.value());
        for (const auto& point : step.retiredVisualPoints) pointHistory_[point.id] = point;
        const auto activeMap = pipeline::mapPoints(snapshot); for (const auto& point : *activeMap) pointHistory_[point.id] = point;
        const auto sensorExtrinsics = pipeline::cameraExtrinsics(snapshot, parameters_.nCameraSystem, parameters_.imu.T_BS);
        if (global_) {
            std::vector<kinematics::Transformation> bodyExtrinsics;
            for (const auto& extrinsic : sensorExtrinsics) bodyExtrinsics.push_back(parameters_.imu.T_BS * extrinsic);
            std::string error;
            try {
                const auto frame = estimator_.multiFrame(current);
                if (!global_->ingest(step, frame, pendingImu_, bodyExtrinsics, pendingAssociation_.trackingAccepted, error)) setError(error);
            }
            catch (const std::exception& e) { setError(std::string("Accepted local state; global update failed: ") + e.what()); }
        }
        auto updates = std::make_shared<pipeline::StateHistory>();
        const auto output = [&](const backend::StateEstimate& value) {
            const auto estimate = global_ ? global_->estimate(value.id) : value; auto state = convert(estimate);
            state.isOnlineExtrinsics = parameters_.camera.online_calibration.do_extrinsics;
            if (state.isOnlineExtrinsics) for (size_t c = 0; c < sensorExtrinsics.size(); ++c) state.extrinsics[c] = sensorExtrinsics[c];
            for (const auto& landmark : snapshot.landmarks) {
                bool sees = false; for (const auto& o : landmark.observations) sees = sees || o.key.frameId == value.id.value();
                if (sees) for (const auto& o : landmark.observations) if (o.key.frameId != value.id.value()) state.covisibleFrameIds.insert(StateId(o.key.frameId));
            }
            (*updates)[state.id] = state; auto history = state; history.previousImuMeasurements.clear(); outputHistory_[state.id] = std::move(history);
            };
        if (global_ && global_->lastLoopClosed()) for (const auto& value : global_->estimates()) output(value);
        else { if (step.retiredState) output(*step.retiredState); for (const auto& value : snapshot.states) output(value); }
        Publication publication; publication.states = updates; publication.state = updates->at(current); publication.points = activeMap;
        if (global_) for (auto& point : *publication.points) {
            const auto landmark = std::find_if(snapshot.landmarks.begin(), snapshot.landmarks.end(), [&](const auto& p) {return p.metadata.id.value() == point.id; });
            if (landmark != snapshot.landmarks.end()) point.point.head<3>() = global_->correctPoint(StateId(landmark->metadata.anchor.frameId), point.point.head<3>());
        }
        auto& tracking = publication.tracking; tracking.id = current; tracking.isKeyframe = publication.state.isKeyframe;
        tracking.isLidarKeyframe = false; tracking.recognisedPlace = global_ && global_->lastLoopClosed(); tracking.isFullGraphOptimising = false;
        const double coverage = pipeline::trackingCoverage(*pendingFrame_, snapshot);
        tracking.trackingQuality = !pendingAssociation_.trackingAccepted || coverage < 0.01 ? TrackingQuality::Lost :
            (coverage < 0.3 ? TrackingQuality::Marginal : TrackingQuality::Good);
        for (auto it = snapshot.states.rbegin(); it != snapshot.states.rend(); ++it) if (it->keyframe) { tracking.currentKeyframeId = it->id; break; }
        if (snapshot.revision==1 || tracking.trackingQuality!=latestTracking_.trackingQuality) {
            if (tracking.trackingQuality==TrackingQuality::Lost && snapshot.revision>1)
                LOG(WARNING) << "[Tracking] frame=" << current.value() << " quality=lost; visual_consensus="
                    << pendingAssociation_.trackingAccepted << " coverage=" << coverage << " inliers=" << pendingAssociation_.inliers;
            else LOG(INFO) << "[Tracking] frame=" << current.value() << " quality=" << trackingName(tracking.trackingQuality)
                << " coverage=" << coverage << " inliers=" << pendingAssociation_.inliers;
        }
        if (snapshot.revision==1 || snapshot.revision%30==0)
            LOG(INFO) << std::setprecision(12) << "[VIO] frame=" << current.value() << " t_ns=" << publication.state.timestamp.toNSec()
                << " quality=" << trackingName(tracking.trackingQuality) << " keyframe=" << tracking.isKeyframe
                << " keypoints=" << pendingFrame_->numKeypoints() << " inliers=" << pendingAssociation_.inliers
                << " map_points=" << activeMap->size() << " position_W=" << publication.state.T_WS.r().transpose()
                << " speed_mps=" << publication.state.v_W.norm();
        if (tracking.recognisedPlace) LOG(INFO) << "[Loop] Verified loop closure applied; frame=" << current.value();
        latestTracking_ = tracking;
        queuePublication(publication);
        try { prepareVisualization(publication, snapshot, step.solve); }
        catch (const std::exception& e) { setError(std::string("Visualization failed: ") + e.what()); }
    }
    void ThreadedSlam::queuePublication(Publication publication) {
        if (blocking_) publicationQueue_.PushBlockingIfFull(publication, 1);
        else if (publicationQueue_.PushNonBlockingDroppingIfFull(publication, 3)) setError("Output queue overrun: oldest publication dropped");
    }
    void ThreadedSlam::prepareVisualization(const Publication& publication, const backend::BackendSnapshot& snapshot,
        const backend::SolveReport& solve) {
        if(!parameters_.output.display_matches && !parameters_.output.display_overhead) return;
        auto data = std::make_shared<ViVisualizer::VisualizationData>();
        data->acceptedFrame=publication.state.id.value(); data->T_WS=publication.state.T_WS;
        if (parameters_.output.display_overhead) {
            const size_t stride=std::max<size_t>(1,(localHistory_.size()+7999)/8000); size_t i=0;
            for(const auto& state:localHistory_) {
                if(i++%stride && state.first!=publication.state.id) continue;
                data->trajectory.push_back({state.first.value(),state.second.T_WS.r(),state.second.isKeyframe});
            }
            for(const auto& state:snapshot.states)
                data->window.push_back({state.id.value(),(state.T_WB*parameters_.imu.T_BS).r(),state.keyframe});
        }
        if (!parameters_.output.display_matches) { visualizationQueue_.PushNonBlockingDroppingIfFull(data,2); return; }
        data->activeLandmarksByCamera=solve.activeLandmarksByCamera;
        data->currentFrames = backend::FrameStore::copyMeasurements(*pendingFrame_, true);
        data->featureDebug=pendingFeatureDebug_;
        estimator_.synchronizeFrame(publication.state.id,*data->currentFrames);
        std::map<uint64_t,const MapPoint*> points;
        for(const auto& point:*publication.points) points.emplace(point.id,&point);
        for(const auto& landmark:snapshot.landmarks) {
            const auto& m=landmark.metadata;
            if (m.initialized && landmark.pointWorld && landmark.pointWorld->allFinite()) ++data->windowMature3d;
            data->trackLifecycles.emplace(m.id.value(),ViVisualizer::VisualizationData::TrackLifecycle{
                m.initializationCount,m.resetCount,m.demotionCount,m.lastResetFrame});
        }
        data->T_WS = publication.state.T_WS; data->T_SCi = pipeline::cameraExtrinsics(snapshot, parameters_.nCameraSystem, parameters_.imu.T_BS);
        data->isKeyframe = publication.state.isKeyframe; data->recognisedPlace = publication.tracking.recognisedPlace;
        data->trackingQuality = publication.tracking.trackingQuality == TrackingQuality::Good ? ViVisualizer::VisualizationData::TrackingQuality::Good :
            (publication.tracking.trackingQuality == TrackingQuality::Marginal ? ViVisualizer::VisualizationData::TrackingQuality::Marginal : ViVisualizer::VisualizationData::TrackingQuality::Lost);
        for (size_t c = 0; c < pendingFrame_->numFrames(); ++c) for (size_t k = 0; k < pendingFrame_->numKeypoints(c); ++k) {
            Observation observation{}; observation.classification = -1; observation.frameId = pendingFrame_->id(); observation.cameraIdx = c; observation.keypointIdx = k;
            pendingFrame_->getKeypoint(c, k, observation.keypointMeasurement); pendingFrame_->getKeypointSize(c, k, observation.keypointSize);
            observation.landmarkId = data->currentFrames->landmarkId(c, k); observation.landmark_W.setZero(); observation.isInitialized = false;
            const auto point = points.find(observation.landmarkId);
            if (point != points.end()) { observation.landmark_W = point->second->point; observation.isInitialized = true; }
            data->observations.push_back(std::move(observation));
        }
        visualizationQueue_.PushNonBlockingDroppingIfFull(data, 2);
    }
    void ThreadedSlam::publishingLoop() {
        const auto publish = [&](const Publication& p) {
            if (optimisedGraphCallback_) try { optimisedGraphCallback_(p.state, p.tracking, p.states, p.points); }
            catch (const std::exception& e) { setError(std::string("Output callback failed: ") + e.what()); }
            catch (...) { setError("Unknown output callback exception"); }
            };
        Publication p; while (publicationQueue_.PopBlocking(&p)) publish(p);
        while (publicationQueue_.PopNonBlocking(&p)) publish(p);
    }
    void ThreadedSlam::visualizationLoop() {
        ViVisualizer::VisualizationData::Ptr data;
        std::map<std::string,cv::Mat> views;
        std::shared_ptr<const ViVisualizer::FeatureDebugData> rendered;
        while (visualizationQueue_.PopBlocking(&data)) try {
            if (data->featureDebug && data->featureDebug!=rendered) {
                const auto debugViews=ViVisualizer::drawFeatureDebug(*data->featureDebug);
                for (const auto& view:debugViews) views[view.first]=view.second;
                rendered=data->featureDebug;
            }
            if (data->currentFrames) {
                views["06 Feature Tracks"]=visualizer_.drawTrackTrails(*data);
                std::vector<cv::Mat> images(data->currentFrames->numFrames());
                for (size_t c=0;c<images.size();++c) if (parameters_.nCameraSystem.cameraType(c).isUsed && !data->currentFrames->image(c).empty())
                    images[c]=visualizer_.drawMatches(data,c);
                views["04 Backend Reprojection"]=ViVisualizer::cameraMosaic(images,"Backend accepted frame="+std::to_string(data->currentFrames->id()));
            }
            if(!data->trajectory.empty()) views["05 Trajectory"]=ViVisualizer::drawTrajectory(*data);
            // Publish all named views together. Reprojection updates must not erase
            // the latest feature/match dashboards when the UI consumes a slower queue.
            visualizationImages_.PushNonBlockingDroppingIfFull(views, 1);
        }
        catch (const std::exception& e) { setError(e.what()); }
        catch (...) { setError("Unknown visualization exception"); }
    }
    void ThreadedSlam::display(std::map<std::string, cv::Mat>& images) {
        std::map<std::string,cv::Mat> cameraViews;
        if (visualizationImages_.PopNonBlocking(&cameraViews)) for (const auto& view:cameraViews) if (!view.second.empty()) images[view.first]=view.second;
        cv::Mat overhead; if (overheadImages_.PopNonBlocking(&overhead)) images["Top Debug View"] = overhead;
    }
    void ThreadedSlam::stopThreading() {
        const auto self = std::this_thread::get_id();
        if ((publishingThread_.joinable() && publishingThread_.get_id() == self) || (optimizationThread_.joinable() && optimizationThread_.get_id() == self)
            || (visualizationThread_.joinable() && visualizationThread_.get_id() == self)) throw Exception("Stop/export must run on the control thread");
        if (stopping_.exchange(true)) return;
        LOG(INFO) << "[VIO] Stopping input and draining processing workers";
        try {
            while (processFrame()) {}
            if (optimizationThread_.joinable()) { optimizationThread_.join(); finishOptimization(); }
        }
        catch (const std::exception& e) { setError(e.what()); }
        catch (...) { setError("Unknown exception while stopping the pipeline"); }
        if (optimizationThread_.joinable()) optimizationThread_.join();
        shutdown_ = true; cameraQueue_.Shutdown(); imuQueue_.Shutdown();
        publicationQueue_.Shutdown(); visualizationQueue_.Shutdown();
        if (publishingThread_.joinable()) publishingThread_.join();
        if (visualizationThread_.joinable()) visualizationThread_.join();
        visualizationImages_.Shutdown(); overheadImages_.Shutdown();
        featureDebugFrames_.clear(); pendingFeatureDebug_.reset();
        LOG(INFO) << "[VIO] Stopped; trajectory_states=" << outputHistory_.size() << " terminal_fault=" << faulted_.load()
            << " queue_dropped_images=" << droppedCameraPackets_.load(std::memory_order_relaxed)
            << " queue_dropped_IMU=" << droppedImuSamples_.load(std::memory_order_relaxed) << " startup_dropped_images=" << startupImageDrops_;
    }
    void ThreadedSlam::writeFinalTrajectoryCsv() {
        stopThreading(); if (!trajectoryFile_.empty() && !pipeline::writeTrajectory(trajectoryFile_, outputHistory_, rpg_)) throw Exception("Trajectory CSV write failed");
        if (!trajectoryFile_.empty()) LOG(INFO) << "[Output] Trajectory saved; states=" << outputHistory_.size() << " path=" << trajectoryFile_;
    }
    bool ThreadedSlam::saveMap() {
        stopThreading(); if (mapFile_.empty()) return false;
        MapPointVector points;
        if (global_) points = global_->mapPoints(); else for (const auto& point : pointHistory_) points.push_back(point.second);
        const bool saved=pipeline::writeMap(mapFile_, points);
        if (saved) LOG(INFO) << "[Output] Map saved; points=" << points.size() << " path=" << mapFile_;
        else LOG(ERROR) << "[Output] Map write failed; path=" << mapFile_;
        return saved;
    }
    void ThreadedSlam::updateGlobalHistory() {
        if (!global_) return;
        const auto extrinsics = global_->cameraExtrinsics();
        for (const auto& body : global_->estimates()) {
            const auto gyro = gyroByFrame_.find(body.id);
            auto state = pipeline::sensorState(body, parameters_.imu.T_BS, {}, gyro == gyroByFrame_.end() ? Eigen::Vector3d::Zero() : Eigen::Vector3d(gyro->second - body.speedBias.segment<3>(3)));
            state.isOnlineExtrinsics = parameters_.camera.online_calibration.do_extrinsics || parameters_.camera.online_calibration.do_extrinsics_final_ba;
            if (state.isOnlineExtrinsics) for (size_t c = 0; c < extrinsics.size(); ++c) state.extrinsics[c] = parameters_.imu.T_BS.inverse() * extrinsics[c];
            outputHistory_[body.id] = std::move(state);
        }
    }
    void ThreadedSlam::doFinalBa() {
        stopThreading();
        if (!global_) throw Exception("Enable do_final_ba before collection to retain original historical measurements");
        LOG(INFO) << "[Backend] Starting final bundle adjustment";
        const auto& calibration = parameters_.camera.online_calibration; std::string error;
        if (!global_->bundleAdjust(100, parameters_.estimator.realtime_num_threads + parameters_.estimator.full_graph_num_threads,
            calibration.do_extrinsics_final_ba ? calibration.sigma_r_final_ba : 0,
            calibration.do_extrinsics_final_ba ? calibration.sigma_alpha_final_ba : 0, error)) throw Exception(error);
        updateGlobalHistory();
        LOG(INFO) << "[Backend] Final bundle adjustment completed; trajectory_states=" << outputHistory_.size();
        if (optimisedGraphCallback_ && !outputHistory_.empty()) {
            auto states = std::make_shared<pipeline::StateHistory>(outputHistory_); auto points = std::make_shared<MapPointVector>(global_->mapPoints());
            latestTracking_.id = states->rbegin()->first; latestTracking_.isKeyframe = states->rbegin()->second.isKeyframe;
            optimisedGraphCallback_(states->rbegin()->second, latestTracking_, states, points);
        }
    }
} // namespace cgmli
