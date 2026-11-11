
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <glog/logging.h>

#include <cameras/PinholeCamera.hpp>
#include <cameras/EquidistantDistortion.hpp>
#include <cameras/RadialTangentialDistortion.hpp>
#include <cameras/RadialTangentialDistortion8.hpp>
#include <cameras/EucmCamera.hpp>
#include <cameras/NoDistortion.hpp>
#include <opencv2/core/core.hpp>
#include <common/ViParametersReader.hpp>

/// \brief cgmli Main namespace of this package.
namespace cgmli {
namespace {
std::string distortionName(const cv::FileNode& node) {
  std::string name = node.empty() ? "none" : std::string(node);
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if (name == "nodistortion" || name == "no_distortion") name = "none";
  return name;
}
bool numericSequence(const cv::FileNode& node, size_t size) {
  if (!node.isSeq() || node.size() != size) return false;
  for (const auto& value : node)
    if ((!value.isInt() && !value.isReal()) || !std::isfinite(double(value))) return false;
  return true;
}
}

// The default constructor.
ViParametersReader::ViParametersReader()
    : readConfigFile_(false) {
}

// The constructor. This calls readConfigFile().
ViParametersReader::ViParametersReader(const std::string& filename) {
  // reads
  readConfigFile(filename);
}

// Read and parse a config file.
void ViParametersReader::readConfigFile(const std::string& filename) {
  // A failed reload must not expose a previous rig or append a second copy of it.
  readConfigFile_ = false;
  viParameters_ = ViParameters{};
  
  // FileStorage's narrow Windows filename path cannot represent every dataset
  // directory. Open UTF-8 paths through filesystem, then parse owned text.
  std::ifstream configInput(std::filesystem::u8path(filename),std::ios::binary);
  CGMLI_ASSERT_TRUE(Exception, bool(configInput), "Could not open config file: " << filename)
  std::ostringstream configText; configText<<configInput.rdbuf();
  cv::FileStorage file(configText.str(), cv::FileStorage::READ | cv::FileStorage::MEMORY);

  CGMLI_ASSERT_TRUE(Exception, file.isOpened(),
                    "Could not open config file: " << filename)
  LOG(INFO) << "Opened configuration file: " << filename;

  // camera calibration
  std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> calibrations;
  if(!getCameraCalibration(calibrations, file)) {
    throw Exception("Invalid camera calibration; see the model-specific diagnostic");
  }

  size_t camIdx = 0;
  for (size_t i = 0; i < calibrations.size(); ++i) {

    std::shared_ptr<const kinematics::Transformation> T_SC_cgmli_ptr(
          new kinematics::Transformation(calibrations[i].T_SC.r(),
                                                calibrations[i].T_SC.q().normalized()));

    if(strcmp(calibrations[i].cameraModel.c_str(), "eucm") == 0){
      std::shared_ptr<cgmli::cameras::EucmCamera> cam;
      cam.reset(new cgmli::cameras::EucmCamera(calibrations[i].imageDimension[0],
                                               calibrations[i].imageDimension[1],
                                               calibrations[i].focalLength[0],
                                               calibrations[i].focalLength[1],
                                               calibrations[i].principalPoint[0],
                                               calibrations[i].principalPoint[1],
                                               calibrations[i].eucmParameters[0],
                                               calibrations[i].eucmParameters[1]));
      cam->initialiseCameraAwarenessMaps();
      viParameters_.nCameraSystem.addCamera(
              T_SC_cgmli_ptr,
              std::static_pointer_cast<const cgmli::cameras::CameraBase>(cam),
              cgmli::cameras::NCameraSystem::NoDistortion, false,
              calibrations[i].cameraType);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "EUCM camera " << camIdx
                << " with T_SC=\n" << s.str();
    }
    else if (calibrations[i].distortionType == "none") {
      auto cam = std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(
          calibrations[i].imageDimension[0], calibrations[i].imageDimension[1],
          calibrations[i].focalLength[0], calibrations[i].focalLength[1],
          calibrations[i].principalPoint[0], calibrations[i].principalPoint[1], cameras::NoDistortion());
      cam->initialiseUndistortMaps();
      cam->initialiseCameraAwarenessMaps();
      viParameters_.nCameraSystem.addCamera(T_SC_cgmli_ptr, cam,
          cameras::NCameraSystem::NoDistortion, false, calibrations[i].cameraType);
    }
    else if (strcmp(calibrations[i].distortionType.c_str(), "equidistant") == 0) {
      std::shared_ptr<cgmli::cameras::PinholeCamera<cgmli::cameras::EquidistantDistortion>> cam;
      cam.reset(new cgmli::cameras::PinholeCamera<
                  cgmli::cameras::EquidistantDistortion>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  cameras::EquidistantDistortion(
                    calibrations[i].distortionCoefficients[0],
                    calibrations[i].distortionCoefficients[1],
                    calibrations[i].distortionCoefficients[2],
                    calibrations[i].distortionCoefficients[3])/*, id ?*/));
      cam->initialiseUndistortMaps(); // set up undistorters
      cam->initialiseCameraAwarenessMaps();
      viParameters_.nCameraSystem.addCamera(
          T_SC_cgmli_ptr,
          std::static_pointer_cast<const cameras::CameraBase>(cam),
          cameras::NCameraSystem::Equidistant, false,
          calibrations[i].cameraType);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Equidistant pinhole camera " << camIdx
                << " with T_SC=\n" << s.str();
    } else if (strcmp(calibrations[i].distortionType.c_str(), "radialtangential") == 0
               || strcmp(calibrations[i].distortionType.c_str(), "plumb_bob") == 0) {
      std::shared_ptr<cameras::PinholeCamera<cameras::RadialTangentialDistortion>> cam;
      cam.reset(new cameras::PinholeCamera<
                  cameras::RadialTangentialDistortion>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  cameras::RadialTangentialDistortion(
                    calibrations[i].distortionCoefficients[0],
                    calibrations[i].distortionCoefficients[1],
                    calibrations[i].distortionCoefficients[2],
                    calibrations[i].distortionCoefficients[3])/*, id ?*/));
      cam->initialiseUndistortMaps(); // set up undistorters
      cam->initialiseCameraAwarenessMaps();
      viParameters_.nCameraSystem.addCamera(
          T_SC_cgmli_ptr,
          std::static_pointer_cast<const cameras::CameraBase>(cam),
          cameras::NCameraSystem::RadialTangential, false,
          calibrations[i].cameraType);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Radial tangential pinhole camera " << camIdx
                << " with T_SC=\n" << s.str();
    } else if (strcmp(calibrations[i].distortionType.c_str(), "radialtangential8") == 0
               || strcmp(calibrations[i].distortionType.c_str(), "plumb_bob8") == 0) {
      std::shared_ptr<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>> cam;
      cam.reset(new cameras::PinholeCamera<cameras::RadialTangentialDistortion8>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  cameras::RadialTangentialDistortion8(
                    calibrations[i].distortionCoefficients[0],
                    calibrations[i].distortionCoefficients[1],
                    calibrations[i].distortionCoefficients[2],
                    calibrations[i].distortionCoefficients[3],
                    calibrations[i].distortionCoefficients[4],
                    calibrations[i].distortionCoefficients[5],
                    calibrations[i].distortionCoefficients[6],
                    calibrations[i].distortionCoefficients[7])/*, id ?*/));
      cam->initialiseUndistortMaps(); // set up undistorters
      cam->initialiseCameraAwarenessMaps();
      viParameters_.nCameraSystem.addCamera(
          T_SC_cgmli_ptr,
          std::static_pointer_cast<const cameras::CameraBase>(cam),
          cameras::NCameraSystem::RadialTangential8, false,
          calibrations[i].cameraType);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Radial tangential 8 pinhole camera " << camIdx
                << " with T_SC=\n" << s.str();
    } else {
      // Skipping a camera would silently renumber every following image stream.
      throw Exception("Unrecognized distortion type: " + calibrations[i].distortionType);
    }
    ++camIdx;
  }
  // All camera models and extrinsics are now available. Compute each directed
  // cross-camera mask once, rather than recomputing the growing rig on insertion.
  viParameters_.nCameraSystem.computeOverlaps();

  if(file["lidar"].isMap()){
    viParameters_.lidar = cgmli::LidarParameters(); 
    if(!getLiDARCalibration(file["lidar"], *viParameters_.lidar)){
      LOG(ERROR) << "Could not parse the LiDAR config file";
    } else {
      std::stringstream s;
      s << (*viParameters_.lidar).T_SL.T();
      LOG(INFO) << "Parsed LiDAR with the following characteristics: \n" 
                << "elevation_resolution_angle: " << std::to_string((*viParameters_.lidar).elevation_resolution_angle) << " \n"
                << "azimuth_resolution_angle: " << std::to_string((*viParameters_.lidar).azimuth_resolution_angle) << "\n"
                << "T_SL:\n " << s.str();
    }
  } else {
    LOG(INFO) << "No LiDAR declared";
  }

  //camera parameters.
  parseEntry(file["camera_parameters"], "timestamp_tolerance",
             viParameters_.camera.timestamp_tolerance);
  parseEntry(file["camera_parameters"], "image_delay",
             viParameters_.camera.image_delay);
  parseEntry(file["camera_parameters"]["online_calibration"], "do_extrinsics",
             viParameters_.camera.online_calibration.do_extrinsics);
  parseEntry(file["camera_parameters"]["online_calibration"], "do_extrinsics_final_ba",
             viParameters_.camera.online_calibration.do_extrinsics_final_ba);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_r",
             viParameters_.camera.online_calibration.sigma_r);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_alpha",
             viParameters_.camera.online_calibration.sigma_alpha);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_r_final_ba",
             viParameters_.camera.online_calibration.sigma_r_final_ba);
  parseEntry(file["camera_parameters"]["online_calibration"], "sigma_alpha_final_ba",
             viParameters_.camera.online_calibration.sigma_alpha_final_ba);
  const auto calibrationNode=file["camera_parameters"]["online_calibration"];
  auto& calibration=viParameters_.camera.online_calibration;
  const auto optionalCalibration=[&](const char* name,auto& value) {
    if (!calibrationNode[name].empty()) parseEntry(calibrationNode,name,value);
  };
  optionalCalibration("estimate_camera_rig_translation",calibration.estimate_camera_rig_translation);
  optionalCalibration("estimate_td_camera",calibration.estimate_td_camera);
  optionalCalibration("td_camera_initial_s",calibration.td_camera_initial_s);
  optionalCalibration("sigma_td_camera_s",calibration.sigma_td_camera_s);
  optionalCalibration("max_td_change_s",calibration.max_td_change_s);
  optionalCalibration("max_rig_translation_change_m",calibration.max_rig_translation_change_m);
  optionalCalibration("max_rig_rotation_change_rad",calibration.max_rig_rotation_change_rad);

  //IMU parameters.
  parseEntry(file["imu_parameters"], "use",
             viParameters_.imu.use);
  Eigen::Matrix4d T_BS;
  parseEntry(file["imu_parameters"], "T_BS", T_BS);
  viParameters_.imu.T_BS = kinematics::Transformation(T_BS);
  parseEntry(file["imu_parameters"], "a_max",
             viParameters_.imu.a_max);
  parseEntry(file["imu_parameters"], "g_max",
             viParameters_.imu.g_max);
  parseEntry(file["imu_parameters"], "sigma_g_c",
             viParameters_.imu.sigma_g_c);
  parseEntry(file["imu_parameters"], "sigma_bg",
             viParameters_.imu.sigma_bg);
  parseEntry(file["imu_parameters"], "sigma_a_c",
             viParameters_.imu.sigma_a_c);
  parseEntry(file["imu_parameters"], "sigma_ba",
             viParameters_.imu.sigma_ba);
  parseEntry(file["imu_parameters"], "sigma_gw_c",
             viParameters_.imu.sigma_gw_c);
  parseEntry(file["imu_parameters"], "sigma_aw_c",
             viParameters_.imu.sigma_aw_c);
  parseEntry(file["imu_parameters"], "a0",
             viParameters_.imu.a0);
  parseEntry(file["imu_parameters"], "g0",
             viParameters_.imu.g0);
  parseEntry(file["imu_parameters"], "g",
             viParameters_.imu.g);
  parseEntry(file["imu_parameters"], "s_a",
             viParameters_.imu.s_a);
  if (!file["imu_parameters"]["max_sample_gap"].empty())
    parseEntry(file["imu_parameters"], "max_sample_gap", viParameters_.imu.max_sample_gap);

  // Parameters for detection etc.
  parseEntry(file["frontend_parameters"], "max_num_keypoints",
             viParameters_.frontend.max_num_keypoints);
  parseEntry(file["frontend_parameters"], "parallelise_detection",
             viParameters_.frontend.parallelise_detection);
  parseEntry(file["frontend_parameters"], "num_matching_threads",
             viParameters_.frontend.num_matching_threads);
  // Optional native ORB controls retain their explicit defaults when omitted.
  const auto frontendNode=file["frontend_parameters"];
  if(!frontendNode["require_keyframe_anchor_for_promotion"].empty())
    parseEntry(frontendNode,"require_keyframe_anchor_for_promotion",viParameters_.frontend.require_keyframe_anchor_for_promotion);
  if(!frontendNode["max_tracking_seeds"].empty()) parseEntry(frontendNode,"max_tracking_seeds",viParameters_.frontend.max_tracking_seeds);
  if(!frontendNode["seed_max_untracked_seconds"].empty()) parseEntry(frontendNode,"seed_max_untracked_seconds",viParameters_.frontend.seed_max_untracked_seconds);
  CGMLI_ASSERT_TRUE(Exception,viParameters_.frontend.max_tracking_seeds>=0
      && std::isfinite(viParameters_.frontend.seed_max_untracked_seconds) && viParameters_.frontend.seed_max_untracked_seconds>=0,
      "invalid tracking seed budget/lifetime")
  if(!frontendNode["keyframe_parallax"].empty()) parseEntry(frontendNode,"keyframe_parallax",viParameters_.frontend.keyframe_parallax);
  if(!frontendNode["keyframe_translation"].empty()) parseEntry(frontendNode,"keyframe_translation",viParameters_.frontend.keyframe_translation);
  if(!frontendNode["keyframe_rotation"].empty()) parseEntry(frontendNode,"keyframe_rotation",viParameters_.frontend.keyframe_rotation);
  if(!frontendNode["keyframe_min_shared_tracks"].empty()) parseEntry(frontendNode,"keyframe_min_shared_tracks",viParameters_.frontend.keyframe_min_shared_tracks);
  if(!frontendNode["keyframe_low_motion_translation"].empty()) parseEntry(frontendNode,"keyframe_low_motion_translation",viParameters_.frontend.keyframe_low_motion_translation);
  if(!frontendNode["keyframe_low_motion_rotation"].empty()) parseEntry(frontendNode,"keyframe_low_motion_rotation",viParameters_.frontend.keyframe_low_motion_rotation);
  const auto& kf=viParameters_.frontend;
  CGMLI_ASSERT_TRUE(Exception, std::isfinite(kf.keyframe_parallax) && kf.keyframe_parallax>0
      && std::isfinite(kf.keyframe_translation) && kf.keyframe_translation>0
      && std::isfinite(kf.keyframe_rotation) && kf.keyframe_rotation>0
      && std::isfinite(kf.keyframe_low_motion_translation) && kf.keyframe_low_motion_translation>0
      && std::isfinite(kf.keyframe_low_motion_rotation) && kf.keyframe_low_motion_rotation>0
      && kf.keyframe_min_shared_tracks>=1, "invalid frontend keyframe parallax/motion thresholds")
  if(!frontendNode["orb_pyramid_levels"].empty())
    parseEntry(frontendNode,"orb_pyramid_levels",viParameters_.frontend.orb_pyramid_levels);
  if(!frontendNode["orb_scale_factor"].empty())
    parseEntry(frontendNode,"orb_scale_factor",viParameters_.frontend.orb_scale_factor);
  if(!frontendNode["max_promotions_per_frame"].empty()) parseEntry(frontendNode,"max_promotions_per_frame",viParameters_.frontend.max_promotions_per_frame);
  if(!frontendNode["max_active_initialized"].empty()) parseEntry(frontendNode,"max_active_initialized",viParameters_.frontend.max_active_initialized);
  CGMLI_ASSERT_TRUE(Exception, viParameters_.frontend.max_promotions_per_frame>=0 && viParameters_.frontend.max_active_initialized>=0, "invalid promotion budget")
  if(!frontendNode["orb_layer_fast"].empty())
    parseEntry(frontendNode,"orb_layer_fast",viParameters_.frontend.orb_layer_fast);
  if(!frontendNode["orb_boundary_fast"].empty())
    parseEntry(frontendNode,"orb_boundary_fast",viParameters_.frontend.orb_boundary_fast);
  if(!frontendNode["orb_adaptive_geometry"].empty())
    parseEntry(frontendNode,"orb_adaptive_geometry",viParameters_.frontend.orb_adaptive_geometry);
  if(!frontendNode["orb_batch_descriptor_coordinates"].empty()) parseEntry(frontendNode,"orb_batch_descriptor_coordinates",viParameters_.frontend.orb_batch_descriptor_coordinates);
  if(!frontendNode["orb_grouped_balancing"].empty())
    parseEntry(frontendNode,"orb_grouped_balancing",viParameters_.frontend.orb_grouped_balancing);
  if(!frontendNode["pose_analytic_absolute_jacobians"].empty())
    parseEntry(frontendNode,"pose_analytic_absolute_jacobians",viParameters_.frontend.pose_analytic_absolute_jacobians);
  if(!frontendNode["pose_adaptive_ransac"].empty())
    parseEntry(frontendNode,"pose_adaptive_ransac",viParameters_.frontend.pose_adaptive_ransac);
  if(!frontendNode["orb_staged_matching"].empty())
    parseEntry(frontendNode,"orb_staged_matching",viParameters_.frontend.orb_staged_matching);
  if(!frontendNode["orb_cuda_matcher"].empty())
    parseEntry(frontendNode,"orb_cuda_matcher",viParameters_.frontend.orb_cuda_matcher);
  if(!frontendNode["orb_cuda_early_integral"].empty())
    parseEntry(frontendNode,"orb_cuda_early_integral",viParameters_.frontend.orb_cuda_early_integral);
  if(!frontendNode["orb_fast_threshold"].empty())
    parseEntry(frontendNode,"orb_fast_threshold",viParameters_.frontend.orb_fast_threshold);
  if(!frontendNode["orb_min_fast_threshold"].empty())
    parseEntry(frontendNode,"orb_min_fast_threshold",viParameters_.frontend.orb_min_fast_threshold);
  if(!frontendNode["orb_virtual_focal_length"].empty())
    parseEntry(frontendNode,"orb_virtual_focal_length",viParameters_.frontend.orb_virtual_focal_length);
  if(!frontendNode["orb_matching_threshold"].empty())
    parseEntry(frontendNode,"orb_matching_threshold",viParameters_.frontend.orb_matching_threshold);

  // Parameters regarding the estimator.
  const auto windowNode=file["estimator_parameters"];
  if(!windowNode["max_imu_interval_seconds"].empty())
    parseEntry(windowNode,"max_imu_interval_seconds",viParameters_.estimator.max_imu_interval_seconds);
  if(!windowNode["incremental_imu"].empty()) parseEntry(windowNode,"incremental_imu",viParameters_.estimator.incremental_imu);
  if(!windowNode["preserve_oldest_visual_information"].empty())
    parseEntry(windowNode,"preserve_oldest_visual_information",viParameters_.estimator.preserve_oldest_visual_information);
  if(!windowNode["backend_dense_schur"].empty()) parseEntry(windowNode,"backend_dense_schur",viParameters_.estimator.backend_dense_schur);
  if(!windowNode["cache_visual_pairs"].empty()) parseEntry(windowNode,"cache_visual_pairs",viParameters_.estimator.cache_visual_pairs);
  auto& visualBatch = viParameters_.estimator.backend_visual_batch;
  if(!windowNode["backend_visual_batch"].empty())
    parseEntry(windowNode,"backend_visual_batch",visualBatch);
  CGMLI_ASSERT_TRUE(Exception, visualBatch=="cpu" || visualBatch=="cuda" || visualBatch=="hybrid" ||
                    visualBatch=="schur" || visualBatch=="audit",
                    "estimator_parameters.backend_visual_batch must be cpu, cuda, hybrid, schur, or audit")
  CGMLI_ASSERT_TRUE(Exception,std::isfinite(viParameters_.estimator.max_imu_interval_seconds)
      && viParameters_.estimator.max_imu_interval_seconds>0,"invalid maximum IMU interval duration")
  CGMLI_ASSERT_TRUE(Exception, windowNode["num_keyframes"].empty() && windowNode["num_imu_frames"].empty(),
                    "Replace obsolete num_keyframes/num_imu_frames with estimator_parameters: window_size (default 10)")
  if(!windowNode["window_size"].empty())
    parseEntry(windowNode,"window_size",viParameters_.estimator.window_size);
  CGMLI_ASSERT_TRUE(Exception, viParameters_.estimator.window_size>=2,
                    "estimator_parameters: window_size must be at least 2")
  parseEntry(file["estimator_parameters"], "do_loop_closures",
             viParameters_.estimator.do_loop_closures);
  parseEntry(file["estimator_parameters"], "do_final_ba",
             viParameters_.estimator.do_final_ba);
  parseEntry(file["estimator_parameters"], "enforce_realtime",
             viParameters_.estimator.enforce_realtime);
  parseEntry(file["estimator_parameters"], "realtime_min_iterations",
             viParameters_.estimator.realtime_min_iterations);
  parseEntry(file["estimator_parameters"], "realtime_max_iterations",
             viParameters_.estimator.realtime_max_iterations);
  parseEntry(file["estimator_parameters"], "realtime_time_limit",
             viParameters_.estimator.realtime_time_limit);
  parseEntry(file["estimator_parameters"], "realtime_num_threads",
             viParameters_.estimator.realtime_num_threads);
  parseEntry(file["estimator_parameters"], "full_graph_iterations",
             viParameters_.estimator.full_graph_iterations);
  parseEntry(file["estimator_parameters"], "full_graph_num_threads",
             viParameters_.estimator.full_graph_num_threads);
  parseEntry(file["estimator_parameters"], "p_dbow",
             viParameters_.estimator.p_dbow);

  // Some options for how and what to output.
  const auto estimatorNode = file["estimator_parameters"];
  if (!estimatorNode["backend_max_retries"].empty())
    parseEntry(estimatorNode, "backend_max_retries", viParameters_.estimator.backend_max_retries);
  if (!estimatorNode["visual_nis_threshold"].empty())
    parseEntry(estimatorNode, "visual_nis_threshold", viParameters_.estimator.visual_nis_threshold);
  if (!estimatorNode["skip_pre_nis_when_tracking_healthy"].empty())
    parseEntry(estimatorNode, "skip_pre_nis_when_tracking_healthy", viParameters_.estimator.skip_pre_nis_when_tracking_healthy);
  if (!estimatorNode["lightweight_non_keyframes"].empty())
    parseEntry(estimatorNode, "lightweight_non_keyframes", viParameters_.estimator.lightweight_non_keyframes);
  if (!estimatorNode["lightweight_new_imu_nis_gate"].empty())
    parseEntry(estimatorNode, "lightweight_new_imu_nis_gate", viParameters_.estimator.lightweight_new_imu_nis_gate);
  if (!estimatorNode["max_consecutive_lightweight_frames"].empty())
    parseEntry(estimatorNode, "max_consecutive_lightweight_frames", viParameters_.estimator.max_consecutive_lightweight_frames);
  if (!estimatorNode["parallel_marginalization_accumulation"].empty())
    parseEntry(estimatorNode, "parallel_marginalization_accumulation", viParameters_.estimator.parallel_marginalization_accumulation);
  if (!estimatorNode["max_visual_imu_rotation_error"].empty())
    parseEntry(estimatorNode, "max_visual_imu_rotation_error", viParameters_.estimator.max_visual_imu_rotation_error);
  if (!estimatorNode["max_visual_imu_position_error"].empty())
    parseEntry(estimatorNode, "max_visual_imu_position_error", viParameters_.estimator.max_visual_imu_position_error);
  if (!estimatorNode["max_visual_imu_position_error_rate"].empty())
    parseEntry(estimatorNode, "max_visual_imu_position_error_rate", viParameters_.estimator.max_visual_imu_position_error_rate);
  parseEntry(file["output_parameters"], "display_matches",
             viParameters_.output.display_matches);
  parseEntry(file["output_parameters"], "display_overhead",
             viParameters_.output.display_overhead);

  // done!
  readConfigFile_ = true;
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name, int& readValue) {
  CGMLI_ASSERT_TRUE(Exception, file[name].isInt(),
                    "missing integer parameter " << file.name() << ": " << name)
  file[name] >> readValue;
}

void ViParametersReader::parseEntry(const cv::FileNode &file,
                                    std::string name, double& readValue) {
  CGMLI_ASSERT_TRUE(Exception, file[name].isReal() || file[name].isInt(),
                    "missing real parameter " << file.name() << ": " << name)
  file[name] >> readValue;
}

// Parses booleans from a cv::FileNode. OpenCV sadly has no implementation like this.
void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name, bool& readValue) {
  CGMLI_ASSERT_TRUE(Exception, file[name].isInt() || file[name].isString(),
                    "missing boolean parameter " << file.name() << ": " << name)
  if (file[name].isInt()) {
    readValue = int(file[name]) != 0;
    return;
  }
  if (file[name].isString()) {
    std::string str = std::string(file[name]);
    // cut out first word. str currently contains everything including comments
    str = str.substr(0,str.find(" "));
    // transform it to all lowercase
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    /* from yaml.org/type/bool.html:
     * Booleans are formatted as English words
     * (“true”/“false”, “yes”/“no” or “on”/“off”)
     * for readability and may be abbreviated as
     * a single character “y”/“n” or “Y”/“N”. */
    if (str.compare("false")  == 0
        || str.compare("no")  == 0
        || str.compare("n")   == 0
        || str.compare("off") == 0) {
      readValue = false;
      return;
    }
    if (str.compare("true")   == 0
        || str.compare("yes") == 0
        || str.compare("y")   == 0
        || str.compare("on")  == 0) {
      readValue = true;
      return;
    }
    CGMLI_THROW(Exception, "Boolean with uninterpretable value " << str)
  }
  return;
}

void ViParametersReader::parseEntry(const cv::FileNode &file, std::string name,
                                    Eigen::Matrix4d& readValue) {
  cv::FileNode T = file[name];
  CGMLI_ASSERT_TRUE(Exception, T.isSeq(),
                    "missing real array parameter " << file.name() << ": " << name)
  readValue << T[0], T[1], T[2], T[3],
               T[4], T[5], T[6], T[7],
               T[8], T[9], T[10], T[11],
               T[12], T[13], T[14], T[15];
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name,
                                    Eigen::Vector3d& readValue) {
  cv::FileNode T = file[name];
  CGMLI_ASSERT_TRUE(Exception, T.isSeq(),
                    "missing real array parameter " << file.name() << ": " << name)
  readValue << T[0], T[1], T[2];
}

void ViParametersReader::parseEntry(const cv::FileNode& file, std::string name,
                                    std::string& readValue) {
  cv::FileNode T = file[name];
  CGMLI_ASSERT_TRUE(Exception, T.isString(),
                    "missing string parameter " << file.name() << ": " << name)
  readValue = std::string(T);
}

bool ViParametersReader::getCameraCalibration(
    std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileStorage& configurationFile) {

  bool success = getCalibrationViaConfig(calibrations, configurationFile["cameras"]);
  return success;
}

// Get the camera calibration via the configuration file.
bool ViParametersReader::getCalibrationViaConfig(
    std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileNode cameraNode) const {

  calibrations.clear();
  bool gotCalibration = false;
  // first check if calibration is available in config file
  if (cameraNode.isSeq()
     && cameraNode.size() > 0) {
    size_t camIdx = 0;
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it) {
      const auto node = *it;
      bool valid = node.isMap() && node["cam_model"].isString()
          && numericSequence(node["T_SC"], 16) && numericSequence(node["image_dimension"], 2)
          && numericSequence(node["focal_length"], 2) && numericSequence(node["principal_point"], 2);
      if (valid) {
        Eigen::Matrix4d extrinsic;
        for (int row = 0; row < 4; ++row) for (int col = 0; col < 4; ++col)
          extrinsic(row, col) = double(node["T_SC"][row * 4 + col]);
        const Eigen::Matrix3d rotation = extrinsic.topLeftCorner<3, 3>();
        valid = (extrinsic.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() < 1e-8
            && (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-4
            && std::abs(rotation.determinant() - 1.0) < 1e-4;
        for (int axis = 0; axis < 2; ++axis)
          valid = valid && node["image_dimension"][axis].isInt() && int(node["image_dimension"][axis]) > 0
              && double(node["focal_length"][axis]) > 0;
        const std::string model = std::string(node["cam_model"]);
        if (model == "eucm") {
          // EUCM has alpha/beta, not a pinhole distortion vector. Legacy padded
          // distortion fields are harmless, but they are never required here.
          valid = valid && numericSequence(node["eucm_parameters"], 2);
          if (valid) valid = double(node["eucm_parameters"][0]) >= 0 && double(node["eucm_parameters"][0]) <= 1
              && double(node["eucm_parameters"][1]) > 0;
        } else if (model == "pinhole") {
          valid = valid && (node["distortion_type"].empty() || node["distortion_type"].isString());
          if (valid) {
            const auto name = distortionName(node["distortion_type"]);
            const auto coefficients = node["distortion_coefficients"];
            if (name == "none") {
              valid = coefficients.empty() || numericSequence(coefficients, coefficients.size());
              if (valid && !coefficients.empty()) for (const auto& value : coefficients) valid = valid && double(value) == 0;
            } else if (name == "equidistant" || name == "radialtangential" || name == "plumb_bob") {
              valid = numericSequence(coefficients, 4);
            } else if (name == "radialtangential8" || name == "plumb_bob8") {
              valid = numericSequence(coefficients, 8);
            } else valid = false;
          }
        } else valid = false;
      }
      if (valid) {
        LOG(INFO) << "Found calibration in configuration file for camera " << camIdx;
        gotCalibration = true;
      } else {
        LOG(WARNING) << "Found invalid model-specific calibration in configuration file for camera " << camIdx
                     << ". Will not use the calibration from the configuration file.";
        return false;
      }

      ++camIdx;
    }
  }
  else
    LOG(INFO) << "Did not find a calibration in the configuration file.";

  if (gotCalibration) {
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it) {

      CameraCalibration calib;

      if((std::string)((*it)["cam_model"]) == "pinhole"){
        cv::FileNode T_SC_node = (*it)["T_SC"];
        cv::FileNode imageDimensionNode = (*it)["image_dimension"];
        cv::FileNode distortionCoefficientNode = (*it)["distortion_coefficients"];
        cv::FileNode focalLengthNode = (*it)["focal_length"];
        cv::FileNode principalPointNode = (*it)["principal_point"];

        // extrinsics
        Eigen::Matrix4d T_SC;
        T_SC << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3],
                T_SC_node[4], T_SC_node[5], T_SC_node[6], T_SC_node[7],
                T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11],
                T_SC_node[12], T_SC_node[13], T_SC_node[14], T_SC_node[15];
        calib.T_SC = kinematics::Transformation(T_SC);

        calib.imageDimension << imageDimensionNode[0], imageDimensionNode[1];
        calib.distortionCoefficients.resize(int(distortionCoefficientNode.size()));
        for(int i=0; i<int(distortionCoefficientNode.size()); ++i) {
          calib.distortionCoefficients[i] = distortionCoefficientNode[i];
        }
        calib.focalLength << focalLengthNode[0], focalLengthNode[1];
        calib.principalPoint << principalPointNode[0], principalPointNode[1];
        calib.distortionType = distortionName((*it)["distortion_type"]);
        calib.cameraModel = "pinhole";
      } else if ( (std::string)((*it)["cam_model"]) == "eucm"){
        cv::FileNode T_SC_node = (*it)["T_SC"];
        cv::FileNode imageDimensionNode = (*it)["image_dimension"];
        cv::FileNode focalLengthNode = (*it)["focal_length"];
        cv::FileNode principalPointNode = (*it)["principal_point"];
        cv::FileNode eucmParamNode= (*it)["eucm_parameters"];

        // extrinsics
        Eigen::Matrix4d T_SC;
        T_SC << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3],
                T_SC_node[4], T_SC_node[5], T_SC_node[6], T_SC_node[7],
                T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11],
                T_SC_node[12], T_SC_node[13], T_SC_node[14], T_SC_node[15];
        calib.T_SC = cgmli::kinematics::Transformation(T_SC);

        calib.imageDimension << imageDimensionNode[0], imageDimensionNode[1];
        calib.focalLength << focalLengthNode[0], focalLengthNode[1];
        calib.principalPoint << principalPointNode[0], principalPointNode[1];
        calib.eucmParameters << eucmParamNode[0], eucmParamNode[1];
        calib.distortionType = "None";
        calib.cameraModel = "eucm";
      }

      // parse additional stuff
      if((*it)["camera_type"].isString()){
        std::string camera_type = std::string((*it)["camera_type"]);
        if(camera_type.compare(0,4,"gray")==0) {
          calib.cameraType.isColour = false;
        } else {
          calib.cameraType.isColour = true;
        }
        if(camera_type.size()>=6){
          if(camera_type.compare(camera_type.size()-6,6,"+depth")==0) {
            calib.cameraType.depthType.isDepthCamera = true;
          } else {
            calib.cameraType.depthType.isDepthCamera = false;
          }
        }
      }
      if((*it)["slam_use"].isString()){
        std::string slam_use = std::string((*it)["slam_use"]);
        if(slam_use.compare(0,5,"cgmli")==0) {
          calib.cameraType.isUsed = true;
        } else {
          calib.cameraType.isUsed = false;
        }
        if(slam_use.size()>=6){
          if(slam_use.compare(slam_use.size()-6,6,"-depth")==0) {
            calib.cameraType.depthType.createDepth = true;
          } else {
            calib.cameraType.depthType.createDepth = false;
          }
        }
        if(slam_use.size()>=8){
          if(slam_use.compare(slam_use.size()-8,8,"-virtual")==0) {
            calib.cameraType.depthType.createVirtual = true;
          } else {
            calib.cameraType.depthType.createVirtual = false;
          }
        }
      }
      if((*it)["sigma_pixels"].isReal()){
        calib.cameraType.depthType.sigmaPixels = ((*it)["sigma_pixels"]);
      }
      if((*it)["sigma_depth"].isReal()){
        calib.cameraType.depthType.sigmaPixels = ((*it)["sigma_depth"]);
      }

      calibrations.push_back(calib);
    }
  }
  return gotCalibration;
}

bool ViParametersReader::getLiDARCalibration(const cv::FileNode& calibrationNode, cgmli::LidarParameters& lidarParameters){
  
  if(!calibrationNode["T_SL"].isSeq() || !calibrationNode["elevation_resolution_angle"].isReal() || !calibrationNode["azimuth_resolution_angle"].isReal()){
    return false;
  }

  cv::FileNode T_SL_node = calibrationNode["T_SL"];
  Eigen::Matrix4d T_SL;
  T_SL << T_SL_node[0], T_SL_node[1], T_SL_node[2], T_SL_node[3],
          T_SL_node[4], T_SL_node[5], T_SL_node[6], T_SL_node[7],
          T_SL_node[8], T_SL_node[9], T_SL_node[10], T_SL_node[11],
          T_SL_node[12], T_SL_node[13], T_SL_node[14], T_SL_node[15];
  lidarParameters.T_SL = cgmli::kinematics::Transformation(T_SL);
  lidarParameters.elevation_resolution_angle = calibrationNode["elevation_resolution_angle"];
  lidarParameters.azimuth_resolution_angle = calibrationNode["azimuth_resolution_angle"];

  return true;
}

}  // namespace cgmli
