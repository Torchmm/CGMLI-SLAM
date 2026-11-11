
#ifndef INCLUDE_CGMLI_PARAMETERS_HPP_
#define INCLUDE_CGMLI_PARAMETERS_HPP_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop
#include <Eigen/Dense>
#include <utils/Time.h>
#include <cameras/NCameraSystem.hpp>
#include <kinematics/Transformation.hpp>
#include <optional>
#include <string>

/// \brief cgmli Main namespace of this package.
namespace cgmli {

/// @brief Struct that contains all the camera calibration information.
struct CameraCalibration {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  cgmli::kinematics::Transformation T_SC;   ///< Transformation from camera to sensor (IMU) frame.
  Eigen::Vector2i imageDimension;           ///< Image dimension. [pixels]
  Eigen::VectorXd distortionCoefficients;   ///< Distortion Coefficients.
  Eigen::Vector2d focalLength;              ///< Focal length.
  Eigen::Vector2d principalPoint;           ///< Principal point.
  std::string cameraModel;                  ///< camera model. ('pinhole' 'eucm')
  Eigen::Vector2d eucmParameters;           ///< alpha, beta eucm parameters

  /// \brief Distortion type. ('radialtangential' 'radialtangential8' 'equdistant')
  std::string distortionType;

  cameras::NCameraSystem::CameraType cameraType; ///< Some additional info about the camera.
};

/*!
 * \brief Camera parameters.
 *
 * A simple struct to specify properties of a Camera.
 *
 */
struct CameraParameters{
  double timestamp_tolerance; ///< Stereo tolerance; Hilti offline uses this as the maximum four-midpoint bundle span [s].

  /// \brief Signed correction added to the raw image timestamp [s].
  double image_delay;
  Time correctedImageTime(const Time& rawTime) const { return rawTime + Duration(image_delay); }

  /**
   * @brief Some parameters to set the online calibrator.
   */
  struct OnlineCalibrationParameters {
    bool do_extrinsics; ///< Do we online-calibrate extrinsics?
    bool do_extrinsics_final_ba; ///< Do we calibrate extrinsics in final BA?
    double sigma_r; ///< T_SCi position prior stdev [m]
    double sigma_alpha; ///< T_SCi orientation prior stdev [rad]
    double sigma_r_final_ba; ///< T_SCi position prior stdev in final BA [m]
    double sigma_alpha_final_ba; ///< T_SCi orientation prior stdev in final BA [rad]
    bool estimate_camera_rig_translation = true; ///< Allow translation after rotation has been calibrated.
    bool estimate_td_camera = false;
    double td_camera_initial_s = 0.0; ///< Residual offset [s]. t_IMU = t_camera_raw + image_delay + td.
    double sigma_td_camera_s = 0.005;
    double max_td_change_s = 0.025; ///< Allowed departure from the initial residual offset.
    double max_rig_translation_change_m = 0.10;
    double max_rig_rotation_change_rad = 0.15;
  };

  OnlineCalibrationParameters online_calibration; ///< Online calibration parameters.
};

/*!
 * \brief IMU parameters.
 *
 * A simple struct to specify properties of an IMU.
 *
 */
struct ImuParameters{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool use; ///< Use the IMU at all?
  cgmli::kinematics::Transformation T_BS; ///< Maps IMU sensor S coordinates into Body B.
  double a_max;  ///< Accelerometer saturation. [m/s^2]
  double g_max;  ///< Gyroscope saturation. [rad/s]
  double sigma_g_c;  ///< Gyroscope noise density.
  double sigma_bg;  ///< Initial gyroscope bias.
  double sigma_a_c;  ///< Accelerometer noise density.
  double sigma_ba;  ///< Initial accelerometer bias
  double sigma_gw_c; ///< Gyroscope drift noise density.
  double sigma_aw_c; ///< Accelerometer drift noise density.
  Eigen::Vector3d g0;  ///< Mean of the prior gyro bias.
  Eigen::Vector3d a0;  ///< Mean of the prior accelerometer bias.
  double g;  ///< Earth acceleration.
  Eigen::Vector3d s_a; ///< Scale factor for accelerometer measurements
  double max_sample_gap = 0.05; ///< Maximum interval between physical IMU samples [s].
};

/**
 * @brief Parameters for detection etc.
 */
struct FrontendParameters {
  int max_num_keypoints; ///< Restrict to a maximum of this many keypoints per img (strongest ones).
  double keyframe_parallax = 0.015; ///< Rotation-compensated S2 parallax [rad], not normalized pinhole pixels.
  double keyframe_translation = 0.10; ///< Accumulated displacement from retained reference keyframe [m].
  double keyframe_rotation = 0.10; ///< Accumulated rotation from reference keyframe [rad].
  int keyframe_min_shared_tracks = 20; ///< Independent same-camera landmark tracks for the parallax decision.
  double keyframe_low_motion_translation = 0.015; ///< Latest-frame low-motion envelope [m].
  double keyframe_low_motion_rotation = 0.015; ///< Latest-frame low-motion envelope [rad].
  bool parallelise_detection; ///< Run parallel detect & describe.
  int num_matching_threads; ///< Parallelise matching with this number of threads.
  // Native ORB settings use FAST thresholds and binary Hamming distance.
  int orb_pyramid_levels = 8;
  double orb_scale_factor = 1.2;
  int orb_fast_threshold = 20;
  int max_promotions_per_frame = 40; ///< Healthy metric tracking; zero disables.
  int max_active_initialized = 1000; ///< Soft promotion ceiling, not forced retirement.
  bool orb_layer_fast = true; ///< Shared per-layer FAST scores, cell-local NMS; false selects ROI reference.
  bool orb_boundary_fast = true; ///< Raw high-threshold FAST only near grid borders.
  bool orb_adaptive_geometry = true; ///< Replenished per-cell geometry batches; false uses the full candidate cap.
  bool orb_batch_descriptor_coordinates = true;
  bool orb_grouped_balancing = true; ///< Exact representative heap instead of all candidates in the heap.
  bool pose_analytic_absolute_jacobians = true;
  bool pose_adaptive_ransac = true; ///< Healthy tracking floor 20, periodic/recovery floor remains 50.
  bool orb_staged_matching = true; ///< Map-first matching with coverage/motion and pose-consensus fallbacks.
  bool orb_cuda_matcher = false; ///< Use CUDA for guided temporal and map matching when available.
  bool orb_cuda_early_integral = true; ///< Overlap CUDA integral images with CPU keypoint selection.
  bool require_keyframe_anchor_for_promotion = true; ///< Keep non-keyframe tracks as seeds until a stable reference is available.
  int max_tracking_seeds = 1000; ///< Independent admission budget; zero disables.
  double seed_max_untracked_seconds = 0.5; ///< Last-observation age before unused seeds expire; zero disables.
  int orb_min_fast_threshold = 7;
  double orb_virtual_focal_length = 0.0; ///< 0 selects camera-0 fu as one shared rig reference.
  int orb_matching_threshold = 64; ///< Absolute Hamming distance in [0,256].
};

/**
 * @brief Parameters regarding the estimator.
 */
struct EstimatorParameters {
  // Health checks use explicit physical units. Pose envelopes are engineering
  // bounds, not covariance-derived probabilities or extra optimization factors.
  int backend_max_retries = 1;
  double visual_nis_threshold = 9.210340371976184; ///< Squared whitened 2D bearing residual.
  bool skip_pre_nis_when_tracking_healthy = true;
  bool lightweight_non_keyframes = false; ///< Refine the two newest VIO epochs before removing the penultimate frame.
  bool lightweight_new_imu_nis_gate = false; ///< Gate the newest IMU edge after a lightweight solve.
  int max_consecutive_lightweight_frames = 4; ///< Zero prevents lightweight solves.
  bool parallel_marginalization_accumulation = true;
    double max_imu_interval_seconds = 10.0;
    bool incremental_imu = true;
    bool preserve_oldest_visual_information = true;
    bool backend_dense_schur = false; ///< Controlled alternative for the small pose/SB separator.
    bool cache_visual_pairs = true; ///< Solver-local camera-pair geometry cache, refreshed at every new point.
    std::string backend_visual_batch = "cpu"; ///< CPU, CUDA, hybrid, Schur, or audit path for visual optimization.
  double max_visual_imu_rotation_error = 0.2; ///< Relative orientation error [rad].
  double max_visual_imu_position_error = 0.5; ///< Position error allowance [m].
  double max_visual_imu_position_error_rate = 2.0; ///< Additional allowance per elapsed second [m/s].
  int window_size = 10; ///< Retained image-frame states; one temporary insertion slot, no sensor-specific quotas.
  bool do_loop_closures; ///< Whether to do VI-SLAM or VIO.
  bool do_final_ba; ///< Whether to run a final full BA.
  bool enforce_realtime; ///< Whether to limit the time budget for optimisation.
  int realtime_min_iterations; ///< Minimum number of iterations always performed.
  int realtime_max_iterations; ///< Never do more than these, even if not converged.
  double realtime_time_limit; ///< Ceres-only soft time allowance shared across refits [s]; requires enforce_realtime.
  int realtime_num_threads; ///< Number of threads for the realtime optimisation.
  int full_graph_iterations; ///< Don't do more than these for the full (background) optimisation.
  int full_graph_num_threads; ///< Number of threads for the full (background) optimisation.
  double p_dbow; ///< Match threshold for dBoW -- unfortunately this varies with setups.
};

/**
 * @brief Some options for how and what to output.
 */
struct OutputParameters {
    bool display_matches = false; ///< Enables feature, stereo, temporal, reprojection and accepted track-trail views.
    bool display_overhead = false; ///< Accepted local trajectory in XY/XZ; rendered on the visualization worker.
};


/// @brief  Struct to specify the parameters of a LiDAR sensor
struct LidarParameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    cgmli::kinematics::Transformation T_SL;   ///< Transformation from LiDAR to sensor (IMU) frame.
    float elevation_resolution_angle; ///< Resolution angle for the elevation of the LiDAR sensor
    float azimuth_resolution_angle; ///< Resolution angle for the azimuth of the LiDAR sensor
};


/// @brief Struct to combine all parameters and settings.
struct ViParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  cgmli::cameras::NCameraSystem nCameraSystem;  ///< Camera extrinsics and intrinsics.
  CameraParameters camera; ///< Camera parameters.
  ImuParameters imu; ///< Imu parameters.
  std::optional<LidarParameters> lidar; ///< LiDAR parameters
  FrontendParameters frontend; ///< Frontend parameters.
  EstimatorParameters estimator; ///< Estimator parameters.
  OutputParameters output; ///< Output parameters.
  CameraCalibration rgb;  ///< RGB parameters.
};

} // namespace cgmli

#endif // INCLUDE_CGMLI_PARAMETERS_HPP_
