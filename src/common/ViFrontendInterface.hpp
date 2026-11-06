

#ifndef INCLUDE_CGMLI_VIOFRONTENDINTERFACE_HPP_
#define INCLUDE_CGMLI_VIOFRONTENDINTERFACE_HPP_

#include <vector>
#include <memory>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <kinematics/Transformation.hpp>

#include <common/Measurements.hpp>
#include <common/Parameters.hpp>
#include <utils/Time.h>
#include <common/FrameTypedefs.hpp>
#include <visual/MultiFrame.hpp>

/// \brief cgmli Main namespace of this package.
namespace cgmli {

class ViSlamBackend;

/// \brief The VI-SLAM estimator class.
using Estimator = ViSlamBackend;

/**
 * @brief The VioFrontendInterface class is an interface for frontends.
 */
class ViFrontendInterface {
 public:
  /// \brief Default constructor.
  ViFrontendInterface() = default;

  /// \brief Default destructor.
  virtual ~ViFrontendInterface() = default;

  /// @name
  /// In the derived class, the following methods (and nothing else) have to be implemented:
  ///@{
  /**
   * @brief Detection and descriptor extraction on a per image basis.
   * @param cameraIndex   Index of camera to do detection and description.
   * @param frameOut      Multiframe containing the frames.
   *                      Resulting keypoints and descriptors are saved in here.
   * @param T_WC          Pose of camera with index cameraIndex at image capture time.
   * @param[in] keypoints If the keypoints are already available from a different source, provide
   *                      them here in order to skip detection.
   * @return True if successful.
   */
  virtual bool detectAndDescribe(
      size_t cameraIndex, std::shared_ptr<cgmli::MultiFrame> frameOut,
      const cgmli::kinematics::Transformation& T_WC,
      const std::vector<cv::KeyPoint> * keypoints) = 0;

  /**
   * @brief Matching as well as initialization of landmarks and state.
   * @param estimator       Estimator.
   * @param params          Configuration parameters.
   * @param framesInOut     Multiframe including the descriptors of all the keypoints.
   * @param kfPrior         A prior to trigger new keyframe from other criteria (e.g. LiDAR overlap)
   * @param[out] asKeyframe Should the frame be a keyframe?
   * @return True if successful.
   */
  virtual bool dataAssociationAndInitialization(
      Estimator& estimator,
      const cgmli::ViParameters & params,
      std::shared_ptr<cgmli::MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) = 0;

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @see cgmli::ceres::ImuError::propagation()
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS_propagated Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @return True on success.
   */
  virtual bool propagation(const cgmli::ImuMeasurementDeque & imuMeasurements,
                           const cgmli::ImuParameters & imuParams,
                           cgmli::kinematics::Transformation& T_WS_propagated,
                           cgmli::SpeedAndBias & speedAndBiases,
                           const cgmli::Time& t_start, const cgmli::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const = 0;

  ///@}
};

}

#endif /* INCLUDE_CGMLI_VIOFRONTENDINTERFACE_HPP_ */
