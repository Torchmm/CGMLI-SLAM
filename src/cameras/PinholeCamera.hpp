
#ifndef INCLUDE_CGMLI_CAMERAS_PINHOLECAMERA_HPP_
#define INCLUDE_CGMLI_CAMERAS_PINHOLECAMERA_HPP_

#include <vector>
#include <memory>
#include <stdint.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#include "cameras/CameraBase.hpp"
#include "cameras/DistortionBase.hpp"
#include "cameras/NoDistortion.hpp"

/// \brief cgmli Main namespace of this package.
namespace cgmli {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

/// \class PinholeCamera
/// \brief This implements a standard pinhole camera projection model.
/// \tparam DISTORTION_T the distortion type, e.g. cgmli::cameras::RadialTangentialDistortion
template<class DISTORTION_T>
class PinholeCamera; // forward declaration

/// \class PinholeCameraBase
/// \brief This is an interface for all the different distortion versions, allowing generic
/// undistortion.
class PinholeCameraBase : public CameraBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief Constructor for width, height and Id
  inline PinholeCameraBase(int imageWidth, int imageHeight, uint64_t id = 0)
        : CameraBase(imageWidth, imageHeight, id)
  {
  }

  /// \brief Destructor.
  virtual ~PinholeCameraBase() override = default;

  /// \brief Initialise undistort maps to defaults, i.e.
  /// undistortedFocalLengh = 0.5 * (focalLengthU() + focalLengthV()) (same for U and V), 
  /// same image dimensions and center in the middle, i.e
  /// undistortedImageCenterU() = 0.5 * imageWith() + 0.5.
  /// \return True on success.
  virtual bool initialiseUndistortMaps() = 0;

  /// \brief Initialise undistort maps, provide custom parameters for the undistorted cam.
  /// @param[in] undistortedImageWidth The width in pixels.
  /// @param[in] undistortedImageHeight The height in pixels.
  /// @param[in] undistortedFocalLengthU The horizontal focal length in pixels.
  /// @param[in] undistortedFocalLengthV The vertical focal length in pixels.
  /// @param[in] undistortedImageCenterU The horizontal centre in pixels.
  /// @param[in] undistortedImageCenterV The vertical centre in pixels.
  /// \return True on success.
  virtual bool initialiseUndistortMaps(int undistortedImageWidth, int undistortedImageHeight, 
      double undistortedFocalLengthU, double undistortedFocalLengthV, 
      double undistortedImageCenterU, double undistortedImageCenterV) = 0;

  /// \brief Get undistorted image -- assumes initialiseUndistortMaps was called
  /// @param[in] srcImg The distorted input image.
  /// @param[out] destImg The undistorted output image.
  /// \return True on success.
  virtual bool undistortImage(const cv::Mat & srcImg, cv::Mat & destImg) const = 0;

  /// \brief Get the model of the undistorted camera.
  /// \return The PinholeCamera without distortion associated with the undistorted image.
  virtual PinholeCamera<NoDistortion> undistortedPinholeCamera() const = 0;

  /// \brief Get the focal length along the u-dimension.
  /// \return The horizontal focal length in pixels.
  virtual double focalLengthU() const = 0;

  /// \brief Get the focal length along the v-dimension.
  /// \return The vertical focal length in pixels.
  virtual double focalLengthV() const = 0;

  /// \brief Get the image centre along the u-dimension.
  /// \return The horizontal centre in pixels.
  virtual double imageCenterU() const = 0;

  /// \brief Get the focal image centre along the v-dimension.
  /// \return The vertical centre in pixels.
  virtual double imageCenterV() const = 0;
};

template<class DISTORTION_T>
class PinholeCamera : public PinholeCameraBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef DISTORTION_T distortion_t; ///< Makes the distortion type accessible.

  /// \brief Constructor that will figure out the type of distortion
  /// @param[in] imageWidth The width in pixels.
  /// @param[in] imageHeight The height in pixels.
  /// @param[in] focalLengthU The horizontal focal length in pixels.
  /// @param[in] focalLengthV The vertical focal length in pixels.
  /// @param[in] imageCenterU The horizontal centre in pixels.
  /// @param[in] imageCenterV The vertical centre in pixels.
  /// @param[in] distortion The distortion object to be used.
  /// @param[in] id Assign a generic ID, if desired.
  PinholeCamera(int imageWidth, int imageHeight, double focalLengthU,
      double focalLengthV, double imageCenterU, double imageCenterV,
      const distortion_t& distortion, uint64_t id = std::numeric_limits<uint64_t>::max()) : PinholeCameraBase(imageWidth, imageHeight, id),
      distortion_(distortion),
      fu_(focalLengthU),
      fv_(focalLengthV),
      cu_(imageCenterU),
      cv_(imageCenterV)
  {
      intrinsics_[0] = fu_;  //< focalLengthU
      intrinsics_[1] = fv_;  //< focalLengthV
      intrinsics_[2] = cu_;  //< imageCenterU
      intrinsics_[3] = cv_;  //< imageCenterV
      one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
      one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
      fu_over_fv_ = fu_ / fv_;  //< fu_ / fv_
  }

  /// \brief Destructor.
  virtual ~PinholeCamera() override = default;

  static const int NumProjectionIntrinsics = 4;  ///< optimisable projection intrinsics
  static const int NumIntrinsics = NumProjectionIntrinsics
      + distortion_t::NumDistortionIntrinsics; ///< total number of intrinsics

  /// \brief Get the focal length along the u-dimension.
  /// \return The horizontal focal length in pixels.
  virtual double focalLengthU() const override final
  {
    return fu_;
  }

  /// \brief Get the focal length along the v-dimension.
  /// \return The vertical focal length in pixels.
  virtual double focalLengthV() const override final
  {
    return fv_;
  }

  /// \brief Get the image centre along the u-dimension.
  /// \return The horizontal centre in pixels.
  virtual double imageCenterU() const override final
  {
    return cu_;
  }

  /// \brief Get the focal image centre along the v-dimension.
  /// \return The vertical centre in pixels.
  virtual double imageCenterV() const override final
  {
    return cv_;
  }

  /// \brief Get the intrinsics as a concatenated vector.
  /// \param intrinsics The intrinsics as a concatenated vector.
  inline void getIntrinsics(Eigen::VectorXd & intrinsics) const override final
  {
      intrinsics = intrinsics_;
      Eigen::VectorXd distortionIntrinsics;
      if (distortion_t::NumDistortionIntrinsics > 0) {
          distortion_.getParameters(distortionIntrinsics);
          intrinsics.tail<distortion_t::NumDistortionIntrinsics>() = distortionIntrinsics;
      }
  }

  /// \brief overwrite all intrinsics - use with caution !
  /// \param[in] intrinsics The intrinsics as a concatenated vector.
  inline bool setIntrinsics(const Eigen::VectorXd& intrinsics) override final
  {
      if (intrinsics.cols() != NumIntrinsics) {
          return false;
      }
      intrinsics_ = intrinsics;
      fu_ = intrinsics[0];  //< focalLengthU
      fv_ = intrinsics[1];  //< focalLengthV
      cu_ = intrinsics[2];  //< imageCenterU
      cv_ = intrinsics[3];  //< imageCenterV
      distortion_.setParameters(
          intrinsics.tail<distortion_t::NumDistortionIntrinsics>());
      one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
      one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
      fu_over_fv_ = fu_ / fv_;  //< fu_ / fv_
      return true;
  }

  /// \brief overwrite undistortion map - use with caution !
  /// \param map1 Rectification map for horizontal direction
  /// \param map2 Rectification map for vertical direction
  virtual void setRectifyMap(const cv::Mat& map1, const cv::Mat& map2)
  {
      map_x_fast_ = map1;
      map_y_fast_ = map2;
  }
  
  /// \brief get rectification map
  virtual void getRectifyMap(cv::Mat& map1, cv::Mat& map2) const
  {
      map1 = map_x_fast_.clone();
      map2 = map_y_fast_.clone();
  }

  /// \brief Get the total number of intrinsics.
  /// \return Number of intrinsics parameters.
  inline int noIntrinsicsParameters() const override final
  {
    return NumIntrinsics;
  }

  /// \brief Initialise undistort maps to defaults, i.e.
  /// undistortedFocalLengh = 0.5 * (focalLengthU() + focalLengthV()) (same for U and V), 
  /// same image dimensions and center in the middle, i.e
  /// undistortedImageCenterU() = 0.5 * imageWith() + 0.5.
  /// \return True on success.
  virtual bool initialiseUndistortMaps() override final
  {
      const double f = (fu_ + fv_) * 0.5;
      return initialiseUndistortMaps(imageWidth(), imageHeight(), f, f,
          double(imageWidth()) * 0.5 + 0.5, double(imageHeight()) * 0.5 + 0.5);
  }

  /// \brief Generate camera awareness maps as required by BRISK 2.
  /// \return True on success.
  virtual bool initialiseCameraAwarenessMaps() override final
  {
      rays_ = cv::Mat(imageHeight_, imageWidth_, CV_32FC3);
      imageJacobians_ = cv::Mat(imageHeight_, imageWidth_, CV_32FC(6));
      for (int v = 0; v < imageHeight_; ++v) {
          for (int u = 0; u < imageWidth_; ++u) {
              Eigen::Vector3d ray;
              Eigen::Matrix<double, 2, 3> jacobian;
              if (backProject(Eigen::Vector2d(u, v), &ray)) {
                  ray.normalize();
              }
              else {
                  ray.setZero();
              }
              rays_.at<cv::Vec3f>(v, u) = cv::Vec3f(ray[0], ray[1], ray[2]);
              Eigen::Vector2d pt;
              if (project(ray, &pt, &jacobian)
                  == cameras::ProjectionStatus::Successful) {
                  cv::Vec6f j;
                  j[0] = jacobian(0, 0);
                  j[1] = jacobian(0, 1);
                  j[2] = jacobian(0, 2);
                  j[3] = jacobian(1, 0);
                  j[4] = jacobian(1, 1);
                  j[5] = jacobian(1, 2);
                  imageJacobians_.at<cv::Vec6f>(v, u) = j;
              }
          }
      }
      return true;
  }

  /// \brief Get camera awareness maps as required by BRISK 2.
  /// @param[out] rays Rays per pixel.
  /// @param[out] imageJacobians Jacobian per pixel.
  /// \return True on success.
  virtual bool getCameraAwarenessMaps(cv::Mat& rays, cv::Mat& imageJacobians) const override final
  {
      rays = rays_;
      imageJacobians = imageJacobians_;
      if (rays_.empty() || imageJacobians_.empty()) {
          return false;
      }
      return true;
  }

  /// \brief Initialise undistort maps, provide custom parameters for the undistorted cam.
  /// @param[in] undistortedImageWidth The width in pixels.
  /// @param[in] undistortedImageHeight The height in pixels.
  /// @param[in] undistortedFocalLengthU The horizontal focal length in pixels.
  /// @param[in] undistortedFocalLengthV The vertical focal length in pixels.
  /// @param[in] undistortedImageCenterU The horizontal centre in pixels.
  /// @param[in] undistortedImageCenterV The vertical centre in pixels.
  /// \return True on success.
  virtual bool initialiseUndistortMaps(int undistortedImageWidth, int undistortedImageHeight,
      double undistortedFocalLengthU, double undistortedFocalLengthV,
      double undistortedImageCenterU, double undistortedImageCenterV) override final
  {
      // store parameters
      undistortedImageWidth_ = undistortedImageWidth;
      undistortedImageHeight_ = undistortedImageHeight;
      undistortedFocalLengthU_ = undistortedFocalLengthU;
      undistortedFocalLengthV_ = undistortedFocalLengthV;
      undistortedImageCenterU_ = undistortedImageCenterU;
      undistortedImageCenterV_ = undistortedImageCenterV;

      // some preparation of the actual and undistorted projections
      Eigen::Matrix2d undistortedK_inv, actualK;
      undistortedK_inv << 1.0 / undistortedFocalLengthU, 0.0, 0.0, 1.0 / undistortedFocalLengthV;
      actualK << fu_, 0.0, 0.0, fv_;
      Eigen::Vector2d actualCenter(cu_, cv_);
      Eigen::Vector2d undistortedCenter(undistortedImageCenterU, undistortedImageCenterV);

      // Create the maps and vectors for points
      cv::Mat map_x(undistortedImageHeight, undistortedImageWidth, CV_32F);
      cv::Mat map_y(undistortedImageHeight, undistortedImageWidth, CV_32F);
      Eigen::Vector2d pixel, imgPlane, projectedPoint, distortedPoint, mappedPixel;
      Eigen::Vector3d ray, rayTransformed;
      Eigen::Vector4d hrayTransformed;
      const double rayLength = 0.25;

      for (int y = 0; y < undistortedImageHeight; y++) {

          pixel(1) = y;

          float* pmap_x = map_x.ptr<float>(y); // for the yth row in the map
          float* pmap_y = map_y.ptr<float>(y);

          for (int x = 0; x < undistortedImageWidth; x++) {

              pixel(0) = x;

              // Convert from pixels to image plane using ideal camera intrinsics
              imgPlane = undistortedK_inv * (pixel - undistortedCenter);

              // Shoot a ray through the (x,y) point
              ray = rayLength * imgPlane.homogeneous();

              // Transform that ray into the frame of the actual camera
              hrayTransformed = /*T_LC * */ ray.homogeneous();
              rayTransformed = hrayTransformed.hnormalized();

              // Project the transformed ray into the actual camera
              projectedPoint = rayTransformed.hnormalized();

              // Apply the distortion model to the projection
              distortion_.distort(projectedPoint, &distortedPoint);

              // Apply the intrinsics model to get a pixel location
              mappedPixel = (actualK * distortedPoint) + actualCenter;

              // Assign that pixel location to the map
              pmap_x[x] = mappedPixel(0); // assign a value to the (x,y) position in the map
              pmap_y[x] = mappedPixel(1);

          }
      }

      // Apply convertMaps for actual fast remapping later when calling undistortImage
      cv::convertMaps(map_x, map_y, map_x_fast_, map_y_fast_, CV_16SC2);

      return true;
  }

  /// \brief Get the model of the undistorted camera.
  /// \return The PinholeCamera without distortion associated with the undistorted image.
  virtual PinholeCamera<NoDistortion> undistortedPinholeCamera() const override final
  {
      assert(map_x_fast_.cols != 0);
      return PinholeCamera<NoDistortion>(undistortedImageWidth_, undistortedImageHeight_,
          undistortedFocalLengthU_, undistortedFocalLengthV_,
          undistortedImageCenterU_, undistortedImageCenterV_,
          NoDistortion());
  }

  /// \brief Get undistorted image -- assumes initialiseUndistortMaps was called
  /// @param[in] srcImg The distorted input image.
  /// @param[out] destImg The undistorted output image.
  /// \return True on success.
  virtual bool undistortImage(const cv::Mat& srcImg, cv::Mat& destImg) const override final
  {
      cv::remap(srcImg, destImg, map_x_fast_, map_y_fast_,
          cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
      return true;
  }

  //////////////////////////////////////////////////////////////
  /// \name Methods to project points
  /// @{

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point      The point in Euclidean coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus project(
      const Eigen::Vector3d& point, Eigen::Vector2d* imagePoint) const override final
  {
      // handle singularity
      if (fabs(point[2]) < 1.0e-12) {
          return ProjectionStatus::Invalid;
      }

      // projection
      Eigen::Vector2d imagePointUndistorted;
      const double rz = 1.0 / point[2];
      imagePointUndistorted[0] = point[0] * rz;
      imagePointUndistorted[1] = point[1] * rz;

      // distortion
      Eigen::Vector2d imagePoint2;
      if (!distortion_.distort(imagePointUndistorted, &imagePoint2)) {
          return ProjectionStatus::Invalid;
      }

      // scale and offset
      (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
      (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

      if (!CameraBase::isInImage(*imagePoint)) {
          return ProjectionStatus::OutsideImage;
      }
      if (CameraBase::isMasked(*imagePoint)) {
          return ProjectionStatus::Masked;
      }
      if (point[2] > 0.0) {
          return ProjectionStatus::Successful;
      }
      else {
          return ProjectionStatus::Behind;
      }
  }

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus project(
      const Eigen::Vector3d& point, Eigen::Vector2d* imagePoint,
      Eigen::Matrix<double, 2, 3>* pointJacobian,
      Eigen::Matrix2Xd* intrinsicsJacobian = nullptr) const override final
  {
      // handle singularity
      if (fabs(point[2]) < 1.0e-12) {
          return ProjectionStatus::Invalid;
      }

      // projection
      Eigen::Vector2d imagePointUndistorted;
      const double rz = 1.0 / point[2];
      const double rz2 = rz * rz;
      imagePointUndistorted[0] = point[0] * rz;
      imagePointUndistorted[1] = point[1] * rz;

      Eigen::Matrix<double, 2, 3> pointJacobianProjection;
      Eigen::Matrix2Xd intrinsicsJacobianProjection;
      Eigen::Matrix2d distortionJacobian;
      Eigen::Matrix2Xd intrinsicsJacobianDistortion;
      Eigen::Vector2d imagePoint2;

      bool distortionSuccess;
      if (intrinsicsJacobian) {
          // get both Jacobians
          intrinsicsJacobian->resize(2, NumIntrinsics);

          distortionSuccess = distortion_.distort(imagePointUndistorted, &imagePoint2,
              &distortionJacobian,
              &intrinsicsJacobianDistortion);
          // compute the intrinsics Jacobian
          intrinsicsJacobian->template topLeftCorner<2, 2>() =
              imagePoint2.asDiagonal();
          intrinsicsJacobian->template block<2, 2>(0, 2) = Eigen::Matrix2d::Identity();

          if (distortion_t::NumDistortionIntrinsics > 0) {
              intrinsicsJacobian
                  ->template bottomRightCorner<2, distortion_t::NumDistortionIntrinsics>() =
                  Eigen::Vector2d(fu_, fv_).asDiagonal() * intrinsicsJacobianDistortion;  // chain rule
          }
      }
      else {
          // only get point Jacobian
          distortionSuccess = distortion_.distort(imagePointUndistorted, &imagePoint2,
              &distortionJacobian);
      }

      // compute the point Jacobian in any case
      Eigen::Matrix<double, 2, 3>& J = *pointJacobian;
      J(0, 0) = fu_ * distortionJacobian(0, 0) * rz;
      J(0, 1) = fu_ * distortionJacobian(0, 1) * rz;
      J(0, 2) = -fu_
          * (point[0] * distortionJacobian(0, 0)
              + point[1] * distortionJacobian(0, 1)) * rz2;
      J(1, 0) = fv_ * distortionJacobian(1, 0) * rz;
      J(1, 1) = fv_ * distortionJacobian(1, 1) * rz;
      J(1, 2) = -fv_
          * (point[0] * distortionJacobian(1, 0)
              + point[1] * distortionJacobian(1, 1)) * rz2;

      // scale and offset
      (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
      (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

      if (!distortionSuccess) {
          return ProjectionStatus::Invalid;
      }
      if (!CameraBase::isInImage(*imagePoint)) {
          return ProjectionStatus::OutsideImage;
      }
      if (CameraBase::isMasked(*imagePoint)) {
          return ProjectionStatus::Masked;
      }
      if (point[2] > 0.0) {
          return ProjectionStatus::Successful;
      }
      else {
          return ProjectionStatus::Behind;
      }
  }

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus projectWithExternalParameters(
      const Eigen::Vector3d& point, const Eigen::VectorXd& parameters,
      Eigen::Vector2d* imagePoint, Eigen::Matrix<double, 2, 3>* pointJacobian,
      Eigen::Matrix2Xd* intrinsicsJacobian = nullptr) const override final
  {
      // handle singularity
      if (fabs(point[2]) < 1.0e-12) {
          return ProjectionStatus::Invalid;
      }

      // parse parameters into human readable form
      const double fu = parameters[0];
      const double fv = parameters[1];
      const double cu = parameters[2];
      const double cv = parameters[3];

      Eigen::VectorXd distortionParameters;
      if (distortion_t::NumDistortionIntrinsics > 0) {
          distortionParameters = parameters
              .template tail<distortion_t::NumDistortionIntrinsics>();
      }

      // projection
      Eigen::Vector2d imagePointUndistorted;
      const double rz = 1.0 / point[2];
      const double rz2 = rz * rz;
      imagePointUndistorted[0] = point[0] * rz;
      imagePointUndistorted[1] = point[1] * rz;

      Eigen::Matrix<double, 2, 3> pointJacobianProjection;
      Eigen::Matrix2Xd intrinsicsJacobianProjection;
      Eigen::Matrix2d distortionJacobian;
      Eigen::Matrix2Xd intrinsicsJacobianDistortion;
      Eigen::Vector2d imagePoint2;

      bool distortionSuccess;
      if (intrinsicsJacobian) {
          // get both Jacobians
          intrinsicsJacobian->resize(2, NumIntrinsics);

          distortionSuccess = distortion_.distortWithExternalParameters(imagePointUndistorted,
              distortionParameters, &imagePoint2,
              &distortionJacobian,
              &intrinsicsJacobianDistortion);
          // compute the intrinsics Jacobian
          intrinsicsJacobian->template topLeftCorner<2, 2>() =
              imagePoint2.asDiagonal();
          intrinsicsJacobian->template block<2, 2>(0, 2) = Eigen::Matrix2d::Identity();

          if (distortion_t::NumDistortionIntrinsics > 0) {
              intrinsicsJacobian
                  ->template bottomRightCorner<2, distortion_t::NumDistortionIntrinsics>() =
                  Eigen::Vector2d(fu, fv).asDiagonal() * intrinsicsJacobianDistortion;  // chain rule
          }
      }
      else {
          // only get point Jacobian
          distortionSuccess = distortion_.distortWithExternalParameters(imagePointUndistorted,
              distortionParameters, &imagePoint2,
              &distortionJacobian);
      }

      // compute the point Jacobian, if requested
      if (pointJacobian) {
          Eigen::Matrix<double, 2, 3>& J = *pointJacobian;
          J(0, 0) = fu * distortionJacobian(0, 0) * rz;
          J(0, 1) = fu * distortionJacobian(0, 1) * rz;
          J(0, 2) = -fu
              * (point[0] * distortionJacobian(0, 0)
                  + point[1] * distortionJacobian(0, 1)) * rz2;
          J(1, 0) = fv * distortionJacobian(1, 0) * rz;
          J(1, 1) = fv * distortionJacobian(1, 1) * rz;
          J(1, 2) = -fv
              * (point[0] * distortionJacobian(1, 0)
                  + point[1] * distortionJacobian(1, 1)) * rz2;
      }

      // scale and offset
      (*imagePoint)[0] = fu * imagePoint2[0] + cu;
      (*imagePoint)[1] = fv * imagePoint2[1] + cv;

      if (!distortionSuccess) {
          return ProjectionStatus::Invalid;
      }
      if (!CameraBase::isInImage(*imagePoint)) {
          return ProjectionStatus::OutsideImage;
      }
      if (CameraBase::isMasked(*imagePoint)) {
          return ProjectionStatus::Masked;
      }
      if (point[2] > 0.0) {
          return ProjectionStatus::Successful;
      }
      else {
          return ProjectionStatus::Behind;
      }
  }

  /// \brief Projects Euclidean points to 2d image points (projection) in a batch.
  ///        Uses projection including distortion models.
  /// @param[in]  points      The points in Euclidean coordinates (one point per column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the projections. See
  ///                         \ref ProjectionStatus for more information.
  inline void projectBatch(
      const Eigen::Matrix3Xd& points, Eigen::Matrix2Xd* imagePoints,
      std::vector<ProjectionStatus>* stati) const override final
  {
      const int numPoints = points.cols();
      for (int i = 0; i < numPoints; ++i) {
          Eigen::Vector3d point = points.col(i);
          Eigen::Vector2d imagePoint;
          ProjectionStatus status = project(point, &imagePoint);
          imagePoints->col(i) = imagePoint;
          if (stati)
              stati->push_back(status);
      }
  }

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point      The point in Homogeneous coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus projectHomogeneous(
      const Eigen::Vector4d& point, Eigen::Vector2d* imagePoint) const override final
  {
      Eigen::Vector3d head = point.head<3>();
      if (point[3] < 0) {
          return project(-head, imagePoint);
      }
      else {
          return project(head, imagePoint);
      }
  }

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus projectHomogeneous(
      const Eigen::Vector4d& point, Eigen::Vector2d* imagePoint,
      Eigen::Matrix<double, 2, 4>* pointJacobian,
      Eigen::Matrix2Xd* intrinsicsJacobian = nullptr) const override final
  {
      Eigen::Vector3d head = point.head<3>();
      Eigen::Matrix<double, 2, 3> pointJacobian3;
      ProjectionStatus status;
      if (point[3] < 0) {
          status = project(-head, imagePoint,
              &pointJacobian3,
              intrinsicsJacobian);
      }
      else {
          status = project(head, imagePoint,
              &pointJacobian3,
              intrinsicsJacobian);
      }
      pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
      pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
      return status;
  }

  /// \brief Projects a point in homogenous coordinates to a 2d image point (projection).
  ///        Uses projection including distortion models.
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  inline ProjectionStatus projectHomogeneousWithExternalParameters(
      const Eigen::Vector4d& point, const Eigen::VectorXd& parameters,
      Eigen::Vector2d* imagePoint,
      Eigen::Matrix<double, 2, 4>* pointJacobian = nullptr,
      Eigen::Matrix2Xd* intrinsicsJacobian = nullptr) const override final
  {
      Eigen::Vector3d head = point.head<3>();
      Eigen::Matrix<double, 2, 3> pointJacobian3;
      ProjectionStatus status;
      if (point[3] < 0) {
          status = projectWithExternalParameters(-head, parameters, imagePoint,
              &pointJacobian3,
              intrinsicsJacobian);
      }
      else {
          status = projectWithExternalParameters(head, parameters, imagePoint,
              &pointJacobian3,
              intrinsicsJacobian);
      }
      pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
      pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
      return status;
  }

  /// \brief Projects points in homogenous coordinates to 2d image points (projection) in a batch.
  ///        Uses projection including distortion models.
  /// @param[in]  points      The points in homogeneous coordinates (one point per column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the projections. See
  ///                         \ref ProjectionStatus for more information.
  inline void projectHomogeneousBatch(
      const Eigen::Matrix4Xd& points, Eigen::Matrix2Xd* imagePoints,
      std::vector<ProjectionStatus>* stati) const override final
  {
      const int numPoints = points.cols();
      for (int i = 0; i < numPoints; ++i) {
          Eigen::Vector4d point = points.col(i);
          Eigen::Vector2d imagePoint;
          ProjectionStatus status = projectHomogeneous(point, &imagePoint);
          imagePoints->col(i) = imagePoint;
          if (stati)
              stati->push_back(status);
      }
  }
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to backproject points
  /// @{

  /// \brief Back-project a 2d image point into Euclidean space (direction vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The Euclidean direction vector.
  /// @return     true on success.
  inline bool backProject(const Eigen::Vector2d& imagePoint,
      Eigen::Vector3d* direction) const override final
  {
      // unscale and center
      Eigen::Vector2d imagePoint2;
      imagePoint2[0] = (imagePoint[0] - cu_) * one_over_fu_;
      imagePoint2[1] = (imagePoint[1] - cv_) * one_over_fv_;

      // undistort
      Eigen::Vector2d undistortedImagePoint;
      bool success = distortion_.undistort(imagePoint2, &undistortedImagePoint);

      // project 1 into z direction
      (*direction)[0] = undistortedImagePoint[0];
      (*direction)[1] = undistortedImagePoint[1];
      (*direction)[2] = 1.0;

      return success;
  }

  /// \brief Back-project a 2d image point into Euclidean space (direction vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The Euclidean direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function  w.r.t. the point.
  /// @return     true on success.
  inline bool backProject(const Eigen::Vector2d& imagePoint,
      Eigen::Vector3d* direction,
      Eigen::Matrix<double, 3, 2>* pointJacobian) const override final
  {
      // unscale and center
      Eigen::Vector2d imagePoint2;
      imagePoint2[0] = (imagePoint[0] - cu_) * one_over_fu_;
      imagePoint2[1] = (imagePoint[1] - cv_) * one_over_fv_;

      // undistort
      Eigen::Vector2d undistortedImagePoint;
      Eigen::Matrix2d pointJacobianUndistortion;
      bool success = distortion_.undistort(imagePoint2, &undistortedImagePoint,
          &pointJacobianUndistortion);

      // project 1 into z direction
      (*direction)[0] = undistortedImagePoint[0];
      (*direction)[1] = undistortedImagePoint[1];
      (*direction)[2] = 1.0;

      // Jacobian w.r.t. imagePoint
      Eigen::Matrix<double, 3, 2> outProjectJacobian =
          Eigen::Matrix<double, 3, 2>::Zero();
      outProjectJacobian(0, 0) = one_over_fu_;
      outProjectJacobian(1, 1) = one_over_fv_;

      (*pointJacobian) = outProjectJacobian * pointJacobianUndistortion;  // chain rule

      return success;
  }

  /// \brief Back-project 2d image points into Euclidean space (direction vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The Euclidean direction vectors (one point per column).
  /// @param[out] success     Success of each of the back-projection
  inline bool backProjectBatch(const Eigen::Matrix2Xd& imagePoints,
      Eigen::Matrix3Xd* directions,
      std::vector<bool>* success) const override final
  {
      const int numPoints = imagePoints.cols();
      for (int i = 0; i < numPoints; ++i) {
          Eigen::Vector2d imagePoint = imagePoints.col(i);
          Eigen::Vector3d point;
          bool suc = backProject(imagePoint, &point);
          if (success)
              success->push_back(suc);
          directions->col(i) = point;
      }
      return true;
  }

  /// \brief Back-project a 2d image point into homogeneous point (direction vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The homogeneous point as direction vector.
  /// @return     true on success.
  inline bool backProjectHomogeneous(const Eigen::Vector2d& imagePoint,
      Eigen::Vector4d* direction) const override final
  {
      Eigen::Vector3d ray;
      bool success = backProject(imagePoint, &ray);
      if(!success || !direction) return false;
      direction->template head<3>() = ray;
      (*direction)[3] = 1.0;  // arbitrary
      return success;
  }

  /// \brief Back-project a 2d image point into homogeneous point (direction vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The homogeneous point as direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function.
  /// @return     true on success.
  inline bool backProjectHomogeneous(
      const Eigen::Vector2d& imagePoint, Eigen::Vector4d* direction,
      Eigen::Matrix<double, 4, 2>* pointJacobian) const override final
  {
      Eigen::Vector3d ray;
      Eigen::Matrix<double, 3, 2> pointJacobian3;
      bool success = backProject(imagePoint, &ray, &pointJacobian3);
      if(!success || !direction || !pointJacobian) return false;
      direction->template head<3>() = ray;
      (*direction)[3] = 1.0;  // arbitrary
      pointJacobian->row(3).setZero();
      pointJacobian->template topLeftCorner<3, 2>() = pointJacobian3;
      return success;
  }

  /// \brief Back-project 2d image points into homogeneous points (direction vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The homogeneous points as direction vectors (one point per column).
  /// @param[out] success     Success of each of the back-projection
  inline bool backProjectHomogeneousBatch(const Eigen::Matrix2Xd& imagePoints,
      Eigen::Matrix4Xd* directions,
      std::vector<bool>* success) const override final
  {
      const int numPoints = imagePoints.cols();
      directions->resize(4,numPoints);
      directions->row(3).setOnes();
      for (int i = 0; i < numPoints; ++i) {
          Eigen::Vector2d imagePoint = imagePoints.col(i);
          Eigen::Vector3d point;
          bool suc = backProject(imagePoint, &point);
          if (success)
              success->push_back(suc);
          directions->template block<3, 1>(0, i) = point;
      }
      return true;
  }
  /// @}

  /// \brief get a test instance
  static std::shared_ptr<CameraBase> createTestObject()
  {
    return std::shared_ptr<CameraBase>(new PinholeCamera(752, 480, 350, 360, 378, 238,
                         distortion_t::testObject()));
  }
  /// \brief get a test instance
  static PinholeCamera testObject()
  {
    return PinholeCamera(752, 480, 350, 360, 378, 238,
                          distortion_t::testObject());
  }

  /// \brief Obtain the projection type
  std::string type() const override final
  {
    return "PinholeCamera<" + distortion_.type() + ">";
  }

  /// \brief Obtain the projection type
  const std::string distortionType() const override final
  {
    return distortion_.type();
  }

 protected:

  /// \brief No default constructor.
  PinholeCamera() = delete;

  distortion_t distortion_;  ///< the distortion to be used

  Eigen::Matrix<double, NumIntrinsics, 1> intrinsics_;  ///< summary of all intrinsics parameters
  double fu_;  ///< focalLengthU
  double fv_;  ///< focalLengthV
  double cu_;  ///< imageCenterU
  double cv_;  ///< imageCenterV
  double one_over_fu_;  ///< 1.0 / fu_
  double one_over_fv_;  ///< 1.0 / fv_
  double fu_over_fv_;  ///< fu_ / fv_

  cv::Mat map_x_fast_; ///< OpenCV undistort fast map x-coordinates
  cv::Mat map_y_fast_; ///< OpenCV undistort fast map x-coordinates

  int undistortedImageWidth_ = 0;  ///< undistortedImageWidth The width in pixels.
  int undistortedImageHeight_ = 0;  ///< undistortedImageHeight The height in pixels.
  double undistortedFocalLengthU_ = 0.0;  ///< The horizontal focal length in pixels.
  double undistortedFocalLengthV_ = 0.0;  ///< The vertical focal length in pixels.
  double undistortedImageCenterU_ = 0.0;  ///< The horizontal centre in pixels.
  double undistortedImageCenterV_ = 0.0;  ///< The vertical centre in pixels.

  cv::Mat rays_;  ///< Rays Rays per pixel.
  cv::Mat imageJacobians_;  ///< Image Jacobian per pixel.

};

// ============= impl ======================

}  // namespace cameras
}  // namespace cgmli

#endif /* INCLUDE_CGMLI_CAMERAS_PINHOLECAMERA_HPP_ */
