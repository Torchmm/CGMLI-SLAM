
#ifndef INCLUDE_CGMLI_CAMERAS_EQUIDISTANTDISTORTION_HPP_
#define INCLUDE_CGMLI_CAMERAS_EQUIDISTANTDISTORTION_HPP_

#include <memory>
#include <Eigen/Core>
#include "cameras/DistortionBase.hpp"

/// \brief cgmli Main namespace of this package.
namespace cgmli {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

/// \class EquidistantDistortion
/// \brief Implements equidistant distortion.
class EquidistantDistortion : public DistortionBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The default constructor with all zero ki
      inline EquidistantDistortion()
  {
      k1_ = (0.0);
      k2_ = (0.0);
      k3_ = (0.0);
      k4_ = (0.0);
      parameters_.setZero();
  }

  /// \brief Constructor initialising ki
  /// @param[in] k1 equidistant parameter 1
  /// @param[in] k2 equidistant parameter 2
  /// @param[in] k3 equidistant parameter 3
  /// @param[in] k4 equidistant parameter 4
  inline EquidistantDistortion(double k1, double k2, double k3, double k4)
  {
      parameters_[0] = k1;
      parameters_[1] = k2;
      parameters_[2] = k3;
      parameters_[3] = k4;
      k1_ = k1;
      k2_ = k2;
      k3_ = k3;
      k4_ = k4;
  }
  //////////////////////////////////////////////////////////////
  /// \name Methods related to generic parameters
  /// @{

  /// \brief set the generic parameters
  /// @param[in] parameters Parameter vector -- length must correspond numDistortionIntrinsics().
  /// @return    True if the requirements were followed.
  inline bool setParameters(const Eigen::VectorXd& parameters) override final
  {
      if (parameters.cols() != NumDistortionIntrinsics) {
          return false;
      }
      parameters_ = parameters;
      k1_ = parameters[0];
      k2_ = parameters[1];
      k3_ = parameters[2];
      k4_ = parameters[3];
      return true;
  }

  /// \brief Obtain the generic parameters.
  inline bool getParameters(Eigen::VectorXd & parameters) const override final
  {
    parameters = parameters_;
    return true;
  }
  /// \brief The class type.
  inline std::string type() const override final
  {
    return "EquidistantDistortion";
  }

  /// \brief Number of distortion parameters.
  inline int numDistortionIntrinsics() const override final
  {
    return NumDistortionIntrinsics;
  }

  static const int NumDistortionIntrinsics = 4;  ///< The Number of distortion parameters.
  /// @}

  /// \brief Unit test support -- create a test distortion object
  static std::shared_ptr<DistortionBase> createTestObject()
  {
    return std::shared_ptr<DistortionBase>(
        new EquidistantDistortion(-0.21, 0.14, 0.0006, 0.0003));
  }
  /// \brief Unit test support -- create a test distortion object
  static EquidistantDistortion testObject()
  {
    return EquidistantDistortion(-0.21, 0.14, 0.0006, 0.0003);
  }

  //////////////////////////////////////////////////////////////
  /// \name Distortion functions
  /// @{

  /// \brief Distortion only
  /// @param[in]  pointUndistorted The undistorted normalised (!) image point.
  /// @param[out] pointDistorted   The distorted normalised (!) image point.
  /// @return     True on success (no singularity)
  inline bool distort(const Eigen::Vector2d& pointUndistorted,
      Eigen::Vector2d* pointDistorted) const override final
  {
      // distortion only:
      const double u0 = pointUndistorted[0];
      const double u1 = pointUndistorted[1];
      const double r = sqrt(u0 * u0 + u1 * u1);
      const double theta = atan(r);
      const double theta2 = theta * theta;
      const double theta4 = theta2 * theta2;
      const double theta6 = theta4 * theta2;
      const double theta8 = theta4 * theta4;
      const double thetad = theta
          * (1.0 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);

      const double scaling = (r > 1e-8) ? thetad / r : 1.0;
      (*pointDistorted)[0] = scaling * u0;
      (*pointDistorted)[1] = scaling * u1;
      return true;
  }

  /// \brief Distortion and Jacobians.
  /// @param[in]  pointUndistorted  The undistorted normalised (!) image point.
  /// @param[out] pointDistorted    The distorted normalised (!) image point.
  /// @param[out] pointJacobian     The Jacobian w.r.t. changes on the image point.
  /// @param[out] parameterJacobian The Jacobian w.r.t. changes on the intrinsics vector.
  /// @return     True on success (no singularity)
  inline bool distort(const Eigen::Vector2d& pointUndistorted,
      Eigen::Vector2d* pointDistorted,
      Eigen::Matrix2d* pointJacobian,
      Eigen::Matrix2Xd* parameterJacobian = nullptr) const override final
  {
      // distortion first:
      const double u0 = pointUndistorted[0];
      const double u1 = pointUndistorted[1];
      const double r = sqrt(u0 * u0 + u1 * u1);
      const double theta = atan(r);
      const double theta2 = theta * theta;
      const double theta4 = theta2 * theta2;
      const double theta6 = theta4 * theta2;
      const double theta8 = theta4 * theta4;
      const double thetad = theta
          * (1.0 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);

      const double scaling = (r > 1e-8) ? thetad / r : 1.0;
      (*pointDistorted)[0] = scaling * u0;
      (*pointDistorted)[1] = scaling * u1;

      Eigen::Matrix2d& J = *pointJacobian;
      if (r > 1e-8) {
          // mostly matlab generated...
          double t2;
          double t3;
          double t4;
          double t6;
          double t7;
          double t8;
          double t9;
          double t11;
          double t17;
          double t18;
          double t19;
          double t20;
          double t25;

          t2 = u0 * u0;
          t3 = u1 * u1;
          t4 = t2 + t3;
          t6 = atan(sqrt(t4));
          t7 = t6 * t6;
          t8 = 1.0 / sqrt(t4);
          t9 = t7 * t7;
          t11 = 1.0 / ((t2 + t3) + 1.0);
          t17 = (((k1_ * t7 + k2_ * t9) + k3_ * t7 * t9) + k4_ * (t9 * t9)) + 1.0;
          t18 = 1.0 / t4;
          t19 = 1.0 / sqrt(t4 * t4 * t4);
          t20 = t6 * t8 * t17;
          t25 = ((k2_ * t6 * t7 * t8 * t11 * u1 * 4.0
              + k3_ * t6 * t8 * t9 * t11 * u1 * 6.0)
              + k4_ * t6 * t7 * t8 * t9 * t11 * u1 * 8.0)
              + k1_ * t6 * t8 * t11 * u1 * 2.0;
          t4 = ((k2_ * t6 * t7 * t8 * t11 * u0 * 4.0
              + k3_ * t6 * t8 * t9 * t11 * u0 * 6.0)
              + k4_ * t6 * t7 * t8 * t9 * t11 * u0 * 8.0)
              + k1_ * t6 * t8 * t11 * u0 * 2.0;
          t7 = t11 * t17 * t18 * u0 * u1;
          J(0, 1) = (t7 + t6 * t8 * t25 * u0) - t6 * t17 * t19 * u0 * u1;
          J(1, 1) = ((t20 - t3 * t6 * t17 * t19) + t3 * t11 * t17 * t18)
              + t6 * t8 * t25 * u1;
          J(0, 0) = ((t20 - t2 * t6 * t17 * t19) + t2 * t11 * t17 * t18)
              + t6 * t8 * t4 * u0;
          J(1, 0) = (t7 + t6 * t8 * t4 * u1) - t6 * t17 * t19 * u0 * u1;

          if (parameterJacobian) {
              Eigen::Matrix2Xd& Ji = *parameterJacobian;
              Ji.resize(2, NumDistortionIntrinsics);
              // mostly matlab generated...
              double t6;
              double t2;
              double t3;
              double t8;
              double t10;

              t6 = u0 * u0 + u1 * u1;
              t2 = atan(sqrt(t6));
              t3 = t2 * t2;
              t8 = t3 * t3;
              t6 = 1.0 / sqrt(t6);
              t10 = t8 * t8;
              Ji(0, 0) = t2 * t3 * t6 * u0;
              Ji(1, 0) = t2 * t3 * t6 * u1;
              Ji(0, 1) = t2 * t8 * t6 * u0;
              Ji(1, 1) = t2 * t8 * t6 * u1;
              Ji(0, 2) = t2 * t3 * t8 * t6 * u0;
              Ji(1, 2) = t2 * t3 * t8 * t6 * u1;
              Ji(0, 3) = t2 * t6 * t10 * u0;
              Ji(1, 3) = t2 * t6 * t10 * u1;

          }
      }
      else {
          // handle limit case for [u0,u1]->0
          if (parameterJacobian) {
              parameterJacobian->resize(2, NumDistortionIntrinsics);
              parameterJacobian->setZero();
          }
          J.setIdentity();
      }

      return true;
  }

  /// \brief Distortion and Jacobians using external distortion intrinsics parameters.
  /// @param[in]  pointUndistorted  The undistorted normalised (!) image point.
  /// @param[in]  parameters        The distortion intrinsics vector.
  /// @param[out] pointDistorted    The distorted normalised (!) image point.
  /// @param[out] pointJacobian     The Jacobian w.r.t. changes on the image point.
  /// @param[out] parameterJacobian The Jacobian w.r.t. changes on the intrinsics vector.
  /// @return     True on success (no singularity)
  inline bool distortWithExternalParameters(
      const Eigen::Vector2d& pointUndistorted,
      const Eigen::VectorXd& parameters, Eigen::Vector2d* pointDistorted,
      Eigen::Matrix2d* pointJacobian = nullptr,
      Eigen::Matrix2Xd* parameterJacobian = nullptr) const override final
  {
      // decompose parameters
      const double k1 = parameters[0];
      const double k2 = parameters[1];
      const double k3 = parameters[2];
      const double k4 = parameters[3];

      // distortion first:
      const double u0 = pointUndistorted[0];
      const double u1 = pointUndistorted[1];
      const double r = sqrt(u0 * u0 + u1 * u1);
      const double theta = atan(r);
      const double theta2 = theta * theta;
      const double theta4 = theta2 * theta2;
      const double theta6 = theta4 * theta2;
      const double theta8 = theta4 * theta4;
      const double thetad = theta
          * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

      const double scaling = (r > 1e-8) ? thetad / r : 1.0;
      (*pointDistorted)[0] = scaling * u0;
      (*pointDistorted)[1] = scaling * u1;

      Eigen::Matrix2d& J = *pointJacobian;
      if (r > 1e-8) {
          // mostly matlab generated...
          double t2;
          double t3;
          double t4;
          double t6;
          double t7;
          double t8;
          double t9;
          double t11;
          double t17;
          double t18;
          double t19;
          double t20;
          double t25;

          t2 = u0 * u0;
          t3 = u1 * u1;
          t4 = t2 + t3;
          t6 = atan(sqrt(t4));
          t7 = t6 * t6;
          t8 = 1.0 / sqrt(t4);
          t9 = t7 * t7;
          t11 = 1.0 / ((t2 + t3) + 1.0);
          t17 = (((k1 * t7 + k2 * t9) + k3 * t7 * t9) + k4 * (t9 * t9)) + 1.0;
          t18 = 1.0 / t4;
          t19 = 1.0 / sqrt(t4 * t4 * t4);
          t20 = t6 * t8 * t17;
          t25 = ((k2 * t6 * t7 * t8 * t11 * u1 * 4.0
              + k3 * t6 * t8 * t9 * t11 * u1 * 6.0)
              + k4 * t6 * t7 * t8 * t9 * t11 * u1 * 8.0)
              + k1 * t6 * t8 * t11 * u1 * 2.0;
          t4 = ((k2 * t6 * t7 * t8 * t11 * u0 * 4.0
              + k3 * t6 * t8 * t9 * t11 * u0 * 6.0)
              + k4 * t6 * t7 * t8 * t9 * t11 * u0 * 8.0)
              + k1 * t6 * t8 * t11 * u0 * 2.0;
          t7 = t11 * t17 * t18 * u0 * u1;
          J(0, 1) = (t7 + t6 * t8 * t25 * u0) - t6 * t17 * t19 * u0 * u1;
          J(1, 1) = ((t20 - t3 * t6 * t17 * t19) + t3 * t11 * t17 * t18)
              + t6 * t8 * t25 * u1;
          J(0, 0) = ((t20 - t2 * t6 * t17 * t19) + t2 * t11 * t17 * t18)
              + t6 * t8 * t4 * u0;
          J(1, 0) = (t7 + t6 * t8 * t4 * u1) - t6 * t17 * t19 * u0 * u1;

          if (parameterJacobian) {
              Eigen::Matrix2Xd& Ji = *parameterJacobian;
              Ji.resize(2, NumDistortionIntrinsics);
              // mostly matlab generated...
              double t6;
              double t2;
              double t3;
              double t8;
              double t10;

              t6 = u0 * u0 + u1 * u1;
              t2 = atan(sqrt(t6));
              t3 = t2 * t2;
              t8 = t3 * t3;
              t6 = 1.0 / sqrt(t6);
              t10 = t8 * t8;
              Ji(0, 0) = t2 * t3 * t6 * u0;
              Ji(1, 0) = t2 * t3 * t6 * u1;
              Ji(0, 1) = t2 * t8 * t6 * u0;
              Ji(1, 1) = t2 * t8 * t6 * u1;
              Ji(0, 2) = t2 * t3 * t8 * t6 * u0;
              Ji(1, 2) = t2 * t3 * t8 * t6 * u1;
              Ji(0, 3) = t2 * t6 * t10 * u0;
              Ji(1, 3) = t2 * t6 * t10 * u1;

          }
      }
      else {
          // handle limit case for [u0,u1]->0
          if (parameterJacobian) {
              parameterJacobian->resize(2, NumDistortionIntrinsics);
              parameterJacobian->setZero();
          }
          J.setIdentity();
      }

      return true;
  }
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Undistortion functions
  /// @{

  /// \brief Undistortion only
  /// @param[in]  pointDistorted   The distorted normalised (!) image point.
  /// @param[out] pointUndistorted The undistorted normalised (!) image point.
  /// @return     True on success (no singularity)
  inline bool undistort(const Eigen::Vector2d& pointDistorted,
      Eigen::Vector2d* pointUndistorted) const override final
  {
      // this is expensive: we solve with Gauss-Newton...
      Eigen::Vector2d x_bar = pointDistorted;  // initialise at distorted point
      const int n = 20;  // just 5 iterations max.
      Eigen::Matrix2d E;  // error Jacobian

      bool success = false;
      for (int i = 0; i < n; i++) {

          Eigen::Vector2d x_tmp;

          distort(x_bar, &x_tmp, &E);

          Eigen::Vector2d e(pointDistorted - x_tmp);
          Eigen::Vector2d du = (E.transpose() * E).inverse() * E.transpose() * e;

          x_bar += du;

          const double chi2 = e.dot(e);
          if (chi2 < 1e-6) {
              success = true;
          }
          if (chi2 < 1e-15) {
              success = true;
              break;
          }

      }
      *pointUndistorted = x_bar;

      return success;
  }

  /// \brief Undistortion only
    /// @param[in]  pointDistorted   The distorted normalised (!) image point.
    /// @param[out] pointUndistorted The undistorted normalised (!) image point.
    /// @param[out] pointJacobian    The Jacobian w.r.t. changes on the image point.
    /// @return     True on success (no singularity)
  inline bool undistort(const Eigen::Vector2d& pointDistorted,
      Eigen::Vector2d* pointUndistorted,
      Eigen::Matrix2d* pointJacobian) const override final
  {
      // this is expensive: we solve with Gauss-Newton...
      Eigen::Vector2d x_bar = pointDistorted;  // initialise at distorted point
      const int n = 5;  // just 5 iterations max.
      Eigen::Matrix2d E;  // error Jacobian

      bool success = false;
      for (int i = 0; i < n; i++) {

          Eigen::Vector2d x_tmp;

          distort(x_bar, &x_tmp, &E);

          Eigen::Vector2d e(pointDistorted - x_tmp);
          Eigen::Vector2d dx = (E.transpose() * E).inverse() * E.transpose() * e;

          x_bar += dx;

          const double chi2 = e.dot(e);
          if (chi2 < 1e-2) {
              success = true;
          }
          if (chi2 < 1e-15) {
              success = true;
              break;
          }

      }
      *pointUndistorted = x_bar;

      // the Jacobian of the inverse map is simply the inverse Jacobian.
      *pointJacobian = E.inverse();

      return success;
  }
  /// @}

 protected:
  Eigen::Matrix<double, NumDistortionIntrinsics, 1> parameters_;  ///< all distortion parameters

  double k1_;  ///< radial parameter 1
  double k2_;  ///< radial parameter 2
  double k3_;  ///< tangential parameter 1
  double k4_;  ///< tangential parameter 2
};

}  // namespace cameras
}  // namespace cgmli


#endif /* INCLUDE_CGMLI_CAMERAS_EQUIDISTANTDISTORTION_HPP_ */
