
#ifndef INCLUDE_CGMLI_CAMERAS_RADIALTANGENTIALDISTORTION_HPP_
#define INCLUDE_CGMLI_CAMERAS_RADIALTANGENTIALDISTORTION_HPP_

#include <memory>
#include <Eigen/Core>
#include "cameras/DistortionBase.hpp"

/// \brief cgmli Main namespace of this package.
namespace cgmli {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

/// \brief Radial-tangential distortion model.
class RadialTangentialDistortion : public DistortionBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The default constructor with all zero ki
  inline RadialTangentialDistortion() : k1_(0.0),
      k2_(0.0),
      p1_(0.0),
      p2_(0.0)
  {
      parameters_.setZero();
  }

  /// \brief Constructor initialising ki
  /// @param[in] k1 radial parameter 1
  /// @param[in] k2 radial parameter 2
  /// @param[in] p1 tangential parameter 1
  /// @param[in] p2 tangential parameter 2
  inline RadialTangentialDistortion(double k1, double k2, double p1, double p2) {
      parameters_[0] = k1;
      parameters_[1] = k2;
      parameters_[2] = p1;
      parameters_[3] = p2;
      k1_ = k1;
      k2_ = k2;
      p1_ = p1;
      p2_ = p2;
  }

  //////////////////////////////////////////////////////////////
  /// \name Methods related to generic parameters
  /// @{

  /// \brief set the generic parameters
  /// @param[in] parameters Parameter vector -- length must correspond numDistortionIntrinsics().
  /// @return    True if the requirements were followed.
  inline bool setParameters(const Eigen::VectorXd & parameters) override final {
      if (parameters.cols() != NumDistortionIntrinsics) {
          return false;
      }
      parameters_ = parameters;
      k1_ = parameters[0];
      k2_ = parameters[1];
      p1_ = parameters[2];
      p2_ = parameters[3];
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
    return "RadialTangentialDistortion";
  }

  /// \brief Number of distortion parameters
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
        new RadialTangentialDistortion(-0.16, 0.15, 0.0003, 0.0002));
  }
  /// \brief Unit test support -- create a test distortion object
  static RadialTangentialDistortion testObject()
  {
    return RadialTangentialDistortion(-0.16, 0.15, 0.0003, 0.0002);
  }

  //////////////////////////////////////////////////////////////
  /// \name Distortion functions
  /// @{

  /// \brief Distortion only
  /// @param[in]  pointUndistorted The undistorted normalised (!) image point.
  /// @param[out] pointDistorted   The distorted normalised (!) image point.
  /// @return     True on success (no singularity)
  inline bool distort(const Eigen::Vector2d & pointUndistorted,
                      Eigen::Vector2d * pointDistorted) const override final {
      // just compute the distorted point
      const double u0 = pointUndistorted[0];
      const double u1 = pointUndistorted[1];
      const double mx_u = u0 * u0;
      const double my_u = u1 * u1;
      const double mxy_u = u0 * u1;
      const double rho_u = mx_u + my_u;
      const double rad_dist_u = k1_ * rho_u + k2_ * rho_u * rho_u;
      (*pointDistorted)[0] = u0 + u0 * rad_dist_u + 2.0 * p1_ * mxy_u
          + p2_ * (rho_u + 2.0 * mx_u);
      (*pointDistorted)[1] = u1 + u1 * rad_dist_u + 2.0 * p2_ * mxy_u
          + p1_ * (rho_u + 2.0 * my_u);
      return true;

  }

  /// \brief Distortion and Jacobians.
  /// @param[in]  pointUndistorted  The undistorted normalised (!) image point.
  /// @param[out] pointDistorted    The distorted normalised (!) image point.
  /// @param[out] pointJacobian     The Jacobian w.r.t. changes on the image point.
  /// @param[out] parameterJacobian The Jacobian w.r.t. changes on the intrinsics vector.
  /// @return     True on success (no singularity)
  inline bool distort(const Eigen::Vector2d & pointUndistorted,
                      Eigen::Vector2d * pointDistorted,
                      Eigen::Matrix2d * pointJacobian,
                      Eigen::Matrix2Xd * parameterJacobian = nullptr) const override final {
      // first compute the distorted point
      const double u0 = pointUndistorted[0];
      const double u1 = pointUndistorted[1];
      const double mx_u = u0 * u0;
      const double my_u = u1 * u1;
      const double mxy_u = u0 * u1;
      const double rho_u = mx_u + my_u;
      const double rad_dist_u = k1_ * rho_u + k2_ * rho_u * rho_u;
      (*pointDistorted)[0] = u0 + u0 * rad_dist_u + 2.0 * p1_ * mxy_u
          + p2_ * (rho_u + 2.0 * mx_u);
      (*pointDistorted)[1] = u1 + u1 * rad_dist_u + 2.0 * p2_ * mxy_u
          + p1_ * (rho_u + 2.0 * my_u);

      // next the Jacobian w.r.t. changes on the undistorted point
      Eigen::Matrix2d& J = *pointJacobian;
      J(0, 0) = 1 + rad_dist_u + k1_ * 2.0 * mx_u + k2_ * rho_u * 4 * mx_u
          + 2.0 * p1_ * u1 + 6 * p2_ * u0;
      J(1, 0) = k1_ * 2.0 * u0 * u1 + k2_ * 4 * rho_u * u0 * u1 + p1_ * 2.0 * u0
          + 2.0 * p2_ * u1;
      J(0, 1) = J(1, 0);
      J(1, 1) = 1 + rad_dist_u + k1_ * 2.0 * my_u + k2_ * rho_u * 4 * my_u
          + 6 * p1_ * u1 + 2.0 * p2_ * u0;

      if (parameterJacobian) {
          // the Jacobian w.r.t. intrinsics parameters
          Eigen::Matrix2Xd& J2 = *parameterJacobian;
          J2.resize(2, NumDistortionIntrinsics);
          const double r2 = rho_u;
          const double r4 = r2 * r2;

          //[ u0*(u0^2 + u1^2), u0*(u0^2 + u1^2)^2,       2*u0*u1, 3*u0^2 + u1^2]
          //[ u1*(u0^2 + u1^2), u1*(u0^2 + u1^2)^2, u0^2 + 3*u1^2,       2*u0*u1]

          J2(0, 0) = u0 * r2;
          J2(0, 1) = u0 * r4;
          J2(0, 2) = 2.0 * u0 * u1;
          J2(0, 3) = r2 + 2.0 * u0 * u0;

          J2(1, 0) = u1 * r2;
          J2(1, 1) = u1 * r4;
          J2(1, 2) = r2 + 2.0 * u1 * u1;
          J2(1, 3) = 2.0 * u0 * u1;
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
      const Eigen::Vector2d & pointUndistorted,
      const Eigen::VectorXd & parameters, Eigen::Vector2d * pointDistorted,
      Eigen::Matrix2d * pointJacobian = nullptr,
      Eigen::Matrix2Xd * parameterJacobian = nullptr) const override final {
      const double k1 = parameters[0];
      const double k2 = parameters[1];
      const double p1 = parameters[2];
      const double p2 = parameters[3];
      // first compute the distorted point
      const double u0 = pointUndistorted[0];
      const double u1 = pointUndistorted[1];
      const double mx_u = u0 * u0;
      const double my_u = u1 * u1;
      const double mxy_u = u0 * u1;
      const double rho_u = mx_u + my_u;
      const double rad_dist_u = k1 * rho_u + k2 * rho_u * rho_u;
      (*pointDistorted)[0] = u0 + u0 * rad_dist_u + 2.0 * p1 * mxy_u
          + p2 * (rho_u + 2.0 * mx_u);
      (*pointDistorted)[1] = u1 + u1 * rad_dist_u + 2.0 * p2 * mxy_u
          + p1 * (rho_u + 2.0 * my_u);

      // next the Jacobian w.r.t. changes on the undistorted point
      Eigen::Matrix2d& J = *pointJacobian;
      J(0, 0) = 1 + rad_dist_u + k1 * 2.0 * mx_u + k2 * rho_u * 4 * mx_u
          + 2.0 * p1 * u1 + 6 * p2 * u0;
      J(1, 0) = k1 * 2.0 * u0 * u1 + k2 * 4 * rho_u * u0 * u1 + p1 * 2.0 * u0
          + 2.0 * p2 * u1;
      J(0, 1) = J(1, 0);
      J(1, 1) = 1 + rad_dist_u + k1 * 2.0 * my_u + k2 * rho_u * 4 * my_u
          + 6 * p1 * u1 + 2.0 * p2 * u0;

      if (parameterJacobian) {
          // the Jacobian w.r.t. intrinsics parameters
          Eigen::Matrix2Xd& J2 = *parameterJacobian;
          J2.resize(2, NumDistortionIntrinsics);
          const double r2 = rho_u;
          const double r4 = r2 * r2;

          //[ u0*(u0^2 + u1^2), u0*(u0^2 + u1^2)^2,       2*u0*u1, 3*u0^2 + u1^2]
          //[ u1*(u0^2 + u1^2), u1*(u0^2 + u1^2)^2, u0^2 + 3*u1^2,       2*u0*u1]

          J2(0, 0) = u0 * r2;
          J2(0, 1) = u0 * r4;
          J2(0, 2) = 2.0 * u0 * u1;
          J2(0, 3) = r2 + 2.0 * u0 * u0;

          J2(1, 0) = u1 * r2;
          J2(1, 1) = u1 * r4;
          J2(1, 2) = r2 + 2.0 * u1 * u1;
          J2(1, 3) = 2.0 * u0 * u1;
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
  inline bool undistort(const Eigen::Vector2d & pointDistorted,
                        Eigen::Vector2d * pointUndistorted) const override final {
      // this is expensive: we solve with Gauss-Newton...
      Eigen::Vector2d x_bar = pointDistorted; // initialise at distorted point
      const int n = 5;  // just 5 iterations max.
      Eigen::Matrix2d E;  // error Jacobian

      bool success = false;
      for (int i = 0; i < n; i++) {

          Eigen::Vector2d x_tmp;

          distort(x_bar, &x_tmp, &E);

          Eigen::Vector2d e(pointDistorted - x_tmp);
          Eigen::Matrix2d E2 = (E.transpose() * E);
          Eigen::Vector2d du = E2.inverse() * E.transpose() * e;

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

      if (!success) {
          //std::cout << "undistortion failed" << std::endl;
      }

      return success;
  }

  /// \brief Undistortion only
  /// @param[in]  pointDistorted   The distorted normalised (!) image point.
  /// @param[out] pointUndistorted The undistorted normalised (!) image point.
  /// @param[out] pointJacobian    The Jacobian w.r.t. changes on the image point.
  /// @return     True on success (no singularity)
  inline bool undistort(const Eigen::Vector2d & pointDistorted,
                        Eigen::Vector2d * pointUndistorted,
                        Eigen::Matrix2d * pointJacobian) const override final {
      // this is expensive: we solve with Gauss-Newton...
      Eigen::Vector2d x_bar = pointDistorted; // initialise at distorted point
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
          if (chi2 < 1e-4) {
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
  double p1_;  ///< tangential parameter 1
  double p2_;  ///< tangential parameter 2
};

}  // namespace cameras
}  // namespace cgmli



#endif /* INCLUDE_CGMLI_CAMERAS_RADIALTANGENTIALDISTORTION_HPP_ */
