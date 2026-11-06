#pragma once

#include <ceres/manifold.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace cgmli { namespace backend {

// Ambient [p, qx, qy, qz, qw], tangent [rho, theta].
// Plus(T,d) = Exp(d) T, Minus(Y,X) = Log(Y X^-1).
// theta is an angle-axis vector in radians; rho is the translational Lie-algebra
// coordinate in the left/output frame. Finite rho is mapped through the SO(3)
// left Jacobian, so this is not independent p+=rho and quaternion multiplication.
// All Jacobian output buffers follow Ceres row-major layout requirements.
class LeftSE3Manifold final : public ::ceres::Manifold {
public:
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }
    /// Apply a left SE(3) increment. Inputs must contain a finite unit quaternion.
    bool Plus(const double* x, const double* delta, double* out) const override;
    /// 7x6 derivative of Plus(x,delta) with respect to delta at delta=0.
    bool PlusJacobian(const double* x, double* jacobian) const override;
    /// Recover [rho,theta] of the principal logarithm of Y*X^-1.
    bool Minus(const double* y, const double* x, double* delta) const override;
    /// 6x7 derivative of Minus(y,x) with respect to ambient y at y=x.
    /// Its product with PlusJacobian(x) is the 6x6 identity on the tangent space.
    bool MinusJacobian(const double* x, double* jacobian) const override;
};

/// Rotation-only counterpart: xyzw storage, 3D angle-axis tangent, Exp(theta)*R.
/// The principal logarithm is locally smooth away from the angle-pi branch cut.
class LeftSO3Manifold final : public ::ceres::Manifold {
public:
    int AmbientSize() const override { return 4; }
    int TangentSize() const override { return 3; }
    bool Plus(const double* x, const double* delta, double* out) const override;
    bool PlusJacobian(const double* x, double* jacobian) const override;
    bool Minus(const double* y, const double* x, double* delta) const override;
    bool MinusJacobian(const double* x, double* jacobian) const override;
};

/// Cross-product matrix: skew(v)*w == v.cross(w).
Eigen::Matrix3d skew(const Eigen::Vector3d& v);
/// SO(3) exponential with a small-angle series for sin(theta/2)/theta.
Eigen::Quaterniond expSO3(const Eigen::Vector3d& theta);
/// Principal angle-axis logarithm; quaternion sign is canonicalized before extraction.
Eigen::Vector3d logSO3(Eigen::Quaterniond q);
/// J_l(theta) used by the SE(3) exponential's translation: p_delta=J_l(theta)*rho.
Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d& theta);

} } // namespace cgmli::backend
