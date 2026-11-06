#include "LeftSE3Manifold.hpp"
#include <cmath>

namespace cgmli { namespace backend {
namespace {
using M43 = Eigen::Matrix<double, 4, 3>;
M43 quaternionLift(const double* x) {
    // Derivative of expSO3(theta)*q at theta=0, with vector part stored first:
    // Q(q) = 0.5 * [qw*I - skew(qv); -qv^T]. This sign is for LEFT multiplication.
    const Eigen::Map<const Eigen::Quaterniond> q(x);
    M43 j;
    j.topRows<3>() = 0.5 * (q.w() * Eigen::Matrix3d::Identity() - skew(q.vec()));
    j.bottomRows<1>() = -0.5 * q.vec().transpose();
    return j;
}
bool validQuaternion(const double* p) {
    const Eigen::Map<const Eigen::Vector4d> q(p);
    return q.allFinite() && std::abs(q.squaredNorm() - 1.0) < 1e-6;
}
}

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}
Eigen::Quaterniond expSO3(const Eigen::Vector3d& t) {
    // Avoid dividing two nearly-zero quantities; theta is a full rotation angle,
    // while quaternion trigonometric arguments use its half angle.
    const double a = t.norm();
    const double s = a < 1e-8 ? 0.5 - a * a / 48.0 : std::sin(a * 0.5) / a;
    return Eigen::Quaterniond(std::cos(a * 0.5), s*t.x(), s*t.y(), s*t.z()).normalized();
}
Eigen::Vector3d logSO3(Eigen::Quaterniond q) {
    // q and -q encode the same rotation. Select nonnegative scalar part to obtain
    // the shortest logarithm and avoid a sign-dependent displacement in the prior.
    q.normalize();
    if (q.w() < 0) q.coeffs() *= -1.0;
    const double s = q.vec().norm();
    return s < 1e-10 ? Eigen::Vector3d(2.0*q.vec())
        : Eigen::Vector3d((2.0*std::atan2(s, q.w())/s)*q.vec());
}
Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d& t) {
    // J_l = I + (1-cos(a))/a^2 * [theta]x + (a-sin(a))/a^3 * [theta]x^2.
    // Taylor coefficients avoid cancellation near identity.
    const double a2 = t.squaredNorm();
    const Eigen::Matrix3d w = skew(t);
    if (a2 < 1e-10) return Eigen::Matrix3d::Identity() + (0.5-a2/24.0)*w
        + (1.0/6.0-a2/120.0)*w*w;
    const double a = std::sqrt(a2);
    return Eigen::Matrix3d::Identity() + ((1.0-std::cos(a))/a2)*w
        + ((a-std::sin(a))/(a2*a))*w*w;
}

bool LeftSO3Manifold::Plus(const double* x, const double* d, double* out) const {
    if (!validQuaternion(x) || !Eigen::Map<const Eigen::Vector3d>(d).allFinite()) return false;
    Eigen::Map<Eigen::Quaterniond> q(out);
    q = (expSO3(Eigen::Map<const Eigen::Vector3d>(d)) * Eigen::Map<const Eigen::Quaterniond>(x)).normalized();
    return true;
}
bool LeftSO3Manifold::PlusJacobian(const double* x, double* j) const {
    if (!validQuaternion(x)) return false;
    Eigen::Map<Eigen::Matrix<double,4,3,Eigen::RowMajor>> out(j);
    out = quaternionLift(x);
    return true;
}
bool LeftSO3Manifold::Minus(const double* y, const double* x, double* d) const {
    if (!validQuaternion(x) || !validQuaternion(y)) return false;
    Eigen::Map<Eigen::Vector3d> out(d);
    out = logSO3(Eigen::Map<const Eigen::Quaterniond>(y) * Eigen::Map<const Eigen::Quaterniond>(x).conjugate());
    return out.allFinite();
}
bool LeftSO3Manifold::MinusJacobian(const double* x, double* j) const {
    // For a unit quaternion Q^T*Q=I/4, so 4*Q^T is a tangent-consistent left inverse.
    // This ambient extension ignores the unphysical radial quaternion direction.
    if (!validQuaternion(x)) return false;
    Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> out(j);
    out = 4.0*quaternionLift(x).transpose();
    return true;
}

bool LeftSE3Manifold::Plus(const double* x, const double* d, double* out) const {
    // Exp([rho,theta]) = [Exp(theta), J_l(theta)*rho]. Left composition also rotates
    // the CURRENT position: p_new = Exp(theta)*p + J_l(theta)*rho.
    if (!validQuaternion(x+3) || !Eigen::Map<const Eigen::Matrix<double,6,1>>(d).allFinite()) return false;
    const Eigen::Vector3d rho = Eigen::Map<const Eigen::Vector3d>(d);
    const Eigen::Vector3d theta = Eigen::Map<const Eigen::Vector3d>(d+3);
    const Eigen::Quaterniond dq = expSO3(theta);
    const Eigen::Vector3d p = dq*Eigen::Map<const Eigen::Vector3d>(x) + leftJacobianSO3(theta)*rho;
    const Eigen::Quaterniond q = (dq*Eigen::Map<const Eigen::Quaterniond>(x+3)).normalized();
    Eigen::Map<Eigen::Vector3d> po(out); po = p;
    Eigen::Map<Eigen::Quaterniond> qo(out+3); qo = q;
    return p.allFinite();
}
bool LeftSE3Manifold::PlusJacobian(const double* x, double* j) const {
    // At zero increment: J_plus = [I, -skew(p); 0, Q(q)]. The top-right block is
    // the coupling missing from an independent Euclidean-translation product manifold.
    if (!validQuaternion(x+3)) return false;
    Eigen::Map<Eigen::Matrix<double,7,6,Eigen::RowMajor>> out(j);
    out.setZero();
    out.topLeftCorner<3,3>().setIdentity();
    out.block<3,3>(0,3) = -skew(Eigen::Map<const Eigen::Vector3d>(x));
    out.bottomRightCorner<4,3>() = quaternionLift(x+3);
    return out.allFinite();
}
bool LeftSE3Manifold::Minus(const double* y, const double* x, double* d) const {
    // First form the relative transform Y*X^-1, then invert its exponential:
    // theta=Log(R_y*R_x^T), rho=J_l(theta)^-1*(p_y-R_delta*p_x).
    if (!validQuaternion(x+3) || !validQuaternion(y+3)) return false;
    const Eigen::Quaterniond dq = Eigen::Map<const Eigen::Quaterniond>(y+3)
        * Eigen::Map<const Eigen::Quaterniond>(x+3).conjugate();
    const Eigen::Vector3d t = logSO3(dq);
    const Eigen::Vector3d p = Eigen::Map<const Eigen::Vector3d>(y) - dq*Eigen::Map<const Eigen::Vector3d>(x);
    const double a2 = t.squaredNorm();
    const Eigen::Matrix3d w = skew(t);
    const double c = a2 < 1e-10 ? 1.0/12.0 + a2/720.0
        : (1.0 - 0.5*std::sqrt(a2)/std::tan(0.5*std::sqrt(a2)))/a2;
    // The inverse left Jacobian has a stable half-angle formula and an identity series.
    Eigen::Map<Eigen::Vector3d> rho(d); rho = (Eigen::Matrix3d::Identity()-0.5*w+c*w*w)*p;
    Eigen::Map<Eigen::Vector3d> theta(d+3); theta = t;
    return Eigen::Map<Eigen::Matrix<double,6,1>>(d).allFinite();
}
bool LeftSE3Manifold::MinusJacobian(const double* x, double* j) const {
    // A compatible ambient derivative is [I, skew(p)*Qminus; 0, Qminus].
    // Multiplication by J_plus cancels translation/rotation coupling exactly on the tangent.
    if (!validQuaternion(x+3)) return false;
    Eigen::Map<Eigen::Matrix<double,6,7,Eigen::RowMajor>> out(j);
    out.setZero();
    const Eigen::Matrix<double,3,4> qm = 4.0*quaternionLift(x+3).transpose();
    out.topLeftCorner<3,3>().setIdentity();
    out.topRightCorner<3,4>() = skew(Eigen::Map<const Eigen::Vector3d>(x))*qm;
    out.bottomRightCorner<3,4>() = qm;
    return out.allFinite();
}
} }
