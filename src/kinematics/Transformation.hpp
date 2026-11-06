
#ifndef INCLUDE_CGMLI_TRANSFORMATION_HPP_
#define INCLUDE_CGMLI_TRANSFORMATION_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "kinematics/operators.hpp"
#include <cmath>

/// \brief cgmli Main namespace of this package.
namespace cgmli {

/// \brief kinematics Namespace for kinematics functionality, i.e. TransformationTs and stuff.
namespace kinematics {

/// \brief Implements sin(x)/x for all x in R.
/// @param[in] x The argument of the sinc function.
/// \return The result.
    inline double sinc(double x) {
        if (fabs(x) > 1.0e-6) {
            return sin(x) / x;
        }
        else {
            static const double c_2 = 1.0 / 6.0;
            static const double c_4 = 1.0 / 120.0;
            static const double c_6 = 1.0 / 5040.0;
            const double x_2 = x * x;
            const double x_4 = x_2 * x_2;
            const double x_6 = x_2 * x_2 * x_2;
            return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
        }
    }

/// \brief Implements the exponential map for quaternions.
/// @param[in] dAlpha a axis*angle (minimal) input in tangent space.
/// \return The corresponding Quaternion.
    inline Eigen::Quaterniond deltaQ(const Eigen::Vector3d& dAlpha)
    {
        Eigen::Vector4d dq;
        const double halfnorm = 0.5 * dAlpha.norm();
        dq.template head<3>() = sinc(halfnorm) * 0.5 * dAlpha;
        dq[3] = cos(halfnorm);
        return Eigen::Quaterniond(dq);
    }

/// \brief Right Jacobian, see Forster et al. RSS 2015 eqn. (8)
    inline Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& PhiVec)
    {
        const double Phi = PhiVec.norm();
        Eigen::Matrix3d retMat = Eigen::Matrix3d::Identity();
        const Eigen::Matrix3d Phi_x = cgmli::kinematics::crossMx(PhiVec);
        const Eigen::Matrix3d Phi_x2 = Phi_x * Phi_x;
        if (Phi < 1.0e-4) {
            retMat += -0.5 * Phi_x + 1.0 / 6.0 * Phi_x2;
        }
        else {
            const double Phi2 = Phi * Phi;
            const double Phi3 = Phi2 * Phi;
            retMat += -(1.0 - cos(Phi)) / (Phi2)*Phi_x + (Phi - sin(Phi)) / Phi3 * Phi_x2;
        }
        return retMat;
    }

/// \brief A class that does homogeneous Transformations.
/// This relates a frame A and B: T_AB; it consists of
///   translation r_AB (represented in frame A) and
///   Quaternion q_AB (as an Eigen Quaternion).
/// see also the RSS'13 / IJRR'14 paper or the Thesis.
/// Follows some of the functionality of the SchweizerMesser library by Paul Furgale,
/// but uses Eigen quaternions underneath.
/// \tparam CACHE_C Whether to cache the rotation matrix or not.
/// \warning This means the convention is different to SchweizerMesser
///          and the RSS'13 / IJRR'14 paper / the Thesis
template<bool CACHE_C = true>
class TransformationT
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  friend class TransformationT<!CACHE_C>;

  /// \brief Default constructor: initialises a unit TransformationT.
  TransformationT();

  /// \brief Copy constructor: nothing fancy but takes care
  ///        of not doing bad stuff with internal caching.
  TransformationT(const TransformationT<CACHE_C> & other);

  /// \brief Copy constructor: nothing fancy but takes care
  ///        of not doing bad stuff with internal caching.
  TransformationT(const TransformationT<!CACHE_C> & other);

  /// \brief Move constructor: nothing fancy but takes care
  ///        of not doing bad stuff with internal caching.
  TransformationT(TransformationT<CACHE_C> && other);

  /// \brief Move constructor: nothing fancy but takes care
  ///        of not doing bad stuff with internal caching.
  TransformationT(TransformationT<!CACHE_C> && other);

  /// \brief Construct from a translation and quaternion.
  /// @param[in] r_AB The translation r_AB (represented in frame A).
  /// @param[in] q_AB The Quaternion q_AB (as an Eigen Quaternion).
  TransformationT(const Eigen::Vector3d & r_AB, const Eigen::Quaterniond& q_AB);

  /// \brief Construct from a homogeneous TransformationT matrix.
  /// @param[in] T_AB The homogeneous TransformationT matrix.
  explicit TransformationT(const Eigen::Matrix4d & T_AB);

  /// \brief Trivial destructor.
  ~TransformationT();

  /// \brief Parameter setting, all 7.
  /// \tparam Derived_coeffs Deducible matrix type.
  /// @param[in] coeffs The parameters as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal convention).
  template<typename Derived_coeffs>
  bool setCoeffs(const Eigen::MatrixBase<Derived_coeffs> & coeffs);

  /// \brief Parameter setting, all 7.
  /// \tparam Derived_coeffs Deducible matrix type.
  /// @param[in] parameters The parameters as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal convention).
  template<typename Derived_coeffs>
  bool setParameters(const Eigen::MatrixBase<Derived_coeffs> & parameters)
  {
    return setCoeffs(parameters);
  }

  /// \brief The underlying homogeneous TransformationT matrix.
  Eigen::Matrix4d T() const;

  /// \brief Returns the rotation matrix (cached).
  Eigen::Matrix3d C() const;

  /// \brief Returns the translation vector r_AB (represented in frame A).
  const Eigen::Map<Eigen::Vector3d> & r() const;

  /// \brief Returns the Quaternion q_AB (as an Eigen Quaternion).
  const Eigen::Map<Eigen::Quaterniond> & q() const;

  /// \brief Get the upper 3x4 part of the homogeneous TransformationT matrix T_AB.
  Eigen::Matrix<double, 3, 4> T3x4() const;

  /// \brief The coefficients (parameters) as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal conv.).
  const Eigen::Matrix<double, 7, 1> & coeffs() const
  {
    return parameters_;
  }

  /// \brief The coefficients (parameters) as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal conv.).
  Eigen::Matrix<double, 7, 1> & coeffs()
  {
    if(CACHE_C) throw(std::runtime_error("can't access writable parameters in cached mode."));
    return parameters_;
  }

  /// \brief The parameters (coefficients) as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal conv.).
  const Eigen::Matrix<double, 7, 1> & parameters() const
  {
    return parameters_;
  }

  /// \brief The parameters (coefficients) as [r_AB,q_AB], q_AB as [x,y,z,w] (Eigen internal conv.).
  Eigen::Matrix<double, 7, 1> & parameters()
  {
    if(CACHE_C) throw(std::runtime_error("can't access writable parameters in cached mode."));
    return parameters_;
  }

  /// \brief Get the parameters --- support for ceres.
  /// \warning USE WITH CARE!
  const double* parameterPtr() const
  {
    return &parameters_[0];
  }

  /// \brief Set this to a random TransformationT.
  void setRandom();

  /// \brief Set this to a random TransformationT with bounded rotation and translation.
  /// @param[in] translationMaxMeters Maximum translation [m].
  /// @param[in] rotationMaxRadians Maximum rotation [rad].
  void setRandom(double translationMaxMeters, double rotationMaxRadians);

  /// \brief Set from a homogeneous TransformationT matrix.
  /// @param[in] T_AB The homogeneous TransformationT matrix.
  void set(const Eigen::Matrix4d & T_AB);

  /// \brief Set from a translation and quaternion.
  /// @param[in] r_AB The translation r_AB (represented in frame A).
  /// @param[in] q_AB The Quaternion q_AB (as an Eigen Quaternion).
  void set(const Eigen::Vector3d & r_AB, const Eigen::Quaternion<double>& q_AB);

  /// \brief Set this TransformationT to identity
  void setIdentity();

  /// \brief Get an identity TransformationT
  static TransformationT<CACHE_C> Identity();

  /// \brief Returns a copy of the TransformationT inverted.
  TransformationT<CACHE_C> inverse() const;

  // operator* (group operator)
  /// \brief Multiplication with another TransformationT object.
  /// @param[in] rhs The right-hand side TransformationT for this to be multiplied with.
  TransformationT<CACHE_C> operator*(const TransformationT<CACHE_C> & rhs) const;

  /// \brief Transform a direction as v_A = C_AB*v_B (with rhs = hp_B)..
  /// \warning This only applies the rotation!
  /// @param[in] rhs The right-hand side direction for this to be multiplied with.
  Eigen::Vector3d operator*(const Eigen::Vector3d & rhs) const;

  /// \brief Transform a homogenous point as hp_B = T_AB*hp_B (with rhs = hp_B).
  /// @param[in] rhs The right-hand side direction for this to be multiplied with.
  Eigen::Vector4d operator*(const Eigen::Vector4d & rhs) const;

  /// \brief Assignment -- copy. Takes care of proper caching.
  /// @param[in] rhs The rhs for this to be assigned to.
  TransformationT<CACHE_C>& operator=(const TransformationT<CACHE_C> & rhs);

  /// \brief Assignment -- copy. Takes care of proper caching.
  /// @param[in] rhs The rhs for this to be assigned to.
  TransformationT<CACHE_C>& operator=(const TransformationT<!CACHE_C> & rhs);

  /// \brief Apply a small update with delta being 6x1.
  /// \tparam Derived_delta Deducible matrix type.
  /// @param[in] delta The 6x1 minimal update.
  /// \return True on success.
  template<typename Derived_delta>
  bool oplus(const Eigen::MatrixBase<Derived_delta> & delta);

  /// \brief Apply a small update with delta being 6x1 --
  ///        the Jacobian is a 7 by 6 matrix.
  /// @param[in] delta The 6x1 minimal update.
  /// @param[out] jacobian The output Jacobian.
  /// \return True on success.
  template<typename Derived_delta, typename Derived_jacobian>
  bool oplus(const Eigen::MatrixBase<Derived_delta> & delta,
             const Eigen::MatrixBase<Derived_jacobian> & jacobian);

  /// \brief Get the Jacobian of the oplus operation (a 7 by 6 matrix).
  /// @param[out] jacobian The output Jacobian.
  /// \return True on success.
  template<typename Derived_jacobian>
  bool oplusJacobian(
      const Eigen::MatrixBase<Derived_jacobian> & jacobian) const;

  /// \brief Gets the jacobian dx/dChi,
  ///        i.e. lift the minimal Jacobian to a full one (as needed by ceres).
  // @param[out] jacobian The output lift Jacobian (6 by 7 matrix).
  /// \return True on success.
  template<typename Derived_jacobian>
  bool liftJacobian(const Eigen::MatrixBase<Derived_jacobian> & jacobian) const;

 protected:
  /// \brief Update the caching of the rotation matrix.
  void updateC();
  Eigen::Matrix<double, 7, 1> parameters_;  ///< Concatenated parameters [r;q].
  Eigen::Map<Eigen::Vector3d> r_;  ///< Translation {_A}r_{B}.
  Eigen::Map<Eigen::Quaterniond> q_;  ///< Quaternion q_{AB}.
  Eigen::Matrix3d C_; ///< The cached DCM C_{AB}.
};

/// \brief Transformation class.
typedef TransformationT<true> Transformation;

/// \brief Transformation class that doesn't cache rotation matrix.
typedef TransformationT<false> TransformationCacheless;


template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(const TransformationT<CACHE_C>& other)
    : parameters_(other.parameters_),
    r_(&parameters_[0]),
    q_(&parameters_[3]),
    C_(other.C_) {
}
template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(const TransformationT<!CACHE_C>& other)
    : parameters_(other.parameters_),
    r_(&parameters_[0]),
    q_(&parameters_[3]),
    C_(other.C_) {
    if (CACHE_C) C_ = q_.toRotationMatrix();
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(TransformationT<CACHE_C>&& other)
    : parameters_(std::move(other.parameters_)),
    r_(&parameters_[0]),
    q_(&parameters_[3]),
    C_(std::move(other.C_)) {
}
template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(TransformationT<!CACHE_C>&& other)
    : parameters_(std::move(other.parameters_)),
    r_(&parameters_[0]),
    q_(&parameters_[3]),
    C_(std::move(other.C_)) {
    if (CACHE_C) C_ = q_.toRotationMatrix();
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT()
    : r_(&parameters_[0]),
    q_(&parameters_[3]),
    C_(Eigen::Matrix3d::Identity()) {
    r_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    q_ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(const Eigen::Vector3d& r_AB,
    const Eigen::Quaterniond& q_AB)
    : r_(&parameters_[0]),
    q_(&parameters_[3]) {
    r_ = r_AB;
    q_ = q_AB.normalized();
    if (CACHE_C) updateC();
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(const Eigen::Matrix4d& T_AB)
    : r_(&parameters_[0]),
    q_(&parameters_[3]),
    C_(T_AB.topLeftCorner<3, 3>()) {
    r_ = (T_AB.topRightCorner<3, 1>());
    q_ = (T_AB.topLeftCorner<3, 3>());
    assert(fabs(T_AB(3, 0)) < 1.0e-12);
    assert(fabs(T_AB(3, 1)) < 1.0e-12);
    assert(fabs(T_AB(3, 2)) < 1.0e-12);
    assert(fabs(T_AB(3, 3) - 1.0) < 1.0e-12);
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::~TransformationT() {

}

template<bool CACHE_C>
template<typename Derived_coeffs>
inline bool TransformationT<CACHE_C>::setCoeffs(
    const Eigen::MatrixBase<Derived_coeffs>& coeffs) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_coeffs, 7);
    parameters_ = coeffs;
    if (CACHE_C) updateC();
    return true;
}

// The underlying TransformationT
template<bool CACHE_C>
inline Eigen::Matrix4d TransformationT<CACHE_C>::T() const {
    Eigen::Matrix4d T_ret;
    T_ret.topLeftCorner<3, 3>() = CACHE_C ? C_ : q_.toRotationMatrix();
    T_ret.topRightCorner<3, 1>() = r_;
    T_ret.bottomLeftCorner<1, 3>().setZero();
    T_ret(3, 3) = 1.0;
    return T_ret;
}

// return the rotation matrix
template<bool CACHE_C>
inline Eigen::Matrix3d TransformationT<CACHE_C>::C() const {
    if (!CACHE_C) return q_.toRotationMatrix();
    return C_;
}

// return the translation vector
template<bool CACHE_C>
inline const Eigen::Map<Eigen::Vector3d>& TransformationT<CACHE_C>::r() const {
    return r_;
}

template<bool CACHE_C>
inline const Eigen::Map<Eigen::Quaterniond>& TransformationT<CACHE_C>::q() const {
    return q_;
}

template<bool CACHE_C>
inline Eigen::Matrix<double, 3, 4> TransformationT<CACHE_C>::T3x4() const {
    Eigen::Matrix<double, 3, 4> T3x4_ret;
    T3x4_ret.topLeftCorner<3, 3>() = CACHE_C ? C_ : q_.toRotationMatrix();
    T3x4_ret.topRightCorner<3, 1>() = r_;
    return T3x4_ret;
}

// Return a copy of the TransformationT inverted.
template<bool CACHE_C>
inline TransformationT<CACHE_C> TransformationT<CACHE_C>::inverse() const {
    return CACHE_C ? TransformationT(-(C_.transpose() * r_), q_.inverse()) :
        TransformationT(-(q_.toRotationMatrix().transpose() * r_), q_.inverse());
}

// Set this to a random TransformationT.
template<bool CACHE_C>
inline void TransformationT<CACHE_C>::setRandom() {
    setRandom(1.0, M_PI);
}
// Set this to a random TransformationT with bounded rotation and translation.
template<bool CACHE_C>
inline void TransformationT<CACHE_C>::setRandom(double translationMaxMeters,
    double rotationMaxRadians) {
    // Create a random unit-length axis.
    Eigen::Vector3d axis = rotationMaxRadians * Eigen::Vector3d::Random();
    // Create a random rotation angle in radians.
    Eigen::Vector3d r = translationMaxMeters * Eigen::Vector3d::Random();
    r_ = r;
    q_ = Eigen::AngleAxisd(axis.norm(), axis.normalized());
    if (CACHE_C) updateC();
}

// Setters
template<bool CACHE_C>
inline void TransformationT<CACHE_C>::set(const Eigen::Matrix4d& T_AB) {
    r_ = (T_AB.topRightCorner<3, 1>());
    q_ = (T_AB.topLeftCorner<3, 3>());
    if (CACHE_C) updateC();
}

template<bool CACHE_C>
inline void TransformationT<CACHE_C>::set(const Eigen::Vector3d& r_AB,
    const Eigen::Quaternion<double>& q_AB) {
    r_ = r_AB;
    q_ = q_AB.normalized();
    if (CACHE_C) updateC();
}

// Set this TransformationT to identity
template<bool CACHE_C>
inline void TransformationT<CACHE_C>::setIdentity() {
    q_.setIdentity();
    r_.setZero();
    if (CACHE_C) C_.setIdentity();
}

template<bool CACHE_C>
inline TransformationT<CACHE_C> TransformationT<CACHE_C>::Identity() {
    return TransformationT();
}

// operator*
template<bool CACHE_C>
inline TransformationT<CACHE_C> TransformationT<CACHE_C>::operator*(
    const TransformationT<CACHE_C>& rhs) const {
    return TransformationT((CACHE_C ? C_ : q_.toRotationMatrix()) * rhs.r_ + r_, q_ * rhs.q_);
}
template<bool CACHE_C>
inline Eigen::Vector3d TransformationT<CACHE_C>::operator*(
    const Eigen::Vector3d& rhs) const {
    return (CACHE_C ? C_ : q_.toRotationMatrix()) * rhs;
}
template<bool CACHE_C>
inline Eigen::Vector4d TransformationT<CACHE_C>::operator*(
    const Eigen::Vector4d& rhs) const {
    const double s = rhs[3];
    Eigen::Vector4d retVec;
    retVec.head<3>() = (CACHE_C ? C_ : q_.toRotationMatrix()) * rhs.head<3>() + r_ * s;
    retVec[3] = s;
    return retVec;
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>& TransformationT<CACHE_C>::operator=(const TransformationT<CACHE_C>& rhs) {
    parameters_ = rhs.parameters_;
    if (CACHE_C) C_ = rhs.C_;
    r_ = Eigen::Map<Eigen::Vector3d>(&parameters_[0]);
    q_ = Eigen::Map<Eigen::Quaterniond>(&parameters_[3]);
    return *this;
}
template<bool CACHE_C>
inline TransformationT<CACHE_C>& TransformationT<CACHE_C>::operator=(const TransformationT<!CACHE_C>& rhs) {
    parameters_ = rhs.parameters_;
    if (CACHE_C) C_ = q_.toRotationMatrix();
    r_ = Eigen::Map<Eigen::Vector3d>(&parameters_[0]);
    q_ = Eigen::Map<Eigen::Quaterniond>(&parameters_[3]);
    return *this;
}

template<bool CACHE_C>
inline void TransformationT<CACHE_C>::updateC() {
    if (CACHE_C) C_ = q_.toRotationMatrix();
    if (!CACHE_C) throw(std::runtime_error("bug"));
}

// apply small update:
template<bool CACHE_C>
template<typename Derived_delta>
inline bool TransformationT<CACHE_C>::oplus(
    const Eigen::MatrixBase<Derived_delta>& delta) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_delta, 6);
    r_ += delta.template head<3>();
    const Eigen::Vector3d dalpha = delta.template tail<3>();
    Eigen::Vector4d dq;
    const double halfnorm = 0.5 * dalpha.norm();
    dq.template head<3>() = sinc(halfnorm) * 0.5 * dalpha;
    dq[3] = cos(halfnorm);
    q_ = (Eigen::Quaterniond(dq) * q_);
    q_.normalize();
    if (CACHE_C) updateC();
    return true;
}

template<bool CACHE_C>
template<typename Derived_delta, typename Derived_jacobian>
inline bool TransformationT<CACHE_C>::oplus(
    const Eigen::MatrixBase<Derived_delta>& delta,
    const Eigen::MatrixBase<Derived_jacobian>& jacobian) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_delta, 6);
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 7, 6);
    if (!oplus(delta)) {
        return false;
    }
    return oplusJacobian(jacobian);
}

template<bool CACHE_C>
template<typename Derived_jacobian>
inline bool TransformationT<CACHE_C>::oplusJacobian(
    const Eigen::MatrixBase<Derived_jacobian>& jacobian) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 7, 6);
    Eigen::Matrix<double, 4, 3> S = Eigen::Matrix<double, 4, 3>::Zero();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).setZero();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian)
        .template topLeftCorner<3, 3>().setIdentity();
    S(0, 0) = 0.5;
    S(1, 1) = 0.5;
    S(2, 2) = 0.5;
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian)
        .template bottomRightCorner<4, 3>() = cgmli::kinematics::oplus(q_) * S;
    return true;
}

template<bool CACHE_C>
template <typename Derived_jacobian>
inline bool TransformationT<CACHE_C>::liftJacobian(
    const Eigen::MatrixBase<Derived_jacobian>& jacobian) const
{
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 6, 7);
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).setZero();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).template topLeftCorner<3, 3>()
        = Eigen::Matrix3d::Identity();
    const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).template bottomRightCorner<3, 4>()
        = 2 * cgmli::kinematics::oplus(q_.inverse()).template topLeftCorner<3, 4>();
    return true;
}


}  // namespace kinematics
}  // namespace cgmli


#endif /* INCLUDE_CGMLI_TRANSFORMATION_HPP_ */
