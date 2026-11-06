
#ifndef CGMLI_PSEUDOINVERSE_HPP_
#define CGMLI_PSEUDOINVERSE_HPP_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <limits>

#include <utils/assert_macros.hpp>


/// \brief cgmli Main namespace of this package.
namespace cgmli {

/// @brief A class to efficiently compute Pseudo-inverses of square PSD matrices.
class PseudoInverse {
public:
  CGMLI_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
/**
 * @brief Pseudo inversion of a symmetric matrix.
 * @warning This uses Eigen-decomposition, it assumes the input is symmetric positive
 * semi-definite (negative Eigenvalues are set to zero).
 * @tparam Derived Matrix type (auto-deducible).
 * @param[in] a Input Matrix
 * @param[out] result Output, i.e. pseudo-inverse.
 * @param[in] epsilon The tolerance.
 * @param[out] rank Optional rank.
 * @return
 */
template<typename Derived>
static bool symm(
    const Eigen::MatrixBase<Derived>&a,
    const Eigen::MatrixBase<Derived>&result, double epsilon =
        std::numeric_limits<typename Derived::Scalar>::epsilon(), int * rank = nullptr) {
    CGMLI_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                          "matrix supplied is not square")

    Eigen::SelfAdjointEigenSolver<Derived> saes(a);

    typename Derived::Scalar tolerance = std::max(
          epsilon, epsilon * a.cols() * saes.eigenvalues().array().maxCoeff());

    const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
        * Eigen::VectorXd(
            (saes.eigenvalues().array() > tolerance).select(
                saes.eigenvalues().array().inverse(), 1.0/tolerance)).asDiagonal()
        * (saes.eigenvectors().transpose());

    if (rank) {
      *rank = 0;
      for (int i = 0; i < a.rows(); ++i) {
        if (saes.eigenvalues()[i] > tolerance)
          (*rank)++;
      }
    }

    return true;
}

/**
 * @brief Pseudo inversion and square root (Cholesky decomposition) of a symmetric matrix.
 * @warning This uses Eigen-decomposition, it assumes the input is symmetric positive
 * semi-definite (negative Eigenvalues are set to zero).
 * @tparam Derived Matrix type (auto-deducible).
 * @param[in] a Input Matrix
 * @param[out] result Output, i.e. the Cholesky decomposition of a pseudo-inverse.
 * @param[in] epsilon The tolerance.
 * @param[out] rank The rank, if of interest.
 * @return
 */
template<typename Derived>
static bool symmSqrt(
    const Eigen::MatrixBase<Derived>&a,
    const Eigen::MatrixBase<Derived>&result, double epsilon =
        std::numeric_limits<typename Derived::Scalar>::epsilon(),
        int* rank = nullptr) {
    CGMLI_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                          "matrix supplied is not square")

    Eigen::SelfAdjointEigenSolver<Derived> saes(a);

    typename Derived::Scalar tolerance = std::max(
          epsilon, epsilon * a.cols() * saes.eigenvalues().array().maxCoeff());

    const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
        * Eigen::VectorXd(
            Eigen::VectorXd(
                (saes.eigenvalues().array() > tolerance).select(
                    saes.eigenvalues().array().inverse(), 1.0/tolerance)).array().sqrt())
            .asDiagonal();

    if (rank) {
      *rank = 0;
      for (int i = 0; i < a.rows(); ++i) {
        if (saes.eigenvalues()[i] > tolerance)
          (*rank)++;
      }
    }

    return true;
}

/**
 * @brief Pseudo inversion and square root (Cholesky decomposition) of a symmetric matrix.
 * @warning This uses Eigen-decomposition, it assumes the input is symmetric positive
 * semi-definite (negative Eigenvalues are set to zero).
 * @tparam Derived Matrix type (auto-deducible).
 * @param[in] a Input Matrix
 * @param[out] result Output, i.e. the Cholesky decomposition of a pseudo-inverse.
 * @param[in] epsilon The tolerance.
 * @param[out] rank The rank, if of interest.
 * @return
 */
template<typename Derived>
static bool symmSqrtU(
    const Eigen::MatrixBase<Derived>&a,
    const Eigen::MatrixBase<Derived>&result, double epsilon =
        std::numeric_limits<typename Derived::Scalar>::epsilon(),
        int* rank = nullptr) {
    CGMLI_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                          "matrix supplied is not square")

    Eigen::SelfAdjointEigenSolver<Derived> saes(a);

    typename Derived::Scalar tolerance = std::max(
          epsilon, epsilon * a.cols() * saes.eigenvalues().array().maxCoeff());

    const_cast<Eigen::MatrixBase<Derived>&>(result) = Eigen::VectorXd(
            Eigen::VectorXd(
                (saes.eigenvalues().array() > tolerance).select(
                    saes.eigenvalues().array().inverse(), 1.0/tolerance)).array().sqrt())
            .asDiagonal() * (saes.eigenvectors().transpose());

    if (rank) {
      *rank = 0;
      for (int i = 0; i < a.rows(); ++i) {
        if (saes.eigenvalues()[i] > tolerance)
          (*rank)++;
      }
    }

    return true;
}
};

}

#endif // CGMLI_PSEUDOINVERSE_HPP_
