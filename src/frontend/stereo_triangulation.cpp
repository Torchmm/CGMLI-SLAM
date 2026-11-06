
#include <frontend/stereo_triangulation.hpp>
#include <kinematics/operators.hpp>
#include <kinematics/Transformation.hpp>

/// \brief cgmli Main namespace of this package.
namespace cgmli {

    /// \brief triangulation A namespace for operations related to triangulation.
    namespace triangulation {

        // Triangulate the intersection of two rays.
        Eigen::Vector4d triangulateFast(const Eigen::Vector3d& p1,
            const Eigen::Vector3d& e1,
            const Eigen::Vector3d& p2,
            const Eigen::Vector3d& e2, double sigma,
            bool& isValid, bool& isParallel) {
            isParallel = false; // This should be the default.
            // But parallel and invalid is not the same. Points at infinity are valid and parallel.
            isValid = true;

            // stolen and adapted from the Kneip toolchain
            // geometric_vision/include/geometric_vision/triangulation/impl/triangulation.hpp
            Eigen::Vector3d t12 = p2 - p1;
            Eigen::Vector2d b;
            b[0] = t12.dot(e1);
            b[1] = t12.dot(e2);
            Eigen::Matrix2d A;
            A(0, 0) = e1.dot(e1);
            A(1, 0) = e1.dot(e2);
            A(0, 1) = -A(1, 0);
            A(1, 1) = -e2.dot(e2);

            bool invertible;
            Eigen::Matrix2d A_inverse;
            A.computeInverseWithCheck(A_inverse, invertible, 1.0e-12);
            Eigen::Vector2d lambda = A_inverse * b;

            if (!invertible) {
                isParallel = true; // let's note this.
                // parallel. that's fine. but A is not invertible. so handle it separately.
                Eigen::Vector4d hp(0, 0, 0, 1);
                isValid = true;
                const Eigen::Vector3d m = p1 + 0.5 * t12;
                const Eigen::Vector3d midpoint = m + 40.0 * std::max(0.01, t12.norm()) * (e1 + e2);
                hp.head<3>() = midpoint;

                // check it
                if (e1.dot((midpoint - p1).normalized()) < cos(2.6 * sigma)) {
                    isValid = false;
                }
                if (e2.dot((midpoint - p2).normalized()) < cos(2.6 * sigma)) {
                    isValid = false;
                }

                return hp;
            }

            if (lambda[0] < 0.01 || lambda[1] < 0.01) {
                isParallel = true;
                // parallel could save this
                Eigen::Vector4d hp(0, 0, 0, 1);
                isValid = true;
                const Eigen::Vector3d m = p1 + 0.5 * t12;
                const Eigen::Vector3d midpoint = m + 40.0 * std::max(0.01, t12.norm()) * (e1 + e2);
                hp.head<3>() = midpoint;

                // check it
                if (e1.dot((midpoint - p1).normalized()) < cos(2.6 * sigma)) {
                    isValid = false;
                }
                if (e2.dot((midpoint - p2).normalized()) < cos(2.6 * sigma)) {
                    isValid = false;
                }

                return hp;
            }

            const Eigen::Vector3d xm = lambda[0] * e1 + p1;
            const Eigen::Vector3d xn = lambda[1] * e2 + p2;
            const Eigen::Vector3d midpoint = (xm + xn) / 2.0;

            // check it
            if (e1.dot((midpoint - p1).normalized()) < cos(2.6 * sigma)) {
                isValid = false;
            }
            if (e2.dot((midpoint - p2).normalized()) < cos(2.6 * sigma)) {
                isValid = false;
            }
            if ((midpoint - p2).normalized().dot((midpoint - p1).normalized()) > cos(6.0 * sigma)) {
                isParallel = true;
            }

            return Eigen::Vector4d(midpoint[0], midpoint[1], midpoint[2], 1.0);
        }

    }

}

