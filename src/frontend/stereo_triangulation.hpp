
#ifndef INCLUDE_CGMLI_TRIANGULATION_STEREO_TRIANGULATION_HPP_
#define INCLUDE_CGMLI_TRIANGULATION_STEREO_TRIANGULATION_HPP_

#include <Eigen/Core>

/// \brief cgmli Main namespace of this package.
namespace cgmli {
    /// \brief
    namespace triangulation {

        /**
         * @brief Triangulate the intersection of two rays.
         * @warning The rays e1 and e2 need to be normalized!
         * @param[in]  p1 Camera center position of frame A in coordinate frame A
         * @param[in]  e1 Ray through keypoint of frame A in coordinate frame A.
         * @param[in]  p2 Camera center position of frame B in coordinate frame A.
         * @param[in]  e2 Ray through keypoint of frame B in coordinate frame A.
         * @param[in]  sigma Ray uncertainty.
         * @param[out] isValid Is the triangulation valid.
         * @param[out] isParallel Are the rays parallel?
         * @return Homogeneous coordinates of triangulated point.
         */
        Eigen::Vector4d triangulateFast(const Eigen::Vector3d& p1,
            const Eigen::Vector3d& e1,
            const Eigen::Vector3d& p2,
            const Eigen::Vector3d& e2, double sigma,
            bool& isValid, bool& isParallel);
    }
}

#endif /* INCLUDE_CGMLI_TRIANGULATION_STEREO_TRIANGULATION_HPP_ */
