
#ifndef INCLUDE_CGMLI_CAMERAS_EUCMCAMERA_HPP_
#define INCLUDE_CGMLI_CAMERAS_EUCMCAMERA_HPP_


#include <vector>
#include <memory>
#include <cmath>
#include <stdint.h>
#include <Eigen/Core>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#include "cameras/CameraBase.hpp"
#include "cameras/DistortionBase.hpp"
#include "logging/logger.hpp"
/// \brief cgmli Main namespace of this package.
namespace cgmli {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {


/// \class EucmCamera

class EucmCamera : public CameraBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// \brief Constructor
    /// @param[in] imageWidth The width in pixels.
    /// @param[in] imageHeight The height in pixels.
    /// @param[in] focalLengthU The horizontal focal length in pixels.
    /// @param[in] focalLengthV The vertical focal length in pixels.
    /// @param[in] imageCenterU The horizontal centre in pixels.
    /// @param[in] imageCenterV The vertical centre in pixels.
    /// @param[in] alpha Parameter alpha in [0,1] of the EUCM model
    /// @param[in] beta Parameter beta > 0 of the EUCM model
    /// @param[in] id Assign a generic ID, if desired.
        inline EucmCamera(int imageWidth, int imageHeight, double focalLengthU,
            double focalLengthV, double imageCenterU, double imageCenterV,
            double alpha, double beta, uint64_t id = -1) : CameraBase(imageWidth, imageHeight, id),
        fu_(focalLengthU),
        fv_(focalLengthV),
        cu_(imageCenterU),
        cv_(imageCenterV),
        alpha_(alpha),
        beta_(beta)
    {

        intrinsics_[0] = fu_;  //< focalLengthU
        intrinsics_[1] = fv_;  //< focalLengthV
        intrinsics_[2] = cu_;  //< imageCenterU
        intrinsics_[3] = cv_;  //< imageCenterV
        intrinsics_[4] = alpha_;  //< alpha
        intrinsics_[5] = beta_;  //< beta
        one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
        one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
        fu_over_fv_ = fu_ / fv_;  //< fu_ / fv_
    }

    /// \brief Destructor.
    virtual ~EucmCamera()
    {
    }

    static const int NumProjectionIntrinsics = 6;  ///< optimisable projection intrinsics
    static const int NumIntrinsics = NumProjectionIntrinsics; ///< total number of intrinsics, distortion is part of projection in EUCM

    /// \brief Get the focal length along the u-dimension.
    /// \return The horizontal focal length in pixels.
    virtual double focalLengthU() const
    {
      return fu_;
    }

    /// \brief Get the focal length along the v-dimension.
    /// \return The vertical focal length in pixels.
    virtual double focalLengthV() const
    {
      return fv_;
    }

    /// \brief Get the image centre along the u-dimension.
    /// \return The horizontal centre in pixels.
    virtual double imageCenterU() const
    {
      return cu_;
    }

    /// \brief Get the focal image centre along the v-dimension.
    /// \return The vertical centre in pixels.
    virtual double imageCenterV() const
    {
      return cv_;
    }

    /// \brief Get alpha in [0,1] of EUCM model
    /// \return alpha
    double alpha() const
    {
      return alpha_;
    }

    /// \brief Get beta >0 of EUCM model
    /// \return beta
    double beta() const
    {
      return beta_;
    }

    /// \brief Get the intrinsics as a concatenated vector.
    /// \return The intrinsics as a concatenated vector.
    inline void getIntrinsics(Eigen::VectorXd& intrinsics) const
    {
        intrinsics = intrinsics_;
    }

    /// \brief overwrite all intrinsics - use with caution !
    /// \param[in] intrinsics The intrinsics as a concatenated vector.
    inline bool setIntrinsics(const Eigen::VectorXd& intrinsics)
    {
        if (intrinsics.cols() != NumIntrinsics && intrinsics.rows() != NumIntrinsics) {
            return false;
        }
        intrinsics_ = intrinsics;
        fu_ = intrinsics[0];  //< focalLengthU
        fv_ = intrinsics[1];  //< focalLengthV
        cu_ = intrinsics[2];  //< imageCenterU
        cv_ = intrinsics[3];  //< imageCenterV
        alpha_ = intrinsics[4];  //< alpha
        beta_ = intrinsics[5];  //< beta
        one_over_fu_ = 1.0 / fu_;  //< 1.0 / fu_
        one_over_fv_ = 1.0 / fv_;  //< 1.0 / fv_
        fu_over_fv_ = fu_ / fv_;  //< fu_ / fv_
        return true;
    }

    /// \brief Not used
    virtual void setRectifyMap(const cv::Mat & /*map1*/, const cv::Mat & /*map2*/) {};

    /// \brief Not used
    virtual void getRectifyMap(cv::Mat & /*map1*/, cv::Mat & /*map2*/) const {};

    /// \brief Get the total number of intrinsics.
    /// \return Number of intrinsics parameters.
    inline int noIntrinsicsParameters() const
    {
      return NumIntrinsics;
    }

    inline virtual bool initialiseCameraAwarenessMaps()
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
    inline virtual bool getCameraAwarenessMaps(cv::Mat& rays, cv::Mat& imageJacobians) const
    {
        rays = rays_;
        imageJacobians = imageJacobians_;
        if (rays_.empty() || imageJacobians_.empty()) {
            return false;
        }
        return true;
    }
    const std::string distortionType() const
    {
      return "NoDistortion";
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
        const Eigen::Vector3d& point, Eigen::Vector2d* imagePoint) const
    {
        // projection
        Eigen::Vector2d imagePoint2;
        double rho = std::sqrt(beta_ * (point[0] * point[0] + point[1] * point[1]) + point[2] * point[2]);
        const double rz = 1.0 / (alpha_ * rho + (1.0 - alpha_) * point[2]);
        imagePoint2[0] = point[0] * rz;
        imagePoint2[1] = point[1] * rz;

        // scale and offset
        (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
        (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

        if (!CameraBase::isInImage(*imagePoint)) {
            return ProjectionStatus::OutsideImage;
        }
        if (CameraBase::isMasked(*imagePoint)) {
            return ProjectionStatus::Masked;
        }
        return ProjectionStatus::Successful;
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
        Eigen::Matrix2Xd* intrinsicsJacobian = NULL) const
    {
        // projection
        Eigen::Vector2d imagePoint2;
        double rho = std::sqrt(beta_ * (point[0] * point[0] + point[1] * point[1]) + point[2] * point[2]);
        const double rz = 1.0 / (alpha_ * rho + (1.0 - alpha_) * point[2]);
        imagePoint2[0] = point[0] * rz;
        imagePoint2[1] = point[1] * rz;

        // scale and offset
        (*imagePoint)[0] = fu_ * imagePoint2[0] + cu_;
        (*imagePoint)[1] = fv_ * imagePoint2[1] + cv_;

        if (intrinsicsJacobian) {
            // get both Jacobians
            double den = alpha_ * rho + (1. - alpha_) * point[2];
            intrinsicsJacobian->resize(2, NumIntrinsics);
            intrinsicsJacobian->setZero();
            (*intrinsicsJacobian)(0, 0) = point[0] / den;
            (*intrinsicsJacobian)(1, 1) = point[1] / den;
            (*intrinsicsJacobian)(0, 2) = 1.0;
            (*intrinsicsJacobian)(1, 3) = 1.0;
            (*intrinsicsJacobian)(0, 4) = -fu_ * point[0] * (rho - point[2]) / (den * den);
            (*intrinsicsJacobian)(1, 4) = -fv_ * point[1] * (rho - point[2]) / (den * den);
            (*intrinsicsJacobian)(0, 5) = -alpha_ * fu_ * point[0] * (point[0] * point[0] + point[1] * point[1]) / (2.0 * rho * den * den);
            (*intrinsicsJacobian)(1, 5) = -alpha_ * fv_ * point[1] * (point[0] * point[0] + point[1] * point[1]) / (2.0 * rho * den * den);
        }

        // compute the point Jacobian in any case
        Eigen::Matrix<double, 2, 3>& J = *pointJacobian;
        J(0, 0) = rz - (alpha_ * beta_ * point[0] * point[0]) * rz * rz / rho;
        J(0, 0) *= fu_;
        J(0, 1) = -(alpha_ * beta_ * point[0] * point[1]) * rz * rz / rho;
        J(0, 1) *= fu_;
        J(0, 2) = -point[0] * rz * rz * (1. - alpha_ + (alpha_ * point[2]) / rho);
        J(0, 2) *= fu_;

        J(1, 0) = -(alpha_ * beta_ * point[0] * point[1]) * rz * rz / rho;
        J(1, 0) *= fv_;
        J(1, 1) = rz - (alpha_ * beta_ * point[1] * point[1]) * rz * rz / rho;
        J(1, 1) *= fv_;
        J(1, 2) = -point[1] * rz * rz * (1. - alpha_ + (alpha_ * point[2]) / rho);
        J(1, 2) *= fv_;


        if (!CameraBase::isInImage(*imagePoint)) {
            return ProjectionStatus::OutsideImage;
        }
        if (CameraBase::isMasked(*imagePoint)) {
            return ProjectionStatus::Masked;
        }
        return ProjectionStatus::Successful;
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
        Eigen::Matrix2Xd* intrinsicsJacobian = NULL) const
    {
        // parse parameters into human readable form
        const double fu = parameters[0];
        const double fv = parameters[1];
        const double cu = parameters[2];
        const double cv = parameters[3];
        const double alpha = parameters[4];
        const double beta = parameters[5];

        // projection
        Eigen::Vector2d imagePoint2;
        double rho = std::sqrt(beta * (point[0] * point[0] + point[1] * point[1]) + point[2] * point[2]);
        const double rz = 1.0 / (alpha * rho + (1.0 - alpha) * point[2]);
        imagePoint2[0] = point[0] * rz;
        imagePoint2[1] = point[1] * rz;

        // scale and offset
        (*imagePoint)[0] = fu * imagePoint2[0] + cu;
        (*imagePoint)[1] = fv * imagePoint2[1] + cv;

        if (intrinsicsJacobian) {
            // get both Jacobians
            double den = alpha * rho + (1. - alpha) * point[2];
            intrinsicsJacobian->resize(2, NumIntrinsics);
            intrinsicsJacobian->setZero();
            (*intrinsicsJacobian)(0, 0) = point[0] / den;
            (*intrinsicsJacobian)(1, 1) = point[1] / den;
            (*intrinsicsJacobian)(0, 2) = 1.0;
            (*intrinsicsJacobian)(1, 3) = 1.0;
            (*intrinsicsJacobian)(0, 4) = -fu * point[0] * (rho - point[2]) / (den * den);
            (*intrinsicsJacobian)(1, 4) = -fv * point[1] * (rho - point[2]) / (den * den);
            (*intrinsicsJacobian)(0, 5) = -alpha * fu * point[0] * (point[0] * point[0] + point[1] * point[1]) / (2.0 * rho * den * den);
            (*intrinsicsJacobian)(1, 5) = -alpha * fv * point[1] * (point[0] * point[0] + point[1] * point[1]) / (2.0 * rho * den * den);
        }

        // compute the point Jacobian in any case
        Eigen::Matrix<double, 2, 3>& J = *pointJacobian;
        J(0, 0) = rz - (alpha * beta * point[0] * point[0]) * rz * rz / rho;
        J(0, 0) *= fu;
        J(0, 1) = -(alpha * beta * point[0] * point[1]) * rz * rz / rho;
        J(0, 1) *= fu;
        J(0, 2) = -point[0] * rz * rz * (1. - alpha + (alpha * point[2]) / rho);
        J(0, 2) *= fu;

        J(1, 0) = -(alpha * beta * point[0] * point[1]) * rz * rz / rho;
        J(1, 0) *= fv;
        J(1, 1) = rz - (alpha * beta * point[1] * point[1]) * rz * rz / rho;
        J(1, 1) *= fv;
        J(1, 2) = -point[1] * rz * rz * (1. - alpha + (alpha * point[2]) / rho);
        J(1, 2) *= fv;


        if (!CameraBase::isInImage(*imagePoint)) {
            return ProjectionStatus::OutsideImage;
        }
        if (CameraBase::isMasked(*imagePoint)) {
            return ProjectionStatus::Masked;
        }
        return ProjectionStatus::Successful;
    }

    /// \brief Projects Euclidean points to 2d image points (projection) in a batch.
    ///        Uses projection including distortion models.
    /// @param[in]  points      The points in Euclidean coordinates (one point per column).
    /// @param[out] imagePoints The image points (one point per column).
    /// @param[out] stati       Get information about the success of the projections. See
    ///                         \ref ProjectionStatus for more information.
    inline void projectBatch(
        const Eigen::Matrix3Xd& points, Eigen::Matrix2Xd* imagePoints,
        std::vector<ProjectionStatus>* stati) const
    {
        const int numPoints = points.cols();
        for (int i = 0; i < numPoints; ++i) {
            Eigen::Vector3d point = points.col(i);
            Eigen::Vector2d imagePoint;
            ProjectionStatus status = project(point, &imagePoint);
            imagePoints->col(i) = imagePoint;
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
        const Eigen::Vector4d& point, Eigen::Vector2d* imagePoint) const
    {
        Eigen::Vector3d point3d = point.head<3>() / point[3];
        return project(point3d, imagePoint);
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
        Eigen::Matrix2Xd* intrinsicsJacobian = NULL) const
    {
        Eigen::Vector3d point3d = point.head<3>() / point[3];
        Eigen::Matrix<double, 2, 3> pointJacobian3;
        ProjectionStatus status;
        status = project(point3d, imagePoint, &pointJacobian3, intrinsicsJacobian);

        pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
        pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3 / point[3]; // scale also Jacobian by homogeneous coordinate
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
        Eigen::Matrix<double, 2, 4>* pointJacobian = NULL,
        Eigen::Matrix2Xd* intrinsicsJacobian = NULL) const
    {
        Eigen::Vector3d point3d = point.head<3>() / point[3];
        Eigen::Matrix<double, 2, 3> pointJacobian3;
        ProjectionStatus status;
        status = projectWithExternalParameters(point3d, parameters, imagePoint, &pointJacobian3, intrinsicsJacobian);

        pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
        pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3 / point[3];
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
        std::vector<ProjectionStatus>* stati) const
    {
        const int numPoints = points.cols();
        for (int i = 0; i < numPoints; ++i) {
            Eigen::Vector4d point = points.col(i);
            Eigen::Vector2d imagePoint;
            ProjectionStatus status = projectHomogeneous(point, &imagePoint);
            imagePoints->col(i) = imagePoint;
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
        Eigen::Vector3d* direction) const
    {
        return backProject(imagePoint,direction,nullptr);
    }

    /// \brief Back-project a 2d image point into Euclidean space (direction vector).
    /// @param[in]  imagePoint         The image point.
    /// @param[out] direction          The Euclidean direction vector.
    /// @param[out] pointJacobian      Jacobian of the back-projection function  w.r.t. the point.
    /// @return     true on success.
    inline bool backProject(const Eigen::Vector2d& imagePoint,
        Eigen::Vector3d* direction,
        Eigen::Matrix<double, 3, 2>* pointJacobian) const
    {
        if(!direction || !imagePoint.allFinite() || !std::isfinite(fu_) || !std::isfinite(fv_)
            || fu_<=0 || fv_<=0 || !std::isfinite(alpha_) || !std::isfinite(beta_)
            || alpha_<0 || alpha_>1 || beta_<=0) return false;
        const double mx=(imagePoint.x()-cu_)/fu_,my=(imagePoint.y()-cv_)/fv_;
        const double radius=mx*mx+my*my;
        const double w=1-(2*alpha_-1)*beta_*radius;
        if(w<0 || (pointJacobian && w<=1e-12)) return false;
        const double root=std::sqrt(w),denominator=alpha_*root+1-alpha_;
        if(denominator<=1e-12) return false;
        const double numerator=1-beta_*alpha_*alpha_*radius;
        const Eigen::Vector3d ray(mx,my,numerator/denominator);
        if(!ray.allFinite() || ray.norm()<1e-12) return false;
        const auto unit=ray.normalized();
        if(pointJacobian) {
            const double dz=(-beta_*alpha_*alpha_*denominator
                +numerator*alpha_*(2*alpha_-1)*beta_/(2*root))/(denominator*denominator);
            Eigen::Matrix<double,3,2> derivative;
            derivative<<1/fu_,0,0,1/fv_,2*mx*dz/fu_,2*my*dz/fv_;
            const Eigen::Matrix<double,3,2> jacobian=(Eigen::Matrix3d::Identity()-unit*unit.transpose())*derivative/ray.norm();
            if(!jacobian.allFinite()) return false;
            *pointJacobian=jacobian;
        }
        *direction=unit; return true;
    }

    /// \brief Back-project 2d image points into Euclidean space (direction vectors).
    /// @param[in]  imagePoints The image points (one point per column).
    /// @param[out] directions  The Euclidean direction vectors (one point per column).
    /// @param[out] success     Success of each of the back-projection
    inline bool backProjectBatch(const Eigen::Matrix2Xd& imagePoints,
        Eigen::Matrix3Xd* directions,
        std::vector<bool>* success) const
    {
        const int numPoints = imagePoints.cols();
        for (int i = 0; i < numPoints; ++i) {
            Eigen::Vector2d imagePoint = imagePoints.col(i);
            Eigen::Vector3d point;
            bool suc = backProject(imagePoint, &point);
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
        Eigen::Vector4d* direction) const
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
        Eigen::Matrix<double, 4, 2>* pointJacobian) const
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
        std::vector<bool>* success) const
    {
        const int numPoints = imagePoints.cols();
        directions->resize(4,numPoints);
        directions->row(3).setOnes();
        for (int i = 0; i < numPoints; ++i) {
            Eigen::Vector2d imagePoint = imagePoints.col(i);
            Eigen::Vector4d point;
            bool suc = backProjectHomogeneous(imagePoint, &point);
            success->push_back(suc);
            directions->col(i) = point;
        }
        return true;
    }

    /// \brief get a test instance
    static std::shared_ptr<CameraBase> createTestObject()
    {
      return std::shared_ptr<CameraBase>(new EucmCamera(640, 480, 320, 320, 320, 240, 0.75, 0.99));
    }
    /// \brief get a test instance
    static EucmCamera testObject()
    {
      return EucmCamera(640, 480, 320, 320, 320, 240, 0.75, 0.99);
    }

    /// \brief Obtain the projection type
    std::string type() const
    {
      return "EUCMCamera";
    }

  protected:

    /// \brief No default constructor.
    EucmCamera() = delete;
    Eigen::Matrix<double, NumIntrinsics, 1> intrinsics_;  ///< summary of all intrinsics parameters
    double fu_;  ///< focalLengthU
    double fv_;  ///< focalLengthV
    double cu_;  ///< imageCenterU
    double cv_;  ///< imageCenterV
    double alpha_; ///< alpha parameter
    double beta_; ///< beta parameter
    double one_over_fu_;  ///< 1.0 / fu_
    double one_over_fv_;  ///< 1.0 / fv_
    double fu_over_fv_;  ///< fu_ / fv_

    cv::Mat rays_;
    cv::Mat imageJacobians_;

};

}  // namespace cameras
}  // namespace cgmli

#endif /*INCLUDE_CGMLI_CAMERAS_EUCMCAMERA_HPP_*/
