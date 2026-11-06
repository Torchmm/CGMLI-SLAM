
#include "cameras/NCameraSystem.hpp"
#include <glog/logging.h>
#include <chrono>
#include <cmath>
#include <utility>

/// \brief cgmli Main namespace of this package.
namespace cgmli {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

void NCameraSystem::transformReferenceFrame(const kinematics::Transformation& T_new_old)
{
  CGMLI_ASSERT_TRUE(Exception, T_new_old.T().allFinite()
      && std::abs(T_new_old.q().squaredNorm()-1.0)<1e-6,
      "Camera reference-frame transform must be a finite rigid transform");
  // Build replacements before publishing. Copied rigs share immutable extrinsic
  // pointers and overlap images; neither may be modified through those pointers.
  auto cameras = T_SC_;
  auto virtualCameras = virtual_T_SC_;
  auto rectifiedCameras = rectify_T_SC_;
  const auto transform = [&](const std::shared_ptr<const kinematics::Transformation>& pose) {
    CGMLI_ASSERT_TRUE(Exception, bool(pose), "Camera extrinsic is missing");
    return std::make_shared<const kinematics::Transformation>(T_new_old * (*pose));
  };
  for (auto& pose : cameras) pose = transform(pose);
  for (auto& pose : virtualCameras) pose = transform(pose);
  for (auto& entry : rectifiedCameras) entry.second = transform(entry.second);
  T_SC_.swap(cameras);
  virtual_T_SC_.swap(virtualCameras);
  rectify_T_SC_.swap(rectifiedCameras);
  // (A*T_SCj)^-1*(A*T_SCi) == T_SCj^-1*T_SCi: no overlap recomputation.
}

const cv::Mat NCameraSystem::selfOverlapMask(size_t cameraIndex) const
{
  const auto& camera = cameraGeometries_.at(cameraIndex);
  std::lock_guard<std::mutex> lock(selfOverlapCache_->mutex);
  auto& mask = selfOverlapCache_->masks[cameraIndex];
  if (mask.empty()) {
    mask = cv::Mat::ones(int(camera->imageHeight()), int(camera->imageWidth()), CV_8UC1);
    VLOG(1) << "[Rig] Created self-overlap mask on demand; camera=" << cameraIndex;
  }
  return mask;
}

/// \brief compute all the overlaps of fields of view. Attention: can be expensive.
void NCameraSystem::computeOverlaps()
{
  const auto started = std::chrono::steady_clock::now();
  LOG(INFO) << "[Rig] Computing angular FOV overlap masks; cameras=" << cameraGeometries_.size();
  CGMLI_ASSERT_TRUE_DBG(
      Exception, T_SC_.size() == cameraGeometries_.size(),
      "Number of extrinsics must match number of camera models!")

  // Build fresh tables before publishing them. Reusing cv::Mat buffers could write
  // into a copied rig's masks, and resizing alone would retain stale overlap flags.
  const size_t count = cameraGeometries_.size();
  std::vector<std::vector<cv::Mat>> masks(count, std::vector<cv::Mat>(count));
  std::vector<std::vector<bool>> visible(count, std::vector<bool>(count, false));
  for (size_t cameraIndexSeenBy = 0; cameraIndexSeenBy < count;
      ++cameraIndexSeenBy) {
    for (size_t cameraIndex = 0; cameraIndex < count;
        ++cameraIndex) {

      // Self visibility needs neither projection nor an allocated image mask.
      if (cameraIndex == cameraIndexSeenBy) {
        visible[cameraIndexSeenBy][cameraIndex] = true;
        continue;
      }
      {
        const auto& camera = cameraGeometries_[cameraIndex];
        // sizing the overlap map:
        const size_t height = camera->imageHeight();
        const size_t width = camera->imageWidth();
        cv::Mat& overlapMat = masks[cameraIndexSeenBy][cameraIndex];
        overlapMat = cv::Mat::zeros(int(height), int(width), CV_8UC1);
        // go through all the pixels:
        std::shared_ptr<const CameraBase> otherCamera =
            cameraGeometries_[cameraIndexSeenBy];
        const cgmli::kinematics::Transformation T_Cother_C =
            T_SC_[cameraIndexSeenBy]->inverse() * (*T_SC_[cameraIndex]);
        for (size_t u = 0; u < width; ++u) {
          for (size_t v = 0; v < height; ++v) {
            // backproject
            Eigen::Vector3d ray_C;
            if(!camera->backProject(Eigen::Vector2d(double(u), double(v)), &ray_C) || !ray_C.allFinite()) continue;
            // project into other camera
            Eigen::Vector3d ray_Cother = T_Cother_C.C() * ray_C;
            // points at infinity, i.e. we only do rotation
            Eigen::Vector2d imagePointInOtherCamera;
            ProjectionStatus status = otherCamera->project(
                ray_Cother, &imagePointInOtherCamera);

            // check the result
            if (status == ProjectionStatus::Successful) {

              Eigen::Vector3d verificationRay;
              if (!otherCamera->backProject(imagePointInOtherCamera,&verificationRay) || !verificationRay.allFinite()) continue;

              // to avoid an artefact of some distortion models, check again
              // note: (this should be fixed in the distortion implementation)
              if(fabs(ray_Cother.normalized().transpose()*verificationRay.normalized()-1.0)
                 <1.0e-10) {
                // fill in the matrix:
                overlapMat.at<uchar>(int(v),int(u)) = 1;
                // and remember there is some overlap at all.
                visible[cameraIndexSeenBy][cameraIndex] = true;
              }
            }
          }
        }
      }
    }
  }
  overlapMats_ = std::move(masks);
  overlaps_ = std::move(visible);
  LOG(INFO) << "[Rig] FOV overlap masks ready; elapsed_ms="
      << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count();
  for (size_t a=0; a<cameraGeometries_.size(); ++a) for (size_t b=a+1; b<cameraGeometries_.size(); ++b) {
    const auto& maskB=overlapMats_[a][b];
    const auto& maskA=overlapMats_[b][a];
    LOG(INFO) << "[Rig] FOV pair=" << a << ',' << b
        << " fraction_cam" << a << '=' << double(cv::countNonZero(maskA))/double(maskA.total())
        << " fraction_cam" << b << '=' << double(cv::countNonZero(maskB))/double(maskB.total());
  }
}

}  // namespace cameras
}  // namespace cgmli

