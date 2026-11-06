
#ifndef INCLUDE_CGMLI_NCAMERASYSTEM_HPP_
#define INCLUDE_CGMLI_NCAMERASYSTEM_HPP_

#include <memory>
#include <map>
#include <mutex>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#include <kinematics/Transformation.hpp>
#include <utils/assert_macros.hpp>
#include "cameras/CameraBase.hpp"

/// \brief cgmli Main namespace of this package.
namespace cgmli {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

/// \class NCameraSystem
/// \brief A class that assembles multiple cameras into a system of
/// (potentially different) cameras.
class NCameraSystem
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CGMLI_DEFINE_EXCEPTION(Exception,std::runtime_error)
  
  /// The enumeration of the currently supported distortion types.
  enum DistortionType  {
    Equidistant = 0, ///< Use with cgmli::cameras::EquidistantDistortion.
    RadialTangential = 1, ///< Use with cgmli::cameras::RadialTangentialDistortion.
    NoDistortion = 2, ///< No distortion.
    RadialTangential8 = 3 ///< Use with cgmli::cameras::RadialTangentialDistortion.
  };

  /// \brief Default constructor
  inline NCameraSystem() : numUsedCameras_(0)
  {
  }

  /// \brief Destructor that doesn't do anything really.
  inline virtual ~NCameraSystem(){};

  /// \brief Helper struct to specify the camera type.
  struct CameraType {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraType(){} ///< Default constructor.
    bool isColour = true; ///< Set true for RGB8, false for mono8.
    bool isUsed = true; ///< Set false if not in use for cgmli.
    bool isUsedMapping = false; ///< Set true if the camera will also be used for mapping 
    /// \brief Helper struct to specify the depth camera type.
    struct DepthType {
      bool isDepthCamera = false; ///< Is it a depth camera?
      Eigen::Vector3d baseline = Eigen::Vector3d(0.0,0.0,0.0); ///< Stereo baseline [m].
      bool createVirtual = false; ///< If true, creating measurements in virtual camera.
      bool createDepth = false; ///< If true, creating measurements constraining depth.
      double sigmaPixels = 1.0; ///< Uncertainty (std. deviation) [pix].
      double sigmaDepth = 0.01; ///< Depth uncertainty (std. deviation) [m].
      bool needRectify = false; ///< Is rectification needed for stereo network?
    };
    DepthType depthType; ///< Depth camera type.
  };

  /// \brief Append with a single camera.
  /// @param[in] T_SC extrinsics.
  /// @param[in] cameraGeometry Camera geometry.
  /// @param[in] distortionType Distortion type.
  /// @param[in] computeOverlaps Explicitly compute cross-camera masks after this insertion.
  /// Defaults to false: append the complete rig, then call computeOverlaps() once.
  /// @param[in] cameraType The type of the camera (colour / gray / depth, etc.).
  inline void addCamera(std::shared_ptr<const cgmli::kinematics::Transformation> T_SC,
                        std::shared_ptr<const cameras::CameraBase> cameraGeometry,
                        DistortionType distortionType,
                        bool computeOverlaps = false,
                        const CameraType & cameraType = CameraType()) {
      T_SC_.push_back(T_SC);
      cameraGeometries_.push_back(cameraGeometry);
      distortionTypes_.push_back(distortionType);
      cameraTypes_.push_back(cameraType); ///< is it a depth camera etc
      invalidateOverlaps();

      if (cameraType.isUsed) {
          numUsedCameras_++;
      }

      if (cameraType.depthType.isDepthCamera && cameraType.depthType.createVirtual) {
          // create an additional virtual camera
          cgmli::kinematics::Transformation T_CCvirtual(
              cameraType.depthType.baseline,
              Eigen::Quaterniond::Identity());
          cgmli::kinematics::Transformation T_SCvirtual(*T_SC * T_CCvirtual);
          virtual_T_SC_.push_back(std::shared_ptr<const cgmli::kinematics::Transformation>(
              new cgmli::kinematics::Transformation(T_SCvirtual)));
          virtual_cameraGeometries_.push_back(cameraGeometry); // TODO: check if this is thread-safe.
          virtual_distortionTypes_.push_back(distortionType);
      }

      // recompute overlaps if requested
      if (computeOverlaps) {
          this->computeOverlaps();
      }
  }

  /// \brief Append with a rectified camera
  /// @param[in] cameraIndex Camera index to be added
  /// @param[in] T_SC Rectified camera extrinsics
  /// @param[in] cameraGeometry Rectified camera geometry
  inline void addRectifyCamera(size_t cameraIndex,
                               std::shared_ptr<const cgmli::kinematics::Transformation> T_SC,
                               std::shared_ptr<const cameras::CameraBase> cameraGeometry) {
      rectify_T_SC_[cameraIndex] = T_SC;
      rectify_cameraGeometries_[cameraIndex] = cameraGeometry;
  }

  /// \brief Obtatin the number of cameras currently added.
  /// @return The number of cameras.
  inline size_t numCameras() const {
      return cameraGeometries_.size();
  }

  /// \brief Obtatin the number of cameras currently added.
  /// @return The number of cameras.
  inline size_t numUsedCameras() const {
      return numUsedCameras_;
  }

  /// \brief Compute directed cross-camera overlap masks after configuring the rig.
  /// Diagonal masks stay empty; overlap(i,i) creates its all-visible mask on demand.
  /// Configuration and recomputation require exclusive access to this rig instance.
  void computeOverlaps();

  /// Change the reference frame of the complete rigid camera assembly.
  /// Every extrinsic is left-multiplied by T_new_old. Relative camera poses,
  /// image-space overlap masks and self-visibility caches remain unchanged.
  /// Individual camera calibration changes must still use setExtrinsics().
  void transformReferenceFrame(const kinematics::Transformation& T_new_old);

  /// \brief Get the pose of the IMU frame S with respect to the camera cameraIndex
  /// @param[in] cameraIndex The camera index for which the extrinsics should be returned.
  /// @return T_SC, the extrinsics.
  inline std::shared_ptr<const cgmli::kinematics::Transformation> T_SC(size_t cameraIndex) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIndex < T_SC_.size(),
          "Camera index " << cameraIndex << "out of range.");
      return T_SC_[cameraIndex];
  }

  /// \brief get the pose of the IMU frame S with respect to the rectified camera cameraIndex
  /// @param[in] cameraIndex The camera index for which the extrinsics should be returned.
  /// @return T_SC, the rectified extrinsics.
  inline std::shared_ptr<const cgmli::kinematics::Transformation> rectifyT_SC(size_t cameraIndex) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, rectify_T_SC_.find(cameraIndex) != rectify_T_SC_.end(),
          "Rectified camera index " << cameraIndex << "out of range.");
      try {
          return rectify_T_SC_.at(cameraIndex);
      }
      catch (const std::out_of_range& ex) {
          throw std::runtime_error("The camera ID is out of range in the rectified camera.");
      }
  }

  /// \brief get the camera geometry of camera cameraIndex
  /// \brief Set the camera extrinsics of camera cameraIndex.
  /// @param[in] cameraIndex The camera index for which the extrinsics should be set.
  /// @param[in] T_SCi The new extrinsics.
  inline void setExtrinsics(size_t cameraIndex, kinematics::Transformation T_SCi) {
      CGMLI_ASSERT_TRUE_DBG(Exception,
          cameraIndex < T_SC_.size(),
          "Camera index " << cameraIndex << "out of range.");
      T_SC_.at(cameraIndex).reset(new kinematics::Transformation(T_SCi));
      invalidateOverlaps();
  }

  /// \brief Get the camera geometry of camera cameraIndex
  /// @param[in] cameraIndex The camera index for which the camera geometry should be returned.
  /// @return The camera geometry.
  inline std::shared_ptr<const cameras::CameraBase> cameraGeometry(size_t cameraIndex) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIndex < cameraGeometries_.size(),
          "Camera index " << cameraIndex << "out of range.");
      return cameraGeometries_[cameraIndex];
  }

  /// \brief get the rectified camera geometry of camera cameraIndex
  /// @param[in] cameraIndex The camera index for which the rectified camera geometry should be returned.
  /// @return The camera geometry.
  inline std::shared_ptr<const cameras::CameraBase> rectifyCameraGeometry(size_t cameraIndex) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, rectify_cameraGeometries_.find(cameraIndex) != rectify_cameraGeometries_.end(),
          "Rectified camera index " << cameraIndex << "out of range.");
      try {
          return rectify_cameraGeometries_.at(cameraIndex);
      }
      catch (const std::out_of_range& ex) {
          throw std::runtime_error("The camera ID is out of range in the rectified camera.");
      }
  }

  /// \brief get the distortion type of the camera
  /// @param[in] cameraIndex The camera index for which the distortion type should be returned.
  /// @return The distortion type
  inline DistortionType distortionType(size_t cameraIndex) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIndex < cameraGeometries_.size(),
          "Camera index " << cameraIndex << "out of range.");
      return distortionTypes_[cameraIndex];
  }

  /// \brief Get the overlap mask. Sorry for the weird syntax, but remember that
  /// cv::Mat is essentially a shared pointer.
  /// @param[in] cameraIndexSeenBy The camera index for one camera.
  /// @param[in] cameraIndex The camera index for the other camera.
  /// @return A 0/1 mask in the second camera's image coordinates. A self mask is
  /// created and cached on first request, without requiring cross-camera computation.
  /// Returned matrices share storage and must be treated as read-only by callers.
  inline const cv::Mat overlap(size_t cameraIndexSeenBy,
                                 size_t cameraIndex) const {
      CGMLI_ASSERT_TRUE(
          Exception, cameraIndexSeenBy < T_SC_.size(),
          "Camera index " << cameraIndexSeenBy << "out of range.");
      CGMLI_ASSERT_TRUE(Exception, cameraIndex < T_SC_.size(),
          "Camera index " << cameraIndex << "out of range.");

      if (cameraIndexSeenBy == cameraIndex) return selfOverlapMask(cameraIndex);
      CGMLI_ASSERT_TRUE(Exception, overlapComputationValid(),
          "Overlap computation not performed or incorrectly computed!");

      return overlapMats_[cameraIndexSeenBy][cameraIndex];
  }

  /// \brief Can the first camera see parts of the FOV of the second camera?
  /// @param[in] cameraIndexSeenBy The camera index for one camera.
  /// @param[in] cameraIndex The camera index for the other camera.
  /// @return True, if there is at least one pixel of overlap.
  inline bool hasOverlap(size_t cameraIndexSeenBy, size_t cameraIndex) const {
      CGMLI_ASSERT_TRUE(
          Exception, cameraIndexSeenBy < T_SC_.size(),
          "Camera index " << cameraIndexSeenBy << "out of range.");
      CGMLI_ASSERT_TRUE(Exception, cameraIndex < T_SC_.size(),
          "Camera index " << cameraIndex << "out of range.");

      // Self visibility is known without allocating a mask or computing other pairs.
      if (cameraIndexSeenBy == cameraIndex) return true;
      CGMLI_ASSERT_TRUE(Exception, overlapComputationValid(),
          "Overlap computation not performed or incorrectly computed!");

      return overlaps_[cameraIndexSeenBy][cameraIndex];
  }

  /// \brief Is it a depth camera?
  /// \param cameraIndex The index of the camera in question.
  /// \return Ture if depth camera.
  inline bool isDepthCamera(size_t cameraIndex) const {
    return cameraTypes_.at(cameraIndex).depthType.isDepthCamera;
  }

  /// \brief The camera type.
  /// \param cameraIndex The index of the camera in question.
  /// \return The camera type.
  CameraType cameraType(size_t cameraIndex) const {
    return cameraTypes_.at(cameraIndex);
  }

  /// \brief Check if the camera type for the given index is configured.
  /// \param cameraIndex The index of the camera in question.
  /// \return True if the camera type is configured.
  bool isCameraConfigured(size_t cameraIndex) const {
    return cameraIndex < cameraTypes_.size();
  }

  inline bool overlapComputationValid() const {
      CGMLI_ASSERT_TRUE_DBG(
          Exception, T_SC_.size() == cameraGeometries_.size(),
          "Number of extrinsics must match number of camera models!");

      if (overlaps_.size() != cameraGeometries_.size()) {
          return false;
      }
      if (overlapMats_.size() != cameraGeometries_.size()) {
          return false;
      }

      // also check for each element
      for (size_t i = 0; i < overlaps_.size(); ++i) {
          if (overlaps_[i].size() != cameraGeometries_.size()) {
              return false;
          }
          if (overlapMats_[i].size() != cameraGeometries_.size()) {
              return false;
          }
      }
      return true;
  }

 protected:
  /// \brief Use this to check overlapMats_ and overlaps_ have correct sizes
  /// @return True, if valid.
  
  std::vector<std::shared_ptr<const cgmli::kinematics::Transformation>> T_SC_; ///< IMU-cam trafo.
  std::vector<std::shared_ptr<const cameras::CameraBase>> cameraGeometries_;  ///< Cam geometries.
  std::vector<DistortionType> distortionTypes_; ///< Distortion types by camera.
  std::vector<std::vector<cv::Mat>> overlapMats_;  ///< Cross-camera masks; diagonal entries remain empty.
  std::vector<std::vector<bool>> overlaps_;  ///< Overlaps between cameras: binary.
  std::vector<CameraType> cameraTypes_; ///< Camera types.
  std::vector<std::shared_ptr<const cgmli::kinematics::Transformation>> virtual_T_SC_; ///< IMU-cam trafo.
  std::vector<std::shared_ptr<const cameras::CameraBase>> virtual_cameraGeometries_;  ///< Cam geometries.
  std::vector<DistortionType> virtual_distortionTypes_; ///< Distortion types by camera.
  std::map<size_t, std::shared_ptr<const cgmli::kinematics::Transformation>> rectify_T_SC_; ///< IMU-cam(rectify) trafo.
  std::map<size_t, std::shared_ptr<const cameras::CameraBase>> rectify_cameraGeometries_;  ///< Rectified Cam geometries.
  size_t numUsedCameras_; ///< Number of cameras used.

  // Rig copies share immutable geometry and may safely share lazily generated self
  // masks. Mutation detaches the cache so independently extended rigs cannot mix IDs.
  struct SelfOverlapCache {
      std::mutex mutex;
      std::map<size_t, cv::Mat> masks;
  };
  std::shared_ptr<SelfOverlapCache> selfOverlapCache_ = std::make_shared<SelfOverlapCache>();
  const cv::Mat selfOverlapMask(size_t cameraIndex) const;
  inline void invalidateOverlaps() {
      overlapMats_.clear();
      overlaps_.clear();
      selfOverlapCache_ = std::make_shared<SelfOverlapCache>();
  }
};

}  // namespace cameras
}  // namespace cgmli

#endif /* INCLUDE_CGMLI_NCAMERASYSTEM_HPP_ */
