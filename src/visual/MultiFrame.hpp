
#ifndef INCLUDE_CGMLI_MULTIFRAME_HPP_
#define INCLUDE_CGMLI_MULTIFRAME_HPP_

#include <memory>
#include <optional>
#include <cstdint>
#include <utils/assert_macros.hpp>
#include <visual/Frame.hpp>
#include <cameras/NCameraSystem.hpp>

/// \brief cgmli Main namespace of this package.
namespace cgmli {

/// \class MultiFrame
/// \brief A multi camera frame that uses cgmli::Frame underneath.
class MultiFrame
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CGMLI_DEFINE_EXCEPTION(Exception,std::runtime_error)

  struct ImageTiming {
      cgmli::Time originalTimestamp; // Unmodified source exposure midpoint.
      int64_t offsetNs=0; // Source midpoint minus source bundle mean, before image_delay.
  };

  /// \brief Default constructor
  inline MultiFrame() : id_(0) {}

  /// \brief Construct from NCameraSystem
  /// @param[in] cameraSystem The camera system for which this is a multi-frame.
  /// @param[in] timestamp The time this frame was recorded.
  /// @param[in] id A unique frame Id.
  inline MultiFrame(const cameras::NCameraSystem & cameraSystem,
                    const cgmli::Time & timestamp, uint64_t id = 0) : timestamp_(timestamp),
      id_(id)
  {
      resetCameraSystemAndFrames(cameraSystem);
  }

  /// \brief Destructor...
  inline virtual ~MultiFrame() {};

  /// \brief (Re)set the NCameraSystem -- which clears the frames as well.
  /// @param[in] cameraSystem The camera system for which this is a multi-frame.
  inline void resetCameraSystemAndFrames(
      const cameras::NCameraSystem & cameraSystem) {
      cameraSystem_ = cameraSystem;
      frames_.clear();  // erase -- for safety
      frames_.resize(cameraSystem.numCameras());
      imageTimings_.clear();
      imageTimings_.resize(cameraSystem.numCameras());

      // copy cameras
      for (size_t c = 0; c < numFrames(); ++c) {
          frames_[c].setGeometry(cameraSystem.cameraGeometry(c));
      }
  }

  /// \brief (Re)set the timestamp
  /// @param[in] timestamp The time this frame was recorded.
  inline void setTimestamp(const cgmli::Time & timestamp) {
      timestamp_ = timestamp;
  }

  /// \brief Set the extrinsics.
  /// @param[in] cameraIdx The camera index that took the image.
  /// @param[in] T_SCi The extrinsics.
  inline void setExtrinsics(size_t cameraIdx, const kinematics::Transformation & T_SCi) {
      cameraSystem_.setExtrinsics(cameraIdx, T_SCi);
  }

  /// Apply one common reference-frame change without altering rig baselines,
  /// image measurements or the precomputed overlap graph.
  inline void transformCameraSystemReferenceFrame(const kinematics::Transformation& T_new_old) {
      cameraSystem_.transformReferenceFrame(T_new_old);
  }

  /// \brief (Re)set the id
  /// @param[in] id A unique frame Id.
  inline void setId(uint64_t id) {
      id_ = id;
  }

  /// \brief Obtain the frame timestamp
  /// \return The time this frame was recorded.
  inline const cgmli::Time & timestamp() const {
      return timestamp_;
  }

  /// Provenance only: all geometric factors still use timestamp() in the current
  /// synchronous approximation. An absent camera has no source-image timing.
  void setImageTiming(size_t camera,const cgmli::Time& original,int64_t offsetNs) {
      imageTimings_.at(camera)=ImageTiming{original,offsetNs};
  }
  const std::optional<ImageTiming>& imageTiming(size_t camera) const { return imageTimings_.at(camera); }

  /// \brief Obtain the frame id
  /// \return The unique frame Id.
  inline uint64_t id() const { return id_; }

  /// \brief The number of frames/cameras.
  /// \return How many individual frames/cameras there are.
  inline size_t numFrames() const {
      return frames_.size();
  }

  /// \brief Get the extrinsics of a camera.
  /// @param[in] cameraIdx The camera index for which the extrinsics are queried.
  /// \return The extrinsics as T_SC.
  inline std::shared_ptr<const cgmli::kinematics::Transformation> T_SC(
      size_t cameraIdx) const
  {
      return cameraSystem_.T_SC(cameraIdx);
  }

  /// \brief Get the N-camera system.
  /// \return Const reference to the N-camera system.
  inline const cameras::NCameraSystem & cameraSystem() const {
      return cameraSystem_;
  }


  //////////////////////////////////////////////////////////////
  /// \name The following mirror the Frame functionality.
  /// @{

  /// \brief Set the frame image;
  /// @param[in] cameraIdx The camera index that took the image.
  /// @param[in] image The image.
  inline void setImage(size_t cameraIdx, const cv::Mat & image) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          frames_[cameraIdx].setImage(image);
  }


  /// \brief Set the frame depth image;
  /// @param[in] cameraIdx The camera index that took the image.
  /// @param[in] depthImage The depth image.
  inline void setDepthImage(size_t cameraIdx, const cv::Mat & depthImage) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          frames_[cameraIdx].setDepthImage(depthImage);
  }

  /// \brief Set the geometry
  /// @param[in] cameraIdx The camera index.
  /// @param[in] cameraGeometry The camera geometry.
  inline void setGeometry(
      size_t cameraIdx,
      std::shared_ptr<const cameras::CameraBase> cameraGeometry) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          frames_[cameraIdx].setGeometry(cameraGeometry);
  }

  /// \brief Set the detector
  /// @param[in] cameraIdx The camera index.
  /// @param[in] detector The detector to be used.
  inline void setDetector(size_t cameraIdx,
                          std::shared_ptr<cv::FeatureDetector> detector) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          frames_[cameraIdx].setDetector(detector);
  }

  /// \brief Set the extractor
  /// @param[in] cameraIdx The camera index.
  /// @param[in] extractor The extractor to be used.
  inline void setExtractor(
      size_t cameraIdx,
      std::shared_ptr<cv::DescriptorExtractor> extractor) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
      frames_[cameraIdx].setExtractor(extractor);
  }

  /// \brief Obtain the image
  /// @param[in] cameraIdx The camera index.
  /// \return The image.
  inline const cv::Mat & image(size_t cameraIdx) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].image();
  }

  /// Const feature views for descriptor matchers; indices align after describe().
  inline const cv::Mat & descriptors(size_t cameraIdx) const {
      return frames_.at(cameraIdx).descriptors();
  }
  const auto& cudaDescriptorRows(size_t cameraIdx) const { return frames_.at(cameraIdx).cudaDescriptorRows(); }
  void setCudaDescriptorRows(size_t cameraIdx,std::shared_ptr<const features::detail::OrbCudaDescriptorRows> rows) {
      frames_.at(cameraIdx).setCudaDescriptorRows(std::move(rows));
  }
  const std::string& descriptorProfile(size_t cameraIdx) const {
      return frames_.at(cameraIdx).descriptorProfile();
  }
  void setDescriptorProfile(size_t cameraIdx,const std::string& profile) {
      frames_.at(cameraIdx).setDescriptorProfile(profile);
  }
  inline const std::vector<cv::KeyPoint> & keypoints(size_t cameraIdx) const {
      return frames_.at(cameraIdx).keypoints();
  }

  /// \brief Obtain the depth image
  /// @param[in] cameraIdx The camera index.
  /// \return The image.
  inline const cv::Mat & depthImage(size_t cameraIdx) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].depthImage();
  }

  /// \brief get the base class geometry (will be slow to use)
  /// @param[in] cameraIdx The camera index.
  /// \return The camera geometry.
  inline std::shared_ptr<const cameras::CameraBase> geometry(
      size_t cameraIdx) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].geometry();
  }

  /// \brief Get the specific geometry (will be fast to use)
  /// \tparam GEOMETRY_T The type for the camera geometry requested.
  /// @param[in] cameraIdx The camera index.
  /// \return The camera geometry.
  template<class GEOMETRY_T>
  inline std::shared_ptr<const GEOMETRY_T> geometryAs(size_t cameraIdx) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].geometryAs<GEOMETRY_T>();
  }

  /// \brief Detect keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// \return The number of detected points.
  inline int detect(size_t cameraIdx) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].detect();
  }

  /// \brief Describe keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// @param[in] cameraIdx The camera index.
  /// \return the number of detected points.
  inline int describe(size_t cameraIdx) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].describe();
  }

  /// \brief Compute the back projections. Caching them should speed up things like repeated
  /// triangulations.
  /// @param[in] cameraIdx The camera index.
  /// \return The number of back projected points.
  inline int computeBackProjections(size_t cameraIdx) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].computeBackProjections();
  }
  /// Share only immutable feature rays; poses/extrinsics and associations remain independent.
  inline bool shareBackProjections(size_t cameraIdx,const MultiFrame& source) {
      return frames_.at(cameraIdx).shareBackProjections(source.frames_.at(cameraIdx));
  }

  /// \brief Get a specific back-projection
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] backProjection The corresponding ray direction.
  /// \return whether or not the operation was successful.
  inline bool getBackProjection(size_t cameraIdx, size_t keypointIdx,
                                Eigen::Vector3d& backProjection) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].getBackProjection(keypointIdx, backProjection);
  }

  /// \brief Access a specific keypoint in OpenCV format
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getCvKeypoint(size_t cameraIdx, size_t keypointIdx,
                            cv::KeyPoint & keypoint) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].getCvKeypoint(keypointIdx, keypoint);
  }

  /// \brief Get a specific keypoint
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getKeypoint(size_t cameraIdx, size_t keypointIdx,
                          Eigen::Vector2d & keypoint) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].getKeypoint(keypointIdx, keypoint);
  }

  /// \brief Get the size of a specific keypoint
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypointSize The requested keypoint's size.
  /// \return whether or not the operation was successful.
  inline bool getKeypointSize(size_t cameraIdx, size_t keypointIdx,
                              double & keypointSize) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].getKeypointSize(keypointIdx, keypointSize);
  }


  /// \brief Access the descriptor -- CAUTION: high-speed version.
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// \return The descriptor data pointer; nullptr if out of bounds.
  inline const unsigned char * keypointDescriptor(size_t cameraIdx,
                                                  size_t keypointIdx) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].keypointDescriptor(keypointIdx);
  }

  /// \brief Set the landmark ID
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[in] landmarkId The landmark Id.
  /// \return whether or not the operation was successful.
  inline bool setLandmarkId(size_t cameraIdx, size_t keypointIdx,
                            uint64_t landmarkId) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].setLandmarkId(keypointIdx, landmarkId);
  }

  /// \brief Set the landmark ID
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[in] landmark The landmark.
  /// @param[in] initialised The landmark initialisation status.
  /// \return whether or not the operation was successful.
  inline bool setLandmark(size_t cameraIdx, size_t keypointIdx, const Eigen::Vector4d & landmark,
                          bool isInitialised) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].setLandmark(keypointIdx, landmark, isInitialised);
  }

  /// \brief Access the landmark ID
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// \return The landmark Id.
  inline uint64_t landmarkId(size_t cameraIdx, size_t keypointIdx) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].landmarkId(keypointIdx);
  }

  /// \brief Set the landmark ID
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] landmark The landmark.
  /// @param[out] isInitialised The landmark initialisation status.
  /// \return whether or not the operation was successful.
  inline bool getLandmark(size_t cameraIdx, size_t keypointIdx, Eigen::Vector4d & landmark,
                          bool & isInitialised) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].getLandmark(keypointIdx, landmark, isInitialised);
  }


  /// \brief provide keypoints externally
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypoints A vector of keyoints.
  /// \return whether or not the operation was successful.
  inline bool resetKeypoints(size_t cameraIdx,
                             const std::vector<cv::KeyPoint> & keypoints) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].resetKeypoints(keypoints);
  }


  /// \brief provide descriptors externally
  /// @param[in] cameraIdx The camera index.
  /// @param[in] descriptors A vector of descriptors.
  /// \return whether or not the operation was successful.
  inline bool resetDescriptors(size_t cameraIdx, const cv::Mat & descriptors) {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].resetDescriptors(descriptors);
  }

  /// \brief the number of keypoints
  /// @param[in] cameraIdx The camera index.
  /// \return The number of keypoints.
  inline size_t numKeypoints(size_t cameraIdx) const {
      CGMLI_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
          return frames_[cameraIdx].numKeypoints();
  }

  /// @}
  /// \brief Get the total number of keypoints in all frames.
  /// \return The total number of keypoints.
  inline size_t numKeypoints() const {
      size_t numKeypoints = 0;
      for (size_t i = 0; i < frames_.size(); ++i) {
          numKeypoints += frames_[i].numKeypoints();
      }
      return numKeypoints;
  }

  /// \brief Get the overlap mask. Sorry for the weird syntax, but remember that
  /// cv::Mat is essentially a shared pointer.
  /// @param[in] cameraIndexSeenBy The camera index for one camera.
  /// @param[in] cameraIndex The camera index for the other camera.
  /// @return The overlap mask image.
  inline const cv::Mat overlap(size_t cameraIndexSeenBy,
                               size_t cameraIndex) const
  {
    return cameraSystem_.overlap(cameraIndexSeenBy, cameraIndex);
  }

  /// \brief Can the first camera see parts of the FOV of the second camera?
  /// @param[in] cameraIndexSeenBy The camera index for one camera.
  /// @param[in] cameraIndex The camera index for the other camera.
  /// @return True, if there is at least one pixel of overlap.
  inline bool hasOverlap(size_t cameraIndexSeenBy, size_t cameraIndex) const
  {
    return cameraSystem_.hasOverlap(cameraIndexSeenBy, cameraIndex);
  }

 protected:
  cgmli::Time timestamp_;  ///< the frame timestamp
  uint64_t id_;  ///< the frame id
  std::vector<Frame, Eigen::aligned_allocator<Frame>> frames_;  ///< the individual frames
  std::vector<std::optional<ImageTiming>> imageTimings_;
  cameras::NCameraSystem cameraSystem_;  ///< the camera system
};

typedef std::shared_ptr<MultiFrame> MultiFramePtr;  ///< For convenience.

}  // namespace cgmli

#endif /* INCLUDE_CGMLI_MULTIFRAME_HPP_ */
