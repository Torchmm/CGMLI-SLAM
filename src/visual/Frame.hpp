
#ifndef INCLUDE_CGMLI_FRAME_HPP_
#define INCLUDE_CGMLI_FRAME_HPP_

#include <memory>
#include <string>
#include <utility>

#include <Eigen/StdVector>
#include <Eigen/Core>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/features2d/features2d.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop

#include <utils/Time.h>
#include <utils/assert_macros.hpp>
#include "cameras/CameraBase.hpp"

/// \brief cgmli Main namespace of this package.
namespace cgmli {
namespace features::detail { struct OrbCudaDescriptorRows; }

/// \class Frame
/// \brief A single camera frame equipped with keypoint detector / extractor.
class Frame
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CGMLI_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// \brief a default constructor
  inline Frame() = default;

  /// \brief A constructor that uses the image, specified geometry,
  /// detector and extractor
  /// @param[in] image The image.
  /// @param[in] cameraGeometry The camera geometry.
  /// @param[in] detector The detector to be used.
  /// @param[in] extractor The extractor to be used.
  inline Frame(const cv::Mat & image,
        std::shared_ptr<cameras::CameraBase> & cameraGeometry,
        std::shared_ptr<cv::FeatureDetector> & detector,
        std::shared_ptr<cv::DescriptorExtractor> & extractor)
      : image_(image),
      cameraGeometry_(cameraGeometry),
      detector_(detector),
      extractor_(extractor)
  {
  }

  /// \brief Copy constructor
  /// @param[in] other The other frame.
  inline Frame(const Frame& other) : image_(other.image_), depthImage_(other.depthImage_),
    cameraGeometry_(other.cameraGeometry_), detector_(other.detector_),
    extractor_(other.extractor_)
  {
  }

  /// \brief Default destructor
  inline virtual ~Frame() = default;

  /// \brief Set the frame image;
  /// @param[in] image The image.
  inline void setImage(const cv::Mat & image) {
      //cv::medianBlur(image,image_,3);
      image_ = image;
  }

  /// \brief Set the frame depth image;
  /// @param[in] depthImage The depth image.
  inline void setDepthImage(const cv::Mat & depthImage) {
      depthImage_ = depthImage;
  }

  /// \brief Set the geometry
  /// @param[in] cameraGeometry The camera geometry.
  inline void setGeometry(std::shared_ptr<const cameras::CameraBase> cameraGeometry) {
      if(cameraGeometry_.get()!=cameraGeometry.get()) backProjectionCache_.reset();
      cameraGeometry_ = cameraGeometry;
  }

  /// \brief Set the detector
  /// @param[in] detector The detector to be used.
  inline void setDetector(std::shared_ptr<cv::FeatureDetector> detector) {
      detector_ = detector;
  }

  /// \brief Set the extractor
  /// @param[in] extractor The extractor to be used.
  inline void setExtractor(std::shared_ptr<cv::DescriptorExtractor> extractor) {
      extractor_ = extractor;
  }

  /// \brief Obtain the image
  /// \return The image.
  inline const cv::Mat & image() const {
      return image_;
  }

  /// Descriptor row i belongs to keypoints()[i] after describe().
  inline const cv::Mat & descriptors() const { return descriptors_; }
  const auto& cudaDescriptorRows() const { return cudaDescriptorRows_; }
  void setCudaDescriptorRows(std::shared_ptr<const features::detail::OrbCudaDescriptorRows> rows) {
      cudaDescriptorRows_=std::move(rows);
  }
  /// Sampling-format provenance, not a claim about a descriptor's visual quality.
  /// External descriptor replacement invalidates it; the producer sets it afterwards.
  const std::string& descriptorProfile() const { return descriptorProfile_; }
  void setDescriptorProfile(const std::string& profile) { descriptorProfile_=profile; }
  inline const std::vector<cv::KeyPoint> & keypoints() const { return keypoints_; }

  /// \brief Obtain the depth image
  /// \return The image.
  inline const cv::Mat & depthImage() const {
      return depthImage_;
  }


  /// \brief get the base class geometry (will be slow to use)
  /// \return The camera geometry.
  inline std::shared_ptr<const cameras::CameraBase> geometry() const {
      return cameraGeometry_;
  }

  /// \brief Get the specific geometry (will be fast to use)
  /// \tparam GEOMETRY_T The type for the camera geometry requested.
  /// \return The camera geometry.
  template<class GEOMETRY_T>
  inline std::shared_ptr<const GEOMETRY_T> geometryAs() const {
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception, std::dynamic_pointer_cast<const GEOMETRY_T>(cameraGeometry_),
          "incorrect pointer cast requested. " << cameraGeometry_->distortionType())
          return std::static_pointer_cast<const GEOMETRY_T>(cameraGeometry_);
#else
      return std::static_pointer_cast<const GEOMETRY_T>(cameraGeometry_);
#endif
  }

  /// \brief Detect keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// \return The number of detected points.
  inline int detect() {
      // make sure things are set to zero for safety
      keypoints_.clear();
      descriptors_.release();
      cudaDescriptorRows_.reset();
      descriptorProfile_.clear();
      invalidateFeatureCaches();

      // resizing and filling in zeros in Frame::describe() as some keypoints are removed there:
      landmarkIds_.clear();
      landmarks_.clear();
      landmarkInitialisations_.clear();

      // run the detector
      CGMLI_ASSERT_TRUE_DBG(Exception, detector_ != nullptr,
          "Detector not initialised!")
          detector_->detect(image_, keypoints_);
      return int(keypoints_.size());
  }

  /// \brief Describe keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// \return The number of detected points.
  inline int describe() {
      cudaDescriptorRows_.reset();
      invalidateFeatureCaches();
      // check initialisation
      CGMLI_ASSERT_TRUE_DBG(Exception, extractor_ != nullptr,
          "Detector not initialised!")

          // extraction
          extractor_->compute(image_, keypoints_, descriptors_);
      descriptorProfile_=extractor_->getDefaultName();

      // resizing
      landmarkIds_ = std::vector<uint64_t>(keypoints_.size(), 0);
      landmarks_ = std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>(
          keypoints_.size(), Eigen::Vector4d(0, 0, 0, 0));
      landmarkInitialisations_ = std::vector<bool>(keypoints_.size(), false);
      return int(keypoints_.size());
  }


  /// \brief Compute the back projections. Caching them should speed up things like repeated
  /// triangulations.
  /// \return The number of back projected points.
  inline int computeBackProjections() {
      auto cache=std::make_shared<BackProjectionCache>();
      cache->rays.assign(keypoints_.size(),Eigen::Vector3d::Zero()); cache->valid.resize(keypoints_.size());
      int ctr = 0;
      for (size_t k = 0; k < keypoints_.size(); ++k) {
          cv::KeyPoint& ckp = keypoints_[k];
          // project ray
          const bool success = cameraGeometry_->backProject(
              Eigen::Vector2d(ckp.pt.x, ckp.pt.y), &cache->rays[k]);
          cache->valid[k] = success;
          if (success) {
              ctr++;
          }
      }
      backProjectionCache_=std::move(cache); return ctr;
  }

  /// Share immutable camera-coordinate rays after an explicit feature copy.
  /// A reset on either frame detaches its pointer; no mutable cache alias escapes.
  inline bool shareBackProjections(const Frame& source) {
      if(this==&source) return bool(backProjectionCache_);
      backProjectionCache_.reset();
      if(cameraGeometry_.get()!=source.cameraGeometry_.get() || keypoints_.size()!=source.keypoints_.size()) return false;
      for(size_t k=0;k<keypoints_.size();++k) if(keypoints_[k].pt!=source.keypoints_[k].pt) return false;
      if(!source.backProjectionCache_ || source.backProjectionCache_->rays.size()!=keypoints_.size()) return false;
      backProjectionCache_=source.backProjectionCache_; return true;
  }

  /// \brief Get a specific back-projection
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] backProjection The corresponding ray direction.
  /// \return whether or not the operation was successful.
  inline bool getBackProjection(size_t keypointIdx, Eigen::Vector3d& backProjection) const {
      if (!backProjectionCache_ || keypointIdx >= backProjectionCache_->rays.size() || keypointIdx >= backProjectionCache_->valid.size()) {
          return false; // detect/describe/resetKeypoints invalidates this cache.
      }
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception,
          keypointIdx < keypoints_.size(),
          "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
          backProjection = backProjectionCache_->rays[keypointIdx];
      return backProjectionCache_->valid[keypointIdx];
#else
      backProjection = backProjectionCache_->rays[keypointIdx];
      return backProjectionCache_->valid[keypointIdx];
#endif
  }

  /// \brief Access a specific keypoint in OpenCV format
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getCvKeypoint(size_t keypointIdx, cv::KeyPoint & keypoint) const {
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception,
          keypointIdx < keypoints_.size(),
          "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
          keypoint = keypoints_[keypointIdx];
      return keypointIdx < keypoints_.size();
#else
      keypoint = keypoints_[keypointIdx];
      return true;
#endif
  }

  /// \brief Get a specific keypoint
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getKeypoint(size_t keypointIdx, Eigen::Vector2d & keypoint) const {
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception,
          keypointIdx < keypoints_.size(),
          "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
          keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x,
              keypoints_[keypointIdx].pt.y);
      return keypointIdx < keypoints_.size();
#else
      keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x, keypoints_[keypointIdx].pt.y);
      return true;
#endif
  }

  /// \brief Get the size of a specific keypoint
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypointSize The requested keypoint's size.
  /// \return whether or not the operation was successful.
  inline bool getKeypointSize(size_t keypointIdx, double & keypointSize) const {
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception,
          keypointIdx < keypoints_.size(),
          "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
          keypointSize = double(keypoints_[keypointIdx].size);
      return keypointIdx < keypoints_.size();
#else
      keypointSize = double(keypoints_[keypointIdx].size);
      return true;
#endif
  }

  /// \brief Access the descriptor -- CAUTION: high-speed version.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// \return The descriptor data pointer; nullptr if out of bounds.
  inline const unsigned char * keypointDescriptor(size_t keypointIdx) const {
      if (keypointIdx >= size_t(descriptors_.rows)) return nullptr;
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception,
          keypointIdx < keypoints_.size(),
          "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
          return descriptors_.ptr<unsigned char>(int(keypointIdx));
#else
      return descriptors_.ptr<unsigned char>(int(keypointIdx));
#endif
  }

  /// \brief Set the landmark ID.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[in] landmarkId The landmark Id.
  /// \return whether or not the operation was successful.
  inline bool setLandmarkId(size_t keypointIdx, uint64_t landmarkId) {
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception,
          keypointIdx < landmarkIds_.size(),
          "keypointIdx " << keypointIdx
          << "out of range: landmarkIds_ has size " << landmarkIds_.size())
          landmarkIds_[keypointIdx] = landmarkId;
      return keypointIdx < keypoints_.size();
#else
      landmarkIds_[keypointIdx] = landmarkId;
      return true;
#endif
  }

  /// \brief Set the landmark.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[in] landmark The landmark.
  /// @param[in] isInitialised The landmark initialisation status.
  /// \return whether or not the operation was successful.
  inline bool setLandmark(size_t keypointIdx, const Eigen::Vector4d & landmark, bool isInitialised) {
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception,
          keypointIdx < landmarks_.size(),
          "keypointIdx " << keypointIdx << "out of range: landmarks_ has size " << landmarks_.size())
          landmarks_[keypointIdx] = landmark;
      landmarkInitialisations_[keypointIdx] = isInitialised;
      return keypointIdx < landmarks_.size();
#else
      landmarks_[keypointIdx] = landmark;
      landmarkInitialisations_[keypointIdx] = isInitialised;
      return true;
#endif
  }

  /// \brief Access the landmark ID.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// \return The landmark Id.
  inline uint64_t landmarkId(size_t keypointIdx) const {
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception,
          keypointIdx < landmarkIds_.size(),
          "keypointIdx " << keypointIdx << "out of range: landmarkIds has size " << landmarkIds_.size())
          return landmarkIds_[keypointIdx];
#else
      return landmarkIds_[keypointIdx];
#endif
  }

  /// \brief Get the landmark.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] landmark The landmark.
  /// @param[out] isInitialised The landmark initialisation status.
  /// \return whether or not the operation was successful.
  inline bool getLandmark(size_t keypointIdx, Eigen::Vector4d & landmark,
                          bool & isInitialised) const {
#ifndef NDEBUG
      CGMLI_ASSERT_TRUE(
          Exception,
          keypointIdx < landmarks_.size(),
          "keypointIdx " << keypointIdx << "out of range: landmarks has size " << landmarks_.size())
          landmark = landmarks_[keypointIdx];
      isInitialised = landmarkInitialisations_[keypointIdx];
      return keypointIdx < landmarks_.size();
#else
      landmark = landmarks_[keypointIdx];
      isInitialised = landmarkInitialisations_[keypointIdx];
      return true;
#endif
  }

  /// \brief Provide keypoints externally.
  /// @param[in] keypoints A vector of keyoints.
  /// \return whether or not the operation was successful.
  inline bool resetKeypoints(const std::vector<cv::KeyPoint> & keypoints) {
      keypoints_ = keypoints;
      descriptors_.release();
      cudaDescriptorRows_.reset();
      descriptorProfile_.clear();
      invalidateFeatureCaches();

      // resizing
      landmarkIds_ = std::vector<uint64_t>(keypoints_.size(), 0);
      landmarks_ = std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>(
          keypoints_.size(), Eigen::Vector4d(0, 0, 0, 0));
      landmarkInitialisations_ = std::vector<bool>(keypoints_.size(), false);

      return true;
  }

  /// \brief provide descriptors externally
  /// @param[in] descriptors A vector of descriptors.
  /// \return whether or not the operation was successful.
  inline bool resetDescriptors(const cv::Mat & descriptors) {
      descriptors_ = descriptors;
      cudaDescriptorRows_.reset();
      descriptorProfile_.clear();
      return true;
  }

  /// \brief Get the number of keypoints.
  /// \return The number of keypoints.
  inline size_t numKeypoints() const {
      return keypoints_.size();
  }

 protected:
  std::string descriptorProfile_;
  inline void invalidateFeatureCaches() {
      backProjectionCache_.reset();
  }
  cv::Mat image_;  ///< the image as OpenCV's matrix
  cv::Mat depthImage_;  ///< the depth image as OpenCV's matrix
  std::shared_ptr<const cameras::CameraBase> cameraGeometry_;  ///< the camera geometry
  std::shared_ptr<cv::FeatureDetector> detector_;  ///< the detector
  std::shared_ptr<cv::DescriptorExtractor> extractor_;  ///< the extractor
  std::vector<cv::KeyPoint> keypoints_;  ///< we store keypoints using OpenCV's struct

  /// \brief We cache backprojections.
  struct BackProjectionCache {
      std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> rays;
      std::vector<bool> valid;
  };
  std::shared_ptr<const BackProjectionCache> backProjectionCache_;
  cv::Mat descriptors_;  ///< we store the descriptors using OpenCV's matrices
  std::shared_ptr<const features::detail::OrbCudaDescriptorRows> cudaDescriptorRows_;
  std::vector<uint64_t> landmarkIds_;  ///< landmark Id, if associated -- 0 otherwise
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> landmarks_; ///< pos in C.
  std::vector<bool> landmarkInitialisations_; ///< Store landmark initialisation stati.
};

}  // namespace cgmli

#endif /* INCLUDE_CGMLI_FRAME_HPP_ */
