
#ifndef INCLUDE_CGMLI_VIOVISUALIZER_HPP_
#define INCLUDE_CGMLI_VIOVISUALIZER_HPP_

#include <utils/assert_macros.hpp>
#include <common/Parameters.hpp>
#include <visual/MultiFrame.hpp>
#include <common/FrameTypedefs.hpp>
#include <optional>
#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include <deque>

/// \brief cgmli Main namespace of this package.
namespace cgmli {

/**
 * @brief This class is responsible to visualize the matching results
 */
class ViVisualizer {
 public:

  /// Owned thumbnails and feature coordinates, independent of mutable backend data.
  /// The control thread captures these after extraction; workers only read them.
  struct FeatureCamera {
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    double scaleX=1, scaleY=1;
    std::optional<int64_t> timeOffsetNs;
  };
  struct FeatureFrame {
    uint64_t id=0,timestampNs=0;
    std::vector<FeatureCamera> cameras;
  };
  struct FeatureDebugData {
    std::shared_ptr<const FeatureFrame> current;
    std::map<uint64_t,std::shared_ptr<const FeatureFrame>> sources;
    std::vector<FeatureMatchDebug> matches;
    bool committed=false,visualConsensus=false;
    size_t inliers=0;
  };
  static std::shared_ptr<const FeatureFrame> captureFeatures(const MultiFrame& frame);
  static cv::Mat cameraMosaic(const std::vector<cv::Mat>& images,const std::string& title);
  static std::map<std::string,cv::Mat> drawFeatureDebug(const FeatureDebugData& data);

  /// @brief This struct contains the relevant data for visualizing
  struct VisualizationData {
    /// \brief Describe tracking quality (qualitatively).
    enum class TrackingQuality {
      Good,
      Marginal,
      Lost
    };

    /// \brief Pointer to VisualizationData.
    typedef std::shared_ptr<VisualizationData> Ptr;

    cgmli::ObservationVector observations;    ///< Vector containing all the keypoint observations.
    std::shared_ptr<cgmli::MultiFrame> currentFrames; ///< Current multiframe.
    cgmli::kinematics::Transformation T_WS;  ///< Pose of the current frame
    AlignedVector<kinematics::Transformation> T_SCi;  ///< Camera extrinsics (may change).
    bool isKeyframe = false; ///< Is it a keyframe?
    bool recognisedPlace = false; ///< is it a place recognition frame?
    TrackingQuality trackingQuality = TrackingQuality::Good; ///< The tracking quality.
    std::shared_ptr<const FeatureDebugData> featureDebug;
    struct TrajectoryPoint { uint64_t id=0; Eigen::Vector3d position=Eigen::Vector3d::Zero(); bool keyframe=false; };
    uint64_t acceptedFrame=0;
    std::vector<size_t> activeLandmarksByCamera;
    size_t windowMature3d=0;
    std::vector<TrajectoryPoint> trajectory,window;
    struct TrackLifecycle { uint32_t initializations=0,resets=0,demotions=0; uint64_t lastReset=0; };
    std::map<uint64_t,TrackLifecycle> trackLifecycles;
  };
  /// Render immutable accepted values on the visualization worker, never in Ceres.
  static cv::Mat drawTrajectory(const VisualizationData& data);
  cv::Mat drawTrackTrails(const VisualizationData& data);

  CGMLI_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /**
   * @brief Constructor.
   * @param parameters Parameters and settings.
   */
  ViVisualizer(cgmli::ViParameters& parameters);

  /// \brief Destructor.
  virtual ~ViVisualizer();

  /**
   * @brief Initialise parameters. Called in constructor.
   * @param parameters Parameters and settings.
   */
  void init(cgmli::ViParameters& parameters);

  /**
   * @brief Circles all keypoints in the current frame, links the matching ones to
   *        the current keyframe and returns the result.
   * @param data Visualization data.
   * @param image_number Index of the frame to display.
   * @return OpenCV matrix with the resulting image.
   */
  cv::Mat drawMatches(VisualizationData::Ptr& data, size_t image_number);
  
 private:
  /**
   * @brief Circles all keypoints in the current frame and returns the result.
   * @param data Visualization data.
   * @param cameraIndex Index of the frame to display.
   * @return OpenCV matrix with the resulting image.
   */
  cv::Mat drawKeypoints(VisualizationData::Ptr& data, size_t cameraIndex);

  /// Parameters and settings.
  cgmli::ViParameters parameters_;
  struct TrackTrail { uint64_t lastFrame=0; std::deque<cv::Point2f> points; };
  std::map<std::pair<uint64_t,size_t>,TrackTrail> trackTrails_;
  uint64_t lastTrailFrame_=0;
};

} /* namespace cgmli */

#endif /* INCLUDE_CGMLI_VIOVISUALIZER_HPP_ */
