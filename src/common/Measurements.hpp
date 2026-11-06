
#ifndef INCLUDE_CGMLI_MEASUREMENTS_HPP_
#define INCLUDE_CGMLI_MEASUREMENTS_HPP_

#include <deque>
#include <vector>
#include <memory>
#include <optional>
#include <cstdint>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#include <map>
#include <Eigen/Dense>
#include <utils/Time.h>
#include <kinematics/Transformation.hpp>

/// \brief cgmli Main namespace of this package.
namespace cgmli {
/**
 * \brief Generic measurements
 *
 * They always come with a timestamp such that we can perform
 * any kind of asynchronous operation.
 * \tparam MEASUREMENT_T Measurement data type.
 */
template<class MEASUREMENT_T>
struct Measurement {
  cgmli::Time timeStamp;      ///< Measurement timestamp
  MEASUREMENT_T measurement;  ///< Actual measurement.
  int sensorId = -1;          ///< Sensor ID. E.g. camera index in a multicamera setup

  /// \brief Copy constructor.
  /// \param other The other measurement.
  Measurement(const Measurement<MEASUREMENT_T>& other) noexcept :
    timeStamp(other.timeStamp), measurement(other.measurement), sensorId(other.sensorId)
  {

  }

  /// \brief Assignment operator.
  /// \param other The other measurement.
  /// \return This.
  Measurement<MEASUREMENT_T>& operator=(const Measurement<MEASUREMENT_T>& other) noexcept {
    timeStamp = other.timeStamp;
    measurement = other.measurement;
    sensorId = other.sensorId;
    return *this;
  }

  /// \brief Default constructor.
  Measurement()
      : timeStamp(0.0) {
  }
  /**
   * @brief Constructor
   * @param timeStamp_ Measurement timestamp.
   * @param measurement_ Actual measurement.
   * @param sensorId Sensor ID (optional).
   */
  Measurement(const cgmli::Time& timeStamp_, const MEASUREMENT_T& measurement_,
              int sensorId = -1)
      : timeStamp(timeStamp_),
        measurement(measurement_),
        sensorId(sensorId) {
  }
};

/// \brief IMU measurements. For now assume they are synchronized:
struct ImuSensorReadings {
  /// \brief Default constructor.
  ImuSensorReadings()
      : gyroscopes(),
        accelerometers() {
  }
  /**
   * @brief Constructor.
   * @param gyroscopes_ Gyroscope measurement.
   * @param accelerometers_ Accelerometer measurement.
   */
  ImuSensorReadings(const Eigen::Matrix<double,3,1,Eigen::DontAlign>& gyroscopes_,
                    const Eigen::Matrix<double,3,1,Eigen::DontAlign>& accelerometers_)
      : gyroscopes(gyroscopes_),
        accelerometers(accelerometers_) {
  }
  Eigen::Matrix<double,3,1,Eigen::DontAlign> gyroscopes;     ///< Gyroscope measurement.
  Eigen::Matrix<double,3,1,Eigen::DontAlign> accelerometers; ///< Accelerometer measurement.
};

/// \brief LiDAR measurements. For now assume they are synchronized:
struct LidarSensorReadings {
    /// \brief Default constructor.
    LidarSensorReadings()
            : rayMeasurement(),
              intensity() {
    }
    /**
     * @brief Constructor.
     * @param rayMeasurement_ lidar measurement.
     * @param intensity_ intensity measurement.
     */
    LidarSensorReadings(const Eigen::Matrix<double,3,1,Eigen::Unaligned>& rayMeasurement_,
                        unsigned int& intensity_)
            : rayMeasurement(rayMeasurement_),
              intensity(intensity_) {
    }
    LidarSensorReadings(const Eigen::Matrix<double,3,1,Eigen::Unaligned>& rayMeasurement_)
            : rayMeasurement(rayMeasurement_),
              intensity(0) {
    }

    Eigen::Matrix<double,3,1,Eigen::Unaligned> rayMeasurement;  ///< Ray in cartesian coordinates.
    unsigned int intensity; ///< Intensity measurement.
};

/// \brief Camera measurement.
struct CameraData {
  
  struct DeepLearningImageData{
    std::vector<int64_t> shape;
    std::shared_ptr<float> data_ptr;
  };

  cv::Mat image;  ///< Image.
  // Diagnostic provenance. The Measurement::timeStamp remains the common node
  // time; these fields must not silently enable motion compensation in a factor.
  std::optional<Time> originalImageTimestamp;
  int64_t imageTimeOffsetNs = 0; // Original midpoint minus the uncorrected bundle mean.
  cv::Mat depthImage;  ///< Image.
  cv::Mat sigmaImage; ///< 1-sigma image.
  std::vector<cv::KeyPoint> keypoints; ///< Keypoints if available.
  std::map<std::string, DeepLearningImageData> deep_learning_data;
  bool deliversKeypoints; ///< Are the keypoints delivered too?
};

// this is how we store raw measurements before more advanced filling into data structures happens:
/// \brief IMU measurement.
typedef Measurement<ImuSensorReadings> ImuMeasurement;

/// \brief IMU measurement queue.
typedef std::deque<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement> > ImuMeasurementDeque;

/// \brief Lidar measurement.
typedef Measurement<LidarSensorReadings> LidarMeasurement;

/// \brief Lidar measurement queue.
typedef std::deque<LidarMeasurement, Eigen::aligned_allocator<LidarMeasurement> > LidarMeasurementDeque;

/// \brief Camera measurement struct.
typedef Measurement<CameraData> CameraMeasurement;

/// \brief Depth measurement queue.
typedef std::deque<CameraMeasurement, Eigen::aligned_allocator<CameraMeasurement> > DepthMeasurementDeque;

/// \brief Keypoint measurement.
struct KeypointData {
  std::vector<cv::KeyPoint> keypoints;  ///< Keypoints.
  std::vector<long unsigned int> landmarkIds; ///< Associated landmark IDs.
  cv::Mat descriptors;  ///< Keypoint descriptors.
};
/// \brief Frame measurement.
struct FrameData {
  /// \brief Shared pointer to FrameData.
  typedef std::shared_ptr<cgmli::FrameData> Ptr;
  CameraData image; ///< Camera measurement, i.e., image.
  KeypointData keypoints; ///< Keypoints.
};

}  // namespace cgmli

#endif // INCLUDE_CGMLI_MEASUREMENTS_HPP_
