
#ifndef INCLUDE_CGMLI_VIPARAMETERSREADER_HPP_
#define INCLUDE_CGMLI_VIPARAMETERSREADER_HPP_

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>
#pragma GCC diagnostic pop
#include <opencv2/calib3d.hpp>

#include <utils/assert_macros.hpp>
#include <common/Parameters.hpp>
#include <cameras/NCameraSystem.hpp>

/// \brief cgmli Main namespace of this package.
namespace cgmli {

/**
 * @brief This class reads and parses config file.
 */
class ViParametersReader{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CGMLI_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The default constructor.
  ViParametersReader();

  /**
   * @brief The constructor. This calls readConfigFile().
   * @param filename Configuration filename.
   */
  ViParametersReader(const std::string& filename);

  /// @brief Trivial destructor.
  ~ViParametersReader() = default;

  /**
   * @brief Read and parse a config file.
   *        To get the result call getParameters().
   * @param filename Configuration filename.
   */
  void readConfigFile(const std::string& filename);

  /**
   * @brief Get parameters.
   * @param[out] parameters A copy of the parameters.
   * @return True if parameters have been read from a configuration file. If it
   *         returns false then the variable \e parameters has not been changed.
   */
  bool getParameters(cgmli::ViParameters& parameters) const{
    if(readConfigFile_)
      parameters = viParameters_;
    return readConfigFile_;
  }

 protected:

  /// @brief Struct that contains all the camera calibration information.
  struct CameraCalibration {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    cgmli::kinematics::Transformation T_SC;   ///< Transformation from camera to sensor (IMU) frame.
    Eigen::Vector2i imageDimension;           ///< Image dimension. [pixels]
    Eigen::VectorXd distortionCoefficients;   ///< Distortion Coefficients.
    Eigen::Vector2d focalLength;              ///< Focal length.
    Eigen::Vector2d principalPoint;           ///< Principal point.
    std::string cameraModel;                  ///< camera model. ('pinhole' 'eucm')
    Eigen::Vector2d eucmParameters;           ///< alpha, beta eucm parameters

    /// \brief Distortion type. ('radialtangential' 'radialtangential8' 'equdistant')
    std::string distortionType;

    cameras::NCameraSystem::CameraType cameraType; ///< Some additional info about the camera.
  };

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode &file, std::string name, int& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode& file, std::string name, double& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode &file, std::string name, bool& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode& file, std::string name, Eigen::Matrix4d& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode &file, std::string name, Eigen::Vector3d& readValue);

  /// \brief Parse yaml file entry.
  /// @param[in] file The yaml file node.
  /// @param[in] name The variable name to be read.
  /// @param[out] readValue The read variable value.
  static void parseEntry(const cv::FileNode &file, std::string name, std::string& readValue);

  bool readConfigFile_; ///< If readConfigFile() has been called at least once this is true.
  cgmli::ViParameters viParameters_;   ///< The parameters.

  /**
   * @brief Get the camera calibration. This looks for the calibration in the
   *        configuration file first. If this fails it will directly get the calibration
   *        from the sensor, if useDriver is set to true.
   * @remark Overload this function if you want to get the calibrations via ROS for example.
   * @param calibrations The calibrations.
   * @param configurationFile The config file.
   * @return True if reading of the calibration was successful.
   */
  bool getCameraCalibration(
      std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
      cv::FileStorage& configurationFile);

  /**
   * @brief Get the camera calibration via the configuration file.
   * @param[out] calibrations Read calibration.
   * @param[in] cameraNode File node pointing to the cameras sequence.
   * @return True if reading and parsing of calibration was successful.
   */
  bool getCalibrationViaConfig(
      std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
      cv::FileNode cameraNode) const;

  /**
   * @brief Get the LiDAR calibration via the configuration file
   * @param[in] cameraNode File node pointing to the LiDAR.
   * @param[out] lidarParameters Read lidar calibration.
   * @return True if reading and parsing of calibration was successful.
   */
  bool getLiDARCalibration(const cv::FileNode& calibrationNode, cgmli::LidarParameters& lidarParameters);


};

}

#endif /* INCLUDE_CGMLI_VIOPARAMETERSREADER_HPP_ */
