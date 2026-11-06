
#ifndef INCLUDE_CGMLI_CERES_IMUERROR_HPP_
#define INCLUDE_CGMLI_CERES_IMUERROR_HPP_

#include <vector>
#include <mutex>
#include <atomic>
#include <string>

#include <ceres/sized_cost_function.h>
#include <common/FrameTypedefs.hpp>
#include <utils/Time.h>
#include <utils/assert_macros.hpp>
#include <common/Measurements.hpp>
#include <common/Parameters.hpp>
#include <factor/ErrorInterface.hpp>

/// \brief cgmli Main namespace of this package.
namespace cgmli {
/// \brief ceres Namespace for ceres-related functionality implemented in cgmli.
namespace ceres {

/// Stationary-start engineering thresholds, expressed in physical sensor units.
/// sqrt(trace(sample covariance)) is the RMS vector fluctuation, not a
/// covariance of the mean or the optimizer's residual covariance.
struct ImuStationarityOptions {
  size_t minimumSamples=10;
  double minimumDuration=0.4,maximumSampleGap=0.05;
  double maximumAccelStd=0.5,maximumGyroStd=0.05,maximumMeanAngularSpeed=0.05;
  double gravity=9.81,maximumGravityError=0.7;
  Eigen::Vector3d accelBias=Eigen::Vector3d::Zero(),gyroBiasPrior=Eigen::Vector3d::Zero();
};
struct ImuStationarityStatistics {
  Eigen::Vector3d meanAccel=Eigen::Vector3d::Zero(),meanGyro=Eigen::Vector3d::Zero();
  Eigen::Matrix3d accelCovariance=Eigen::Matrix3d::Zero(),gyroCovariance=Eigen::Matrix3d::Zero();
  double accelStd=0,gyroStd=0,duration=0,meanAngularSpeed=0,gravityError=0;
  size_t samples=0;
  std::string reason;
};

/// \brief Implements a nonlinear IMU factor.
class ImuErrorBase : public ::ceres::SizedCostFunction<
                       15 /* number of residuals */,
                       7 /* size of first parameter (PoseParameterBlock k) */,
                       9 /* size of second parameter (SpeedAndBiasParameterBlock k) */,
                       7 /* size of third parameter (PoseParameterBlock k+1) */,
                       9 /* size of fourth parameter (SpeedAndBiasParameterBlock k+1) */>,
                     public ErrorInterface
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CGMLI_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// \brief The base in ceres we derive from
  typedef ::ceres::SizedCostFunction<15, 7, 9, 7, 9> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 15;

  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 15, 15> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  typedef Eigen::Matrix<double, 15, 15> jacobian_t;

  /// \brief The type of the Jacobian w.r.t. poses --
  /// \warning This is w.r.t. minimal tangential space coordinates...
  typedef Eigen::Matrix<double, 15, 7> jacobian0_t;

  /// \brief The type of Jacobian w.r.t. Speed and biases
  typedef Eigen::Matrix<double, 15, 9> jacobian1_t;

  /// \brief Trivial destructor.
  virtual ~ImuErrorBase() override = default;

  /// \brief (Re)set the start time.
  /// \@param[in] t_0 Start time.
  void setT0(const cgmli::Time &t_0) { t0_ = t_0; }

  /// \brief (Re)set the start time.
  /// \@param[in] t_1 End time.
  void setT1(const cgmli::Time &t_1) { t1_ = t_1; }

  // getters

  /// \brief Get the start time.
  cgmli::Time t0() const { return t0_; }

  /// \brief Get the end time.
  cgmli::Time t1() const { return t1_; }

  /// \brief Get a clone of this.
  /// \return A clone as std::shared_ptr.
  virtual std::shared_ptr<ImuErrorBase> clone() const = 0;

  // sizes
  /// \brief Residual dimension.
  int residualDim() const override final {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  virtual int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  int parameterBlockDim(int parameterBlockId) const override final {
    return int(base_t::parameter_block_sizes().at(size_t(parameterBlockId)));
  }

  protected:
  // times
  cgmli::Time t0_; ///< The start time (i.e. time of the first set of states).
  cgmli::Time t1_; ///< The end time (i.e. time of the sedond set of states).
};

/// \brief Implements a nonlinear IMU factor.
class ImuError :
    public ImuErrorBase {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CGMLI_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief If switched to true, all error terms will be always re-doing propagation (very slow).
  static std::atomic_bool redoPropagationAlways;

  /// \brief Default constructor -- assumes information recomputation.
  ImuError() = default;

  /// \brief Trivial destructor.
  virtual ~ImuError() override = default;

  /// \brief Construct with measurements and parameters.
  /// \@param[in] imuMeasurements All the IMU measurements.
  /// \@param[in] imuParameters The parameters to be used.
  /// \@param[in] t_0 Start time.
  /// \@param[in] t_1 End time.
  ImuError(const cgmli::ImuMeasurementDeque & imuMeasurements,
           const cgmli::ImuParameters & imuParameters, const cgmli::Time& t_0,
           const cgmli::Time& t_1);

  /// \brief Append with measurements and parameters.
  /// \@param[in] imuMeasurements All the IMU measurements.
  /// \@param[in] t_1 End time.
  int append(const cgmli::kinematics::Transformation& T_WS,
             const cgmli::SpeedAndBias & speedAndBiases,
             const cgmli::ImuMeasurementDeque & imuMeasurements,
             const cgmli::Time& t_1);

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @remark This can be used externally to perform propagation
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @return Number of integration steps.
   */
  static int propagation(const cgmli::ImuMeasurementDeque & imuMeasurements,
                         const cgmli::ImuParameters & imuParams,
                         cgmli::kinematics::Transformation& T_WS,
                         cgmli::SpeedAndBias & speedAndBiases,
                         const cgmli::Time& t_start, const cgmli::Time& t_end,
                         covariance_t* covariance = nullptr,
                         jacobian_t* jacobian = nullptr);

  /// Initialize gravity alignment ONLY after the complete interval passes
  /// stationarity, finite-value and sampling checks. Failure preserves T_WS.
  /// IMU-only stationarity cannot distinguish rest from constant linear velocity.
  static bool initPose(const cgmli::ImuMeasurementDeque& imuMeasurements,
                       cgmli::kinematics::Transformation& T_WS,
                       const ImuStationarityOptions& options = ImuStationarityOptions(),
                       ImuStationarityStatistics* statistics = nullptr);

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @warning This is not actually const, since the re-propagation must somehow be stored...
   * @param[in] T_WS Start pose.
   * @param[in] speedAndBiases Start speed and biases.
   * @return Number of integration steps.
   */
  int redoPreintegration(const cgmli::kinematics::Transformation& T_WS,
                         const cgmli::SpeedAndBias & speedAndBiases) const;

  /// \brief Synchronise from other IMU error.
  /// \param other The other IMU error.
  void syncFrom(const ImuError & other);

  // setters

  /// \brief (Re)set the parameters.
  /// \@param[in] imuParameters The parameters to be used.
  void setImuParameters(const cgmli::ImuParameters& imuParameters);

  /// \brief (Re)set the measurements
  /// \@param[in] imuMeasurements All the IMU measurements.
  void setImuMeasurements(const cgmli::ImuMeasurementDeque& imuMeasurements) {
    imuMeasurements_ = imuMeasurements;
  }

  // getters
  /// \brief Get a clone of this.
  /// \return A clone as std::shared_ptr.
  virtual std::shared_ptr<ImuErrorBase> clone() const override final;

  /// \brief Get the IMU Parameters.
  /// \return the IMU parameters.
  const cgmli::ImuParameters& imuParameters() const {
    return imuParameters_;
  }

  /// \brief Get the IMU measurements.
  const cgmli::ImuMeasurementDeque& imuMeasurements() const {
    return imuMeasurements_;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  /**
   * @brief This evaluates the cost and additionally computes
   *        the gradient and Hessian in the minimal internal representation.
   * @param[in] parameters Pointer to the parameters (see ceres)
   * @param[out] cost Pointer to the scalar cost.
   * @param[out] gradient Pointer to the gradient.
   * @param[out] hessian Pointer to the Hessian.
   * @return Success of the evaluation.
   */
  bool EvaluateWithSigmaGradientAndHessian(double const* const* parameters,
                                            double* cost, double* gradient, double* hessian) const;

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const override final {
    return "ImuError";
  }

 protected:
  // parameters
  cgmli::ImuParameters imuParameters_; ///< The IMU parameters.

  // measurements
  cgmli::ImuMeasurementDeque imuMeasurements_; ///< The IMU measurements. used. Must span t0_ - t1_.

  // preintegration stuff. the mutable is a TERRIBLE HACK, but what can I do.
  mutable std::mutex preintegrationMutex_; ///< Protect access of intermediate results.
  // increments (initialise with identity)
  mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0); ///< Intermediate result
  mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero(); ///< Intermediate result
  mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero(); ///< Intermediate result

  // cross matrix accumulatrion
  mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  // sub-Jacobians
  mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  /// \brief The Jacobian of the increment (w/o biases).
  mutable Eigen::Matrix<double,15,15> P_delta_ = Eigen::Matrix<double,15,15>::Zero();

  /// \brief Reference biases that are updated when called redoPreintegration.
  mutable SpeedAndBias speedAndBiases_ref_ = SpeedAndBias::Zero();

  mutable bool redo_ = true; ///< Keeps track of whether or not redoPreintegration() is needed.
  // Transactional factors freeze their preintegration before sharing the cost
  // between graph checkpoints. Ordinary legacy users retain adaptive integration.
  bool fixedLinearization_ = false;
  mutable int redoCounter_ = 0; ///< Counts the number of preintegrations for statistics.

  // information matrix and its square root
  mutable information_t information_; ///< The information matrix for this error term.
  mutable information_t squareRootInformation_; ///< The square root information of this error term.

  /// \brief For gradient/hessian w.r.t. the sigmas.
  mutable AlignedVector<Eigen::Matrix<double,15,15>> dPdsigma_;

};

/// \brief Implements a linear pseudo-IMU factor as a constant velocity / orientation prior.
class PseudoImuError : public ImuErrorBase {
  public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CGMLI_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Default constructor -- assumes information recomputation.
  PseudoImuError() = default;

  /// \brief Construct with times.
  /// \@param[in] t_0 Start time.
  /// \@param[in] t_1 End time.
  PseudoImuError(const cgmli::Time& t_0, const cgmli::Time& t_1);

  /// \brief Trivial destructor.
  virtual ~PseudoImuError() override = default;

  /**
   * @brief Propagates pose, speeds and biases with simple constant velocity / orientation prior.
   * @remark This can be used externally to perform propagation
   * @param[inout] T_WS Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @return Number of integration steps.
   */
  static int propagation(cgmli::kinematics::Transformation& T_WS,
                         cgmli::SpeedAndBias & speedAndBiases,
                         const cgmli::Time& t_start, const cgmli::Time& t_end);

  /// \brief Synchronise from other IMU error.
  /// \param other The other IMU error.
  void syncFrom(const PseudoImuError & other);

  // getters

  // getters
  /// \brief Get a clone of this.
  /// \return A clone as std::shared_ptr.
  virtual std::shared_ptr<ImuErrorBase> clone() const override final;

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const override final {
    return "PseudoImuError";
  }

  protected:

};

}  // namespace ceres
}  // namespace cgmli

#endif /* INCLUDE_CGMLI_CERES_IMUERROR_HPP_ */
