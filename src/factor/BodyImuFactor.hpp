#pragma once

#include <backend/BackendTypes.hpp>
#include <common/Parameters.hpp>
#include <common/Measurements.hpp>

namespace cgmli { namespace factor {
struct BodyImuDiagnostics {
    Eigen::Matrix<double,15,1> physical=Eigen::Matrix<double,15,1>::Zero();
    double nis=0,deltaGyroBias=0,deltaAccelBias=0,duration=0;
    Eigen::Matrix<double,6,1> referenceBias=Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,15,15> covariance=Eigen::Matrix<double,15,15>::Zero();
};
/// Read-only diagnostics; physical ordering is p, 2*quaternion-vector, v, bg, ba.
bool diagnoseBodyImu(const ::ceres::CostFunction&,double const* const*,BodyImuDiagnostics&);
struct BodyImuJacobianCheck {
    size_t columns=0,failedColumns=0;
    double maxScaledError=0,maxAbsoluteError=0;
    int worstBlock=-1,worstColumn=-1;
};
/// Central differences of the SAME immutable factor at actual accepted values.
/// Poses use the backend left-SE3 tangent, SB uses [v,bg,ba]. This validates
/// derivatives, not IMU noise calibration or the physical measurement model.
bool checkBodyImuJacobians(const ::ceres::CostFunction&,double const* const*,BodyImuJacobianCheck&);
/// Physical samples must bracket the interval, increase strictly and have no gap
/// exceeding the configured bound inside that interval. Earlier cached gaps do not
/// invalidate an unrelated later integration interval.
bool validImuInterval(const ImuMeasurementDeque&,const Time&,const Time&,std::string&,double maximumGapSeconds=0.05);
Eigen::Vector3d gyroAt(const ImuMeasurementDeque&,const Time&);
/// Stationary startup: gravity alignment, zero velocity and observed gyro bias.
/// Failure leaves outputs unchanged; moving startup waits for a stationary interval.
bool initializeImu(const ImuMeasurementDeque&,const ImuParameters&,const Time&,
    kinematics::Transformation&,SpeedAndBias&,std::string&);
/// Safe public sensor-S wrapper around the existing IMU integration implementation.
bool propagateImu(const ImuMeasurementDeque&,const ImuParameters&,
    kinematics::Transformation&,SpeedAndBias&,const Time&,const Time&,
    Eigen::Matrix<double,15,15>* covariance=nullptr,Eigen::Matrix<double,15,15>* jacobian=nullptr);
/// Native body-B cost with [pose(7), SB(9), pose(7), SB(9)] layout. The existing
/// sensor-S IMU equations are reused with pose, velocity lever arm and bias chain
/// derivatives. Preintegration/whitening are fixed before sharing the cost.
std::shared_ptr<::ceres::CostFunction> makeBodyImuFactor(const ImuMeasurementDeque&,
    const ImuParameters&,const backend::StateEstimate&,const backend::StateEstimate&);
/// Production graph record, including bounded raw samples for SECOND_NEW replay.
backend::FactorRecord makeBodyImuRecord(const ImuMeasurementDeque&,const ImuParameters&,
    const backend::StateEstimate&,const backend::StateEstimate&,double maximumDuration=10.0);
/// One duration policy for optimization, refits, Schur selection and fallback.
void configureBodyImuDuration(backend::FactorRecord&,double maximumDuration,
    const backend::StateEstimate&,const backend::StateEstimate&);
/// Replace only a live native cost at a solve boundary. Shared old costs and
/// historical priors remain immutable; returns false when no refresh is needed.
bool bodyImuNeedsRefresh(const ::ceres::CostFunction&,const SpeedAndBias&,double gyroThreshold,double accelThreshold);
/// Concatenate two adjacent native records, deduplicating their shared bracket
/// samples. Reintegrate once at the accepted left state. The caller must REMOVE
/// both original factors without marginalizing them; the new edge replaces them.
bool mergeBodyImuRecords(const backend::FactorRecord&,const backend::FactorRecord&,
    const backend::StateEstimate&,const backend::StateEstimate&,size_t sampleLimit,
    backend::FactorRecord&,std::string&,double maximumDuration=10.0,bool incremental=true);
} }
