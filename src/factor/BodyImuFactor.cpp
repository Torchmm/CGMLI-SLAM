#include "BodyImuFactor.hpp"
#include "ImuError.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <iterator>

namespace cgmli { namespace factor {
bool validImuInterval(const ImuMeasurementDeque& samples,const Time& start,const Time& end,std::string& error,double maximumGapSeconds) {
    if (!std::isfinite(maximumGapSeconds) || maximumGapSeconds <= 0) {
        error="Invalid maximum IMU sample gap"; return false;
    }
    if(samples.size()<2 || end<=start || samples.front().timeStamp>start || samples.back().timeStamp<end) {
        error="IMU samples do not bracket a positive integration interval"; return false;
    }
    for(size_t i=0;i<samples.size();++i) {
        if(!samples[i].measurement.accelerometers.allFinite() || !samples[i].measurement.gyroscopes.allFinite()
            || (i && samples[i].timeStamp<=samples[i-1].timeStamp)) {
            error="Non-finite or non-increasing IMU samples"; return false;
        }
        if (i && samples[i-1].timeStamp < end && samples[i].timeStamp > start
            && (samples[i].timeStamp-samples[i-1].timeStamp).toSec() > maximumGapSeconds + 1e-9) {
            error="IMU sample gap exceeds configured limit inside integration interval"; return false;
        }
    }
    return true;
}
Eigen::Vector3d gyroAt(const ImuMeasurementDeque& samples,const Time& t) {
    if(samples.empty()) return Eigen::Vector3d::Zero();
    const auto next=std::lower_bound(samples.begin(),samples.end(),t,[](const auto& m,const Time& epoch){return m.timeStamp<epoch;});
    if(next==samples.begin()) return next->measurement.gyroscopes;
    if(next==samples.end()) return samples.back().measurement.gyroscopes;
    const auto before=std::prev(next); const double span=(next->timeStamp-before->timeStamp).toSec();
    if(span<=0) throw std::invalid_argument("Invalid IMU timestamp order");
    const double alpha=(t-before->timeStamp).toSec()/span;
    return (1-alpha)*before->measurement.gyroscopes+alpha*next->measurement.gyroscopes;
}
bool initializeImu(const ImuMeasurementDeque& samples,const ImuParameters& p,const Time& time,
    kinematics::Transformation& pose,SpeedAndBias& sb,std::string& error) {
    ImuMeasurementDeque interval;
    const Time beginning=time.toSec()<0.5?Time(0,0):time-Duration(0.5);
    for(const auto& m:samples) if(m.timeStamp>=beginning && m.timeStamp<=time) interval.push_back(m);
    if(interval.size()<10 || (interval.back().timeStamp-interval.front().timeStamp).toSec()<0.4) {
        error="Waiting for at least 0.4 seconds of startup IMU"; return false;
    }
    std::string reason;
    if(!validImuInterval(interval,interval.front().timeStamp,interval.back().timeStamp,reason,p.max_sample_gap)) { error=reason; return false; }
    cgmli::ceres::ImuStationarityOptions stationary;
    stationary.gravity=p.g; stationary.maximumSampleGap=p.max_sample_gap;
    stationary.accelBias=p.a0; stationary.gyroBiasPrior=p.g0;
    cgmli::ceres::ImuStationarityStatistics statistics;
    kinematics::Transformation initialized;
    if(!cgmli::ceres::ImuError::initPose(interval,initialized,stationary,&statistics) || !initialized.T().allFinite()) {
        error="Stationary IMU initialization withheld: "+statistics.reason; return false;
    }
    SpeedAndBias value=SpeedAndBias::Zero(); value.segment<3>(3)=statistics.meanGyro; value.tail<3>()=p.a0;
    pose=initialized; sb=value; return true;
}
bool propagateImu(const ImuMeasurementDeque& samples,const ImuParameters& p,
    kinematics::Transformation& pose,SpeedAndBias& sb,const Time& start,const Time& end,
    Eigen::Matrix<double,15,15>* covariance,Eigen::Matrix<double,15,15>* jacobian) {
    std::string error; if(!validImuInterval(samples,start,end,error,p.max_sample_gap)) return false;
    auto nextPose=pose; auto nextSb=sb;
    Eigen::Matrix<double,15,15> cov=Eigen::Matrix<double,15,15>::Zero(),jac=cov;
    if(cgmli::ceres::ImuError::propagation(samples,p,nextPose,nextSb,start,end,covariance?&cov:nullptr,jacobian?&jac:nullptr)<=0
        || !nextPose.T().allFinite() || !nextSb.allFinite() || !cov.allFinite() || !jac.allFinite()) return false;
    pose=nextPose; sb=nextSb; if(covariance) *covariance=cov; if(jacobian) *jacobian=jac; return true;
}
namespace {
using Vector16=Eigen::Matrix<double,16,1>;
using Matrix16=Eigen::Matrix<double,16,16>;
using J7=Eigen::Matrix<double,15,7,Eigen::RowMajor>;
using J9=Eigen::Matrix<double,15,9,Eigen::RowMajor>;
Vector16 sensorCoordinates(const Vector16& x,const kinematics::Transformation& T_BS,const Eigen::Vector3d& rawGyro) {
    const kinematics::Transformation body(x.head<3>(),Eigen::Quaterniond(x[6],x[3],x[4],x[5]).normalized());
    const auto sensor=body*T_BS;
    const Eigen::Vector3d omegaS=rawGyro-x.segment<3>(10);
    Vector16 y; y.head<3>()=sensor.r(); y.segment<4>(3)=sensor.q().coeffs(); y.tail<9>()=x.tail<9>();
    y.segment<3>(7)+=(sensor.C()*omegaS).cross(body.C()*T_BS.r());
    return y;
}
class FrozenImu final : public cgmli::ceres::ImuError {
public:
    void exportData(backend::detail::FrozenImuData& out) const {
        const auto copy=[](const auto& value,double* target) { std::copy_n(value.data(),value.size(),target); };
        out.duration=(t1_-t0_).toSec(); out.gravity=imuParameters_.g;
        copy(Delta_q_.coeffs(),out.deltaQuaternion);
        copy(acc_integral_,out.deltaVelocity); copy(acc_doubleintegral_,out.deltaPosition);
        copy(speedAndBiases_ref_.tail<6>(),out.referenceBias);
        copy(dalpha_db_g_,out.rotationGyro); copy(dv_db_g_,out.velocityGyro); copy(dp_db_g_,out.positionGyro);
        copy(C_integral_,out.velocityAccel); copy(C_doubleintegral_,out.positionAccel);
        copy(squareRootInformation_,out.whitening);
    }
    const SpeedAndBias& reference() const { return speedAndBiases_ref_; }
    const Eigen::Matrix<double,15,15>& covariance() const { return P_delta_; }
    Eigen::Matrix<double,15,1> unwhiten(const Eigen::Matrix<double,15,1>& r) const {
        return squareRootInformation_.fullPivLu().solve(r);
    }
    FrozenImu(const ImuMeasurementDeque& m,const ImuParameters& p,const Time& start,const Time& end,const Vector16& x)
        : ImuError(m,p,start,end) {
        const kinematics::Transformation pose(x.head<3>(),Eigen::Quaterniond(x[6],x[3],x[4],x[5]));
        if(redoPreintegration(pose,x.tail<9>())<=0 || !squareRootInformation_.allFinite())
            throw std::invalid_argument("IMU preintegration failed");
        fixedLinearization_=true; redo_=false; redoCounter_=1;
    }
    FrozenImu(const FrozenImu& previous,const ImuMeasurementDeque& samples,const Time& end) : ImuError() {
        syncFrom(previous);
        if(end<t1_) throw std::invalid_argument("incremental IMU endpoint moved backwards");
        if(end>t1_) {
            auto begin=std::lower_bound(samples.begin(),samples.end(),t1_,[](const auto& m,const Time& t){return m.timeStamp<t;});
            if(begin!=samples.begin()) --begin;
            const ImuMeasurementDeque tail(begin,samples.end());
            if(append(kinematics::Transformation(),speedAndBiases_ref_,tail,end)<=0 || !squareRootInformation_.allFinite())
                throw std::invalid_argument("incremental IMU append failed");
        }
        fixedLinearization_=true; redo_=false;
    }
};
class BodyImuCost final : public ::ceres::SizedCostFunction<15,7,9,7,9> {
public:
    void exportData(backend::detail::FrozenImuData& out) const {
        cost_->exportData(out);
        const auto q=T_BS_.q().coeffs(); const auto p=T_BS_.r();
        std::copy_n(q.data(),4,out.sensorQuaternion); std::copy_n(p.data(),3,out.sensorPosition);
        for(int i=0;i<2;++i) std::copy_n(gyro_[i].data(),3,out.gyros+3*i);
    }
    bool needsRefresh(const SpeedAndBias& sb,double gyro,double accel) const {
        const auto delta=sb-cost_->reference();
        return delta.segment<3>(3).norm()>gyro || delta.tail<3>().norm()>accel;
    }
    bool diagnose(double const* const* x,BodyImuDiagnostics& out) const {
        Eigen::Matrix<double,15,1> r;
        if(!Evaluate(x,r.data(),nullptr)) return false;
        out.physical=cost_->unwhiten(r); out.nis=r.squaredNorm();
        out.covariance=cost_->covariance();
        out.referenceBias=cost_->reference().tail<6>();
        const SpeedAndBias delta=Eigen::Map<const SpeedAndBias>(x[1])-cost_->reference();
        out.deltaGyroBias=delta.segment<3>(3).norm(); out.deltaAccelBias=delta.tail<3>().norm();
        out.duration=(cost_->t1()-cost_->t0()).toSec();
        return out.physical.allFinite() && std::isfinite(out.nis);
    }
    BodyImuCost(const ImuMeasurementDeque& m,const ImuParameters& p,
        const backend::StateEstimate& first,const backend::StateEstimate& second,
        const BodyImuCost* previous=nullptr,bool* appended=nullptr)
        : T_BS_(p.T_BS),gyro_{gyroAt(m,first.timestamp),gyroAt(m,second.timestamp)} {
        if(appended) *appended=false;
        Vector16 x; x.head<3>()=first.T_WB.r(); x.segment<4>(3)=first.T_WB.q().coeffs(); x.tail<9>()=first.speedBias;
        const auto sensor=sensorCoordinates(x,T_BS_,gyro_[0]);
        auto boundary=std::upper_bound(m.begin(),m.end(),second.timestamp,[](const Time& t,const auto& sample){return t<sample.timeStamp;});
        if(boundary==m.begin()) throw std::invalid_argument("missing IMU prefix boundary");
        --boundary;
        // Cache ONLY complete IMU sample intervals. Appending after a partial
        // image-time tail would change midpoint integration and its covariance.
        // Rebuild that final fractional interval from this immutable prefix.
        const Time prefixEnd=boundary->timeStamp;
        if(prefixEnd>first.timestamp) {
            const bool reuse=previous && previous->prefix_ && previous->prefix_->t0()==first.timestamp
                && previous->prefix_->t1()<=prefixEnd && !previous->needsRefresh(first.speedBias,0.0003,0.02);
            if(reuse) {
                prefix_=std::make_shared<FrozenImu>(*previous->prefix_,m,prefixEnd);
                if(appended) *appended=true;
            } else prefix_=std::make_shared<FrozenImu>(m,p,first.timestamp,prefixEnd,sensor);
            cost_=prefixEnd==second.timestamp?prefix_:std::make_shared<FrozenImu>(*prefix_,m,second.timestamp);
        } else cost_=std::make_shared<FrozenImu>(m,p,first.timestamp,second.timestamp,sensor);
    }
    bool Evaluate(double const* const* parameters,double* residuals,double** jacobians) const override {
        std::array<Vector16,2> body,sensor;
        for(int i=0;i<2;++i) {
            body[i].head<7>()=Eigen::Map<const Eigen::Matrix<double,7,1>>(parameters[2*i]);
            body[i].tail<9>()=Eigen::Map<const SpeedAndBias>(parameters[2*i+1]);
            if(!body[i].allFinite() || body[i].segment<4>(3).norm()<1e-8) return false;
            sensor[i]=sensorCoordinates(body[i],T_BS_,gyro_[i]);
        }
        const double* inputs[]{sensor[0].data(),sensor[0].data()+7,sensor[1].data(),sensor[1].data()+7};
        J7 poseJac[2]; J9 sbJac[2];
        double* sensorJac[]{poseJac[0].data(),sbJac[0].data(),poseJac[1].data(),sbJac[1].data()};
        if(!cost_->Evaluate(inputs,residuals,jacobians?sensorJac:nullptr)
            || !Eigen::Map<const Eigen::Matrix<double,15,1>>(residuals).allFinite()) return false;
        if(jacobians) for(int i=0;i<2;++i) {
            if(!jacobians[2*i] && !jacobians[2*i+1]) continue;
            // Differentiate only the inexpensive coordinate map; preintegration and
            // its analytic whitened Jacobians are evaluated once, not per perturbation.
            Matrix16 chain;
            for(int column=0;column<16;++column) {
                auto plus=body[i],minus=body[i]; const double h=1e-6*std::max(1.0,std::abs(body[i][column]));
                plus[column]+=h; minus[column]-=h;
                chain.col(column)=(sensorCoordinates(plus,T_BS_,gyro_[i])-sensorCoordinates(minus,T_BS_,gyro_[i]))/(2*h);
            }
            Eigen::Matrix<double,15,16> sensorDerivative; sensorDerivative<<poseJac[i],sbJac[i];
            const Eigen::Matrix<double,15,16> result=sensorDerivative*chain;
            if(!result.allFinite()) return false;
            if(jacobians[2*i]) { Eigen::Map<J7> out(jacobians[2*i]); out=result.leftCols<7>(); }
            if(jacobians[2*i+1]) { Eigen::Map<J9> out(jacobians[2*i+1]); out=result.rightCols<9>(); }
        }
        return true;
    }
private:
    kinematics::Transformation T_BS_;
    std::array<Eigen::Vector3d,2> gyro_;
    std::shared_ptr<const FrozenImu> cost_,prefix_;
};
class SuppressedImuCost final : public ::ceres::SizedCostFunction<15,7,9,7,9> {
    bool Evaluate(double const* const*,double*,double**) const override { return false; }
};
}
bool diagnoseBodyImu(const ::ceres::CostFunction& cost,double const* const* x,BodyImuDiagnostics& out) {
    const auto native=dynamic_cast<const BodyImuCost*>(&cost); return native && native->diagnose(x,out);
}
bool exportFrozenImu(const ::ceres::CostFunction& cost,backend::detail::FrozenImuData& out) {
    const auto* native=dynamic_cast<const BodyImuCost*>(&cost);
    if(!native) return false;
    native->exportData(out);
    return true;
}
bool checkBodyImuJacobians(const ::ceres::CostFunction& cost,double const* const* x,BodyImuJacobianCheck& out) {
    out=BodyImuJacobianCheck();
    if(!dynamic_cast<const BodyImuCost*>(&cost)) return false;
    J7 poseJac[2]; J9 sbJac[2];
    double* jac[]{poseJac[0].data(),sbJac[0].data(),poseJac[1].data(),sbJac[1].data()};
    Eigen::Matrix<double,15,1> residual;
    if(!cost.Evaluate(x,residual.data(),jac)) return false;
    const backend::LeftSE3Manifold manifold;
    for(int block=0;block<4;++block) {
        const bool isPose=block%2==0; const int tangent=isPose?6:9,ambient=isPose?7:9;
        Eigen::MatrixXd analytic(15,tangent);
        if(isPose) {
            Eigen::Matrix<double,7,6,Eigen::RowMajor> lift;
            if(!manifold.PlusJacobian(x[block],lift.data())) return false;
            analytic=poseJac[block/2]*lift;
        } else analytic=sbJac[block/2];
        if(!analytic.allFinite()) return false;
        for(int column=0;column<tangent;++column) {
            const double h=1e-6;
            Eigen::VectorXd plus=Eigen::Map<const Eigen::VectorXd>(x[block],ambient),minus=plus;
            if(isPose) {
                Eigen::Matrix<double,6,1> delta=Eigen::Matrix<double,6,1>::Zero(); delta[column]=h;
                if(!manifold.Plus(x[block],delta.data(),plus.data())) return false;
                delta[column]=-h; if(!manifold.Plus(x[block],delta.data(),minus.data())) return false;
            } else { plus[column]+=h; minus[column]-=h; }
            const double* parameters[]{x[0],x[1],x[2],x[3]};
            Eigen::Matrix<double,15,1> rp,rm;
            parameters[block]=plus.data(); if(!cost.Evaluate(parameters,rp.data(),nullptr)) return false;
            parameters[block]=minus.data(); if(!cost.Evaluate(parameters,rm.data(),nullptr)) return false;
            const Eigen::Matrix<double,15,1> numeric=(rp-rm)/(2*h);
            if(!numeric.allFinite()) return false;
            const double absolute=(analytic.col(column)-numeric).norm();
            const double scaled=absolute/(1e-6+1e-6*numeric.norm());
            ++out.columns; if(scaled>1) ++out.failedColumns;
            out.maxAbsoluteError=std::max(out.maxAbsoluteError,absolute);
            if(scaled>out.maxScaledError) { out.maxScaledError=scaled; out.worstBlock=block; out.worstColumn=column; }
        }
    }
    return true;
}
bool bodyImuNeedsRefresh(const ::ceres::CostFunction& cost,const SpeedAndBias& sb,double gyro,double accel) {
    const auto native=dynamic_cast<const BodyImuCost*>(&cost); return native && native->needsRefresh(sb,gyro,accel);
}
std::shared_ptr<::ceres::CostFunction> makeBodyImuFactor(const ImuMeasurementDeque& samples,const ImuParameters& p,
    const backend::StateEstimate& first,const backend::StateEstimate& second) {
    std::string error;
    if(!validImuInterval(samples,first.timestamp,second.timestamp,error,p.max_sample_gap)) throw std::invalid_argument(error);
    return std::make_shared<BodyImuCost>(samples,p,first,second);
}
backend::FactorRecord makeBodyImuRecord(const ImuMeasurementDeque& samples,const ImuParameters& p,
    const backend::StateEstimate& first,const backend::StateEstimate& second,double maximumDuration) {
    if(!std::isfinite(maximumDuration) || maximumDuration<=0) throw std::invalid_argument("invalid IMU duration policy");
    std::string error;
    if(!validImuInterval(samples,first.timestamp,second.timestamp,error,p.max_sample_gap)) throw std::invalid_argument(error);
    auto replay=std::make_shared<backend::ImuFactorReplay>();
    replay->parameters=std::make_shared<const ImuParameters>(p);
    replay->start=first.timestamp; replay->end=second.timestamp;
    // Retain only the samples bracketing this edge, not the startup/history buffer.
    auto begin=std::lower_bound(samples.begin(),samples.end(),first.timestamp,
        [](const auto& m,const Time& t) { return m.timeStamp<t; });
    if(begin!=samples.begin()) --begin;
    auto end=std::lower_bound(begin,samples.end(),second.timestamp,
        [](const auto& m,const Time& t) { return m.timeStamp<t; });
    if(end!=samples.end()) ++end;
    replay->samples.assign(begin,end);
    backend::FactorRecord record; record.kind=backend::FactorKind::Imu;
    record.epochs={first.id,second.id};
    record.keys={backend::BlockKey::pose(first.id),backend::BlockKey::speedBias(first.id),
        backend::BlockKey::pose(second.id),backend::BlockKey::speedBias(second.id)};
    record.solveEnabled=(second.timestamp-first.timestamp).toSec()<=maximumDuration;
    record.cost=record.solveEnabled?makeBodyImuFactor(replay->samples,p,first,second):std::make_shared<SuppressedImuCost>();
    if(!record.solveEnabled) { replay->integrationMode="suppressed_duration"; record.reason="IMU interval exceeds maximum duration"; }
    record.imuReplay=std::move(replay); return record;
}
void configureBodyImuDuration(backend::FactorRecord& record,double maximumDuration,
    const backend::StateEstimate& first,const backend::StateEstimate& last) {
    if(!record.imuReplay) return;
    if(!std::isfinite(maximumDuration) || maximumDuration<=0) throw std::invalid_argument("invalid IMU duration policy");
    if(record.imuReplay->start!=first.timestamp || record.imuReplay->end!=last.timestamp)
        throw std::invalid_argument("IMU interval metadata does not match its endpoint epochs");
    const bool enabled=(record.imuReplay->end-record.imuReplay->start).toSec()<=maximumDuration;
    if(!enabled) {
        record.solveEnabled=false; record.cost=std::make_shared<SuppressedImuCost>();
        record.reason="IMU interval exceeds maximum duration";
    } else if(!record.solveEnabled) {
        if(!record.imuReplay->completeSamples) throw std::invalid_argument("cannot reactivate a pruned IMU interval");
        record.cost=makeBodyImuFactor(record.imuReplay->samples,*record.imuReplay->parameters,first,last);
        record.solveEnabled=true; record.reason.clear();
    }
}
bool mergeBodyImuRecords(const backend::FactorRecord& left,const backend::FactorRecord& right,
    const backend::StateEstimate& first,const backend::StateEstimate& last,size_t limit,
    backend::FactorRecord& out,std::string& error,double maximumDuration,bool incremental) {
    if(limit<4 || !std::isfinite(maximumDuration) || maximumDuration<=0) {
        error="invalid IMU merge storage/duration policy"; return false;
    }
    if(!left.imuReplay || !right.imuReplay || !left.imuReplay->parameters || !right.imuReplay->parameters
        || left.loss || right.loss || left.keys.size()!=4 || right.keys.size()!=4
        || left.keys[2]!=right.keys[0] || left.keys[3]!=right.keys[1]
        || left.keys[0]!=backend::BlockKey::pose(first.id) || right.keys[2]!=backend::BlockKey::pose(last.id)
        || std::isfinite(left.nisThreshold) || std::isfinite(right.nisThreshold)) {
        error="IMU merge requires adjacent native ungated/unrobustified edges"; return false;
    }
    const StateId middle(left.keys[2].id);
    const std::vector<backend::BlockKey> expectedLeft{backend::BlockKey::pose(first.id),backend::BlockKey::speedBias(first.id),
        backend::BlockKey::pose(middle),backend::BlockKey::speedBias(middle)};
    const std::vector<backend::BlockKey> expectedRight{backend::BlockKey::pose(middle),backend::BlockKey::speedBias(middle),
        backend::BlockKey::pose(last.id),backend::BlockKey::speedBias(last.id)};
    if(left.kind!=backend::FactorKind::Imu || right.kind!=backend::FactorKind::Imu
        || left.keys!=expectedLeft || right.keys!=expectedRight || !(first.id<middle && middle<last.id)
        || left.imuReplay->start!=first.timestamp || right.imuReplay->end!=last.timestamp
        || left.imuReplay->end!=right.imuReplay->start) {
        error="IMU merge endpoint/layout mismatch"; return false;
    }
    const auto& a=*left.imuReplay->parameters; const auto& b=*right.imuReplay->parameters;
    // These are the parameters used by BodyImuCost/FrozenImu. A calibration or
    // noise change must never silently reinterpret one half of the interval.
    if((a.T_BS.T()-b.T_BS.T()).norm()>1e-12 || a.g!=b.g || a.a_max!=b.a_max || a.g_max!=b.g_max
        || a.sigma_g_c!=b.sigma_g_c || a.sigma_a_c!=b.sigma_a_c
        || a.sigma_gw_c!=b.sigma_gw_c || a.sigma_aw_c!=b.sigma_aw_c || a.max_sample_gap!=b.max_sample_gap) {
        error="IMU merge calibration/noise mismatch"; return false;
    }
    const bool suppressed=(last.timestamp-first.timestamp).toSec()>maximumDuration;
    if(!suppressed && (!left.imuReplay->completeSamples || !right.imuReplay->completeSamples)) {
        error="cannot reintegrate an incomplete IMU interval"; return false;
    }
    ImuMeasurementDeque merged;
    auto i=left.imuReplay->samples.begin(),j=right.imuReplay->samples.begin();
    const auto ie=left.imuReplay->samples.end(),je=right.imuReplay->samples.end();
    while(i!=ie || j!=je) {
        if(i!=ie && j!=je && i->timeStamp==j->timeStamp) {
            if((i->measurement.accelerometers-j->measurement.accelerometers).norm()>1e-12
                || (i->measurement.gyroscopes-j->measurement.gyroscopes).norm()>1e-12) {
                error="conflicting IMU samples at shared timestamp"; return false;
            }
            merged.push_back(*i++); ++j;
        } else if(j==je || (i!=ie && i->timeStamp<j->timeStamp)) merged.push_back(*i++);
        else merged.push_back(*j++);
        if(!suppressed && merged.size()>limit) { error="merged IMU interval exceeds sample budget"; return false; }
    }
    try {
        if(suppressed) {
            auto replay=std::make_shared<backend::ImuFactorReplay>(); replay->parameters=left.imuReplay->parameters;
            replay->start=first.timestamp; replay->end=last.timestamp; replay->integrationMode="suppressed_duration";
            replay->completeSamples=left.imuReplay->completeSamples && right.imuReplay->completeSamples && merged.size()<=limit;
            // Once a suppressed interval exceeds the raw storage budget, retain
            // endpoint brackets only. It cannot become short again or enter a
            // numerical factor; log completeness explicitly rather than bridge it.
            if(!replay->completeSamples && merged.size()>4) {
                ImuMeasurementDeque brackets{merged[0],merged[1],merged[merged.size()-2],merged.back()};
                merged=std::move(brackets);
            }
            replay->samples=std::move(merged); out=backend::FactorRecord(); out.kind=backend::FactorKind::Imu;
            out.epochs={first.id,last.id}; out.keys={backend::BlockKey::pose(first.id),backend::BlockKey::speedBias(first.id),
                backend::BlockKey::pose(last.id),backend::BlockKey::speedBias(last.id)};
            out.imuReplay=replay; out.solveEnabled=false; out.cost=std::make_shared<SuppressedImuCost>();
            out.reason="IMU interval exceeds maximum duration";
        } else {
            if(!validImuInterval(merged,first.timestamp,last.timestamp,error,a.max_sample_gap)) return false;
            auto replay=std::make_shared<backend::ImuFactorReplay>(); replay->parameters=left.imuReplay->parameters;
            replay->start=first.timestamp; replay->end=last.timestamp; replay->samples=std::move(merged);
            const auto old=incremental?dynamic_cast<const BodyImuCost*>(left.cost.get()):nullptr;
            bool appended=false;
            out=backend::FactorRecord(); out.kind=backend::FactorKind::Imu; out.epochs={first.id,last.id};
            out.keys={backend::BlockKey::pose(first.id),backend::BlockKey::speedBias(first.id),
                backend::BlockKey::pose(last.id),backend::BlockKey::speedBias(last.id)};
            out.cost=std::make_shared<BodyImuCost>(replay->samples,a,first,last,old,&appended);
            replay->integrationMode=appended?"incremental":"full"; out.imuReplay=std::move(replay);
        }
    }
    catch(const std::exception& e) { error=e.what(); return false; }
    return true;
}
} }
