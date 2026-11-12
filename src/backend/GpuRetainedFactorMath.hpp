#pragma once

#include "GpuRetainedFactors.hpp"
#include "VisualSphereBatchMath.hpp"

namespace cgmli::backend::detail::retained_math {

using sphere_batch::V3;
using sphere_batch::M3;
using sphere_batch::finite;
using sphere_batch::skewMatrix;
using Q=Eigen::Quaterniond;

EIGEN_DEVICE_FUNC inline Q rotationExp(const V3& angle) {
    const double a=angle.norm();
    const double s=a<1e-8 ? 0.5-a*a/48.0 : ::sin(0.5*a)/a;
    return Q(::cos(0.5*a),s*angle.x(),s*angle.y(),s*angle.z());
}

EIGEN_DEVICE_FUNC inline M3 rotationJacobian(const V3& angle) {
    const double a2=angle.squaredNorm();
    const double a=::sqrt(a2);
    const M3 w=skewMatrix(angle);
    return M3::Identity()+(a2<1e-10?0.5-a2/24:(1-::cos(a))/a2)*w
        +(a2<1e-10?1.0/6-a2/120:(a-::sin(a))/(a2*a))*w*w;
}

EIGEN_DEVICE_FUNC inline void posePlus(const double* x,const double* d,double* y,bool rotationOnly) {
    const V3 angle=Eigen::Map<const V3>(d+(rotationOnly?0:3));
    const Q q=rotationExp(angle);
    const V3 position=Eigen::Map<const V3>(x);
    Eigen::Map<V3> yp(y);
    yp=position;
    if(!rotationOnly) yp=q*position+rotationJacobian(angle)*Eigen::Map<const V3>(d);
    Eigen::Map<Q> yq(y+3);
    yq=(q*Eigen::Map<const Q>(x+3)).normalized();
}

EIGEN_DEVICE_FUNC inline void poseMinus(const double* x,const double* ref,double* d) {
    Q q=Eigen::Map<const Q>(x+3)*Eigen::Map<const Q>(ref+3).conjugate();
    q.normalize();
    if(q.w()<0) q.coeffs()*=-1;
    const double s=q.vec().norm();
    const V3 theta=(s<1e-10?2.0:2.0*::atan2(s,q.w())/s)*q.vec();
    const double a2=theta.squaredNorm(),a=::sqrt(a2);
    const M3 w=skewMatrix(theta);
    const double c=a2<1e-10?1.0/12+a2/720:(1-0.5*a/::tan(0.5*a))/a2;
    Eigen::Map<V3> p(d),r(d+3);
    p=(M3::Identity()-0.5*w+c*w*w)*(Eigen::Map<const V3>(x)-q*Eigen::Map<const V3>(ref));
    r=theta;
}

struct SensorState {
    V3 p,v,arm,lever;
    Q q;
};

EIGEN_DEVICE_FUNC inline SensorState sensorState(const FrozenImuData& imu,const double* pose,const double* sb,int end) {
    const Q q=Eigen::Map<const Q>(pose+3).normalized();
    SensorState s;
    s.q=(q*Eigen::Map<const Q>(imu.sensorQuaternion)).normalized();
    s.arm=q*Eigen::Map<const V3>(imu.sensorPosition);
    s.lever=(s.q*(Eigen::Map<const V3>(imu.gyros+3*end)-Eigen::Map<const V3>(sb+3))).cross(s.arm);
    s.p=Eigen::Map<const V3>(pose)+s.arm;
    s.v=Eigen::Map<const V3>(sb)+s.lever;
    return s;
}

// Tangents are the backend's left SE(3) pose and additive world velocity/bias.
// The lever-arm derivative is analytic, including its coupling to gyro bias.
EIGEN_DEVICE_FUNC inline bool evaluateImu(const GpuImuFactor& f,const double* x,int columns,double* r,double* jacobian) {
    const auto& imu=f.imu;
    const auto* sb0=x+f.blocks[1].offset;
    const auto* sb1=x+f.blocks[3].offset;
    const auto s0=sensorState(imu,x+f.blocks[0].offset,sb0,0);
    const auto s1=sensorState(imu,x+f.blocks[2].offset,sb1,1);
    const Eigen::Map<const M3> A(imu.rotationGyro),V(imu.velocityGyro),P(imu.positionGyro),
        Va(imu.velocityAccel),Pa(imu.positionAccel);
    const M3 R=s0.q.conjugate().toRotationMatrix();
    const double dt=imu.duration;
    const V3 gravity(0,0,imu.gravity),bg=Eigen::Map<const V3>(sb0+3)-Eigen::Map<const V3>(imu.referenceBias),
        ba=Eigen::Map<const V3>(sb0+6)-Eigen::Map<const V3>(imu.referenceBias+3);
    const V3 dp=s0.p-s1.p+dt*s0.v-0.5*dt*dt*gravity,dv=s0.v-s1.v-dt*gravity,phi=-A*bg;
    const Q correction=rotationExp(phi),dq=correction*Eigen::Map<const Q>(imu.deltaQuaternion);
    const Q tail=Eigen::Map<const Q>(imu.deltaQuaternion)*s1.q.conjugate()*s0.q;
    Eigen::Matrix<double,15,1> e;
    e.template segment<3>(0)=R*dp+Eigen::Map<const V3>(imu.deltaPosition)+P*bg-Pa*ba;
    e.template segment<3>(3)=2*(dq*(s1.q.conjugate()*s0.q)).vec();
    e.template segment<3>(6)=R*dv+Eigen::Map<const V3>(imu.deltaVelocity)+V*bg-Va*ba;
    e.template tail<6>()=Eigen::Map<const Eigen::Matrix<double,6,1>>(sb0+3)-Eigen::Map<const Eigen::Matrix<double,6,1>>(sb1+3);
    const Eigen::Map<const Eigen::Matrix<double,15,15>> W(imu.whitening);
    Eigen::Map<Eigen::Matrix<double,15,1>> residual(r);
    residual=W*e;
    if(!finite(residual) || !jacobian) return finite(residual);
    const M3 gyroAngle=-rotationJacobian(-phi)*A;
    M3 Jrotation,Jbias;
    for(int j=0;j<3;++j) {
        const V3 axis=V3::Unit(j), db=gyroAngle.col(j);
        Jrotation.col(j)=(dq*s1.q.conjugate()*Q(0,axis.x(),axis.y(),axis.z())*s0.q).vec();
        Jbias.col(j)=(correction*Q(0,db.x(),db.y(),db.z())*tail).vec();
    }
    for(int k=0;k<4;++k) {
        const auto& block=f.blocks[k];
        if(!block.width) continue;
        const bool first=k<2;
        const double sign=first?1:-1;
        const auto& s=first?s0:s1;
        Eigen::Matrix<double,15,9> J=Eigen::Matrix<double,15,9>::Zero();
        if(k%2==0) {
            J.template block<3,3>(0,0)=sign*R;
            J.template block<3,3>(0,3)=-sign*R*skewMatrix(s.p);
            J.template block<3,3>(3,3)=sign*Jrotation;
            J.template block<3,3>(6,3)=-sign*R*skewMatrix(s.lever);
            if(first) {
                J.template block<3,3>(0,3)+=R*(skewMatrix(dp)-dt*skewMatrix(s.lever));
                J.template block<3,3>(6,3)+=R*skewMatrix(dv);
            }
        } else {
            const M3 leverBias=skewMatrix(s.arm)*s.q.toRotationMatrix();
            J.template block<3,3>(6,0)=sign*R;
            J.template block<3,3>(6,3)=sign*R*leverBias;
            J.template block<3,3>(9,3)=sign*M3::Identity();
            J.template block<3,3>(12,6)=sign*M3::Identity();
            if(first) {
                J.template block<3,3>(0,0)=dt*R;
                J.template block<3,3>(0,3)=dt*R*leverBias+P;
                J.template block<3,3>(0,6)=-Pa;
                J.template block<3,3>(3,3)=Jbias;
                J.template block<3,3>(6,3)+=V;
                J.template block<3,3>(6,6)=-Va;
            }
        }
        for(int j=0;j<block.width;++j) {
            const Eigen::Matrix<double,15,1> column=W*J.col(j);
            if(!finite(column)) return false;
            for(int i=0;i<15;++i) jacobian[i*columns+block.column+j]=column[i];
        }
    }
    return true;
}

}
