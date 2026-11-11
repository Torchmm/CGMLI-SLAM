#pragma once

#include "VisualCudaBatch.hpp"
#include <Eigen/Geometry>

namespace cgmli::backend::detail {
namespace sphere_batch {

using V3=Eigen::Vector3d;
using M3=Eigen::Matrix3d;
using M23=Eigen::Matrix<double,2,3>;
using R3=Eigen::Matrix<double,3,3,Eigen::RowMajor>;
using R23=Eigen::Matrix<double,2,3,Eigen::RowMajor>;

template<class Derived>
EIGEN_DEVICE_FUNC inline bool finite(const Eigen::DenseBase<Derived>& value) {
    for(int row=0;row<value.rows();++row)
        for(int col=0;col<value.cols();++col)
            if(!Eigen::numext::isfinite(value.derived().coeff(row,col))) return false;
    return true;
}

EIGEN_DEVICE_FUNC inline M3 skewMatrix(const V3& v) {
    M3 result;
    result << 0,-v.z(),v.y(), v.z(),0,-v.x(), -v.y(),v.x(),0;
    return result;
}

EIGEN_DEVICE_FUNC inline V3 timeBearing(const double* raw,const double* velocity,double dt,
    bool differentiate,V3& derivative) {
    const Eigen::Map<const V3> bearing(raw), v(velocity);
    if (!differentiate) {
        const V3 u=v*dt;
        const double angle=u.norm();
        if(angle<1e-8) return (bearing+u).normalized();
        return ::cos(angle)*bearing+(::sin(angle)/angle)*u;
    }
    const double speed=v.norm(), angle=speed*dt;
    if (::fabs(angle)<1e-8) {
        const V3 direction=bearing+v*dt;
        const double length=direction.norm();
        const V3 unit=direction/length;
        derivative=-(M3::Identity()-unit*unit.transpose())*v/length;
        return unit;
    }
    const double sine=::sin(angle),cosine=::cos(angle);
    derivative=speed*sine*bearing-cosine*v;
    return cosine*bearing+(sine/speed)*v;
}

EIGEN_DEVICE_FUNC inline bool writePose(const double* pose,double* output,
    const M23& translation,const M23& rotation) {
    if (!output) return true;
    const Eigen::Map<const Eigen::Quaterniond> q(pose+3);
    if (!finite(q.coeffs()) || ::fabs(q.squaredNorm()-1)>1e-6) return false;
    Eigen::Matrix<double,3,4> lift;
    lift.leftCols<3>()=2*(q.w()*M3::Identity()+skewMatrix(q.vec()));
    lift.rightCols<1>()=-2*q.vec();
    Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> J(output);
    J.leftCols<3>()=translation;
    J.rightCols<4>()=rotation*lift;
    return finite(J);
}

// Independent scalar implementation of the analytic S² factor. Ceres still
// owns loss/manifold correction, damping, the trust region and factor gates.
EIGEN_DEVICE_FUNC inline bool evaluate(const VisualBatchInput& in,const double* parameters,
    double* residual,double* derivatives) {
    const auto* anchor=parameters+in.offsets[0];
    const auto* observer=parameters+in.offsets[in.observer];
    const auto* rig=parameters+in.offsets[in.extrinsic];
    const double rho=parameters[in.offsets[in.range]];
    const double td=in.time<0?in.anchorTime:parameters[in.offsets[in.time]];
    if (!(rho>0) || !Eigen::numext::isfinite(rho) || !Eigen::numext::isfinite(td)) return false;
    for(int i=0;i<in.blocks;++i) if(in.sizes[i]==7) {
        const double* state=parameters+in.offsets[i];
        const Eigen::Map<const Eigen::Matrix<double,7,1>> all(state);
        if(!finite(all) || all.tail<4>().squaredNorm()<=1e-16) return false;
    }
    double* j[5]{};
    int column=0;
    for (int i=0;i<in.blocks;++i) {
        if (!(in.derivativeMask & (1u<<i))) continue;
        if (derivatives) j[i]=derivatives+column;
        column+=2*in.sizes[i];
    }
    const bool timeDerivative=in.time>=0 && j[in.time];
    V3 ba=Eigen::Map<const V3>(in.anchorStatic), bb=Eigen::Map<const V3>(in.measuredStatic);
    V3 da=V3::Zero(), db=V3::Zero();
    if(in.movingAnchor) ba=timeBearing(in.anchor,in.anchorVelocity,in.anchorTime-td,timeDerivative,da);
    if(in.movingMeasured) bb=timeBearing(in.measured,in.measuredVelocity,in.measuredTime-td,timeDerivative,db);
    if(!finite(ba) || !finite(bb)) return false;

    const Eigen::Map<const R3> Rca(in.anchorRotation), Rcb(in.measuredRotation);
    const Eigen::Map<const V3> tca(in.anchorTranslation),tcb(in.measuredTranslation);
    const V3 local=Rca*(ba/rho)+tca;
    M3 Ra,Rb,Re,W,cameraRotation;
    V3 body,relative,rotatedBody,observerBody,pc,te;
    if (in.observer==0) {
        pc=Rcb.transpose()*(local-tcb);
        cameraRotation=Rcb.transpose()*Rca;
    } else {
        Ra=Eigen::Map<const Eigen::Quaterniond>(anchor+3).normalized().toRotationMatrix();
        Rb=Eigen::Map<const Eigen::Quaterniond>(observer+3).normalized().toRotationMatrix();
        Re=Eigen::Map<const Eigen::Quaterniond>(rig+3).normalized().toRotationMatrix();
        te=Eigen::Map<const V3>(rig);
        W=(Rb*Re*Rcb).transpose();
        cameraRotation=W*(Ra*Re*Rca);
        const V3 separation=Eigen::Map<const V3>(anchor)-Eigen::Map<const V3>(observer);
        const V3 offset=W*(Ra*(Re*tca+te)-Rb*(Re*tcb+te)+separation);
        pc=cameraRotation*(ba/rho)+offset;
        if(j[0] || j[in.observer] || j[in.extrinsic]) {
            body=Re*local+te;
            rotatedBody=Ra*body;
            relative=rotatedBody+separation;
            observerBody=Rb.transpose()*relative;
        }
    }
    const double range=pc.norm();
    if(!finite(pc) || !Eigen::numext::isfinite(range) || range<1e-10) return false;
    const V3 n=pc/range;
    const double c=::fmin(1.0,::fmax(-1.0,bb.dot(n))),s=bb.cross(n).norm();
    const Eigen::Map<const V3> measured(in.measured);
    const double denominator=in.movingMeasured?1+measured.dot(bb):2;
    if(c<-1+1e-8 || denominator<1e-8) return false;
    const V3 u=n-c*bb;
    const double d=1-c;
    const bool series=c>0 && s<1e-3;
    const double theta=series?0 : ::atan2(s,c);
    const double f=s<1e-8?1.0:(series?1+d/3+2*d*d/15+2*d*d*d/35:theta/s);
    const V3 logarithm=f*u;
    const Eigen::Map<const R23> measurement(in.measurement);
    M23 P=measurement;
    M3 K=M3::Zero();
    if(in.movingMeasured) {
        K=skewMatrix(measured.cross(bb));
        const M3 transport=M3::Identity()+K+K*K/denominator;
        P=measurement*transport.transpose();
    }
    Eigen::Map<Eigen::Vector2d> out(residual);
    out=P*logarithm;
    if(!finite(out)) return false;
    if(!derivatives) return true;
    const double g=s<1e-8?0.0:(series?(-1.0/3.0-4*d/15.0-6*d*d/35.0):(theta*c-s)/(s*s*s));
    const M23 projected=f*P+(g*(P*u)-f*(P*bb))*bb.transpose();
    const M23 reduce=(projected-(projected*n)*n.transpose())/range;
    if(in.observer==0) {
        if(j[0]) Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>>(j[0]).setZero();
        if(j[in.extrinsic]) Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>>(j[in.extrinsic]).setZero();
    } else if(j[0] || j[in.observer] || j[in.extrinsic]) {
        const M23 D=reduce*W;
        if(!writePose(anchor,j[0],D,-D*skewMatrix(rotatedBody)) ||
            !writePose(observer,j[in.observer],-D,D*skewMatrix(relative))) return false;
        if(j[in.extrinsic] && !writePose(rig,j[in.extrinsic],D*(Ra-Rb),
            D*(-Ra*skewMatrix(body-te)+Rb*skewMatrix(observerBody-te)))) return false;
    }
    if(j[in.range]) {
        Eigen::Map<Eigen::Vector2d> J(j[in.range]);
        J=reduce*cameraRotation*(-ba/(rho*rho));
        if(!finite(J)) return false;
    }
    if(timeDerivative) {
        Eigen::Map<Eigen::Vector2d> J(j[in.time]);
        if(!in.movingAnchor && !in.movingMeasured) J.setZero();
        else {
            const M3 Jb=f*(-bb*n.transpose()-c*M3::Identity())+g*u*n.transpose();
            const M3 dK=skewMatrix(measured.cross(db));
            const M3 dR=dK+(dK*K+K*dK)/denominator-(K*K)*(measured.dot(db)/(denominator*denominator));
            J=reduce*cameraRotation*(da/rho)+P*Jb*db+measurement*dR.transpose()*logarithm;
            if(!finite(J)) return false;
        }
    }
    return true;
}

}
}
