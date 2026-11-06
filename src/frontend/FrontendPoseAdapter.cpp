#include <frontend/FrontendPoseAdapter.hpp>
#include <visual/MultiFrame.hpp>
#include <common/Parameters.hpp>
#include <backend/LeftSE3Manifold.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

namespace cgmli { namespace frontend_pose {
bool Options::valid() const {
    // Frontend input is ordered by observation/landmark identity, not match quality.
    // PROSAC requires a quality ordering that this ingestion path does not provide.
    return !solver.ransac.progressive_sampling
        && minimumAbsoluteInliers>=3 && minimumRelativeInliers>=6 && minimumLoopLandmarks>=3
        && std::isfinite(absoluteInlierRatio) && absoluteInlierRatio>0 && absoluteInlierRatio<=1
        && std::isfinite(relativeInlierRatio) && relativeInlierRatio>0 && relativeInlierRatio<=1
        && std::isfinite(rotationDominanceRatio) && rotationDominanceRatio>0 && rotationDominanceRatio<=1
        && std::isfinite(loopPixelSigma) && loopPixelSigma>0
        && trackingMinIterations>0 && fullRansacPeriod>0 && minimumBudgetCorrespondences>=minimumAbsoluteInliers
        && std::isfinite(budgetPredictionInlierRatio) && budgetPredictionInlierRatio>0 && budgetPredictionInlierRatio<=1
        && std::isfinite(budgetPredictionRmsFraction) && budgetPredictionRmsFraction>0 && budgetPredictionRmsFraction<=1
        && std::isfinite(solver.angularThreshold) && solver.angularThreshold>0 && solver.angularThreshold<0.5;
}
bool absolutePredictionHealthy(const pose_solver::AbsolutePoseCorrespondences& data,
    const kinematics::Transformation& T_WB,const Options& o) {
    if(data.size()<o.minimumBudgetCorrespondences || !T_WB.T().allFinite()) return false;
    const auto T_BW=T_WB.inverse(); std::size_t count=0; double squared=0;
    for(const auto& m:data) {
        const Eigen::Vector3d p=T_BW.C()*m.point_W+T_BW.r()-m.ray_S.origin;
        const double length=m.ray_S.direction.norm();
        if(!p.allFinite() || !std::isfinite(length) || length<1e-12) return false;
        const Eigen::Vector3d b=m.ray_S.direction/length; const double dot=p.dot(b);
        if(dot<=o.solver.minRayDepth) continue;
        const double angle=std::atan2(p.cross(b).stableNorm(),dot);
        if(angle<o.solver.angularThreshold) { ++count; squared+=angle*angle; }
    }
    return count>=o.minimumAbsoluteInliers && double(count)>=o.budgetPredictionInlierRatio*data.size()
        && std::sqrt(squared/std::max(size_t(1),count))<=o.budgetPredictionRmsFraction*o.solver.angularThreshold;
}
bool makeRay(const MultiFrame& frame,const KeypointIdentifier& key,const Extrinsics& extrinsics,
    pose_solver::GeneralizedRay& ray) {
    if(key.frameId!=frame.id() || extrinsics.size()!=frame.numFrames()
        || key.cameraIndex>=frame.numFrames() || key.keypointIndex>=frame.numKeypoints(key.cameraIndex)) return false;
    const auto camera=frame.geometry(key.cameraIndex); Eigen::Vector2d pixel;
    if(!camera || !frame.getKeypoint(key.cameraIndex,key.keypointIndex,pixel) || !pixel.allFinite()
        || pixel.x()<0 || pixel.y()<0 || pixel.x()>=camera->imageWidth() || pixel.y()>=camera->imageHeight()
        || (camera->hasMask() && camera->mask().at<uchar>(static_cast<int>(pixel.y()),static_cast<int>(pixel.x()))==0)) return false;
    Eigen::Vector3d bearing;
    if(!frame.getBackProjection(key.cameraIndex,key.keypointIndex,bearing) && !camera->backProject(pixel,&bearing)) return false;
    return pose_solver::makeGeneralizedRay(extrinsics[key.cameraIndex],bearing,ray);
}
AbsoluteInput absoluteInput(const MultiFrame& frame,const MapPoints& landmarks,const Extrinsics& extrinsics) {
    AbsoluteInput output;
    for(size_t camera=0;camera<frame.numFrames();++camera) for(size_t keypoint=0;keypoint<frame.numKeypoints(camera);++keypoint) {
        const LandmarkId id(frame.landmarkId(camera,keypoint)); auto point=landmarks.find(id);
        if(!id.isInitialised() || point==landmarks.end() || !point->second.isInitialised
            || !point->second.point.allFinite() || std::abs(point->second.point[3])<1e-12) continue;
        pose_solver::AbsolutePoseCorrespondence match;
        match.point_W=point->second.point.head<3>()/point->second.point[3];
        const KeypointIdentifier key(frame.id(),camera,keypoint);
        if(!match.point_W.allFinite() || !makeRay(frame,key,extrinsics,match.ray_S)) continue;
        output.data.push_back(match); output.observations.push_back({key,id});
    }
    return output;
}
AbsoluteInput absoluteInput(const MultiFrame& frame,const std::map<LandmarkId,Eigen::Vector3d>& points,const Extrinsics& extrinsics) {
    AbsoluteInput output; output.data.reserve(frame.numKeypoints()); output.observations.reserve(frame.numKeypoints());
    for(size_t camera=0;camera<frame.numFrames();++camera) for(size_t keypoint=0;keypoint<frame.numKeypoints(camera);++keypoint) {
        const LandmarkId id(frame.landmarkId(camera,keypoint)); const auto point=points.find(id);
        if(!id.isInitialised() || point==points.end() || !point->second.allFinite()) continue;
        const KeypointIdentifier key(frame.id(),camera,keypoint); pose_solver::AbsolutePoseCorrespondence match;
        match.point_W=point->second;
        if(!makeRay(frame,key,extrinsics,match.ray_S)) continue;
        output.data.push_back(match); output.observations.push_back({key,id});
    }
    return output;
}
RelativeInput relativeInput(const MultiFrame& first,const MultiFrame& second,
    const Extrinsics& firstExtrinsics,const Extrinsics& secondExtrinsics) {
    if(first.id()==second.id()) return {};
    using Ray=std::pair<KeypointIdentifier,pose_solver::GeneralizedRay>;
    using Groups=std::map<LandmarkId,std::vector<Ray>>;
    const auto groups=[](const MultiFrame& frame,const Extrinsics& extrinsics) {
        Groups result;
        for(size_t camera=0;camera<frame.numFrames();++camera) for(size_t keypoint=0;keypoint<frame.numKeypoints(camera);++keypoint) {
            const LandmarkId id(frame.landmarkId(camera,keypoint)); if(!id.isInitialised()) continue;
            const KeypointIdentifier key(frame.id(),camera,keypoint); pose_solver::GeneralizedRay ray;
            if(makeRay(frame,key,extrinsics,ray)) result[id].emplace_back(key,ray);
        }
        return result;
    };
    const auto a=groups(first,firstExtrinsics),b=groups(second,secondExtrinsics);
    std::map<std::pair<size_t,size_t>,size_t> used;
    RelativeInput output;
    for(const auto& item:a) {
        auto other=b.find(item.first); if(other==b.end()) continue;
        const Ray *bestA=nullptr,*bestB=nullptr;
        std::tuple<size_t,bool,size_t,size_t,size_t,size_t> score;
        for(const auto& x:item.second) for(const auto& y:other->second) {
            const auto candidate=std::make_tuple(used[{x.first.cameraIndex,y.first.cameraIndex}],
                x.first.cameraIndex!=y.first.cameraIndex,x.first.cameraIndex,y.first.cameraIndex,
                x.first.keypointIndex,y.first.keypointIndex);
            if(!bestA || candidate<score) { bestA=&x; bestB=&y; score=candidate; }
        }
        if(!bestA) continue;
        ++used[{bestA->first.cameraIndex,bestB->first.cameraIndex}];
        output.data.push_back({bestA->second,bestB->second});
        output.observations.push_back({bestA->first,bestB->first,item.first});
    }
    return output;
}
bool relativeWorldSeed(const kinematics::Transformation& firstPose,const kinematics::Transformation& predicted,
    const pose_solver::PoseSolverResult& result,bool rotationOnly,kinematics::Transformation& seed) {
    const auto valid=[](const kinematics::Transformation& pose) {
        return pose.T().allFinite() && std::abs(pose.q().squaredNorm()-1.0)<1e-6;
    };
    if(!result.success() || !valid(firstPose) || !valid(predicted) || !valid(result.pose)) return false;
    kinematics::Transformation candidate;
    if(!rotationOnly && result.translationScaleObservable) candidate=firstPose*result.pose;
    else candidate=kinematics::Transformation(predicted.r(),firstPose.q()*result.pose.q());
    if(!valid(candidate)) return false;
    seed=candidate; return true;
}
bool inertialPoseConsistent(const kinematics::Transformation& prediction,
    const kinematics::Transformation& candidate,double elapsedSeconds,
    const EstimatorParameters& options,std::string* reason) {
    const auto fail=[&](const char* text) { if(reason) *reason=text; return false; };
    if (!prediction.T().allFinite() || !candidate.T().allFinite()
        || std::abs(prediction.q().squaredNorm()-1)>1e-6 || std::abs(candidate.q().squaredNorm()-1)>1e-6
        || !std::isfinite(elapsedSeconds) || elapsedSeconds<0
        || !std::isfinite(options.max_visual_imu_rotation_error) || options.max_visual_imu_rotation_error<=0
        || !std::isfinite(options.max_visual_imu_position_error) || options.max_visual_imu_position_error<=0
        || !std::isfinite(options.max_visual_imu_position_error_rate) || options.max_visual_imu_position_error_rate<0)
        return fail("Invalid inertial pose consistency input/bounds");
    const double rotation=Eigen::AngleAxisd(candidate.q()*prediction.q().conjugate()).angle();
    if (rotation>options.max_visual_imu_rotation_error)
        return fail("Visual/optimized rotation disagrees with IMU prediction");
    const double allowance=options.max_visual_imu_position_error+elapsedSeconds*options.max_visual_imu_position_error_rate;
    if ((candidate.r()-prediction.r()).norm()>allowance)
        return fail("Visual/optimized sensor position disagrees with IMU prediction");
    return true;
}
namespace {
Eigen::Matrix<double,3,2> basis(const Eigen::Vector3d& bearing) {
    const Eigen::Vector3d axis=std::abs(bearing.x())<0.8?Eigen::Vector3d::UnitX():Eigen::Vector3d::UnitY();
    Eigen::Matrix<double,3,2> result;
    result.col(0)=(axis-bearing*bearing.dot(axis)).normalized(); result.col(1)=bearing.cross(result.col(0)); return result;
}
bool logarithm(const Eigen::Vector3d& predicted,const Eigen::Vector3d& observed,Eigen::Vector3d& value) {
    if(!predicted.allFinite() || predicted.norm()<1e-12) return false;
    const auto direction=predicted.normalized(); const double cosine=std::clamp(direction.dot(observed),-1.0,1.0);
    const double sine=direction.cross(observed).norm(); if(cosine<-1.0+1e-8) return false;
    value=direction-cosine*observed; if(sine>1e-10) value*=std::atan2(sine,cosine)/sine;
    return value.allFinite();
}
}
bool absoluteInformation(const MultiFrame& frame,const Extrinsics& extrinsics,const AbsoluteInput& input,
    const pose_solver::PoseSolverResult& result,double pixelSigma,double pyramidScale,
    Eigen::Matrix<double,6,6>& information) {
    if(!result.success() || input.data.size()!=input.observations.size() || extrinsics.size()!=frame.numFrames()
        || !std::isfinite(pixelSigma) || pixelSigma<=0 || !std::isfinite(pyramidScale) || pyramidScale<=1) return false;
    Eigen::Matrix<double,6,6> total=Eigen::Matrix<double,6,6>::Zero();
    const backend::LeftSE3Manifold manifold;
    std::set<size_t> unique;
    for(auto index:result.inliers) {
        if(index>=input.data.size() || !unique.insert(index).second) return false;
        const auto& key=input.observations[index].key;
        if(key.frameId!=frame.id() || key.cameraIndex>=frame.numFrames()
            || key.keypointIndex>=frame.numKeypoints(key.cameraIndex)) return false;
        const auto camera=frame.geometry(key.cameraIndex); if(!camera) return false;
        const auto& point=frame.keypoints(key.cameraIndex)[key.keypointIndex];
        Eigen::Vector3d observed; Eigen::Matrix<double,3,2> derivative;
        if(!camera->backProject(Eigen::Vector2d(point.pt.x,point.pt.y),&observed,&derivative)
            || !observed.allFinite() || !derivative.allFinite() || observed.norm()<1e-12 || point.octave<0) return false;
        const double norm=observed.norm(); observed/=norm; const auto tangent=basis(observed);
        if((input.data[index].ray_S.origin-extrinsics[key.cameraIndex].r()).norm()>1e-6
            || (input.data[index].ray_S.direction-extrinsics[key.cameraIndex].C()*observed).norm()>1e-6) return false;
        const Eigen::Matrix<double,2,2> pixelToTangent=tangent.transpose()
            *(Eigen::Matrix3d::Identity()-observed*observed.transpose())*derivative/norm;
        const double sigma=pixelSigma*std::pow(pyramidScale,point.octave);
        const Eigen::Matrix2d covariance=sigma*sigma*pixelToTangent*pixelToTangent.transpose();
        Eigen::LLT<Eigen::Matrix2d> llt(covariance); if(llt.info()!=Eigen::Success) return false;
        const Eigen::Matrix2d whiten=llt.matrixL().solve(Eigen::Matrix2d::Identity());
        const auto residual=[&](const kinematics::Transformation& pose,Eigen::Vector2d& value) {
            const Eigen::Vector3d bodyPoint=pose.C().transpose()*(input.data[index].point_W-pose.r());
            const auto& T_BC=extrinsics[key.cameraIndex];
            const Eigen::Vector3d cameraPoint=T_BC.C().transpose()*(bodyPoint-T_BC.r());
            Eigen::Vector3d angular; if(!logarithm(cameraPoint,observed,angular)) return false;
            value=whiten*tangent.transpose()*angular; return value.allFinite();
        };
        Eigen::Matrix<double,2,6> J;
        for(int column=0;column<6;++column) {
            const double h=column<3?1e-6*std::max(1.0,result.pose.r().norm()):1e-7;
            Eigen::Matrix<double,6,1> delta=Eigen::Matrix<double,6,1>::Zero();
            Eigen::Matrix<double,7,1> plus,minus;
            delta[column]=h; if(!manifold.Plus(result.pose.parameters().data(),delta.data(),plus.data())) return false;
            delta[column]=-h; if(!manifold.Plus(result.pose.parameters().data(),delta.data(),minus.data())) return false;
            const auto transform=[](const Eigen::Matrix<double,7,1>& value) {
                return kinematics::Transformation(value.head<3>(),Eigen::Quaterniond(value[6],value[3],value[4],value[5]));
            };
            Eigen::Vector2d rp,rm; if(!residual(transform(plus),rp) || !residual(transform(minus),rm)) return false;
            J.col(column)=(rp-rm)/(2*h);
        }
        if(!J.allFinite()) return false; total+=J.transpose()*J;
    }
    if(!total.allFinite() || total.diagonal().minCoeff()<=0) return false;
    const Eigen::Matrix<double,6,1> scale=total.diagonal().cwiseSqrt().cwiseInverse();
    const Eigen::Matrix<double,6,6> normalized=scale.asDiagonal()*total*scale.asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> spectrum(normalized);
    if(spectrum.info()!=Eigen::Success || spectrum.eigenvalues().minCoeff()<=1e-9*spectrum.eigenvalues().maxCoeff()) return false;
    information=0.5*(total+total.transpose()); return true;
}
} }
