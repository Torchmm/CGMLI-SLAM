// Standalone regression source. Not compiled/executed in this static-only task.
#include <debug/ModuleDebug.hpp>
#include <debug/ModuleDebugArtifacts.hpp>
#include <frontend/FrontendPoseAdapter.hpp>
#include <visual/pose_solver/AbsolutePoseSolver.hpp>
#include <visual/pose_solver/RelativePoseSolver.hpp>
#include <visual/pose_solver/RotationPoseSolver.hpp>
#include <visual/pose_solver/PoseEstimatorAdapter.hpp>
#include <visual/MultiFrame.hpp>
#include <cameras/PinholeCamera.hpp>
#include <cameras/EucmCamera.hpp>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <set>
#include <random>
#include <array>
#include <algorithm>
#include <stdexcept>

using namespace cgmli;
namespace fp=frontend_pose;
namespace ps=pose_solver;
namespace {
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
class SamplingProbe final : public ps::detail::PoseEstimatorAdapter {
public:
    SamplingProbe(size_t count,size_t sampleSize,const ps::PoseSolverOptions& options)
        : PoseEstimatorAdapter(count,sampleSize,3,options) {}
    const ps::detail::Indices& next() { return drawMinimalSample(); }
protected:
    double angularError(const ps::detail::Pose&,size_t) const override { return 0; }
    bool residualForMatch(const ps::detail::Pose&,size_t,Eigen::VectorXd& residual) const override {
        residual=Eigen::Vector3d::Zero(); return true;
    }
};
void minimalSamplingBuffers() {
    // Exercise the actual shared adapter, without invoking a polynomial solver.
    // N==K catches impossible-sample loops; 32 draws cross the PROSAC cutoff.
    for(size_t sampleSize:{size_t(3),size_t(5),size_t(6)}) for(size_t count:{sampleSize,sampleSize+11})
        for(bool progressive:{false,true}) {
            ps::PoseSolverOptions options; options.ransac.progressive_sampling=progressive;
            options.ransac.max_prosac_iterations=8;
            SamplingProbe first(count,sampleSize,options),second(count,sampleSize,options);
            const size_t* buffer=nullptr;
            for(int iteration=0;iteration<32;++iteration) {
                const auto& sample=first.next(); const auto& repeat=second.next();
                require(sample.size()==sampleSize,"PoseLib sampling output has allocated size");
                require(sample==repeat,"Sampler seed remains deterministic");
                std::set<size_t> unique(sample.begin(),sample.end());
                require(unique.size()==sampleSize && *unique.rbegin()<count,"Minimal sample indices are unique and in range");
                if(!buffer) buffer=sample.data();
                require(sample.data()==buffer,"Sampling reuses one preallocated buffer");
            }
        }
    ps::PoseSolverOptions options; options.ransac.progressive_sampling=true;
    bool rejected=false;
    try { SamplingProbe invalid(2,3,options); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"N<K must be rejected before PROSAC initialization");
    rejected=false;
    try { SamplingProbe invalid(3,0,options); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"An empty minimal sample must be rejected");
}
double rotationError(const kinematics::Transformation& a,const kinematics::Transformation& b) {
    return Eigen::AngleAxisd(a.q()*b.q().conjugate()).angle();
}
struct Fixture {
    cameras::NCameraSystem rig;
    fp::Extrinsics extrinsics;
    MapPoints points;
    kinematics::Transformation first{Eigen::Vector3d(0.2,-0.1,0.3),Eigen::Quaterniond(Eigen::AngleAxisd(0.1,Eigen::Vector3d::UnitZ()))};
    kinematics::Transformation relative{Eigen::Vector3d(0.3,0.02,0.05),Eigen::Quaterniond(Eigen::AngleAxisd(0.06,Eigen::Vector3d::UnitY()))};
    Fixture() {
        rig.addCamera(std::make_shared<kinematics::Transformation>(),
            std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(640,480,260,260,320,240,cameras::NoDistortion()),
            cameras::NCameraSystem::NoDistortion,false);
        rig.addCamera(std::make_shared<kinematics::Transformation>(Eigen::Vector3d(0.25,0,0),Eigen::Quaterniond::Identity()),
            std::make_shared<cameras::EucmCamera>(640,480,140,150,320,240,0.7,1.0),
            cameras::NCameraSystem::NoDistortion,false);
        const kinematics::Transformation T_BS(Eigen::Vector3d(0.05,-0.03,0.02),
            Eigen::Quaterniond(Eigen::AngleAxisd(0.1,Eigen::Vector3d::UnitY())));
        for(size_t c=0;c<rig.numCameras();++c) extrinsics.push_back(T_BS*(*rig.T_SC(c)));
        for(uint64_t i=0;i<21;++i) {
            MapPoint2 point{}; point.id=LandmarkId(i+1); point.isInitialised=true;
            const Eigen::Vector3d local=i==20?Eigen::Vector3d(3,0.4,-0.8):
                Eigen::Vector3d((static_cast<int>(i%5)-2)*0.7+0.1,(static_cast<int>(i/5)-2)*0.4,3.5+0.31*(i%7)+0.17*(i/3));
            point.point.head<3>()=first.C()*local+first.r(); point.point[3]=1; points.emplace(point.id,point);
        }
    }
    std::shared_ptr<MultiFrame> frame(uint64_t id,const kinematics::Transformation& body) const {
        auto frame=std::make_shared<MultiFrame>(rig,Time(static_cast<uint32_t>(id),0),id);
        for(size_t camera=0;camera<rig.numCameras();++camera) {
            std::vector<cv::KeyPoint> keypoints; std::vector<LandmarkId> ids;
            const auto T_CW=(body*extrinsics[camera]).inverse();
            for(const auto& item:points) {
                const Eigen::Vector3d point=T_CW.C()*item.second.point.head<3>()+T_CW.r();
                Eigen::Vector2d pixel;
                if(rig.cameraGeometry(camera)->project(point,&pixel)!=cameras::ProjectionStatus::Successful) continue;
                keypoints.emplace_back(static_cast<float>(pixel.x()),static_cast<float>(pixel.y()),31.0f); ids.push_back(item.first);
            }
            frame->resetKeypoints(camera,keypoints);
            for(size_t i=0;i<ids.size();++i) frame->setLandmarkId(camera,i,ids[i].value());
        }
        return frame;
    }
};
ps::PoseSolverOptions options() {
    ps::PoseSolverOptions options; options.angularThreshold=0.003; options.minInliers=10;
    options.ransac.max_iterations=300; options.ransac.min_iterations=30; return options;
}
void absoluteMappingAndBodyConvention() {
    Fixture f; const auto truth=f.first*f.relative; auto frame=f.frame(2,truth);
    auto points=f.points; points.at(LandmarkId(1)).isInitialised=false; points.at(LandmarkId(2)).point[3]=0;
    points.at(LandmarkId(7)).point.x()+=2; // A known bad 3D association, not a deleted raw row.
    const auto input=fp::absoluteInput(*frame,points,f.extrinsics);
    std::map<LandmarkId,Eigen::Vector3d> coordinates;
    for(const auto& item:points) if(item.second.isInitialised && item.second.point.allFinite() && std::abs(item.second.point[3])>=1e-12)
        coordinates.emplace(item.first,item.second.point.head<3>()/item.second.point[3]);
    for(size_t camera=0;camera<frame->numFrames();++camera) frame->computeBackProjections(camera);
    const auto cached=fp::absoluteInput(*frame,coordinates,f.extrinsics);
    require(cached.data.size()==input.data.size(),"Coordinate/ray caches changed absolute correspondence count");
    for(size_t i=0;i<input.data.size();++i) {
        require(cached.observations[i].key==input.observations[i].key && cached.observations[i].landmark==input.observations[i].landmark,
            "Coordinate cache changed original observation mapping");
        require((cached.data[i].point_W-input.data[i].point_W).norm()<1e-12
            && (cached.data[i].ray_S.origin-input.data[i].ray_S.origin).norm()<1e-12
            && (cached.data[i].ray_S.direction-input.data[i].ray_S.direction).norm()<1e-12,"Cached coordinates/rays changed solver geometry");
    }
    require(input.data.size()==input.observations.size() && input.observations.front().key.keypointIndex>=2,"filtered rows retain explicit original keypoint mapping");
    bool rear=false;
    for(size_t i=0;i<input.data.size();++i) {
        const auto& key=input.observations[i].key;
        require((input.data[i].ray_S.origin-f.extrinsics[key.cameraIndex].r()).norm()<1e-12,"solver uses live T_BC origins when body differs from IMU");
        require(input.observations[i].landmark!=LandmarkId(1) && input.observations[i].landmark!=LandmarkId(2),"uninitialized and infinite points excluded");
        rear=rear || input.data[i].ray_S.direction.z()<0;
    }
    require(rear,"EUCM rear-hemisphere rays survive adaptation");
    const auto result=ps::AbsolutePoseSolver(options()).solve(input.data,&truth);
    require(result.success() && (result.pose.r()-truth.r()).norm()<1e-3 && rotationError(result.pose,truth)<1e-3,"absolute output is T_WB, not its inverse");
    for(auto index:result.inliers) require(input.observations[index].landmark!=LandmarkId(7),"inliers map to the correct original association after filtering");
    Eigen::Matrix<double,6,6> H;
    require(fp::absoluteInformation(*frame,f.extrinsics,input,result,1.0,1.2,H),"mixed-model left-tangent conditional information");
    require(H.allFinite() && (H-H.transpose()).norm()<1e-9 && Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>>(H).eigenvalues().minCoeff()>0,"information is symmetric and full rank without damping");
    const auto before=H;
    require(!fp::absoluteInformation(*frame,f.extrinsics,input,result,0.0,1.2,H) && (H-before).norm()==0,"invalid information request leaves output unchanged");
}
void pooledRelativeAndLoopFrames() {
    Fixture f; const auto truth=f.first*f.relative;
    const auto first=f.frame(1,f.first),second=f.frame(2,truth);
    const auto input=fp::relativeInput(*first,*second,f.extrinsics,f.extrinsics);
    std::set<LandmarkId> unique; std::set<std::pair<size_t,size_t>> pairs;
    for(const auto& observation:input.observations) {
        require(unique.insert(observation.landmark).second,"one independent pair per landmark");
        require(observation.first.frameId==1 && observation.second.frameId==2,"relative index map preserves old/new order");
        pairs.emplace(observation.first.cameraIndex,observation.second.cameraIndex);
    }
    require(pairs.size()>1,"camera pairs pooled rather than separate monocular solves");
    const auto result=ps::RelativePoseSolver(options()).solve(input.data,&f.relative);
    require(result.success() && result.translationScaleObservable && (result.pose.r()-f.relative.r()).norm()<1e-3
        && rotationError(result.pose,f.relative)<1e-3,"generalized relative output is old-body from new-body with metric scale");
    kinematics::Transformation seed;
    require(fp::relativeWorldSeed(f.first,truth,result,false,seed) && (seed.T()-truth.T()).norm()<1e-3,"world composition uses T_WBold*T_BoldBnew");
    auto oldPoints=f.points;
    const auto T_old_W=f.first.inverse();
    for(auto& item:oldPoints) item.second.point.head<3>()=(T_old_W.C()*item.second.point.head<3>()+T_old_W.r()).eval();
    const auto loopInput=fp::absoluteInput(*second,oldPoints,f.extrinsics);
    const auto loop=ps::AbsolutePoseSolver(options()).solve(loopInput.data,&f.relative);
    require(loop.success() && (loop.pose.T()-f.relative.T()).norm()<1e-3,"old-frame loop points return T_BoldBnew directly");
}
void centralScaleAndRotationFallback() {
    Fixture f; const auto first=f.frame(1,f.first),second=f.frame(2,f.first*f.relative);
    ps::RelativePoseCorrespondences central;
    for(size_t k=0;k<std::min(first->numKeypoints(0),second->numKeypoints(0));++k) {
        ps::RelativePoseCorrespondence match;
        require(fp::makeRay(*first,{1,0,k},f.extrinsics,match.ray_S1)
            && fp::makeRay(*second,{2,0,k},f.extrinsics,match.ray_S2),"central rays"); central.push_back(match);
    }
    const auto centralResult=ps::RelativePoseSolver(options()).solve(central,&f.relative);
    require(centralResult.success() && !centralResult.translationScaleObservable,"single-centre translation is explicitly nonmetric");
    const kinematics::Transformation prediction(Eigen::Vector3d(4,-2,1),Eigen::Quaterniond::Identity());
    kinematics::Transformation seed;
    require(fp::relativeWorldSeed(f.first,prediction,centralResult,false,seed)
        && (seed.r()-prediction.r()).norm()==0,"unit camera-centre baseline does not overwrite predicted metric position");
    ps::RelativePoseCorrespondences rotationData;
    for(const auto& point:f.points) {
        ps::RelativePoseCorrespondence match;
        match.ray_S2.direction=point.second.point.head<3>().normalized();
        match.ray_S1.direction=f.relative.C()*match.ray_S2.direction;
        match.ray_S1.origin=Eigen::Vector3d(0.2,0,0); match.ray_S2.origin=Eigen::Vector3d(-0.1,0,0);
        rotationData.push_back(match);
    }
    const auto rotation=ps::RotationPoseSolver(options()).solve(rotationData);
    require(rotation.success() && !rotation.translationScaleObservable && rotation.pose.r().norm()==0
        && rotationError(rotation.pose,f.relative)<1e-6,"Wahba/PoseLib fallback estimates rotation only");
    for(auto& item:rotationData) { item.ray_S1.direction=Eigen::Vector3d::UnitZ(); item.ray_S2.direction=Eigen::Vector3d::UnitZ(); }
    require(!ps::RotationPoseSolver(options()).solve(rotationData).success(),"parallel rays cannot fabricate observable rotation");
}
enum class RecoveryFixture { Clean, NoisyOutliers, WrongPrediction };
void poseRecovery(bool relative,RecoveryFixture fixture) {
    // Independent metric geometry: do not construct a frame with already-correct
    // landmark ownership or pass ground truth as PoseLib's score_initial_model.
    const kinematics::Transformation truth(Eigen::Vector3d(0.55,-0.18,0.22),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.19,Eigen::Vector3d(0.3,0.8,-0.2).normalized())));
    const std::array<Eigen::Vector3d,4> centres{{{0.4,0,0.05},{0,0.35,-0.08},{-0.35,-0.1,0.1},{0.1,-0.4,-0.06}}};
    auto summary=debug::artifactFile("pose_recovery.csv");
    if(summary) summary<<"seed,relative,sigma_radians,outlier_fraction,initial_model_supplied,success,translation_error_m,rotation_error_rad,precision,recall,inliers,iterations,generated_models,rms_angle_rad\n";
    for(unsigned seed:{17u,43u,89u}) {
        const bool contaminated=fixture!=RecoveryFixture::Clean;
        const double sigma=contaminated?1e-4:0;
        std::mt19937 random(seed); std::uniform_real_distribution<double> uniform(-1,1);
        std::normal_distribution<double> noise(0,1);
        const auto noisy=[&](const Eigen::Vector3d& ray) {
            Eigen::Vector3d e(noise(random),noise(random),noise(random));
            e-=ray*ray.dot(e); return Eigen::Vector3d((ray+sigma*e).normalized());
        };
        ps::AbsolutePoseCorrespondences absoluteData;
        ps::RelativePoseCorrespondences relativeData;
        std::vector<uint8_t> outliers;
        for(size_t i=0;i<160;++i) {
            const Eigen::Vector3d point(4*uniform(random),3*uniform(random),4+3*uniform(random));
            const Eigen::Vector3d origin1=centres[i%4],origin2=centres[(i/4+1)%4];
            const Eigen::Vector3d point2=truth.C().transpose()*(point-truth.r());
            ps::AbsolutePoseCorrespondence a; a.point_W=point; a.ray_S.origin=origin2;
            a.ray_S.direction=noisy((point2-origin2).normalized());
            ps::RelativePoseCorrespondence r; r.ray_S1.origin=origin1; r.ray_S2.origin=origin2;
            r.ray_S1.direction=noisy((point-origin1).normalized()); r.ray_S2.direction=a.ray_S.direction;
            const bool outlier=contaminated && i%5==0; // Exactly 20%; keep all centre pairs represented.
            if(outlier) {
                // Fixed large angular corruption; no accidental almost-correct label.
                const Eigen::Vector3d axis=a.ray_S.direction.unitOrthogonal();
                a.ray_S.direction=Eigen::AngleAxisd(0.45+0.2*std::abs(uniform(random)),axis)*a.ray_S.direction;
                r.ray_S2.direction=a.ray_S.direction;
            }
            absoluteData.push_back(a); relativeData.push_back(r); outliers.push_back(uint8_t(outlier));
        }
        auto settings=options(); settings.angularThreshold=1e-3; settings.minInliers=60;
        settings.ransac.max_iterations=2000; settings.ransac.min_iterations=100;
        settings.ransac.success_prob=0.999; settings.ransac.seed=seed;
        const kinematics::Transformation wrong=truth*kinematics::Transformation(Eigen::Vector3d(0.8,-0.5,0.4),
            Eigen::Quaterniond(Eigen::AngleAxisd(0.3,Eigen::Vector3d::UnitX())));
        const auto* initial=fixture==RecoveryFixture::WrongPrediction?&wrong:nullptr;
        if(initial) require(rotationError(*initial,truth)>0.2 && (initial->r()-truth.r()).norm()>0.5,
            "wrong-prediction fixture must start outside the final acceptance bounds");
        const auto result=relative?ps::RelativePoseSolver(settings).solve(relativeData,initial)
            :ps::AbsolutePoseSolver(settings).solve(absoluteData,initial);
        std::vector<uint8_t> selected(outliers.size(),0); size_t truePositive=0,falsePositive=0;
        for(size_t i:result.inliers) {
            require(i<selected.size() && !selected[i],"recovery inlier indices must be original, unique rows");
            selected[i]=1; if(outliers[i]) ++falsePositive; else ++truePositive;
        }
        const double good=double(std::count(outliers.begin(),outliers.end(),uint8_t(0)));
        const double precision=result.inliers.empty()?0:double(truePositive)/result.inliers.size();
        const double recall=double(truePositive)/good;
        const double translation=(result.pose.r()-truth.r()).norm(),rotation=rotationError(result.pose,truth);
        if(summary) { summary<<seed<<','<<relative<<','<<sigma<<','<<(contaminated?0.2:0)<<','<<(initial!=nullptr)<<','<<result.success()
            <<','<<translation<<','<<rotation<<','<<precision<<','<<recall<<','<<result.inliers.size()<<','<<result.iterations
            <<','<<result.generatedModels<<','<<result.rmsAngularError<<'\n'; summary.flush(); }
        auto rows=debug::artifactFile("seed_"+std::to_string(seed)+"_correspondences.csv");
        if(rows) {
            rows<<"row,cx1,cy1,cz1,bx1,by1,bz1,cx2,cy2,cz2,bx2,by2,bz2,px,py,pz,labelled_outlier,selected_inlier\n";
            for(size_t i=0;i<outliers.size();++i) {
                rows<<i;
                for(const Eigen::Vector3d& v:{relativeData[i].ray_S1.origin,relativeData[i].ray_S1.direction,
                    relativeData[i].ray_S2.origin,relativeData[i].ray_S2.direction,absoluteData[i].point_W})
                    for(int k=0;k<3;++k) rows<<','<<v[k];
                rows<<','<<int(outliers[i])<<','<<int(selected[i])<<'\n';
            }
        }
        require(result.success() && result.generatedModels>0 && result.iterations>0,
            "minimal solver and LO-RANSAC must recover without a ground-truth initial model");
        require(!relative || result.translationScaleObservable,"diverse camera centres must recover metric translation");
        const std::string prefix="seed_"+std::to_string(seed);
        debug::recordError(prefix+"_translation_error_m",translation,contaminated?(relative?0.03:0.01):1e-5);
        debug::recordError(prefix+"_rotation_error_rad",rotation,contaminated?0.003:1e-6);
        debug::recordError(prefix+"_false_positive_fraction",1-precision,0.02);
        debug::recordError(prefix+"_missed_inlier_fraction",1-recall,0.10);
    }
}
// Check the actual pose-solver update chart, including a nonzero world origin
// and rear-hemisphere observations. This is independent of backend SE3 lifting.
void absoluteAnalyticLinearizationAgreement() {
    auto rows=debug::artifactFile("absolute_analytic_jacobians.csv");
    if(rows) rows<<"case,column,absolute_error,reference_norm\n";
    for(double angle:{0.0,1e-9,1e-5,0.02,0.6,1.8}) for(double sign:{-1.0,1.0}) {
        kinematics::Transformation T(Eigen::Vector3d(1.1,-0.7,0.3),
            Eigen::Quaterniond(Eigen::AngleAxisd(0.2,Eigen::Vector3d(1,2,3).normalized())));
        const Eigen::Vector3d observed=Eigen::Vector3d(0.2,-0.4,sign).normalized(),origin(0.3,-0.2,0.1);
        const Eigen::Vector3d predicted=4.0*(Eigen::AngleAxisd(angle,observed.unitOrthogonal())*observed);
        const Eigen::Vector3d point=T.C().transpose()*(predicted-T.r()+origin),rotated=T.C()*point;
        Eigen::Vector3d r; Eigen::Matrix3d D;
        require(ps::detail::sphericalLinearization(predicted,observed,r,D),"S2 analytic linearization must support both hemispheres");
        Eigen::Matrix<double,3,6> analytic; analytic.leftCols<3>()=D; analytic.rightCols<3>()=-D*ps::detail::crossMatrix(rotated);
        for(int column=0;column<6;++column) {
            ps::detail::Vector6 step=ps::detail::Vector6::Zero(); const double h=1e-7; step[column]=h;
            auto plus=T,minus=T; plus.oplus(step); minus.oplus(-step); Eigen::Vector3d a,b;
            require(ps::detail::sphericalResidual(plus.C()*point+plus.r()-origin,observed,a)
                && ps::detail::sphericalResidual(minus.C()*point+minus.r()-origin,observed,b),"central S2 reference must evaluate");
            const Eigen::Vector3d reference=(a-b)/(2*h); const double error=(analytic.col(column)-reference).norm();
            if(rows) rows<<angle*sign<<','<<column<<','<<error<<','<<reference.norm()<<'\n';
            debug::recordDerivative("absolute_chart_"+std::to_string(angle*sign)+"_"+std::to_string(column),error,reference.norm(),2e-8,2e-7);
        }
    }
    Eigen::Vector3d r; Eigen::Matrix3d D;
    require(!ps::detail::sphericalLinearization(-Eigen::Vector3d::UnitZ(),Eigen::Vector3d::UnitZ(),r,D),"antipodal log remains undefined");
}
void absoluteAnalyticRecoveryAndBudgetGuard() {
    // A physical linear residual with an already negligible optimal update must
    // stop before performing another inlier-set trial evaluation.
    ps::detail::Pose tiny; ps::PoseSolverTiming timing; auto stopping=options();
    const ps::detail::ResidualFunction residual=[](const auto& p,size_t,Eigen::VectorXd& r) { r=Eigen::VectorXd::Constant(1,p.r().x()+1e-11); return true; };
    const ps::detail::LinearizationFunction linear=[](const auto& p,size_t,Eigen::VectorXd& r,Eigen::MatrixXd& j) {
        r=Eigen::VectorXd::Constant(1,p.r().x()+1e-11); j=Eigen::MatrixXd::Zero(1,6); j(0,0)=1; return true;
    };
    require(ps::detail::refine(tiny,{0},1,stopping,residual,[](auto&) { return true; },linear,&timing)
        && timing.smallStepStops==1 && timing.acceptedLoSteps==0,"vanishing LO update must terminate without a trial traversal");

    ps::AbsolutePoseCorrespondences data;
    const kinematics::Transformation truth(Eigen::Vector3d(0.2,-0.1,0.3),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.08,Eigen::Vector3d::UnitY())));
    std::mt19937 random(733); std::uniform_real_distribution<double> uniform(-1,1);
    for(int i=0;i<100;++i) {
        ps::AbsolutePoseCorrespondence m; m.point_W=Eigen::Vector3d(3*uniform(random),2*uniform(random),5+uniform(random));
        m.ray_S.origin=Eigen::Vector3d(0.2*(i%4),0.1*(i%2),0);
        const Eigen::Vector3d p=truth.C().transpose()*(m.point_W-truth.r())-m.ray_S.origin;
        m.ray_S.direction=p.normalized(); data.push_back(m);
    }
    fp::Options guard; require(fp::absolutePredictionHealthy(data,truth,guard),"clean well-supported prediction permits reduced sampling floor");
    const kinematics::Transformation wrong(Eigen::Vector3d(0.8,-0.5,0.7),Eigen::Quaterniond::Identity());
    require(!fp::absolutePredictionHealthy(data,wrong,guard),"wrong prediction must retain full RANSAC budget");
    require(!fp::absolutePredictionHealthy(ps::AbsolutePoseCorrespondences(data.begin(),data.begin()+20),truth,guard),"small support must retain full budget");
    for(size_t i=0;i<data.size();++i) {
        const double angle=i%5==0?0.4:1e-4*uniform(random);
        data[i].ray_S.direction=Eigen::AngleAxisd(angle,data[i].ray_S.direction.unitOrthogonal())*data[i].ray_S.direction;
    }
    require(!fp::absolutePredictionHealthy(data,truth,guard),"outlier-rich prediction must retain full budget");
    auto analytic=options(),numeric=analytic; analytic.angularThreshold=1e-3; numeric=analytic;
    numeric.analyticAbsoluteJacobians=false;
    const auto a=ps::AbsolutePoseSolver(analytic).solve(data,&wrong),b=ps::AbsolutePoseSolver(numeric).solve(data,&wrong);
    require(a.success() && b.success(),"analytic and numeric LO must independently recover a noisy contaminated pose");
    require(a.inliers==b.inliers,"analytic LO must preserve clean/outlier classification");
    debug::recordError("analytic_numeric_translation",(a.pose.r()-b.pose.r()).norm(),2e-5);
    debug::recordError("analytic_numeric_rotation",rotationError(a.pose,b.pose),2e-6);
    require(a.timing.jacobianEvaluations>0 && b.timing.jacobianEvaluations==0,"analytic/reference paths must be distinguishable in diagnostics");
    require(a.timing.residualEvaluations<b.timing.residualEvaluations,"analytic LO must reduce point residual evaluations on the same fixture");
}
void rotationAnalyticAgreementAndRecovery() {
    auto rows=debug::artifactFile("rotation_analytic_jacobians.csv");
    if(rows) rows<<"angle,hemisphere,column,absolute_error,reference_norm\n";
    for(double angle:{0.0,1e-9,0.02,0.6,1.8}) for(double sign:{-1.0,1.0}) {
        const kinematics::Transformation T(Eigen::Vector3d(2,-1,3),
            Eigen::Quaterniond(Eigen::AngleAxisd(0.2,Eigen::Vector3d(1,2,3).normalized())));
        const Eigen::Vector3d observed=Eigen::Vector3d(0.2,-0.4,sign).normalized();
        const Eigen::Vector3d predicted=Eigen::AngleAxisd(angle,observed.unitOrthogonal())*observed;
        const Eigen::Vector3d ray=T.C().transpose()*predicted;
        Eigen::Vector3d residual; Eigen::Matrix3d D;
        require(ps::detail::sphericalLinearization(predicted,observed,residual,D),"rotation S2 derivative must exist on both hemispheres");
        Eigen::Matrix<double,3,6> J; J.leftCols<3>().setZero(); J.rightCols<3>()=-D*ps::detail::crossMatrix(predicted);
        for(int column=0;column<6;++column) {
            ps::detail::Vector6 step=ps::detail::Vector6::Zero(); step[column]=1e-7;
            auto plus=T,minus=T; plus.oplus(step); minus.oplus(-step); Eigen::Vector3d a,b;
            require(ps::detail::sphericalResidual(plus.C()*ray,observed,a)
                && ps::detail::sphericalResidual(minus.C()*ray,observed,b),"rotation chart numeric reference");
            const Eigen::Vector3d reference=(a-b)/2e-7; const double error=(J.col(column)-reference).norm();
            if(rows) rows<<angle<<','<<sign<<','<<column<<','<<error<<','<<reference.norm()<<'\n';
            debug::recordDerivative("rotation_chart_"+std::to_string(angle*sign)+"_"+std::to_string(column),error,reference.norm(),2e-8,2e-7);
        }
    }
    const kinematics::Transformation truth(Eigen::Vector3d::Zero(),Eigen::Quaterniond(Eigen::AngleAxisd(0.17,Eigen::Vector3d(1,-2,3).normalized())));
    ps::RelativePoseCorrespondences data;
    for(int i=0;i<80;++i) {
        ps::RelativePoseCorrespondence match;
        match.ray_S2.direction=Eigen::Vector3d(std::sin(0.13*i),std::cos(0.19*i),std::cos(0.37*i)).normalized();
        match.ray_S1.direction=truth.C()*match.ray_S2.direction;
        match.ray_S1.direction=Eigen::AngleAxisd(i%7==0?0.2:1e-5,match.ray_S1.direction.unitOrthogonal())*match.ray_S1.direction;
        data.push_back(match);
    }
    auto analytic=options(),numeric=analytic; numeric.analyticRotationJacobians=false;
    const auto a=ps::RotationPoseSolver(analytic).solve(data),b=ps::RotationPoseSolver(numeric).solve(data);
    require(a.success() && b.success() && a.inliers==b.inliers,"analytic and numeric rotation LO must recover the same contaminated full-sphere consensus");
    debug::recordError("rotation_recovery_truth",rotationError(a.pose,truth),1e-4);
    debug::recordError("rotation_analytic_numeric",rotationError(a.pose,b.pose),2e-6);
    require(a.timing.jacobianEvaluations>0 && b.timing.jacobianEvaluations==0
        && a.timing.residualEvaluations<b.timing.residualEvaluations,"rotation analytic LO must remove finite-difference residual calls");
}
void absoluteUnseededRecovery() { poseRecovery(false,RecoveryFixture::Clean); }
void absoluteNoisyOutlierRecovery() { poseRecovery(false,RecoveryFixture::NoisyOutliers); }
void absoluteWrongPredictionRecovery() { poseRecovery(false,RecoveryFixture::WrongPrediction); }
void relativeUnseededRecovery() { poseRecovery(true,RecoveryFixture::Clean); }
void relativeNoisyOutlierRecovery() { poseRecovery(true,RecoveryFixture::NoisyOutliers); }
void relativeWrongPredictionRecovery() { poseRecovery(true,RecoveryFixture::WrongPrediction); }
}
#ifdef CGMLI_EMBED_MODULE_TESTS
void cgmli::debug::registerPoseCases(std::vector<cgmli::debug::ModuleCase>& cases) {
    cases.push_back({"jacobian","rotationAnalyticAgreementAndRecovery",[]() { rotationAnalyticAgreementAndRecovery(); }});
    cases.push_back({"jacobian","absoluteAnalyticLinearizationAgreement",[]() { absoluteAnalyticLinearizationAgreement(); }});
    cases.push_back({"tracking","absoluteAnalyticRecoveryAndBudgetGuard",[]() { absoluteAnalyticRecoveryAndBudgetGuard(); }});
    cases.push_back({"tracking","minimalSamplingBuffers",[]() { minimalSamplingBuffers(); }});
    cases.push_back({"tracking","absoluteMappingAndBodyConvention",[]() { absoluteMappingAndBodyConvention(); }});
    cases.push_back({"tracking","pooledRelativeAndLoopFrames",[]() { pooledRelativeAndLoopFrames(); }});
    cases.push_back({"tracking","centralScaleAndRotationFallback",[]() { centralScaleAndRotationFallback(); }});
    cases.push_back({"tracking","absoluteUnseededRecovery",[]() { absoluteUnseededRecovery(); }});
    cases.push_back({"tracking","absoluteNoisyOutlierRecovery",[]() { absoluteNoisyOutlierRecovery(); }});
    cases.push_back({"tracking","absoluteWrongPredictionRecovery",[]() { absoluteWrongPredictionRecovery(); }});
    cases.push_back({"tracking","relativeUnseededRecovery",[]() { relativeUnseededRecovery(); }});
    cases.push_back({"tracking","relativeNoisyOutlierRecovery",[]() { relativeNoisyOutlierRecovery(); }});
    cases.push_back({"tracking","relativeWrongPredictionRecovery",[]() { relativeWrongPredictionRecovery(); }});
}
#else
int main() {
    try { rotationAnalyticAgreementAndRecovery(); absoluteAnalyticLinearizationAgreement(); absoluteAnalyticRecoveryAndBudgetGuard(); minimalSamplingBuffers(); absoluteMappingAndBodyConvention(); pooledRelativeAndLoopFrames(); centralScaleAndRotationFallback();
        absoluteUnseededRecovery(); absoluteNoisyOutlierRecovery(); absoluteWrongPredictionRecovery();
        relativeUnseededRecovery(); relativeNoisyOutlierRecovery(); relativeWrongPredictionRecovery();
        std::cout<<"frontend pose regression passed\n"; return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}

#endif // CGMLI_EMBED_MODULE_TESTS
