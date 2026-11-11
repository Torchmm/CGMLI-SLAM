// Standalone numerical/integration regressions. Not compiled or executed by the
// static-only workflow; register a dedicated test target before running these.
#include <debug/ModuleDebug.hpp>
#include <debug/ModuleDebugArtifacts.hpp>
#include <pipeline/ThreadedSlam.hpp>
#include <factor/BodyImuFactor.hpp>
#include <factor/ImuError.hpp>
#include <backend/BackendFactors.hpp>
#include <backend/FrameStore.hpp>
#include <backend/ProjectBackendAdapter.hpp>
#include <limits>
#include <cameras/PinholeCamera.hpp>
#include <cameras/EucmCamera.hpp>
#include <common/ViParametersReader.hpp>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <mutex>

namespace cgmli {
struct ThreadedSlamRegressionAccess {
    static bool finish(ThreadedSlam& slam) { return slam.waitForOptimization(); }
    static bool takeImages(ThreadedSlam& slam,std::vector<CameraMeasurement>& packet) { return slam.cameraQueue_.PopNonBlocking(&packet); }
    static bool calibrationRetryKeepsEligibleBlocks() {
        using backend::BlockKind;
        ThreadedSlam::CameraCalibrationDecision joint;
        joint.rotation=true; joint.translation=true; joint.timeOffset=true; joint.phase="CALIB_FULL";
        const auto rigFixed=ThreadedSlam::calibrationRetryDecision(joint,{{BlockKind::CameraExtrinsic,0}});
        const auto clockFixed=ThreadedSlam::calibrationRetryDecision(joint,{{BlockKind::CameraTimeOffset,0}});
        const auto bothFixed=ThreadedSlam::calibrationRetryDecision(joint,
            {{BlockKind::CameraTimeOffset,0},{BlockKind::CameraExtrinsic,0}});
        return !rigFixed.rotation && !rigFixed.translation && rigFixed.timeOffset
            && clockFixed.rotation && clockFixed.translation && !clockFixed.timeOffset
            && !bothFixed.rotation && !bothFixed.translation && !bothFixed.timeOffset;
    }
    static bool motionCalibrationGate(ThreadedSlam& slam) {
        using backend::SensorHealth;
        std::vector<cv::KeyPoint> points;
        for (int i=0; i<20; ++i) points.emplace_back(50.0f+3*i,80.0f,31.0f);
        std::string error;
        const Eigen::Vector3d gyro[]{{0.2,0,0},{0,0.2,0},{-0.2,0,0},{0,-0.2,0},{0,0,0}};
        for (int i=0; i<5; ++i) {
            const uint64_t id=i+1;
            Time stamp; stamp.fromNSec(1000000000ULL+100000000ULL*i);
            MultiFrame frame(slam.parameters_.nCameraSystem,stamp,id);
            if (i==0 || i==4) frame.resetKeypoints(0,points);
            backend::StateEstimate state; state.id=StateId(id); state.timestamp=stamp;
            state.T_WB=kinematics::Transformation(Eigen::Vector3d(0.04*i,0.005*i*i,0),Eigen::Quaterniond::Identity());
            state.speedBias.head<3>()=Eigen::Vector3d(0.2+0.08*i,0.015*i*i,0);
            state.visual=SensorHealth::Valid; state.keyframe=i==0;
            if (!slam.estimator_.addStates(frame,state,{},error)) throw std::runtime_error(error);
            slam.gyroByFrame_.emplace(state.id,gyro[i]);
        }
        const auto appendTracks=[&](size_t begin,size_t end,bool timed) {
            backend::ObservationOptions options;
            // All image measurements have y=cy, so the camera's Y axis is tangent.
            if (timed) options.bearingVelocity.y()=0.1;
            for (size_t i=begin; i<end; ++i) {
                const LandmarkId id(i+1);
                if (!slam.estimator_.addLandmark(id,{1,0,i},0.2,error)
                    || !slam.estimator_.addObservation(id,StateId(5),0,i,error,options)) throw std::runtime_error(error);
            }
            slam.pendingVisualLandmarks_=end;
        };
        appendTracks(0,10,false);
        const auto rotation=slam.cameraCalibrationDecision();
        appendTracks(10,20,true);
        const auto joint=slam.cameraCalibrationDecision();
        slam.rotationCalibrationUpdates_=3; slam.timeCalibrationUpdates_=3;
        const auto full=slam.cameraCalibrationDecision();
        for (auto& sample : slam.gyroByFrame_) sample.second.setZero();
        const auto clock=slam.cameraCalibrationDecision();
        if (!slam.estimator_.setHealth(StateId(5),SensorHealth::Degraded,SensorHealth::Invalid))
            throw std::runtime_error("cannot mark the test epoch degraded");
        const auto frozen=slam.cameraCalibrationDecision();
        return rotation.rotation && !rotation.timeOffset && !rotation.translation
            && joint.rotation && joint.timeOffset && !joint.translation
            && full.rotation && full.timeOffset && full.translation
            && !clock.rotation && clock.timeOffset && !clock.translation
            && !frozen.rotation && !frozen.timeOffset && !frozen.translation;
    }
    static void injectFailingFrameAndPrepareNext(ThreadedSlam& slam,const Time& failedTime,const Time& nextTime,const cv::Mat& image) {
        const auto prepare=[&](const Time& time) {
            auto frame=std::make_shared<MultiFrame>(slam.parameters_.nCameraSystem,time,slam.nextId_++);
            for(size_t c=0;c<frame->numFrames();++c) frame->setImage(c,image);
            State state{};
            if(!slam.collectImu(time) || !slam.predict(*frame,state)) throw std::runtime_error("regression prediction failed");
            slam.extract(frame,state); return std::make_pair(frame,state);
        };
        const auto failed=prepare(failedTime);
        if(!slam.stage(failed.first,failed.second)) throw std::runtime_error("regression staging failed");
        // A negative time budget is deterministically rejected before Ceres runs.
        // Only the control thread writes task inputs, before launch.
        slam.pendingBudget_.timeSeconds=-1.0;
        slam.launchOptimization();
        const auto next=prepare(nextTime);
        slam.preparedFrame_=next.first; slam.preparedPrediction_=next.second;
        slam.preparedFromState_=slam.latestLocal_.id;
    }
};
}

using namespace cgmli;
namespace {
void require(bool value,const char* reason) { if(!value) throw std::runtime_error(reason); }
Time timeAt(double seconds) { Time t; t.fromSec(seconds); return t; }
ViParameters parameters() {
    ViParameters p{};
    p.nCameraSystem.addCamera(std::make_shared<kinematics::Transformation>(),
        std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(160,160,100,100,80,80,cameras::NoDistortion()),cameras::NCameraSystem::NoDistortion,false);
    p.nCameraSystem.addCamera(std::make_shared<kinematics::Transformation>(Eigen::Vector3d(0.2,0,0),Eigen::Quaterniond::Identity()),
        std::make_shared<cameras::EucmCamera>(160,160,80,80,80,80,0.6,1.0),cameras::NCameraSystem::NoDistortion,false);
    p.imu.use=true; p.imu.g=9.81; p.imu.g_max=100; p.imu.a_max=100;
    p.imu.sigma_g_c=0.002; p.imu.sigma_a_c=0.02; p.imu.sigma_gw_c=0.0002; p.imu.sigma_aw_c=0.002;
    p.imu.sigma_bg=0.01; p.imu.sigma_ba=0.1; p.imu.s_a.setOnes(); p.imu.g0.setZero(); p.imu.a0.setZero();
    p.frontend.max_num_keypoints=200; p.frontend.orb_pyramid_levels=1;
    p.estimator.window_size=3; p.estimator.realtime_max_iterations=6;
    p.estimator.realtime_num_threads=1; p.estimator.full_graph_num_threads=1; p.estimator.full_graph_iterations=10;
    return p;
}
ImuMeasurementDeque samples(double end) {
    ImuMeasurementDeque data;
    for(int i=0;i<=int(end*200);++i) data.emplace_back(timeAt(i/200.0),ImuSensorReadings(Eigen::Vector3d::Zero(),Eigen::Vector3d(0,0,9.81)));
    return data;
}
void averagedImageInputPreservesTimes() {
    auto p=parameters(); p.camera.image_delay=0.002; ThreadedSlam slam(p,"");
    const cv::Mat image(160,160,CV_8UC1,cv::Scalar(80));
    Time mean,early,late; mean.fromNSec(1000000000); early.fromNSec(994999000); late.fromNSec(1005001000);
    require(slam.addImagesWithTimestamps(mean,{{0,image},{1,image}},{{0,early},{1,late}}),"submit averaged image packet");
    std::vector<CameraMeasurement> packet;
    require(ThreadedSlamRegressionAccess::takeImages(slam,packet) && packet.size()==2,"image packet must retain configured camera slots");
    for(const auto& camera:packet) require(!camera.measurement.image.empty() && camera.timeStamp.toNSec()==1002000000,
        "a positive image_delay moves the common node 2 ms later");
    require(packet[0].measurement.originalImageTimestamp->toNSec()==early.toNSec()
        && packet[0].measurement.imageTimeOffsetNs==-5001000
        && packet[1].measurement.imageTimeOffsetNs==5001000,"raw midpoint times and signed offsets must survive enqueue");
    require(!slam.addImagesWithTimestamps(mean,{{0,image},{1,image}},{{0,early}}),"partial timestamp metadata rejected");
    require(slam.addImages(mean,{{0,image},{1,image}}),"legacy common-time API remains valid");
    require(ThreadedSlamRegressionAccess::takeImages(slam,packet) && packet[0].measurement.imageTimeOffsetNs==0,
        "legacy API uses zero group offset");
    p.camera.image_delay=-0.002;
    require(p.camera.correctedImageTime(mean).toNSec()==998000000,
        "a negative image_delay moves the common node 2 ms earlier");
}
void calibrationRetryFreezesOnlyRejectedBlocks() {
    require(ThreadedSlamRegressionAccess::calibrationRetryKeepsEligibleBlocks(),
        "retry keeps calibration blocks that are inside their own ranges variable");
}
void motionCalibrationRespectsTrackSupport() {
    auto p=parameters(); p.estimator.window_size=5;
    auto& calibration=p.camera.online_calibration;
    calibration.do_extrinsics=true; calibration.estimate_td_camera=true;
    calibration.sigma_r=0.01; calibration.sigma_alpha=0.01;
    ThreadedSlam slam(p,"");
    require(ThreadedSlamRegressionAccess::motionCalibrationGate(slam),
        "motion gate releases rotation, time and translation only with their respective support");
}
void featureDebugSnapshotsAndRendering() {
    const auto p=parameters(); MultiFrame previous(p.nCameraSystem,timeAt(1),1),current(p.nCameraSystem,timeAt(2),2);
    cv::Mat source(160,160,CV_8UC1,cv::Scalar(80));
    for(size_t c=0;c<2;++c) {
        previous.setImage(c,source); current.setImage(c,source);
        previous.resetKeypoints(c,{cv::KeyPoint(20,20,31),cv::KeyPoint(60,60,31)});
        current.resetKeypoints(c,{cv::KeyPoint(20,20,31),cv::KeyPoint(60,60,31)});
    }
    current.setImageTiming(0,timeAt(2.005),5000000);
    ViVisualizer::FeatureDebugData data; data.current=ViVisualizer::captureFeatures(current);
    data.sources.emplace(1,ViVisualizer::captureFeatures(previous));
    data.matches={{{2,0,0},{2,1,0},10,true},{{2,0,1},{2,1,1},50,false},{{1,0,0},{2,0,0},8,true}};
    source.setTo(0);
    require(cv::countNonZero(data.current->cameras[0].image)==160*160,"debug snapshot must own pixels independently of the producer");
    require(data.current->cameras[0].timeOffsetNs==5000000,"debug snapshot retains mean-time offset metadata");
    const auto views=ViVisualizer::drawFeatureDebug(data);
    require(views.size()==3 && !views.at("01 ORB Features").empty() && !views.at("03 Temporal Matches").empty(),"feature/stereo/temporal dashboards must render");
    cv::Mat green,red;
    cv::inRange(views.at("02 Stereo Matches"),cv::Scalar(0,150,0),cv::Scalar(100,255,100),green);
    cv::inRange(views.at("02 Stereo Matches"),cv::Scalar(0,0,150),cv::Scalar(120,120,255),red);
    require(cv::countNonZero(green)>0 && cv::countNonZero(red)>0,"kept and discarded candidate classes must both be visible");
    data.sources.clear(); data.matches.push_back({{2,0,9999},{2,1,9999},60,false});
    require(!ViVisualizer::drawFeatureDebug(data).at("03 Temporal Matches").empty(),"missing reference images and invalid diagnostic indices must not crash rendering");
    require(cv::countNonZero(data.current->cameras[0].image)==160*160,"rendering must not paint into cached input thumbnails");
    ViVisualizer::VisualizationData accepted;
    accepted.acceptedFrame=2; accepted.currentFrames=backend::FrameStore::copyMeasurements(current,true);
    accepted.currentFrames->setLandmarkId(0,0,42);
    accepted.currentFrames->setLandmark(0,0,Eigen::Vector4d(0,0,4,1),true);
    accepted.trajectory={{1,Eigen::Vector3d::Zero(),true},{2,Eigen::Vector3d(1,0.2,0.1),false}};
    accepted.window=accepted.trajectory;
    accepted.trackLifecycles.emplace(42,ViVisualizer::VisualizationData::TrackLifecycle{2,3,1,2});
    const auto untouched=accepted.currentFrames->image(0).clone();
    auto visualParameters=p; ViVisualizer visualizer(visualParameters);
    const auto trajectory=ViVisualizer::drawTrajectory(accepted),tracks=visualizer.drawTrackTrails(accepted);
    require(!trajectory.empty() && !tracks.empty(),"accepted trajectory and identity trails render");
    require(cv::norm(untouched,accepted.currentFrames->image(0),cv::NORM_INF)==0,"trail rendering cannot modify input image");
    debug::saveImage("accepted_trajectory.png",trajectory); debug::saveImage("accepted_feature_tracks.png",tracks);
}
class CalibrationReaderProbe : public ViParametersReader {
public:
    bool accepts(const std::string& yaml) {
        cv::FileStorage file(yaml,cv::FileStorage::READ|cv::FileStorage::MEMORY);
        std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> values;
        return getCalibrationViaConfig(values,file["cameras"]);
    }
};
void modelSpecificCameraCalibration() {
    CalibrationReaderProbe reader;
    const std::string common="%YAML:1.0\ncameras:\n  - T_SC: [1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.]\n"
        "    image_dimension: [160,160]\n    focal_length: [100.,100.]\n    principal_point: [80.,80.]\n";
    require(reader.accepts(common+"    cam_model: eucm\n    eucm_parameters: [0.6,1.]\n"),"EUCM must not require a pinhole distortion vector");
    require(reader.accepts(common+"    cam_model: pinhole\n    distortion_type: None\n"),"NoDistortion accepts an empty distortion vector");
    require(!reader.accepts(common+"    cam_model: unknown\n"),"unknown camera model must fail before stream renumbering");
    require(!reader.accepts(common+"    cam_model: eucm\n    eucm_parameters: [1.2,1.]\n"),"invalid EUCM domain rejected");
    require(!reader.accepts(common+"    cam_model: pinhole\n    distortion_type: radialtangential8\n    distortion_coefficients: [0.,0.,0.,0.]\n"),"eight-parameter distortion cannot index a four-element vector");
}
void imuGapAndAtomicPropagation() {
    auto p=parameters(); auto data=samples(1.0);
    data.erase(std::remove_if(data.begin(),data.end(),[](const auto& m){return m.timeStamp>timeAt(0.55) && m.timeStamp<timeAt(0.65);}),data.end());
    std::string error;
    require(!factor::validImuInterval(data,timeAt(0.5),timeAt(0.7),error,p.imu.max_sample_gap),"interior IMU gap rejected");
    require(factor::validImuInterval(data,timeAt(0.8),timeAt(0.9),error,p.imu.max_sample_gap),"old cached gap outside the requested interval is irrelevant");
    kinematics::Transformation pose; SpeedAndBias sb=SpeedAndBias::Zero(); const auto before=pose.T();
    require(!factor::propagateImu(data,p.imu,pose,sb,timeAt(0.5),timeAt(0.7)),"gap cannot become a valid propagation");
    require((pose.T()-before).norm()==0 && sb.norm()==0,"gap failure does not mutate output state");
}
void inertialPoseAcceptanceBounds() {
    const auto p=parameters(); kinematics::Transformation prediction;
    const kinematics::Transformation reasonable(Eigen::Vector3d(0.1,0,0),Eigen::Quaterniond(Eigen::AngleAxisd(0.01,Eigen::Vector3d::UnitX())));
    require(frontend_pose::inertialPoseConsistent(prediction,reasonable,0.1,p.estimator),"small inertial correction accepted");
    const kinematics::Transformation wrongRotation(Eigen::Vector3d::Zero(),Eigen::Quaterniond(Eigen::AngleAxisd(0.5,Eigen::Vector3d::UnitY())));
    require(!frontend_pose::inertialPoseConsistent(prediction,wrongRotation,0.1,p.estimator),"large rotation disagreement rejected");
    const kinematics::Transformation wrongTranslation(Eigen::Vector3d(3,0,0),Eigen::Quaterniond::Identity());
    require(!frontend_pose::inertialPoseConsistent(prediction,wrongTranslation,0.1,p.estimator),"large metric position disagreement rejected");
}
void failedSolveRollbackPreservesNextFrame() {
    for(const int retries:{0,1,3}) {
        auto p=parameters(); p.estimator.backend_max_retries=retries;
        std::atomic<int> count{0}; std::atomic<bool> publishedRejected{false}; std::atomic<uint64_t> latest{0};
        ThreadedSlam slam(p,""); slam.setBlocking(true);
        slam.setOptimisedGraphCallback([&](const State& state,const TrackingState&,std::shared_ptr<const AlignedMap<StateId,State>>,std::shared_ptr<const MapPointVector>) {
            if(state.id.value()==2) publishedRejected=true;
            latest=state.id.value(); ++count;
        });
        for(const auto& m:samples(1.0)) require(slam.addImuMeasurement(m.timeStamp,m.measurement.accelerometers,m.measurement.gyroscopes),"enqueue rollback IMU");
        cv::Mat blank(160,160,CV_8UC1,cv::Scalar(128));
        require(slam.addImages(timeAt(0.5),{{0,blank},{1,blank}}) && slam.processFrame(),"stage accepted reference");
        require(ThreadedSlamRegressionAccess::finish(slam),"commit reference before injected failure");
        ThreadedSlamRegressionAccess::injectFailingFrameAndPrepareNext(slam,timeAt(0.55),timeAt(0.6),blank);
        for(int attempt=0;attempt<retries;++attempt) {
            require(!slam.processFrame() && slam.hasPendingImages(),"retryable rejection preserves extracted next frame");
            require(slam.processFrame(),"launch a bounded retry");
        }
        require(slam.processFrame(),"exhausted retry rolls back and stages the next frame");
        require(ThreadedSlamRegressionAccess::finish(slam),"next frame commits across the dropped insertion");
        slam.stopThreading();
        require(count.load()==2 && latest.load()==3 && !publishedRejected.load(),"only accepted frames 1 and 3 are published");
        require(!slam.hasPendingWork(),"failed input does not occupy the writer slot forever");
    }
}
void imuGapStopsCurrentSegment() {
    auto p=parameters(); auto data=samples(1.0);
    data.erase(std::remove_if(data.begin(),data.end(),[](const auto& m){return m.timeStamp>timeAt(0.6) && m.timeStamp<timeAt(0.7);}),data.end());
    ThreadedSlam slam(p,""); slam.setBlocking(true);
    for(const auto& m:data) require(slam.addImuMeasurement(m.timeStamp,m.measurement.accelerometers,m.measurement.gyroscopes),"enqueue gap test IMU");
    cv::Mat blank(160,160,CV_8UC1,cv::Scalar(128));
    require(slam.addImages(timeAt(0.5),{{0,blank},{1,blank}}) && slam.processFrame(),"initialize before physical gap");
    require(ThreadedSlamRegressionAccess::finish(slam),"accept pre-gap estimate");
    require(slam.addImages(timeAt(0.75),{{0,blank},{1,blank}}),"enqueue image after gap");
    require(!slam.processFrame() && slam.hasFatalError(),"physical gap terminates the current segment explicitly");
    require(!slam.addImages(timeAt(0.8),{{0,blank}}),"faulted segment rejects further input");
    slam.stopThreading();
}
void eucmBackProjectionJacobian() {
    cameras::EucmCamera camera(640,480,220,210,320,240,0.6,1.0);
    for(const Eigen::Vector2d pixel:std::vector<Eigen::Vector2d>{{320,240},{100,100},{550,390}}) {
        Eigen::Vector3d ray; Eigen::Matrix<double,3,2> analytic,numeric;
        require(camera.backProject(pixel,&ray,&analytic),"EUCM valid pixel");
        for(int c=0;c<2;++c) {
            Eigen::Vector2d plus=pixel,minus=pixel; plus[c]+=1e-4; minus[c]-=1e-4; Eigen::Vector3d a,b;
            require(camera.backProject(plus,&a) && camera.backProject(minus,&b),"EUCM derivative samples"); numeric.col(c)=(a-b)/2e-4;
        }
        require(analytic.norm()>0,"EUCM bearing derivative must not vanish");
        debug::recordDerivative("eucm_pixel_"+std::to_string(int(pixel.x()))+"_"+std::to_string(int(pixel.y())),
            (analytic-numeric).norm(),numeric.norm(),1e-6,1e-10);
    }
}
void stationaryPoseEntryRejectsMotion() {
    const auto data=samples(0.5);
    cgmli::ceres::ImuStationarityOptions options;
    cgmli::ceres::ImuStationarityStatistics statistics;
    const kinematics::Transformation original(Eigen::Vector3d(1,2,3),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.3,Eigen::Vector3d::UnitX())));
    const auto reject=[&](const ImuMeasurementDeque& input,const char* label) {
        auto pose=original;
        require(!cgmli::ceres::ImuError::initPose(input,pose,options,&statistics),label);
        require((pose.T()-original.T()).norm()==0 && !statistics.reason.empty(),"failed direct init preserves pose and provides reason");
    };
    reject({},"empty direct initialization");
    reject(ImuMeasurementDeque(data.begin(),data.begin()+10),"short duration cannot prove stationarity");
    auto invalid=data; invalid[1].timeStamp=invalid[0].timeStamp; reject(invalid,"duplicate timestamp");
    invalid=data; invalid.erase(invalid.begin()+20,invalid.begin()+40); reject(invalid,"IMU sampling gap");
    invalid=data; invalid[3].measurement.gyroscopes.x()=std::numeric_limits<double>::quiet_NaN(); reject(invalid,"non-finite gyroscope");
    auto turning=data; for(auto& m:turning) m.measurement.gyroscopes.z()=0.3;
    reject(turning,"constant rotation must fail even with zero covariance");
    auto vibrating=data;
    for(size_t i=0;i<vibrating.size();++i) vibrating[i].measurement.accelerometers.x()=i%2?2:-2;
    reject(vibrating,"accelerometer covariance rejects vibration");
    vibrating=data;
    for(size_t i=0;i<vibrating.size();++i) vibrating[i].measurement.gyroscopes.x()=i%2?0.1:-0.1;
    reject(vibrating,"gyroscope covariance rejects angular oscillation");
    invalid=data; for(auto& m:invalid) m.measurement.accelerometers.setZero(); reject(invalid,"free fall is not stationary gravity");
    invalid=data; for(auto& m:invalid) m.measurement.accelerometers.z()=12; reject(invalid,"constant force with zero covariance must fail gravity gate");
    auto pose=original;
    require(cgmli::ceres::ImuError::initPose(data,pose),"legacy two-argument entry enforces valid stationary startup");
    auto noisy=data; Eigen::Vector3d mean=Eigen::Vector3d::Zero();
    for(size_t i=0;i<noisy.size();++i) {
        noisy[i].measurement.accelerometers.x()=i%2?0.02:-0.02;
        noisy[i].measurement.accelerometers.y()=i%3?0.01:-0.01;
        mean+=noisy[i].measurement.accelerometers;
    }
    mean/=double(noisy.size()); Eigen::Matrix3d reference=Eigen::Matrix3d::Zero();
    for(const auto& m:noisy) { const Eigen::Vector3d d=m.measurement.accelerometers-mean; reference+=d*d.transpose(); }
    reference/=double(noisy.size()-1);
    require(cgmli::ceres::ImuError::initPose(noisy,pose,options,&statistics),"low stationary sensor noise accepted");
    debug::recordError("startup_sample_covariance_error",(statistics.accelCovariance-reference).norm(),1e-12);
    auto csv=debug::artifactFile("startup_stationarity_covariance.csv");
    if(csv) {
        csv<<"row,column,welford,independent_two_pass\n";
        for(int r=0;r<3;++r) for(int c=0;c<3;++c) csv<<r<<','<<c<<','<<statistics.accelCovariance(r,c)<<','<<reference(r,c)<<'\n';
    }
    auto inverted=data; for(auto& m:inverted) m.measurement.accelerometers.z()=-9.81;
    require(cgmli::ceres::ImuError::initPose(inverted,pose),"antiparallel stationary gravity remains supported");
    debug::recordError("antiparallel_gravity_error",(pose.C()*(-Eigen::Vector3d::UnitZ())-Eigen::Vector3d::UnitZ()).norm(),1e-12);
    auto calibrated=data;
    options.gravity=9.80665; options.accelBias=Eigen::Vector3d(0.2,-0.1,0.3); options.gyroBiasPrior=Eigen::Vector3d(0.1,0,0);
    for(auto& m:calibrated) {
        m.measurement.accelerometers=options.accelBias+options.gravity*Eigen::Vector3d::UnitZ();
        m.measurement.gyroscopes=options.gyroBiasPrior;
    }
    require(cgmli::ceres::ImuError::initPose(calibrated,pose,options,&statistics),"known sensor biases and local gravity are respected");
    debug::recordError("calibrated_stationary_gravity_error",(pose.C()-Eigen::Matrix3d::Identity()).norm(),1e-12);
}
void stationaryInitializationAndAtomicFailure() {
    const auto p=parameters(); const auto data=samples(1.0);
    kinematics::Transformation pose; SpeedAndBias sb=SpeedAndBias::Zero(); std::string error;
    require(factor::initializeImu(data,p.imu,timeAt(0.5),pose,sb,error),"stationary initialization");
    require((pose.C()-Eigen::Matrix3d::Identity()).norm()<1e-12,"parallel gravity must remain finite and aligned");
    debug::recordError("stationary_gravity_rotation_error",(pose.C()-Eigen::Matrix3d::Identity()).norm(),1e-12);
    auto init=debug::artifactFile("initial_state.csv");
    if(init) {
        init<<"variable,index,value\n";
        for(int i=0;i<3;++i) init<<"position,"<<i<<','<<pose.r()[i]<<'\n';
        for(int i=0;i<9;++i) init<<"speed_bias,"<<i<<','<<sb[i]<<'\n';
    }
    const auto originalPose=pose; const auto originalSb=sb;
    require(!factor::propagateImu({},p.imu,pose,sb,timeAt(0.5),timeAt(0.6)),"missing IMU rejected");
    require((pose.T()-originalPose.T()).norm()==0 && (sb-originalSb).norm()==0,"propagation failure is atomic");
    auto moving=data; for(size_t i=0;i<moving.size();++i) moving[i].measurement.gyroscopes.x()=i%2?1.0:-1.0;
    require(!factor::initializeImu(moving,p.imu,timeAt(0.5),pose,sb,error),"moving startup cannot claim zero velocity");
    require((pose.T()-originalPose.T()).norm()==0 && (sb-originalSb).norm()==0,"moving initialization preserves both pose and speed/bias");
}
void nativeSecondNewestImuReplay() {
    const auto p=parameters(); const auto data=samples(1.2);
    backend::BackendOptions options; options.windowSize=2;
    backend::SlidingWindowManager window(options); std::string error;
    const auto state=[](uint64_t id) {
        backend::StateEstimate s; s.id=StateId(id); s.timestamp=timeAt(0.2*double(id)); return s;
    };
    for(uint64_t id=1;id<=2;++id) require(window.append(state(id),error),"IMU replay state");
    require(window.addFactor(factor::makeBodyImuRecord(data,p.imu,state(1),state(2))).accepted(),"initial replayable IMU");
    auto trace=debug::artifactFile("second_newest_imu_replay.csv");
    if(trace) trace<<"frame,raw_samples,active_imu,prior_reused,residual_error,jacobian_error\n";
    for(uint64_t id=3;id<=4;++id) {
        require(window.append(state(id),error),"replay append");
        require(window.addFactor(factor::makeBodyImuRecord(data,p.imu,state(id-1),state(id))).accepted(),"next replayable IMU");
        const auto report=window.marginalize(backend::MarginalizationPolicy::MARGIN_SECOND_NEW);
        require(report.success && (report.imuMode=="incremental" || report.imuMode=="reintegrated") && report.replacedImuFactors==2
            && report.activeImuFactors==1 && report.priorReused && !window.graph().prior,
            "replace two edges with one nonlinear interval, without an IMU prior duplicate");
        require(window.graph().factors.size()==1,"replay leaves exactly one edge");
        const auto& merged=window.graph().factors.begin()->second;
        const auto direct=factor::makeBodyImuRecord(data,p.imu,state(1),state(id));
        require(merged.keys==direct.keys && merged.imuReplay && merged.imuReplay->samples.size()==direct.imuReplay->samples.size(),
            "concatenated support equals direct integration support");
        for(size_t i=1;i<merged.imuReplay->samples.size();++i)
            require(merged.imuReplay->samples[i-1].timeStamp<merged.imuReplay->samples[i].timeStamp,"shared brackets are deduplicated");
        std::vector<Eigen::VectorXd> storage;
        for(auto key:merged.keys) storage.push_back(window.graph().blocks.at(key).value);
        storage[2][0]+=0.01; storage[3][1]-=0.02; storage[1][3]+=1e-4;
        std::vector<const double*> x; std::vector<Eigen::VectorXd> ja,jb; std::vector<double*> pa,pb;
        for(const auto& value:storage) { x.push_back(value.data()); ja.emplace_back(15*value.size()); jb.emplace_back(15*value.size()); }
        for(size_t i=0;i<ja.size();++i) { pa.push_back(ja[i].data()); pb.push_back(jb[i].data()); }
        Eigen::Matrix<double,15,1> ra,rb;
        require(merged.cost->Evaluate(x.data(),ra.data(),pa.data()) && direct.cost->Evaluate(x.data(),rb.data(),pb.data()),"merged/direct factor evaluation");
        debug::recordError("merged_imu_residual_"+std::to_string(id),(ra-rb).norm(),1e-9);
        double maximumError=0;
        for(size_t i=0;i<ja.size();++i) {
            const double e=(ja[i]-jb[i]).norm(); maximumError=std::max(maximumError,e);
            debug::recordDerivative("merged_imu_J_"+std::to_string(id)+"_"+std::to_string(i),e,jb[i].norm(),1e-8,1e-8);
        }
        require(window.valid(error),"merged frame payload and graph remain consistent");
        if(trace) trace<<id<<','<<merged.imuReplay->samples.size()<<','<<report.activeImuFactors<<','<<report.priorReused<<','<<(ra-rb).norm()<<','<<maximumError<<'\n';
    }
    require(window.append(state(5),error),"merge rollback append");
    auto conflicting=factor::makeBodyImuRecord(data,p.imu,state(4),state(5));
    auto payload=std::make_shared<backend::ImuFactorReplay>(*conflicting.imuReplay);
    payload->samples.front().measurement.accelerometers.x()+=1;
    conflicting.imuReplay=payload;
    require(window.addFactor(std::move(conflicting)).accepted(),"stage conflicting replay material");
    const auto before=window.graph(); const auto historySize=window.history().size();
    const auto failed=window.marginalize(backend::MarginalizationPolicy::MARGIN_SECOND_NEW);
    require(!failed.success && window.size()==3 && window.frames().find(StateId(4))
        && window.history().size()==historySize && window.graph().factors.size()==before.factors.size(),
        "failed merge must not evict a state, consume evidence or partially insert a replacement");
    for(const auto& item:before.factors) require(window.graph().factors.at(item.first).cost==item.second.cost
        && window.graph().factors.at(item.first).imuReplay==item.second.imuReplay,"failed merge keeps original immutable factors");
}
void scopedPoseVelocityKeepsWindowAndCalibration() {
    using namespace backend;
    const auto p=parameters();
    const auto data=samples(0.8);
    BackendOptions options; options.windowSize=2;
    std::vector<cv::KeyPoint> points;
    for(int i=0;i<8;++i) points.emplace_back(36.f+22.f*(i%4),36.f+24.f*(i/4),31.f);
    const auto populate=[&](FixedLagBackend& target,uint64_t gyroBiasStepFrame) {
        std::string error;
        require(target.initializeCalibration(makeRigCalibration(p.nCameraSystem,kinematics::Transformation()).parameterBlocks(),error),
            "scoped calibration setup");
        StateEstimate previous;
        for(uint64_t i=1;i<=3;++i) {
            StateEstimate state; state.id=StateId(i); state.timestamp=timeAt(0.2*double(i));
            state.visual=SensorHealth::Valid; state.keyframe=i==1;
            if(i>=2) {
                const double direction=i==2?-1.0:1.0;
                state.T_WB.set(Eigen::Vector3d(direction*0.001,0,0),Eigen::Quaterniond::Identity());
                state.speedBias[0]=direction*0.005;
            }
            // A bias step leaves a large residual on just one edge because the local solve holds biases fixed.
            if(gyroBiasStepFrame && i>=gyroBiasStepFrame) state.speedBias[3]=0.002;
            MultiFrame frame(p.nCameraSystem,state.timestamp,i);
            frame.resetKeypoints(0,points);
            require(target.append(state,frame,data,error),"append scoped fixture");
            if(i>1) require(target.addFactor(factor::makeBodyImuRecord(data,p.imu,previous,state)).accepted(),"scoped IMU edge");
            previous=state;
        }
        for(size_t i=0;i<points.size();++i) {
            const LandmarkId id(i+1);
            require(target.addLandmark(id,{1,0,i},0.25,error),"scoped anchor");
            require(target.addObservation(id,{2,0,i},error) && target.addObservation(id,{3,0,i},error),"scoped live observations");
        }
        require(target.window().canOptimizeLatestTwo(),"eligible two-frame fixture");
    };
    FixedLagBackend backend(options);
    populate(backend,0);
    const auto before=backend.window().graph();
    const auto pose=BlockKey::pose(StateId(3)),speed=BlockKey::speedBias(StateId(3));
    SolveBudget budget; budget.latestTwoPoseVelocityOnly=true;
    budget.minIterations=1;
    const auto rejected=backend.step({}, {}, [](const BackendGraph&,std::string& reason) {
        reason="forced scoped rejection"; return false;
    }, {}, {}, budget);
    require(!rejected.accepted && !backend.snapshot() && backend.window().size()==3,
        "failed scoped solve leaves the same epoch ready for a full retry");
    for(const auto& item:before.blocks)
        require((backend.window().graph().blocks.at(item.first).value-item.second.value).norm()==0,"scoped rollback restores all values");
    for(const auto& item:before.factors)
        require(backend.window().graph().factors.at(item.first).lifecycle==item.second.lifecycle,"scoped rollback restores factor lifecycle");

    int calibrationRequests=0;
    const auto accepted=backend.step({},[&](BlockKey,const BackendGraph&) {++calibrationRequests; return true;}, {}, {}, {}, budget);
    require(accepted.accepted && accepted.solve.effectiveParameters==18 && accepted.solve.linearSolverUsed=="DENSE_QR"
        && accepted.solve.residualBlocks==2*points.size()+2 && accepted.solve.activeLandmarks==0
        && accepted.solve.postSolveResiduals[1].factors==2,
        "scoped problem includes both frames' measurements and 18 free coordinates");
    require(calibrationRequests==0 && accepted.marginalization &&
        accepted.marginalization->removed==StateId(2) && accepted.marginalization->replacedImuFactors==2,
        "scoped solve freezes calibration and still merges IMU on normal eviction");
    const auto& after=backend.window().graph();
    for(const auto& item:after.blocks) {
        const auto& old=before.blocks.at(item.first);
        if(item.first==pose) continue;
        const Eigen::VectorXd delta=item.second.value-old.value;
        require((item.first==speed?delta.tail<6>().norm():delta.norm())==0,"scoped solve changes only pose and velocity");
    }
    require((after.blocks.at(pose).value.head<3>()).norm()<0.001 &&
        std::abs(after.blocks.at(speed).value[0])<0.005,"scoped solve corrects latest pose and velocity");
    require(accepted.retiredState && accepted.retiredState->id==StateId(2)
        && accepted.retiredState->T_WB.r().norm()<0.001 && std::abs(accepted.retiredState->speedBias[0])<0.005
        && (accepted.retiredState->speedBias.tail<6>()-before.blocks.at(BlockKey::speedBias(StateId(2))).value.tail<6>()).norm()==0,
        "scoped solve corrects penultimate pose and velocity with fixed biases before eviction");

    budget.skipPreNis=true;
    auto gatedBudget=budget;
    gatedBudget.gateLightweightNewImuNis=true;
    FixedLagBackend oldConflict(options);
    populate(oldConflict,2);
    const auto oldOnly=oldConflict.step({}, {}, {}, {}, {}, gatedBudget);
    require(oldOnly.accepted && oldOnly.solve.lightweightOldImu.after>60 && oldOnly.solve.lightweightNewImu.after<=60,
        "large old-edge NIS is diagnostic when the new edge passes");
    require(oldOnly.solve.lightweightOldImu.factor!=oldOnly.solve.lightweightNewImu.factor
        && std::isfinite(oldOnly.solve.lightweightOldImu.before) && std::isfinite(oldOnly.solve.lightweightNewImu.before),
        "both directed edges report pre-solve NIS even when the general pre-gate is skipped");
    FixedLagBackend newConflict(options);
    populate(newConflict,3);
    const auto gated=newConflict.step({}, {}, {}, {}, {}, gatedBudget);
    require(!gated.accepted && !newConflict.snapshot() && newConflict.window().size()==3
        && gated.solve.lightweightNewImu.after>60 && gated.solve.lightweightOldImu.after<=60,
        "the enabled newest-edge gate rejects the same staged epoch before eviction or publication");
    const auto newOnly=newConflict.step({}, {}, {}, {}, {}, budget);
    require(newOnly.accepted && newConflict.snapshot() && newConflict.window().size()==2
        && newOnly.solve.lightweightNewImu.after>60 && newOnly.solve.lightweightOldImu.after<=60,
        "large finite new-edge NIS is diagnostic and the scoped solve still commits");
    require(newOnly.marginalization && newOnly.marginalization->removed==StateId(2)
        && newOnly.marginalization->replacedImuFactors==2,
        "scoped solve with large new-edge NIS still evicts and merges the IMU bridge");
}
void longImuIntervalsAreStorageOnly() {
    const auto p=parameters(); const auto data=samples(25.0); backend::BackendOptions options;
    options.windowSize=2; options.maxImuSamplesPerEpoch=1600;
    backend::SlidingWindowManager window(options); std::string error;
    const auto state=[](uint64_t id) {
        backend::StateEstimate s; s.id=StateId(id); s.timestamp=timeAt(id==1?0.1:6.0*double(id-1)); return s;
    };
    for(uint64_t id=1;id<=3;++id) {
        require(window.append(state(id),error),"long interval append");
        if(id>1) require(window.addFactor(factor::makeBodyImuRecord(data,p.imu,state(id-1),state(id))).accepted(),"long interval input edge");
    }
    const auto first=window.marginalize(backend::MarginalizationPolicy::MARGIN_SECOND_NEW);
    require(first.success && first.activeImuFactors==0 && first.suppressedImuFactors==1
        && first.consumedFactors==0 && !window.graph().prior,"overlong replacement cannot enter a new prior");
    const auto& stored=window.graph().factors.begin()->second;
    require(!stored.solveEnabled && !stored.imuReplay->completeSamples && stored.imuReplay->samples.size()<=options.maxImuSamplesPerEpoch,
        "suppressed raw storage remains bounded with explicit incomplete flag");
    require(window.append(state(4),error) && window.addFactor(factor::makeBodyImuRecord(data,p.imu,state(3),state(4))).accepted(),
        "new short IMU edge remains eligible beside stored long interval");
    auto graph=window.graph(); backend::FactorRecord pose=backend::makeParameterPrior(graph.blocks.at(backend::BlockKey::pose(StateId(3))),
        Eigen::Matrix<double,6,6>::Identity()); pose.id=1000; pose.lifecycle=backend::FactorLifecycle::Active; graph.factors.emplace(pose.id,pose);
    auto sb=backend::makeParameterPrior(graph.blocks.at(backend::BlockKey::speedBias(StateId(3))),Eigen::Matrix<double,9,9>::Identity());
    sb.id=1001; sb.lifecycle=backend::FactorLifecycle::Active; graph.factors.emplace(sb.id,sb);
    const auto solved=backend::BackendOptimizer(options).optimize(graph);
    require(solved.accepted && solved.residualBlocks==3,"Ceres uses the short edge and priors, never the deliberately unevaluable long cost");
    auto referenceGraph=window.graph();
    const auto prior=backend::Marginalizer(options).compute(referenceGraph,{backend::BlockKey::pose(StateId(1)),backend::BlockKey::speedBias(StateId(1))});
    require(prior.success && prior.consumed.empty() && !prior.prior,"long edge excluded from OLD/general Schur selection too");
    const auto second=window.marginalize(backend::MarginalizationPolicy::MARGIN_SECOND_NEW);
    require(second.success && second.suppressedImuFactors==1 && !window.graph().prior && window.valid(error),
        "suppressed interval continues concatenating without a prior-bridge fallback");
}
void fractionalIncrementalImuMatchesFullReplay() {
    auto p=parameters(); ImuMeasurementDeque data;
    for(int i=0;i<=200;++i) {
        const double t=i/200.0;
        data.emplace_back(timeAt(t),ImuSensorReadings(Eigen::Vector3d(0.15+0.1*t,-0.07,0.08*t),
            Eigen::Vector3d(0.2*t,0.1*std::sin(t),p.imu.g+0.03)));
    }
    backend::StateEstimate a,b,c; a.id=StateId(1); b.id=StateId(2); c.id=StateId(3);
    a.timestamp=timeAt(0.1013); b.timestamp=timeAt(0.4077); c.timestamp=timeAt(0.7352);
    a.speedBias.segment<3>(3)=Eigen::Vector3d(0.002,-0.001,0.003); a.speedBias.tail<3>()=Eigen::Vector3d(0.01,-0.02,0.03);
    b.speedBias=a.speedBias; c.speedBias=a.speedBias;
    const auto left=factor::makeBodyImuRecord(data,p.imu,a,b),right=factor::makeBodyImuRecord(data,p.imu,b,c);
    backend::FactorRecord merged; std::string error;
    require(factor::mergeBodyImuRecords(left,right,a,c,1000,merged,error)
        && merged.imuReplay->integrationMode=="incremental","fractional image boundaries use complete-sample prefix cache");
    const auto direct=factor::makeBodyImuRecord(data,p.imu,a,c);
    Eigen::Matrix<double,7,1> poses[2];
    poses[0]<<a.T_WB.r(),a.T_WB.q().coeffs();
    poses[1]<<c.T_WB.r(),c.T_WB.q().coeffs();
    const double* x[]{poses[0].data(),a.speedBias.data(),poses[1].data(),c.speedBias.data()};
    factor::BodyImuDiagnostics m,d;
    require(factor::diagnoseBodyImu(*merged.cost,x,m) && factor::diagnoseBodyImu(*direct.cost,x,d),"fractional replay diagnostics");
    debug::recordDerivative("fractional_incremental_covariance",(m.covariance-d.covariance).norm(),d.covariance.norm(),1e-10,1e-18);
    debug::recordDerivative("fractional_incremental_physical_residual",(m.physical-d.physical).norm(),d.physical.norm(),1e-9,1e-10);
    Eigen::Matrix<double,15,1> rm,rd; std::vector<Eigen::VectorXd> jm,jd; std::vector<double*> pm,pd;
    for(int n:{7,9,7,9}) { jm.emplace_back(15*n); jd.emplace_back(15*n); }
    for(size_t i=0;i<jm.size();++i) { pm.push_back(jm[i].data()); pd.push_back(jd[i].data()); }
    require(merged.cost->Evaluate(x,rm.data(),pm.data()) && direct.cost->Evaluate(x,rd.data(),pd.data()),"fractional cached/full derivatives");
    for(size_t i=0;i<jm.size();++i) debug::recordDerivative("fractional_incremental_J_"+std::to_string(i),
        (jm[i]-jd[i]).norm(),jd[i].norm(),1e-8,1e-8);
    a.speedBias[3]+=0.01;
    require(factor::mergeBodyImuRecords(left,right,a,c,1000,merged,error)
        && merged.imuReplay->integrationMode=="full","changed reference bias triggers full reintegration");
}
void bodyImuJacobianAndFrozenCost() {
    auto p=parameters(); p.imu.T_BS=kinematics::Transformation(Eigen::Vector3d(0.12,-0.04,0.03),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.2,Eigen::Vector3d::UnitY())));
    State sensor{}; sensor.id=StateId(1); sensor.timestamp=timeAt(0.5); sensor.v_W.setZero(); sensor.b_g.setZero(); sensor.b_a.setZero(); sensor.omega_S.setZero();
    auto a=pipeline::bodyPrediction(sensor,p.imu.T_BS); sensor.id=StateId(2); sensor.timestamp=timeAt(0.55);
    auto b=pipeline::bodyPrediction(sensor,p.imu.T_BS); const auto cost=factor::makeBodyImuFactor(samples(1),p.imu,a,b);
    Eigen::Matrix<double,7,1> poses[2]; poses[0]<<a.T_WB.r(),a.T_WB.q().coeffs(); poses[1]<<b.T_WB.r(),b.T_WB.q().coeffs();
    SpeedAndBias biases[2]{a.speedBias,b.speedBias};
    const double* x[]{poses[0].data(),biases[0].data(),poses[1].data(),biases[1].data()};
    Eigen::Matrix<double,15,7,Eigen::RowMajor> jp[2]; Eigen::Matrix<double,15,9,Eigen::RowMajor> jb[2];
    double* j[]{jp[0].data(),jb[0].data(),jp[1].data(),jb[1].data()}; Eigen::Matrix<double,15,1> base;
    require(cost->Evaluate(x,base.data(),j) && base.norm()<1e-5,"stationary body/sensor residual");
    const backend::LeftSE3Manifold manifold;
    for(int block=0;block<4;++block) {
        const int tangent=block%2?9:6; Eigen::Matrix<double,15,Eigen::Dynamic> numeric(15,tangent),analytic(15,tangent);
        if(block%2) analytic=jb[block/2];
        else { Eigen::Matrix<double,7,6,Eigen::RowMajor> lift; manifold.PlusJacobian(x[block],lift.data()); analytic=jp[block/2]*lift; }
        for(int c=0;c<tangent;++c) {
            const double h=1e-6; Eigen::VectorXd plus(block%2?9:7),minus(block%2?9:7);
            if(block%2) { plus=biases[block/2]; minus=plus; plus[c]+=h; minus[c]-=h; }
            else { Eigen::Matrix<double,6,1> delta=Eigen::Matrix<double,6,1>::Zero(); delta[c]=h; manifold.Plus(x[block],delta.data(),plus.data()); delta[c]=-h; manifold.Plus(x[block],delta.data(),minus.data()); }
            const double* input[]{x[0],x[1],x[2],x[3]}; Eigen::Matrix<double,15,1> rp,rm;
            input[block]=plus.data(); require(cost->Evaluate(input,rp.data(),nullptr),"positive Jacobian sample");
            input[block]=minus.data(); require(cost->Evaluate(input,rm.data(),nullptr),"negative Jacobian sample"); numeric.col(c)=(rp-rm)/(2*h);
        }
        debug::recordDerivative("imu_block_"+std::to_string(block),(numeric-analytic).norm(),numeric.norm());
        auto csv=debug::artifactFile("imu_block_"+std::to_string(block)+".csv");
        if(csv) {
            csv<<"row,column,analytic,numeric,absolute_error\n";
            for(int row=0;row<15;++row) for(int column=0;column<tangent;++column)
                csv<<row<<','<<column<<','<<analytic(row,column)<<','<<numeric(row,column)<<','<<std::abs(analytic(row,column)-numeric(row,column))<<'\n';
        }
    }
    biases[0][3]=0.01; Eigen::Matrix<double,15,1> other,repeated;
    require(cost->Evaluate(x,other.data(),nullptr),"biased sample"); biases[0][3]=0;
    require(cost->Evaluate(x,repeated.data(),nullptr) && (repeated-base).norm()<1e-10,"factor cache cannot alter a rollback reference");
    require(!factor::bodyImuNeedsRefresh(*cost,a.speedBias,0.0003,0.02),"fresh cost retains its reference");
    auto shifted=a; shifted.speedBias[3]+=0.001; shifted.speedBias[6]+=0.03;
    require(factor::bodyImuNeedsRefresh(*cost,shifted.speedBias,0.0003,0.02),"bias drift requests immutable refresh");
    const auto refreshed=factor::makeBodyImuFactor(samples(1),p.imu,shifted,b);
    require(!factor::bodyImuNeedsRefresh(*refreshed,shifted.speedBias,0.0003,0.02),"replacement uses the new bias");
    require(cost->Evaluate(x,repeated.data(),nullptr) && (repeated-base).norm()<1e-10,"replacement preserves old checkpoint cost");
    poses[1][0]+=0.01; biases[1][0]+=0.02; biases[1][3]+=0.001; biases[1][6]+=0.03;
    factor::BodyImuDiagnostics d;
    require(factor::diagnoseBodyImu(*cost,x,d) && d.physical.segment<3>(0).norm()>0
        && d.physical.segment<3>(6).norm()>0 && d.physical.segment<3>(9).norm()>0
        && d.physical.segment<3>(12).norm()>0,"physical position/velocity/bias diagnostics");
    require(cost->Evaluate(x,repeated.data(),nullptr),"diagnostic NIS reference");
    debug::recordError("imu_diagnostic_nis_error",std::abs(d.nis-repeated.squaredNorm()),1e-9);
}
void dynamicBodyImuJacobiansAndIntervals() {
    auto p=parameters();
    p.imu.T_BS=kinematics::Transformation(Eigen::Vector3d(0.25,-0.12,0.07),
        backend::expSO3(Eigen::Vector3d(0.2,-0.15,0.1)));
    const Eigen::Quaterniond R0=backend::expSO3(Eigen::Vector3d(0.3,-0.2,0.4));
    const Eigen::Vector3d acceleration(0.2,-0.1,0.05),velocity(1.1,-0.4,0.2),origin(3,-2,1);
    const Eigen::Vector3d bg(0.01,-0.02,0.005),ba(0.03,-0.015,0.02);
    const double angularRate=0.7;
    const Eigen::Vector3d omega=R0.conjugate()*Eigen::Vector3d(0,0,angularRate);
    const auto sensor=[&](double t,uint64_t id) {
        State s{}; s.id=StateId(id); s.timestamp=timeAt(t);
        s.T_WS=kinematics::Transformation(Eigen::Vector3d(origin+velocity*t+0.5*acceleration*t*t),
            backend::expSO3(Eigen::Vector3d(0,0,angularRate*t))*R0);
        s.v_W=velocity+acceleration*t; s.b_g=bg; s.b_a=ba; s.omega_S=omega; return s;
    };
    ImuMeasurementDeque data;
    for(int i=0;i<=160;++i) {
        const double t=i/200.0; const auto s=sensor(t,1);
        data.emplace_back(timeAt(t),ImuSensorReadings(Eigen::Vector3d(omega+bg),
            Eigen::Vector3d(s.T_WS.C().transpose()*(acceleration+Eigen::Vector3d(0,0,p.imu.g))+ba)));
    }
    const auto s0=sensor(0.1,1),s1=sensor(0.7,3);
    const auto a=pipeline::bodyPrediction(s0,p.imu.T_BS),b=pipeline::bodyPrediction(s1,p.imu.T_BS);
    const auto mid=pipeline::bodyPrediction(sensor(0.4,2),p.imu.T_BS);
    const auto direct=factor::makeBodyImuRecord(data,p.imu,a,b);
    auto propagated=s0.T_WS; SpeedAndBias sb; sb<<s0.v_W,bg,ba;
    require(factor::propagateImu(data,p.imu,propagated,sb,s0.timestamp,s1.timestamp),"dynamic sensor propagation");
    debug::recordError("dynamic_propagation_position",(propagated.r()-s1.T_WS.r()).norm(),1e-5);
    debug::recordError("dynamic_propagation_velocity",(sb.head<3>()-s1.v_W).norm(),2e-5);
    debug::recordError("dynamic_propagation_rotation",backend::logSO3(propagated.q()*s1.T_WS.q().conjugate()).norm(),1e-6);
    Eigen::Matrix<double,7,1> pose[2]; pose[0]<<a.T_WB.r(),a.T_WB.q().coeffs(); pose[1]<<b.T_WB.r(),b.T_WB.q().coeffs();
    SpeedAndBias biases[2]{a.speedBias,b.speedBias};
    const double* x[]{pose[0].data(),biases[0].data(),pose[1].data(),biases[1].data()};
    factor::BodyImuDiagnostics nominal;
    require(factor::diagnoseBodyImu(*direct.cost,x,nominal),"dynamic physical diagnostics");
    debug::recordError("dynamic_nominal_position",nominal.physical.head<3>().norm(),1e-5);
    debug::recordError("dynamic_nominal_rotation",nominal.physical.segment<3>(3).norm(),1e-6);
    debug::recordError("dynamic_nominal_velocity",nominal.physical.segment<3>(6).norm(),2e-5);
    // Deliberately evaluate away from the frozen bias reference and away from
    // zero residual. A stationary zero-bias case cannot expose Exp(phi)'s Jr.
    biases[0].segment<3>(3)+=Eigen::Vector3d(0.012,-0.009,0.007);
    biases[0].tail<3>()+=Eigen::Vector3d(0.03,0.01,-0.02);
    biases[1].head<3>()+=Eigen::Vector3d(0.02,-0.01,0.015); pose[1][0]+=0.01;
    Eigen::Matrix<double,15,7,Eigen::RowMajor> jp[2];
    Eigen::Matrix<double,15,9,Eigen::RowMajor> jb[2];
    double* J[]{jp[0].data(),jb[0].data(),jp[1].data(),jb[1].data()};
    Eigen::Matrix<double,15,1> r;
    require(direct.cost->Evaluate(x,r.data(),J),"dynamic nonzero residual Evaluate");
    factor::BodyImuJacobianCheck runtimeAudit;
    require(factor::checkBodyImuJacobians(*direct.cost,x,runtimeAudit) && runtimeAudit.columns==30,
        "runtime derivative audit covers both pose and SB tangents");
    debug::recordError("dynamic_runtime_audit_scaled_error",runtimeAudit.maxScaledError,1.0);
    const backend::LeftSE3Manifold manifold;
    auto csv=debug::artifactFile("dynamic_imu_jacobians.csv");
    if(csv) csv<<"block,row,column,analytic,numeric\n";
    for(int block=0;block<4;++block) {
        const int tangent=block%2?9:6; Eigen::MatrixXd analytic(15,tangent),numeric(15,tangent);
        if(block%2) analytic=jb[block/2];
        else {
            Eigen::Matrix<double,7,6,Eigen::RowMajor> lift;
            require(manifold.PlusJacobian(x[block],lift.data()),"dynamic left-SE3 lift"); analytic=jp[block/2]*lift;
        }
        for(int c=0;c<tangent;++c) {
            Eigen::VectorXd plus=Eigen::Map<const Eigen::VectorXd>(x[block],block%2?9:7),minus=plus;
            constexpr double h=1e-6;
            if(block%2) { plus[c]+=h; minus[c]-=h; }
            else {
                Eigen::Matrix<double,6,1> delta=Eigen::Matrix<double,6,1>::Zero(); delta[c]=h;
                require(manifold.Plus(x[block],delta.data(),plus.data()),"dynamic plus");
                delta[c]=-h; require(manifold.Plus(x[block],delta.data(),minus.data()),"dynamic minus");
            }
            const double* input[]{x[0],x[1],x[2],x[3]}; Eigen::Matrix<double,15,1> rp,rm;
            input[block]=plus.data(); require(direct.cost->Evaluate(input,rp.data(),nullptr),"dynamic FD plus");
            input[block]=minus.data(); require(direct.cost->Evaluate(input,rm.data(),nullptr),"dynamic FD minus");
            numeric.col(c)=(rp-rm)/(2*h);
        }
        debug::recordDerivative("dynamic_imu_block_"+std::to_string(block),(analytic-numeric).norm(),numeric.norm(),1e-6,1e-6);
        for(int c=0;c<tangent;++c) debug::recordDerivative("dynamic_imu_column_"+std::to_string(block)+"_"+std::to_string(c),
            (analytic.col(c)-numeric.col(c)).norm(),numeric.col(c).norm(),1e-6,1e-6);
        if(csv) for(int row=0;row<15;++row) for(int c=0;c<tangent;++c)
            csv<<block<<','<<row<<','<<c<<','<<analytic(row,c)<<','<<numeric(row,c)<<'\n';
    }
    backend::FactorRecord merged; std::string error;
    require(factor::mergeBodyImuRecords(factor::makeBodyImuRecord(data,p.imu,a,mid),
        factor::makeBodyImuRecord(data,p.imu,mid,b),a,b,10000,merged,error),"dynamic adjacent interval merge");
    Eigen::Matrix<double,15,1> mergedResidual;
    require(merged.cost->Evaluate(x,mergedResidual.data(),nullptr),"dynamic merged Evaluate");
    debug::recordDerivative("dynamic_old_interval_merge",(mergedResidual-r).norm(),r.norm(),1e-10,1e-9);
    const auto saved=r; const auto savedCovariance=nominal.covariance;
    for(int i=0;i<3;++i) require(direct.cost->Evaluate(x,r.data(),nullptr),"old interval repeated evaluation");
    factor::BodyImuDiagnostics repeated;
    require(factor::diagnoseBodyImu(*direct.cost,x,repeated),"old interval covariance diagnostic");
    debug::recordError("old_interval_residual_immutability",(r-saved).norm(),1e-12);
    debug::recordError("old_interval_covariance_immutability",(repeated.covariance-savedCovariance).norm(),1e-18);
}
void imuNoiseDensityUnits() {
    auto p=parameters(); backend::StateEstimate a,b; a.id=StateId(1); b.id=StateId(2);
    a.timestamp=timeAt(0.1); b.timestamp=timeAt(0.5);
    const auto measure=[&](const ImuParameters& imu,int rate) {
        ImuMeasurementDeque data;
        for(int i=0;i<=int(0.6*rate);++i)
            data.emplace_back(timeAt(double(i)/rate),ImuSensorReadings(Eigen::Vector3d::Zero(),Eigen::Vector3d(0,0,imu.g)));
        const auto cost=factor::makeBodyImuFactor(data,imu,a,b);
        Eigen::Matrix<double,7,1> pose; pose<<0,0,0,0,0,0,1;
        const double* x[]{pose.data(),a.speedBias.data(),pose.data(),b.speedBias.data()};
        factor::BodyImuDiagnostics d; require(factor::diagnoseBodyImu(*cost,x,d),"noise covariance diagnostic"); return d.covariance;
    };
    const auto base=measure(p.imu,200); auto doubled=p.imu;
    doubled.sigma_g_c*=2; doubled.sigma_a_c*=2; doubled.sigma_gw_c*=2; doubled.sigma_aw_c*=2;
    const auto scaled=measure(doubled,200);
    debug::recordDerivative("noise_density_squared_scaling",(scaled-4*base).norm(),(4*base).norm(),1e-10,1e-16);
    const double dt=0.4;
    debug::recordError("gyro_bias_diffusion_units",std::abs(base(9,9)-p.imu.sigma_gw_c*p.imu.sigma_gw_c*dt),1e-16);
    debug::recordError("accel_bias_diffusion_units",std::abs(base(12,12)-p.imu.sigma_aw_c*p.imu.sigma_aw_c*dt),1e-16);
    auto isolated=p.imu; isolated.sigma_g_c=0; isolated.sigma_gw_c=0; isolated.sigma_aw_c=0;
    const auto at200=measure(isolated,200),at400=measure(isolated,400);
    const double expected=isolated.sigma_a_c*isolated.sigma_a_c*dt;
    debug::recordError("continuous_accel_density_200Hz",std::abs(at200(8,8)-expected),1e-14);
    debug::recordError("continuous_accel_density_400Hz",std::abs(at400(8,8)-expected),1e-14);
    auto csv=debug::artifactFile("imu_noise_units.csv");
    if(csv) csv<<"quantity,expected,actual\nvelocity_variance_200,"<<expected<<','<<at200(8,8)
        <<"\nvelocity_variance_400,"<<expected<<','<<at400(8,8)<<'\n';
}
void nativePipelineImuWindowAndHistory() {
    // Cover both explicit per-frame draining and continuous submission. In the
    // latter schedule the next image is available while the preceding backend
    // worker is still joinable; every input must survive window eviction/EOF.
    for (const bool drainEachFrame : {false, true}) {
        auto p=parameters(); std::atomic<int> count{0}; std::atomic<uint64_t> latest{0};
        auto trace=debug::artifactFile(drainEachFrame?"stationary_tracking_drained.csv":"stationary_tracking_async.csv");
        if(trace) trace<<"frame,timestamp_ns,x,y,z,speed_mps,quality\n";
        std::mutex traceMutex; double maximumSpeed=0;
        ThreadedSlam slam(p,""); slam.setBlocking(true);
        slam.setOptimisedGraphCallback([&](const State& state,const TrackingState& tracking,std::shared_ptr<const AlignedMap<StateId,State>> updates,std::shared_ptr<const MapPointVector>) {
            require(state.T_WS.T().allFinite() && state.v_W.norm()<0.05,"stationary IMU-only state remains finite");
            {
                std::lock_guard<std::mutex> lock(traceMutex); maximumSpeed=std::max(maximumSpeed,state.v_W.norm());
                if(trace) trace<<state.id.value()<<','<<state.timestamp.toNSec()<<','<<state.T_WS.r().x()<<','<<state.T_WS.r().y()<<','<<state.T_WS.r().z()
                    <<','<<state.v_W.norm()<<','<<int(tracking.trackingQuality)<<'\n';
            }
            require(updates->count(state.id),"state and revision agree"); latest=state.id.value(); ++count;
        });
        for(const auto& m:samples(1.4)) require(slam.addImuMeasurement(m.timeStamp,m.measurement.accelerometers,m.measurement.gyroscopes),"enqueue IMU");
        cv::Mat blank(160,160,CV_8UC1,cv::Scalar(128));
        for(int i=0;i<10;++i) {
            require(slam.addImages(timeAt(0.5+i*0.05),{{0,blank},{1,blank}}),"enqueue image");
            require(slam.processFrame(),"stage native state");
            require(!slam.hasPendingImages(),"consumed input does not wait for backend completion");
            if (drainEachFrame) slam.processFrame();
        }
        slam.stopThreading(); require(count.load()==10 && latest.load()==10,"window eviction does not suppress trajectory output");
        debug::recordError(drainEachFrame?"stationary_speed_drained":"stationary_speed_async",maximumSpeed,0.05);
        require(!slam.hasPendingWork(),"successful stop drains worker and prepared frame");
    }
}
void historicalSphericalImuBundleAdjustment() {
    auto p=parameters(); p.estimator.do_final_ba=true; backend::GlobalSlamGraph graph(p);
    // A translated, yawed stationary rig exposes accidental direction-only
    // transforms in historical point storage and the global bearing residual.
    const kinematics::Transformation T_WB(Eigen::Vector3d(1.0,-0.5,0.2),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.25,Eigen::Vector3d::UnitZ())));
    const auto imu=samples(1.0); std::vector<kinematics::Transformation> extrinsics;
    for(size_t c=0;c<2;++c) extrinsics.push_back(*p.nCameraSystem.T_SC(c));
    std::vector<backend::LandmarkEstimate> landmarks(20);
    for(int frameId=1;frameId<=3;++frameId) {
        auto frame=std::make_shared<MultiFrame>(p.nCameraSystem,timeAt(0.45+frameId*0.05),frameId);
        for(size_t c=0;c<2;++c) {
            std::vector<cv::KeyPoint> keypoints;
            for(int i=0;i<20;++i) {
                const Eigen::Vector3d local((i%5-2)*0.12,(i/5-2)*0.12,3.0+0.02*i);
                const Eigen::Vector3d world=T_WB.C()*local+T_WB.r();
                const auto T_WC=T_WB*extrinsics[c];
                const Eigen::Vector3d cameraPoint=T_WC.C().transpose()*(world-T_WC.r());
                Eigen::Vector2d pixel; p.nCameraSystem.cameraGeometry(c)->project(cameraPoint,&pixel);
                keypoints.emplace_back(float(pixel.x()),float(pixel.y()),31.0f);
                auto& point=landmarks[i]; point.metadata.id=LandmarkId(i+1); point.metadata.anchor={1,0,size_t(i)}; point.pointWorld=world;
                backend::WindowObservation o; o.key={uint64_t(frameId),c,size_t(i)}; o.landmark=point.metadata.id;
                o.bearing=cameraPoint.normalized(); o.sqrtInformation=Eigen::Matrix2d::Identity()*100; point.observations.push_back(o);
            }
            frame->resetKeypoints(c,keypoints); frame->resetDescriptors(c,cv::Mat(20,32,CV_8UC1,cv::Scalar(1))); frame->setDescriptorProfile(c,"regression-orb");
            for(int i=0;i<20;++i) frame->setLandmarkId(c,i,i+1);
        }
        backend::StateEstimate state; state.id=StateId(frameId); state.timestamp=frame->timestamp(); state.keyframe=true;
        state.T_WB=T_WB;
        auto snapshot=std::make_shared<backend::BackendSnapshot>(); snapshot->states.push_back(state);
        snapshot->calibration[{backend::BlockKind::CameraExtrinsic,0}]=backend::ParameterBlock::pose({backend::BlockKind::CameraExtrinsic,0},extrinsics[0]).value;
        backend::StepReport step; step.accepted=true; step.snapshot=snapshot; step.optimizedVisualHistory=landmarks;
        std::string error; require(graph.ingest(step,frame,imu,extrinsics,true,error),"archive raw measurements");
    }
    std::string error; require(graph.bundleAdjust(10,1,0,0,error),"historical BA uses real spherical and IMU costs");
    require(graph.estimates().size()==3 && graph.mapPoints().size()==20,"historical BA retains states and metric geometry");
    for(const auto& point:graph.mapPoints()) {
        const int i=int(point.id)-1;
        const Eigen::Vector3d local((i%5-2)*0.12,(i/5-2)*0.12,3.0+0.02*i);
        const Eigen::Vector3d expected=T_WB.C()*local+T_WB.r();
        require((point.point.head<3>()-expected).norm()<1e-5,"historical BA preserves translated metric point coordinates");
    }
}
void historicalLoopClosesMetricDrift() {
    auto p=parameters(); p.imu.use=false; p.estimator.do_loop_closures=true;
    backend::GlobalSlamGraph graph(p); std::vector<kinematics::Transformation> extrinsics;
    for(size_t c=0;c<2;++c) extrinsics.push_back(*p.nCameraSystem.T_SC(c));
    cv::Mat descriptors[2]; cv::RNG random(991);
    for(auto& d:descriptors) { d.create(40,32,CV_8UC1); random.fill(d,cv::RNG::UNIFORM,0,256); }
    bool closed=false;
    for(int frameId=1;frameId<=22;++frameId) {
        auto frame=std::make_shared<MultiFrame>(p.nCameraSystem,timeAt(0.5+frameId*0.5),frameId);
        std::vector<backend::LandmarkEstimate> landmarks(40);
        for(size_t c=0;c<2;++c) {
            std::vector<cv::KeyPoint> keypoints;
            for(int i=0;i<40;++i) {
                const Eigen::Vector3d world((i%8-3.5)*0.4,(i/8-2)*0.45,3.0+0.15*(i%3));
                const Eigen::Vector3d cameraPoint=extrinsics[c].C().transpose()*(world-extrinsics[c].r());
                Eigen::Vector2d pixel; p.nCameraSystem.cameraGeometry(c)->project(cameraPoint,&pixel);
                keypoints.emplace_back(float(pixel.x()),float(pixel.y()),31.0f);
                auto& point=landmarks[i]; point.metadata.id=LandmarkId(i+1); point.metadata.anchor={1,0,size_t(i)}; point.pointWorld=world;
                backend::WindowObservation o; o.key={uint64_t(frameId),c,size_t(i)}; o.landmark=point.metadata.id;
                o.bearing=cameraPoint.normalized(); o.sqrtInformation=Eigen::Matrix2d::Identity()*100; point.observations.push_back(o);
            }
            if(frameId>20) random.fill(descriptors[c],cv::RNG::UNIFORM,0,256);
            frame->resetKeypoints(c,keypoints); frame->resetDescriptors(c,descriptors[c]); frame->setDescriptorProfile(c,"regression-orb");
            for(int i=0;i<40;++i) frame->setLandmarkId(c,i,i+1);
        }
        backend::StateEstimate state; state.id=StateId(frameId); state.timestamp=frame->timestamp(); state.keyframe=true;
        state.T_WB=kinematics::Transformation(Eigen::Vector3d(0.02*(frameId-1),0,0),Eigen::Quaterniond::Identity());
        auto snapshot=std::make_shared<backend::BackendSnapshot>(); snapshot->states.push_back(state);
        snapshot->calibration[{backend::BlockKind::CameraExtrinsic,0}]=backend::ParameterBlock::pose({backend::BlockKind::CameraExtrinsic,0},extrinsics[0]).value;
        backend::StepReport step; step.accepted=true; step.snapshot=snapshot; step.optimizedVisualHistory=std::move(landmarks);
        std::string error; require(graph.ingest(step,frame,{},extrinsics,true,error),"archive and verify historical loops");
        closed=closed || graph.lastLoopClosed();
        if(frameId>20) require(!graph.lastLoopClosed(),"unrelated descriptors must not create a geometric loop");
    }
    require(closed,"historical ORB/GP3P candidate must pass temporal consistency and close a loop");
    require(graph.estimates().back().T_WB.r().norm()<0.1,"independent pose graph corrects the injected metric drift");
}
}
#ifdef CGMLI_EMBED_MODULE_TESTS
void cgmli::debug::registerPipelineCases(std::vector<cgmli::debug::ModuleCase>& cases) {
    cases.push_back({"initialization","averagedImageInputPreservesTimes",[]() { averagedImageInputPreservesTimes(); }});
    cases.push_back({"backend","calibrationRetryFreezesOnlyRejectedBlocks",[]() { calibrationRetryFreezesOnlyRejectedBlocks(); }});
    cases.push_back({"backend","motionCalibrationRespectsTrackSupport",[]() { motionCalibrationRespectsTrackSupport(); }});
    cases.push_back({"backend","scopedPoseVelocityKeepsWindowAndCalibration",[]() { scopedPoseVelocityKeepsWindowAndCalibration(); }});
    cases.push_back({"initialization","modelSpecificCameraCalibration",[]() { modelSpecificCameraCalibration(); }});
    cases.push_back({"initialization","stationaryInitializationAndAtomicFailure",[]() { stationaryInitializationAndAtomicFailure(); }});
    cases.push_back({"initialization","stationaryPoseEntryRejectsMotion",[]() { stationaryPoseEntryRejectsMotion(); }});
    cases.push_back({"jacobian","eucmBackProjectionJacobian",[]() { eucmBackProjectionJacobian(); }});
    cases.push_back({"jacobian","bodyImuJacobianAndFrozenCost",[]() { bodyImuJacobianAndFrozenCost(); }});
    cases.push_back({"jacobian","dynamicBodyImuJacobiansAndIntervals",[]() { dynamicBodyImuJacobiansAndIntervals(); }});
    cases.push_back({"backend","imuNoiseDensityUnits",[]() { imuNoiseDensityUnits(); }});
    cases.push_back({"marginalization","nativeSecondNewestImuReplay",[]() { nativeSecondNewestImuReplay(); }});
    cases.push_back({"marginalization","longImuIntervalsAreStorageOnly",[]() { longImuIntervalsAreStorageOnly(); }});
    cases.push_back({"jacobian","fractionalIncrementalImuMatchesFullReplay",[]() { fractionalIncrementalImuMatchesFullReplay(); }});
    cases.push_back({"tracking","featureDebugSnapshotsAndRendering",[]() { featureDebugSnapshotsAndRendering(); }});
    cases.push_back({"tracking","imuGapAndAtomicPropagation",[]() { imuGapAndAtomicPropagation(); }});
    cases.push_back({"tracking","inertialPoseAcceptanceBounds",[]() { inertialPoseAcceptanceBounds(); }});
    cases.push_back({"tracking","failedSolveRollbackPreservesNextFrame",[]() { failedSolveRollbackPreservesNextFrame(); }});
    cases.push_back({"tracking","imuGapStopsCurrentSegment",[]() { imuGapStopsCurrentSegment(); }});
    cases.push_back({"tracking","nativePipelineImuWindowAndHistory",[]() { nativePipelineImuWindowAndHistory(); }});
}
#else
int main() {
    try { featureDebugSnapshotsAndRendering(); averagedImageInputPreservesTimes(); modelSpecificCameraCalibration(); imuGapAndAtomicPropagation(); inertialPoseAcceptanceBounds();
        calibrationRetryFreezesOnlyRejectedBlocks();
        motionCalibrationRespectsTrackSupport();
        scopedPoseVelocityKeepsWindowAndCalibration();
        failedSolveRollbackPreservesNextFrame(); imuGapStopsCurrentSegment();
        nativeSecondNewestImuReplay();
        longImuIntervalsAreStorageOnly(); fractionalIncrementalImuMatchesFullReplay();
        dynamicBodyImuJacobiansAndIntervals(); imuNoiseDensityUnits();
        eucmBackProjectionJacobian(); stationaryInitializationAndAtomicFailure(); stationaryPoseEntryRejectsMotion(); bodyImuJacobianAndFrozenCost(); nativePipelineImuWindowAndHistory(); historicalSphericalImuBundleAdjustment(); historicalLoopClosesMetricDrift();
        std::cout<<"native SLAM regressions passed\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

#endif // CGMLI_EMBED_MODULE_TESTS
