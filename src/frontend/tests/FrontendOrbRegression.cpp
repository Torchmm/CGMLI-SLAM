// Standalone source regression; intentionally not built or executed in this task.
// Link OrbFrontendFeatures, the ORB feature sources and project camera/time sources.
#include <debug/ModuleDebug.hpp>
#include <debug/ModuleDebugArtifacts.hpp>
#include <frontend/OrbFrontendFeatures.hpp>
#include <visual/MultiFrame.hpp>
#include <visual/features/OrbMatcher.hpp>
#include <cameras/PinholeCamera.hpp>
#include <cameras/EucmCamera.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <cstring>
#include <opencv2/imgproc.hpp>

using namespace cgmli;
using namespace cgmli::features;
namespace {
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
template<class F> void throws(F operation,const char* message) {
    bool failed=false; try { operation(); } catch(const std::exception&) { failed=true; } require(failed,message);
}
void sparseGuidedMatchingEquivalence() {
    auto camera=std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(
        128,128,70,70,64,64,cameras::NoDistortion());
    cv::Mat queryStorage(24,35,CV_8UC1),trainStorage(64,37,CV_8UC1);
    cv::Mat query=queryStorage(cv::Rect(1,0,32,24)),train=trainStorage(cv::Rect(2,0,32,64));
    cv::RNG rng(1743); rng.fill(query,cv::RNG::UNIFORM,0,256); rng.fill(train,cv::RNG::UNIFORM,0,256);
    query.copyTo(train.rowRange(0,24)); query.row(0).copyTo(train.row(1));
    std::vector<cv::KeyPoint> q,t;
    for(int i=0;i<24;++i) q.emplace_back(float(10+(i%6)*20),float(10+(i/6)*20),31);
    for(int j=0;j<64;++j) t.push_back(j<24?q[j]:cv::KeyPoint(float((j*19)%127),float((j*31)%127),31));
    OrbMatchGuidance guidance; guidance.queryCamera=camera; guidance.trainCamera=camera;
    guidance.useSphericalGeometry=false; guidance.searchRadiusPixels=12;
    guidance.queryEnabled.resize(q.size()); guidance.trainEnabled.resize(t.size());
    for(size_t i=0;i<q.size();++i) { guidance.queryEnabled[i]=i%11!=0; guidance.predictedTrainPixels.push_back(q[i].pt); }
    for(size_t j=0;j<t.size();++j) guidance.trainEnabled[j]=j%13!=0;
    guidance.allowedPairs=cv::Mat(24,64,CV_8UC1,cv::Scalar(255));
    const auto equal=[](const auto& a,const auto& b) {
        require(a.size()==b.size(),"sparse/dense reference match count differs");
        for(size_t i=0;i<a.size();++i) require(a[i].queryIdx==b[i].queryIdx && a[i].trainIdx==b[i].trainIdx
            && a[i].distance==b[i].distance,"sparse/dense reference match identity differs");
    };
    auto combinations=debug::artifactFile("matching_policy_combinations.csv");
    if(combinations) combinations<<"mutual,reverse_ratio,allow_single,reference_matches,guided_matches,candidate_visits,hamming_tests\n";
    for(bool mutual:{false,true}) for(bool reverseRatio:{false,true}) for(bool single:{false,true}) {
        OrbMatcherOptions options; options.maxOctaveDifference=-1; options.mutualCheck=mutual;
        options.reverseRatioTest=reverseRatio; options.allowSingleCandidate=single;
        OrbMatcher matcher(options); cv::Mat permitted=cv::Mat::zeros(24,64,CV_8UC1);
        for(int i=0;i<24;++i) for(int j=0;j<64;++j) {
            if((i+j)%7==0) guidance.allowedPairs.at<uchar>(i,j)=0;
            const auto d=t[j].pt-guidance.predictedTrainPixels[i];
            if(guidance.queryEnabled[i] && guidance.trainEnabled[j] && guidance.allowedPairs.at<uchar>(i,j)
                && d.dot(d)<=144) permitted.at<uchar>(i,j)=255;
        }
        std::vector<cv::DMatch> expected,actual; OrbMatchStatistics statistics;
        matcher.matchFiltered(query,train,expected,permitted);
        matcher.matchGuided(query,train,q,t,guidance,actual,&statistics);
        equal(expected,actual);
        if(combinations) combinations<<mutual<<','<<reverseRatio<<','<<single<<','<<expected.size()<<','<<actual.size()
            <<','<<statistics.candidatePairs<<','<<statistics.descriptorComparisons<<'\n';
        if(mutual && reverseRatio) {
            const std::string prefix=single?"single_candidate_matches":"guided_matches";
            debug::recordError(prefix+"_dense_count_error",std::abs(double(expected.size())-double(actual.size())),0);
            auto csv=debug::artifactFile(prefix+".csv");
            if(csv) {
                csv<<"query,train,hamming\n";
                for(const auto& match:actual) csv<<match.queryIdx<<','<<match.trainIdx<<','<<match.distance<<'\n';
            }
            if(!debug::caseDirectory.empty()) {
                cv::Mat left(128,128,CV_8UC1,cv::Scalar(220)),right=left.clone(),view;
                cv::drawMatches(left,q,right,t,actual,view,cv::Scalar(0,180,0),cv::Scalar(100,100,100));
                cv::resize(view,view,cv::Size(),3,3,cv::INTER_NEAREST);
                cv::putText(view,"SYNTHETIC descriptor/geometry check",{8,20},cv::FONT_HERSHEY_SIMPLEX,0.5,{0,0,200},1);
                debug::saveImage(prefix+".png",view);
            }
        }
        require(statistics.candidatePairs>0 && statistics.candidatePairs<24*64,"pixel grid must reduce real candidate visits");
        // A reverse ratio needs two admissible QUERY neighbors. This fixture has
        // isolated predictions, so a correctly empty result is legal in strict mode.
        // Single-candidate mode is a separate positive control against two empty paths.
        if(single) require(!actual.empty(),"single-candidate positive control must recover copied descriptors");
        if(mutual && reverseRatio && !single)
            require(actual.empty(),"isolated reverse neighbors must fail the strict ratio test");
    }
    // Exact octave partitions must preserve forward/reverse second neighbors,
    // including ties, ratio rejection and pre-existing feature/pair gates.
    guidance.predictedTrainPixels.clear();
    for(size_t i=0;i<q.size();++i) q[i].octave=int(i%4);
    for(size_t j=0;j<t.size();++j) t[j].octave=int(j%4);
    for(int difference:{0,1,3}) for(bool mutual:{false,true}) {
        OrbMatcherOptions options; options.maxOctaveDifference=difference; options.mutualCheck=mutual;
        options.allowSingleCandidate=true; OrbMatcher matcher(options);
        cv::Mat permitted=cv::Mat::zeros(24,64,CV_8UC1);
        for(int i=0;i<24;++i) for(int j=0;j<64;++j)
            if(guidance.queryEnabled[i] && guidance.trainEnabled[j] && guidance.allowedPairs.at<uchar>(i,j)
                && std::abs(q[i].octave-t[j].octave)<=difference) permitted.at<uchar>(i,j)=255;
        std::vector<cv::DMatch> expected,actual; OrbMatchStatistics statistics;
        matcher.matchFiltered(query,train,expected,permitted);
        matcher.matchGuided(query,train,q,t,guidance,actual,&statistics); equal(expected,actual);
        require(statistics.setupMs>=0 && statistics.traversalMs>=0 && statistics.selectionMs>=0,"matching phase timing must be nonnegative");
    }
    // Compare the cached dot-product gate to the original symmetric asin gate.
    guidance.predictedTrainPixels.clear(); guidance.allowedPairs.release(); guidance.useSphericalGeometry=true;
    guidance.T_train_query=kinematics::Transformation(Eigen::Vector3d(0.2,0.03,0.01),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.02,Eigen::Vector3d::UnitY())));
    OrbBearings qRays(q.size()),tRays(t.size());
    for(size_t i=0;i<q.size();++i) { camera->backProject(Eigen::Vector2d(q[i].pt.x,q[i].pt.y),&qRays[i]); qRays[i].normalize(); }
    for(size_t j=0;j<t.size();++j) { camera->backProject(Eigen::Vector2d(t[j].pt.x,t[j].pt.y),&tRays[j]); tRays[j].normalize(); }
    guidance.queryBearings=&qRays; guidance.trainBearings=&tRays;
    OrbMatcherOptions options; options.maxOctaveDifference=-1; options.allowSingleCandidate=true;
    OrbMatcher matcher(options); cv::Mat permitted=cv::Mat::zeros(24,64,CV_8UC1);
    const Eigen::Vector3d translation=guidance.T_train_query.r().normalized();
    for(int i=0;i<24;++i) for(int j=0;j<64;++j) {
        if(!guidance.queryEnabled[i] || !guidance.trainEnabled[j]) continue;
        const Eigen::Vector3d rotated=guidance.T_train_query.C()*qRays[i];
        const Eigen::Vector3d nt=translation.cross(rotated),nq=guidance.T_train_query.C().transpose()*tRays[j].cross(translation);
        if(nt.norm()<1e-12 || nq.norm()<1e-12) continue;
        const double a=std::asin(std::clamp(std::abs(tRays[j].dot(nt))/nt.norm(),0.0,1.0));
        const double b=std::asin(std::clamp(std::abs(qRays[i].dot(nq))/nq.norm(),0.0,1.0));
        if(std::max(a,b)<=options.epipolarThresholdRadians) permitted.at<uchar>(i,j)=255;
    }
    std::vector<cv::DMatch> expected,actual;
    matcher.matchFiltered(query,train,expected,permitted); matcher.matchGuided(query,train,q,t,guidance,actual); equal(expected,actual);
    OrbMatchGuidance duplicates; duplicates.useSphericalGeometry=false;
    cv::Mat zeroQuery(1,32,CV_8UC1,cv::Scalar(0)),zeroTrain(2,32,CV_8UC1,cv::Scalar(0));
    matcher.matchGuided(zeroQuery,zeroTrain,{cv::KeyPoint(10,10,31)},
        {cv::KeyPoint(10,10,31),cv::KeyPoint(11,10,31)},duplicates,actual);
    require(actual.empty(),"two equal zero-distance neighbors must fail strict ratio");
}
struct OverlapRig : cameras::NCameraSystem {
    bool diagonalMasksEmpty() const {
        for(size_t i=0;i<overlapMats_.size();++i) if(!overlapMats_[i][i].empty()) return false;
        return true;
    }
    size_t selfMaskCount() const {
        std::lock_guard<std::mutex> lock(selfOverlapCache_->mutex);
        return selfOverlapCache_->masks.size();
    }
};
void deferredRigOverlaps() {
    OverlapRig rig;
    const auto camera=[](int width=24) {
        return std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(
            width,18,20,20,width/2.0,9,cameras::NoDistortion());
    };
    const kinematics::Transformation opposite(Eigen::Vector3d::Zero(),
        Eigen::Quaterniond(Eigen::AngleAxisd(std::acos(-1.0),Eigen::Vector3d::UnitY())));
    for(size_t i=0;i<4;++i) rig.addCamera(std::make_shared<kinematics::Transformation>(
        i==2?opposite:kinematics::Transformation()),camera(),cameras::NCameraSystem::NoDistortion);
    require(!rig.overlapComputationValid(),"camera insertion defers cross-camera computation");
    require(rig.hasOverlap(0,0) && rig.selfMaskCount()==0,"self visibility requires no masks");
    throws([&]{rig.overlap(0,1);},"cross-camera access requires explicit finalization");
    rig.computeOverlaps();
    require(rig.overlapComputationValid() && rig.diagonalMasksEmpty() && rig.selfMaskCount()==0,
        "finalization computes only cross-camera image masks");
    require(rig.hasOverlap(0,1) && !rig.hasOverlap(0,2),"forward and opposite camera visibility");
    const int initialSupport=cv::countNonZero(rig.overlap(0,1));
    // Floating-point round trips can move border pixels just outside the image.
    require(initialSupport>=22*16 && cv::countNonZero(rig.overlap(0,2))==0,
        "directed overlap masks distinguish coincident and opposite views");

    // A common body/sensor calibration update changes no camera-to-camera ray
    // geometry. Preserve the existing tables and leave the original rig alone.
    auto rebased=rig;
    const kinematics::Transformation common(Eigen::Vector3d(0.3,-0.1,0.2),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.4,Eigen::Vector3d::UnitZ())));
    rebased.transformReferenceFrame(common);
    require(rebased.overlapComputationValid() && !rebased.hasOverlap(0,2)
        && rebased.overlap(0,1).data==rig.overlap(0,1).data,
        "common rig rebase preserves overlap validity and cached image storage");
    for(size_t i=0;i<rig.numCameras();++i) {
        require((rebased.T_SC(i)->T()-(common*(*rig.T_SC(i))).T()).norm()<1e-12,
            "common transform updates every camera without writing into the source");
        const auto relative=rebased.T_SC(0)->inverse()*(*rebased.T_SC(i));
        const auto original=rig.T_SC(0)->inverse()*(*rig.T_SC(i));
        require((relative.T()-original.T()).norm()<1e-12,"relative rig baselines remain unchanged");
    }
    auto incomplete=rig; incomplete.setExtrinsics(1,opposite);
    incomplete.transformReferenceFrame(common);
    require(!incomplete.overlapComputationValid(),"rebase cannot make stale overlap data valid");

    // Concurrent first access, including a copied const rig, must allocate one
    // shared self mask without changing the cross-camera table's diagonal entries.
    const OverlapRig copy=rig;
    std::array<cv::Mat,4> masks;
    std::array<std::exception_ptr,4> errors{};
    std::array<std::thread,4> workers;
    for(size_t i=0;i<workers.size();++i) workers[i]=std::thread([&,i]{
        try { masks[i]=(i%2?copy:rig).overlap(0,0); } catch(...) { errors[i]=std::current_exception(); }
    });
    for(auto& worker:workers) worker.join();
    for(const auto& error:errors) if(error) std::rethrow_exception(error);
    for(const auto& mask:masks) require(mask.data==masks[0].data && cv::countNonZero(mask)==24*18,
        "concurrent self-mask requests reuse one all-visible image");
    require(rig.selfMaskCount()==1 && rig.diagonalMasksEmpty(),"self allocation is lazy and separate");

    // Recomputing a modified copy must neither retain an old true flag nor write
    // through OpenCV's shared image storage into the original rig's published mask.
    auto changed=rig;
    changed.setExtrinsics(1,opposite);
    require(!changed.overlapComputationValid(),"extrinsic update invalidates cross-camera masks");
    changed.computeOverlaps();
    require(!changed.hasOverlap(0,1) && rig.hasOverlap(0,1),"recomputation clears stale visibility flags");
    require(cv::countNonZero(rig.overlap(0,1))==initialSupport && cv::countNonZero(changed.overlap(0,1))==0,
        "recomputation preserves another rig's image masks");
    auto narrow=rig,wide=rig;
    narrow.addCamera(std::make_shared<kinematics::Transformation>(),camera(20),cameras::NCameraSystem::NoDistortion);
    wide.addCamera(std::make_shared<kinematics::Transformation>(),camera(30),cameras::NCameraSystem::NoDistortion);
    require(!narrow.overlapComputationValid() && !wide.overlapComputationValid(),"extension invalidates old tables");
    require(narrow.overlap(4,4).cols==20 && wide.overlap(4,4).cols==30,
        "independently extended copies do not share an incompatible self-mask cache");
    throws([&]{rig.hasOverlap(4,4);},"invalid self camera index is rejected");
}
cameras::NCameraSystem mixedRig() {
    cameras::NCameraSystem rig;
    rig.addCamera(std::make_shared<kinematics::Transformation>(),
        std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(160,160,120,120,80,80,cameras::NoDistortion()),
        cameras::NCameraSystem::NoDistortion,false);
    rig.addCamera(std::make_shared<kinematics::Transformation>(),
        std::make_shared<cameras::EucmCamera>(160,160,80,90,80,80,0.6,1.0),
        cameras::NCameraSystem::NoDistortion,false);
    return rig;
}
cv::Mat texture() {
    cv::Mat image(160,160,CV_8UC1); cv::RNG random(81); random.fill(image,cv::RNG::UNIFORM,0,256); return image;
}
void hammingAndRowLayout() {
    std::array<uchar,33> a{},b{}; b.fill(255);
    require(OrbFrontendFeatures::hammingDistance(a.data()+1,b.data()+1)==256,"unaligned full 256-bit distance");
    require(OrbFrontendFeatures::hammingDistance(a.data()+1,a.data()+1)==0,"identical descriptors");
    b.fill(0); b[1]=8;
    require(OrbFrontendFeatures::hammingDistance(a.data()+1,b.data()+1)==1,"one-bit distance");
    auto rig=mixedRig(); MultiFrame frame(rig,Time(1,0),1);
    frame.resetKeypoints(0,{cv::KeyPoint(80,80,31),cv::KeyPoint(90,80,31)});
    cv::Mat backing(2,64,CV_8UC1,cv::Scalar(0));
    frame.resetDescriptors(0,backing.colRange(0,32)); // Valid rows with a 64-byte stride.
    OrbFrontendFeatures::validateDescriptors(frame,0);
    frame.resetDescriptors(0,cv::Mat::zeros(2,48,CV_8UC1));
    throws([&]{OrbFrontendFeatures::validateDescriptors(frame,0);},"48-byte descriptors rejected before matching");
    frame.resetDescriptors(0,cv::Mat(2,0,CV_8UC1));
    throws([&]{OrbFrontendFeatures::validateDescriptors(frame,0);},"zero-width rows cannot satisfy a nonempty keypoint set");
}
void mixedCamerasReuseAndProfiles() {
    auto rig=mixedRig(); OrbExtractorOptions options; options.pyramidLevels=1;
    OrbFrontendFeatures frontend(2,options);
    MultiFrame frame(rig,Time(1,0),1);
    for(size_t c=0;c<2;++c) frame.setImage(c,texture());
    const std::vector<cv::KeyPoint> input{cv::KeyPoint(80,80,31),cv::KeyPoint(0,0,31)};
    // Camera 1 initializes first; common focal selection must still come from C0.
    require(frontend.detectAndDescribe(1,frame,kinematics::Transformation(),&input),"EUCM extraction");
    require(frontend.detectAndDescribe(0,frame,kinematics::Transformation(),&input),"pinhole extraction");
    require(frame.numKeypoints(0)==1 && frame.numKeypoints(1)==1,"unsupported supplied keypoints removed with rows");
    frontend.validateFrame(frame);
    require(frame.descriptorProfile(0)==frame.descriptorProfile(1),"one common angular/profile convention across cameras");
    require(frame.descriptorProfile(0).find("f=120")!=std::string::npos,"reference independent of initialization order");
    const cv::Mat first=frame.descriptors(0).clone();
    MultiFrame next(rig,Time(2,0),2); for(size_t c=0;c<2;++c) next.setImage(c,texture());
    for(size_t c=0;c<2;++c) frontend.detectAndDescribe(c,next,kinematics::Transformation(),&input);
    require(cv::norm(first,next.descriptors(0),cv::NORM_HAMMING)==0,"cache reuse preserves output for identical inputs");
    throws([&]{frontend.setOptions(options);},"live descriptor profile cannot be reconfigured");
    frame.setDescriptorProfile(0,"ordinary-opencv-orb");
    throws([&]{frontend.validateFrame(frame);},"same-length incompatible profile rejected");
    frontend.clear(); options.virtualFocalLength=100; frontend.setOptions(options);
    for(size_t c=0;c<2;++c) frontend.detectAndDescribe(c,next,kinematics::Transformation(),&input);
    frontend.validateFrame(next);
    require(next.descriptorProfile(0).find("f=100")!=std::string::npos,"clear allows explicit reference reconfiguration");
}
void startupCameraPreparation() {
    const auto rig=mixedRig(); OrbExtractorOptions options; options.pyramidLevels=3;
    OrbFrontendFeatures prepared(2,options),lazy(2,options);
    const MultiFrame calibration(rig,Time(0,0));
    prepared.prepare(calibration,true); prepared.prepare(calibration,false);
    require(calibration.numKeypoints()==0 && calibration.image(0).empty() && calibration.image(1).empty(),
        "Startup preparation must not consume/fabricate image measurements");
    MultiFrame ready(rig,Time(1,0),1),reference(rig,Time(1,0),1);
    for(size_t c=0;c<2;++c) {
        ready.setImage(c,texture()); reference.setImage(c,texture());
        prepared.detectAndDescribe(c,ready,kinematics::Transformation());
        lazy.detectAndDescribe(c,reference,kinematics::Transformation());
        require(ready.numKeypoints(c)==reference.numKeypoints(c)
            && ready.descriptorProfile(c)==reference.descriptorProfile(c),"Prepared/lazy feature layout differs");
        for(size_t k=0;k<ready.numKeypoints(c);++k)
            require(ready.keypoints(c)[k].pt==reference.keypoints(c)[k].pt
                && ready.keypoints(c)[k].octave==reference.keypoints(c)[k].octave,"Startup preparation changed feature selection");
        require(ready.descriptors(c).empty()==reference.descriptors(c).empty()
            && (ready.descriptors(c).empty() || cv::norm(ready.descriptors(c),reference.descriptors(c),cv::NORM_HAMMING)==0),
            "Startup preparation changed descriptor bits");
    }
    prepared.validateFrame(ready);
    auto differentRig=mixedRig(); const MultiFrame changed(differentRig,Time(0,0));
    throws([&]{prepared.prepare(changed,true);},"Prepared calibration identity cannot silently change");
    throws([&]{prepared.setOptions(options);},"Prepared descriptor configuration is frozen");
    prepared.clear(); options.virtualFocalLength=100; prepared.setOptions(options);
    prepared.prepare(calibration,false);
}
void groupedMapNeighborsAndPreparedLayouts() {
    auto camera=std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(
        128,128,70,70,64,64,cameras::NoDistortion());
    cv::Mat query(4,32,CV_8UC1,cv::Scalar(0)),train(2,32,CV_8UC1,cv::Scalar(0));
    query.row(2).colRange(0,8).setTo(255); query.row(3).setTo(255); train.row(1).setTo(255);
    std::vector<cv::KeyPoint> q,t;
    for(int i=0;i<4;++i) q.emplace_back(float(50+i),50,31);
    t.emplace_back(55,55,31); t.emplace_back(60,60,31);
    OrbMatcherOptions options; options.maxOctaveDifference=-1; OrbMatcher matcher(options);
    OrbMatchGuidance guidance; guidance.useSphericalGeometry=false;
    guidance.queryCamera=camera; guidance.trainCamera=camera;
    std::vector<cv::DMatch> ordinary,grouped,cached;
    matcher.matchGuided(query,train,q,t,guidance,ordinary);
    const std::vector<uint64_t> identities{1,1,2,3}; guidance.queryIdentities=&identities;
    matcher.matchGuided(query,train,q,t,guidance,grouped);
    require(ordinary.size()==1 && grouped.size()==2,"Two descriptors of one landmark must not corrupt reverse ratio");
    require(identities[size_t(grouped[0].queryIdx)]==1 && identities[size_t(grouped[1].queryIdx)]==3,
        "Grouped map matching must retain distinct identity and best descriptor owner");
    OrbWordRows qw(4),tw(2);
    for(int i=0;i<4;++i) std::memcpy(qw[i].data(),query.ptr(i),32);
    for(int i=0;i<2;++i) std::memcpy(tw[i].data(),train.ptr(i),32);
    auto index=OrbMatcher::prepareTrainIndex(t,camera.get(),guidance.searchRadiusPixels);
    guidance.queryWords=&qw; guidance.trainWords=&tw; guidance.trainIndex=&index;
    matcher.matchGuided(query,train,q,t,guidance,cached);
    require(cached.size()==grouped.size(),"Prepared index changed map matches");
    for(size_t i=0;i<cached.size();++i) require(cached[i].queryIdx==grouped[i].queryIdx
        && cached[i].trainIdx==grouped[i].trainIdx && cached[i].distance==grouped[i].distance,"Prepared words/index changed exact result");
    index.cellSize+=1;
    throws([&]{matcher.matchGuided(query,train,q,t,guidance,cached);},"Stale search-radius index must be rejected");
}
void filteredTrainBucketsPreserveNeighbors() {
    auto camera=std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(
        128,128,70,70,64,64,cameras::NoDistortion());
    cv::Mat query(12,32,CV_8UC1),train(48,32,CV_8UC1); cv::RNG rng(9317);
    rng.fill(query,cv::RNG::UNIFORM,0,256); rng.fill(train,cv::RNG::UNIFORM,0,256);
    query.copyTo(train.rowRange(0,12));
    std::vector<cv::KeyPoint> q,t;
    for(int j=0;j<48;++j) t.emplace_back(float(8+(j%8)*15),float(8+(j/8)*18),31,-1,1,j%3);
    q.assign(t.begin(),t.begin()+12);
    OrbMatchGuidance g; g.useSphericalGeometry=false; g.queryCamera=camera; g.trainCamera=camera;
    g.trainEnabled.assign(48,1); for(int j=1;j<48;j+=3) g.trainEnabled[j]=0;
    g.trainRegionMask=cv::Mat::zeros(128,128,CV_8UC1); g.trainRegionMask.colRange(0,80).setTo(255);
    auto index=OrbMatcher::prepareTrainIndex(t,camera.get(),g.searchRadiusPixels); g.trainIndex=&index;
    const auto originalGrid=index.grid; const auto originalOctaves=index.octaves;
    for(bool predicted:{false,true}) for(int octaveGate:{-1,0,1}) for(bool mutual:{false,true}) {
        g.predictedTrainPixels.clear();
        if(predicted) for(const auto& key:q) g.predictedTrainPixels.push_back(key.pt);
        OrbMatcherOptions options; options.maxOctaveDifference=octaveGate; options.mutualCheck=mutual;
        options.allowSingleCandidate=true; OrbMatcher matcher(options);
        cv::Mat permitted=cv::Mat::zeros(12,48,CV_8UC1);
        size_t visits=0;
        for(int i=0;i<12;++i) for(int j=0;j<48;++j) {
            if(!g.trainEnabled[j] || t[j].pt.x>=80) continue;
            if(octaveGate>=0 && std::abs(q[i].octave-t[j].octave)>octaveGate) continue;
            const auto d=t[j].pt-q[i].pt;
            if(!predicted || d.dot(d)<=g.searchRadiusPixels*g.searchRadiusPixels) permitted.at<uchar>(i,j)=255;
            ++visits;
        }
        std::vector<cv::DMatch> expected,actual; OrbMatchStatistics stats;
        matcher.matchFiltered(query,train,expected,permitted);
        matcher.matchGuided(query,train,q,t,g,actual,&stats);
        require(expected.size()==actual.size(),"Filtered train buckets changed ratio/mutual matches");
        for(size_t k=0;k<actual.size();++k) require(actual[k].queryIdx==expected[k].queryIdx && actual[k].trainIdx==expected[k].trainIdx
            && actual[k].distance==expected[k].distance,"Filtered buckets changed exact neighbor identity");
        if(!predicted) require(stats.candidatePairs==visits,"Disabled/overlap-excluded rows must not enter candidate traversal");
        require(index.grid==originalGrid && index.octaves==originalOctaves,"A job must not mutate the shared training index");
    }
}
void existingIdentityWinsBeforeNewTexture() {
    cv::Mat query(3,32,CV_8UC1,cv::Scalar(0)),train(2,32,CV_8UC1,cv::Scalar(0));
    query.at<uchar>(0,0)=1; query.at<uchar>(1,0)=15; train.row(1).setTo(255);
    const std::vector<cv::KeyPoint> q{cv::KeyPoint(10,10,31),cv::KeyPoint(20,10,31),cv::KeyPoint(30,10,31)};
    const std::vector<cv::KeyPoint> t{cv::KeyPoint(10,10,31),cv::KeyPoint(40,10,31)};
    OrbMatchGuidance guidance; guidance.useSphericalGeometry=false;
    OrbMatcherOptions options; OrbMatcher matcher(options);
    std::vector<cv::DMatch> ordinary,retained; OrbMatchStatistics baseline,prioritized;
    matcher.matchGuided(query,train,q,t,guidance,ordinary,&baseline);
    require(ordinary.size()==1 && ordinary[0].queryIdx==2,"unowned exact lookalike must expose the ordinary reciprocal competition");
    const std::vector<uint8_t> ownership{1,1,0}; guidance.queryPriorities=&ownership;
    matcher.matchGuided(query,train,q,t,guidance,retained,&prioritized);
    require(retained.size()==1 && retained[0].queryIdx==0 && retained[0].trainIdx==0,
        "established seed with distinct reciprocal neighbor must survive new texture competition");
    require(prioritized.descriptorComparisons==baseline.descriptorComparisons && prioritized.ownedMatches==1
        && prioritized.prioritySuppressed==1,"ownership must reuse one Hamming pass and report the displaced new proposal");
    options.maxHammingDistance=0; OrbMatcher strict(options);
    strict.matchGuided(query,train,q,t,guidance,retained);
    require(retained.size()==1 && retained[0].queryIdx==2,"ownership cannot bypass the Hamming gate or block a valid new match after failure");
    guidance.queryEnabled={1,0,1};
    matcher.matchGuided(query,train,q,t,guidance,retained);
    require(retained.size()==1 && retained[0].queryIdx==2,"a lone owned reverse candidate cannot fabricate a strict reverse ratio");
}
void parallelCameraInitialization() {
    auto rig=mixedRig(); OrbExtractorOptions options; options.pyramidLevels=1;
    OrbFrontendFeatures frontend(2,options); MultiFrame frame(rig,Time(1,0),1);
    for(size_t c=0;c<2;++c) frame.setImage(c,texture());
    std::array<std::exception_ptr,2> errors{};
    std::array<std::thread,2> workers;
    for(size_t c=0;c<2;++c) workers[c]=std::thread([&,c]{
        try { const std::vector<cv::KeyPoint> points{cv::KeyPoint(80,80,31)};
            frontend.detectAndDescribe(c,frame,kinematics::Transformation(),&points);
        } catch(...) { errors[c]=std::current_exception(); }
    });
    for(auto& worker:workers) worker.join();
    for(const auto& error:errors) if(error) std::rethrow_exception(error);
    frontend.validateFrame(frame);
    require(frame.numKeypoints()==2,"parallel first use has independent camera caches");
    auto otherRig=mixedRig(); MultiFrame changed(otherRig,Time(2,0),2); changed.setImage(0,texture());
    throws([&]{frontend.detectAndDescribe(0,changed,kinematics::Transformation());},"model identity cannot silently change within a session");
}
}
#ifdef CGMLI_EMBED_MODULE_TESTS
void cgmli::debug::registerOrbCases(std::vector<cgmli::debug::ModuleCase>& cases) {
    cases.push_back({"matching","existingIdentityWinsBeforeNewTexture",[]() { existingIdentityWinsBeforeNewTexture(); }});
    cases.push_back({"extraction","mixedCamerasReuseAndProfiles",[]() { mixedCamerasReuseAndProfiles(); }});
    cases.push_back({"extraction","startupCameraPreparation",[]() { startupCameraPreparation(); }});
    cases.push_back({"extraction","parallelCameraInitialization",[]() { parallelCameraInitialization(); }});
    cases.push_back({"matching","deferredRigOverlaps",[]() { deferredRigOverlaps(); }});
    cases.push_back({"matching","sparseGuidedMatchingEquivalence",[]() { sparseGuidedMatchingEquivalence(); }});
    cases.push_back({"matching","hammingAndRowLayout",[]() { hammingAndRowLayout(); }});
    cases.push_back({"matching","groupedMapNeighborsAndPreparedLayouts",[]() { groupedMapNeighborsAndPreparedLayouts(); }});
    cases.push_back({"matching","filteredTrainBucketsPreserveNeighbors",[]() { filteredTrainBucketsPreserveNeighbors(); }});
}
#else
int main() {
    try { deferredRigOverlaps(); existingIdentityWinsBeforeNewTexture(); sparseGuidedMatchingEquivalence(); hammingAndRowLayout(); mixedCamerasReuseAndProfiles(); startupCameraPreparation(); groupedMapNeighborsAndPreparedLayouts(); filteredTrainBucketsPreserveNeighbors(); parallelCameraInitialization();
        std::cout<<"frontend ORB regression passed\n"; return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}

#endif // CGMLI_EMBED_MODULE_TESTS
