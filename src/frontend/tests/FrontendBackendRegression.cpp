// Standalone regression source. Not compiled or executed in this static-only task.
// Link Frontend, frontend helpers, backend, cameras/time, ORB, DBoW2 and PoseLib.
#include <debug/ModuleDebug.hpp>
#include <debug/ModuleDebugArtifacts.hpp>
#include <frontend/Frontend.hpp>
#include <cameras/PinholeCamera.hpp>
#include <cameras/EucmCamera.hpp>
#include <backend/FrameStore.hpp>
#include <backend/WindowVisualFactor.hpp>
#include <iostream>
#include <stdexcept>
#include <set>

using namespace cgmli;
namespace {
void require(bool pass,const char* message) { if(!pass) throw std::runtime_error(message); }
struct Scene {
    cameras::NCameraSystem rig;
    kinematics::Transformation T_BS{Eigen::Vector3d(0.12,-0.03,0.01),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.07,Eigen::Vector3d::UnitY()))};
    std::vector<Eigen::Vector3d> points;
    bool stationary=false;
    explicit Scene(bool holdStill=false):stationary(holdStill) {
        rig.addCamera(std::make_shared<kinematics::Transformation>(Eigen::Vector3d(0.03,0,0),Eigen::Quaterniond::Identity()),
            std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(640,480,220,220,320,240,cameras::NoDistortion()),
            cameras::NCameraSystem::NoDistortion,false);
        rig.addCamera(std::make_shared<kinematics::Transformation>(Eigen::Vector3d(0.25,0,0),Eigen::Quaterniond::Identity()),
            std::make_shared<cameras::EucmCamera>(640,480,120,120,320,240,0.75,1.0),
            cameras::NCameraSystem::NoDistortion,false);
        for(int i=0;i<30;++i) points.emplace_back(-1.0+0.4*(i%6),-0.8+0.4*(i/6),3.0+0.4*((i*7)%9));
        points.emplace_back(2,0,-0.3); // Valid rear-hemisphere EUCM feature, invisible to pinhole.
        rig.computeOverlaps();
    }
    backend::StateEstimate state(uint64_t id) const {
        backend::StateEstimate s; s.id=StateId(id); s.timestamp=Time(uint32_t(id),0);
        const double motion=stationary?0:double(id-1);
        s.T_WB=kinematics::Transformation(Eigen::Vector3d(0.18*motion,0.02*motion,0),
            Eigen::Quaterniond(Eigen::AngleAxisd(0.01*motion,Eigen::Vector3d::UnitZ())));
        s.visual=backend::SensorHealth::Valid; s.lidar=backend::SensorHealth::Valid; return s;
    }
    std::shared_ptr<MultiFrame> frame(uint64_t id) const {
        auto f=std::make_shared<MultiFrame>(rig,state(id).timestamp,id);
        for(size_t c=0;c<rig.numCameras();++c) {
            const auto T_WC=state(id).T_WB*T_BS*(*rig.T_SC(c));
            const auto T_CW=T_WC.inverse();
            std::vector<cv::KeyPoint> keys; cv::Mat descriptors;
            for(size_t i=0;i<points.size();++i) {
                Eigen::Vector2d pixel;
                if(rig.cameraGeometry(c)->project(T_CW.C()*points[i]+T_CW.r(),&pixel)!=cameras::ProjectionStatus::Successful) continue;
                keys.emplace_back(float(pixel.x()),float(pixel.y()),31.0f,-1.0f,1.0f,0,int(i));
                cv::Mat row(1,32,CV_8UC1); cv::RNG rng(891+int(i)*119); rng.fill(row,cv::RNG::UNIFORM,0,256);
                descriptors.push_back(row);
            }
            f->resetKeypoints(c,keys); f->resetDescriptors(c,descriptors); f->setDescriptorProfile(c,"synthetic32");
        }
        return f;
    }
    void initialize(ViSlamBackend& backend) const {
        std::string error;
        require(backend.initializeRig(rig,kinematics::Transformation(),error,0,0,T_BS),"mixed body/IMU rig initialization");
    }
};
void exportedRingRigKeepsStereoGate() {
    Scene s; s.rig=cameras::NCameraSystem(); s.points.clear();
    const double pi=std::acos(-1.0);
    for(size_t c=0;c<4;++c) {
        const double yaw=double(c)*pi/2;
        s.rig.addCamera(std::make_shared<kinematics::Transformation>(
            Eigen::Vector3d(0.1*std::sin(yaw),0,0.1*std::cos(yaw)),
            Eigen::Quaterniond(Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitY()))),
            std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(
                64,48,24,24,32,24,cameras::NoDistortion()),cameras::NCameraSystem::NoDistortion);
        // Points near the middle of adjacent-camera overlap, with distinct
        // descriptors and finite depth, exercise matching as well as graph export.
        for(int i=0;i<9;++i) {
            const double angle=yaw+pi/4+0.015*(i%3-1);
            const Eigen::Vector3d point(4*std::sin(angle),0.3*(i/3-1),4*std::cos(angle));
            s.points.push_back(s.T_BS.C()*point+s.T_BS.r());
        }
    }
    s.rig.computeOverlaps();
    ViSlamBackend backend; s.initialize(backend); std::string error;
    auto frame=s.frame(1); require(backend.addStates(*frame,s.state(1),{},error),"append ring-rig measurements");
    const auto exported=backend.multiFrame(StateId(1));
    require(exported->cameraSystem().overlapComputationValid(),"backend export must preserve valid FOV masks");
    for(size_t a=0;a<4;++a) for(size_t b=a+1;b<4;++b) {
        const bool adjacent=(b-a==1 || b-a==3);
        require(exported->hasOverlap(a,b)==adjacent && exported->hasOverlap(b,a)==adjacent,
            "exported ring permits only adjacent same-epoch camera pairs");
        require(exported->overlap(a,b).data==s.rig.overlap(a,b).data,
            "export reuses precomputed masks instead of recomputing per frame");
    }
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0; options.collectDebugMatches=true;
    options.matcher.allowSingleCandidate=true;
    const auto plan=frontend_association::propose(backend,options);
    require(!plan.debugMatches.empty(),"ring fixture must exercise actual guided matching");
    options.matchingThreads=4; const auto parallel=frontend_association::propose(backend,options);
    require(parallel.debugMatches.size()==plan.debugMatches.size() && parallel.batch.landmarks.size()==plan.batch.landmarks.size(),
        "parallel matching must preserve candidate and track counts");
    for(size_t i=0;i<plan.debugMatches.size();++i) require(parallel.debugMatches[i].first==plan.debugMatches[i].first
        && parallel.debugMatches[i].second==plan.debugMatches[i].second && parallel.debugMatches[i].distance==plan.debugMatches[i].distance,
        "parallel job scheduling must preserve ordered match identities");
    for(size_t i=0;i<plan.batch.landmarks.size();++i) require(parallel.batch.landmarks[i].id==plan.batch.landmarks[i].id
        && parallel.batch.landmarks[i].anchor==plan.batch.landmarks[i].anchor,"parallel matching must preserve monotonic track ownership");
    for(const auto& match:plan.debugMatches) {
        const size_t a=match.first.cameraIndex,b=match.second.cameraIndex;
        require(b-a==1 || b-a==3,"non-overlapping stereo pairs cannot produce debug candidates");
    }
    // A caller that never finalizes a rig must receive an explicit error, even
    // when it has no feature descriptors; an invalid cache is not an all-pairs mode.
    s.rig.setExtrinsics(0,*s.rig.T_SC(0));
    ViSlamBackend invalid; s.initialize(invalid);
    auto stale=s.frame(1); require(invalid.addStates(*stale,s.state(1),{},error),"append stale-cache fixture");
    bool rejected=false;
    try { frontend_association::propose(invalid,options); } catch(const std::logic_error&) { rejected=true; }
    require(rejected,"invalid overlap graph must not bypass the stereo gate");
}
void healthyPromotionBudgetPreservesSeeds() {
    Scene scene; scene.points.clear();
    for(int i=0;i<100;++i) scene.points.emplace_back(-1.2+0.24*(i%10),-0.9+0.2*(i/10),4+0.1*(i%7));
    ViSlamBackend backend; scene.initialize(backend); std::string error;
    auto first=scene.frame(1); require(backend.addStates(*first,scene.state(1),{},error),"budget fixture first state");
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0; options.matcher.allowSingleCandidate=true;
    options.replenishSeedGaps=false; // Isolate rho admission from spatial seed replenishment.
    auto seed=frontend_association::propose(backend,options);
    frontend_association::triangulate(backend,seed,options,scene.state(1).T_WB,false);
    require(backend.applyFrontendBatch(StateId(1),seed.batch,std::nullopt,backend::SensorHealth::Valid,true,error),"budget fixture stereo bootstrap");
    for(int i=0;i<60;++i) scene.points.emplace_back(-1.1+0.22*(i%10),-0.7+0.24*(i/10),5.5+0.1*(i%9));
    auto second=scene.frame(2); require(backend.addStates(*second,scene.state(2),{},error),"budget fixture second state");
    const auto proposal=frontend_association::propose(backend,options);
    require(proposal.finitePoints.size()>=60,"budget fixture must enter established-tracking branch");
    const auto count=[](const auto& plan) { size_t n=plan.batch.initializations.size(); for(const auto& p:plan.batch.landmarks) n+=bool(p.inverseDepth); return n; };
    options.maxPromotionsPerFrame=0; options.maxActiveInitialized=0;
    auto full=proposal; frontend_association::triangulate(backend,full,options,scene.state(2).T_WB,true);
    require(count(full)>3,"budget fixture must offer multiple admissible new ranges");
    options.maxPromotionsPerFrame=3;
    auto capped=proposal; frontend_association::triangulate(backend,capped,options,scene.state(2).T_WB,true);
    require(count(capped)==3 && capped.deferredPromotionBudget==count(full)-3,"per-frame budget must defer rather than drop seed tracks");
    require(capped.batch.landmarks.size()==proposal.batch.landmarks.size() && capped.batch.observations.size()==proposal.batch.observations.size(),"promotion budget preserves raw views and existing associations");
    size_t active=0; for(const auto& p:backend.window().landmarks().landmarks()) active+=p.second.initialized;
    options.maxPromotionsPerFrame=0; options.maxActiveInitialized=active+2;
    auto bounded=proposal; frontend_association::triangulate(backend,bounded,options,scene.state(2).T_WB,true);
    require(count(bounded)==2,"window budget limits only additional initialized ranges");
    require(backend.window().landmarks().landmarks().size()>=active,"proposal cannot retire authoritative landmarks");
}
void proposalOwnershipAndRange() {
    Scene s; ViSlamBackend backend; s.initialize(backend); std::string error;
    auto first=s.frame(1); require(backend.addStates(*first,s.state(1),{},error),"append frozen first measurements");
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0; options.matcher.allowSingleCandidate=true;
    auto plan=frontend_association::propose(backend,options);
    require(backend.numLandmarks()==0 && !plan.batch.landmarks.empty(),"proposal is isolated from authoritative store");
    auto discarded=frontend_association::propose(backend,options);
    const auto anchor=discarded.batch.landmarks.front().anchor;
    const auto removedId=discarded.batch.landmarks.front().id;
    frontend_association::reject(discarded,{anchor});
    for(const auto& observation:discarded.batch.observations) require(observation.landmark!=removedId,"rejected anchor removes the whole new track");
    frontend_association::triangulate(backend,plan,options,s.state(1).T_WB,false);
    size_t mature=0;
    for(const auto& landmark:plan.batch.landmarks) if(landmark.inverseDepth) ++mature;
    require(mature>=10,"metric stereo depth without temporal translation");
    auto transient=frontend_association::propose(backend,options);
    frontend_association::triangulate(backend,transient,options,s.state(1).T_WB,false,false);
    require(transient.deferredPromotions>0 && transient.batch.initializations.empty(),"transient anchor defers promotion");
    for(const auto& point:transient.batch.landmarks) require(!point.inverseDepth,"transient seeds keep measurements without a disposable depth block");
    require(backend.applyFrontendBatch(StateId(1),plan.batch,std::nullopt,backend::SensorHealth::Degraded,true,error),"commit stereo batch");
    for(const auto& item:backend.window().landmarks().landmarks()) {
        auto estimate=backend.window().landmark(item.first); if(!estimate->pointWorld) continue;
        const auto& key=item.second.anchor;
        const auto worldIndex=first->keypoints(key.cameraIndex)[key.keypointIndex].class_id;
        require((*estimate->pointWorld-s.points.at(size_t(worldIndex))).norm()<1e-3,"inverse RANGE on original bearing, with B different from S");
    }
    auto second=s.frame(2); require(backend.addStates(*second,s.state(2),{},error),"append second");
    auto temporal=frontend_association::propose(backend,options);
    frontend_association::filterMap(backend,temporal,s.state(2).T_WB,options.maximumAngularError);
    frontend_association::triangulate(backend,temporal,options,s.state(2).T_WB,false);
    for(const auto& point:temporal.batch.landmarks)
        if(point.anchor.frameId!=2) require(!point.inverseDepth,"unverified temporal motion cannot produce range");
    frontend_association::triangulate(backend,temporal,options,s.state(2).T_WB,true);
    require(backend.applyFrontendBatch(StateId(2),temporal.batch,std::nullopt,backend::SensorHealth::Valid,false,error),"commit temporal observations");
    bool rear=false;
    for(const auto& key:backend.window().landmarks().inFrame(StateId(2))) {
        Eigen::Vector3d ray; const auto& kp=second->keypoints(key.cameraIndex)[key.keypointIndex];
        second->geometry(key.cameraIndex)->backProject(Eigen::Vector2d(kp.pt.x,kp.pt.y),&ray);
        rear=rear || ray.z()<0;
    }
    require(rear,"rear-hemisphere temporal observation retained");
    const auto retired=backend.window().landmarks().landmarks().begin()->first;
    const auto consumed=backend.window().landmarks().find(retired)->anchor;
    require(backend.removeLandmark(retired,error) && backend.isObservationConsumed(consumed),"retired evidence tombstone");
    auto third=s.frame(3); require(backend.addStates(*third,s.state(3),{},error),"append third");
    auto next=frontend_association::propose(backend,options);
    for(const auto& point:next.batch.landmarks) require(!backend.isObservationConsumed(point.anchor),"consumed image evidence cannot become a new anchor");
    for(const auto& observation:next.batch.observations) require(observation.key.frameId==3,"no duplicate historical measurement submission");
    require(backend.window().valid(error),"valid managed graph after plan and retirement");
}
void promotionSurvivesImpendingEviction() {
    Scene scene; backend::BackendOptions settings; settings.windowSize=2;
    ViSlamBackend estimator(settings); scene.initialize(estimator); std::string error;
    for(uint64_t id=1;id<=3;++id) {
        const auto frame=scene.frame(id);
        require(estimator.addStates(*frame,scene.state(id),{},error) && estimator.setKeyframe(StateId(id),true),"promotion window append");
    }
    require(estimator.addLandmark(LandmarkId(1),{1,0,0},std::nullopt,error)
        && estimator.addObservation(LandmarkId(1),StateId(2),0,0,error),"seed anchored in impending oldest victim");
    require(estimator.addLandmark(LandmarkId(2),{2,0,1},std::nullopt,error),"seed with surviving anchor");
    require(estimator.addLandmark(LandmarkId(3),{2,0,2},std::nullopt,error)
        && estimator.addObservation(LandmarkId(3),StateId(1),0,2,error),
        "retained seed anchor also has a fresh outgoing view");
    frontend_association::Plan plan; plan.current=scene.frame(3);
    plan.sources.emplace(StateId(1),scene.frame(1)); plan.sources.emplace(StateId(2),scene.frame(2));
    plan.batch.observations={ {LandmarkId(1),{3,0,0},{}},{LandmarkId(2),{3,0,1},{}},{LandmarkId(3),{3,0,2},{}} };
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0;
    frontend_association::triangulate(estimator,plan,options,scene.state(3).T_WB,true,true);
    require(plan.deferredPromotions==2 && plan.deferredEvictionSupport==1 && plan.batch.initializations.size()==1
        && plan.batch.initializations.front().id==LandmarkId(2),"defer victim-anchor seed while preserving supported promotion");
    require(estimator.applyFrontendBatch(StateId(3),plan.batch,std::nullopt,backend::SensorHealth::Valid,true,error),"commit retained-support promotion");
    require(!estimator.isLandmarkInitialised(LandmarkId(1)) && estimator.isLandmarkInitialised(LandmarkId(2))
        && !estimator.isLandmarkInitialised(LandmarkId(3)),
        "deferral retains identity and current tracking without transient range optimization");
    require(estimator.window().graph().blocks.count({backend::BlockKind::Landmark,3})==0,
        "deferred range never enters Ceres");
}
void nonKeyframeRetainsSeedsWithoutRho() {
    Scene scene; ViSlamBackend estimator; scene.initialize(estimator); std::string error;
    const auto first=scene.frame(1); require(estimator.addStates(*first,scene.state(1),{},error),"reference frame");
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0;
    options.replenishSeedGaps=false; // Exercise identity lifetime even in already covered cells.
    auto initial=frontend_association::propose(estimator,options);
    frontend_association::triangulate(estimator,initial,options,scene.state(1).T_WB,false,true);
    require(estimator.applyFrontendBatch(StateId(1),initial.batch,std::nullopt,backend::SensorHealth::Valid,true,error),"initial map");
    const size_t oldPointCount=scene.points.size();
    for(int i=0;i<8;++i) scene.points.emplace_back(-0.8+0.2*i,0.3,4.2+0.1*i);
    const auto second=scene.frame(2); require(estimator.addStates(*second,scene.state(2),{},error),"non-keyframe with new texture");
    auto plan=frontend_association::propose(estimator,options);
    frontend_association::triangulate(estimator,plan,options,scene.state(2).T_WB,true,false);
    std::set<LandmarkId> newIds;
    for(const auto& point:plan.batch.landmarks) {
        const auto& source=point.anchor.frameId==2?*plan.current:*plan.sources.at(StateId(point.anchor.frameId));
        if(source.keypoints(point.anchor.cameraIndex)[point.anchor.keypointIndex].class_id>=int(oldPointCount)) {
            require(!point.inverseDepth,"new non-keyframe texture stays a seed"); newIds.insert(point.id);
        }
    }
    require(!newIds.empty(),"new non-keyframe seed identities are admitted");
    require(estimator.applyFrontendBatch(StateId(2),plan.batch,std::nullopt,backend::SensorHealth::Valid,false,error),"commit seeds and existing-ID extensions");
    for(auto id:newIds) {
        require(!estimator.isLandmarkInitialised(id) && !estimator.window().graph().blocks.count({backend::BlockKind::Landmark,id.value()}),"seed adds no Ceres coordinate");
        for(const auto& key:estimator.window().landmarks().ofLandmark(id))
            require(estimator.window().landmarks().find(key)->use==backend::ObservationUse::TrackingOnly,"seed views add no visual factors");
    }
    for(uint64_t frameId:{3u,4u}) {
        const auto frame=scene.frame(frameId); require(estimator.addStates(*frame,scene.state(frameId),{},error),"track seed again");
        auto next=frontend_association::propose(estimator,options);
        const size_t continued=std::count_if(next.batch.observations.begin(),next.batch.observations.end(),[&](const auto& o) { return newIds.count(o.landmark)!=0; });
        require(continued>0,"subsequent frames extend the SAME seed IDs");
        frontend_association::triangulate(estimator,next,options,scene.state(frameId).T_WB,true,frameId==4);
        require(estimator.applyFrontendBatch(StateId(frameId),next.batch,std::nullopt,backend::SensorHealth::Valid,frameId==4,error),"atomic tracking/reference/promotion batch");
        if(frameId==3) for(auto id:newIds) require(!estimator.isLandmarkInitialised(id),"no stable keyframe reference yet");
    }
    size_t promoted=0;
    for(auto id:newIds) if(estimator.isLandmarkInitialised(id)) {
        ++promoted; require(estimator.window().landmarks().find(id)->anchor.frameId==4,"promoted seed uses its new keyframe reference");
    }
    require(promoted>0 && estimator.window().valid(error),"supported seeds mature without replacing their IDs");
}
void seedAdmissionIsBoundedAndExpiresAtomically() {
    Scene scene; ViSlamBackend estimator; scene.initialize(estimator); std::string error;
    const auto first=scene.frame(1); require(estimator.addStates(*first,scene.state(1),{},error),"bounded seeds frame");
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0; options.maxTrackingSeeds=2;
    auto initial=frontend_association::propose(estimator,options);
    require(initial.batch.landmarks.size()==2 && initial.newSeedCells==2,"new seeds cover multiple spatial cells within their independent cap");
    require(estimator.applyFrontendBatch(StateId(1),initial.batch,std::nullopt,backend::SensorHealth::Valid,false,error),"bounded seed commit");
    const auto second=scene.frame(2); require(estimator.addStates(*second,scene.state(2),{},error),"full seed pool continuation");
    auto full=frontend_association::propose(estimator,options);
    require(full.batch.landmarks.empty() && !full.batch.observations.empty() && full.seedBudgetSkipped>0,
        "full seed pool still extends existing tracks");
    options.seedMaxUntrackedSeconds=0.5;
    auto expired=frontend_association::propose(estimator,options);
    require(expired.expiredSeeds==2 && expired.batch.retiredLandmarks.size()==2 && !expired.batch.landmarks.empty(),
        "expiry reclaims pure seeds even if their old image is still a selected source");
    auto broken=expired.batch; broken.observations.push_back({broken.landmarks.front().id,{2,99,0},{}});
    require(!estimator.applyFrontendBatch(StateId(2),broken,std::nullopt,backend::SensorHealth::Valid,false,error)
        && estimator.numLandmarks()==2 && estimator.isLandmarkAdded(initial.batch.landmarks.front().id),"failed admission restores expired seed identities");
    require(estimator.applyFrontendBatch(StateId(2),expired.batch,std::nullopt,backend::SensorHealth::Valid,false,error)
        && estimator.numLandmarks()<=2 && !estimator.isLandmarkAdded(initial.batch.landmarks.front().id)
        && estimator.window().valid(error),"expiry and fresh seed admission commit atomically");
}
void seedReferenceUpdateCannotRewriteOptimizedGeometry() {
    Scene scene; ViSlamBackend estimator; scene.initialize(estimator); std::string error;
    const auto first=scene.frame(1); require(estimator.addStates(*first,scene.state(1),{},error),"seed reference old view");
    require(estimator.addLandmark(LandmarkId(1),{1,0,0},std::nullopt,error),"pure seed reference");
    const auto second=scene.frame(2); require(estimator.addStates(*second,scene.state(2),{},error),"seed reference new view");
    backend::VisualUpdateBatch batch; batch.observations.push_back({LandmarkId(1),{2,0,0},{}});
    batch.seedAnchors.push_back({LandmarkId(1),{2,99,0}});
    require(!estimator.applyFrontendBatch(StateId(2),batch,std::nullopt,backend::SensorHealth::Valid,true,error)
        && estimator.window().landmarks().find(LandmarkId(1))->anchor.frameId==1 && !estimator.isObserved({2,0,0}),"bad reference rolls back the newly added view too");
    batch.seedAnchors.front().anchor={2,0,0};
    const auto camera=scene.state(2).T_WB*scene.T_BS*(*scene.rig.T_SC(0));
    batch.initializations.push_back({LandmarkId(1),1.0/(scene.points[0]-camera.r()).norm()});
    require(estimator.applyFrontendBatch(StateId(2),batch,std::nullopt,backend::SensorHealth::Valid,true,error),"select fresh reference before creating rho");
    const auto point=estimator.window().landmark(LandmarkId(1));
    require(point && point->pointWorld && (*point->pointWorld-scene.points[0]).norm()<1e-5,"new rho chart reproduces the physical point");
    backend::VisualUpdateBatch unsafe; unsafe.seedAnchors.push_back({LandmarkId(1),{1,0,0}});
    require(!estimator.applyFrontendBatch(StateId(2),unsafe,std::nullopt,backend::SensorHealth::Valid,true,error)
        && estimator.window().landmarks().find(LandmarkId(1))->anchor.frameId==2 && estimator.window().valid(error),"a seed-reference update cannot move an already optimized coordinate");
    require(estimator.removeLandmark(LandmarkId(1),error) && estimator.isObservationConsumed({1,0,0}),"old optimized measurements become consumed evidence");
    require(estimator.addLandmark(LandmarkId(2),{2,0,1},std::nullopt,error),"fresh independent seed");
    const auto prior=estimator.window().graph().prior;
    backend::VisualUpdateBatch replay; replay.seedAnchors.push_back({LandmarkId(2),{1,0,0}});
    require(!estimator.applyFrontendBatch(StateId(2),replay,std::nullopt,backend::SensorHealth::Valid,true,error)
        && estimator.window().graph().prior==prior && estimator.isObservationConsumed({1,0,0})
        && !estimator.window().graph().blocks.count({backend::BlockKind::Landmark,2}),"seed selection cannot replay an absorbed ray or change its prior");
}
void imageTimingSurvivesFrozenCopies() {
    Scene s; ViSlamBackend backend; s.initialize(backend); std::string error;
    auto frame=s.frame(1);
    Time early,late; early.fromNSec(994999000); late.fromNSec(1005001000);
    frame->setImageTiming(0,early,-5001000); frame->setImageTiming(1,late,5001000);
    require(backend.addStates(*frame,s.state(1),{},error),"append mean-time frame with source timing metadata");
    const auto frozen=backend.multiFrame(StateId(1));
    require(frozen->timestamp().toNSec()==1000000000 && frozen->imageTiming(0)
        && frozen->imageTiming(0)->originalTimestamp.toNSec()==early.toNSec()
        && frozen->imageTiming(1)->offsetNs==5001000,"Frozen backend frame lost original image timing");
    require(frontend_association::sameMeasurements(*frame,*frozen),"Identical source timing must pass frozen ownership check");
    frame->setImageTiming(0,late,5001000);
    require(!frontend_association::sameMeasurements(*frame,*frozen),"Changing source timing must not mutate backend measurements");
    require(frozen->imageTiming(0)->offsetNs==-5001000,"Caller mutation leaked into frozen timing");
}
void sameCameraTimingUsesExposureMidpoints() {
    Scene scene; ViSlamBackend estimator; scene.initialize(estimator); std::string error;
    auto first=scene.frame(1), current=scene.frame(2);
    auto firstState=scene.state(1), currentState=scene.state(2);
    currentState.timestamp=Time(1,100000000); current->setTimestamp(currentState.timestamp);
    first->setImageTiming(0,Time(1,2000000),2000000);
    current->setImageTiming(0,Time(1,107000000),7000000);
    require(estimator.addStates(*first,firstState,{},error) && estimator.addStates(*current,currentState,{},error),
        "append same-camera track with asynchronous exposure midpoints");
    frontend_association::Plan plan; plan.current=current;
    plan.batch.landmarks.push_back({LandmarkId(1),{1,0,0},std::nullopt,{}});
    plan.batch.observations.push_back({LandmarkId(1),{2,0,0},{}});
    plan.batch.observations.push_back({LandmarkId(1),{2,1,0},{}});
    frontend_association::prepareBearingVelocities(estimator,plan);
    const auto& anchor=plan.batch.landmarks.front().options;
    const auto& observation=plan.batch.observations.front().options;
    require(anchor.referenceTimeOffset && observation.referenceTimeOffset
        && plan.batch.observations.back().options.bearingVelocity.squaredNorm()==0,
        "only same-camera tracks supply the spherical time derivative");
    debug::recordError("anchor_offset_seconds",std::abs(*anchor.referenceTimeOffset+0.002),1e-12);
    debug::recordError("view_offset_seconds",std::abs(*observation.referenceTimeOffset+0.007),1e-12);
    Eigen::Vector3d a,b;
    const auto ray=[](const MultiFrame& frame,Eigen::Vector3d& out) {
        const auto& pixel=frame.keypoints(0).front().pt;
        return backend::pixelToBearing(*frame.geometry(0),Eigen::Vector2d(pixel.x,pixel.y),out);
    };
    require(ray(*first,a) && ray(*current,b),"usable raw bearings");
    const auto angle=[](const Eigen::Vector3d& x,const Eigen::Vector3d& y) { return std::atan2(x.cross(y).norm(),x.dot(y)); };
    const double measuredAngle=angle(a,b);
    require(measuredAngle>1e-3,"nonzero track motion");
    debug::recordError("exposure_velocity_scale",std::abs(observation.bearingVelocity.norm()-measuredAngle/0.105),1e-10);
    const auto corrected=backend::timeBearing(b,observation.bearingVelocity,*observation.referenceTimeOffset-0.003);
    debug::recordError("positive_delay_rewinds_bearing",std::abs(angle(a,corrected)-measuredAngle*(0.095/0.105)),1e-10);
}
void frontendCallOrderAndFrozenIndices() {
    Scene s; ViSlamBackend backend; s.initialize(backend);
    features::OrbExtractorOptions extraction; extraction.pyramidLevels=1;
    Frontend frontend(2,extraction);
    frontend_association::Options matching; matching.seedMaxUntrackedSeconds=0; matching.matcher.allowSingleCandidate=true; frontend.setAssociationOptions(matching);
    auto profileFrame=s.frame(99);
    for(size_t c=0;c<2;++c) {
        profileFrame->setImage(c,cv::Mat(480,640,CV_8UC1,cv::Scalar(90)));
        const std::vector<cv::KeyPoint> one{cv::KeyPoint(320,240,31)};
        frontend.detectAndDescribe(c,profileFrame,s.state(1).T_WB*s.T_BS*(*s.rig.T_SC(c)),&one);
    }
    // Synthetic descriptor identities isolate association/pose/backend contracts
    // from photometric extraction, which has its own independent regression source.
    const auto frame=[&](uint64_t id) {
        auto f=s.frame(id); for(size_t c=0;c<2;++c) f->setDescriptorProfile(c,profileFrame->descriptorProfile(c)); return f;
    };
    ViParameters params{}; params.frontend.seed_max_untracked_seconds=0; params.nCameraSystem=s.rig; params.estimator.do_loop_closures=false;
    std::string error; bool keyframe=false;
    for(uint64_t id=1;id<=2;++id) {
        auto f=frame(id); require(backend.addStates(*f,s.state(id),{},error),"append before frontend");
        require(frontend.dataAssociationAndInitialization(backend,params,f,false,&keyframe),"managed frontend ingestion");
        require(frontend.lastAssociationReport().committed && !backend.snapshot(),"ingestion does not optimize or publish");
        const auto frozen=backend.multiFrame(StateId(id));
        for(size_t c=0;c<2;++c) for(size_t i=0;i<f->numKeypoints(c);++i)
            require(frozen->landmarkId(c,i)==f->landmarkId(c,i),"caller IDs equal authoritative observation table");
        if(id==2) require(frontend.lastAssociationReport().trackingAccepted && frontend.isInitialized(),"absolute consensus uses finite stereo map");
    }
    auto f=frame(3); require(backend.addStates(*f,s.state(3),{},error),"append mismatch fixture");
    cv::Mat changed=f->descriptors(0).clone(); changed.at<uchar>(0,0)^=1;
    const auto profile=f->descriptorProfile(0); f->resetDescriptors(0,changed); f->setDescriptorProfile(0,profile);
    const auto count=backend.numLandmarks(); keyframe=true;
    require(!frontend.dataAssociationAndInitialization(backend,params,f,false,&keyframe),"changed frozen descriptor rejected");
    require(!frontend.lastAssociationReport().committed && backend.numLandmarks()==count
        && backend.window().landmarks().inFrame(StateId(3)).empty() && keyframe,"failed association leaves backend and keyframe output unchanged");
}
void capacityPreservesExistingObservations() {
    Scene s; backend::BackendOptions limits; limits.maxLandmarks=3;
    ViSlamBackend estimator(limits); s.initialize(estimator); std::string error;
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0; options.matcher.allowSingleCandidate=true;
    auto first=s.frame(1); require(estimator.addStates(*first,s.state(1),{},error),"capacity first frame");
    auto seed=frontend_association::propose(estimator,options);
    require(seed.admittedTrackLimit==3 && seed.batch.landmarks.size()==3,"new seeds must respect remaining capacity");
    frontend_association::triangulate(estimator,seed,options,s.state(1).T_WB,false);
    require(estimator.applyFrontendBatch(StateId(1),seed.batch,std::nullopt,backend::SensorHealth::Valid,true,error),"capacity seed commit");
    require(estimator.window().remainingLandmarkCapacity()==0,"fixture must fill landmark capacity");
    for(uint64_t id=2;id<=3;++id) {
        auto frame=s.frame(id); require(estimator.addStates(*frame,s.state(id),{},error),"append while landmark budget full");
        auto next=frontend_association::propose(estimator,options);
        require(next.batch.landmarks.empty() && !next.batch.observations.empty(),"full landmark budget must retain existing-track observations");
        require(estimator.applyFrontendBatch(StateId(id),next.batch,std::nullopt,backend::SensorHealth::Valid,false,error),"full capacity must not reject entire visual batch");
        require(!estimator.window().landmarks().inFrame(StateId(id)).empty() && estimator.window().valid(error),"visual updates must continue beyond capacity boundary");
    }
}
void staleSeedRetirementIsAtomic() {
    Scene s; backend::BackendOptions limits; limits.maxLandmarks=2; limits.maxObservations=4;
    ViSlamBackend estimator(limits); s.initialize(estimator); std::string error;
    for(uint64_t id=1;id<=3;++id) {
        auto frame=s.frame(id); require(estimator.addStates(*frame,s.state(id),{},error),"stale seed fixture frames");
        if(id==1) {
            require(estimator.addLandmark(LandmarkId(1),{1,0,0},std::nullopt,error),"tentative old anchor");
            require(estimator.addObservation(LandmarkId(1),StateId(1),1,0,error),"tentative old stereo observation");
        }
    }
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0; options.maxSourceFrames=1; options.matcher.allowSingleCandidate=true;
    auto plan=frontend_association::propose(estimator,options);
    require(plan.retiredSeeds==1 && plan.admittedTrackLimit==2 && !plan.batch.landmarks.empty(),"stale seed frees both track and observation capacity");
    const auto used=plan.batch.landmarks.size()+plan.batch.observations.size();
    require(used<=4,"admission must count anchors and all added views");
    auto malformed=plan.batch; malformed.observations.push_back({plan.batch.landmarks.front().id,{3,99,0},{}});
    require(!estimator.applyFrontendBatch(StateId(3),malformed,std::nullopt,backend::SensorHealth::Valid,false,error),"late failure must roll back retirement and insertions");
    require(estimator.isLandmarkAdded(LandmarkId(1)) && estimator.numLandmarks()==1
        && estimator.window().remainingObservationCapacity()==2,"failed transaction must restore tentative evidence and budgets");
    require(estimator.applyFrontendBatch(StateId(3),plan.batch,std::nullopt,backend::SensorHealth::Valid,false,error),"stale seed admission transaction");
    require(!estimator.isLandmarkAdded(LandmarkId(1)) && estimator.numLandmarks()==plan.batch.landmarks.size()
        && estimator.window().remainingObservationCapacity()==4-used && estimator.window().valid(error),"recycled capacity has consistent reverse indices");
}
void immutableRayCacheSurvivesCopies() {
    Scene s; ViSlamBackend estimator; s.initialize(estimator); std::string error;
    auto source=s.frame(1);
    for(size_t c=0;c<source->numFrames();++c) source->computeBackProjections(c);
    Eigen::Vector3d expected,actual;
    require(source->getBackProjection(0,0,expected),"source ray cache fixture");
    require(estimator.addStates(*source,s.state(1),{},error),"freeze cached feature rays");
    const auto exported=estimator.multiFrame(StateId(1));
    require(exported->getBackProjection(0,0,actual) && (actual-expected).norm()==0,"frozen/exported frame lost feature ray cache");
    auto working=backend::FrameStore::copyMeasurements(*exported,false);
    require(working->getBackProjection(0,0,actual) && (actual-expected).norm()==0,"working copy lost shared rays");
    source->resetKeypoints(0,{});
    require(exported->getBackProjection(0,0,actual) && working->getBackProjection(0,0,actual),"caller reset must detach without invalidating frozen cache");
    auto other=s.rig.cameraGeometry(1); working->setGeometry(0,other);
    require(!working->getBackProjection(0,0,actual) && exported->getBackProjection(0,0,actual),"geometry replacement must invalidate only local rays");
}
void stagedMapMatchingAndCoverageFallback() {
    Scene s; ViSlamBackend estimator; s.initialize(estimator); std::string error;
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0; options.matcher.allowSingleCandidate=true;
    auto first=s.frame(1); require(estimator.addStates(*first,s.state(1),{},error),"staged map bootstrap");
    auto seeds=frontend_association::propose(estimator,options);
    frontend_association::triangulate(estimator,seeds,options,s.state(1).T_WB,false);
    require(estimator.applyFrontendBatch(StateId(1),seeds.batch,std::nullopt,backend::SensorHealth::Valid,true,error),"staged map finite points");
    auto second=s.frame(2); require(estimator.addStates(*second,s.state(2),{},error),"staged map next image");
    options.minimumStageMatchesPerCamera=5;
    auto staged=frontend_association::propose(estimator,options);
    options.stagedMatching=false; auto reference=frontend_association::propose(estimator,options);
    require(staged.stagedUsed && staged.mapJobs>0,"finite points must use landmark-organized map matching");
    std::map<KeypointIdentifier,LandmarkId> expected,actual;
    for(const auto& observation:reference.batch.observations) if(estimator.isLandmarkInitialised(observation.landmark)) expected[observation.key]=observation.landmark;
    for(const auto& observation:staged.batch.observations) if(estimator.isLandmarkInitialised(observation.landmark)) actual[observation.key]=observation.landmark;
    require(actual.size()>=10,"staged fixture must retain a usable finite map");
    for(const auto& observation:actual) require(expected.count(observation.first) && expected.at(observation.first)==observation.second,
        "map-first matching changed physical point identity");
    options.stagedMatching=true; options.minimumStageMatchesPerCamera=10000;
    auto broad=frontend_association::propose(estimator,options);
    auto stages=debug::artifactFile("coverage_fallback.csv");
    if(stages) {
        stages<<"strategy,seed_jobs,expansion_jobs,broad_jobs,broad_flag,reason\n";
        stages<<"exhaustive,"<<reference.seedJobs<<','<<reference.expansionJobs<<','<<reference.broadJobs<<','<<reference.broadFallback<<','<<reference.stageFallbackReason<<'\n';
        stages<<"coverage_fallback,"<<broad.seedJobs<<','<<broad.expansionJobs<<','<<broad.broadJobs<<','<<broad.broadFallback<<','<<broad.stageFallbackReason<<'\n';
    }
    require(broad.seedJobs==reference.seedJobs && broad.expansionJobs+broad.broadJobs>0,
        "insufficient support must recover every temporal camera-pair job");
    require(broad.broadFallback==(broad.broadJobs>0),"broad flag describes the broad stage, not earlier expansion");
    std::map<KeypointIdentifier,LandmarkId> expanded;
    for(const auto& observation:broad.batch.observations)
        if(estimator.isLandmarkInitialised(observation.landmark)) expanded[observation.key]=observation.landmark;
    require(expanded==expected,"coverage recovery must preserve exhaustive finite-point associations");
    require(estimator.window().landmarks().inFrame(StateId(2)).empty(),"all matching passes must remain private until commit");

    // One starved target must expand its own history without forcing the healthy
    // target to search every source camera. Keep the backend snapshot immutable.
    ViSlamBackend partial; s.initialize(partial);
    require(partial.addStates(*first,s.state(1),{},error),"partial coverage bootstrap");
    options.minimumStageMatchesPerCamera=5;
    auto initial=frontend_association::propose(partial,options);
    frontend_association::triangulate(partial,initial,options,s.state(1).T_WB,false);
    require(partial.applyFrontendBatch(StateId(1),initial.batch,std::nullopt,backend::SensorHealth::Valid,true,error),"partial coverage map");
    auto sparse=s.frame(2); const auto keys=sparse->keypoints(1); const auto descriptors=sparse->descriptors(1).row(0).clone();
    sparse->resetKeypoints(1,{keys.front()}); sparse->resetDescriptors(1,descriptors); sparse->setDescriptorProfile(1,"synthetic32");
    require(partial.addStates(*sparse,s.state(2),{},error),"partial coverage next image");
    const auto local=frontend_association::propose(partial,options);
    options.stagedMatching=false; const auto complete=frontend_association::propose(partial,options);
    require(local.stagePrimaryMatches[0]>=5 && local.stagePrimaryMatches[1]<5,"partial fixture must isolate the weak target camera");
    require(local.expansionJobs>0 && local.seedJobs<complete.seedJobs,"Weak camera must not expand healthy-camera jobs");
    require(partial.window().landmarks().inFrame(StateId(2)).empty(),"Per-camera expansion must remain a private proposal");
}
void seedReplenishmentPreservesExistingCoverage() {
    Scene scene; scene.points.clear();
    for(int i=0;i<100;++i) scene.points.emplace_back(-1.2+0.24*(i%10),-0.9+0.2*(i/10),4+0.1*(i%7));
    ViSlamBackend estimator; scene.initialize(estimator); std::string error;
    frontend_association::Options options; options.seedMaxUntrackedSeconds=0; options.matcher.allowSingleCandidate=true;
    auto first=scene.frame(1); require(estimator.addStates(*first,scene.state(1),{},error),"density reference state");
    auto map=frontend_association::propose(estimator,options);
    frontend_association::triangulate(estimator,map,options,scene.state(1).T_WB,false,true);
    require(estimator.applyFrontendBatch(StateId(1),map.batch,std::nullopt,backend::SensorHealth::Valid,true,error),"density reference map");
    for(int i=0;i<60;++i) scene.points.emplace_back(-1.1+0.22*(i%10),-0.7+0.24*(i/10),5.5+0.1*(i%9));
    auto second=scene.frame(2); require(estimator.addStates(*second,scene.state(2),{},error),"density current frame");
    const auto gaps=frontend_association::propose(estimator,options);
    options.replenishSeedGaps=false;
    const auto full=frontend_association::propose(estimator,options);
    require(gaps.seedCoverageSkipped>0 && !full.batch.landmarks.empty()
        && gaps.batch.landmarks.size()<full.batch.landmarks.size(),"healthy covered cells must stop manufacturing new seed identities");
    std::map<KeypointIdentifier,LandmarkId> a,b;
    for(const auto& view:gaps.batch.observations) if(estimator.isLandmarkInitialised(view.landmark)) a.emplace(view.key,view.landmark);
    for(const auto& view:full.batch.observations) if(estimator.isLandmarkInitialised(view.landmark)) b.emplace(view.key,view.landmark);
    require(a==b && a.size()>=60,"density admission cannot remove established map extensions");
    options.stagedMatching=false; options.replenishSeedGaps=true;
    const auto recovery=frontend_association::propose(estimator,options);
    require(recovery.seedCoverageSkipped==0,"exhaustive recovery must bypass healthy-map replenishment quotas");
    auto detached=estimator.associationFrame(StateId(1));
    const auto frozen=estimator.window().frames().measurements(StateId(1));
    require(detached && detached->descriptors(0).data!=frozen->descriptors(0).data,"proposal descriptors must be writable detached copies");
    detached->setLandmarkId(0,0,0);
    require(estimator.multiFrame(StateId(1))->landmarkId(0,0)!=0,"editing a proposal cannot change an authoritative ID");
    require(estimator.window().valid(error),"density proposals and detached edits leave graph/index invariants intact");
}
void missingOrFailedPropagation() {
    Frontend frontend(1); kinematics::Transformation pose; SpeedAndBias sb=SpeedAndBias::Zero();
    Eigen::Matrix<double,15,15> cov=Eigen::Matrix<double,15,15>::Identity(),jac=cov;
    require(!frontend.propagation({},ImuParameters{},pose,sb,Time(1,0),Time(2,0),&cov,&jac),"missing IMU module is explicit failure");
    require(cov.isIdentity() && jac.isIdentity() && sb.norm()==0,"unavailable propagation preserves outputs");
    frontend.setImuPropagator([](const ImuMeasurementDeque&,const ImuParameters&,kinematics::Transformation& pose,
        SpeedAndBias& sb,const Time&,const Time&,Eigen::Matrix<double,15,15>* cov,Eigen::Matrix<double,15,15>* jac) {
        pose=kinematics::Transformation(Eigen::Vector3d(9,8,7),Eigen::Quaterniond::Identity());
        sb.setOnes(); if(cov) cov->setZero(); if(jac) jac->setZero(); return false;
    });
    const ImuMeasurementDeque measurements(2);
    require(!frontend.propagation(measurements,ImuParameters{},pose,sb,Time(1,0),Time(2,0),&cov,&jac),"failed injected prediction");
    require(pose.r().norm()==0 && sb.norm()==0 && cov.isIdentity() && jac.isIdentity(),"failed prediction rolls back every output");
}
void stationaryKeyframesPreserveMatureAnchors() {
    Scene scene(true); backend::BackendOptions settings; settings.windowSize=10; settings.maxIterations=6; settings.maxSolverTimeSeconds=30;
    ViSlamBackend estimator(settings); scene.initialize(estimator); std::string error;
    frontend_association::Options matching; matching.seedMaxUntrackedSeconds=0; matching.matcher.allowSingleCandidate=true;
    FrontendParameters keyframes{}; std::set<LandmarkId> original;
    auto trace=debug::artifactFile("stationary_window.csv");
    if(trace) trace<<"frame,keyframe,reason,shared,parallax_rad,policy,removed,retired_mature,original_mature_survivors\n";
    for(uint64_t id=1;id<=35;++id) {
        const auto frame=scene.frame(id); require(estimator.addStates(*frame,scene.state(id),{},error),"append stationary textured rig");
        auto plan=frontend_association::propose(estimator,matching);
        frontend_association::triangulate(estimator,plan,matching,scene.state(id).T_WB,true);
        const auto decision=frontend_association::decideKeyframe(estimator,*plan.current,scene.state(id).T_WB,keyframes,true);
        if(id>2) require(!decision.keyframe && decision.sharedTracks>=20,"stationary shared bearings must not create keyframes from sparse grid coverage");
        require(estimator.applyFrontendBatch(StateId(id),plan.batch,std::nullopt,backend::SensorHealth::Valid,decision.keyframe,error),"commit stationary observations");
        if(id==1) for(const auto& point:estimator.window().landmarks().landmarks()) if(point.second.initialized) original.insert(point.first);
        require(original.size()>=20,"stationary test must start with a real stereo map");
        const auto result=estimator.optimize(); require(result.accepted,"stationary textured window must optimize and marginalize");
        size_t surviving=0; for(auto point:original) if(estimator.window().landmarks().find(point)) ++surviving;
        if(trace) trace<<id<<','<<decision.keyframe<<','<<decision.reason<<','<<decision.sharedTracks<<','<<decision.parallax<<','
            <<(result.marginalization?(result.marginalization->policy==backend::MarginalizationPolicy::MARGIN_SECOND_NEW?"SECOND_NEW":"OLD"):"NONE")<<','
            <<(result.retiredState?result.retiredState->id.value():0)<<','<<(result.marginalization?result.marginalization->retiredMature:0)<<','<<surviving<<'\n';
        require(surviving==original.size(),"stationary SECOND_NEW must retain original mature anchors");
        if(id>10) require(result.marginalization && result.marginalization->policy==backend::MarginalizationPolicy::MARGIN_SECOND_NEW
            && result.retiredState->id==StateId(id-1),"stationary overflow must remove the penultimate frame");
    }
    // Positive control: moving geometry still creates a new keyframe.
    Scene moving; ViSlamBackend other; moving.initialize(other);
    for(uint64_t id=1;id<=2;++id) {
        auto frame=moving.frame(id); require(other.addStates(*frame,moving.state(id),{},error),"motion-control append");
        auto plan=frontend_association::propose(other,matching); frontend_association::triangulate(other,plan,matching,moving.state(id).T_WB,true);
        require(other.applyFrontendBatch(StateId(id),plan.batch,std::nullopt,backend::SensorHealth::Valid,true,error),"motion-control reference");
    }
    auto current=moving.frame(3); require(other.addStates(*current,moving.state(3),{},error),"motion-control current");
    const auto plan=frontend_association::propose(other,matching);
    const auto decision=frontend_association::decideKeyframe(other,*plan.current,moving.state(3).T_WB,keyframes,true);
    require(decision.keyframe && decision.reason=="accumulated_motion","accumulated motion must not be suppressed as stationarity");
}
}
#ifdef CGMLI_EMBED_MODULE_TESTS
void cgmli::debug::registerAssociationCases(std::vector<cgmli::debug::ModuleCase>& cases) {
    cases.push_back({"tracking","seedReplenishmentPreservesExistingCoverage",[]() { seedReplenishmentPreservesExistingCoverage(); }});
    cases.push_back({"matching","exportedRingRigKeepsStereoGate",[]() { exportedRingRigKeepsStereoGate(); }});
    cases.push_back({"initialization","imageTimingSurvivesFrozenCopies",[]() { imageTimingSurvivesFrozenCopies(); }});
    cases.push_back({"tracking","sameCameraTimingUsesExposureMidpoints",[]() { sameCameraTimingUsesExposureMidpoints(); }});
    cases.push_back({"backend","proposalOwnershipAndRange",[]() { proposalOwnershipAndRange(); }});
    cases.push_back({"backend","healthyPromotionBudgetPreservesSeeds",[]() { healthyPromotionBudgetPreservesSeeds(); }});
    cases.push_back({"backend","promotionSurvivesImpendingEviction",[]() { promotionSurvivesImpendingEviction(); }});
    cases.push_back({"backend","nonKeyframeRetainsSeedsWithoutRho",[]() { nonKeyframeRetainsSeedsWithoutRho(); }});
    cases.push_back({"backend","seedAdmissionIsBoundedAndExpiresAtomically",[]() { seedAdmissionIsBoundedAndExpiresAtomically(); }});
    cases.push_back({"backend","seedReferenceUpdateCannotRewriteOptimizedGeometry",[]() { seedReferenceUpdateCannotRewriteOptimizedGeometry(); }});
    cases.push_back({"backend","capacityPreservesExistingObservations",[]() { capacityPreservesExistingObservations(); }});
    cases.push_back({"backend","staleSeedRetirementIsAtomic",[]() { staleSeedRetirementIsAtomic(); }});
    cases.push_back({"backend","immutableRayCacheSurvivesCopies",[]() { immutableRayCacheSurvivesCopies(); }});
    cases.push_back({"tracking","frontendCallOrderAndFrozenIndices",[]() { frontendCallOrderAndFrozenIndices(); }});
    cases.push_back({"tracking","stagedMapMatchingAndCoverageFallback",[]() { stagedMapMatchingAndCoverageFallback(); }});
    cases.push_back({"tracking","missingOrFailedPropagation",[]() { missingOrFailedPropagation(); }});
    cases.push_back({"tracking","stationaryKeyframesPreserveMatureAnchors",[]() { stationaryKeyframesPreserveMatureAnchors(); }});
}
#else
int main() {
    try { seedReplenishmentPreservesExistingCoverage(); exportedRingRigKeepsStereoGate(); imageTimingSurvivesFrozenCopies(); proposalOwnershipAndRange(); frontendCallOrderAndFrozenIndices(); capacityPreservesExistingObservations(); staleSeedRetirementIsAtomic(); immutableRayCacheSurvivesCopies(); stagedMapMatchingAndCoverageFallback(); missingOrFailedPropagation(); stationaryKeyframesPreserveMatureAnchors();
        healthyPromotionBudgetPreservesSeeds(); promotionSurvivesImpendingEviction();
        nonKeyframeRetainsSeedsWithoutRho(); seedAdmissionIsBoundedAndExpiresAtomically(); seedReferenceUpdateCannotRewriteOptimizedGeometry();
        sameCameraTimingUsesExposureMidpoints();
        std::cout<<"frontend/backend regression passed\n"; return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}

#endif // CGMLI_EMBED_MODULE_TESTS
