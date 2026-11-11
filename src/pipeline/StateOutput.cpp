#include "StateOutput.hpp"
#include <factor/BodyImuFactor.hpp>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cmath>
#include <stdexcept>

namespace cgmli { namespace pipeline {
backend::StateEstimate bodyPrediction(const State& sensor,const kinematics::Transformation& T_BS) {
    backend::StateEstimate body; body.id=sensor.id; body.timestamp=sensor.timestamp;
    body.T_WB=sensor.T_WS*T_BS.inverse(); body.speedBias<<sensor.v_W,sensor.b_g,sensor.b_a;
    body.speedBias.head<3>()-=(sensor.T_WS.C()*sensor.omega_S).cross(body.T_WB.C()*T_BS.r());
    body.visual=backend::SensorHealth::Degraded; body.keyframe=sensor.isKeyframe; return body;
}
State sensorState(const backend::StateEstimate& body,const kinematics::Transformation& T_BS,
    const ImuMeasurementDeque& samples,const Eigen::Vector3d& fallbackOmega) {
    State state{}; state.id=body.id; state.timestamp=body.timestamp; state.T_WS=body.T_WB*T_BS;
    state.b_g=body.speedBias.segment<3>(3); state.b_a=body.speedBias.tail<3>();
    state.omega_S=samples.empty()?fallbackOmega:factor::gyroAt(samples,body.timestamp)-state.b_g;
    state.v_W=body.speedBias.head<3>()+(state.T_WS.C()*state.omega_S).cross(body.T_WB.C()*T_BS.r());
    state.previousImuMeasurements=samples; state.isKeyframe=body.keyframe; state.stateChanged=true; return state;
}
AlignedVector<kinematics::Transformation> cameraExtrinsics(const backend::BackendSnapshot& snapshot,
    const cameras::NCameraSystem& rig,const kinematics::Transformation& T_BS) {
    const auto it=snapshot.calibration.find({backend::BlockKind::CameraExtrinsic,0});
    if(it==snapshot.calibration.end()) throw std::invalid_argument("Snapshot has no camera rig extrinsic");
    backend::ParameterBlock block; block.value=it->second;
    const auto T_SR=T_BS.inverse()*backend::blockPose(block);
    AlignedVector<kinematics::Transformation> result;
    for(size_t c=0;c<rig.numCameras();++c) result.push_back(T_SR*rig.T_SC(0)->inverse()*(*rig.T_SC(c)));
    return result;
}
std::shared_ptr<MapPointVector> mapPoints(const backend::BackendSnapshot& snapshot) {
    auto result=std::make_shared<MapPointVector>();
    for(const auto& point:snapshot.landmarks) if(point.pointWorld) {
        uint64_t latest=0; for(const auto& observation:point.observations) latest=std::max(latest,observation.key.frameId);
        const auto& p=*point.pointWorld;
        result->emplace_back(point.metadata.id.value(),Eigen::Vector4d(p.x(),p.y(),p.z(),1),point.metadata.quality,latest);
    }
    return result;
}
static double gridTrackingCoverage(const MultiFrame& frame,const std::set<KeypointIdentifier>& tracked) {
    size_t total=0,matched=0;
    for(size_t c=0;c<frame.numFrames();++c) {
        std::set<int> all,good; const auto camera=frame.geometry(c);
        for(size_t k=0;k<frame.numKeypoints(c);++k) {
            const auto pixel=frame.keypoints(c)[k].pt;
            const int cell=std::clamp(int(pixel.y*6/camera->imageHeight()),0,5)*8+std::clamp(int(pixel.x*8/camera->imageWidth()),0,7);
            all.insert(cell); if(tracked.count({frame.id(),c,k})) good.insert(cell);
        }
        total+=all.size(); matched+=good.size();
    }
    return total?double(matched)/total:0;
}
double trackingCoverage(const MultiFrame& frame,const backend::BackendSnapshot& snapshot) {
    std::set<KeypointIdentifier> tracked;
    for(const auto& point:snapshot.landmarks) if(point.pointWorld) for(const auto& o:point.observations)
        if(o.key.frameId==frame.id() && o.use==backend::ObservationUse::LiveFactor) tracked.insert(o.key);
    return gridTrackingCoverage(frame,tracked);
}
double trackingCoverage(const MultiFrame& frame,const backend::LandmarkStore& landmarks,const backend::BackendGraph& graph) {
    std::set<KeypointIdentifier> tracked;
    for(const auto& key:landmarks.inFrame(StateId(frame.id()))) {
        const auto* observation=landmarks.find(key);
        if(!observation || observation->use!=backend::ObservationUse::LiveFactor) continue;
        const auto factor=graph.factors.find(observation->factorId);
        if(factor!=graph.factors.end() && factor->second.participates()) tracked.insert(key);
    }
    return gridTrackingCoverage(frame,tracked);
}
bool writeTrajectory(const std::string& path,const StateHistory& history,bool rpg) {
    std::ofstream file(std::filesystem::u8path(path)); if(!file) return false; file<<std::setprecision(17);
    file<<(rpg?"# timestamp tx ty tz qx qy qz qw\n":"timestamp,px,py,pz,qw,qx,qy,qz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz\n");
    for(const auto& item:history) {
        const auto& s=item.second; const auto& p=s.T_WS.r(); const auto q=s.T_WS.q();
        if(rpg) file<<s.timestamp.toSec()<<' '<<p.transpose()<<' '<<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<'\n';
        else file<<s.timestamp.toNSec()<<','<<p.x()<<','<<p.y()<<','<<p.z()<<','<<q.w()<<','<<q.x()<<','<<q.y()<<','<<q.z()
            <<','<<s.v_W.x()<<','<<s.v_W.y()<<','<<s.v_W.z()<<','<<s.b_g.x()<<','<<s.b_g.y()<<','<<s.b_g.z()
            <<','<<s.b_a.x()<<','<<s.b_a.y()<<','<<s.b_a.z()<<'\n';
    }
    return bool(file);
}
bool writeMap(const std::string& path,const MapPointVector& points) {
    std::ofstream file(std::filesystem::u8path(path)); if(!file) return false; file<<std::setprecision(17)<<"id,x,y,z,quality,state_id\n";
    for(const auto& p:points) if(p.point.allFinite() && std::abs(p.point.w())>1e-12)
        file<<p.id<<','<<p.point.x()/p.point.w()<<','<<p.point.y()/p.point.w()<<','<<p.point.z()/p.point.w()<<','<<p.quality<<','<<p.stateId<<'\n';
    return bool(file);
}
} }
