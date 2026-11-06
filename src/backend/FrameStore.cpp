#include "FrameStore.hpp"
#include "WindowVisualFactor.hpp"
#include <visual/MultiFrame.hpp>
#include <cameras/CameraBase.hpp>
#include <Eigen/Cholesky>
#include <cmath>

namespace cgmli { namespace backend {
const WindowFrameData* FrameStore::find(StateId id) const {
    auto it=records_.find(id); return it==records_.end()?nullptr:&it->second;
}
std::shared_ptr<const MultiFrame> FrameStore::measurements(StateId id) const {
    auto record=find(id); return record?record->measurements:nullptr;
}
bool FrameStore::insert(const EpochNode& epoch) {
    if (epochs_.count(epoch.id)) return false;
    epochs_.emplace(epoch.id,epoch); records_.emplace(epoch.id,WindowFrameData()); return true;
}
void FrameStore::erase(StateId id) { records_.erase(id); epochs_.erase(id); }
std::shared_ptr<MultiFrame> FrameStore::copyMeasurements(const MultiFrame& source,bool keepImages) {
    auto result=std::make_shared<MultiFrame>(source.cameraSystem(),source.timestamp(),source.id());
    for (size_t c=0;c<source.numFrames();++c) {
        result->setGeometry(c,source.geometry(c));
        result->resetKeypoints(c,source.keypoints(c));
        result->resetDescriptors(c,source.descriptors(c).clone());
        result->setDescriptorProfile(c,source.descriptorProfile(c));
        result->shareBackProjections(c,source);
        if (const auto& timing=source.imageTiming(c))
            result->setImageTiming(c,timing->originalTimestamp,timing->offsetNs);
        if (keepImages) result->setImage(c,source.image(c).clone());
    }
    return result;
}
bool FrameStore::attach(const MultiFrame& frame,bool keepImages,std::string& error) {
    const StateId id(frame.id()); auto epoch=epochs_.find(id);
    if (epoch==epochs_.end() || epoch->second.timestamp.toNSec()!=frame.timestamp().toNSec()) {
        error="frame ID/time does not match an active epoch"; return false;
    }
    if (records_.at(id).measurements || !frame.numFrames()
        || frame.numFrames()!=frame.cameraSystem().numCameras() || !frame.T_SC(0)
        || !ParameterBlock::pose({BlockKind::CameraExtrinsic,0},*frame.T_SC(0)).valid()) {
        error="frame already attached or invalid camera rig"; return false;
    }
    WindowFrameData candidate=records_.at(id);
    const auto T_RS=frame.T_SC(0)->inverse();
    for(size_t c=0;c<frame.numFrames();++c) {
        if (!frame.geometry(c) || frame.geometry(c).get()!=frame.cameraSystem().cameraGeometry(c).get() || !frame.T_SC(c)
            || !ParameterBlock::pose({BlockKind::CameraExtrinsic,0},*frame.T_SC(c)).valid()) {
            error="invalid frame camera calibration"; return false;
        }
        const auto& descriptors=frame.descriptors(c);
        if (!descriptors.empty() && (descriptors.rows!=static_cast<int>(frame.keypoints(c).size())
            || descriptors.type()!=CV_8UC1)) { error="descriptor/keypoint layout mismatch"; return false; }
        for(const auto& kp:frame.keypoints(c)) if (!std::isfinite(kp.pt.x) || !std::isfinite(kp.pt.y)) {
            error="nonfinite keypoint"; return false;
        }
        candidate.T_RC.push_back(T_RS*(*frame.T_SC(c)));
    }
    // Factory baselines and camera models must remain fixed within one active rig.
    // A common online T_BR update is allowed because it cancels from T_RC.
    if(!rigBaselines_.empty()) {
        if(rigBaselines_.size()!=candidate.T_RC.size()) { error="camera count changed inside window"; return false; }
        for(size_t c=0;c<candidate.T_RC.size();++c) if(
            rigModels_[c].get()!=frame.geometry(c).get()
            || (rigBaselines_[c].T()-candidate.T_RC[c].T()).norm()>1e-8) {
            error="camera model or fixed rig baseline changed inside window"; return false;
        }
    }
    candidate.measurements=copyMeasurements(frame,keepImages);
    if(rigBaselines_.empty()) {
        rigBaselines_=candidate.T_RC;
        for(size_t c=0;c<candidate.T_RC.size();++c) rigModels_.push_back(frame.geometry(c));
    }
    records_.at(id)=std::move(candidate); return true;
}
bool FrameStore::attachImu(StateId id,const ImuMeasurementDeque& samples,size_t limit,std::string& error) {
    if (!records_.count(id) || records_.at(id).imuSamples || samples.size()>limit) {
        error="invalid/duplicate IMU attachment or sample budget exceeded"; return false;
    }
    uint64_t last=0; bool first=true;
    for(const auto& sample:samples) {
        if ((!first && sample.timeStamp.toNSec()<=last) || !sample.measurement.gyroscopes.allFinite()
            || !sample.measurement.accelerometers.allFinite()) { error="invalid IMU sample sequence"; return false; }
        last=sample.timeStamp.toNSec(); first=false;
    }
    records_.at(id).imuSamples=std::make_shared<const ImuMeasurementDeque>(samples); return true;
}
bool FrameStore::contains(const KeypointIdentifier& key) const {
    const auto record=find(StateId(key.frameId));
    const auto* frame=record?record->measurements.get():nullptr;
    return frame && key.cameraIndex<frame->numFrames() && key.keypointIndex<frame->numKeypoints(key.cameraIndex);
}
bool FrameStore::cameraPose(const BackendGraph& graph,StateId id,size_t camera,
    kinematics::Transformation& T_WC,std::string& error) const {
    auto record=find(id); auto pose=graph.blocks.find(BlockKey::pose(id));
    auto extrinsic=graph.blocks.find({BlockKind::CameraExtrinsic,0});
    if(!record || camera>=record->T_RC.size() || pose==graph.blocks.end() || extrinsic==graph.blocks.end()) {
        error="camera pose requires active frame, body pose and global T_BR"; return false;
    }
    T_WC=blockPose(pose->second)*blockPose(extrinsic->second)*record->T_RC[camera]; return true;
}
bool FrameStore::observation(const KeypointIdentifier& key,double sigma,WindowObservation& out,std::string& error) const {
    if(!contains(key) || !std::isfinite(sigma) || sigma<=0) { error="invalid observation key/pixel sigma"; return false; }
    auto frame=measurements(StateId(key.frameId)); auto camera=frame->geometry(key.cameraIndex);
    const auto& kp=frame->keypoints(key.cameraIndex)[key.keypointIndex];
    WindowObservation candidate; candidate.key=key; candidate.pixel=Eigen::Vector2d(kp.pt.x,kp.pt.y);
    if(candidate.pixel.x()<0 || candidate.pixel.y()<0 || candidate.pixel.x()>=camera->imageWidth()
        || candidate.pixel.y()>=camera->imageHeight() || (camera->hasMask()
        && camera->mask().at<unsigned char>(static_cast<int>(kp.pt.y),static_cast<int>(kp.pt.x))==0)) {
        error="observation is outside valid camera image"; return false;
    }
    Eigen::Vector3d ray; Eigen::Matrix<double,3,2> J;
    if(!camera->backProject(candidate.pixel,&ray,&J) || !ray.allFinite() || !J.allFinite() || ray.norm()<1e-12) {
        error="camera back-projection/Jacobian failed"; return false;
    }
    candidate.bearing=ray.normalized();
    const Eigen::Matrix<double,3,2> D=(Eigen::Matrix3d::Identity()-candidate.bearing*candidate.bearing.transpose())*J/ray.norm();
    const Eigen::Matrix<double,2,3> B=sphereBasis(candidate.bearing).transpose();
    const Eigen::Matrix2d covariance=sigma*sigma*(B*D)*(B*D).transpose();
    Eigen::LLT<Eigen::Matrix2d> llt(covariance);
    if(llt.info()!=Eigen::Success) { error="singular bearing covariance"; return false; }
    candidate.sqrtInformation=llt.matrixL().solve(Eigen::Matrix2d::Identity());
    if(!candidate.sqrtInformation.allFinite()) { error="invalid bearing whitening"; return false; }
    out=candidate; return true;
}
} }
