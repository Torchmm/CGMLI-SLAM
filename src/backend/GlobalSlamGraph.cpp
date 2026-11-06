#include "GlobalSlamGraph.hpp"
#include "BackendFactors.hpp"
#include "FrameStore.hpp"
#include <factor/BodyImuFactor.hpp>
#include <visual/MultiFrame.hpp>
#include <visual/features/OrbMatcher.hpp>
#include <visual/pose_solver/AbsolutePoseSolver.hpp>
#include <ceres/ceres.h>
#include <ceres/numeric_diff_cost_function.h>
#include <DBoW2/DBoW2.h>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iterator>

namespace cgmli { namespace backend {
namespace {
using PoseVector=Eigen::Matrix<double,7,1>;
PoseVector coordinates(const kinematics::Transformation& pose) {
    PoseVector p; p<<pose.r(),pose.q().coeffs(); return p;
}
kinematics::Transformation transform(const double* p) {
    return {Eigen::Vector3d(p[0],p[1],p[2]),Eigen::Quaterniond(p[6],p[3],p[4],p[5]).normalized()};
}
Eigen::Vector3d transformPoint(const kinematics::Transformation& pose,const Eigen::Vector3d& point) {
    // Transformation::operator*(Vector3d) rotates directions only. Map points
    // require the full affine action, including translation, in every frame.
    return pose.C()*point+pose.r();
}
class GravityPoseManifold final : public ::ceres::Manifold {
public:
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 4; }
    bool Plus(const double* x,const double* d,double* y) const override {
        const auto pose=transform(x);
        const kinematics::Transformation result(pose.r()+Eigen::Map<const Eigen::Vector3d>(d),expSO3(Eigen::Vector3d(0,0,d[3]))*pose.q());
        Eigen::Map<PoseVector> out(y); out=coordinates(result); return out.allFinite();
    }
    bool PlusJacobian(const double* x,double* data) const override {
        Eigen::Map<Eigen::Matrix<double,7,4,Eigen::RowMajor>> j(data); j.setZero(); j.topLeftCorner<3,3>().setIdentity();
        Eigen::Matrix<double,4,3,Eigen::RowMajor> q; LeftSO3Manifold().PlusJacobian(x+3,q.data()); j.col(3).tail<4>()=q.col(2); return true;
    }
    bool Minus(const double* y,const double* x,double* d) const override {
        const auto a=transform(x),b=transform(y); Eigen::Map<Eigen::Vector3d> p(d); p=b.r()-a.r(); d[3]=logSO3(b.q()*a.q().conjugate()).z(); return true;
    }
    bool MinusJacobian(const double* x,double* data) const override {
        Eigen::Map<Eigen::Matrix<double,4,7,Eigen::RowMajor>> j(data); j.setZero(); j.topLeftCorner<3,3>().setIdentity();
        Eigen::Matrix<double,3,4,Eigen::RowMajor> q; LeftSO3Manifold().MinusJacobian(x+3,q.data()); j.row(3).tail<4>()=q.row(2); return true;
    }
};
struct RelativeCost {
    kinematics::Transformation measured;
    Eigen::Matrix<double,6,6> S;
    bool operator()(const double* first,const double* second,double* residual) const {
        const auto prediction=coordinates(transform(first).inverse()*transform(second));
        const auto reference=coordinates(measured); Eigen::Matrix<double,6,1> error;
        if(!LeftSE3Manifold().Minus(prediction.data(),reference.data(),error.data())) return false;
        Eigen::Map<Eigen::Matrix<double,6,1>> r(residual); r=S*error; return r.allFinite();
    }
};
void selectLinearSolver(::ceres::Solver::Options& options,bool schur) {
    options.linear_solver_type=schur?::ceres::DENSE_SCHUR: ::ceres::DENSE_QR;
    if(::ceres::IsSparseLinearAlgebraLibraryTypeAvailable(::ceres::EIGEN_SPARSE)) {
        options.sparse_linear_algebra_library_type=::ceres::EIGEN_SPARSE;
        options.linear_solver_type=schur?::ceres::SPARSE_SCHUR: ::ceres::SPARSE_NORMAL_CHOLESKY;
    } else if(::ceres::IsSparseLinearAlgebraLibraryTypeAvailable(::ceres::SUITE_SPARSE)) {
        options.sparse_linear_algebra_library_type=::ceres::SUITE_SPARSE;
        options.linear_solver_type=schur?::ceres::SPARSE_SCHUR: ::ceres::SPARSE_NORMAL_CHOLESKY;
    }
}
struct BearingCost {
    WindowObservation observation;
    kinematics::Transformation T_RC;
    bool operator()(const double* pose,const double* point,const double* rig,double* residual) const {
        const auto camera=transform(pose)*transform(rig)*T_RC;
        const Eigen::Vector3d pointWorld=Eigen::Map<const Eigen::Vector3d>(point);
        const Eigen::Vector3d p=transformPoint(camera.inverse(),pointWorld);
        if(!p.allFinite() || p.norm()<1e-8) return false;
        const Eigen::Vector3d b=observation.bearing,y=p.normalized(); const double c=std::clamp(b.dot(y),-1.0,1.0),s=b.cross(y).norm();
        if(c<-1+1e-8) return false;
        const Eigen::Vector3d axis=std::abs(b.x())<0.8?Eigen::Vector3d::UnitX():Eigen::Vector3d::UnitY();
        Eigen::Matrix<double,3,2> basis; basis.col(0)=(axis-b*b.dot(axis)).normalized(); basis.col(1)=b.cross(basis.col(0));
        Eigen::Vector3d logarithm=y-c*b; if(s>1e-9) logarithm*=std::atan2(s,c)/s;
        Eigen::Map<Eigen::Vector2d> r(residual); r=observation.sqrtInformation*basis.transpose()*logarithm; return r.allFinite();
    }
};
std::map<int,double> words(const MultiFrame& frame) {
    std::map<int,double> result; size_t count=0;
    // Deterministic appearance proposals only. Hash collisions never constitute a
    // loop: mutual ORB matching and metric multi-camera geometry must verify it.
    for(size_t c=0;c<frame.numFrames();++c) for(int k=0;k<frame.descriptors(c).rows;++k) {
        const auto* p=frame.descriptors(c).ptr<uchar>(k);
        for(int bank=0;bank<4;++bank) { ++result[bank*256+p[bank*7]]; ++count; }
    }
    if(count) for(auto& item:result) item.second/=double(count); return result;
}
WindowObservation measurement(const MultiFrame& frame,const KeypointIdentifier& key,LandmarkId id,double scale) {
    WindowObservation o; o.key=key; o.landmark=id; const auto& point=frame.keypoints(key.cameraIndex)[key.keypointIndex];
    o.pixel={point.pt.x,point.pt.y}; Eigen::Matrix<double,3,2> derivative;
    if(!frame.geometry(key.cameraIndex)->backProject(o.pixel,&o.bearing,&derivative) || o.bearing.norm()<1e-8)
        throw std::invalid_argument("Invalid archived camera ray");
    const double norm=o.bearing.norm(); o.bearing/=norm;
    const Eigen::Vector3d axis=std::abs(o.bearing.x())<0.8?Eigen::Vector3d::UnitX():Eigen::Vector3d::UnitY();
    Eigen::Matrix<double,3,2> basis; basis.col(0)=(axis-o.bearing*o.bearing.dot(axis)).normalized(); basis.col(1)=o.bearing.cross(basis.col(0));
    const Eigen::Matrix2d j=basis.transpose()*(Eigen::Matrix3d::Identity()-o.bearing*o.bearing.transpose())*derivative/norm;
    const double sigma=std::pow(scale,point.octave); const Eigen::Matrix2d covariance=sigma*sigma*j*j.transpose();
    Eigen::LLT<Eigen::Matrix2d> llt(covariance); if(llt.info()!=Eigen::Success) throw std::invalid_argument("Singular archived bearing noise");
    o.sqrtInformation=llt.matrixL().solve(Eigen::Matrix2d::Identity()); return o;
}
}
class GlobalSlamGraph::PlaceIndex {
public:
    OrbDatabase database;
    std::vector<StateId> identities;
    std::string profile;
};
GlobalSlamGraph::GlobalSlamGraph(const ViParameters& p):parameters_(p) {}
GlobalSlamGraph::~GlobalSlamGraph()=default;
bool GlobalSlamGraph::loadVocabulary(const std::string& path,const std::string& profile,std::string& error) {
    if(!nodes_.empty()) { error="Load a global ORB vocabulary before collecting states"; return false; }
    try {
        cv::FileStorage file(path,cv::FileStorage::READ);
        if(!file.isOpened() || profile.empty() || static_cast<std::string>(file["descriptor_profile"])!=profile) {
            error="Global vocabulary descriptor_profile differs from the ORB sampling format"; return false;
        }
        const auto vocabularyNode=file["vocabulary"];
        if(vocabularyNode["nodes"].empty() || vocabularyNode["words"].empty()) { error="Empty global ORB vocabulary"; return false; }
        for(const auto& node:vocabularyNode["nodes"]) {
            std::istringstream stream(static_cast<std::string>(node["descriptor"])); int byte=0,count=0;
            while(stream>>byte) { if(byte<0 || byte>255) { error="Invalid ORB vocabulary byte"; return false; } ++count; }
            if(count!=32 || !stream.eof()) { error="ORB vocabulary nodes must contain 32 bytes"; return false; }
        }
        OrbVocabulary vocabulary; vocabulary.load(file); if(vocabulary.empty()) { error="Empty ORB words"; return false; }
        auto index=std::make_unique<PlaceIndex>(); index->profile=profile; index->database.setVocabulary(vocabulary,false,0);
        places_=std::move(index); return true;
    } catch(const std::exception& exception) { error=exception.what(); return false; }
}
uint64_t GlobalSlamGraph::canonical(uint64_t id) const {
    for(size_t depth=0;depth<=merged_.size();++depth) {
        const auto it=merged_.find(id); if(it==merged_.end() || it->second==id) return id; id=it->second;
    }
    throw std::logic_error("Cyclic global track aliases");
}
kinematics::Transformation GlobalSlamGraph::globalPose(StateId id) const {
    const auto& node=nodes_.at(id);
    if(bundleAdjusted_ || node.reference==id || !nodes_.count(node.reference)) return node.global;
    const auto& reference=nodes_.at(node.reference);
    return reference.global*reference.local.T_WB.inverse()*node.local.T_WB;
}
std::optional<GlobalSlamGraph::Loop> GlobalSlamGraph::findLoop(StateId current,const std::set<StateId>& active) const {
    const auto& query=nodes_.at(current); if(!query.frame || keyframes_.size()<16) return std::nullopt;
    std::vector<std::pair<double,StateId>> candidates;
    std::map<StateId,double> trainedScores;
    if(places_) {
        std::vector<cv::Mat> descriptors;
        for(size_t c=0;c<query.frame->numFrames();++c) for(int k=0;k<query.frame->descriptors(c).rows;++k) descriptors.push_back(query.frame->descriptors(c).row(k));
        DBoW2::QueryResults results; places_->database.query(descriptors,results,-1);
        for(const auto& result:results) if(result.Id<places_->identities.size()) trainedScores[places_->identities[result.Id]]=result.Score;
    }
    for(size_t i=0;i+15<keyframes_.size();++i) {
        const auto id=keyframes_[i]; const auto& old=nodes_.at(id);
        if(active.count(id) || !old.frame || (query.local.timestamp-old.local.timestamp).toSec()<5.0) continue;
        double score=0;
        if(places_) { const auto match=trainedScores.find(id); if(match!=trainedScores.end()) score=match->second; }
        else for(const auto& word:query.words) { const auto it=old.words.find(word.first); if(it!=old.words.end()) score+=std::min(word.second,it->second); }
        const double threshold=places_?parameters_.estimator.p_dbow:0.08;
        if(std::isfinite(score) && score>threshold) candidates.emplace_back(score,id);
    }
    std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.first>b.first;});
    if(candidates.size()>6) candidates.resize(6);
    for(const auto& candidate:candidates) {
        const auto oldId=candidate.second; const auto& old=nodes_.at(oldId);
        cv::Mat descriptors; std::vector<LandmarkId> identities;
        for(size_t c=0;c<old.frame->numFrames();++c) for(size_t k=0;k<old.frame->numKeypoints(c);++k) {
            const auto id=canonical(old.frame->landmarkId(c,k));
            if(!id || !points_.count(id)) continue;
            descriptors.push_back(old.frame->descriptors(c).row(int(k))); identities.emplace_back(id);
        }
        if(descriptors.rows<20) continue;
        struct Match { KeypointIdentifier key; LandmarkId point; float distance; };
        std::vector<Match> matches; features::OrbMatcherOptions matching;
        matching.maxHammingDistance=parameters_.frontend.orb_matching_threshold; features::OrbMatcher matcher(matching);
        for(size_t c=0;c<query.frame->numFrames();++c) {
            if(query.frame->descriptorProfile(c)!=old.frame->descriptorProfile(0)) continue;
            std::vector<cv::DMatch> found; matcher.matchFiltered(query.frame->descriptors(c),descriptors,found);
            for(const auto& m:found) matches.push_back({{current.value(),c,size_t(m.queryIdx)},identities.at(m.trainIdx),m.distance});
        }
        std::sort(matches.begin(),matches.end(),[](const auto& a,const auto& b){return a.distance<b.distance;});
        frontend_pose::AbsoluteInput input; std::set<LandmarkId> used;
        for(const auto& m:matches) if(used.insert(m.point).second) {
            const auto& point=points_.at(m.point.value());
            pose_solver::AbsolutePoseCorrespondence data;
            data.point_W=transformPoint(globalPose(oldId).inverse(),transformPoint(globalPose(point.anchor),point.position));
            if(!frontend_pose::makeRay(*query.frame,m.key,query.extrinsics,data.ray_S)) continue;
            input.data.push_back(data); input.observations.push_back({m.key,m.point});
        }
        if(input.data.size()<20) continue;
        pose_solver::PoseSolverOptions options; options.minInliers=20; options.angularThreshold=0.01;
        const auto seed=globalPose(oldId).inverse()*globalPose(current);
        const auto result=pose_solver::AbsolutePoseSolver(options).solve(input.data,&seed);
        if(!result.success() || result.inlierRatio<0.5) continue;
        std::set<std::pair<size_t,int>> coverage;
        for(auto index:result.inliers) {
            const auto key=input.observations.at(index).key; const auto p=query.frame->keypoints(key.cameraIndex)[key.keypointIndex].pt;
            const auto model=query.frame->geometry(key.cameraIndex);
            coverage.emplace(key.cameraIndex,int(p.y*6/model->imageHeight())*8+int(p.x*8/model->imageWidth()));
        }
        if(coverage.size()<6) continue;
        if(parameters_.imu.use) {
            const auto recovered=globalPose(oldId)*result.pose;
            if((recovered.C().transpose()*Eigen::Vector3d::UnitZ()-globalPose(current).C().transpose()*Eigen::Vector3d::UnitZ()).norm()>0.2) continue;
        }
        Eigen::Matrix<double,6,6> information;
        if(!frontend_pose::absoluteInformation(*query.frame,query.extrinsics,input,result,1.0,
            parameters_.frontend.orb_scale_factor,information)) continue;
        // The map is uncertain; reduce the conditional fixed-map information instead
        // of treating it as an exact covariance of a joint historical estimate.
        Eigen::LLT<Eigen::Matrix<double,6,6>> llt(0.1*information); if(llt.info()!=Eigen::Success) continue;
        Loop loop; loop.edge.first=oldId; loop.edge.second=current; loop.edge.relative=result.pose;
        loop.edge.sqrtInformation=llt.matrixU(); loop.edge.loop=true; loop.score=candidate.first;
        for(auto index:result.inliers) loop.associations.emplace_back(input.observations.at(index).key,input.observations.at(index).landmark);
        return loop;
    }
    return std::nullopt;
}
bool GlobalSlamGraph::optimizePoseGraph(std::string& error) {
    std::map<StateId,PoseVector> poses; ::ceres::Problem problem;
    for(auto id:keyframes_) {
        poses.emplace(id,coordinates(nodes_.at(id).global)); auto* value=poses.at(id).data();
        problem.AddParameterBlock(value,7,parameters_.imu.use?static_cast<::ceres::Manifold*>(new GravityPoseManifold):new LeftSE3Manifold);
    }
    if(poses.empty()) return true;
    problem.SetParameterBlockConstant(poses.begin()->second.data());
    for(const auto& edge:edges_) {
        using Cost=::ceres::NumericDiffCostFunction<RelativeCost,::ceres::CENTRAL,6,7,7>;
        problem.AddResidualBlock(new Cost(new RelativeCost{edge.relative,edge.sqrtInformation}),edge.loop?new ::ceres::HuberLoss(2.5):nullptr,
            poses.at(edge.first).data(),poses.at(edge.second).data());
    }
    ::ceres::Solver::Options options; options.max_num_iterations=parameters_.estimator.full_graph_iterations;
    options.num_threads=parameters_.estimator.full_graph_num_threads; selectLinearSolver(options,false);
    ::ceres::Solver::Summary summary; ::ceres::Solve(options,&problem,&summary);
    if(!summary.IsSolutionUsable() || !std::isfinite(summary.final_cost) || summary.final_cost>summary.initial_cost+1e-8*std::max(1.0,summary.initial_cost)) {
        error="Global pose graph rejected: "+summary.BriefReport(); return false;
    }
    for(const auto& pose:poses) if(!pose.second.allFinite()) { error="Non-finite global pose"; return false; }
    for(const auto& pose:poses) nodes_.at(pose.first).global=transform(pose.second.data()); return true;
}
bool GlobalSlamGraph::ingest(const StepReport& step,std::shared_ptr<const MultiFrame> frame,const ImuMeasurementDeque& imu,
    const std::vector<kinematics::Transformation>& extrinsics,bool visualConsensus,std::string& error) {
    if(!step.accepted || !step.snapshot || !frame) { error="Global archive requires an accepted frame"; return false; }
    lastLoopClosed_=false; const StateId current(frame->id()); std::set<StateId> active;
    const auto update=[&](const StateEstimate& state) {
        auto it=nodes_.find(state.id);
        if(it==nodes_.end()) {
            Node node; node.local=state; node.global=state.T_WB;
            node.reference=state.keyframe || keyframes_.empty()?state.id:keyframes_.back();
            if(!keyframes_.empty()) {
                const auto reference=keyframes_.back();
                node.global=globalPose(reference)*nodes_.at(reference).local.T_WB.inverse()*state.T_WB;
            }
            nodes_.emplace(state.id,std::move(node));
        } else {
            const auto correction=globalPose(state.id)*it->second.local.T_WB.inverse();
            it->second.local=state; it->second.global=correction*state.T_WB;
        }
    };
    if(step.retiredState) update(*step.retiredState);
    for(const auto& state:step.snapshot->states) { active.insert(state.id); update(state); }
    auto& node=nodes_.at(current);
    if(node.local.keyframe) {
        auto owned=FrameStore::copyMeasurements(*frame,false);
        for(size_t c=0;c<frame->numFrames();++c) for(size_t k=0;k<frame->numKeypoints(c);++k)
            owned->setLandmarkId(c,k,frame->landmarkId(c,k));
        node.frame=owned; node.words=words(*frame);
    }
    node.imu=imu; node.extrinsics=extrinsics;
    const auto rig=step.snapshot->calibration.find({BlockKind::CameraExtrinsic,0});
    if(rig!=step.snapshot->calibration.end()) rigExtrinsic_.assign(rig->second.data(),rig->second.data()+rig->second.size());
    if(node.local.keyframe && std::find(keyframes_.begin(),keyframes_.end(),current)==keyframes_.end()) {
        if(places_) {
            std::vector<cv::Mat> descriptors;
            for(size_t c=0;c<frame->numFrames();++c) for(int k=0;k<frame->descriptors(c).rows;++k) descriptors.push_back(frame->descriptors(c).row(k));
            if(!descriptors.empty()) {
                const auto entry=places_->database.add(descriptors);
                if(entry!=places_->identities.size()) throw std::logic_error("Global vocabulary entry identity mismatch");
                places_->identities.push_back(current);
            }
        }
        if(!keyframes_.empty()) {
            const auto previous=keyframes_.back(); Edge edge; edge.first=previous; edge.second=current;
            edge.relative=nodes_.at(previous).local.T_WB.inverse()*node.local.T_WB;
            // Explicit conservative odometry model for the pose graph, in metres
            // and radians. This is not advertised as a recovered joint covariance.
            edge.sqrtInformation.diagonal()<<10,10,10,20,20,20; edges_.push_back(edge);
        }
        node.reference=current; keyframes_.push_back(current);
    }
    for(auto& edge:edges_) if(!edge.loop && active.count(edge.first) && active.count(edge.second))
        edge.relative=nodes_.at(edge.first).local.T_WB.inverse()*nodes_.at(edge.second).local.T_WB;
    for(const auto& point:step.optimizedVisualHistory) {
        if(point.pointWorld && nodes_.count(StateId(point.metadata.anchor.frameId)) && canonical(point.metadata.id.value())==point.metadata.id.value()) {
            Point p; p.anchor=StateId(point.metadata.anchor.frameId); p.quality=point.metadata.quality;
            p.position=transformPoint(nodes_.at(p.anchor).local.T_WB.inverse(),*point.pointWorld); points_[point.metadata.id.value()]=p;
        }
        for(const auto& o:point.observations) observations_[o.key]=o;
    }
    for(const auto& key:step.rejectedVisualObservations) observations_.erase(key);
    if(!parameters_.estimator.do_loop_closures || !node.local.keyframe || !visualConsensus) return true;
    const auto loop=findLoop(current,active);
    if(!loop) { lastCandidate_=StateId(); candidateStreak_=0; return true; }
    const auto previousCandidate=std::find(keyframes_.begin(),keyframes_.end(),lastCandidate_);
    const auto thisCandidate=std::find(keyframes_.begin(),keyframes_.end(),loop->edge.first);
    const bool consistent=previousCandidate!=keyframes_.end() && thisCandidate!=keyframes_.end()
        && std::abs(std::distance(keyframes_.begin(),previousCandidate)-std::distance(keyframes_.begin(),thisCandidate))<=4;
    candidateStreak_=consistent?candidateStreak_+1:1; lastCandidate_=loop->edge.first;
    if(candidateStreak_<2) return true;
    edges_.push_back(loop->edge);
    if(!optimizePoseGraph(error)) { edges_.pop_back(); return false; }
    for(const auto& association:loop->associations) {
        const auto key=association.first; const auto target=canonical(association.second.value());
        const auto existing=observations_.find(key);
        if(existing!=observations_.end()) {
            const auto original=canonical(existing->second.landmark.value());
            if(original && original!=target) merged_[original]=target;
        } else observations_[key]=measurement(*frame,key,LandmarkId(target),parameters_.frontend.orb_scale_factor);
    }
    lastLoopClosed_=true; candidateStreak_=0; return true;
}
std::vector<StateEstimate> GlobalSlamGraph::estimates() const {
    std::vector<StateEstimate> result; result.reserve(nodes_.size());
    for(const auto& item:nodes_) {
        auto state=item.second.local; const auto correction=globalPose(item.first)*state.T_WB.inverse();
        state.T_WB=globalPose(item.first); state.speedBias.head<3>()=correction.C()*state.speedBias.head<3>();
        const auto sb=bundleSpeedBias_.find(item.first); if(sb!=bundleSpeedBias_.end()) state.speedBias=sb->second;
        result.push_back(state);
    }
    return result;
}
StateEstimate GlobalSlamGraph::estimate(StateId id) const {
    auto state=nodes_.at(id).local; const auto correction=globalPose(id)*state.T_WB.inverse();
    state.T_WB=globalPose(id); state.speedBias.head<3>()=correction.C()*state.speedBias.head<3>();
    const auto sb=bundleSpeedBias_.find(id); if(sb!=bundleSpeedBias_.end()) state.speedBias=sb->second; return state;
}
Eigen::Vector3d GlobalSlamGraph::correctPoint(StateId anchor,const Eigen::Vector3d& point) const {
    return transformPoint(globalPose(anchor),transformPoint(nodes_.at(anchor).local.T_WB.inverse(),point));
}
MapPointVector GlobalSlamGraph::mapPoints() const {
    MapPointVector result;
    for(const auto& item:points_) if(canonical(item.first)==item.first && nodes_.count(item.second.anchor)) {
        const auto world=transformPoint(globalPose(item.second.anchor),item.second.position);
        result.emplace_back(item.first,Eigen::Vector4d(world.x(),world.y(),world.z(),1),item.second.quality,item.second.anchor.value());
    }
    return result;
}
std::vector<kinematics::Transformation> GlobalSlamGraph::cameraExtrinsics() const {
    if(rigExtrinsic_.size()!=7) throw std::logic_error("Global archive has no rig calibration");
    const auto rig=transform(rigExtrinsic_.data()); std::vector<kinematics::Transformation> result;
    for(size_t c=0;c<parameters_.nCameraSystem.numCameras();++c)
        result.push_back(rig*parameters_.nCameraSystem.T_SC(0)->inverse()*(*parameters_.nCameraSystem.T_SC(c)));
    return result;
}
bool GlobalSlamGraph::bundleAdjust(int iterations,int threads,double sigmaTranslation,double sigmaRotation,std::string& error) {
    if(iterations<1 || threads<1 || !std::isfinite(sigmaTranslation) || !std::isfinite(sigmaRotation)
        || !((sigmaTranslation==0 && sigmaRotation==0) || (sigmaTranslation>0 && sigmaRotation>0))) {
        error="Invalid final BA controls or calibration prior sigmas"; return false;
    }
    if(nodes_.empty() || rigExtrinsic_.size()!=7) { error="No historical measurements for BA"; return false; }
    std::map<StateId,PoseVector> poses; std::map<StateId,SpeedAndBias> speedBias;
    for(const auto& state:estimates()) { poses.emplace(state.id,coordinates(state.T_WB)); speedBias.emplace(state.id,state.speedBias); }
    std::map<uint64_t,Eigen::Vector3d> points; std::map<uint64_t,std::set<std::pair<StateId,size_t>>> views;
    for(const auto& item:observations_) {
        const auto root=canonical(item.second.landmark.value()); const StateId frame(item.first.frameId);
        if(points_.count(root) && poses.count(frame)) views[root].emplace(frame,item.first.cameraIndex);
    }
    for(const auto& item:points_) if(canonical(item.first)==item.first && views[item.first].size()>=2)
        points.emplace(item.first,transformPoint(globalPose(item.second.anchor),item.second.position));
    if(points.empty()) { error="No triangulated multi-view observations for BA"; return false; }
    ::ceres::Problem::Options ownership; ownership.cost_function_ownership=::ceres::DO_NOT_TAKE_OWNERSHIP;
    auto rig=rigExtrinsic_; std::vector<std::shared_ptr<::ceres::CostFunction>> costs;
    ::ceres::Problem problem(ownership); problem.AddParameterBlock(rig.data(),7,new LeftSE3Manifold);
    if(sigmaTranslation<=0 || sigmaRotation<=0) problem.SetParameterBlockConstant(rig.data());
    else {
        const auto ref=ParameterBlock::pose({BlockKind::CameraExtrinsic,0},transform(rig.data()));
        Eigen::Matrix<double,6,6> S=Eigen::Matrix<double,6,6>::Identity(); S.topLeftCorner<3,3>()/=sigmaTranslation; S.bottomRightCorner<3,3>()/=sigmaRotation;
        auto prior=makeParameterPrior(ref,S,FactorKind::Calibration); costs.push_back(prior.cost); problem.AddResidualBlock(prior.cost.get(),nullptr,rig.data());
    }
    for(auto& pose:poses) problem.AddParameterBlock(pose.second.data(),7,new LeftSE3Manifold);
    for(auto& point:points) problem.AddParameterBlock(point.second.data(),3);
    std::map<StateId,size_t> observationCounts; size_t residualCount=0;
    struct VisualInput { uint64_t point; StateId state; BearingCost measurement; };
    std::vector<VisualInput> acceptedViews;
    std::map<uint64_t,std::set<std::pair<StateId,size_t>>> actualViews;
    for(const auto& item:observations_) {
        const auto& o=item.second; const auto root=canonical(o.landmark.value()); const StateId id(o.key.frameId);
        if(!points.count(root) || !poses.count(id)) continue;
        if(o.key.cameraIndex>=parameters_.nCameraSystem.numCameras()) continue;
        const auto baseline=parameters_.nCameraSystem.T_SC(0)->inverse()*(*parameters_.nCameraSystem.T_SC(o.key.cameraIndex));
        const BearingCost measurement{o,baseline}; Eigen::Vector2d initial;
        if(!measurement(poses.at(id).data(),points.at(root).data(),rig.data(),initial.data()) || initial.norm()>15) continue;
        // Track aliases created by a loop may overlap. Keep one raw observation
        // per point/camera/epoch, and require two actual usable views for free 3D.
        if(!actualViews[root].emplace(id,o.key.cameraIndex).second) continue;
        acceptedViews.push_back({root,id,measurement});
    }
    for(const auto& input:acceptedViews) {
        if(actualViews[input.point].size()<2) continue;
        using Cost=::ceres::NumericDiffCostFunction<BearingCost,::ceres::CENTRAL,2,7,3,7>;
        auto cost=std::make_shared<Cost>(new BearingCost(input.measurement)); costs.push_back(cost);
        problem.AddResidualBlock(cost.get(),new ::ceres::HuberLoss(2.5),poses.at(input.state).data(),points.at(input.point).data(),rig.data());
        ++observationCounts[input.state]; ++residualCount;
    }
    if(residualCount<10) { error="Too few usable historical bearing factors"; return false; }
    if(parameters_.imu.use) {
        for(auto& sb:speedBias) problem.AddParameterBlock(sb.second.data(),9);
        for(auto it=std::next(nodes_.begin());it!=nodes_.end();++it) {
            const auto previous=std::prev(it); auto a=previous->second.local,b=it->second.local;
            a.T_WB=transform(poses.at(a.id).data()); a.speedBias=speedBias.at(a.id);
            b.T_WB=transform(poses.at(b.id).data()); b.speedBias=speedBias.at(b.id);
            try {
                const auto cost=factor::makeBodyImuFactor(it->second.imu,parameters_.imu,a,b); costs.push_back(cost);
                problem.AddResidualBlock(cost.get(),nullptr,poses.at(a.id).data(),speedBias.at(a.id).data(),poses.at(b.id).data(),speedBias.at(b.id).data());
            } catch(const std::exception& exception) { error=exception.what(); return false; }
        }
        const auto id=nodes_.begin()->first;
        const auto ref=ParameterBlock::euclidean(BlockKey::speedBias(id),nodes_.begin()->second.local.speedBias,id);
        Eigen::Matrix<double,9,9> S=Eigen::Matrix<double,9,9>::Zero();
        S.diagonal().head<3>().setConstant(20); S.diagonal().segment<3>(3).setConstant(1/std::max(1e-4,parameters_.imu.sigma_bg));
        S.diagonal().tail<3>().setConstant(1/std::max(1e-3,parameters_.imu.sigma_ba));
        const auto prior=makeParameterPrior(ref,S); costs.push_back(prior.cost); problem.AddResidualBlock(prior.cost.get(),nullptr,speedBias.at(id).data());
    } else for(auto& pose:poses) if(!observationCounts[pose.first]) problem.SetParameterBlockConstant(pose.second.data());
    if(parameters_.imu.use) {
        const auto id=poses.begin()->first; const auto ref=ParameterBlock::pose(BlockKey::pose(id),transform(poses.begin()->second.data()),id);
        Eigen::Matrix<double,4,6> S=Eigen::Matrix<double,4,6>::Zero(); S.topLeftCorner<3,3>()=Eigen::Matrix3d::Identity()*1e4; S(3,5)=1e4;
        const auto prior=makeParameterPrior(ref,S); costs.push_back(prior.cost); problem.AddResidualBlock(prior.cost.get(),nullptr,poses.begin()->second.data());
    } else problem.SetParameterBlockConstant(poses.begin()->second.data());
    ::ceres::Solver::Options options; options.max_num_iterations=iterations; options.num_threads=threads;
    selectLinearSolver(options,true);
    options.linear_solver_ordering=std::make_shared<::ceres::ParameterBlockOrdering>();
    for(auto& point:points) options.linear_solver_ordering->AddElementToGroup(point.second.data(),0);
    for(auto& pose:poses) options.linear_solver_ordering->AddElementToGroup(pose.second.data(),1);
    if(parameters_.imu.use) for(auto& sb:speedBias) options.linear_solver_ordering->AddElementToGroup(sb.second.data(),1);
    options.linear_solver_ordering->AddElementToGroup(rig.data(),1);
    ::ceres::Solver::Summary summary; ::ceres::Solve(options,&problem,&summary);
    if(!summary.IsSolutionUsable() || !std::isfinite(summary.final_cost) || summary.final_cost>summary.initial_cost+1e-8*std::max(1.0,summary.initial_cost)) {
        error="Historical BA rejected: "+summary.BriefReport(); return false;
    }
    for(const auto& pose:poses) if(!pose.second.allFinite()) { error="Non-finite BA pose"; return false; }
    for(const auto& point:points) if(!point.second.allFinite()) { error="Non-finite BA point"; return false; }
    for(const auto& sb:speedBias) if(!sb.second.allFinite()) { error="Non-finite BA speed/bias"; return false; }
    for(const auto& pose:poses) nodes_.at(pose.first).global=transform(pose.second.data());
    bundleAdjusted_=true; if(parameters_.imu.use) bundleSpeedBias_=std::move(speedBias);
    for(const auto& point:points) points_.at(point.first).position=transformPoint(globalPose(points_.at(point.first).anchor).inverse(),point.second);
    rigExtrinsic_=std::move(rig); return true;
}
} }
