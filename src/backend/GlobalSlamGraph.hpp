#pragma once
#include "ViSlamBackend.hpp"
#include <frontend/FrontendPoseAdapter.hpp>
#include <array>

namespace cgmli { namespace backend {
/**
 * Separate historical archive. It never mutates the marginalized local graph.
 * Loop proposals use ORB appearance, archived 3D points and generalized GP3P;
 * only old keyframes outside the local window are eligible. Accepted loops drive
 * an independent pose graph. Offline BA rebuilds original visual/IMU measurements,
 * so marginalized local priors and pose-graph odometry are not counted again.
 * All methods run on the pipeline's single estimator writer.
 */
class GlobalSlamGraph {
public:
    explicit GlobalSlamGraph(const ViParameters&);
    ~GlobalSlamGraph();
    bool loadVocabulary(const std::string& path,const std::string& descriptorProfile,std::string& error);
    bool ingest(const StepReport&,std::shared_ptr<const MultiFrame>,const ImuMeasurementDeque&,
        const std::vector<kinematics::Transformation>& T_BC,bool visualConsensus,std::string& error);
    bool bundleAdjust(int iterations,int threads,double sigmaTranslation,double sigmaRotation,std::string& error);
    std::vector<StateEstimate> estimates() const;
    StateEstimate estimate(StateId) const;
    Eigen::Vector3d correctPoint(StateId anchor,const Eigen::Vector3d& localWorldPoint) const;
    MapPointVector mapPoints() const;
    std::vector<kinematics::Transformation> cameraExtrinsics() const;
    bool lastLoopClosed() const { return lastLoopClosed_; }
private:
    class PlaceIndex;
    std::unique_ptr<PlaceIndex> places_;
    struct Node {
        StateEstimate local;
        kinematics::Transformation global;
        StateId reference;
        std::shared_ptr<const MultiFrame> frame;
        ImuMeasurementDeque imu;
        std::vector<kinematics::Transformation> extrinsics;
        std::map<int,double> words;
    };
    struct Point { StateId anchor; Eigen::Vector3d position=Eigen::Vector3d::Zero(); double quality=0; };
    struct Edge {
        StateId first,second;
        kinematics::Transformation relative;
        Eigen::Matrix<double,6,6> sqrtInformation=Eigen::Matrix<double,6,6>::Identity();
        bool loop=false;
    };
    struct Loop {
        Edge edge;
        std::vector<std::pair<KeypointIdentifier,LandmarkId>> associations;
        double score=0;
    };
    kinematics::Transformation globalPose(StateId) const;
    uint64_t canonical(uint64_t) const;
    std::optional<Loop> findLoop(StateId,const std::set<StateId>& active) const;
    bool optimizePoseGraph(std::string& error);
    ViParameters parameters_;
    std::map<StateId,Node> nodes_;
    std::vector<StateId> keyframes_;
    std::map<uint64_t,Point> points_;
    std::map<KeypointIdentifier,WindowObservation> observations_;
    std::map<uint64_t,uint64_t> merged_;
    std::vector<Edge> edges_;
    std::vector<double> rigExtrinsic_;
    StateId lastCandidate_;
    int candidateStreak_=0;
    bool lastLoopClosed_=false;
    bool bundleAdjusted_=false;
    std::map<StateId,SpeedAndBias> bundleSpeedBias_;
};
} }
