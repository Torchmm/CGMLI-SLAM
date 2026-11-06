#pragma once

#include <atomic>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <pipeline/StateOutput.hpp>
#include <backend/GlobalSlamGraph.hpp>
#include <frontend/Frontend.hpp>
#include <visualization/ViVisualizer.hpp>
#include <utils/ThreadsafeQueue.hpp>
#include <utils/Timer.hpp>

namespace cgmli {
/**
 * Native visual/inertial SLAM pipeline. One estimator writer stages real inputs,
 * solves/marginalizes atomically, then publishes an immutable successful result.
 * Historical loop closure/BA owns a separate graph and never overwrites the local
 * marginal prior. Configure output callbacks before the first processFrame().
 * Current-image extraction overlaps the preceding backend transaction using a
 * coherent accepted-state cache. Association/insertion waits for that writer.
 * processFrame, stopThreading and exports run on the same control thread.
 */
class ThreadedSlam : public ViInterface {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CGMLI_DEFINE_EXCEPTION(Exception,std::runtime_error)
    ThreadedSlam(ViParameters&,std::string vocabularyPath);
    ~ThreadedSlam() override;
    bool addImages(const Time&,const std::map<size_t,cv::Mat>&,
        const std::map<size_t,cv::Mat>& depthImages={}) override final;
    /// Synchronous approximation at referenceTime; preserve per-camera source
    /// timestamps for diagnostics and bearing compensation. The fixed image_delay is added once.
    bool addImagesWithTimestamps(const Time& referenceTime,const std::map<size_t,cv::Mat>&,
        const std::map<size_t,Time>& originalTimes,const std::map<size_t,cv::Mat>& depthImages={});
    bool addImuMeasurement(const Time&,const Eigen::Vector3d&,const Eigen::Vector3d&) override final;
    bool addLidarMeasurement(const Time&,const Eigen::Vector3d&) override final;
    bool addDepthMeasurement(const Time&,const cv::Mat&,const std::optional<cv::Mat>& = std::nullopt) override final;
    void setBlocking(bool value) override final { blocking_=value; }
    bool processFrame();
    /// Control-thread diagnostic for replay/drain loops; waiting for IMU still
    /// counts as pending input. A terminal fault quarantines that input while
    /// any previously launched worker still counts. Do not poll from an output worker.
    bool hasPendingWork() const { return optimizationThread_.joinable() || (!faulted_ && (staged_ || preparedFrame_ || !cameraQueue_.Empty())); }
    /// Input-consumption diagnostic for a single-thread replay producer. A frame
    /// already handed to the backend does not prevent feeding the next image/IMU.
    bool hasPendingImages() const { return !faulted_ && (preparedFrame_ || !cameraQueue_.Empty()); }
    /// A broken physical IMU interval cannot be repaired by retrying an optimizer.
    /// Stop this instance and explicitly initialize a new estimation segment.
    bool hasFatalError() const { return faulted_; }
    void display(std::map<std::string,cv::Mat>&) override final;
    void stopThreading();
    void setFinalTrajectoryCsvFile(const std::string& path,bool rpg=false) { trajectoryFile_=path; rpg_=rpg; }
    void setMapCsvFile(const std::string& path) { mapFile_=path; }
    void writeFinalTrajectoryCsv();
    void doFinalBa();
    bool saveMap();
    // LiDAR/depth submapping remains deferred and reports unavailable.
    bool exportSe2Map() { return false; }
    bool createFinalSe2Map() { return false; }
    bool needsNewLidarKeyframe() { return false; }
    void writeDebugStatisticsCsv(const std::string&);
    using AlignmentPublishCallback=std::function<void(const Time&,const kinematics::Transformation&,
        const std::vector<Eigen::Vector3f,Eigen::aligned_allocator<Eigen::Vector3f>>&,bool)>;
    void setAlignmentPublishCallback(const AlignmentPublishCallback& callback) { alignmentCallback_=callback; }
    std::string lastError() const;
private:
    // Standalone regression access permits deterministic failed-solve injection
    // without exposing a production callback that could mutate a running writer.
    friend struct ThreadedSlamRegressionAccess;
    struct Publication {
        State state{};
        TrackingState tracking{};
        std::shared_ptr<pipeline::StateHistory> states;
        std::shared_ptr<MapPointVector> points;
    };
    void initialize();
    bool collectImu(const Time&);
    bool predict(const MultiFrame&,State&);
    void extract(const MultiFramePtr&,const State&);
    void cacheFeatureDebug(const MultiFramePtr&);
    void prepareFeatureDebug(const MultiFramePtr&,const FrontendAssociationReport&);
    bool stage(const MultiFramePtr&,const State&);
    void launchOptimization();
    bool waitForOptimization();
    bool finishOptimization();
    void optimizeAndPublish();
    struct CameraCalibrationDecision {
        bool rotation=false, timeOffset=false, translation=false;
        double angularVariation=0, angularSecondAxis=0, accelerationVariation=0;
        double velocityVariation=0, translationSpan=0, speed=0, gateMs=0;
        size_t temporalTracks=0, timedTracks=0;
        const char* phase="CALIB_FIXED";
        const char* reason="disabled";
    };
    CameraCalibrationDecision cameraCalibrationDecision();
    static CameraCalibrationDecision calibrationRetryDecision(CameraCalibrationDecision,
        const std::vector<backend::BlockKey>& rejected);
    void recordCameraCalibration(const CameraCalibrationDecision&, const backend::StepReport&);
    std::string cameraCalibrationViolation(const backend::ParameterBlock&) const;
    bool validateSolution(const backend::BackendGraph&,std::string&) const;
    void rollbackPendingFrame();
    void publishStep(const backend::StepReport&);
    void prepareVisualization(const Publication&,const backend::BackendSnapshot&,const backend::SolveReport&);
    void publishingLoop();
    void visualizationLoop();
    void setError(std::string);
    void setFatalError(std::string);
    void updateGlobalHistory();
    void queuePublication(Publication);
    ViParameters parameters_;
    backend::BackendOptions options_;
    ViSlamBackend estimator_;
    Frontend frontend_;
    ViVisualizer visualizer_;
    std::unique_ptr<backend::GlobalSlamGraph> global_;
    std::atomic_bool blocking_{false},shutdown_{false},stopping_{false},faulted_{false};
    // Input callbacks may have independent producers. Keep diagnostic counters
    // per instance rather than using old glog macros with shared static counters.
    std::atomic<uint64_t> droppedCameraPackets_{0},droppedImuSamples_{0};
    uint64_t startupImageDrops_=0,imuCoverageWaits_=0; // Control-thread diagnostics only.
    std::thread optimizationThread_,publishingThread_,visualizationThread_;
    threadsafe::Queue<std::vector<CameraMeasurement>> cameraQueue_;
    threadsafe::Queue<ImuMeasurement,Eigen::aligned_allocator<ImuMeasurement>> imuQueue_;
    threadsafe::Queue<Publication> publicationQueue_;
    threadsafe::Queue<ViVisualizer::VisualizationData::Ptr> visualizationQueue_;
    threadsafe::Queue<std::map<std::string,cv::Mat>> visualizationImages_;
    threadsafe::Queue<cv::Mat> overheadImages_;
    ImuMeasurementDeque imuBuffer_;
    AlignedMap<StateId,ImuMeasurementDeque> imuByFrame_;
    AlignedMap<StateId,Eigen::Vector3d> gyroByFrame_;
    pipeline::StateHistory localHistory_,outputHistory_;
    std::map<uint64_t,MapPoint> pointHistory_;
    State latestLocal_{},previousLocal_{};
    // Control-thread cache: refresh pose and calibration together only after join.
    // Reading the live snapshot during extraction could mix two accepted revisions.
    std::shared_ptr<const backend::BackendSnapshot> extractionSnapshot_;
    // Control-thread thumbnail cache follows accepted window membership, plus the
    // pending/prepared images. Immutable snapshots remain alive for the visualizer.
    std::map<uint64_t,std::shared_ptr<const ViVisualizer::FeatureFrame>> featureDebugFrames_;
    std::shared_ptr<const ViVisualizer::FeatureDebugData> pendingFeatureDebug_;
    // One extracted next frame may wait here while the previous staged insertion
    // is retried. It is not yet visible to the estimator or the backend worker.
    MultiFramePtr preparedFrame_;
    State preparedPrediction_{};
    StateId preparedFromState_;
    // Worker input remains unchanged from launch until the corresponding join.
    MultiFramePtr pendingFrame_;
    ImuMeasurementDeque pendingImu_;
    FrontendAssociationReport pendingAssociation_;
    backend::SolveBudget pendingBudget_;
    // Exact pre-insertion checkpoint: includes numeric blocks, frozen prior and
    // managed observations. Rollback never tries to subtract a committed Schur prior.
    std::optional<ViSlamBackend> stagedCheckpoint_;
    backend::StateEstimate pendingPrediction_;
    double pendingPredictionSeconds_=0;
    size_t pendingVisualLandmarks_=0;
    int optimizationAttempts_=0;
    // Worker-owned stage progress. A failed solve cannot advance either stage.
    unsigned rotationCalibrationUpdates_=0, timeCalibrationUpdates_=0, trackingCalibrationUpdates_=0;
    std::optional<CameraCalibrationDecision> pendingCalibrationRetry_;
    std::string cameraCalibrationPhase_;
    bool accepted_=false,staged_=false,vocabularyLoaded_=false;
    bool optimizationRetryable_=true; // Worker writes; control thread reads only after join.
    std::exception_ptr optimizationError_;
    uint64_t nextId_=1;
    std::string vocabularyPath_,trajectoryFile_,mapFile_;
    bool rpg_=false;
    TrackingState latestTracking_{};
    mutable std::mutex errorMutex_;
    std::string error_;
    AlignmentPublishCallback alignmentCallback_;
};
}
