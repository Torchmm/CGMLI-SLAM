#pragma once

#include <atomic>
#include <functional>
#include <utility>
#include <frontend/OrbFrontendFeatures.hpp>
#include <frontend/FrontendAssociation.hpp>
#include <common/ViFrontendInterface.hpp>
#include <utils/assert_macros.hpp>

namespace cgmli {

    /// A verified local-window candidate, not an inserted loop factor. H is conditional
    /// fixed-map information in LEFT [rho,theta] coordinates of T_Bold_Bnew.
    struct FrontendLoopCandidate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            StateId oldState, currentState;
        kinematics::Transformation T_Bold_Bnew;
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    };
    struct FrontendAssociationReport {
        bool committed = false, trackingAccepted = false, metricPose = false;
        size_t newTracks = 0, newObservations = 0, initializedTracks = 0, inliers = 0;
        std::string message;
        std::vector<FrontendLoopCandidate> localLoopCandidates;
        std::vector<FeatureMatchDebug> debugMatches;
        frontend_association::KeyframeDecision keyframe;
    };

    /**
     * ORB frontend for the managed ViSlamBackend. All association methods run on its
     * single writer thread. Per-camera feature extraction may run concurrently.
     *
     * Contract: initializeRig -> extract all cameras -> addStates(frame,prediction,imu)
     * -> dataAssociationAndInitialization -> supply other factors -> optimize.
     * A successful association stages measurements; it never solves, marginalizes or
     * publishes a state. Inspect lastAssociationReport() for visual tracking quality.
     */
    class Frontend : public ViFrontendInterface {
    public:
        CGMLI_DEFINE_EXCEPTION(Exception, std::runtime_error)
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            explicit Frontend(size_t numCameras, const features::OrbExtractorOptions& options = {});
        ~Frontend() override;
        bool detectAndDescribe(size_t cameraIndex, std::shared_ptr<MultiFrame> frameOut,
            const kinematics::Transformation& T_WC, const std::vector<cv::KeyPoint>* keypoints) override final;
        /// Feature indices/descriptors must equal the frozen addStates payload. On any
        /// rejected plan the backend and caller associations remain unchanged. Return
        /// true means committed ingestion, including bootstrap/low-parallax seed tracks;
        /// trackingAccepted is a separate result, never inferred from a match count.
        bool dataAssociationAndInitialization(Estimator& estimator, const ViParameters& params,
            std::shared_ptr<MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) override final;

        using ImuPropagator = std::function<bool(const ImuMeasurementDeque&, const ImuParameters&,
            kinematics::Transformation&, SpeedAndBias&, const Time&, const Time&,
            Eigen::Matrix<double, 15, 15>*, Eigen::Matrix<double, 15, 15>*)>;
        /// Bind the real IMU prediction/preintegration module before starting workers.
        /// This legacy interface uses T_WS, whereas backend predictions use T_WB.
        /// The built-in IMU implementation is used by default. An override is optional;
        /// failure leaves every output unchanged.
        void setImuPropagator(ImuPropagator propagator) { imuPropagator_ = std::move(propagator); }
        bool propagation(const ImuMeasurementDeque&, const ImuParameters&,
            kinematics::Transformation&, SpeedAndBias&, const Time&, const Time&,
            Eigen::Matrix<double, 15, 15>*, Eigen::Matrix<double, 15, 15>*) const override final;

        features::OrbExtractorOptions orbOptions() const { return orbFeatures_.options(); }
        void setOrbOptions(const features::OrbExtractorOptions& options);
        /// Prepare calibrated LUTs/static masks before input starts; consumes no frame.
        void prepareOrb(const cameras::NCameraSystem& rig, bool parallel = true);
        int orbMatchingThreshold() const { return orbMatchingThreshold_; }
        void setOrbMatchingThreshold(int threshold);
        void setPoseSolverOptions(const frontend_pose::Options& options);
        void setAssociationOptions(const frontend_association::Options& options);
        const FrontendAssociationReport& lastAssociationReport() const { return lastReport_; }
        /// Metric visual pose tracking has succeeded against finite-range tracks.
        /// This is not an IMU bias/gravity/scale initialization certificate.
        bool isInitialized() const { return isInitialized_; }
        bool loadOrbVocabulary(const std::string& filename, const std::string& expectedProfile, std::string& error);
        /// Legacy full-graph import is unavailable in the local window backend.
        bool loadComponent(std::string, const ImuParameters&, const cameras::NCameraSystem&, bool componentFixed = true);
        /// Clear together with backend/map when restarting or changing ORB profiles.
        void clear();
        /// Drop uncommitted tracking diagnostics after a backend rollback without
        /// changing extractors, descriptor profiles or the accepted window's map.
        void resetTrackingState();

    private:
        struct PoseDecision {
            kinematics::Transformation T_WB;
            size_t inliers = 0;
            bool metric = false;
            bool reducedBudget = false;
            std::set<KeypointIdentifier> rejected;
        };
        bool runRansac3d2d(const Estimator&, const frontend_association::Plan&, PoseDecision&,bool forceFullBudget=false) const;
        bool runRansac2d2d(const Estimator&, const frontend_association::Plan&, PoseDecision&) const;
        class DBoW;
        std::unique_ptr<DBoW> dBow_;
        int getFilteredDBoWResult(const std::unique_ptr<DBoW>&, const std::vector<cv::Mat>&,
            std::vector<std::pair<StateId, double>>&) const;
        bool verifyRecognisedPlace(const Estimator&, const ViParameters&,
            std::shared_ptr<const MultiFrame>, std::shared_ptr<const MultiFrame>,
            kinematics::Transformation&, Eigen::Matrix<double, 6, 6>&, int minInliers = 10);
        void queryLocalPlaces(const Estimator&, const ViParameters&);

        features::OrbFrontendFeatures orbFeatures_;
        bool isInitialized_ = false;
        const size_t numCameras_;
        int orbMatchingThreshold_ = 64;
        frontend_pose::Options poseOptions_;
        frontend_association::Options associationOptions_;
        std::atomic_bool trackingLost_{ false };
        StateId lastAssociatedState_;
        FrontendAssociationReport lastReport_;
        std::size_t absoluteBudgetEpochs_=0;
        ImuPropagator imuPropagator_;
    };
}
