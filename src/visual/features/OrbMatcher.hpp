#pragma once

#include <cameras/CameraBase.hpp>
#include <kinematics/Transformation.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <vector>
#include <cstdint>
#include <array>
#include <map>

namespace cgmli {
    class Frame;
    class MultiFrame;
    namespace features {
        using OrbBearings = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
        using OrbDescriptorWords=std::array<uint64_t,4>;
        using OrbWordRows=std::vector<OrbDescriptorWords>;
        struct OrbTrainIndex {
            size_t count=0;
            const cv::KeyPoint* keypoints=nullptr;
            const cameras::CameraBase* camera=nullptr;
            int width=0,height=0,columns=0,rows=0;
            double cellSize=16;
            std::vector<std::vector<int>> grid;
            std::map<int,std::vector<int>> octaves;
        };
        struct OrbMatchStatistics {
            uint64_t candidatePairs=0, descriptorComparisons=0;
            double setupMs=0, traversalMs=0, selectionMs=0; ///< Per-job elapsed time; traversal includes fused Hamming.
            uint64_t predictedQueries=0, unpredictedQueries=0;
            uint64_t ownedQueries=0,ownedMatches=0,prioritySuppressed=0;
        };
        struct OrbMatcherOptions {
            int maxHammingDistance = 64;
            float ratioThreshold = 0.8f;
            bool mutualCheck = true;
            bool reverseRatioTest = true;
            bool allowSingleCandidate = false; ///< Ratio cannot be established with one neighbor.
            int maxOctaveDifference = 1; ///< Guided path only; -1 disables this gate.
            double epipolarThresholdRadians = 0.01;
            double rotationThresholdRadians = 0.02; ///< Zero-baseline/pure-rotation guidance.
            double minimumBaseline = 1e-9;
        };
        struct OrbMatchGuidance {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                std::shared_ptr<const cameras::CameraBase> queryCamera, trainCamera;
            /// p_Ctrain = C_train_query * p_Cquery + r_train_query.
            kinematics::Transformation T_train_query;
            bool useSphericalGeometry = true;
            /// Query/train IMAGE masks, e.g. the rig's directed overlap masks; 0 excludes.
            cv::Mat queryRegionMask, trainRegionMask;
            /// Optional descriptor-pair mask: query rows x train rows, CV_8UC1.
            cv::Mat allowedPairs;
            /// Optional per-feature gates and immutable per-frame unit-ray caches.
            /// Non-finite cached rays exclude a feature. Cache owners outlive matchGuided().
            std::vector<uint8_t> queryEnabled, trainEnabled;
            const OrbBearings* queryBearings=nullptr;
            const OrbBearings* trainBearings=nullptr;
            /// Optional immutable per-proposal layouts. Per-job gates are still applied.
            const OrbWordRows* queryWords=nullptr;
            const OrbWordRows* trainWords=nullptr;
            const OrbTrainIndex* trainIndex=nullptr;
            /// Optional landmark identities for pooled map descriptors. Ratio/mutual
            /// neighbors are distinct identities, never two views of the same point.
            const std::vector<uint64_t>* queryIdentities=nullptr;
            /// Optional ownership tier (0=new texture, 1=existing identity). Existing
            /// tracks may also establish reciprocal uniqueness against existing IDs;
            /// the full reverse test remains available when that pool is too sparse.
            /// Forward ratio, geometry and distance gates remain unchanged. All views
            /// grouped by queryIdentities must have the same tier. Empty means tier 0.
            const std::vector<uint8_t>* queryPriorities=nullptr;
            /// Optional prediction for every query keypoint in TRAIN pixels. Non-finite
            /// predictions mean unavailable/not visible and exclude that query feature.
            std::vector<cv::Point2f> predictedTrainPixels;
            /// Frontend seed tracks without depth keep spherical search; a known
            /// map point outside the image must instead be disabled via queryEnabled.
            bool allowUnpredictedQueries=false;
            float searchRadiusPixels = 30.0f;
        };

        /// CPU Hamming reference matcher. BFMatcher already derives from DescriptorMatcher.
        /// Raw OpenCV match/knnMatch/radiusMatch retain OpenCV semantics. SLAM callers
        /// use matchFiltered/matchGuided for ratio, mutual and geometric filtering.
        class OrbMatcher final : public cv::BFMatcher {
        public:
            explicit OrbMatcher(const OrbMatcherOptions& options = {});
            cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const override;
            cv::String getDefaultName() const override { return "cgmli.ORB.GuidedHamming"; }
            const OrbMatcherOptions& options() const { return options_; }
            static OrbTrainIndex prepareTrainIndex(const std::vector<cv::KeyPoint>& points,
                const cameras::CameraBase* camera,float searchRadiusPixels);
            void matchFiltered(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors,
                std::vector<cv::DMatch>& matches, const cv::Mat& allowedPairs = {}) const;
            void matchGuided(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors,
                const std::vector<cv::KeyPoint>& queryKeypoints,
                const std::vector<cv::KeyPoint>& trainKeypoints,
                const OrbMatchGuidance& guidance, std::vector<cv::DMatch>& matches,
                OrbMatchStatistics* statistics=nullptr) const;
            void matchFrames(const Frame& query, const Frame& train, const OrbMatchGuidance& guidance,
                std::vector<cv::DMatch>& matches) const;
            /// T_Scurrent_Sprevious comes from the upper-layer IMU prediction. Pixel
            /// predictions must include depth when available; rotation-only windows for
            /// nearby points must be widened by the caller to account for translation.
            void matchTemporal(const MultiFrame& previous, std::size_t queryCamera,
                const MultiFrame& current, std::size_t trainCamera,
                const kinematics::Transformation& T_Scurrent_Sprevious,
                const std::vector<cv::Point2f>& predictedTrainPixels,
                std::vector<cv::DMatch>& matches, float searchRadiusPixels = 30.0f) const;
            /// Same capture time. Requires the rig's overlap graph to have been computed.
            /// Uses fixed camera-to-rig extrinsics; online common rig/body motion cancels.
            void matchCrossCamera(const MultiFrame& frame, std::size_t queryCamera,
                std::size_t trainCamera, std::vector<cv::DMatch>& matches) const;

        protected:
            void knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches,
                int k, cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;
            void radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches,
                float maxDistance, cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;
        private:
            OrbMatcherOptions options_;
        };
    }
}
