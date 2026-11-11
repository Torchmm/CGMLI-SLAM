#pragma once

#include <cameras/CameraBase.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <vector>
#include <utility>

namespace cgmli {
    namespace features {
        namespace detail { class OrbCuda; struct OrbCudaDescriptorRows; }

        struct OrbExtractorOptions {
            enum class DescriptorMode { Standard, GravityAligned, GravityCameraAware };
            enum class SamplingMode { SparseBox, AffineGaussianReference };
            // Rig default. Requires a CameraBase and a per-frame gravity update.
            DescriptorMode descriptorMode = DescriptorMode::GravityCameraAware;
            SamplingMode samplingMode = SamplingMode::SparseBox;
            double virtualFocalLength = 0.0; ///< 0: use camera intrinsics[0] (fu); >0: explicit shared reference.
            double gravityParallelSinThreshold = 0.1; ///< Texture fallback near gravity poles (~6 degrees).
            double maximumWarpCondition = 12.0; ///< Reject severely anisotropic/singular patches.
            double maximumWarpStretch = 4.0; ///< Largest singular value in level pixels/reference pixel.
            int maxFeatures = 1000;
            int pyramidLevels = 8;
            float scaleFactor = 1.2f;
            int cellSize = 32; ///< Pixels at each pyramid level, not base-image pixels.
            int initialFastThreshold = 20;
            int minimumFastThreshold = 7;
            bool layerFast = true; ///< Share FAST-9 scores per layer; retain cell-local NMS. False: ROI reference.
            bool boundaryFast = true; ///< Raw FAST scans only grid-border strips; false keeps the full-layer raw reference.
            bool adaptiveGeometryCandidates = true; ///< Start with a small valid batch per cell, replenish to the original cap if short.
            bool batchDescriptorCoordinates = true; ///< Batched affine sampling coordinates; false keeps the scalar reference.
            bool groupedBalancing = true; ///< Exact (bin,cell) representative heap; false retains flat-heap reference.
            bool budgetedFastFallback = true; ///< Fill uncovered coarse cells first; false retains per-cell low FAST.
            bool compactGrayIntegral = true; ///< Exact int32 prefix sums when the image area is overflow-safe.
            int candidatesPerCell = 16;
            int maxFeaturesPerCell = 8;
            int edgeThreshold = 31;
            bool balanceBearings = true; ///< Requires a camera model; default SLAM path.
            int azimuthBins = 16;
            int zBins = 8; ///< Uniform z bands + azimuth sectors have equal solid angle.
            bool useCuda = false; ///< The SLAM frontend enables this when the CUDA build is available.
            bool cudaEarlyIntegral = true; ///< Prefetch integrals while the CPU selects keypoints.
            bool keepCudaDescriptorRows = false; ///< Allow the rig matcher to reuse current-frame device rows.
        };

        struct OrbDistribution {
            std::vector<int> perLevel, imageCells, directionBins;
            int occupiedImageCells = 0, occupiedDirectionBins = 0;
        };
        struct OrbExtractionTiming {
            double pyramidMs=0,detectionMs=0,descriptorMs=0;
            double fastMs=0,candidateSortMs=0,candidateFilterMs=0,balanceMs=0;
            double fastRawScanMs=0,fastNmsScanMs=0,fastBoundaryMs=0,fastBucketMs=0,fastFallbackMs=0;
            bool maskCacheHit=false;
            // Counts include candidates later rejected by geometry/descriptor support.
            size_t geometryEvaluations=0,geometryCacheHits=0,fastCandidates=0,grayIntegralBuilds=0;
            size_t fastCalls=0,grayIntegral32Builds=0;
            size_t fastRawCorners=0,fastFallbackCalls=0;
            size_t emptyMaskCells=0,quotaFallbackCells=0,geometryFallbackCells=0;
            size_t budgetSkippedCells=0;
            size_t fastRawPixels=0,balanceHeapPushes=0;
            size_t geometryMemoHits=0,geometryExpansionRounds=0;
            bool usedCuda=false;
            double gpuUploadMs=0,gpuPyramidMs=0,gpuFastMs=0,gpuCornersDownloadMs=0;
            double gpuIntegralMs=0,gpuDescriptorMs=0,gpuDescriptorsDownloadMs=0;
        };

        /// CPU Reference for framework V1.10 sections 8/8.1.
        /// FeatureDetector and DescriptorExtractor are both aliases of Feature2D in OpenCV 4.
        /// One instance can therefore fill both Frame::detector_ and Frame::extractor_.
        /// Grid FAST/Top-K + image/S2 balancing and 256-bit ORB descriptors (WTA_K=2).
        /// GravityCameraAware uses the BRISK tangent-plane geometry with ORB point pairs.
        /// Calibration/mask LUTs are immutable snapshots; recreate after calibration changes.
        /// Update gravity before each frame. Do not mutate/use one instance concurrently.
        class OrbFeatureExtractor final : public cv::Feature2D {
        public:
            explicit OrbFeatureExtractor(std::shared_ptr<const cameras::CameraBase> camera,
                const OrbExtractorOptions& options = {});
            ~OrbFeatureExtractor() override;
            using cv::Feature2D::detect;
            using cv::Feature2D::compute;
            void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints,
                cv::InputArray mask = cv::noArray()) override;
            void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints,
                cv::OutputArray descriptors) override;
            void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                bool useProvidedKeypoints = false) override;
            int descriptorSize() const override { return 32; }
            int descriptorType() const override { return CV_8UC1; }
            int defaultNorm() const override { return cv::NORM_HAMMING; }
            bool empty() const override { return false; }
            cv::String getDefaultName() const override {
                switch (options_.descriptorMode) {
                case OrbExtractorOptions::DescriptorMode::GravityAligned:
                    return options_.samplingMode == OrbExtractorOptions::SamplingMode::SparseBox ?
                        "cgmli.ORB.Gravity.SparseBox.v3" : "cgmli.ORB.Gravity.v2";
                case OrbExtractorOptions::DescriptorMode::GravityCameraAware:
                    return options_.samplingMode == OrbExtractorOptions::SamplingMode::SparseBox ?
                        "cgmli.ORB.GravityCameraAware.SparseBox.v3" : "cgmli.ORB.GravityCameraAware.v2";
                default: return "cgmli.ORB.GridSphere";
                }
            }
            const OrbExtractorOptions& options() const { return options_; }
            const std::shared_ptr<const cameras::CameraBase>& camera() const { return camera_; }
            /// Normalized gravity direction in CAMERA coordinates, not acceleration containing motion.
            void setExtractionDirection(const Eigen::Vector3d& gravityCamera);
            void clearExtractionDirection();
            bool hasExtractionDirection() const { return hasGravity_; }
            bool isCameraAware() const {
                return options_.descriptorMode == OrbExtractorOptions::DescriptorMode::GravityCameraAware;
            }
            /// Read-only LUT views for a future GPU upload; do not modify their storage.
            const cv::Mat& rayDirections() const { return rayDirections_; } ///< CV_32FC3, unit bearings.
            const cv::Mat& projectionJacobians() const { return projectionJacobians_; } ///< CV_32FC(6), row-major 2x3.
            const cv::Mat& geometryValidMask() const { return geometryValidMask_; }
            double virtualFocalLength() const { return virtualFocalLength_; }
            /// Reference-plane coordinates -> keypoint pyramid-level offsets.
            /// Returns false for invalid geometry; patch/mask support is checked during compute().
            /// With camera-aware gravity poles this returns an unrotated reference basis;
            /// compute() estimates texture orientation with sparse reference-plane probes
            /// (or a rectified patch in the optional Gaussian reference backend).
            /// With GravityAligned poles the caller supplies the image texture angle.
            bool samplingWarp(const cv::KeyPoint& keypoint, Eigen::Matrix2d& warp,
                bool& gravityAligned) const;
            OrbDistribution distribution(const std::vector<cv::KeyPoint>& keypoints) const;
            const OrbExtractionTiming& lastTiming() const { return lastTiming_; }
            /// Build image-independent pyramid masks before the first image arrives.
            /// Requires calibrated geometry; neither an image nor gravity is needed.
            /// Repeated calls reuse the cache. Calibration/mask changes require a new instance.
            void prepareStaticCaches();
            bool cudaReady() const;
            std::shared_ptr<const detail::OrbCudaDescriptorRows> cudaDescriptorRows() const { return cudaDescriptorRows_; }

        private:
            struct KeypointGeometry {
                // Owned by a single detectAndCompute() call, never persisted across
                // frames or reused by separate detect()/compute() calls.
                Eigen::Matrix2d warp=Eigen::Matrix2d::Identity();
                cv::Point2f levelPixel;
                bool gravityAligned=false;
                int bin=0;
            };
            struct Candidate { cv::KeyPoint keypoint; int cell,bin; KeypointGeometry geometry; };
            struct GeometryMemo { bool valid=false,geometricallyValid=false; Candidate candidate; };
            struct Pyramid {
                std::vector<cv::Mat> images,safeMasks,samplingMasks,invalidIntegrals,cellMasks;
                detail::OrbCuda* cuda=nullptr;
            };
            cv::Mat grayscale(cv::InputArray image) const;
            cv::Mat combinedMask(const cv::Mat& gray, cv::InputArray mask) const;
            void initialiseLevels(const cv::Size& size);
            void buildPyramid(const cv::Mat& gray, cv::InputArray mask, Pyramid& pyramid);
            void buildMasks(const cv::Mat& baseMask, Pyramid& pyramid) const;
#ifdef CGMLI_ENABLE_CUDA_ORB
            void prepareCuda();
            void disableCuda(const char* reason);
            void computeCuda(const Pyramid& pyramid,std::vector<cv::KeyPoint>& keypoints,
                cv::OutputArray descriptors,const std::vector<KeypointGeometry>& geometry);
#endif
            std::vector<int> levelBudgets(int activeLevels) const;
            int directionBin(const cv::Point2f& pixel) const;
            int directionBin(const Eigen::Vector3d& bearing) const;
            float orientation(const cv::Mat& image, const cv::Point2f& pixel) const;
            cv::Point2f levelToBase(const cv::Point2f& pixel, int octave) const;
            cv::Point2f baseToLevel(const cv::Point2f& pixel, int octave) const;
            void detectImpl(const Pyramid& pyramid,std::vector<cv::KeyPoint>& keypoints,
                std::vector<KeypointGeometry>& geometry);
            void computeImpl(const cv::Mat& gray,const Pyramid& pyramid,std::vector<cv::KeyPoint>& keypoints,
                cv::OutputArray descriptors,const std::vector<KeypointGeometry>* detectedGeometry);
            bool samplingWarpImpl(const cv::KeyPoint& keypoint,Eigen::Matrix2d& warp,
                bool& gravityAligned,Eigen::Vector3d* bearing) const;
            bool geometryAt(const cv::Point2f& pixel, Eigen::Vector3d& bearing,
                Eigen::Matrix<double, 2, 3>& jacobian) const;
            void initialiseGeometry();
            std::shared_ptr<const cameras::CameraBase> camera_;
            OrbExtractorOptions options_;
            cv::Mat cameraValidMask_; ///< Snapshot of mask AND the back-projection domain.
            cv::Mat rayDirections_, projectionJacobians_, geometryValidMask_;
            Eigen::Vector3d gravityCamera_ = Eigen::Vector3d::Zero();
            bool hasGravity_ = false;
            double virtualFocalLength_ = 0.0;
            OrbExtractionTiming lastTiming_;
            std::shared_ptr<const detail::OrbCudaDescriptorRows> cudaDescriptorRows_;
            // Masks depend only on the immutable calibration and pyramid layout.
            // External masks take a per-call path and never overwrite this cache.
            cv::Size pyramidBaseSize_;
            std::vector<cv::Size> levelSizes_;
            std::vector<double> levelScales_,levelBaseX_,levelBaseY_;
            std::vector<cv::Mat> staticSafeMasks_,staticSamplingMasks_,staticInvalidIntegrals_,staticCellMasks_;
            std::vector<cv::Mat> imageScratch_,intensityScratch_;
            std::vector<cv::Mat> fastScoreScratch_;
            std::vector<cv::KeyPoint> rawFastScratch_,maximaFastScratch_;
            std::vector<cv::KeyPoint> cellFastScratch_,stripFastScratch_;
            std::vector<std::vector<cv::KeyPoint>> layerCellScratch_;
            std::vector<std::vector<cv::KeyPoint>> lowCellScratch_;
            std::vector<std::vector<std::pair<int,int>>> retainedPixelScratch_;
            std::vector<Candidate> candidateScratch_,validCandidateScratch_,combinedCandidateScratch_;
            std::vector<int> groupNextScratch_,candidateOrderScratch_;
            std::vector<GeometryMemo> geometryMemoScratch_;
            std::vector<cv::Mat> geometryMemoIndexScratch_;
            bool staticMasksReady_=false;
#ifdef CGMLI_ENABLE_CUDA_ORB
            std::unique_ptr<detail::OrbCuda> cuda_;
            bool cudaDisabled_=false;
#endif
        };
    }
}
