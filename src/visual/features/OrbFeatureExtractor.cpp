#include <visual/features/OrbFeatureExtractor.hpp>
#include <visual/features/OrbBriefPattern.hpp>
#include <visual/features/OrbCameraAwareSampling.hpp>
#include <visual/features/OrbSparseSampling.hpp>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <queue>
#include <chrono>
#include <limits>
#ifdef CGMLI_ENABLE_CUDA_ORB
#include <visual/features/OrbCuda.hpp>
#include <glog/logging.h>
#endif

namespace cgmli {
    namespace features {
        namespace {
            constexpr int kPatchSize = 31, kPatchRadius = 15;
            constexpr double kPi = 3.14159265358979323846;
            // Rotated 31x31 BRIEF patch plus the 7x7 descriptor smoothing support.
            constexpr int kMaskRadius = 25;
            bool betterResponse(const cv::KeyPoint& a, const cv::KeyPoint& b) {
                if (a.response != b.response) return a.response > b.response;
                if (a.pt.y != b.pt.y) return a.pt.y < b.pt.y;
                return a.pt.x < b.pt.x;
            }
            bool safePoint(const cv::Mat& mask, const cv::Point2f& point) {
                if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0 || point.y < 0 ||
                    point.x >= mask.cols || point.y >= mask.rows) return false;
                const int x = cvRound(point.x), y = cvRound(point.y);
                return x >= 0 && y >= 0 && x < mask.cols && y < mask.rows && mask.at<uchar>(y, x) != 0;
            }
            Eigen::Matrix2d rotation(double degrees) {
                const double a = degrees * kPi / 180.0, c = std::cos(a), s = std::sin(a);
                Eigen::Matrix2d result; result << c, -s, s, c; return result;
            }
            int fastNineScore(const cv::Mat& image,int x,int y) {
                // Independently evaluate the FAST-9 definition: largest threshold
                // for a strict contiguous 9-of-16 bright/dark arc. Sliding min/max
                // queues cover all 16 circular arcs in linear work, without allocation.
                constexpr int dx[16]={0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
                constexpr int dy[16]={3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3};
                int difference[24],minimum[24],maximum[24],minHead=0,minTail=0,maxHead=0,maxTail=0,best=0;
                const int centre=image.at<uchar>(y,x);
                for(int i=0;i<24;++i) {
                    difference[i]=i<16?centre-int(image.ptr<uchar>(y+dy[i])[x+dx[i]]):difference[i-16];
                    while(minHead<minTail && minimum[minHead]<=i-9) ++minHead;
                    while(maxHead<maxTail && maximum[maxHead]<=i-9) ++maxHead;
                    while(minHead<minTail && difference[minimum[minTail-1]]>=difference[i]) --minTail;
                    while(maxHead<maxTail && difference[maximum[maxTail-1]]<=difference[i]) --maxTail;
                    minimum[minTail++]=i; maximum[maxTail++]=i;
                    if(i>=8) best=std::max(best,std::max(difference[minimum[minHead]],-difference[maximum[maxHead]]));
                }
                return std::max(0,best-1);
            }
        }

        OrbFeatureExtractor::OrbFeatureExtractor(std::shared_ptr<const cameras::CameraBase> camera,
            const OrbExtractorOptions& options)
            : camera_(std::move(camera)), options_(options) {
            if (options_.maxFeatures <= 0 || options_.pyramidLevels < 1 || options_.pyramidLevels > 16 ||
                !std::isfinite(options_.scaleFactor) || options_.scaleFactor <= 1.0f || options_.scaleFactor > 2.0f ||
                options_.cellSize < 8 || options_.initialFastThreshold < 1 || options_.initialFastThreshold > 255 ||
                options_.minimumFastThreshold < 1 || options_.minimumFastThreshold > options_.initialFastThreshold ||
                options_.candidatesPerCell < 1 || options_.maxFeaturesPerCell < 1 ||
                options_.maxFeaturesPerCell > options_.candidatesPerCell || options_.edgeThreshold < kMaskRadius ||
                options_.azimuthBins < 1 || options_.azimuthBins > 128 || options_.zBins < 1 || options_.zBins > 128)
                throw std::invalid_argument("Invalid ORB extraction options");
            if (options_.balanceBearings && !camera_)
                throw std::invalid_argument("S2 ORB balancing requires a CameraBase model");
            const auto mode = options_.descriptorMode;
            if (mode != OrbExtractorOptions::DescriptorMode::Standard &&
                mode != OrbExtractorOptions::DescriptorMode::GravityAligned &&
                mode != OrbExtractorOptions::DescriptorMode::GravityCameraAware)
                throw std::invalid_argument("Unknown ORB descriptor mode");
            if (options_.samplingMode != OrbExtractorOptions::SamplingMode::SparseBox &&
                options_.samplingMode != OrbExtractorOptions::SamplingMode::AffineGaussianReference)
                throw std::invalid_argument("Unknown ORB sampling mode");
            if (!std::isfinite(options_.virtualFocalLength) || options_.virtualFocalLength < 0 ||
                !std::isfinite(options_.gravityParallelSinThreshold) || options_.gravityParallelSinThreshold <= 0 ||
                options_.gravityParallelSinThreshold >= 1 || !std::isfinite(options_.maximumWarpCondition) ||
                options_.maximumWarpCondition < 1 || !std::isfinite(options_.maximumWarpStretch) ||
                options_.maximumWarpStretch < 1 || options_.maximumWarpStretch > detail::kOrbMaximumSupersampling)
                throw std::invalid_argument("Invalid gravity/camera-aware ORB options");
            if (mode != OrbExtractorOptions::DescriptorMode::Standard && !camera_)
                throw std::invalid_argument("Gravity ORB requires a CameraBase model");
            if (camera_) {
                const cv::Size size(static_cast<int>(camera_->imageWidth()), static_cast<int>(camera_->imageHeight()));
                if (size.width <= 0 || size.height <= 0) throw std::invalid_argument("Invalid ORB camera dimensions");
                const cv::Mat& cameraMask = camera_->mask();
                if (!cameraMask.empty() && (cameraMask.type() != CV_8UC1 || cameraMask.size() != size))
                    throw std::invalid_argument("Camera mask must be CV_8UC1 at the calibrated image size");
                cameraValidMask_ = cv::Mat::zeros(size, CV_8UC1);
                initialiseLevels(size);
                // Gravity modes fill validity and geometry in one back-projection
                // pass below. Standard mode needs only the domain snapshot.
                if(mode==OrbExtractorOptions::DescriptorMode::Standard)
                  for (int y = 0; y < size.height; ++y) for (int x = 0; x < size.width; ++x) {
                    if (!cameraMask.empty() && cameraMask.at<uchar>(y, x) == 0) continue;
                    Eigen::Vector3d bearing;
                    if (camera_->backProject(Eigen::Vector2d(x, y), &bearing) && bearing.allFinite() &&
                        std::isfinite(bearing.stableNorm()) && bearing.stableNorm() > 1e-12)
                        cameraValidMask_.at<uchar>(y, x) = 255;
                }
            }
            if (mode != OrbExtractorOptions::DescriptorMode::Standard) initialiseGeometry();
        }

        OrbFeatureExtractor::~OrbFeatureExtractor() = default;

        bool OrbFeatureExtractor::cudaReady() const {
#ifdef CGMLI_ENABLE_CUDA_ORB
            return bool(cuda_);
#else
            return false;
#endif
        }

        void OrbFeatureExtractor::initialiseGeometry() {
            virtualFocalLength_ = options_.virtualFocalLength;
            if (isCameraAware() && virtualFocalLength_ == 0) {
                // All project EUCM/Pinhole variants expose fu as the first intrinsic.
                // An external CameraBase with another ordering must pass an explicit value.
                Eigen::VectorXd intrinsics; camera_->getIntrinsics(intrinsics);
                if (intrinsics.size() == 0 || !std::isfinite(intrinsics[0]) || intrinsics[0] <= 0)
                    throw std::invalid_argument("Set an explicit ORB virtual focal length for this camera");
                virtualFocalLength_ = intrinsics[0];
            }
            const cv::Size size = cameraValidMask_.size();
            const cv::Mat& cameraMask=camera_->mask();
            rayDirections_ = cv::Mat::zeros(size, CV_32FC3);
            projectionJacobians_ = cv::Mat::zeros(size, CV_32FC(6));
            geometryValidMask_ = cv::Mat::zeros(size, CV_8UC1);
            // Build our own zero-initialized snapshot: existing camera maps can contain
            // undefined Jacobians at invalid pixels. Never divide a bearing by its z.
            for (int y = 0; y < size.height; ++y) for (int x = 0; x < size.width; ++x) {
                if (!cameraMask.empty() && cameraMask.at<uchar>(y,x)==0) continue;
                Eigen::Vector3d bearing;
                if (!camera_->backProject(Eigen::Vector2d(x, y), &bearing) || !bearing.allFinite()) continue;
                const double norm = bearing.stableNorm();
                if (!std::isfinite(norm) || norm < 1e-12) continue;
                cameraValidMask_.at<uchar>(y,x)=255;
                bearing /= norm;
                Eigen::Vector2d projected;
                Eigen::Matrix<double, 2, 3> jacobian;
                if (camera_->project(bearing, &projected, &jacobian) != cameras::ProjectionStatus::Successful ||
                    !projected.allFinite() || !jacobian.allFinite() ||
                    (projected - Eigen::Vector2d(x, y)).norm() > 0.25) continue;
                const auto jf = jacobian.cast<float>().eval();
                if (!jf.allFinite()) continue;
                rayDirections_.at<cv::Vec3f>(y, x) = cv::Vec3f(
                    static_cast<float>(bearing.x()), static_cast<float>(bearing.y()), static_cast<float>(bearing.z()));
                projectionJacobians_.at<cv::Vec<float, 6>>(y, x) = cv::Vec<float, 6>(
                    jf(0, 0), jf(0, 1), jf(0, 2), jf(1, 0), jf(1, 1), jf(1, 2));
                geometryValidMask_.at<uchar>(y, x) = 255;
            }
        }

        void OrbFeatureExtractor::setExtractionDirection(const Eigen::Vector3d& gravityCamera) {
            const double norm = gravityCamera.stableNorm();
            if (!gravityCamera.allFinite() || !std::isfinite(norm) || norm < 1e-12)
                throw std::invalid_argument("ORB extraction direction must be finite and nonzero");
            gravityCamera_ = gravityCamera / norm; hasGravity_ = true;
        }
        void OrbFeatureExtractor::clearExtractionDirection() {
            gravityCamera_.setZero(); hasGravity_ = false;
        }

        bool OrbFeatureExtractor::geometryAt(const cv::Point2f& pixel, Eigen::Vector3d& bearing,
            Eigen::Matrix<double, 2, 3>& jacobian) const {
            if (geometryValidMask_.empty() || !std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
                pixel.x < 0 || pixel.y < 0 || pixel.x >= rayDirections_.cols - 1 ||
                pixel.y >= rayDirections_.rows - 1) return false;
            const int x = static_cast<int>(std::floor(pixel.x)), y = static_cast<int>(std::floor(pixel.y));
            const double dx = pixel.x - x, dy = pixel.y - y;
            bearing.setZero(); jacobian.setZero();
            for (int v = 0; v < 2; ++v) for (int u = 0; u < 2; ++u) {
                if (geometryValidMask_.at<uchar>(y + v, x + u) == 0) return false;
                const double weight = (u ? dx : 1 - dx) * (v ? dy : 1 - dy);
                const auto b = rayDirections_.at<cv::Vec3f>(y + v, x + u);
                const auto j = projectionJacobians_.at<cv::Vec<float, 6>>(y + v, x + u);
                for (int i = 0; i < 3; ++i) bearing[i] += weight * b[i];
                for (int i = 0; i < 6; ++i) jacobian(i / 3, i % 3) += weight * j[i];
            }
            const double norm = bearing.stableNorm();
            if (!bearing.allFinite() || !jacobian.allFinite() || norm < 1e-12) return false;
            bearing /= norm; return true;
        }

        bool OrbFeatureExtractor::samplingWarp(const cv::KeyPoint& keypoint, Eigen::Matrix2d& warp,
            bool& gravityAligned) const {
            return samplingWarpImpl(keypoint,warp,gravityAligned,nullptr);
        }
        bool OrbFeatureExtractor::samplingWarpImpl(const cv::KeyPoint& keypoint,Eigen::Matrix2d& warp,
            bool& gravityAligned,Eigen::Vector3d* outputBearing) const {
            warp.setIdentity(); gravityAligned = false;
            if (!std::isfinite(keypoint.pt.x) || !std::isfinite(keypoint.pt.y) ||
                keypoint.octave < 0 || keypoint.octave >= options_.pyramidLevels) return false;
            if (options_.descriptorMode == OrbExtractorOptions::DescriptorMode::Standard) {
                if (!std::isfinite(keypoint.angle) || keypoint.angle < 0) return false;
                warp = rotation(keypoint.angle); return true;
            }
            if (!hasGravity_) throw std::logic_error("Set ORB camera gravity before detecting/describing each frame");
            Eigen::Vector3d bearing;
            Eigen::Matrix<double, 2, 3> jacobian;
            if (!geometryAt(keypoint.pt, bearing, jacobian)) return false;
            if(outputBearing) *outputBearing=bearing;
            const double scale = levelScales_.at(size_t(keypoint.octave));
            const double width = camera_->imageWidth(), height = camera_->imageHeight();
            const int levelWidth = levelSizes_.at(size_t(keypoint.octave)).width;
            const int levelHeight = levelSizes_.at(size_t(keypoint.octave)).height;
            if (levelWidth < 1 || levelHeight < 1) return false;
            // Camera J is in original-image pixels. Angular support grows by the nominal
            // pyramid scale, but the actual rounded resize has separate x/y scales.
            jacobian.row(0) *= scale * levelWidth / width;
            jacobian.row(1) *= scale * levelHeight / height;
            Eigen::Vector3d eu = gravityCamera_.cross(bearing);
            gravityAligned = eu.norm() >= options_.gravityParallelSinThreshold;
            if (gravityAligned) eu.normalize();
            else {
                if (!isCameraAware()) {
                    if (!std::isfinite(keypoint.angle) || keypoint.angle < 0) return false;
                    warp = rotation(keypoint.angle); return true;
                }
                // Establish a neutral tangent basis from image x, independent of unstable
                // gravity at the pole. compute() estimates texture rotation in this plane.
                const Eigen::Matrix2d gram = jacobian * jacobian.transpose();
                if (!gram.allFinite()) return false;
                const auto gramSingular = detail::singularValues2x2(gram);
                if (gramSingular[1] <= 1e-12 ||
                    gramSingular[0] / gramSingular[1] > options_.maximumWarpCondition * options_.maximumWarpCondition)
                    return false;
                eu = jacobian.transpose() * gram.inverse() * Eigen::Vector2d::UnitX();
                eu -= bearing * bearing.dot(eu);
                if (!eu.allFinite() || eu.norm() < 1e-12) return false;
                eu.normalize();
            }
            const Eigen::Vector2d imageX = jacobian * eu;
            if (!imageX.allFinite() || imageX.norm() < 1e-12) return false;
            if (!isCameraAware()) {
                warp = rotation(std::atan2(imageX.y(), imageX.x()) * 180.0 / kPi); return true;
            }
            const Eigen::Vector3d ev = bearing.cross(eu).normalized();
            warp.col(0) = imageX / virtualFocalLength_;
            warp.col(1) = jacobian * ev / virtualFocalLength_;
            if (!warp.allFinite()) return false;
            const auto singular = detail::singularValues2x2(warp);
            // Resize conversion is already incorporated above; do not divide by scale again.
            return singular[1] > 1e-6 && singular[0] <= options_.maximumWarpStretch &&
                singular[0] / singular[1] <= options_.maximumWarpCondition;
        }

        cv::Mat OrbFeatureExtractor::grayscale(cv::InputArray input) const {
            const cv::Mat image = input.getMat();
            if (image.empty()) return {};
            if (image.depth() != CV_8U) throw std::invalid_argument("ORB expects 8-bit input images");
            if (camera_ && (image.cols != static_cast<int>(camera_->imageWidth()) ||
                image.rows != static_cast<int>(camera_->imageHeight())))
                throw std::invalid_argument("ORB image size must match camera calibration");
            cv::Mat gray;
            if (image.channels() == 1) gray = image;
            else if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            else if (image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
            else throw std::invalid_argument("ORB supports gray, BGR, or BGRA input");
            return gray;
        }

        cv::Mat OrbFeatureExtractor::combinedMask(const cv::Mat& gray, cv::InputArray input) const {
            cv::Mat mask = cameraValidMask_.empty() ? cv::Mat(gray.size(), CV_8UC1, cv::Scalar(255))
                : cameraValidMask_.clone();
            if (!input.empty()) {
                const cv::Mat extra = input.getMat();
                if (extra.type() != CV_8UC1 || extra.size() != gray.size())
                    throw std::invalid_argument("ORB mask must be CV_8UC1 with the image dimensions");
                cv::Mat binary; cv::compare(extra, 0, binary, cv::CMP_NE);
                cv::bitwise_and(mask, binary, mask);
            }
            return mask;
        }

        void OrbFeatureExtractor::initialiseLevels(const cv::Size& size) {
            if(pyramidBaseSize_==size && !levelSizes_.empty()) return;
            pyramidBaseSize_=size; levelSizes_.clear(); levelScales_.clear(); levelBaseX_.clear(); levelBaseY_.clear();
            staticMasksReady_=false; staticSafeMasks_.clear(); staticSamplingMasks_.clear(); staticInvalidIntegrals_.clear();
            staticCellMasks_.clear();
            for(int level=0;level<options_.pyramidLevels;++level) {
                const double scale=std::pow(static_cast<double>(options_.scaleFactor),level);
                const cv::Size layer(cvRound(size.width/scale),cvRound(size.height/scale));
                // Keep all configured level transforms for public samplingWarp(),
                // even when a small image cannot support detection at a high level.
                levelSizes_.push_back(layer); levelScales_.push_back(scale);
                levelBaseX_.push_back(layer.width>0?double(size.width)/layer.width:0);
                levelBaseY_.push_back(layer.height>0?double(size.height)/layer.height:0);
            }
            imageScratch_.resize(levelSizes_.size()); intensityScratch_.resize(levelSizes_.size());
            fastScoreScratch_.resize(levelSizes_.size());
            geometryMemoIndexScratch_.resize(levelSizes_.size());
        }
        void OrbFeatureExtractor::buildMasks(const cv::Mat& baseMask,Pyramid& pyramid) const {
            // Camera-aware descriptors validate their actual warped support separately.
            // A fixed 25px erosion would discard valid compressed patches near mask edges.
            const int radius = isCameraAware() ? 3 : kMaskRadius;
            const cv::Mat kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(2*radius+1,2*radius+1));
            cv::Mat previousMask=baseMask;
            for (size_t level = 0; level < levelSizes_.size(); ++level) {
                const cv::Size size=levelSizes_[level];
                if (size.width <= 2 * options_.edgeThreshold + 6 || size.height <= 2 * options_.edgeThreshold + 6) break;
                cv::Mat levelMask, safe;
                // A level pixel is valid only if its interpolation support was valid.
                // Resize in float: uint8 rounding can turn a partly invalid support back into 255.
                cv::Mat floatingMask, weights;
                previousMask.convertTo(floatingMask, CV_32F, 1.0 / 255.0);
                cv::resize(floatingMask, weights, size, 0, 0, cv::INTER_LINEAR);
                cv::compare(weights, 1.0, levelMask, cv::CMP_GE);
                cv::erode(levelMask, safe, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
                const int edge = options_.edgeThreshold;
                safe.rowRange(0, edge).setTo(0); safe.rowRange(size.height - edge, size.height).setTo(0);
                safe.colRange(0, edge).setTo(0); safe.colRange(size.width - edge, size.width).setTo(0);
                const int width=size.width-2*edge,height=size.height-2*edge;
                cv::Mat cells((height+options_.cellSize-1)/options_.cellSize,(width+options_.cellSize-1)/options_.cellSize,CV_8UC1);
                for(int row=0;row<cells.rows;++row) for(int col=0;col<cells.cols;++col) {
                    const cv::Rect core(edge+col*options_.cellSize,edge+row*options_.cellSize,
                        std::min(options_.cellSize,width-col*options_.cellSize),std::min(options_.cellSize,height-row*options_.cellSize));
                    cells.at<uchar>(row,col)=cv::countNonZero(safe(core))?255:0;
                }
                pyramid.cellMasks.push_back(cells);
                pyramid.safeMasks.push_back(safe); pyramid.samplingMasks.push_back(levelMask);
                pyramid.invalidIntegrals.push_back(options_.descriptorMode==OrbExtractorOptions::DescriptorMode::Standard?
                    cv::Mat():detail::invalidMaskIntegral(levelMask));
                previousMask = levelMask;
            }
        }
        void OrbFeatureExtractor::prepareStaticCaches() {
            if(cameraValidMask_.empty()) throw std::logic_error("ORB startup preparation requires a calibrated camera");
            initialiseLevels(cameraValidMask_.size());
            if(!staticMasksReady_) {
                // Only calibration and mask data are read during startup.
                Pyramid pyramid; buildMasks(cameraValidMask_,pyramid);
                staticSafeMasks_=std::move(pyramid.safeMasks);
                staticSamplingMasks_=std::move(pyramid.samplingMasks);
                staticInvalidIntegrals_=std::move(pyramid.invalidIntegrals);
                staticCellMasks_=std::move(pyramid.cellMasks);
                staticMasksReady_=true;
            }
#ifdef CGMLI_ENABLE_CUDA_ORB
            prepareCuda();
#endif
        }
#ifdef CGMLI_ENABLE_CUDA_ORB
        void OrbFeatureExtractor::disableCuda(const char* reason) {
            LOG(WARNING)<<"[ORB CUDA] CPU fallback for camera "<<camera_->id()<<": "<<reason;
            cuda_.reset();
            cudaDisabled_=true;
        }

        void OrbFeatureExtractor::prepareCuda() {
            if(cuda_ || cudaDisabled_ || !options_.useCuda || !isCameraAware() || !options_.layerFast ||
                options_.samplingMode!=OrbExtractorOptions::SamplingMode::SparseBox || staticSafeMasks_.empty()) return;
            const auto size=staticSafeMasks_.front().size();
            if(size.width>4096 || size.height>4096 ||
                size_t(size.width)*size.height>size_t(std::numeric_limits<int>::max()/255)) {
                disableCuda("unsupported image size");
                return;
            }
            std::vector<cv::Mat> integralStorage(staticSafeMasks_.size());
            std::vector<detail::OrbCudaLevel> levels;
            levels.reserve(staticSafeMasks_.size());
            for(size_t i=0;i<staticSafeMasks_.size();++i) {
                staticInvalidIntegrals_[i].convertTo(integralStorage[i],CV_32S);
                levels.push_back({levelSizes_[i].width,levelSizes_[i].height,integralStorage[i].ptr<int>()});
            }
            try {
                cuda_=std::make_unique<detail::OrbCuda>(levels,options_.edgeThreshold,options_.cellSize,
                    options_.minimumFastThreshold,&detail::kSparseOrbPattern.points[0][0],
                    &detail::kSparseOrbPattern.pairs[0][0],options_.maxFeatures);
            } catch(const detail::OrbCudaError& error) { disableCuda(error.what()); }
        }
#endif
        void OrbFeatureExtractor::buildPyramid(const cv::Mat& gray,cv::InputArray externalMask,Pyramid& pyramid) {
            initialiseLevels(gray.size()); pyramid={};
            const bool reusable=externalMask.empty();
            const bool rebuild=!reusable || !staticMasksReady_;
            lastTiming_.maskCacheHit=reusable && staticMasksReady_;
            cv::Mat previous=gray;
            for(size_t level=0;level<levelSizes_.size();++level) {
                const cv::Size size=levelSizes_[level];
                if(size.width<=2*options_.edgeThreshold+6 || size.height<=2*options_.edgeThreshold+6) break;
                cv::Mat& image=imageScratch_[level];
                if(level==0) image=gray;
                else cv::resize(previous,image,size,0,0,cv::INTER_LINEAR_EXACT);
                pyramid.images.push_back(image); previous=image;
            }
            if(rebuild) buildMasks(combinedMask(gray,externalMask),pyramid);
            if(reusable) {
                if(rebuild) {
                    staticSafeMasks_=pyramid.safeMasks; staticSamplingMasks_=pyramid.samplingMasks;
                    staticInvalidIntegrals_=pyramid.invalidIntegrals; staticMasksReady_=true;
                    staticCellMasks_=pyramid.cellMasks;
                } else {
                    pyramid.safeMasks=staticSafeMasks_; pyramid.samplingMasks=staticSamplingMasks_;
                    pyramid.invalidIntegrals=staticInvalidIntegrals_;
                    pyramid.cellMasks=staticCellMasks_;
                }
            }
        }

        std::vector<int> OrbFeatureExtractor::levelBudgets(int count) const {
            if (count == 0) return {};
            std::vector<int> budgets(count, 0);
            std::vector<std::pair<double, int>> remainders;
            double totalWeight = 0;
            for (int level = 0; level < count; ++level) totalWeight += std::pow(options_.scaleFactor, -level);
            int assigned = 0;
            for (int level = 0; level < count; ++level) {
                const double exact = options_.maxFeatures * std::pow(options_.scaleFactor, -level) / totalWeight;
                budgets[level] = static_cast<int>(std::floor(exact)); assigned += budgets[level];
                remainders.emplace_back(exact - budgets[level], level);
            }
            std::sort(remainders.begin(), remainders.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
                });
            for (int i = 0; i < options_.maxFeatures - assigned; ++i) ++budgets[remainders[i].second];
            return budgets;
        }

        int OrbFeatureExtractor::directionBin(const cv::Point2f& pixel) const {
            if (!options_.balanceBearings) return 0;
            Eigen::Vector3d bearing;
            Eigen::Matrix<double,2,3> unusedJacobian;
            if(!rayDirections_.empty() && geometryAt(pixel,bearing,unusedJacobian)) return directionBin(bearing);
            if (!camera_->backProject(Eigen::Vector2d(pixel.x, pixel.y), &bearing) || !bearing.allFinite()) return -1;
            const double norm = bearing.stableNorm();
            if (!std::isfinite(norm) || norm < 1e-12) return -1;
            bearing /= norm;
            return directionBin(bearing);
        }
        int OrbFeatureExtractor::directionBin(const Eigen::Vector3d& bearing) const {
            if(!options_.balanceBearings) return 0;
            const double azimuth = std::atan2(bearing.y(), bearing.x()) + kPi;
            const int a = std::clamp(static_cast<int>(azimuth / (2 * kPi) * options_.azimuthBins), 0, options_.azimuthBins - 1);
            const int z = std::clamp(static_cast<int>((bearing.z() + 1) * 0.5 * options_.zBins), 0, options_.zBins - 1);
            return z * options_.azimuthBins + a;
        }

        float OrbFeatureExtractor::orientation(const cv::Mat& image, const cv::Point2f& pixel) const {
            const int cx = cvRound(pixel.x), cy = cvRound(pixel.y);
            double momentX = 0, momentY = 0;
            // Independently implemented intensity-centroid orientation on a circular patch.
            for (int dy = -kPatchRadius; dy <= kPatchRadius; ++dy) {
                const int halfWidth = static_cast<int>(std::floor(std::sqrt(kPatchRadius * kPatchRadius - dy * dy)));
                const uchar* row = image.ptr<uchar>(cy + dy);
                for (int dx = -halfWidth; dx <= halfWidth; ++dx) {
                    const double value = row[cx + dx]; momentX += dx * value; momentY += dy * value;
                }
            }
            return cv::fastAtan2(static_cast<float>(momentY), static_cast<float>(momentX));
        }

        cv::Point2f OrbFeatureExtractor::levelToBase(const cv::Point2f& pixel, int octave) const {
            const double scale = levelScales_.at(size_t(octave));
            if (options_.descriptorMode == OrbExtractorOptions::DescriptorMode::Standard)
                return pixel * static_cast<float>(scale);
            const double sx=levelBaseX_.at(size_t(octave)),sy=levelBaseY_.at(size_t(octave));
            // OpenCV resize uses pixel-centre coordinates, including the half-pixel offset.
            return cv::Point2f(static_cast<float>((pixel.x + 0.5) * sx - 0.5),
                static_cast<float>((pixel.y + 0.5) * sy - 0.5));
        }
        cv::Point2f OrbFeatureExtractor::baseToLevel(const cv::Point2f& pixel, int octave) const {
            const double scale = levelScales_.at(size_t(octave));
            if (options_.descriptorMode == OrbExtractorOptions::DescriptorMode::Standard)
                return pixel * static_cast<float>(1.0 / scale);
            const double sx=levelBaseX_.at(size_t(octave)),sy=levelBaseY_.at(size_t(octave));
            return cv::Point2f(static_cast<float>((pixel.x + 0.5) / sx - 0.5),
                static_cast<float>((pixel.y + 0.5) / sy - 0.5));
        }

        void OrbFeatureExtractor::detectImpl(const Pyramid& pyramid,std::vector<cv::KeyPoint>& keypoints,
            std::vector<KeypointGeometry>& retainedGeometry) {
            const auto milliseconds=[](const auto& start) { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); };
            const auto& images=pyramid.images; const auto& masks=pyramid.safeMasks;
            keypoints.clear(); retainedGeometry.clear();
            const auto betterCandidate=[](const Candidate& a,const Candidate& b) { return betterResponse(a.keypoint,b.keypoint); };
            const auto budgets = levelBudgets(static_cast<int>(images.size()));
            const int bins = options_.balanceBearings ? options_.azimuthBins * options_.zBins : 1;
            std::vector<int> binOccupancy(bins, 0);
            for (std::size_t level = 0; level < images.size(); ++level) {
                const cv::Mat& image = images[level]; const cv::Mat& mask = masks[level];
                const auto imageSize=mask.size();
                const int edge = options_.edgeThreshold, width = imageSize.width - 2 * edge, height = imageSize.height - 2 * edge;
                const int cols = (width + options_.cellSize - 1) / options_.cellSize;
                const int rows = (height + options_.cellSize - 1) / options_.cellSize;
                const int cellCount = cols * rows;
                if(budgets[level]==0) continue;
                if(!cv::countNonZero(pyramid.cellMasks[level])) {
                    lastTiming_.emptyMaskCells+=size_t(cellCount); continue;
                }
                const int targetPerCell = std::min(options_.maxFeaturesPerCell,
                    std::max(1, budgets[level] / cellCount + (budgets[level] % cellCount != 0)));
                const float scale = static_cast<float>(levelScales_[level]);
                const cv::Mat& invalidIntegral=pyramid.invalidIntegrals[level];
                auto& candidates=candidateScratch_; candidates.clear();
                candidates.reserve(size_t(cellCount)*options_.candidatesPerCell);
                auto& fastScratch=cellFastScratch_;
                auto& layerCells=layerCellScratch_;
                if(layerCells.size()<size_t(cellCount)) layerCells.resize(size_t(cellCount));
                for(auto& cell:layerCells) cell.clear();
                auto& lowCells=lowCellScratch_; if(lowCells.size()<size_t(cellCount)) lowCells.resize(size_t(cellCount));
                std::vector<uint8_t> lowReady(size_t(cellCount),0);
                auto& geometryMemo=geometryMemoScratch_; geometryMemo.clear();
                auto& memoIndices=geometryMemoIndexScratch_[level];
                if(options_.adaptiveGeometryCandidates) { memoIndices.create(imageSize,CV_32SC1); memoIndices.setTo(-1); }
                int geometryQuota=options_.adaptiveGeometryCandidates?std::min(options_.candidatesPerCell,std::max(2,targetPerCell)):options_.candidatesPerCell;
#ifdef CGMLI_ENABLE_CUDA_ORB
                if(pyramid.cuda) {
                    const auto bucketStarted=std::chrono::steady_clock::now();
                    for(int cell=0;cell<cellCount;++cell) lowCells[cell].clear();
                    const auto* corners=pyramid.cuda->corners(level);
                    const int count=pyramid.cuda->cornerCount(level);
                    for(int k=0;k<count;++k) {
                        const auto corner=corners[k];
                        const int cell=(corner.y()-edge)/options_.cellSize*cols+(corner.x()-edge)/options_.cellSize;
                        const cv::KeyPoint point(float(corner.x()),float(corner.y()),7,-1,float(corner.score()));
                        if(corner.score()>=options_.initialFastThreshold) layerCells[cell].push_back(point);
                        if(options_.minimumFastThreshold<options_.initialFastThreshold) lowCells[cell].push_back(point);
                    }
                    lastTiming_.fastBucketMs+=milliseconds(bucketStarted);
                } else
#endif
                if(options_.layerFast) {
                    const auto fastStarted=std::chrono::steady_clock::now();
                    auto& scores=fastScoreScratch_[level]; scores.create(image.size(),CV_8UC1); scores.setTo(0);
                    // Share the HIGH-threshold pass. Low-threshold corners are
                    // requested only by cells that fail their geometric quota;
                    // scanning the entire layer at the low threshold wastes work
                    // on textured regions. Cell-local NMS remains the reference.
                    const auto rawStarted=std::chrono::steady_clock::now();
                    rawFastScratch_.clear();
                    if(options_.boundaryFast) {
                        // Only borders (and their immediate neighbours) require raw
                        // scores. Each strip includes the FAST circle's 3px halo.
                        // Overlapping horizontal/vertical strips are deduplicated below.
                        const auto scanStrip=[&](const cv::Rect& core) {
                            const cv::Rect roi(core.x-3,core.y-3,core.width+6,core.height+6);
                            cv::FAST(image(roi),stripFastScratch_,options_.initialFastThreshold,false,cv::FastFeatureDetector::TYPE_9_16);
                            ++lastTiming_.fastCalls; lastTiming_.fastRawPixels+=size_t(roi.area());
                            for(auto p:stripFastScratch_) { p.pt+=cv::Point2f(float(roi.x),float(roi.y)); rawFastScratch_.push_back(p); }
                        };
                        for(int boundary=edge;boundary<edge+width;boundary+=options_.cellSize) {
                            const int left=std::max(edge,boundary-2),right=std::min(edge+width,boundary+2);
                            scanStrip(cv::Rect(left,edge,right-left,height));
                        }
                        scanStrip(cv::Rect(std::max(edge,edge+width-2),edge,std::min(2,width),height));
                        for(int boundary=edge;boundary<edge+height;boundary+=options_.cellSize) {
                            const int top=std::max(edge,boundary-2),bottom=std::min(edge+height,boundary+2);
                            scanStrip(cv::Rect(edge,top,width,bottom-top));
                        }
                        scanStrip(cv::Rect(edge,std::max(edge,edge+height-2),width,std::min(2,height)));
                    } else {
                        cv::FAST(image,rawFastScratch_,options_.initialFastThreshold,false,cv::FastFeatureDetector::TYPE_9_16);
                        ++lastTiming_.fastCalls; lastTiming_.fastRawPixels+=image.total();
                    }
                    lastTiming_.fastRawScanMs+=milliseconds(rawStarted);
                    lastTiming_.fastRawCorners+=rawFastScratch_.size();
                    const auto nmsStarted=std::chrono::steady_clock::now();
                    cv::FAST(image,maximaFastScratch_,options_.initialFastThreshold,true,cv::FastFeatureDetector::TYPE_9_16);
                    lastTiming_.fastNmsScanMs+=milliseconds(nmsStarted);
                    ++lastTiming_.fastCalls;
                    const auto boundaryStarted=std::chrono::steady_clock::now();
                    size_t uniqueBoundary=0;
                    for(const auto point:rawFastScratch_) {
                        const int x=cvRound(point.pt.x),y=cvRound(point.pt.y);
                        if(x<edge || x>=edge+width || y<edge || y>=edge+height) continue;
                        const int left=edge+(x-edge)/options_.cellSize*options_.cellSize;
                        const int top=edge+(y-edge)/options_.cellSize*options_.cellSize;
                        const int right=std::min(left+options_.cellSize,edge+width),bottom=std::min(top+options_.cellSize,edge+height);
                        if((x-left<=1 || right-1-x<=1 || y-top<=1 || bottom-1-y<=1) && !scores.at<uchar>(y,x)) {
                            scores.at<uchar>(y,x)=uchar(fastNineScore(image,x,y));
                            rawFastScratch_[uniqueBoundary++]=point;
                        }
                    }
                    rawFastScratch_.resize(uniqueBoundary);
                    lastTiming_.fastBoundaryMs+=milliseconds(boundaryStarted);
                    const auto bucketStarted=std::chrono::steady_clock::now();
                    for(const auto& point:maximaFastScratch_) {
                        const int x=cvRound(point.pt.x),y=cvRound(point.pt.y);
                        if(x<edge || x>=edge+width || y<edge || y>=edge+height) continue;
                        const int col=(x-edge)/options_.cellSize,row=(y-edge)/options_.cellSize;
                        const int left=edge+col*options_.cellSize,top=edge+row*options_.cellSize;
                        const int right=std::min(left+options_.cellSize,edge+width),bottom=std::min(top+options_.cellSize,edge+height);
                        if(x>left && x<right-1 && y>top && y<bottom-1)
                            layerCells[size_t(row)*cols+col].push_back(point);
                    }
                    for(const auto& point:rawFastScratch_) {
                        const int x=cvRound(point.pt.x),y=cvRound(point.pt.y);
                        if(x<edge || x>=edge+width || y<edge || y>=edge+height) continue;
                        const int score=scores.at<uchar>(y,x);
                        if(score<options_.initialFastThreshold) continue;
                        const int col=(x-edge)/options_.cellSize,row=(y-edge)/options_.cellSize;
                        const int left=edge+col*options_.cellSize,top=edge+row*options_.cellSize;
                        const int right=std::min(left+options_.cellSize,edge+width),bottom=std::min(top+options_.cellSize,edge+height);
                        if(x!=left && x!=right-1 && y!=top && y!=bottom-1) continue;
                        bool maximum=true;
                        // Legacy ROI FAST has a 3-pixel circle halo and zero score
                        // outside the core. Preserve that cell-local strict NMS;
                        // global image NMS would change corners on grid boundaries.
                        for(int yy=std::max(y-1,top);maximum && yy<=std::min(y+1,bottom-1);++yy)
                            for(int xx=std::max(x-1,left);xx<=std::min(x+1,right-1);++xx)
                                if((xx!=x || yy!=y) && score<=scores.at<uchar>(yy,xx)) { maximum=false; break; }
                        if(maximum) {
                            auto corner=point; corner.response=float(score);
                            layerCells[size_t(row)*cols+col].push_back(corner);
                        }
                    }
                    lastTiming_.fastBucketMs+=milliseconds(bucketStarted);
                    lastTiming_.fastMs+=milliseconds(fastStarted);
                }
                // Coarse regions are sized by the final layer budget, not the
                // much larger detector grid. Keep high-threshold detection over
                // EVERY fine cell, then spend low-threshold work on coverage gaps.
                const int coarseCols=std::min(cols,std::max(1,int(std::ceil(std::sqrt(double(budgets[level])*width/height)))));
                const int coarseRows=std::min(rows,std::max(1,(budgets[level]+coarseCols-1)/coarseCols));
                std::vector<size_t> coarseSupport(size_t(coarseCols)*coarseRows,0);
                const auto region=[&](int row,int col) {
                    const int x=std::min(coarseCols-1,int((col+0.5)*coarseCols/cols));
                    const int y=std::min(coarseRows-1,int((row+0.5)*coarseRows/rows));
                    return size_t(y)*coarseCols+x;
                };
                std::vector<size_t> deferred;
                std::vector<int> highValid(size_t(cellCount),0);
                std::vector<uint8_t> highHadCandidates(size_t(cellCount),0);
                auto& retainedPixels=retainedPixelScratch_;
                if(retainedPixels.size()<size_t(cellCount)) retainedPixels.resize(size_t(cellCount));
                for(auto& cell:retainedPixels) cell.clear();
                size_t selectableCapacity=0;
                const auto processCell=[&](int row,int col,bool lowOnly,bool expanding=false) {
                    // Cached from the actual safe mask, including external-mask paths.
                    // No admissible center means neither threshold can yield a feature.
                    if(!pyramid.cellMasks[level].at<uchar>(row,col)) { ++lastTiming_.emptyMaskCells; return; }
                    const cv::Rect core(edge + col * options_.cellSize, edge + row * options_.cellSize,
                        std::min(options_.cellSize, width - col * options_.cellSize),
                        std::min(options_.cellSize, height - row * options_.cellSize));
                    const cv::Rect roi(core.x - 3, core.y - 3, core.width + 6, core.height + 6);
                    auto localCandidates = [&](int threshold,int& geometricallyValid) -> const std::vector<Candidate>& {
                        auto& points=fastScratch; points.clear(); auto& valid=validCandidateScratch_; valid.clear();
                        valid.reserve(size_t(options_.candidatesPerCell));
                        const bool sharedCandidates=options_.layerFast && threshold==options_.initialFastThreshold;
                        const size_t cell=size_t(row)*cols+col;
                        const bool cachedLow=options_.adaptiveGeometryCandidates && threshold==options_.minimumFastThreshold
                            && options_.minimumFastThreshold<options_.initialFastThreshold;
                        if(pyramid.cuda) {
                            // Device candidates use layer coordinates. Threshold
                            // selection and per-cell quotas still run here.
                            points=sharedCandidates?layerCells[cell]:lowCells[cell];
                            if(cachedLow) lowReady[cell]=1;
                        } else if(sharedCandidates) {
                            for(const auto& point:layerCells[size_t(row)*cols+col]) if(point.response>=threshold) points.push_back(point);
                        } else if(cachedLow && lowReady[cell]) {
                            points=lowCells[cell];
                        } else {
                            const auto fastStarted=std::chrono::steady_clock::now();
                            cv::FAST(image(roi), points, threshold, true, cv::FastFeatureDetector::TYPE_9_16);
                            const double elapsed=milliseconds(fastStarted);
                            ++lastTiming_.fastCalls; lastTiming_.fastMs+=elapsed;
                            if(options_.layerFast) { ++lastTiming_.fastFallbackCalls; lastTiming_.fastFallbackMs+=elapsed; }
                            if(cachedLow) { lowCells[cell]=points; lowReady[cell]=1; }
                        }
                        lastTiming_.fastCandidates+=points.size();
                        // The response order is independent of the geometry gate.
                        // Walk strongest-first and replenish rejected candidates until
                        // the current valid batch is filled or FAST is exhausted.
                        // Capacity replenishment can grow this batch to the original cap.
                        const auto sortStarted=std::chrono::steady_clock::now();
                        std::sort(points.begin(),points.end(),betterResponse);
                        lastTiming_.candidateSortMs+=milliseconds(sortStarted);
                        const auto filteringStarted=std::chrono::steady_clock::now();
                        for (auto point : points) {
                            if(!pyramid.cuda && !sharedCandidates) point.pt += cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
                            if (point.pt.x < core.x || point.pt.x >= core.x + core.width ||
                                point.pt.y < core.y || point.pt.y >= core.y + core.height || !safePoint(mask, point.pt)) continue;
                            int memoIndex=-1;
                            if(options_.adaptiveGeometryCandidates) {
                                int& index=memoIndices.at<int>(cvRound(point.pt.y),cvRound(point.pt.x));
                                if(index>=0) {
                                    ++lastTiming_.geometryMemoHits;
                                    const auto& cached=geometryMemo[size_t(index)];
                                    if(cached.geometricallyValid) ++geometricallyValid;
                                    if(cached.valid) {
                                        valid.push_back(cached.candidate);
                                        if(valid.size()>=size_t(geometryQuota)) break;
                                    }
                                    continue;
                                }
                                // Cache failures too: low-FAST/replenishment must
                                // not repeatedly evaluate an unusable warped patch.
                                memoIndex=int(geometryMemo.size()); index=memoIndex; geometryMemo.emplace_back();
                            }
                            KeypointGeometry geometry; geometry.levelPixel=point.pt;
                            cv::KeyPoint base=point;
                            base.pt=levelToBase(point.pt,int(level)); base.octave=int(level); base.size=kPatchSize*scale;
                            // Description uses the rounded base-image keypoint, as
                            // it did before caching, rather than the raw FAST pixel.
                            geometry.levelPixel=baseToLevel(base.pt,int(level));
                            if (options_.descriptorMode != OrbExtractorOptions::DescriptorMode::Standard) {
                                base.angle = isCameraAware() ? 0.0f : orientation(image, point.pt);
                                auto& warp=geometry.warp; auto& aligned=geometry.gravityAligned;
                                Eigen::Vector3d ray; ++lastTiming_.geometryEvaluations;
                                if (!samplingWarpImpl(base, warp, aligned,&ray)) continue;
                                // A pole's texture rotation is not known until its patch is rectified.
                                // Enclose all possible rotations so unusable candidates do not consume quotas.
                                const bool pole = isCameraAware() && !aligned;
                                const Eigen::Vector2d centre(point.pt.x, point.pt.y);
                                if (options_.samplingMode == OrbExtractorOptions::SamplingMode::SparseBox) {
                                    if (!detail::sparseSupportValid(imageSize, invalidIntegral, centre, warp, pole)) continue;
                                } else {
                                    const double support = pole ? std::sqrt(2.0) * detail::kOrbTileRadius : detail::kOrbTileRadius;
                                    if (!detail::warpedSupportValid(image.size(), invalidIntegral, centre, warp, support)) continue;
                                }
                                base.angle = isCameraAware() && !aligned ? -1.0f :
                                    cv::fastAtan2(static_cast<float>(warp(1, 0)), static_cast<float>(warp(0, 0)));
                                geometry.bin=directionBin(ray);
                            } else geometry.bin=directionBin(base.pt);
                            ++geometricallyValid;
                            if(memoIndex>=0) geometryMemo[size_t(memoIndex)].geometricallyValid=true;
                            if(geometry.bin<0) continue;
                            valid.push_back({base,row*cols+col,geometry.bin,geometry});
                            if(options_.adaptiveGeometryCandidates) {
                                auto& cached=geometryMemo[size_t(memoIndex)]; cached.valid=true; cached.candidate=valid.back();
                            }
                            if(valid.size()>=size_t(geometryQuota)) break;
                        }
                        lastTiming_.candidateFilterMs+=milliseconds(filteringStarted);
                        // Caller consumes this scratch before requesting another cell.
                        return valid;
                        };
                    int geometricallyValid=0; const size_t cell=size_t(row)*cols+col;
                    auto& points=combinedCandidateScratch_;
                    points=localCandidates(lowOnly?options_.minimumFastThreshold:options_.initialFastThreshold,geometricallyValid);
                    if(lowOnly && !expanding) {
                        ++lastTiming_.quotaFallbackCells;
                        if(highValid[cell]==0 && highHadCandidates[cell]) ++lastTiming_.geometryFallbackCells;
                    } else if(!lowOnly && !expanding) {
                        highValid[cell]=geometricallyValid;
                        highHadCandidates[cell]=!fastScratch.empty();
                        if(geometricallyValid<targetPerCell && options_.minimumFastThreshold<options_.initialFastThreshold) {
                            if(options_.budgetedFastFallback) deferred.push_back(cell);
                            else {
                                ++lastTiming_.quotaFallbackCells;
                                if(geometricallyValid==0 && !fastScratch.empty()) ++lastTiming_.geometryFallbackCells;
                                int lowerValid=0; const auto& lower=localCandidates(options_.minimumFastThreshold,lowerValid);
                                points.insert(points.end(),lower.begin(),lower.end());
                            }
                        }
                    }
                    std::sort(points.begin(), points.end(), betterCandidate);
                    auto& seen=retainedPixels[cell]; seen.reserve(size_t(options_.candidatesPerCell));
                    int retained = int(seen.size());
                    for (const auto& point : points) {
                        if(retained>=options_.candidatesPerCell) break;
                        const auto pixel=std::make_pair(cvRound(point.geometry.levelPixel.x),cvRound(point.geometry.levelPixel.y));
                        if(std::find(seen.begin(),seen.end(),pixel)!=seen.end()) continue;
                        seen.push_back(pixel);
                        candidates.push_back(point);
                        ++coarseSupport[region(row,col)];
                        if(retained<options_.maxFeaturesPerCell) ++selectableCapacity;
                        if (++retained == options_.candidatesPerCell) break;
                    }
                };
                for(int row=0;row<rows;++row) for(int col=0;col<cols;++col) processCell(row,col,false);
                std::vector<uint8_t> attempted(deferred.size(),0);
                for(size_t i=0;i<deferred.size();++i) {
                    const int row=int(deferred[i])/cols,col=int(deferred[i])%cols;
                    if(coarseSupport[region(row,col)]==0) { processCell(row,col,true); attempted[i]=1; }
                }
                // Replenish high-threshold batches before spending low-threshold
                // work merely because the first small batch underestimated capacity.
                // Never truncate to the final 800 points before geometry validation.
                while(options_.adaptiveGeometryCandidates && selectableCapacity<size_t(budgets[level]) && geometryQuota<options_.candidatesPerCell) {
                    geometryQuota=std::min(options_.candidatesPerCell,2*geometryQuota); ++lastTiming_.geometryExpansionRounds;
                    for(int row=0;row<rows;++row) for(int col=0;col<cols;++col) processCell(row,col,false,true);
                    for(size_t cell=0;cell<lowReady.size();++cell) if(lowReady[cell]) processCell(int(cell)/cols,int(cell)%cols,true,true);
                }
                // If coverage-driven requests still leave the layer short, retain
                // a deterministic fill pass rather than sacrificing feature count.
                for(size_t i=0;i<deferred.size() && selectableCapacity<size_t(budgets[level]);++i)
                    if(!attempted[i]) { processCell(int(deferred[i])/cols,int(deferred[i])%cols,true); attempted[i]=1; }
                for(auto value:attempted) if(!value) ++lastTiming_.budgetSkippedCells;
                const auto balanceStarted=std::chrono::steady_clock::now();
                std::vector<int> cellOccupancy(cols * rows, 0);
                struct Ranked { int index,binLoad,cellLoad; };
                const auto worse=[&](const Ranked& a,const Ranked& b) {
                    if(a.binLoad!=b.binLoad) return a.binLoad>b.binLoad;
                    if(a.cellLoad!=b.cellLoad) return a.cellLoad>b.cellLoad;
                    const auto& ka=candidates[a.index].keypoint; const auto& kb=candidates[b.index].keypoint;
                    if(betterResponse(ka,kb)) return false;
                    if(betterResponse(kb,ka)) return true;
                    return a.index>b.index;
                };
                // Candidates sharing (bearing bin, image cell) have identical
                // occupancy keys. Only their strongest unselected representative
                // needs a heap entry. This preserves the exact global ordering,
                // but avoids refreshing all 16 candidates when one load changes.
                auto& order=candidateOrderScratch_; order.resize(candidates.size()); std::iota(order.begin(),order.end(),0);
                auto& next=groupNextScratch_; next.assign(candidates.size(),-1);
                std::vector<Ranked> heap; heap.reserve(candidates.size());
                if(options_.groupedBalancing) {
                    std::sort(order.begin(),order.end(),[&](int a,int b) {
                        if(candidates[a].bin!=candidates[b].bin) return candidates[a].bin<candidates[b].bin;
                        if(candidates[a].cell!=candidates[b].cell) return candidates[a].cell<candidates[b].cell;
                        if(betterCandidate(candidates[a],candidates[b])) return true;
                        if(betterCandidate(candidates[b],candidates[a])) return false;
                        return a<b;
                    });
                    // Sparse groups use O(candidate count) storage, never a dense
                    // number-of-bins x number-of-cells allocation.
                    int previous=-1;
                    for(int index:order) {
                        const auto& c=candidates[index];
                        if(previous>=0 && c.bin==candidates[previous].bin && c.cell==candidates[previous].cell) next[previous]=index;
                        else heap.push_back({index,options_.balanceBearings?binOccupancy[c.bin]:0,0});
                        previous=index;
                    }
                } else for(size_t i=0;i<candidates.size();++i) heap.push_back({int(i),options_.balanceBearings?binOccupancy[candidates[i].bin]:0,0});
                lastTiming_.balanceHeapPushes+=heap.size();
                std::priority_queue<Ranked,std::vector<Ranked>,decltype(worse)> ranking(worse,std::move(heap));
                for (int selected = 0; selected < budgets[level]; ++selected) {
                    int best = -1;
                    // Occupancy only increases: old heap keys are lower bounds.
                    // Refresh stale heads lazily, preserving the original global
                    // bin/cell/response ordering without rescanning every candidate.
                    while(!ranking.empty()) {
                        auto head=ranking.top(); ranking.pop(); const auto& c=candidates[head.index];
                        if(cellOccupancy[c.cell]>=options_.maxFeaturesPerCell) continue;
                        const int bin=options_.balanceBearings?binOccupancy[c.bin]:0,cell=cellOccupancy[c.cell];
                        if(head.binLoad!=bin || head.cellLoad!=cell) { ranking.push({head.index,bin,cell}); ++lastTiming_.balanceHeapPushes; continue; }
                        best=head.index; break;
                    }
                    if (best < 0) break;
                    const auto& selectedCandidate = candidates[best];
                    ++cellOccupancy[selectedCandidate.cell]; ++binOccupancy[selectedCandidate.bin];
                    if(next[best]>=0 && cellOccupancy[selectedCandidate.cell]<options_.maxFeaturesPerCell) {
                        ranking.push({next[best],options_.balanceBearings?binOccupancy[selectedCandidate.bin]:0,cellOccupancy[selectedCandidate.cell]});
                        ++lastTiming_.balanceHeapPushes;
                    }
                    auto point = selectedCandidate.keypoint;
                    if (options_.descriptorMode == OrbExtractorOptions::DescriptorMode::Standard)
                        point.angle = orientation(image, baseToLevel(point.pt, static_cast<int>(level)));
                    keypoints.push_back(point);
                    retainedGeometry.push_back(selectedCandidate.geometry);
                }
                lastTiming_.balanceMs+=milliseconds(balanceStarted);
            }
        }

        void OrbFeatureExtractor::computeImpl(const cv::Mat& gray,const Pyramid& pyramid,
            std::vector<cv::KeyPoint>& keypoints,cv::OutputArray descriptors,
            const std::vector<KeypointGeometry>* detectedGeometry) {
            const auto& images=pyramid.images; const auto& masks=pyramid.safeMasks;
            if (options_.descriptorMode != OrbExtractorOptions::DescriptorMode::Standard && !hasGravity_)
                throw std::logic_error("Set ORB camera gravity before describing each frame");
            std::vector<cv::KeyPoint> valid; valid.reserve(keypoints.size());
            std::vector<KeypointGeometry> geometry; geometry.reserve(keypoints.size());
            if(detectedGeometry && detectedGeometry->size()!=keypoints.size())
                throw std::logic_error("Detected keypoints and geometry rows differ");
            for (size_t k=0;k<keypoints.size();++k) {
                auto point=keypoints[k];
                if (point.octave < 0 || point.octave >= static_cast<int>(images.size())) continue;
                const float scale = static_cast<float>(levelScales_[size_t(point.octave)]);
                const cv::Point2f levelPixel = baseToLevel(point.pt, point.octave);
                if (!safePoint(masks[point.octave], levelPixel)) continue;
                // Camera-aware gravity poles estimate orientation on a rectified patch.
                point.angle = isCameraAware() ? 0.0f : orientation(images[point.octave], levelPixel);
                point.size = kPatchSize * scale;
                KeypointGeometry prepared; prepared.levelPixel=levelPixel;
                if(options_.descriptorMode!=OrbExtractorOptions::DescriptorMode::Standard) {
                    if(detectedGeometry) { prepared=(*detectedGeometry)[k]; ++lastTiming_.geometryCacheHits; }
                    else {
                        Eigen::Vector3d ray; ++lastTiming_.geometryEvaluations;
                        if(!samplingWarpImpl(point,prepared.warp,prepared.gravityAligned,&ray)) continue;
                        prepared.bin=directionBin(ray);
                    }
                } else prepared.bin=directionBin(point.pt);
                if(prepared.bin<0) continue;
                valid.push_back(point);
                geometry.push_back(prepared);
            }
            keypoints.swap(valid);
            if (keypoints.empty()) { descriptors.release(); return; }
#ifdef CGMLI_ENABLE_CUDA_ORB
            if(pyramid.cuda) { computeCuda(pyramid,keypoints,descriptors,geometry); return; }
#endif
            if (options_.descriptorMode != OrbExtractorOptions::DescriptorMode::Standard) {
                const auto& invalidIntegral=pyramid.invalidIntegrals;
                // Reuse allocations, NEVER pixel values: every described layer is
                // integrated again for this image, even on a static-mask cache hit.
                std::vector<bool> intensityReady(images.size(),false);
                std::vector<cv::KeyPoint> described; described.reserve(keypoints.size());
                cv::Mat result(static_cast<int>(keypoints.size()), 32, CV_8UC1, cv::Scalar(0));
                cv::Mat patch, filtered; // Reuse tile storage across features.
                for (size_t k=0;k<keypoints.size();++k) {
                    auto point=keypoints[k]; auto warp=geometry[k].warp; const bool aligned=geometry[k].gravityAligned;
                    const cv::Point2f levelPixel=geometry[k].levelPixel;
                    const Eigen::Vector2d centre(levelPixel.x, levelPixel.y);
                    const cv::Mat& image = images[point.octave];
                    const cv::Mat& integral = invalidIntegral[point.octave];
                    if (options_.samplingMode == OrbExtractorOptions::SamplingMode::SparseBox) {
                        if (!detail::sparseSupportValid(image.size(), integral, centre, warp)) continue;
                        cv::Mat& intensities = intensityScratch_[size_t(point.octave)];
                        if (!intensityReady[point.octave]) {
                            const bool compact=options_.compactGrayIntegral && image.total()<=size_t(std::numeric_limits<int>::max()/255);
                            cv::integral(image, intensities, compact?CV_32S:CV_64F); intensityReady[point.octave]=true;
                            if(compact) ++lastTiming_.grayIntegral32Builds;
                            ++lastTiming_.grayIntegralBuilds;
                        }
                        if (isCameraAware() && !aligned) {
                            double textureAngle;
                            if (!detail::sparseTextureOrientation(intensities, centre, warp, textureAngle)) continue;
                            warp = (warp * rotation(textureAngle)).eval();
                            if (!detail::sparseSupportValid(image.size(), integral, centre, warp)) continue;
                        }
                        std::array<uchar, 32> bits{};
                        if (!detail::sparseOrbDescriptor(intensities, centre, warp, bits,options_.batchDescriptorCoordinates)) continue;
                        point.angle = cv::fastAtan2(static_cast<float>(warp(1, 0)), static_cast<float>(warp(0, 0)));
                        std::copy(bits.begin(), bits.end(), result.ptr<uchar>(static_cast<int>(described.size())));
                        described.push_back(point);
                        continue; // Default path never allocates/filters a per-feature tile.
                    }
                    if (!detail::rectifyPatch(image, integral, centre, warp, patch)) continue;
                    if (isCameraAware() && !aligned) {
                        double textureAngle;
                        if (!detail::rectifiedTextureOrientation(patch, textureAngle)) continue;
                        warp = (warp * rotation(textureAngle)).eval();
                        // Resample from the source, avoiding a second interpolation of the first tile.
                        if (!detail::rectifyPatch(image, integral, centre, warp, patch)) continue;
                    }
                    detail::smoothRectifiedPatch(patch, filtered);
                    const int samplingScale = detail::patchSamplingScale(filtered);
                    const int patchCentre = detail::kOrbTileRadius * samplingScale;
                    std::array<uchar, 32> bits{};
                    for (int bit = 0; bit < 256; ++bit) {
                        const int* pair = detail::kOrbPattern[bit];
                        // Integer reference-plane coordinates; no extra warp or image angle here.
                        const float a = filtered.at<float>(pair[1] * samplingScale + patchCentre, pair[0] * samplingScale + patchCentre);
                        const float b = filtered.at<float>(pair[3] * samplingScale + patchCentre, pair[2] * samplingScale + patchCentre);
                        if (a < b) bits[bit / 8] |= static_cast<uchar>(1u << (bit % 8));
                    }
                    // Diagnostic image angle only. Warp already includes orientation;
                    // rotating by keypoint.angle again would apply gravity twice.
                    point.angle = cv::fastAtan2(static_cast<float>(warp(1, 0)), static_cast<float>(warp(0, 0)));
                    std::copy(bits.begin(), bits.end(), result.ptr<uchar>(static_cast<int>(described.size())));
                    described.push_back(point);
                }
                keypoints.swap(described);
                if (keypoints.empty()) descriptors.release();
                else result.rowRange(0, static_cast<int>(keypoints.size())).copyTo(descriptors);
                return;
            }
            // OpenCV's compute() does NOT compute orientation for provided keypoints.
            // It can reorder/drop points; keep its returned keypoints and rows together.
            auto orb = cv::ORB::create(options_.maxFeatures, options_.scaleFactor, options_.pyramidLevels,
                options_.edgeThreshold, 0, 2, cv::ORB::FAST_SCORE, kPatchSize, options_.initialFastThreshold);
            orb->compute(gray, keypoints, descriptors);
        }

#ifdef CGMLI_ENABLE_CUDA_ORB
        void OrbFeatureExtractor::computeCuda(const Pyramid& pyramid,std::vector<cv::KeyPoint>& keypoints,
            cv::OutputArray descriptors,const std::vector<KeypointGeometry>& geometry) {
            std::vector<detail::OrbCudaPatch> patches;
            std::vector<cv::KeyPoint> described;
            patches.reserve(keypoints.size()); described.reserve(keypoints.size());
            for(size_t i=0;i<keypoints.size();++i) {
                const auto& point=keypoints[i];
                const auto& prepared=geometry[i];
                const Eigen::Vector2d centre(prepared.levelPixel.x,prepared.levelPixel.y);
                const auto& warp=prepared.warp;
                if(!detail::sparseSupportValid(pyramid.safeMasks[point.octave].size(),
                    pyramid.invalidIntegrals[point.octave],centre,warp)) continue;
                patches.push_back({centre.x(),centre.y(),warp(0,0),warp(0,1),warp(1,0),warp(1,1),
                    point.octave,prepared.gravityAligned});
                described.push_back(point);
            }
            const auto* results=pyramid.cuda->describe(patches,options_.keepCudaDescriptorRows);
            cv::Mat result(static_cast<int>(patches.size()),32,CV_8UC1);
            size_t accepted=0;
            for(size_t i=0;i<patches.size();++i) {
                if(!results[i].valid) continue;
                auto point=described[i];
                point.angle=cv::fastAtan2(results[i].c,results[i].a);
                std::copy_n(results[i].bytes,32,result.ptr<uchar>(static_cast<int>(accepted)));
                described[accepted++]=point;
            }
            described.resize(accepted);
            keypoints.swap(described);
            if(!accepted) descriptors.release();
            else result.rowRange(0,static_cast<int>(accepted)).copyTo(descriptors);
            cudaDescriptorRows_=pyramid.cuda->descriptorRows(accepted);
        }
#endif

        void OrbFeatureExtractor::detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask) {
            cudaDescriptorRows_.reset();
            lastTiming_={};
            const cv::Mat gray = grayscale(image);
            if (gray.empty()) { keypoints.clear(); return; }
            if (options_.descriptorMode != OrbExtractorOptions::DescriptorMode::Standard && !hasGravity_)
                throw std::logic_error("Set ORB camera gravity before detecting each frame");
            Pyramid pyramid; std::vector<KeypointGeometry> geometry;
            buildPyramid(gray,mask,pyramid);
            detectImpl(pyramid,keypoints,geometry);
        }
        void OrbFeatureExtractor::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) {
            detectAndCompute(image, cv::noArray(), keypoints, descriptors, true);
        }
        void OrbFeatureExtractor::detectAndCompute(cv::InputArray image, cv::InputArray mask,
            std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
            bool useProvidedKeypoints) {
            cudaDescriptorRows_.reset();
            lastTiming_={}; const auto started=std::chrono::steady_clock::now();
            const auto milliseconds=[](const auto& start) { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); };
            const cv::Mat gray = grayscale(image);
            if (gray.empty()) { keypoints.clear(); if (descriptors.needed()) descriptors.release(); return; }
            if (options_.descriptorMode != OrbExtractorOptions::DescriptorMode::Standard && !hasGravity_)
                throw std::logic_error("Set ORB camera gravity before detecting/describing each frame");
            Pyramid pyramid; std::vector<KeypointGeometry> geometry;
#ifdef CGMLI_ENABLE_CUDA_ORB
            if(options_.useCuda && isCameraAware() && mask.empty() && gray.size()==cameraValidMask_.size()) {
                const bool cacheHit=staticMasksReady_;
                prepareStaticCaches();
                if(cuda_) {
                    auto working=useProvidedKeypoints?keypoints:std::vector<cv::KeyPoint>{};
                    try {
                        pyramid.cuda=cuda_.get();
                        pyramid.images.resize(staticSafeMasks_.size()); // Device images have no cv::Mat view.
                        pyramid.safeMasks=staticSafeMasks_;
                        pyramid.invalidIntegrals=staticInvalidIntegrals_;
                        pyramid.cellMasks=staticCellMasks_;
                        cuda_->begin(gray.ptr<uchar>(),gray.step,!useProvidedKeypoints,
                            options_.cudaEarlyIntegral && !useProvidedKeypoints && descriptors.needed());
                        const double fastWall=cuda_->timing().fastMs+cuda_->timing().cornersDownloadMs;
                        lastTiming_.pyramidMs=std::max(0.0,milliseconds(started)-fastWall);
                        const auto detectionStarted=std::chrono::steady_clock::now();
                        if(!useProvidedKeypoints) detectImpl(pyramid,working,geometry);
                        lastTiming_.detectionMs=milliseconds(detectionStarted)+fastWall;
                        const auto descriptorStarted=std::chrono::steady_clock::now();
                        if(descriptors.needed()) computeImpl(gray,pyramid,working,descriptors,useProvidedKeypoints?nullptr:&geometry);
                        lastTiming_.descriptorMs=milliseconds(descriptorStarted);
                        const auto& gpu=cuda_->timing();
                        lastTiming_.usedCuda=true;
                        lastTiming_.maskCacheHit=cacheHit;
                        lastTiming_.gpuUploadMs=gpu.uploadMs;
                        lastTiming_.gpuPyramidMs=gpu.pyramidMs;
                        lastTiming_.gpuFastMs=gpu.fastMs;
                        lastTiming_.fastMs=gpu.fastMs;
                        lastTiming_.gpuCornersDownloadMs=gpu.cornersDownloadMs;
                        lastTiming_.gpuIntegralMs=gpu.integralMs;
                        lastTiming_.gpuDescriptorMs=gpu.descriptorMs;
                        lastTiming_.gpuDescriptorsDownloadMs=gpu.descriptorsDownloadMs;
                        keypoints.swap(working);
                        return;
                    } catch(const detail::OrbCudaError& error) {
                        disableCuda(error.what());
                        lastTiming_={};
                    }
                }
            }
#endif
            buildPyramid(gray,mask,pyramid);
            lastTiming_.pyramidMs=milliseconds(started); const auto detectionStarted=std::chrono::steady_clock::now();
            if (!useProvidedKeypoints) detectImpl(pyramid,keypoints,geometry);
            lastTiming_.detectionMs=milliseconds(detectionStarted); const auto descriptorStarted=std::chrono::steady_clock::now();
            if (descriptors.needed()) computeImpl(gray,pyramid,keypoints,descriptors,useProvidedKeypoints?nullptr:&geometry);
            lastTiming_.descriptorMs=milliseconds(descriptorStarted);
        }

        OrbDistribution OrbFeatureExtractor::distribution(const std::vector<cv::KeyPoint>& keypoints) const {
            OrbDistribution result; result.perLevel.resize(options_.pyramidLevels, 0);
            result.directionBins.resize(options_.balanceBearings ? options_.azimuthBins * options_.zBins : 1, 0);
            const int cols = camera_ ? (static_cast<int>(camera_->imageWidth()) + options_.cellSize - 1) / options_.cellSize : 0;
            const int rows = camera_ ? (static_cast<int>(camera_->imageHeight()) + options_.cellSize - 1) / options_.cellSize : 0;
            result.imageCells.resize(cols * rows, 0);
            for (const auto& point : keypoints) {
                if (point.octave >= 0 && point.octave < options_.pyramidLevels) ++result.perLevel[point.octave];
                if (!std::isfinite(point.pt.x) || !std::isfinite(point.pt.y)) continue;
                const int bin = directionBin(point.pt); if (bin >= 0) ++result.directionBins[bin];
                if (cols&& rows&& point.pt.x >= 0 && point.pt.y >= 0 &&
                    point.pt.x < camera_->imageWidth() && point.pt.y < camera_->imageHeight())
                    ++result.imageCells[static_cast<int>(point.pt.y) / options_.cellSize * cols + static_cast<int>(point.pt.x) / options_.cellSize];
            }
            result.occupiedImageCells = static_cast<int>(std::count_if(result.imageCells.begin(), result.imageCells.end(), [](int n) {return n > 0; }));
            result.occupiedDirectionBins = static_cast<int>(std::count_if(result.directionBins.begin(), result.directionBins.end(), [](int n) {return n > 0; }));
            return result;
        }
    }
}
