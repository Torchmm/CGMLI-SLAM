// Regression executable source. Not compiled/run by the static-only workflow.
#include <debug/ModuleDebug.hpp>
#include <debug/ModuleDebugArtifacts.hpp>
#include <visual/features/OrbFeatureExtractor.hpp>
#include <visual/features/OrbFrameAdapter.hpp>
#include <visual/features/OrbCameraAwareSampling.hpp>
#include <visual/features/OrbSparseSampling.hpp>
#include <visual/Frame.hpp>
#include <visual/MultiFrame.hpp>
#include <cameras/PinholeCamera.hpp>
#include <cameras/EucmCamera.hpp>
#include <cameras/EquidistantDistortion.hpp>
#include <cameras/RadialTangentialDistortion.hpp>
#include <cameras/RadialTangentialDistortion8.hpp>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <map>
#ifdef CGMLI_ENABLE_CUDA_ORB
#include <cuda_runtime_api.h>
#endif

using namespace cgmli;
using namespace cgmli::features;
namespace {
    void require(bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    }
    OrbExtractorOptions options() {
        OrbExtractorOptions o;
        o.descriptorMode = OrbExtractorOptions::DescriptorMode::GravityCameraAware;
        o.virtualFocalLength = 120; o.pyramidLevels = 3;
        return o;
    }
    auto pinhole(double fv = 120) {
        return std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(
            161, 161, 120, fv, 80, 80, cameras::NoDistortion());
    }
    cv::KeyPoint centre(int octave = 0) {
        return cv::KeyPoint(80, 80, 31, 0, 1, octave, 7);
    }
    cv::Mat texture() {
        cv::Mat image(161, 161, CV_8UC1);
        cv::RNG rng(12345); rng.fill(image, cv::RNG::UNIFORM, 0, 256);
        return image;
    }
    void cachedExtractionAndDynamicMasks() {
        auto settings=options(); settings.maxFeatures=120;
        OrbFeatureExtractor orb(pinhole(),settings);
        const Eigen::Vector3d gravity(0,1,0); orb.setExtractionDirection(gravity);
        const auto image=texture(); std::vector<cv::KeyPoint> baseline,warm,dynamic;
        cv::Mat expected,actual,masked;
        const auto equal=[](const auto& a,const cv::Mat& da,const auto& b,const cv::Mat& db) {
            require(a.size()==b.size() && da.rows==db.rows,"Cached extraction changed feature rows");
            for(size_t i=0;i<a.size();++i) require(a[i].pt==b[i].pt && a[i].octave==b[i].octave
                && a[i].angle==b[i].angle,"Cached extraction changed feature identity/orientation");
            require(da.empty()==db.empty() && (da.empty() || cv::norm(da,db,cv::NORM_HAMMING)==0),"Cached descriptor differs from recomputed result");
        };
        orb.detectAndCompute(image,cv::noArray(),baseline,expected);
        require(!baseline.empty() && !orb.lastTiming().maskCacheHit,"First frame must build static masks");
        require(orb.lastTiming().geometryCacheHits>=baseline.size(),"Combined extraction must reuse detected warps");
        OrbFeatureExtractor prepared(pinhole(),settings);
        const auto* rays=prepared.rayDirections().data;
        prepared.prepareStaticCaches(); prepared.prepareStaticCaches();
        require(!prepared.hasExtractionDirection() && prepared.lastTiming().grayIntegralBuilds==0
            && prepared.rayDirections().data==rays,"Startup preparation must be idempotent and independent of image/gravity");
        prepared.setExtractionDirection(gravity);
        prepared.detectAndCompute(image,cv::noArray(),warm,actual);
        require(prepared.lastTiming().maskCacheHit,"Prepared first image must reuse static masks");
        equal(baseline,expected,warm,actual);
        orb.detectAndCompute(image,cv::noArray(),warm,actual);
        require(orb.lastTiming().maskCacheHit,"Second frame must reuse static masks"); equal(baseline,expected,warm,actual);
        cv::Mat allValid(image.size(),CV_8UC1,cv::Scalar(255));
        orb.detectAndCompute(image,allValid,dynamic,masked);
        require(!orb.lastTiming().maskCacheHit,"Explicit external mask must take the uncached path"); equal(baseline,expected,dynamic,masked);
        cv::Mat holes=allValid.clone(); holes(cv::Rect(60,55,30,40)).setTo(0);
        orb.detectAndCompute(image,holes,dynamic,masked);
        OrbFeatureExtractor reference(pinhole(),settings); reference.setExtractionDirection(gravity);
        std::vector<cv::KeyPoint> referenceKeys; cv::Mat referenceRows;
        reference.detectAndCompute(image,holes,referenceKeys,referenceRows); equal(dynamic,masked,referenceKeys,referenceRows);
        cv::Mat emptyMask=cv::Mat::zeros(image.size(),CV_8UC1);
        orb.detectAndCompute(image,emptyMask,dynamic,masked);
        require(dynamic.empty() && masked.empty() && orb.lastTiming().emptyMaskCells>0 && orb.lastTiming().fastCalls==0,
            "Empty safe-mask cells must never request low-threshold FAST");
        orb.detectAndCompute(image,cv::noArray(),warm,actual);
        require(orb.lastTiming().maskCacheHit,"Dynamic masks must not poison the static cache"); equal(baseline,expected,warm,actual);
        orb.detect(image,warm); orb.compute(image,warm,actual);
        require(orb.lastTiming().geometryCacheHits==0,"Separate compute must not reuse a previous detection's gravity/warp");
        equal(baseline,expected,warm,actual);
        const Eigen::Vector3d changedGravity(1,0,0); orb.setExtractionDirection(changedGravity);
        reference.setExtractionDirection(changedGravity);
        orb.detectAndCompute(image,cv::noArray(),warm,actual);
        reference.detectAndCompute(image,cv::noArray(),referenceKeys,referenceRows); equal(warm,actual,referenceKeys,referenceRows);

        // Reused allocations must never retain a previous image's gray integral.
        auto first=std::vector<cv::KeyPoint>{centre()}; cv::Mat originalRows;
        orb.compute(image,first,originalRows);
        cv::Mat inverted; cv::bitwise_not(image,inverted);
        auto second=std::vector<cv::KeyPoint>{centre()},fresh=second;
        orb.compute(inverted,second,actual); reference.compute(inverted,fresh,referenceRows);
        equal(second,actual,fresh,referenceRows);
        require(orb.lastTiming().grayIntegralBuilds==1 && !actual.empty()
            && cv::norm(originalRows,actual,cv::NORM_HAMMING)>0,"Image change must rebuild gray integrals and change descriptor bits");
    }
    void sparseSamplerAndAnalyticWarpAgreement() {
        const auto image=texture(); cv::Mat integral; cv::integral(image,integral,CV_64F);
        cv::Mat storage(integral.rows+2,integral.cols+7,CV_64FC1);
        cv::Mat strided=storage(cv::Rect(3,1,integral.cols,integral.rows)); integral.copyTo(strided);
        for(const auto& half:std::vector<Eigen::Vector2d>{{0.5,0.5},{0.75,1.25},{3.46,5.2}}) {
            ::cgmli::features::detail::SparseBoxSampler sampler(strided,half);
            for(const auto& p:std::vector<Eigen::Vector2d>{{40.125,60.75},{80,80},Eigen::Vector2d(half-Eigen::Vector2d::Constant(0.5))}) {
                float value=0; require(sampler.sample(p,value),"Strided box sampler rejected valid support");
                const Eigen::Vector2d c=p+Eigen::Vector2d::Constant(0.5),lo=c-half,hi=c+half;
                const float expected=float((::cgmli::features::detail::fractionalIntegral(strided,hi.x(),hi.y())
                    -::cgmli::features::detail::fractionalIntegral(strided,lo.x(),hi.y())-::cgmli::features::detail::fractionalIntegral(strided,hi.x(),lo.y())
                    +::cgmli::features::detail::fractionalIntegral(strided,lo.x(),lo.y()))/(4*half.x()*half.y()));
                require(value==expected,"Optimized box sampler changed reference interpolation/rounding");
            }
            float value; require(!sampler.sample(Eigen::Vector2d(-1,0),value),"Box sampler must reject invalid support");
        }
        for(double x:{0.1,1.0,4.0,12.0}) for(double y:{0.1,0.9,2.0}) {
            Eigen::Matrix2d matrix; matrix<<x,0.3,-0.1,y;
            const Eigen::Vector2d analytic=::cgmli::features::detail::singularValues2x2(matrix);
            const Eigen::Vector2d reference=Eigen::JacobiSVD<Eigen::Matrix2d>(matrix).singularValues();
            require((analytic-reference).norm()<1e-11*std::max(1.0,reference.norm()),"Analytic singular values disagree with SVD");
        }
        require(::cgmli::features::detail::singularValues2x2(Eigen::Matrix2d::Zero()).isZero(),"Zero warp must remain singular");
    }
    void layerFastAndCompactIntegralAgreement() {
        cv::Mat storage(161,181,CV_8UC1,cv::Scalar(0));
        cv::Mat strided=storage(cv::Rect(7,0,161,161)); texture().copyTo(strided);
        cv::Mat boundaries(161,161,CV_8UC1,cv::Scalar(80));
        cv::Mat lowContrast; texture().convertTo(lowContrast,CV_8UC1,0.05,100);
        for(int y:{30,31,32,62,63,64,94,95,96}) for(int x:{30,31,32,62,63,64,94,95,96})
            cv::rectangle(boundaries,cv::Rect(x,y,5,5),cv::Scalar((x+y)%2?255:0),cv::FILLED);
        for(auto mode:{OrbExtractorOptions::DescriptorMode::Standard,OrbExtractorOptions::DescriptorMode::GravityAligned,
            OrbExtractorOptions::DescriptorMode::GravityCameraAware}) for(const auto& image:{strided,boundaries,lowContrast}) {
            auto referenceOptions=options(); referenceOptions.maxFeatures=120; referenceOptions.descriptorMode=mode;
            referenceOptions.layerFast=false; referenceOptions.compactGrayIntegral=false;
            referenceOptions.adaptiveGeometryCandidates=false; referenceOptions.groupedBalancing=false; referenceOptions.boundaryFast=false;
            auto optimizedOptions=referenceOptions; optimizedOptions.layerFast=true; optimizedOptions.compactGrayIntegral=true; optimizedOptions.groupedBalancing=true;
            auto camera=pinhole(); OrbFeatureExtractor reference(camera,referenceOptions),optimized(camera,optimizedOptions);
            reference.setExtractionDirection(Eigen::Vector3d(0,1,0)); optimized.setExtractionDirection(Eigen::Vector3d(0,1,0));
            cv::Mat mask(image.size(),CV_8UC1,cv::Scalar(255)); mask(cv::Rect(70,50,7,20)).setTo(0);
            for(const auto& selectedMask:{cv::Mat(),mask}) {
                std::vector<cv::KeyPoint> expected,actual; cv::Mat a,b;
                reference.detectAndCompute(image,selectedMask,expected,a); optimized.detectAndCompute(image,selectedMask,actual,b);
                require(expected.size()==actual.size(),"Shared layer FAST changed cell quota/feature count");
                for(size_t k=0;k<expected.size();++k) require(expected[k].pt==actual[k].pt && expected[k].response==actual[k].response
                    && expected[k].octave==actual[k].octave && expected[k].angle==actual[k].angle,"Shared FAST changed boundary NMS/response/order");
                require(a.empty()==b.empty() && (a.empty() || cv::norm(a,b,cv::NORM_HAMMING)==0),"Compact integral changed descriptor bits");
                require(optimized.lastTiming().fastCalls-optimized.lastTiming().fastFallbackCalls<=2*size_t(optimizedOptions.pyramidLevels),
                    "Shared high-threshold FAST must run at most twice per layer, plus selective ROI fallback");
                auto stripOptions=optimizedOptions; stripOptions.boundaryFast=true;
                OrbFeatureExtractor strip(camera,stripOptions); strip.setExtractionDirection(Eigen::Vector3d(0,1,0));
                std::vector<cv::KeyPoint> stripPoints; cv::Mat stripDescriptors;
                strip.detectAndCompute(image,selectedMask,stripPoints,stripDescriptors);
                require(stripPoints.size()==expected.size(),"Boundary-strip FAST changed feature count");
                for(size_t k=0;k<expected.size();++k) require(expected[k].pt==stripPoints[k].pt && expected[k].response==stripPoints[k].response
                    && expected[k].octave==stripPoints[k].octave && expected[k].angle==stripPoints[k].angle,"Boundary-strip FAST or grouped heap changed exact selection");
                require(a.empty()==stripDescriptors.empty() && (a.empty() || cv::norm(a,stripDescriptors,cv::NORM_HAMMING)==0),"Boundary-strip FAST changed descriptor bits");
                if(image.data==lowContrast.data) require(optimized.lastTiming().fastFallbackCalls>0,
                    "Low-contrast cells must retain the low-threshold ROI fallback");
                if(mode!=OrbExtractorOptions::DescriptorMode::Standard && !actual.empty())
                    require(optimized.lastTiming().grayIntegral32Builds>0,"Small images must use overflow-safe int32 prefixes");
            }
        }
        cv::Mat integral32,integral64; const auto image=texture();
        cv::integral(image,integral32,CV_32S); cv::integral(image,integral64,CV_64F);
        for(const auto& half:std::vector<Eigen::Vector2d>{{0.5,0.5},{3.46,5.2}}) {
            float a=0,b=0; require(::cgmli::features::detail::sparseBoxIntensity(integral32,Eigen::Vector2d(80.125,80.75),half,a)
                && ::cgmli::features::detail::sparseBoxIntensity(integral64,Eigen::Vector2d(80.125,80.75),half,b) && a==b,"Integer/double fractional prefixes must agree");
        }
    }
    void batchedDescriptorCoordinatesMatchScalar() {
        const auto image=texture(); cv::Mat a,b; cv::integral(image,a,CV_32S); cv::integral(image,b,CV_64F);
        for(const auto& integral:{a,b}) for(double angle:{0.0,0.2,1.5}) {
            Eigen::Matrix2d W; W<<std::cos(angle),-0.7*std::sin(angle),std::sin(angle),0.7*std::cos(angle);
            std::array<uchar,32> reference{},actual{};
            require(::cgmli::features::detail::sparseOrbDescriptor(integral,Eigen::Vector2d(80.125,80.75),W,reference,false)
                && ::cgmli::features::detail::sparseOrbDescriptor(integral,Eigen::Vector2d(80.125,80.75),W,actual,true),"Both sampling coordinate paths must evaluate");
            require(reference==actual,"Batched coordinates must preserve all descriptor bits");
        }
    }
    void adaptiveGeometryBatchesPreserveSupport() {
        auto settings=options(); settings.maxFeatures=120;
        auto referenceOptions=settings; referenceOptions.adaptiveGeometryCandidates=false;
        OrbFeatureExtractor reference(pinhole(),referenceOptions),optimized(pinhole(),settings);
        reference.setExtractionDirection(Eigen::Vector3d(0,1,0)); optimized.setExtractionDirection(Eigen::Vector3d(0,1,0));
        auto image=texture();
        auto rows=debug::artifactFile("adaptive_geometry_batches.csv");
        if(rows) rows<<"mask,reference_count,batch_count,reference_geometry,batch_geometry,batch_expansions,image_cells,direction_bins\n";
        cv::Mat mask(image.size(),CV_8UC1,cv::Scalar(0)); mask(cv::Rect(55,55,51,51)).setTo(255);
        int iteration=0;
        for(const auto& selectedMask:{cv::Mat(),mask}) {
            std::vector<cv::KeyPoint> a,b; cv::Mat da,db;
            reference.detectAndCompute(image,selectedMask,a,da); optimized.detectAndCompute(image,selectedMask,b,db);
            const auto distribution=optimized.distribution(b);
            if(rows) rows<<iteration++<<','<<a.size()<<','<<b.size()<<','<<reference.lastTiming().geometryEvaluations<<','
                <<optimized.lastTiming().geometryEvaluations<<','<<optimized.lastTiming().geometryExpansionRounds<<','
                <<distribution.occupiedImageCells<<','<<distribution.occupiedDirectionBins<<'\n';
            require(a.size()==b.size(),"Adaptive geometry batches must replenish available feature capacity");
            require(db.rows==int(b.size()),"Every adaptively selected point must retain a descriptor row");
            for(const auto& point:b) {
                Eigen::Matrix2d warp; bool aligned=false;
                require(optimized.samplingWarp(point,warp,aligned),"Adaptive batching must retain the camera-aware geometry gate");
                require(selectedMask.empty() || selectedMask.at<uchar>(cvRound(point.pt.y),cvRound(point.pt.x)),"Dynamic mask must remain enforced");
            }
            const auto baseline=reference.distribution(a);
            require(distribution.occupiedImageCells*10>=baseline.occupiedImageCells*8 && distribution.occupiedDirectionBins*10>=baseline.occupiedDirectionBins*8,
                "Adaptive batches must retain spatial and spherical coverage on the controlled fixture");
        }
    }
    void budgetDrivenFallbackPreservesCoverage() {
        auto camera=std::make_shared<cameras::PinholeCamera<cameras::NoDistortion>>(
            640,480,300,300,320,240,cameras::NoDistortion());
        auto settings=options(); settings.descriptorMode=OrbExtractorOptions::DescriptorMode::Standard;
        settings.pyramidLevels=1; settings.maxFeatures=80; settings.balanceBearings=false;
        cv::Mat image(480,640,CV_8UC1,cv::Scalar(128)); cv::RNG rng(4907);
        for(int y=40;y<430;y+=64) for(int x=40;x<580;x+=64) {
            auto tile=image(cv::Rect(x,y,18,18)); rng.fill(tile,cv::RNG::UNIFORM,0,256);
        }
        OrbFeatureExtractor adaptive(camera,settings);
        settings.budgetedFastFallback=false; OrbFeatureExtractor reference(camera,settings);
        std::vector<cv::KeyPoint> a,b; cv::Mat da,db;
        adaptive.detectAndCompute(image,cv::noArray(),a,da); reference.detectAndCompute(image,cv::noArray(),b,db);
        require(a.size()==80 && b.size()==80,"Budgeted fallback must fill a feasible layer budget");
        require(adaptive.lastTiming().budgetSkippedCells>0 && adaptive.lastTiming().fastFallbackCalls<reference.lastTiming().fastFallbackCalls,
            "Coarse coverage must suppress redundant low-threshold calls");
        std::map<std::pair<int,int>,size_t> cells;
        for(const auto& key:a) ++cells[{int(key.pt.x-31)/32,int(key.pt.y-31)/32}];
        for(const auto& cell:cells) require(cell.second<=size_t(settings.maxFeaturesPerCell),"Budgeted fallback must retain the fine-cell occupancy limit");
        debug::recordError("feature_budget_error",std::abs(double(a.size())-80),0);
        if(!debug::caseDirectory.empty()) {
            cv::Mat adaptiveView,referenceView;
            cv::drawKeypoints(image,a,adaptiveView,cv::Scalar(0,200,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::drawKeypoints(image,b,referenceView,cv::Scalar(0,200,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            for(auto* view:{&adaptiveView,&referenceView}) {
                for(int x=31;x<view->cols-31;x+=32) cv::line(*view,{x,31},{x,view->rows-31},cv::Scalar(100,100,100));
                for(int y=31;y<view->rows-31;y+=32) cv::line(*view,{31,y},{view->cols-31,y},cv::Scalar(100,100,100));
                cv::putText(*view,"SYNTHETIC: fine-grid feature distribution",{10,20},cv::FONT_HERSHEY_SIMPLEX,0.5,{0,255,0},1);
            }
            debug::saveImage("adaptive_features.png",adaptiveView); debug::saveImage("reference_features.png",referenceView);
            auto csv=debug::artifactFile("adaptive_points.csv"); csv<<"x,y,octave,response,angle\n";
            for(const auto& key:a) csv<<key.pt.x<<','<<key.pt.y<<','<<key.octave<<','<<key.response<<','<<key.angle<<'\n';
        }
    }
    void missingAndInvalidGravity() {
        OrbFeatureExtractor orb(pinhole(), options());
        auto points = std::vector<cv::KeyPoint>{centre()}; cv::Mat descriptors;
        bool missing = false;
        try { orb.compute(texture(), points, descriptors); }
        catch (const std::logic_error&) { missing = true; }
        require(missing, "Missing gravity must not silently yield texture ORB");
        bool invalid = false;
        try { orb.setExtractionDirection(Eigen::Vector3d::Zero()); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid && !orb.hasExtractionDirection(), "Zero gravity must be rejected");
        invalid = false;
        try { orb.setExtractionDirection(Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 1, 0)); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "Nonfinite gravity must be rejected");
    }
    void gravityRollInvariance() {
        OrbFeatureExtractor a(pinhole(), options()), b(pinhole(), options());
        a.setExtractionDirection(Eigen::Vector3d(0, 1, 0));
        b.setExtractionDirection(Eigen::Vector3d(-1, 0, 0));
        const auto original = texture(); cv::Mat rotated;
        cv::rotate(original, rotated, cv::ROTATE_90_CLOCKWISE);
        auto ka = std::vector<cv::KeyPoint>{centre()}, kb = ka;
        cv::Mat da, db; a.compute(original, ka, da); b.compute(rotated, kb, db);
        require(ka.size() == 1 && kb.size() == 1, "Rotation fixture must retain the centre feature");
        require(da.type() == CV_8UC1 && da.cols == 32, "ORB layout must remain 256-bit");
        require(cv::norm(da, db, cv::NORM_HAMMING) == 0, "90-degree camera roll must preserve descriptor bits");
        require(std::abs(kb[0].angle - 90) < 1e-4, "Image diagnostic angle must agree with warp");
    }
    void defaultModeAndStandardOptOut() {
        OrbFeatureExtractor defaultOrb(pinhole());
        require(defaultOrb.isCameraAware(), "GravityCameraAware must be the default");
        require(defaultOrb.getDefaultName() == "cgmli.ORB.GravityCameraAware.SparseBox.v3", "Descriptor profile must record new sampling semantics");
        require(defaultOrb.options().samplingMode == OrbExtractorOptions::SamplingMode::SparseBox, "Sparse sampling must be the default");
        bool missing = false; std::vector<cv::KeyPoint> points;
        try { defaultOrb.detect(texture(), points); } catch (const std::logic_error&) { missing = true; }
        require(missing, "Default detection requires gravity too");
        OrbExtractorOptions standard; standard.descriptorMode = OrbExtractorOptions::DescriptorMode::Standard;
        standard.balanceBearings = false;
        OrbFeatureExtractor legacy(nullptr, standard);
        points = {centre()}; cv::Mat descriptor; legacy.compute(texture(), points, descriptor);
        require(points.size() == 1 && descriptor.cols == 32, "Explicit Standard mode works without camera or gravity");
    }
    void affineSmoothingAgreement() {
        const cv::Mat image = texture(), mask(image.size(), CV_8UC1, cv::Scalar(255));
        const cv::Mat integral = ::cgmli::features::detail::invalidMaskIntegral(mask);
        Eigen::Matrix2d warp; warp << 1.8, 0.6, 0.3, 1.2;
        const Eigen::Vector2d centrePixel(80, 80);
        cv::Mat patch, filtered;
        require(::cgmli::features::detail::rectifyPatch(image, integral, centrePixel, warp, patch), "Anisotropic fixture must fit");
        ::cgmli::features::detail::smoothRectifiedPatch(patch, filtered);
        const int scale = ::cgmli::features::detail::patchSamplingScale(patch), centreIndex = ::cgmli::features::detail::kOrbTileRadius * scale;
        require(scale >= 2, "Stretched sampling must supersample before smoothing");
        const int radius = ::cgmli::features::detail::kOrbBlurRadius * scale;
        const cv::Mat kernel = cv::getGaussianKernel(2 * radius + 1, ::cgmli::features::detail::kOrbSigma * scale, CV_64F);
        const Eigen::Vector2d sample(5, -3); double expected = 0;
        // Independent direct warped-kernel sum: filter offsets must be transformed too.
        for (int y = -radius; y <= radius; ++y) for (int x = -radius; x <= radius; ++x) {
            const Eigen::Vector2d pixel = centrePixel + warp * (sample + Eigen::Vector2d(double(x) / scale, double(y) / scale));
            const int ix = static_cast<int>(std::floor(pixel.x())), iy = static_cast<int>(std::floor(pixel.y()));
            const double dx = pixel.x() - ix, dy = pixel.y() - iy;
            const double value = (1 - dy) * ((1 - dx) * image.at<uchar>(iy, ix) + dx * image.at<uchar>(iy, ix + 1)) +
                dy * ((1 - dx) * image.at<uchar>(iy + 1, ix) + dx * image.at<uchar>(iy + 1, ix + 1));
            expected += value * kernel.at<double>(y + radius) * kernel.at<double>(x + radius);
        }
        const double actual = filtered.at<float>(centreIndex - 3 * scale, centreIndex + 5 * scale);
        require(std::abs(actual - expected) < 0.002, "Rectified Gaussian must agree with the full affine filter");
        cv::Mat fixedBlur; cv::GaussianBlur(image, fixedBlur, cv::Size(7, 7), 2, 2);
        double difference = 0;
        for (int y = -6; y <= 6; y += 3) for (int x = -6; x <= 6; x += 3) {
            const Eigen::Vector2d pixel = centrePixel + warp * Eigen::Vector2d(x, y); float oldValue;
            require(::cgmli::features::detail::bilinearIntensity(fixedBlur, pixel.x(), pixel.y(), oldValue), "Old sampling fixture");
            difference += std::abs(oldValue - filtered.at<float>(centreIndex + y * scale, centreIndex + x * scale));
        }
        require(difference > 0.1, "Warped smoothing must differ from the previous fixed image blur");
    }
    void sparsePatternAndFractionalArea() {
        require(::cgmli::features::detail::kSparseOrbPattern.count == 375, "ORB point deduplication must preserve 375 positions");
        for (int bit = 0; bit < 256; ++bit) for (int end = 0; end < 2; ++end) {
            const int index = ::cgmli::features::detail::kSparseOrbPattern.pairs[bit][end];
            require(index >= 0 && index < 375 && ::cgmli::features::detail::kSparseOrbPattern.points[index][0] == ::cgmli::features::detail::kOrbPattern[bit][2 * end] &&
                ::cgmli::features::detail::kSparseOrbPattern.points[index][1] == ::cgmli::features::detail::kOrbPattern[bit][2 * end + 1], "Deduplication must preserve comparison endpoints");
        }
        cv::Mat image(17, 19, CV_8UC1);
        for (int y = 0; y < image.rows; ++y) for (int x = 0; x < image.cols; ++x)
            image.at<uchar>(y, x) = static_cast<uchar>(10 * y + x);
        cv::Mat integral; cv::integral(image, integral, CV_64F);
        const Eigen::Vector2d point(3.25, 4.75), half(1.3, 0.6);
        const Eigen::Vector2d lower = point + Eigen::Vector2d::Constant(0.5) - half;
        const Eigen::Vector2d upper = point + Eigen::Vector2d::Constant(0.5) + half;
        double sum = 0;
        for (int y = 0; y < image.rows; ++y) for (int x = 0; x < image.cols; ++x) {
            const double wx = std::max(0.0, std::min(double(x + 1), upper.x()) - std::max(double(x), lower.x()));
            const double wy = std::max(0.0, std::min(double(y + 1), upper.y()) - std::max(double(y), lower.y()));
            sum += image.at<uchar>(y, x) * wx * wy;
        }
        float actual;
        require(::cgmli::features::detail::sparseBoxIntensity(integral, point, half, actual) &&
            std::abs(actual - sum / (4 * half.x() * half.y())) < 1e-4, "Fractional integral box must agree with explicit area integration");
    }
    void sparseAndReferenceProfiles() {
        auto sparseOptions = options(), referenceOptions = options();
        referenceOptions.samplingMode = OrbExtractorOptions::SamplingMode::AffineGaussianReference;
        OrbFeatureExtractor sparse(pinhole(), sparseOptions), reference(pinhole(), referenceOptions);
        require(sparse.getDefaultName() != reference.getDefaultName(), "Different smoothing backends require different descriptor profiles");
        sparse.setExtractionDirection(Eigen::Vector3d(0, 1, 0)); reference.setExtractionDirection(Eigen::Vector3d(0, 1, 0));
        const cv::Mat image = texture(); auto a = std::vector<cv::KeyPoint>{centre()}, b = a; cv::Mat da, db;
        sparse.compute(image, a, da); reference.compute(image, b, db);
        require(a.size() == 1 && b.size() == 1 && da.cols == 32 && db.cols == 32, "Both backends must retain the ORB output interface");
    }
    void smoothingMaskAndCompressedSupport() {
        const cv::Mat image = texture(); cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
        mask.at<uchar>(80, 115) = 0; // W=2 Gaussian support reaches 36px, rather than the old 33px.
        const Eigen::Matrix2d warp = 2 * Eigen::Matrix2d::Identity();
        require(!::cgmli::features::detail::warpedSupportValid(image.size(), ::cgmli::features::detail::invalidMaskIntegral(mask), Eigen::Vector2d(80, 80), warp),
            "Warped Gaussian mask support must be validated");
        auto camera = pinhole(); mask.setTo(255); mask.at<uchar>(80, 100) = 0; camera->setMask(mask);
        auto o = options(); o.virtualFocalLength = 240; // compressed support radius=9px.
        OrbFeatureExtractor orb(camera, o); orb.setExtractionDirection(Eigen::Vector3d(0, 1, 0));
        auto points = std::vector<cv::KeyPoint>{centre()}; cv::Mat descriptor; orb.compute(image, points, descriptor);
        require(points.size() == 1 && !descriptor.empty(), "Mask holes outside compressed support must not reject the feature");
    }
    void canonicalPoleOrientation() {
        OrbFeatureExtractor a(pinhole(), options()), b(pinhole(), options());
        a.setExtractionDirection(Eigen::Vector3d(0, 0, 1)); b.setExtractionDirection(Eigen::Vector3d(0, 0, 1));
        const cv::Mat image = texture(); cv::Mat rotated; cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
        auto ka = std::vector<cv::KeyPoint>{centre()}, kb = ka; cv::Mat da, db;
        a.compute(image, ka, da); b.compute(rotated, kb, db);
        require(ka.size() == 1 && kb.size() == 1 && cv::norm(da, db, cv::NORM_HAMMING) <= 3,
            "Pole fallback should estimate rotation on the canonical patch");
        cv::Mat flat(::cgmli::features::detail::kOrbTileSize, ::cgmli::features::detail::kOrbTileSize, CV_32FC1, cv::Scalar(128)); double angle;
        require(!::cgmli::features::detail::rectifiedTextureOrientation(flat, angle), "Unobservable canonical direction must be rejected");
    }
    void poleFallbackAndOctaveScale() {
        OrbFeatureExtractor orb(pinhole(), options());
        orb.setExtractionDirection(Eigen::Vector3d(0, 0, 1));
        Eigen::Matrix2d w; bool aligned;
        require(orb.samplingWarp(centre(), w, aligned) && !aligned && w.allFinite(),
            "Parallel gravity requires a finite texture fallback");
        require((w - Eigen::Matrix2d::Identity()).norm() < 1e-6, "Centre fallback should have unit scale");
        orb.setExtractionDirection(Eigen::Vector3d(0, 1, 0));
        Eigen::Matrix2d w2;
        require(orb.samplingWarp(centre(2), w2, aligned) && aligned, "Octave warp must be valid");
        const double nominal = std::pow(double(options().scaleFactor), 2);
        const double resizeCorrection = nominal * cvRound(161 / nominal) / 161;
        require((resizeCorrection * w - w2).norm() < 1e-6, "Warp must use rounded resize dimensions without applying octave twice");
    }
    void projectionDerivativeAgreement() {
        auto camera = pinhole(); OrbFeatureExtractor orb(camera, options());
        const Eigen::Vector3d gravity(0, 1, 0); orb.setExtractionDirection(gravity);
        cv::KeyPoint keypoint(100, 90, 31, 0);
        Eigen::Matrix2d w; bool aligned;
        require(orb.samplingWarp(keypoint, w, aligned) && aligned, "Off-axis warp must be valid");
        Eigen::Vector3d ray;
        require(camera->backProject(Eigen::Vector2d(100, 90), &ray), "Backproject fixture");
        ray.normalize();
        const Eigen::Vector3d x = gravity.cross(ray).normalized(), y = ray.cross(x).normalized();
        const double epsilon = 1e-6;
        for (int axis = 0; axis < 2; ++axis) {
            const Eigen::Vector3d tangent = axis == 0 ? x : y;
            Eigen::Vector2d plus, minus;
            require(camera->project(ray + epsilon * tangent, &plus) == cameras::ProjectionStatus::Successful,
                "Positive numerical derivative must project");
            require(camera->project(ray - epsilon * tangent, &minus) == cameras::ProjectionStatus::Successful,
                "Negative numerical derivative must project");
            const Eigen::Vector2d numeric=(plus-minus)/(2*epsilon*120);
            debug::recordDerivative("projection_warp_axis_"+std::to_string(axis),
                (numeric-w.col(axis)).norm(),numeric.norm());
        }
    }
    void fullWarpedMaskSupport() {
        auto camera = pinhole(); cv::Mat mask(161, 161, CV_8UC1, cv::Scalar(255));
        mask.at<uchar>(80, 108) = 0; camera->setMask(mask);
        auto o = options(); o.virtualFocalLength = 60; // W=2: mask hole beyond the old 25-pixel support.
        OrbFeatureExtractor orb(camera, o); orb.setExtractionDirection(Eigen::Vector3d(0, 1, 0));
        auto points = std::vector<cv::KeyPoint>{centre()}; cv::Mat descriptor;
        orb.compute(texture(), points, descriptor);
        require(points.empty() && descriptor.empty(), "Mask holes inside the warped patch must reject the feature");
    }
    void rejectExcessiveStretch() {
        OrbFeatureExtractor orb(pinhole(1200), options());
        orb.setExtractionDirection(Eigen::Vector3d(0, 1, 0));
        Eigen::Matrix2d w; bool aligned;
        require(!orb.samplingWarp(centre(), w, aligned), "Excessive warp stretch must be rejected");
    }
#ifdef CGMLI_ENABLE_CUDA_ORB
    bool hasCudaDevice() {
        int count=0;
        return cudaGetDeviceCount(&count)==cudaSuccess && count>0;
    }
    void cudaExtractionAgreement() {
        const auto compare=[](OrbFeatureExtractor& cpu,OrbFeatureExtractor& gpu,const cv::Mat& image,
            const cv::Mat& mask,const std::vector<cv::KeyPoint>& input) {
            auto expected=input,actual=input;
            cv::Mat cpuRows,gpuRows;
            cpu.detectAndCompute(image,mask,expected,cpuRows,!input.empty());
            gpu.detectAndCompute(image,mask,actual,gpuRows,!input.empty());
            require(gpu.lastTiming().usedCuda==mask.empty(),"CUDA extraction or dynamic-mask fallback was not exercised");
            if(input.empty()) require(cpu.lastTiming().fastCandidates==gpu.lastTiming().fastCandidates &&
                cpu.lastTiming().geometryExpansionRounds==gpu.lastTiming().geometryExpansionRounds,
                "CUDA ORB changed the candidate replenishment policy");
            if(expected.size()!=actual.size())
                throw std::runtime_error("CUDA ORB keypoint count differs: "+std::to_string(expected.size())+
                    " CPU, "+std::to_string(actual.size())+" GPU");
            for(size_t i=0;i<expected.size();++i) {
                const auto& a=expected[i]; const auto& b=actual[i];
                require(a.pt==b.pt && a.octave==b.octave && a.response==b.response &&
                    a.size==b.size && a.angle==b.angle && a.class_id==b.class_id,
                    "CUDA ORB changed keypoint identity, order or orientation");
            }
            require(cpuRows.empty()==gpuRows.empty() &&
                (cpuRows.empty() || cv::norm(cpuRows,gpuRows,cv::NORM_HAMMING)==0),
                "CUDA ORB changed descriptor bits");
        };
        auto settings=options(); settings.maxFeatures=120;
        auto cudaSettings=settings; cudaSettings.useCuda=true;
        auto camera=pinhole();
        OrbFeatureExtractor cpu(camera,settings),gpu(camera,cudaSettings);
        gpu.prepareStaticCaches();
        require(gpu.cudaReady(),"CUDA ORB startup failed on an available GPU");
        cpu.setExtractionDirection(Eigen::Vector3d::UnitY());
        gpu.setExtractionDirection(Eigen::Vector3d::UnitY());
        cv::Mat storage(161,181,CV_8UC1,cv::Scalar(0));
        auto image=storage(cv::Rect(7,0,161,161)); texture().copyTo(image);
        compare(cpu,gpu,image,{},{}); // strided input and several pyramid levels
        cv::Mat lowContrast; image.convertTo(lowContrast,CV_8UC1,0.05,100);
        compare(cpu,gpu,lowContrast,{},{}); // low threshold and cell borders
        cv::Mat mask(image.size(),CV_8UC1,cv::Scalar(255)); mask(cv::Rect(64,65,14,25)).setTo(0);
        compare(cpu,gpu,image,mask,{});
        cpu.setExtractionDirection(Eigen::Vector3d::UnitZ());
        gpu.setExtractionDirection(Eigen::Vector3d::UnitZ());
        compare(cpu,gpu,image,{},{centre(),centre(1),cv::KeyPoint(82.3f,75.7f,31,0,1,2)});

        auto eucm=std::make_shared<cameras::EucmCamera>(601,601,100,100,300,300,0.6,1.0);
        settings.virtualFocalLength=100; settings.maximumWarpStretch=8; settings.maximumWarpCondition=100;
        cudaSettings=settings; cudaSettings.useCuda=true;
        OrbFeatureExtractor cpuWide(eucm,settings),gpuWide(eucm,cudaSettings);
        cpuWide.setExtractionDirection(Eigen::Vector3d::UnitY());
        gpuWide.setExtractionDirection(Eigen::Vector3d::UnitY());
        cv::Mat wide(601,601,CV_8UC1);
        cv::RNG rng(216); rng.fill(wide,cv::RNG::UNIFORM,0,256);
        compare(cpuWide,gpuWide,wide,{},{cv::KeyPoint(480,300,31),cv::KeyPoint(300,300,31)});
        compare(cpuWide,gpuWide,wide,{},{});
    }
#endif

    void rearHemisphereEucm() {
        auto camera = std::make_shared<cameras::EucmCamera>(601, 601, 100, 100, 300, 300, 0.6, 1.0);
        auto o = options(); o.virtualFocalLength = 100;
        o.maximumWarpStretch = 8; o.maximumWarpCondition = 100;
        OrbFeatureExtractor orb(camera, o); orb.setExtractionDirection(Eigen::Vector3d(0, 1, 0));
        cv::KeyPoint keypoint(480, 300, 31, 0);
        Eigen::Vector3d ray;
        require(camera->backProject(Eigen::Vector2d(480, 300), &ray) && ray.z() < 0,
            "EUCM fixture must see the rear hemisphere");
        Eigen::Matrix2d w; bool aligned;
        require(orb.samplingWarp(keypoint, w, aligned) && w.allFinite(), "Valid rear-facing rays must not be discarded by z");
    }
    template<class Distortion> void cameraVariant(const Distortion& distortion) {
        auto camera = std::make_shared<cameras::PinholeCamera<Distortion>>(161, 161, 120, 120, 80, 80, distortion);
        OrbFeatureExtractor orb(camera, options()); orb.setExtractionDirection(Eigen::Vector3d(0, 1, 0));
        Eigen::Matrix2d w; bool aligned;
        require(orb.samplingWarp(cv::KeyPoint(95, 85, 31, 0), w, aligned) && w.allFinite(),
            "Distortion variant must work through CameraBase");
    }
    void frameAdapterAndRepeatability() {
        Frame frame; frame.setGeometry(pinhole()); frame.setImage(texture());
        auto orb = configureOrb(frame, options());
        const Eigen::Quaterniond q(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()));
        const kinematics::Transformation T_WC(Eigen::Vector3d(20, 30, 40), q);
        updateOrbGravity(*orb, T_WC, Eigen::Vector3d(0, 1, 0));
        OrbFeatureExtractor direct(frame.geometry(), options());
        direct.setExtractionDirection(q.conjugate() * Eigen::Vector3d(0, 1, 0));
        auto a = std::vector<cv::KeyPoint>{centre()}, b = a;
        cv::Mat da, db; orb->compute(frame.image(), a, da); direct.compute(frame.image(), b, db);
        require(!da.empty() && cv::norm(da, db, cv::NORM_HAMMING) == 0, "World-to-camera gravity must ignore translation");
        frame.detect(); frame.describe();
        require(frame.descriptors().rows == static_cast<int>(frame.keypoints().size()), "Frame rows must follow retained keypoints");
        cv::Mat again; auto points = a; orb->compute(frame.image(), points, again);
        require(cv::norm(da, again, cv::NORM_HAMMING) == 0, "Repeated extraction must be deterministic");
    }
    void multiCameraGravityAndReuse() {
        cameras::NCameraSystem rig;
        const auto first = pinhole(), second = pinhole();
        const kinematics::Transformation T_SC0(Eigen::Vector3d(1, 2, 3), Eigen::Quaterniond::Identity());
        const kinematics::Transformation T_SC1(Eigen::Vector3d(-2, 1, 4),
            Eigen::Quaterniond(Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ())));
        rig.addCamera(std::make_shared<kinematics::Transformation>(T_SC0), first, cameras::NCameraSystem::NoDistortion, false);
        rig.addCamera(std::make_shared<kinematics::Transformation>(T_SC1), second, cameras::NCameraSystem::NoDistortion, false);
        MultiFrame frame(rig, cgmli::Time());
        auto extractors = configureOrb(frame, options());
        const kinematics::Transformation T_WS(Eigen::Vector3d(6, 7, 8),
            Eigen::Quaterniond(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitX())));
        const Eigen::Vector3d gravityWorld(0, 1, 0);
        updateOrbGravity(frame, extractors, T_WS, gravityWorld);
        for (std::size_t camera = 0; camera < 2; ++camera) {
            OrbFeatureExtractor direct(frame.geometry(camera), options());
            updateOrbGravity(direct, T_WS * *frame.T_SC(camera), gravityWorld);
            Eigen::Matrix2d a, b; bool aa, ab;
            require(extractors[camera]->samplingWarp(centre(), a, aa) && direct.samplingWarp(centre(), b, ab),
                "Each rig camera must have a valid gravity warp");
            require((a - b).norm() < 1e-10, "Rig gravity must agree with composed T_WC");
        }
        MultiFrame next(rig, cgmli::Time()); bindOrb(next, extractors);
        auto wrong = extractors; wrong[1] = std::make_shared<OrbFeatureExtractor>(pinhole(), options());
        bool rejected = false;
        try { bindOrb(next, wrong); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Reusing an extractor with another calibration must fail");
    }
}
#ifdef CGMLI_EMBED_MODULE_TESTS
void cgmli::debug::registerExtractionCases(std::vector<cgmli::debug::ModuleCase>& cases) {
    cases.push_back({"extraction","defaultModeAndStandardOptOut",[]() { defaultModeAndStandardOptOut(); }});
    cases.push_back({"extraction","missingAndInvalidGravity",[]() { missingAndInvalidGravity(); }});
    cases.push_back({"extraction","gravityRollInvariance",[]() { gravityRollInvariance(); }});
    cases.push_back({"extraction","poleFallbackAndOctaveScale",[]() { poleFallbackAndOctaveScale(); }});
    cases.push_back({"extraction","affineSmoothingAgreement",[]() { affineSmoothingAgreement(); }});
    cases.push_back({"extraction","smoothingMaskAndCompressedSupport",[]() { smoothingMaskAndCompressedSupport(); }});
    cases.push_back({"extraction","canonicalPoleOrientation",[]() { canonicalPoleOrientation(); }});
    cases.push_back({"extraction","sparsePatternAndFractionalArea",[]() { sparsePatternAndFractionalArea(); }});
    cases.push_back({"extraction","sparseAndReferenceProfiles",[]() { sparseAndReferenceProfiles(); }});
    cases.push_back({"extraction","cachedExtractionAndDynamicMasks",[]() { cachedExtractionAndDynamicMasks(); }});
    cases.push_back({"extraction","sparseSamplerAndAnalyticWarpAgreement",[]() { sparseSamplerAndAnalyticWarpAgreement(); }});
    cases.push_back({"extraction","batchedDescriptorCoordinatesMatchScalar",[]() { batchedDescriptorCoordinatesMatchScalar(); }});
    cases.push_back({"extraction","adaptiveGeometryBatchesPreserveSupport",[]() { adaptiveGeometryBatchesPreserveSupport(); }});
    cases.push_back({"extraction","layerFastAndCompactIntegralAgreement",[]() { layerFastAndCompactIntegralAgreement(); }});
    cases.push_back({"extraction","budgetDrivenFallbackPreservesCoverage",[]() { budgetDrivenFallbackPreservesCoverage(); }});
    cases.push_back({"extraction","fullWarpedMaskSupport",[]() { fullWarpedMaskSupport(); }});
    cases.push_back({"extraction","rejectExcessiveStretch",[]() { rejectExcessiveStretch(); }});
    cases.push_back({"extraction","rearHemisphereEucm",[]() { rearHemisphereEucm(); }});
#ifdef CGMLI_ENABLE_CUDA_ORB
    if(hasCudaDevice()) cases.push_back({"extraction","cudaExtractionAgreement",[]() { cudaExtractionAgreement(); }});
#endif
    cases.push_back({"extraction","frameAdapterAndRepeatability",[]() { frameAdapterAndRepeatability(); }});
    cases.push_back({"extraction","multiCameraGravityAndReuse",[]() { multiCameraGravityAndReuse(); }});
    cases.push_back({"jacobian","projectionDerivativeAgreement",[]() { projectionDerivativeAgreement(); }});
    cases.push_back({"extraction","cameraVariant_equidistant",[]() { cameraVariant(cameras::EquidistantDistortion(-0.01,0.001,0,0)); }});
    cases.push_back({"extraction","cameraVariant_radtan",[]() { cameraVariant(cameras::RadialTangentialDistortion(-0.01,0.001,0.0001,0.0001)); }});
    cases.push_back({"extraction","cameraVariant_radtan8",[]() { cameraVariant(cameras::RadialTangentialDistortion8(-0.01,0.001,0.0001,0.0001,0,0,0,0)); }});
}
#else
int main() {
    static_assert(std::is_base_of<cv::FeatureDetector, OrbFeatureExtractor>::value, "Frame detector interface");
    static_assert(std::is_base_of<cv::DescriptorExtractor, OrbFeatureExtractor>::value, "Frame extractor interface");
    try {
        defaultModeAndStandardOptOut(); missingAndInvalidGravity(); gravityRollInvariance(); poleFallbackAndOctaveScale();
        affineSmoothingAgreement(); smoothingMaskAndCompressedSupport(); canonicalPoleOrientation();
        sparsePatternAndFractionalArea(); sparseAndReferenceProfiles();
        cachedExtractionAndDynamicMasks(); sparseSamplerAndAnalyticWarpAgreement(); batchedDescriptorCoordinatesMatchScalar(); adaptiveGeometryBatchesPreserveSupport(); layerFastAndCompactIntegralAgreement(); budgetDrivenFallbackPreservesCoverage();
        projectionDerivativeAgreement(); fullWarpedMaskSupport(); rejectExcessiveStretch();
        rearHemisphereEucm();
#ifdef CGMLI_ENABLE_CUDA_ORB
        if(hasCudaDevice()) cudaExtractionAgreement();
#endif
        cameraVariant(cameras::EquidistantDistortion(-0.01, 0.001, 0, 0));
        cameraVariant(cameras::RadialTangentialDistortion(-0.01, 0.001, 0.0001, 0.0001));
        cameraVariant(cameras::RadialTangentialDistortion8(-0.01, 0.001, 0.0001, 0.0001, 0, 0, 0, 0));
        frameAdapterAndRepeatability(); multiCameraGravityAndReuse();
        std::cout << "Camera-aware ORB regressions passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}

#endif // CGMLI_EMBED_MODULE_TESTS
