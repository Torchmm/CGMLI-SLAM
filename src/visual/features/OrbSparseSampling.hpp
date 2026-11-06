#pragma once
#include <visual/features/OrbBriefPattern.hpp>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>

namespace cgmli {
    namespace features {
        namespace detail {
            struct SparseOrbPattern {
                int points[512][2]{};
                int pairs[256][2]{};
                int count = 0;
            };
            constexpr SparseOrbPattern uniqueOrbPattern() {
                SparseOrbPattern result;
                int indices[31][31]{}; // index+1; bounded linear work at compile time.
                for (int bit = 0; bit < 256; ++bit) for (int end = 0; end < 2; ++end) {
                    const int x = kOrbPattern[bit][2 * end], y = kOrbPattern[bit][2 * end + 1];
                    int& entry = indices[y + 15][x + 15];
                    if (entry == 0) {
                        result.points[result.count][0] = x; result.points[result.count][1] = y;
                        entry = ++result.count;
                    }
                    result.pairs[bit][end] = entry - 1;
                }
                return result;
            }
            inline constexpr SparseOrbPattern kSparseOrbPattern = uniqueOrbPattern();
            static_assert(kSparseOrbPattern.count == 375, "OpenCV ORB pattern must contain 375 unique positions");

            inline Eigen::Vector2d sparseBoxHalfSize(const Eigen::Matrix2d& warp) {
                // Uniform-box variance h^2/3 matches the diagonal of sigma_ref^2 W W^T,
                // sigma_ref=2. Correlation/shear of the filter itself is intentionally approximated.
                return Eigen::Vector2d(std::max(0.5, 3.4641016151377546 * warp.row(0).norm()),
                    std::max(0.5, 3.4641016151377546 * warp.row(1).norm()));
            }
            inline bool sparseSupportValid(const cv::Size& size, const cv::Mat& invalidIntegral,
                const Eigen::Vector2d& centre, const Eigen::Matrix2d& warp, bool anyRotation = false) {
                if (!warp.allFinite() || !centre.allFinite() || invalidIntegral.type() != CV_64FC1 ||
                    invalidIntegral.cols != size.width + 1 || invalidIntegral.rows != size.height + 1) return false;
                const Eigen::Vector2d half = sparseBoxHalfSize(warp);
                Eigen::Vector2d extent;
                // Full reference 31x31 enclosure, including texture orientation probes.
                for (int axis = 0; axis < 2; ++axis)
                    extent[axis] = (anyRotation ? 21.213203435596427 * warp.row(axis).norm()
                        : 15 * warp.row(axis).cwiseAbs().sum()) + half[axis];
                const Eigen::Vector2d lower = centre + Eigen::Vector2d::Constant(0.5) - extent;
                const Eigen::Vector2d upper = centre + Eigen::Vector2d::Constant(0.5) + extent;
                if (!lower.allFinite() || !upper.allFinite() || lower.x() < 0 || lower.y() < 0 ||
                    upper.x() > size.width || upper.y() > size.height) return false;
                const int left = static_cast<int>(std::floor(lower.x())), top = static_cast<int>(std::floor(lower.y()));
                const int right = static_cast<int>(std::ceil(upper.x())), bottom = static_cast<int>(std::ceil(upper.y()));
                return invalidIntegral.at<double>(bottom, right) - invalidIntegral.at<double>(top, right) -
                    invalidIntegral.at<double>(bottom, left) + invalidIntegral.at<double>(top, left) == 0;
            }
            inline double fractionalIntegral(const cv::Mat& integral, double x, double y) {
                // Coordinates are pixel-cell boundaries. Bilinear interpolation of the prefix
                // table exactly integrates the piecewise-constant pixel image at fractional boundaries.
                const int ix = std::min(static_cast<int>(std::floor(x)), integral.cols - 2);
                const int iy = std::min(static_cast<int>(std::floor(y)), integral.rows - 2);
                const double dx = x - ix, dy = y - iy;
                const double* a = integral.ptr<double>(iy), * b = integral.ptr<double>(iy + 1);
                return (1 - dy) * ((1 - dx) * a[ix] + dx * a[ix + 1]) +
                    dy * ((1 - dx) * b[ix] + dx * b[ix + 1]);
            }
            template<class Pixel> class BasicSparseBoxSampler {
            public:
                BasicSparseBoxSampler(const cv::Mat& integral,const Eigen::Vector2d& half)
                    : integral_(integral),half_(half),area_(4*half.x()*half.y()),
                      valid_(integral.type()==cv::traits::Type<Pixel>::value && integral.cols>=2 && integral.rows>=2
                          && half.allFinite() && half.minCoeff()>=0.5) {}
                bool valid() const { return valid_; }
                bool sample(const Eigen::Vector2d& pixel,float& value) const {
                    if(!valid_ || !pixel.allFinite()) return false;
                    const double x0=pixel.x()+0.5-half_.x(),x1=pixel.x()+0.5+half_.x();
                    const double y0=pixel.y()+0.5-half_.y(),y1=pixel.y()+0.5+half_.y();
                    if(x0<0 || y0<0 || x1>integral_.cols-1 || y1>integral_.rows-1) return false;
                    // Four boundaries, not eight repeated floor operations for
                    // four corners. Preserve the reference interpolation/sum order.
                    const int ix0=std::min(int(std::floor(x0)),integral_.cols-2);
                    const int ix1=std::min(int(std::floor(x1)),integral_.cols-2);
                    const int iy0=std::min(int(std::floor(y0)),integral_.rows-2);
                    const int iy1=std::min(int(std::floor(y1)),integral_.rows-2);
                    const double dx0=x0-ix0,dx1=x1-ix1,dy0=y0-iy0,dy1=y1-iy1;
                    const auto query=[](const Pixel* a,const Pixel* b,int x,double dx,double dy) {
                        return (1-dy)*((1-dx)*a[x]+dx*a[x+1])+dy*((1-dx)*b[x]+dx*b[x+1]);
                    };
                    const Pixel* top0=integral_.ptr<Pixel>(iy0); const Pixel* bottom0=integral_.ptr<Pixel>(iy0+1);
                    const Pixel* top1=integral_.ptr<Pixel>(iy1); const Pixel* bottom1=integral_.ptr<Pixel>(iy1+1);
                    const double sum=query(top1,bottom1,ix1,dx1,dy1)-query(top1,bottom1,ix0,dx0,dy1)
                        -query(top0,bottom0,ix1,dx1,dy0)+query(top0,bottom0,ix0,dx0,dy0);
                    value=static_cast<float>(sum/area_); return std::isfinite(value);
                }
            private:
                const cv::Mat& integral_;
                Eigen::Vector2d half_;
                double area_;
                bool valid_;
            };
            using SparseBoxSampler=BasicSparseBoxSampler<double>; // Independent double reference remains available.
            inline bool sparseBoxIntensity(const cv::Mat& integral, const Eigen::Vector2d& pixel,
                const Eigen::Vector2d& half, float& value) {
                if(integral.type()==CV_32SC1) return BasicSparseBoxSampler<int>(integral,half).sample(pixel,value);
                return SparseBoxSampler(integral,half).sample(pixel,value);
            }
            template<class Pixel> inline bool sparseTextureOrientationTyped(const cv::Mat& integral, const Eigen::Vector2d& centre,
                const Eigen::Matrix2d& warp, double& angleDegrees) {
                const Eigen::Vector2d half = sparseBoxHalfSize(warp);
                const BasicSparseBoxSampler<Pixel> sampler(integral,half); if(!sampler.valid()) return false;
                double mx = 0, my = 0, mass = 0;
                // 29 fixed probes in a reference-plane disk; no dense pole tile either.
                for (int y = -3; y <= 3; ++y) for (int x = -3; x <= 3; ++x) {
                    if (x * x + y * y > 9) continue;
                    float intensity;
                    if (!sampler.sample(centre + warp * Eigen::Vector2d(4 * x, 4 * y), intensity)) return false;
                    mx += x * intensity; my += y * intensity; mass += std::abs(intensity);
                }
                if (std::hypot(mx, my) <= 1e-6 * std::max(1.0, 3 * mass)) return false;
                angleDegrees = std::atan2(my, mx) * 180.0 / 3.14159265358979323846;
                return true;
            }
            inline bool sparseTextureOrientation(const cv::Mat& integral,const Eigen::Vector2d& centre,
                const Eigen::Matrix2d& warp,double& angleDegrees) {
                return integral.type()==CV_32SC1?sparseTextureOrientationTyped<int>(integral,centre,warp,angleDegrees)
                    :sparseTextureOrientationTyped<double>(integral,centre,warp,angleDegrees);
            }
            template<class Pixel> inline bool sparseOrbDescriptorTyped(const cv::Mat& integral, const Eigen::Vector2d& centre,
                const Eigen::Matrix2d& warp, std::array<uchar, 32>& bits,bool batchCoordinates=true) {
                const Eigen::Vector2d half = sparseBoxHalfSize(warp);
                const BasicSparseBoxSampler<Pixel> sampler(integral,half); if(!sampler.valid()) return false;
                std::array<float, kSparseOrbPattern.count> values{};
                using Coordinates=Eigen::Matrix<double,2,kSparseOrbPattern.count>;
                static const Coordinates pattern=[]() {
                    Coordinates p;
                    for(int i=0;i<kSparseOrbPattern.count;++i) {
                        p(0,i)=kSparseOrbPattern.points[i][0]; p(1,i)=kSparseOrbPattern.points[i][1];
                    }
                    return p;
                }();
                Coordinates pixels;
                if(batchCoordinates) { pixels.noalias()=warp*pattern; pixels.colwise()+=centre; }
                for(int i=0;i<kSparseOrbPattern.count;++i) {
                    const Eigen::Vector2d pixel=batchCoordinates?Eigen::Vector2d(pixels.col(i)):
                        Eigen::Vector2d(centre+warp*Eigen::Vector2d(kSparseOrbPattern.points[i][0],kSparseOrbPattern.points[i][1]));
                    if(!sampler.sample(pixel,values[i])) return false;
                }
                bits.fill(0);
                for (int bit = 0; bit < 256; ++bit)
                    if (values[kSparseOrbPattern.pairs[bit][0]] < values[kSparseOrbPattern.pairs[bit][1]])
                        bits[bit / 8] |= static_cast<uchar>(1u << (bit % 8));
                return true;
            }
            inline bool sparseOrbDescriptor(const cv::Mat& integral,const Eigen::Vector2d& centre,
                const Eigen::Matrix2d& warp,std::array<uchar,32>& bits,bool batchCoordinates=true) {
                // Dispatch once per feature, not once per sampling location. Int32
                // prefix sums are exact; interpolation still uses double arithmetic.
                return integral.type()==CV_32SC1?sparseOrbDescriptorTyped<int>(integral,centre,warp,bits,batchCoordinates)
                    :sparseOrbDescriptorTyped<double>(integral,centre,warp,bits,batchCoordinates);
            }
        }
    }
}
