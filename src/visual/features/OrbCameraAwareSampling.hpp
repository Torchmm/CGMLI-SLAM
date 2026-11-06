#pragma once
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>

namespace cgmli {
    namespace features {
        namespace detail {
            // CPU reference layout: a 31x31 patch padded by the 7-tap Gaussian radius.
            // A CUDA implementation can rectify/filter tiles and reuse their values for
            // all 256 pairs. Large supersampled tiles require tiling or global scratch.
            inline constexpr int kOrbPatchRadius = 15;
            inline constexpr int kOrbBlurRadius = 3;
            inline constexpr int kOrbTileRadius = kOrbPatchRadius + kOrbBlurRadius;
            inline constexpr int kOrbTileSize = 2 * kOrbTileRadius + 1;
            inline constexpr double kOrbSigma = 2.0;
            inline constexpr int kOrbMaximumSupersampling = 8;

            inline Eigen::Vector2d singularValues2x2(const Eigen::Matrix2d& matrix) {
                // Singular values only: no U/V matrices or iterative SVD. Compute
                // the small value through |det(A)|/sigma_max to avoid cancellation.
                if(!matrix.allFinite()) return Eigen::Vector2d::Zero();
                const double a=matrix.col(0).squaredNorm(),c=matrix.col(1).squaredNorm();
                const double b=matrix.col(0).dot(matrix.col(1));
                const double largest=std::sqrt(std::max(0.0,0.5*(a+c+std::hypot(a-c,2*b))));
                if(!std::isfinite(largest) || largest==0) return Eigen::Vector2d::Zero();
                const double determinant=matrix(0,0)*matrix(1,1)-matrix(0,1)*matrix(1,0);
                return Eigen::Vector2d(largest,std::min(largest,std::abs(determinant)/largest));
            }

            inline int samplingResolution(const Eigen::Matrix2d& warp) {
                if (!warp.allFinite()) return 0;
                const Eigen::Matrix2d covariance = warp * warp.transpose();
                const double trace = covariance.trace();
                const double difference = covariance(0, 0) - covariance(1, 1);
                const double discriminant = std::hypot(difference, 2 * covariance(0, 1));
                const double stretch = std::sqrt(std::max(0.0, 0.5 * (trace + discriminant)));
                if (!std::isfinite(stretch) || stretch <= 0 || stretch > kOrbMaximumSupersampling) return 0;
                // Rectification spacing in image space is <= one pixel. A stretched patch
                // must not alias BEFORE Gaussian filtering by skipping source pixels.
                // Avoid resolution changes caused only by roundoff during a rigid rotation.
                return std::max(1, static_cast<int>(std::ceil(stretch - 1e-6)));
            }

            inline int patchSamplingScale(const cv::Mat& patch) {
                if (patch.type() != CV_32FC1 || patch.rows != patch.cols ||
                    (patch.cols - 1) % (2 * kOrbTileRadius) != 0) return 0;
                const int scale = (patch.cols - 1) / (2 * kOrbTileRadius);
                return scale >= 1 && scale <= kOrbMaximumSupersampling ? scale : 0;
            }

            inline cv::Mat invalidMaskIntegral(const cv::Mat& mask) {
                cv::Mat invalid, integral;
                cv::compare(mask, 0, invalid, cv::CMP_EQ);
                cv::integral(invalid, integral, CV_64F);
                return integral;
            }

            inline bool warpedSupportValid(const cv::Size& size, const cv::Mat& invalidIntegral,
                const Eigen::Vector2d& centre, const Eigen::Matrix2d& warp, double radius = kOrbTileRadius) {
                if (!centre.allFinite() || !warp.allFinite() || !std::isfinite(radius) || radius <= 0 ||
                    invalidIntegral.type() != CV_64FC1 || invalidIntegral.cols != size.width + 1 ||
                    invalidIntegral.rows != size.height + 1 || samplingResolution(warp) == 0) return false;
                Eigen::Vector2d minimum = centre, maximum = centre;
                for (int y : {-1, 1}) for (int x : {-1, 1}) {
                    const Eigen::Vector2d corner = centre + warp * Eigen::Vector2d(x * radius, y * radius);
                    minimum = minimum.cwiseMin(corner); maximum = maximum.cwiseMax(corner);
                }
                // W warps Gaussian support too. Only bilinear support is added in image coordinates.
                // Test bounds before converting potentially large coordinates to integer indices.
                if (!minimum.allFinite() || !maximum.allFinite() || minimum.x() < 0 || minimum.y() < 0 ||
                    maximum.x() >= size.width - 1 || maximum.y() >= size.height - 1) return false;
                const int left = static_cast<int>(std::floor(minimum.x()));
                const int top = static_cast<int>(std::floor(minimum.y()));
                const int right = static_cast<int>(std::floor(maximum.x())) + 2;
                const int bottom = static_cast<int>(std::floor(maximum.y())) + 2;
                const double masked = invalidIntegral.at<double>(bottom, right) - invalidIntegral.at<double>(top, right) -
                    invalidIntegral.at<double>(bottom, left) + invalidIntegral.at<double>(top, left);
                // Conservative whole-area mask enclosure, including holes between sample positions.
                return masked == 0;
            }

            inline bool bilinearIntensity(const cv::Mat& image, double x, double y, float& value) {
                if (image.type() != CV_8UC1 || !std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 ||
                    x >= image.cols - 1 || y >= image.rows - 1) return false;
                const int ix = static_cast<int>(std::floor(x)), iy = static_cast<int>(std::floor(y));
                const double dx = x - ix, dy = y - iy;
                const uchar* top = image.ptr<uchar>(iy);
                const uchar* bottom = image.ptr<uchar>(iy + 1);
                value = static_cast<float>((1 - dy) * ((1 - dx) * top[ix] + dx * top[ix + 1]) +
                    dy * ((1 - dx) * bottom[ix] + dx * bottom[ix + 1]));
                return true;
            }

            inline bool rectifyPatch(const cv::Mat& image, const cv::Mat& invalidIntegral,
                const Eigen::Vector2d& centre, const Eigen::Matrix2d& warp, cv::Mat& patch) {
                if (!warpedSupportValid(image.size(), invalidIntegral, centre, warp)) return false;
                const int scale = samplingResolution(warp), radius = kOrbTileRadius * scale;
                patch.create(2 * radius + 1, 2 * radius + 1, CV_32FC1);
                for (int y = -radius; y <= radius; ++y) {
                    float* row = patch.ptr<float>(y + radius);
                    for (int x = -radius; x <= radius; ++x) {
                        const Eigen::Vector2d pixel = centre + warp * Eigen::Vector2d(double(x) / scale, double(y) / scale);
                        if (!bilinearIntensity(image, pixel.x(), pixel.y(), row[x + radius])) return false;
                    }
                }
                return true;
            }

            inline void smoothRectifiedPatch(const cv::Mat& patch, cv::Mat& filtered) {
                // In image space the nominal Gaussian covariance is sigma^2 * W * W^T.
                // This discrete reference transforms both sample positions and filter offsets,
                // rather than approximating a nonsymmetric W with one scalar sigma.
                const int scale = patchSamplingScale(patch);
                if (scale == 0) throw std::invalid_argument("Invalid camera-aware ORB reference tile");
                const int size = 2 * kOrbBlurRadius * scale + 1;
                cv::GaussianBlur(patch, filtered, cv::Size(size, size), kOrbSigma * scale, kOrbSigma * scale, cv::BORDER_CONSTANT);
                // Descriptor coordinates are inside the central 31x31 crop, so no synthetic
                // border pixels enter a descriptor's filter support.
            }

            inline bool rectifiedTextureOrientation(const cv::Mat& patch, double& angleDegrees) {
                const int scale = patchSamplingScale(patch);
                if (scale == 0) return false;
                const int radius = kOrbPatchRadius * scale, centre = kOrbTileRadius * scale;
                double momentX = 0, momentY = 0, mass = 0;
                for (int y = -radius; y <= radius; ++y) {
                    const int halfWidth = static_cast<int>(std::floor(std::sqrt(radius * radius - y * y)));
                    const float* row = patch.ptr<float>(y + centre);
                    for (int x = -halfWidth; x <= halfWidth; ++x) {
                        const double value = row[x + centre];
                        momentX += x * value; momentY += y * value; mass += std::abs(value);
                    }
                }
                // Reject an unobservable texture direction instead of returning atan2(0,0).
                if (!std::isfinite(momentX) || !std::isfinite(momentY) ||
                    std::hypot(momentX, momentY) <= 1e-6 * std::max(1.0, radius * mass)) return false;
                angleDegrees = std::atan2(momentY, momentX) * 180.0 / 3.14159265358979323846;
                return true;
            }
        }
    }
}
