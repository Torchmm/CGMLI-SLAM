#include <visual/features/OrbMatcher.hpp>
#include <visual/Frame.hpp>
#include <visual/MultiFrame.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <array>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <map>
#include <limits>
#include <numeric>
#ifdef CGMLI_ENABLE_CUDA_ORB
#include "OrbMatchCuda.hpp"
#endif

namespace cgmli {
    namespace features {
        namespace {
            void validateDescriptors(const cv::Mat& descriptors) {
                if (!descriptors.empty() && (descriptors.type() != CV_8UC1 || descriptors.cols != 32))
                    throw std::invalid_argument("ORB matching requires 32-byte CV_8UC1 WTA_K=2 descriptors");
            }
            void validatePairs(const cv::Mat& mask, int queries, int trains) {
                if (!mask.empty() && (mask.type() != CV_8UC1 || mask.rows != queries || mask.cols != trains))
                    throw std::invalid_argument("Allowed-pair mask must be query rows x train rows, CV_8UC1");
            }
            void validateRegion(const cv::Mat& mask, const cameras::CameraBase* camera) {
                if (!mask.empty() && (!camera || mask.type() != CV_8UC1 ||
                    mask.cols != static_cast<int>(camera->imageWidth()) || mask.rows != static_cast<int>(camera->imageHeight())))
                    throw std::invalid_argument("Region mask requires a matching camera and calibrated image dimensions");
            }
            bool validPixel(const cv::Point2f& point, const cameras::CameraBase* camera, const cv::Mat& region) {
                if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0 || point.y < 0) return false;
                if (camera && (point.x >= camera->imageWidth() || point.y >= camera->imageHeight())) return false;
                if (!camera) return region.empty();
                const int x = static_cast<int>(point.x), y = static_cast<int>(point.y);
                if (!region.empty() && region.at<uchar>(y, x) == 0) return false;
                if (camera && !camera->mask().empty() && camera->mask().at<uchar>(y, x) == 0) return false;
                return true;
            }
            bool bearingFor(const cv::KeyPoint& keypoint, const cameras::CameraBase* camera,
                const cv::Mat& region, Eigen::Vector3d& bearing, bool needBearing) {
                if (!validPixel(keypoint.pt, camera, region)) return false;
                if (!needBearing) return true;
                if (!camera->backProject(Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y), &bearing) || !bearing.allFinite()) return false;
                const double norm = bearing.stableNorm();
                if (!std::isfinite(norm) || norm < 1e-12) return false;
                bearing /= norm; return true;
            }
            bool ratioPass(const std::vector<cv::DMatch>& neighbors, const OrbMatcherOptions& o) {
                if (neighbors.empty() || neighbors[0].distance > o.maxHammingDistance) return false;
                if (neighbors.size() < 2) return o.allowSingleCandidate;
                // Strict comparison also rejects equal zero-distance duplicate descriptors.
                return neighbors[0].distance < o.ratioThreshold * neighbors[1].distance;
            }
            struct TopTwo {
                int index[2]{-1,-1}, distance[2]{257,257};
                void add(int other,int value) {
                    if(other==index[0]) { distance[0]=std::min(distance[0],value); return; }
                    if(other==index[1]) {
                        distance[1]=std::min(distance[1],value);
                        if(distance[1]<distance[0] || (distance[1]==distance[0] && index[1]<index[0])) {
                            std::swap(index[0],index[1]); std::swap(distance[0],distance[1]);
                        }
                        return;
                    }
                    const auto better=[&](int slot) { return value<distance[slot]
                        || (value==distance[slot] && (index[slot]<0 || other<index[slot])); };
                    if(better(0)) { index[1]=index[0]; distance[1]=distance[0]; index[0]=other; distance[0]=value; }
                    else if(better(1)) { index[1]=other; distance[1]=value; }
                }
                bool passes(const OrbMatcherOptions& options) const {
                    return index[0]>=0 && distance[0]<=options.maxHammingDistance
                        && (index[1]<0?options.allowSingleCandidate:distance[0]<options.ratioThreshold*distance[1]);
                }
            };
            using DescriptorWords=OrbDescriptorWords;
            int hamming(const DescriptorWords& a,const DescriptorWords& b) {
                int total=0;
                // Portable SWAR popcount: no CPU-specific POPCNT requirement and no
                // per-comparison cv::Mat construction or descriptor alignment assumption.
                for(size_t word=0;word<4;++word) {
                    uint64_t x=a[word]^b[word];
                    x-= (x>>1)&UINT64_C(0x5555555555555555);
                    x=(x&UINT64_C(0x3333333333333333))+((x>>2)&UINT64_C(0x3333333333333333));
                    x=(x+(x>>4))&UINT64_C(0x0f0f0f0f0f0f0f0f);
                    total+=int((x*UINT64_C(0x0101010101010101))>>56);
                }
                return total;
            }
            void validateCameraIndex(const MultiFrame& frame, std::size_t camera) {
                if (camera >= frame.numFrames()) throw std::out_of_range("ORB camera index out of range");
            }
        }

        OrbMatcher::OrbMatcher(const OrbMatcherOptions& options) : cv::BFMatcher(cv::NORM_HAMMING, false), options_(options) {
            if (options_.maxHammingDistance < 0 || options_.maxHammingDistance > 256 ||
                !std::isfinite(options_.ratioThreshold) || options_.ratioThreshold <= 0 || options_.ratioThreshold >= 1 ||
                options_.maxOctaveDifference < -1 ||
                !std::isfinite(options_.epipolarThresholdRadians) || options_.epipolarThresholdRadians <= 0 || options_.epipolarThresholdRadians >= 0.5 ||
                !std::isfinite(options_.rotationThresholdRadians) || options_.rotationThresholdRadians <= 0 || options_.rotationThresholdRadians >= 0.5 ||
                !std::isfinite(options_.minimumBaseline) || options_.minimumBaseline < 0)
                throw std::invalid_argument("Invalid ORB matcher options");
        }
        cv::Ptr<cv::DescriptorMatcher> OrbMatcher::clone(bool emptyTrainData) const {
            auto copy = cv::makePtr<OrbMatcher>(options_);
            if (!emptyTrainData) {
                std::vector<cv::Mat> descriptors;
                for (const auto& mat : trainDescCollection) descriptors.push_back(mat.clone());
                for (const auto& mat : utrainDescCollection) descriptors.push_back(mat.getMat(cv::ACCESS_READ).clone());
                if (!descriptors.empty()) copy->add(descriptors);
            }
            return copy;
        }
        void OrbMatcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches,
            int k, cv::InputArrayOfArrays masks, bool compactResult) {
            validateDescriptors(queryDescriptors.getMat());
            for (const auto& descriptors : trainDescCollection) validateDescriptors(descriptors);
            for (const auto& descriptors : utrainDescCollection) validateDescriptors(descriptors.getMat(cv::ACCESS_READ));
            cv::BFMatcher::knnMatchImpl(queryDescriptors, matches, k, masks, compactResult);
        }
        void OrbMatcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches,
            float maxDistance, cv::InputArrayOfArrays masks, bool compactResult) {
            validateDescriptors(queryDescriptors.getMat());
            for (const auto& descriptors : trainDescCollection) validateDescriptors(descriptors);
            for (const auto& descriptors : utrainDescCollection) validateDescriptors(descriptors.getMat(cv::ACCESS_READ));
            cv::BFMatcher::radiusMatchImpl(queryDescriptors, matches, maxDistance, masks, compactResult);
        }
        void OrbMatcher::matchFiltered(const cv::Mat& query, const cv::Mat& train,
            std::vector<cv::DMatch>& matches, const cv::Mat& allowedPairs) const {
            matches.clear(); validateDescriptors(query); validateDescriptors(train);
            validatePairs(allowedPairs, query.rows, train.rows);
            if (query.empty() || train.empty()) return;
            std::vector<std::vector<cv::DMatch>> forward, reverse;
            knnMatch(query, train, forward, 2, allowedPairs, false);
            if (options_.mutualCheck) {
                cv::Mat reversedMask;
                if (!allowedPairs.empty()) cv::transpose(allowedPairs, reversedMask);
                knnMatch(train, query, reverse, options_.reverseRatioTest ? 2 : 1, reversedMask, false);
            }
            for (const auto& neighbors : forward) {
                if (!ratioPass(neighbors, options_)) continue;
                const auto& best = neighbors[0];
                if (options_.mutualCheck) {
                    const auto& back = reverse[best.trainIdx];
                    if (back.empty() || back[0].trainIdx != best.queryIdx ||
                        (options_.reverseRatioTest && !ratioPass(back, options_))) continue;
                }
                matches.push_back(best);
            }
            // Mutual NN guarantees a unique train feature. Without it, resolve collisions
            // by Hamming distance, deterministically, so one observation is never reused.
            std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
                if (a.distance != b.distance) return a.distance < b.distance;
                return a.queryIdx != b.queryIdx ? a.queryIdx < b.queryIdx : a.trainIdx < b.trainIdx;
                });
            std::vector<bool> used(train.rows, false);
            matches.erase(std::remove_if(matches.begin(), matches.end(), [&](const cv::DMatch& match) {
                if (used[match.trainIdx]) return true;
                used[match.trainIdx] = true; return false;
                }), matches.end());
            std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {return a.queryIdx < b.queryIdx; });
        }

        OrbTrainIndex OrbMatcher::prepareTrainIndex(const std::vector<cv::KeyPoint>& points,
            const cameras::CameraBase* camera,float radius) {
            OrbTrainIndex index; index.count=points.size(); index.keypoints=points.data(); index.camera=camera;
            index.cellSize=std::isfinite(radius)?std::max(16.0,double(radius)):16.0;
            if(camera) {
                index.width=int(camera->imageWidth()); index.height=int(camera->imageHeight());
                index.columns=std::max(1,int(std::ceil(index.width/index.cellSize)));
                index.rows=std::max(1,int(std::ceil(index.height/index.cellSize)));
                if(uint64_t(index.columns)*uint64_t(index.rows)<=1000000) index.grid.resize(size_t(index.columns)*index.rows);
            }
            for(size_t j=0;j<points.size();++j) if(validPixel(points[j].pt,camera,{})) {
                index.octaves[points[j].octave].push_back(int(j));
                if(!index.grid.empty()) index.grid[size_t(int(points[j].pt.y/index.cellSize))*index.columns
                    +int(points[j].pt.x/index.cellSize)].push_back(int(j));
            }
            return index;
        }
        void OrbMatcher::matchGuided(const cv::Mat& query, const cv::Mat& train,
            const std::vector<cv::KeyPoint>& queryPoints, const std::vector<cv::KeyPoint>& trainPoints,
            const OrbMatchGuidance& guidance, std::vector<cv::DMatch>& matches, OrbMatchStatistics* statistics) const {
            matches.clear(); validateDescriptors(query); validateDescriptors(train);
            if(statistics) *statistics={};
            const auto setupStarted=std::chrono::steady_clock::now();
            const auto milliseconds=[](const auto& start) { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); };
            if (query.rows != static_cast<int>(queryPoints.size()) || train.rows != static_cast<int>(trainPoints.size()))
                throw std::invalid_argument("ORB descriptor rows must align with keypoint indices");
            validatePairs(guidance.allowedPairs, query.rows, train.rows);
            validateRegion(guidance.queryRegionMask, guidance.queryCamera.get());
            validateRegion(guidance.trainRegionMask, guidance.trainCamera.get());
            if((!guidance.queryEnabled.empty() && guidance.queryEnabled.size()!=queryPoints.size())
                || (!guidance.trainEnabled.empty() && guidance.trainEnabled.size()!=trainPoints.size())
                || (guidance.queryBearings && guidance.queryBearings->size()!=queryPoints.size())
                || (guidance.trainBearings && guidance.trainBearings->size()!=trainPoints.size())
                || (guidance.queryWords && guidance.queryWords->size()!=queryPoints.size())
                || (guidance.trainWords && guidance.trainWords->size()!=trainPoints.size()))
                throw std::invalid_argument("Guided feature gates/ray caches must match descriptor rows");
            if(guidance.queryIdentities && guidance.queryIdentities->size()!=queryPoints.size())
                throw std::invalid_argument("Map descriptor identities must match query rows");
            if(guidance.queryPriorities && guidance.queryPriorities->size()!=queryPoints.size())
                throw std::invalid_argument("Query priorities must match descriptor rows");
            if (!guidance.predictedTrainPixels.empty() &&
                (guidance.predictedTrainPixels.size() != queryPoints.size() ||
                    !std::isfinite(guidance.searchRadiusPixels) || guidance.searchRadiusPixels <= 0))
                throw std::invalid_argument("Guided matching needs one prediction per query and a positive search radius");
            if (guidance.useSphericalGeometry && (!guidance.queryCamera || !guidance.trainCamera))
                throw std::invalid_argument("Spherical ORB matching requires both camera models");
            if (guidance.useSphericalGeometry && (!guidance.T_train_query.T().allFinite() ||
                std::abs(guidance.T_train_query.q().squaredNorm() - 1.0) > 1e-6))
                throw std::invalid_argument("Invalid ORB guidance transform");
            if (query.empty() || train.empty()) return;
            OrbBearings bearingsQ(query.rows), bearingsT(train.rows);
            std::vector<bool> validQ(query.rows), validT(train.rows);
            const auto prepare=[&](const auto& keys,const auto& camera,const cv::Mat& region,
                const auto& enabled,const OrbBearings* cached,OrbBearings& rays,std::vector<bool>& valid) {
                for(size_t i=0;i<keys.size();++i) {
                    if(!enabled.empty() && !enabled[i]) continue;
                    if(cached && guidance.useSphericalGeometry) {
                        rays[i]=(*cached)[i];
                        valid[i]=validPixel(keys[i].pt,camera.get(),region) && rays[i].allFinite()
                            && std::abs(rays[i].squaredNorm()-1.0)<1e-6;
                    } else valid[i]=bearingFor(keys[i],camera.get(),region,rays[i],guidance.useSphericalGeometry);
                }
            };
            prepare(queryPoints,guidance.queryCamera,guidance.queryRegionMask,guidance.queryEnabled,guidance.queryBearings,bearingsQ,validQ);
            prepare(trainPoints,guidance.trainCamera,guidance.trainRegionMask,guidance.trainEnabled,guidance.trainBearings,bearingsT,validT);

            OrbWordRows localQ,localT;
            if(!guidance.queryWords) { localQ.resize(query.rows); for(int i=0;i<query.rows;++i) std::memcpy(localQ[i].data(),query.ptr(i),32); }
            if(!guidance.trainWords) { localT.resize(train.rows); for(int j=0;j<train.rows;++j) std::memcpy(localT[j].data(),train.ptr(j),32); }
            const auto& wordsQ=guidance.queryWords?*guidance.queryWords:localQ;
            const auto& wordsT=guidance.trainWords?*guidance.trainWords:localT;
            std::vector<int> groups(query.rows); int groupCount=query.rows;
            if(guidance.queryIdentities) {
                std::map<uint64_t,int> identities; groupCount=0;
                for(int i=0;i<query.rows;++i) {
                    auto inserted=identities.emplace((*guidance.queryIdentities)[i],groupCount);
                    if(inserted.second) ++groupCount;
                    groups[i]=inserted.first->second;
                }
            } else std::iota(groups.begin(),groups.end(),0);
            std::vector<int> priorities(groupCount,-1);
            for(int i=0;i<query.rows;++i) {
                const int tier=guidance.queryPriorities?int((*guidance.queryPriorities)[i]):0;
                if(tier>1 || (priorities[groups[i]]>=0 && priorities[groups[i]]!=tier))
                    throw std::invalid_argument("One descriptor identity must have one ownership priority");
                priorities[groups[i]]=tier;
            }
            std::vector<TopTwo> forward(groupCount),reverse(train.rows),ownedReverse(guidance.queryPriorities?train.rows:0);
            std::vector<int> representatives(groupCount,-1);
            const double baseline=guidance.T_train_query.r().stableNorm();
            const bool rotationOnly=baseline<=options_.minimumBaseline;
            Eigen::Vector3d translation=Eigen::Vector3d::Zero();
            if(!rotationOnly) translation=guidance.T_train_query.r()/baseline;
            const double sineThreshold=std::sin(options_.epipolarThresholdRadians);
            const double cosineThreshold=std::cos(options_.rotationThresholdRadians);
            std::vector<double> trainPlaneNorm(train.rows);
            if(guidance.useSphericalGeometry && !rotationOnly) for(int j=0;j<train.rows;++j)
                if(validT[j]) trainPlaneNorm[j]=bearingsT[j].cross(translation).norm();

            // Finite metric predictions use a pixel grid. Seed tracks with no
            // depth retain full spherical search; no translation-free depth is invented.
            OrbTrainIndex localIndex;
            if(!guidance.trainIndex) localIndex=prepareTrainIndex(trainPoints,guidance.trainCamera.get(),guidance.searchRadiusPixels);
            const auto& index=guidance.trainIndex?*guidance.trainIndex:localIndex;
            const double cellSize=std::isfinite(guidance.searchRadiusPixels)?std::max(16.0,double(guidance.searchRadiusPixels)):16.0;
            if(index.count!=trainPoints.size() || index.keypoints!=trainPoints.data() || index.camera!=guidance.trainCamera.get() || index.cellSize!=cellSize
                || (guidance.trainCamera && (index.width!=int(guidance.trainCamera->imageWidth()) || index.height!=int(guidance.trainCamera->imageHeight()))))
                throw std::invalid_argument("Cached ORB index differs from points/camera/search radius");
            // The shared index contains all camera-valid points. Compact a job's
            // overlap/enable/ray gates ONCE, before queries traverse the buckets.
            // Retain original row indices/order: nearest-neighbor ties, ratios
            // and the reverse pass must see exactly the same admissible pairs.
            std::vector<std::vector<int>> filteredGrid;
            std::map<int,std::vector<int>> filteredOctaves;
            const bool filterTrain=std::find(validT.begin(),validT.end(),false)!=validT.end();
            if(filterTrain) {
                filteredGrid.resize(index.grid.size());
                for(size_t cell=0;cell<index.grid.size();++cell)
                    for(int j:index.grid[cell]) if(validT[j]) filteredGrid[cell].push_back(j);
                for(const auto& bucket:index.octaves) {
                    auto& destination=filteredOctaves[bucket.first];
                    for(int j:bucket.second) if(validT[j]) destination.push_back(j);
                }
            }
            const int columns=index.columns,rows=index.rows;
            const auto& grid=filterTrain?filteredGrid:index.grid;
            const double radiusSquared = double(guidance.searchRadiusPixels) * guidance.searchRadiusPixels;
            // Partition unpredicted spherical searches by the existing octave gate.
            // This visits exactly the admissible octave sets, preserving all top-two
            // neighbors and deterministic index tie-breaking without a dense scan.
            const auto& trainOctaves=filterTrain?filteredOctaves:index.octaves;
            if(statistics) statistics->setupMs=milliseconds(setupStarted);
            const auto traversalStarted=std::chrono::steady_clock::now();
            bool usedCuda=false;
#ifdef CGMLI_ENABLE_CUDA_ORB
            if(guidance.cudaSession && std::size_t(groupCount)*train.rows<=8000000 &&
                std::size_t(query.rows)*train.rows<=50000000) {
                detail::OrbCudaMatchInput input;
                input.queryWords=&wordsQ; input.trainWords=&wordsT;
                input.deviceQuery=guidance.deviceQuery; input.deviceTrain=guidance.deviceTrain;
                input.cacheQuery=guidance.queryWords && !guidance.queryIdentities;
                input.cacheTrain=guidance.trainWords!=nullptr;
                input.geometry=guidance.useSphericalGeometry; input.rotationOnly=rotationOnly;
                input.reverse=options_.mutualCheck; input.owned=input.reverse && !ownedReverse.empty();
                input.groups=groupCount; input.gridColumns=columns;
                input.maxOctaveDifference=options_.maxOctaveDifference;
                input.radiusSquared=radiusSquared; input.sineThreshold=sineThreshold; input.cosineThreshold=cosineThreshold;
                input.priorities=priorities;
                input.gridOffsets.push_back(0);
                for(const auto& bucket:grid) {
                    input.gridEntries.insert(input.gridEntries.end(),bucket.begin(),bucket.end());
                    input.gridOffsets.push_back(int(input.gridEntries.size()));
                }
                std::map<int,int> octaveOffsets;
                for(const auto& bucket:trainOctaves) {
                    octaveOffsets[bucket.first]=int(input.octaveEntries.size());
                    input.octaveEntries.insert(input.octaveEntries.end(),bucket.second.begin(),bucket.second.end());
                }
                if(!guidance.allowedPairs.empty()) {
                    input.allowed.reserve(std::size_t(query.rows)*train.rows);
                    for(int i=0;i<query.rows;++i) {
                        const auto* row=guidance.allowedPairs.ptr<uchar>(i);
                        input.allowed.insert(input.allowed.end(),row,row+train.rows);
                    }
                }
                input.train.resize(train.rows);
                for(int j=0;j<train.rows;++j) if(validT[j]) {
                    auto& output=input.train[j];
                    output.x=trainPoints[j].pt.x; output.y=trainPoints[j].pt.y; output.octave=trainPoints[j].octave;
                    if(input.geometry) std::copy_n(bearingsT[j].data(),3,output.bearing);
                    output.planeNorm=trainPlaneNorm[j];
                }
                input.queries.resize(query.rows);
                std::uint64_t predictedCount=0,unpredictedCount=0,ownedCount=0;
                const auto octaveOffset=[&](const auto& it) {
                    return it==octaveOffsets.end()?int(input.octaveEntries.size()):it->second;
                };
                for(int i=0;i<query.rows;++i) {
                    if(!validQ[i]) continue;
                    auto& q=input.queries[i];
                    q.predicted=!guidance.predictedTrainPixels.empty() &&
                        std::isfinite(guidance.predictedTrainPixels[i].x) && std::isfinite(guidance.predictedTrainPixels[i].y);
                    if(!guidance.predictedTrainPixels.empty() && !q.predicted && !guidance.allowUnpredictedQueries) continue;
                    if(q.predicted) ++predictedCount; else ++unpredictedCount;
                    if(priorities[groups[i]]) ++ownedCount;
                    if(input.geometry) {
                        const Eigen::Vector3d rotated=guidance.T_train_query.C()*bearingsQ[i];
                        const Eigen::Vector3d direction=rotationOnly?rotated:translation.cross(rotated).eval();
                        if(!rotationOnly) { q.planeNorm=direction.norm(); if(q.planeNorm<1e-12) continue; }
                        std::copy_n(direction.data(),3,q.direction);
                    }
                    q.group=groups[i]; q.octave=queryPoints[i].octave;
                    q.useGrid=q.predicted && !grid.empty();
                    if(q.predicted) {
                        const auto p=guidance.predictedTrainPixels[i];
                        q.predictedX=p.x; q.predictedY=p.y;
                        if(q.useGrid) {
                            const double radius=guidance.searchRadiusPixels;
                            const auto cell=[&](double coordinate,int limit) {
                                return int(std::clamp(std::floor(coordinate/cellSize),0.0,double(limit-1)));
                            };
                            q.minX=cell(p.x-radius,columns); q.maxX=cell(p.x+radius,columns);
                            q.minY=cell(p.y-radius,rows); q.maxY=cell(p.y+radius,rows);
                        }
                    }
                    if(!q.useGrid) {
                        if(options_.maxOctaveDifference>=0) {
                            const auto lo=static_cast<long long>(q.octave)-options_.maxOctaveDifference;
                            const auto hi=static_cast<long long>(q.octave)+options_.maxOctaveDifference;
                            const auto lower=octaveOffsets.lower_bound(int(std::max(lo,static_cast<long long>(std::numeric_limits<int>::min()))));
                            const auto upper=hi>=std::numeric_limits<int>::max()?octaveOffsets.end():octaveOffsets.upper_bound(int(hi));
                            q.first=octaveOffset(lower); q.last=octaveOffset(upper);
                        } else q.last=int(input.octaveEntries.size());
                    }
                }
                detail::OrbCudaMatchOutput output;
                usedCuda=guidance.cudaSession->match(input,output);
                if(usedCuda) {
                    const auto assign=[](const auto& from,auto& to) {
                        for(std::size_t i=0;i<from.size();++i) for(int k=0;k<2;++k) {
                            to[i].index[k]=from[i].index[k]; to[i].distance[k]=from[i].distance[k];
                        }
                    };
                    assign(output.forward,forward); assign(output.reverse,reverse); assign(output.owned,ownedReverse);
                    for(int i=0;i<groupCount;++i) representatives[i]=output.forward[i].representative;
                    if(statistics) {
                        statistics->usedCuda=true; statistics->candidatePairs=output.candidatePairs;
                        statistics->residentDescriptorSets=output.residentDescriptorSets;
                        statistics->descriptorComparisons=output.descriptorComparisons;
                        statistics->predictedQueries=predictedCount; statistics->unpredictedQueries=unpredictedCount;
                        statistics->ownedQueries=ownedCount;
                    }
                }
            }
#endif
            if(!usedCuda) for (int i = 0; i < query.rows; ++i) {
                if (!validQ[i]) continue;
                const bool predicted=!guidance.predictedTrainPixels.empty()
                    && std::isfinite(guidance.predictedTrainPixels[i].x) && std::isfinite(guidance.predictedTrainPixels[i].y);
                if(!guidance.predictedTrainPixels.empty() && !predicted && !guidance.allowUnpredictedQueries) continue;
                if(statistics) { if(predicted) ++statistics->predictedQueries; else ++statistics->unpredictedQueries; }
                if(statistics && priorities[groups[i]]) ++statistics->ownedQueries;
                Eigen::Vector3d rotated=Eigen::Vector3d::Zero(),normal=rotated;
                double queryPlaneNorm=0;
                if(guidance.useSphericalGeometry) {
                    rotated=guidance.T_train_query.C()*bearingsQ[i];
                    if(!rotationOnly) { normal=translation.cross(rotated); queryPlaneNorm=normal.norm(); if(queryPlaneNorm<1e-12) continue; }
                }
                const auto consider=[&](int j) {
                    if(statistics) ++statistics->candidatePairs;
                    if (!validT[j] || (!guidance.allowedPairs.empty() && guidance.allowedPairs.at<uchar>(i, j) == 0)) return;
                    if (options_.maxOctaveDifference >= 0 &&
                        std::abs(static_cast<long long>(queryPoints[i].octave) - trainPoints[j].octave) > options_.maxOctaveDifference) return;
                    if (predicted) {
                        const cv::Point2f displacement = trainPoints[j].pt - guidance.predictedTrainPixels[i];
                        if (double(displacement.x) * displacement.x + double(displacement.y) * displacement.y > radiusSquared) return;
                    }
                    if(guidance.useSphericalGeometry) {
                        if(rotationOnly) { if(rotated.dot(bearingsT[j])<cosineThreshold) return; }
                        // Equivalent symmetric angular gate without per-pair asin,
                        // rotation, normalization or a second cross product.
                        else if(trainPlaneNorm[j]<1e-12 || std::abs(bearingsT[j].dot(normal))
                            >sineThreshold*std::min(queryPlaneNorm,trainPlaneNorm[j])) return;
                    }
                    if(statistics) ++statistics->descriptorComparisons;
                    const int distance=hamming(wordsQ[i],wordsT[j]);
                    const int group=groups[i]; auto& best=forward[group];
                    const int previousIndex=best.index[0],previousDistance=best.distance[0];
                    best.add(j,distance);
                    if(best.index[0]==j && best.distance[0]==distance
                        && (previousIndex!=j || previousDistance>distance || representatives[group]<0 || i<representatives[group]))
                        representatives[group]=i;
                    if(options_.mutualCheck) {
                        // Ownership is decided before reciprocal-neighbor competition:
                        // an unassigned lookalike must not break a valid existing seed
                        // merely because assignment would prioritize that seed later.
                        reverse[j].add(group,distance);
                        if(priorities[group]) ownedReverse[j].add(group,distance);
                    }
                };
                if(predicted && !grid.empty()) {
                    const auto p=guidance.predictedTrainPixels[i]; const double radius=guidance.searchRadiusPixels;
                    const auto cell=[&](double coordinate,int limit) { return int(std::clamp(std::floor(coordinate/cellSize),0.0,double(limit-1))); };
                    for(int y=cell(p.y-radius,rows);y<=cell(p.y+radius,rows);++y)
                        for(int x=cell(p.x-radius,columns);x<=cell(p.x+radius,columns);++x)
                            for(int j:grid[size_t(y)*columns+x]) consider(j);
                } else if(options_.maxOctaveDifference>=0) {
                    const long long lo=static_cast<long long>(queryPoints[i].octave)-options_.maxOctaveDifference;
                    const long long hi=static_cast<long long>(queryPoints[i].octave)+options_.maxOctaveDifference;
                    const int first=int(std::max(lo,static_cast<long long>(std::numeric_limits<int>::min())));
                    for(auto bucket=trainOctaves.lower_bound(first);bucket!=trainOctaves.end() && bucket->first<=hi;++bucket)
                        for(int j:bucket->second) consider(j);
                } else for(const auto& bucket:trainOctaves) for(int j:bucket.second) consider(j);
            }
            if(statistics) statistics->traversalMs=milliseconds(traversalStarted);
            const auto selectionStarted=std::chrono::steady_clock::now();
            // One distance pass supplies BOTH directions, including each direction's
            // second neighbor. Preserve strict ratio, mutual and uniqueness semantics.
            std::vector<uint8_t> claimedTrain(train.rows,0);
            for(int tier=1;tier>=0;--tier) for(int group=0;group<groupCount;++group)
                if(priorities[group]==tier && forward[group].passes(options_)) {
                const int j=forward[group].index[0];
                if(options_.mutualCheck) {
                    const auto reciprocal=[&](const TopTwo& neighbors) {
                        return neighbors.index[0]==group && (!options_.reverseRatioTest || neighbors.passes(options_));
                    };
                    if(!reciprocal(reverse[j]) && !(tier && reciprocal(ownedReverse[j]))) continue;
                    if(claimedTrain[j]) { if(statistics && !tier) ++statistics->prioritySuppressed; continue; }
                    claimedTrain[j]=1;
                }
                matches.emplace_back(representatives[group],j,float(forward[group].distance[0]));
            }
            if(!options_.mutualCheck) {
                std::sort(matches.begin(),matches.end(),[&](const auto& a,const auto& b) {
                    if(priorities[groups[a.queryIdx]]!=priorities[groups[b.queryIdx]])
                        return priorities[groups[a.queryIdx]]>priorities[groups[b.queryIdx]];
                    if(a.distance!=b.distance) return a.distance<b.distance;
                    return a.queryIdx!=b.queryIdx?a.queryIdx<b.queryIdx:a.trainIdx<b.trainIdx;
                });
                std::vector<bool> used(train.rows,false);
                matches.erase(std::remove_if(matches.begin(),matches.end(),[&](const auto& match) {
                    if(used[match.trainIdx]) return true; used[match.trainIdx]=true; return false;
                }),matches.end());
                }
            // Public output remains query ordered regardless of ownership traversal.
            std::sort(matches.begin(),matches.end(),[](const auto& a,const auto& b) { return a.queryIdx<b.queryIdx; });
            if(statistics) for(const auto& match:matches) if(priorities[groups[match.queryIdx]]) ++statistics->ownedMatches;
            if(statistics) statistics->selectionMs=milliseconds(selectionStarted);
        }
        void OrbMatcher::matchFrames(const Frame& query, const Frame& train, const OrbMatchGuidance& input,
            std::vector<cv::DMatch>& matches) const {
            OrbMatchGuidance guidance(input);
            if (!guidance.queryCamera) guidance.queryCamera = query.geometry();
            if (!guidance.trainCamera) guidance.trainCamera = train.geometry();
            matchGuided(query.descriptors(), train.descriptors(), query.keypoints(), train.keypoints(), guidance, matches);
        }
        void OrbMatcher::matchTemporal(const MultiFrame& previous, std::size_t queryCamera,
            const MultiFrame& current, std::size_t trainCamera,
            const kinematics::Transformation& T_Scurrent_Sprevious,
            const std::vector<cv::Point2f>& predictedTrainPixels,
            std::vector<cv::DMatch>& matches, float searchRadiusPixels) const {
            matches.clear(); validateCameraIndex(previous, queryCamera); validateCameraIndex(current, trainCamera);
            if (predictedTrainPixels.size() != previous.numKeypoints(queryCamera))
                throw std::invalid_argument("Temporal ORB requires an IMU-guided prediction for every query feature");
            if (!std::isfinite(searchRadiusPixels) || searchRadiusPixels <= 0)
                throw std::invalid_argument("Invalid temporal ORB search radius");
            const auto queryExtrinsics = previous.cameraSystem().T_SC(queryCamera);
            const auto trainExtrinsics = current.cameraSystem().T_SC(trainCamera);
            if (!queryExtrinsics || !trainExtrinsics) throw std::invalid_argument("Missing ORB camera extrinsics");
            OrbMatchGuidance guidance;
            guidance.queryCamera = previous.geometry(queryCamera); guidance.trainCamera = current.geometry(trainCamera);
            guidance.T_train_query = trainExtrinsics->inverse() * T_Scurrent_Sprevious * (*queryExtrinsics);
            guidance.predictedTrainPixels = predictedTrainPixels; guidance.searchRadiusPixels = searchRadiusPixels;
            matchGuided(previous.descriptors(queryCamera), current.descriptors(trainCamera),
                previous.keypoints(queryCamera), current.keypoints(trainCamera), guidance, matches);
        }
        void OrbMatcher::matchCrossCamera(const MultiFrame& frame, std::size_t queryCamera,
            std::size_t trainCamera, std::vector<cv::DMatch>& matches) const {
            matches.clear(); validateCameraIndex(frame, queryCamera); validateCameraIndex(frame, trainCamera);
            if (queryCamera == trainCamera) throw std::invalid_argument("Cross-camera ORB requires two different cameras");
            const auto& rig = frame.cameraSystem();
            if (!rig.overlapComputationValid()) throw std::logic_error("Compute the camera overlap graph before cross-camera ORB matching");
            if (!rig.hasOverlap(trainCamera, queryCamera) || !rig.hasOverlap(queryCamera, trainCamera)) return;
            const auto queryExtrinsics = rig.T_SC(queryCamera), trainExtrinsics = rig.T_SC(trainCamera);
            if (!queryExtrinsics || !trainExtrinsics) throw std::invalid_argument("Missing ORB camera extrinsics");
            OrbMatchGuidance guidance;
            guidance.queryCamera = frame.geometry(queryCamera); guidance.trainCamera = frame.geometry(trainCamera);
            guidance.T_train_query = trainExtrinsics->inverse() * (*queryExtrinsics);
            // overlap(seenBy, source): mask pixels are in the SOURCE camera image.
            guidance.queryRegionMask = rig.overlap(trainCamera, queryCamera);
            guidance.trainRegionMask = rig.overlap(queryCamera, trainCamera);
            matchGuided(frame.descriptors(queryCamera), frame.descriptors(trainCamera),
                frame.keypoints(queryCamera), frame.keypoints(trainCamera), guidance, matches);
        }
    }
}
