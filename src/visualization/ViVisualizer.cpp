
#include <kinematics/Transformation.hpp>

#include <cameras/NCameraSystem.hpp>
#include <common/FrameTypedefs.hpp>

#include "visualization/ViVisualizer.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <utility>

// cameras and distortions
#include <cameras/PinholeCamera.hpp>
#include <cameras/EquidistantDistortion.hpp>
#include <cameras/RadialTangentialDistortion.hpp>
#include <cameras/RadialTangentialDistortion8.hpp>
#include <cameras/EucmCamera.hpp>

/// \brief cgmli Main namespace of this package.
namespace cgmli {

namespace {
const cv::Scalar debugYellow(0,220,255), debugGreen(50,230,50), debugRed(60,60,255), debugWhite(230,230,230);
void debugText(cv::Mat& image,const std::string& text,int x,int y,double scale=0.45) {
  cv::putText(image,text,cv::Point(x,y),cv::FONT_HERSHEY_SIMPLEX,scale,debugWhite,1,cv::LINE_AA);
}
cv::Mat colorCopy(const cv::Mat& image) {
  cv::Mat result;
  if(image.empty()) return result;
  if(image.channels()==1) cv::cvtColor(image,result,cv::COLOR_GRAY2BGR);
  else if(image.channels()==4) cv::cvtColor(image,result,cv::COLOR_BGRA2BGR);
  else result=image.clone();
  return result;
}
cv::Mat debugBoard(const std::vector<cv::Mat>& panels,const std::string& title,const std::string& legend) {
  if(panels.empty()) return {};
  const int columns=std::min(2,int(panels.size())),rows=(int(panels.size())+columns-1)/columns;
  const int width=panels.front().cols,height=panels.front().rows;
  cv::Mat result(rows*height+48,columns*width,CV_8UC3,cv::Scalar(20,20,20));
  debugText(result,title,8,18); debugText(result,legend,8,38,0.40);
  for(size_t i=0;i<panels.size();++i) panels[i].copyTo(result(cv::Rect(int(i)%columns*width,48+int(i)/columns*height,width,height)));
  return result;
}
using FeatureCamera=ViVisualizer::FeatureCamera;
cv::Mat matchPanel(const FeatureCamera& left,const FeatureCamera& right,
    const std::vector<const FeatureMatchDebug*>& matches,const std::string& title) {
  cv::Mat result(250,640,CV_8UC3,cv::Scalar(16,16,16));
  const auto putImage=[&](const FeatureCamera& camera,int startX,cv::Point& origin,double& scale) {
    scale=camera.image.empty()?1.0:std::min(320.0/camera.image.cols,180.0/camera.image.rows);
    if(camera.image.empty()) { origin={startX,54}; return; }
    cv::Mat resized; cv::resize(colorCopy(camera.image),resized,cv::Size(),scale,scale,cv::INTER_AREA);
    origin={startX+(320-resized.cols)/2,54+(180-resized.rows)/2};
    resized.copyTo(result(cv::Rect(origin.x,origin.y,resized.cols,resized.rows)));
  };
  cv::Point originA,originB; double scaleA=1,scaleB=1;
  putImage(left,0,originA,scaleA); putImage(right,320,originB,scaleB);
  size_t retained=0; for(const auto* match:matches) if(match->retained) ++retained;
  size_t seenKept=0,seenDiscarded=0,shown=0;
  // Sample each class separately so a busy image does not hide all discarded
  // candidates. Coordinates always come from the actual matcher keypoint IDs.
  const size_t keptStride=std::max<size_t>(1,(retained+59)/60);
  const size_t discardedStride=std::max<size_t>(1,(matches.size()-retained+59)/60);
  for(const auto* match:matches) {
    const size_t index=match->retained?seenKept++:seenDiscarded++;
    if(index%(match->retained?keptStride:discardedStride)) continue;
    if(match->first.keypointIndex>=left.keypoints.size() || match->second.keypointIndex>=right.keypoints.size()) continue;
    const auto a=left.keypoints[match->first.keypointIndex].pt,b=right.keypoints[match->second.keypointIndex].pt;
    if(!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(b.x) || !std::isfinite(b.y)) continue;
    const cv::Point p(originA.x+cvRound(a.x*left.scaleX*scaleA),originA.y+cvRound(a.y*left.scaleY*scaleA));
    const cv::Point q(originB.x+cvRound(b.x*right.scaleX*scaleB),originB.y+cvRound(b.y*right.scaleY*scaleB));
    if(p.y<54 || q.y<54 || p.x<0 || p.x>=320 || q.x<320 || q.x>=640 || p.y>=234 || q.y>=234) continue;
    const auto color=match->retained?debugGreen:debugRed;
    cv::line(result,p,q,color,1,cv::LINE_AA); cv::circle(result,p,3,color,1,cv::LINE_AA); cv::circle(result,q,3,color,1,cv::LINE_AA); ++shown;
  }
  debugText(result,title,6,17);
  debugText(result,"guided="+std::to_string(matches.size())+" kept="+std::to_string(retained)
      +" discarded="+std::to_string(matches.size()-retained)+" shown="+std::to_string(shown),6,38,0.42);
  return result;
}
cv::Mat noMatchesPanel(const std::string& message) {
  cv::Mat panel(250,640,CV_8UC3,cv::Scalar(16,16,16)); debugText(panel,message,12,100); return panel;
}
}

std::shared_ptr<const ViVisualizer::FeatureFrame> ViVisualizer::captureFeatures(const MultiFrame& frame) {
  auto result=std::make_shared<FeatureFrame>(); result->id=frame.id(); result->timestampNs=frame.timestamp().toNSec();
  result->cameras.resize(frame.numFrames());
  for(size_t c=0;c<frame.numFrames();++c) {
    auto& output=result->cameras[c]; const auto& image=frame.image(c);
    output.keypoints=frame.keypoints(c);
    if(const auto& timing=frame.imageTiming(c)) output.timeOffsetNs=timing->offsetNs;
    if(image.empty()) continue;
    const double scale=std::min(1.0,std::min(640.0/image.cols,360.0/image.rows));
    cv::resize(image,output.image,cv::Size(std::max(1,cvRound(image.cols*scale)),std::max(1,cvRound(image.rows*scale))),0,0,cv::INTER_AREA);
    output.scaleX=double(output.image.cols)/image.cols; output.scaleY=double(output.image.rows)/image.rows;
  }
  return result;
}

cv::Mat ViVisualizer::cameraMosaic(const std::vector<cv::Mat>& images,const std::string& title) {
  std::vector<cv::Mat> panels;
  for(size_t c=0;c<images.size();++c) {
    cv::Mat panel(420,640,CV_8UC3,cv::Scalar(16,16,16)); debugText(panel,"camera "+std::to_string(c),8,19);
    if(!images[c].empty()) {
      const double scale=std::min(640.0/images[c].cols,392.0/images[c].rows);
      cv::Mat image; cv::resize(colorCopy(images[c]),image,cv::Size(),scale,scale,cv::INTER_AREA);
      image.copyTo(panel(cv::Rect((640-image.cols)/2,28+(392-image.rows)/2,image.cols,image.rows)));
    }
    panels.push_back(std::move(panel));
  }
  return debugBoard(panels,title,"Reprojection residuals: cyan projection to measured keypoint; backend accepted state");
}

std::map<std::string,cv::Mat> ViVisualizer::drawFeatureDebug(const FeatureDebugData& data) {
  std::map<std::string,cv::Mat> views; if(!data.current) return views;
  const auto& frame=*data.current; const size_t cameras=frame.cameras.size();
  const std::string legend="yellow=features; green=retained track (seed or mature); red=discarded candidate; before backend acceptance";
  const std::string caption="frame="+std::to_string(frame.id)+" node_ns="+std::to_string(frame.timestampNs)
      +" committed="+std::to_string(data.committed)+" visual_consensus="+std::to_string(data.visualConsensus)
      +" pose_inliers="+std::to_string(data.inliers);
  std::vector<cv::Mat> panels;
  for(size_t c=0;c<cameras;++c) {
    const auto& camera=frame.cameras[c]; cv::Mat panel(420,640,CV_8UC3,cv::Scalar(16,16,16));
    const int x=(640-camera.image.cols)/2,y=56+(360-camera.image.rows)/2;
    if(!camera.image.empty()) {
      const auto color=colorCopy(camera.image); color.copyTo(panel(cv::Rect(x,y,color.cols,color.rows)));
      for(const auto& key:camera.keypoints) {
        if(!std::isfinite(key.pt.x) || !std::isfinite(key.pt.y)) continue;
        const cv::Point p(x+cvRound(key.pt.x*camera.scaleX),y+cvRound(key.pt.y*camera.scaleY));
        cv::circle(panel,p,3,debugYellow,1,cv::LINE_AA);
        if(std::isfinite(key.angle) && key.angle>=0) {
          const double angle=key.angle*std::acos(-1.0)/180.0;
          cv::line(panel,p,p+cv::Point(cvRound(7*std::cos(angle)),cvRound(7*std::sin(angle))),debugYellow,1,cv::LINE_AA);
        }
      }
    }
    debugText(panel,"camera "+std::to_string(c)+" keypoints="+std::to_string(camera.keypoints.size()),8,19);
    std::ostringstream timing; timing<<"source offset=";
    if(camera.timeOffsetNs) timing<<std::fixed<<std::setprecision(3)<<double(*camera.timeOffsetNs)*1e-6<<" ms (sync approximation)";
    else timing<<"unavailable";
    debugText(panel,timing.str(),8,40); panels.push_back(std::move(panel));
  }
  views["01 ORB Features"]=debugBoard(panels,caption,legend); panels.clear();
  for(size_t a=0;a<cameras;++a) for(size_t b=a+1;b<cameras;++b) {
    std::vector<const FeatureMatchDebug*> matches;
    for(const auto& match:data.matches) if(match.first.frameId==frame.id && match.second.frameId==frame.id
        && match.first.cameraIndex==a && match.second.cameraIndex==b) matches.push_back(&match);
    panels.push_back(matchPanel(frame.cameras[a],frame.cameras[b],matches,"same epoch: camera "+std::to_string(a)+" -> "+std::to_string(b)));
  }
  if(panels.empty()) panels.push_back(noMatchesPanel("No cross-camera pairs"));
  views["02 Stereo Matches"]=debugBoard(panels,"Same-epoch cross-camera matches | "+caption,legend); panels.clear();
  uint64_t sourceId=0;
  for(const auto& match:data.matches) if(match.first.frameId!=frame.id && data.sources.count(match.first.frameId)) sourceId=std::max(sourceId,match.first.frameId);
  if(sourceId) {
    const auto& source=*data.sources.at(sourceId);
    struct Pair { size_t a,b,count; }; std::vector<Pair> pairs;
    for(size_t a=0;a<source.cameras.size();++a) for(size_t b=0;b<cameras;++b) {
      size_t count=0;
      for(const auto& match:data.matches) if(match.first.frameId==sourceId && match.second.frameId==frame.id
          && match.first.cameraIndex==a && match.second.cameraIndex==b) ++count;
      if(count) pairs.push_back({a,b,count});
    }
    std::sort(pairs.begin(),pairs.end(),[](const Pair& a,const Pair& b) {
      return a.count!=b.count?a.count>b.count:std::tie(a.a,a.b)<std::tie(b.a,b.b);
    });
    if(pairs.size()>6) pairs.resize(6);
    for(const auto& pair:pairs) {
      std::vector<const FeatureMatchDebug*> matches;
      for(const auto& match:data.matches) if(match.first.frameId==sourceId && match.second.frameId==frame.id
          && match.first.cameraIndex==pair.a && match.second.cameraIndex==pair.b) matches.push_back(&match);
      panels.push_back(matchPanel(source.cameras[pair.a],frame.cameras[pair.b],matches,
          "frame "+std::to_string(sourceId)+" cam "+std::to_string(pair.a)+" -> frame "+std::to_string(frame.id)+" cam "+std::to_string(pair.b)));
    }
  }
  if(panels.empty()) panels.push_back(noMatchesPanel("No cached temporal candidate matches (bootstrap or tracking failure)"));
  views["03 Temporal Matches"]=debugBoard(panels,"Newest matched reference; top 6 camera pairs | "+caption,legend);
  return views;
}

cv::Mat ViVisualizer::drawTrajectory(const VisualizationData& data) {
  std::vector<cv::Mat> panels;
  const Eigen::Vector3d current=data.trajectory.empty()?Eigen::Vector3d(data.T_WS.r()):data.trajectory.back().position;
  for(int vertical:{1,2}) {
    cv::Mat image(540,640,CV_8UC3,cv::Scalar(16,16,16));
    double x0=current.x()-1,x1=x0+2,y0=current[vertical]-1,y1=y0+2;
    for(const auto& p:data.trajectory) if(p.position.allFinite()) {
      x0=std::min(x0,p.position.x()); x1=std::max(x1,p.position.x());
      y0=std::min(y0,p.position[vertical]); y1=std::max(y1,p.position[vertical]);
    }
    const double scale=std::min(560.0/(x1-x0),440.0/(y1-y0));
    if(!std::isfinite(scale) || scale<=0) { panels.push_back(image); continue; }
    const auto pixel=[&](const Eigen::Vector3d& p) {
      return cv::Point(cvRound(320+scale*(p.x()-0.5*(x0+x1))),cvRound(285-scale*(p[vertical]-0.5*(y0+y1))));
    };
    for(int x=40;x<640;x+=80) cv::line(image,{x,60},{x,510},cv::Scalar(35,35,35));
    for(int y=70;y<520;y+=80) cv::line(image,{40,y},{600,y},cv::Scalar(35,35,35));
    bool havePrevious=false; cv::Point previous;
    for(const auto& p:data.trajectory) if(p.position.allFinite()) {
      const auto next=pixel(p.position);
      if(havePrevious) cv::line(image,previous,next,cv::Scalar(190,190,190),1,cv::LINE_AA);
      previous=next; havePrevious=true;
    }
    for(const auto& p:data.window) if(p.position.allFinite())
      cv::circle(image,pixel(p.position),p.keyframe?4:2,debugYellow,p.keyframe?2:cv::FILLED,cv::LINE_AA);
    cv::circle(image,pixel(current),6,debugGreen,2,cv::LINE_AA);
    debugText(image,vertical==1?"X-Y trajectory [m]":"X-Z trajectory [m]",10,23);
    std::ostringstream legend; legend<<std::fixed<<std::setprecision(3)<<"current="<<current.transpose()
      <<" | grid="<<80/scale<<" m";
    debugText(image,legend.str(),10,45,0.42); panels.push_back(std::move(image));
  }
  return debugBoard(panels,"Accepted local trajectory | frame="+std::to_string(data.acceptedFrame),
    "gray=accepted history; yellow=active window (rings=keyframes); green=current; gravity-aligned local frame, no GT alignment");
}

cv::Mat ViVisualizer::drawTrackTrails(const VisualizationData& data) {
  if(!data.currentFrames || data.currentFrames->id()<lastTrailFrame_) return {};
  const auto& frame=*data.currentFrames; std::vector<cv::Mat> panels;
  for(size_t camera=0;camera<frame.numFrames();++camera) {
    const size_t active=camera<data.activeLandmarksByCamera.size()?data.activeLandmarksByCamera[camera]:0;
    const auto cameraLabel="camera "+std::to_string(camera)+" active_landmarks="+std::to_string(active);
    if(frame.image(camera).empty()) {
      cv::Mat panel(400,640,CV_8UC3,cv::Scalar(16,16,16));
      debugText(panel,cameraLabel,8,18);
      debugText(panel,"No camera image",8,38);
      panels.push_back(std::move(panel)); continue;
    }
    cv::Mat panel; cv::resize(colorCopy(frame.image(camera)),panel,cv::Size(640,400));
    const float sx=640.0f/frame.image(camera).cols,sy=400.0f/frame.image(camera).rows;
    std::vector<std::pair<uint64_t,TrackTrail*>> visible;
    for(size_t k=0;k<frame.numKeypoints(camera);++k) {
      const auto id=frame.landmarkId(camera,k); if(!id) continue;
      const auto& kp=frame.keypoints(camera)[k]; const cv::Point2f p(kp.pt.x*sx,kp.pt.y*sy);
      auto& trail=trackTrails_[{id,camera}];
      if(frame.id()>trail.lastFrame && frame.id()-trail.lastFrame>30) trail.points.clear();
      if(trail.lastFrame!=frame.id()) { trail.points.push_back(p); trail.lastFrame=frame.id(); }
      while(trail.points.size()>12) trail.points.pop_front();
      Eigen::Vector4d point; bool mature=false; frame.getLandmark(camera,k,point,mature);
      const auto life=data.trackLifecycles.find(id);
      const bool reset=life!=data.trackLifecycles.end() && life->second.lastReset>0
        && life->second.lastReset>=lastTrailFrame_;
      cv::circle(panel,p,2,reset?cv::Scalar(0,140,255):mature?debugGreen:cv::Scalar(255,150,50),1,cv::LINE_AA);
      visible.emplace_back(id,&trail);
    }
    std::sort(visible.begin(),visible.end(),[](const auto& a,const auto& b) {
      if(a.second->points.size()!=b.second->points.size()) return a.second->points.size()>b.second->points.size();
      return a.first<b.first;
    });
    for(size_t i=0;i<std::min<size_t>(64,visible.size());++i) {
      const auto id=visible[i].first; const auto& points=visible[i].second->points;
      const cv::Scalar color(80+(id*37)%175,80+(id*67)%175,80+(id*97)%175);
      for(size_t j=1;j<points.size();++j) cv::line(panel,points[j-1],points[j],color,1,cv::LINE_AA);
      if(i<12 && !points.empty()) {
        std::string label=std::to_string(id); const auto life=data.trackLifecycles.find(id);
        if(life!=data.trackLifecycles.end()) label+=" i"+std::to_string(life->second.initializations)
          +" r"+std::to_string(life->second.resets)+" d"+std::to_string(life->second.demotions);
        cv::putText(panel,label,points.back(),cv::FONT_HERSHEY_SIMPLEX,0.35,debugWhite,1,cv::LINE_AA);
      }
    }
    debugText(panel,cameraLabel+" associated="+std::to_string(visible.size()),8,18);
    panels.push_back(std::move(panel));
  }
  lastTrailFrame_=frame.id();
  for(auto it=trackTrails_.begin();it!=trackTrails_.end();) {
    if(frame.id()>it->second.lastFrame && frame.id()-it->second.lastFrame>30) it=trackTrails_.erase(it); else ++it;
  }
  return debugBoard(panels,"Accepted feature tracks | frame="+std::to_string(frame.id())
    +" | window_mature_3d="+std::to_string(data.windowMature3d),
    "green=mature; blue=seed; orange=recent reset; ID i=initializations r=resets d=demotions; up to 12 rendered positions, 64 trails/camera");
}

ViVisualizer::ViVisualizer(ViParameters &parameters)
    : parameters_(parameters) {
  if (parameters.nCameraSystem.numCameras() > 0) {
    init(parameters);
  }
}

ViVisualizer::~ViVisualizer() {
}

void ViVisualizer::init(ViParameters &parameters) {
  parameters_ = parameters;
}

cv::Mat ViVisualizer::drawMatches(VisualizationData::Ptr& data,
                                   size_t image_number) {

  const kinematics::Transformation T_WS = data->T_WS;
  const kinematics::Transformation T_SC = data->T_SCi.at(image_number);
  //const kinematics::Transformation T_SC = *parameters_.nCameraSystem.T_SC(image_number);
  const kinematics::Transformation T_WC = T_WS*T_SC;
  const kinematics::Transformation T_CW = T_WC.inverse();

  std::shared_ptr<cgmli::MultiFrame> frame = data->currentFrames;

  // allocate an image
  const int im_cols = frame->image(image_number).cols;
  const int im_rows = frame->image(image_number).rows;

  cv::Mat outimg(im_rows, im_cols, CV_8UC3);
  cv::Mat current = outimg;

  cv::cvtColor(frame->image(image_number), current, cv::COLOR_GRAY2BGR);

  // loop-closure frame?
  cv::Mat tmp(im_rows, im_cols, CV_8UC3);
  if(data->recognisedPlace) {
    cv::Scalar colour(255,0,0);
    tmp.setTo(colour);
    cv::addWeighted(tmp, 0.5, current, 1.0 - 0.5, 0, current);
    cv::putText(current, "Place recognised", cv::Point2f(200,10), cv::FONT_HERSHEY_COMPLEX, 0.3,
                colour, 1, cv::LINE_AA);
  }

  // keyframe?
  if(data->isKeyframe) {
    cv::Scalar colour(255,255,0);
    tmp.setTo(colour);
    cv::addWeighted(tmp, 0.3, current, 1.0 - 0.3, 0, current);
    cv::putText(current, "Keyframe", cv::Point2f(120,10), cv::FONT_HERSHEY_COMPLEX, 0.3, colour, 1,
                cv::LINE_AA);
  }

  // Quality
  cv::Scalar trackingColour;
  if(data->trackingQuality == VisualizationData::TrackingQuality::Lost) {
    trackingColour = cv::Scalar(0,0,255);
    cv::putText(current, "TRACKING LOST", cv::Point2f(5,10), cv::FONT_HERSHEY_COMPLEX, 0.3,
                trackingColour, 1, cv::LINE_AA);
  } else if (data->trackingQuality == VisualizationData::TrackingQuality::Marginal) {
    trackingColour = cv::Scalar(0,255,255);
    cv::putText(current, "Tracking marginal", cv::Point2f(5,10), cv::FONT_HERSHEY_COMPLEX, 0.3,
                trackingColour, 1, cv::LINE_AA);
  } else {
    trackingColour = cv::Scalar(0,255,0);
    cv::putText(current, "Tracking good", cv::Point2f(5,10), cv::FONT_HERSHEY_COMPLEX, 0.3,
                trackingColour, 1, cv::LINE_AA);
  }

  for (auto it = data->observations.begin(); it != data->observations.end();
      ++it) {
    if (it->cameraIdx != image_number)
      continue;

    cv::Scalar color;

    if (it->landmarkId != 0) {
      color = cv::Scalar(255, 0, 0);  // blue
    } else {
      color = cv::Scalar(0, 0, 255);  // red
    }

    // draw matches
    auto keypoint = it->keypointMeasurement;
    if (fabs(it->landmark_W[3]) > 1.0e-8) {
      Eigen::Vector4d hPoint = it->landmark_W;
      if (it->isInitialized) {
        color = cv::Scalar(0, 255, 0);  // green
      } else {
        color = cv::Scalar(0, 255, 255);  // yellow
      }
      Eigen::Vector2d projection;
      const Eigen::Vector4d hP_C = T_CW * hPoint;
      // Use the actual camera model. Distortion enums cannot distinguish EUCM
      // from a pinhole without distortion, and valid fisheye rays may have z<0.
      const bool isVisible = frame->geometry(image_number)->projectHomogeneous(hP_C, &projection)
          == cameras::ProjectionStatus::Successful;
      // draw projection
      if(isVisible) {
        cv::Point2f projectionCv(projection[0], projection[1]);
        cv::line(current, projectionCv,
            cv::Point2f(float(keypoint[0]), float(keypoint[1])), color, 1, cv::LINE_AA);
        cv::circle(current, projectionCv, 2,
            cv::Scalar(255,255,0), cv::FILLED, cv::LINE_AA);
        std::stringstream idText;
        idText << it->landmarkId;
        cv::putText(current, idText.str(),
                    cv::Point2f(float(projection[0]), float(projection[1])) + cv::Point2f(10,5),
                     cv::FONT_HERSHEY_COMPLEX, 0.35, cv::Scalar(255,255,0), 1, cv::LINE_AA);
      }
    }
    // draw keypoint
    const double r = 0.5 * it->keypointSize;
    cv::circle(current, cv::Point2f(float(keypoint[0]), float(keypoint[1])), int(r), color, 1,
        cv::LINE_AA);
    cv::KeyPoint cvKpt;
    frame->getCvKeypoint(it->cameraIdx, it->keypointIdx, cvKpt);
    const float angle = cvKpt.angle / 180.0f * float(M_PI);
    cv::line(outimg, cvKpt.pt,
             cv::Point2f(cvKpt.pt.x + float(r) * cos(angle), cvKpt.pt.y + float(r) * sin(angle)),
             color, 1, cv::LINE_AA);
  }
  return outimg;
}


cv::Mat ViVisualizer::drawKeypoints(VisualizationData::Ptr& data,
                                     size_t cameraIndex) {

  std::shared_ptr<cgmli::MultiFrame> currentFrames = data->currentFrames;
  cv::Mat currentImage = currentFrames->image(cameraIndex);

  cv::Mat outimg;
  cv::cvtColor(currentImage, outimg, cv::COLOR_GRAY2BGR);
  cv::Scalar greenColor(0, 255, 0);  // green

  cv::KeyPoint keypoint;
  for (size_t k = 0; k < currentFrames->numKeypoints(cameraIndex); ++k) {
    currentFrames->getCvKeypoint(cameraIndex, k, keypoint);

    const float radius = keypoint.size;
    const float angle = keypoint.angle / 180.0f * float(M_PI);

    cv::circle(outimg, keypoint.pt, int(radius), greenColor);
    cv::line(
        outimg,
        keypoint.pt,
        cv::Point2f(keypoint.pt.x + radius * cos(angle),
                    keypoint.pt.y - radius * sin(angle)),
        greenColor);
  }

  return outimg;
}

} /* namespace cgmli */
