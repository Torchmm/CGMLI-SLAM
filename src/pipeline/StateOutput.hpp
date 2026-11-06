#pragma once
#include <backend/ViSlamBackend.hpp>
#include <common/ViInterface.hpp>

namespace cgmli { namespace pipeline {
using StateHistory=AlignedMap<StateId,State>;
backend::StateEstimate bodyPrediction(const State&,const kinematics::Transformation& T_BS);
State sensorState(const backend::StateEstimate&,const kinematics::Transformation& T_BS,
    const ImuMeasurementDeque&,const Eigen::Vector3d& fallbackOmega=Eigen::Vector3d::Zero());
AlignedVector<kinematics::Transformation> cameraExtrinsics(const backend::BackendSnapshot&,
    const cameras::NCameraSystem&,const kinematics::Transformation& T_BS);
std::shared_ptr<MapPointVector> mapPoints(const backend::BackendSnapshot&);
double trackingCoverage(const MultiFrame&,const backend::BackendSnapshot&);
bool writeTrajectory(const std::string&,const StateHistory&,bool rpg);
bool writeMap(const std::string&,const MapPointVector&);
} }
