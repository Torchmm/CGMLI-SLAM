#include "FixedLagBackend.hpp"
#include <factor/BodyImuFactor.hpp>
#include <Eigen/Cholesky>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <set>
#include <glog/logging.h>
#include <visual/MultiFrame.hpp>

namespace cgmli { namespace backend {
FixedLagBackend::FixedLagBackend(const BackendOptions& o) : options_(o),window_(o),optimizer_(o) { options_.validate(); }
bool FixedLagBackend::append(const StateEstimate& prediction,const MultiFrame& frame,
    const ImuMeasurementDeque& imu,std::string& error) {
    auto staged=window_;
    if(!staged.append(prediction,error) || !staged.attachFrame(frame,error) || !staged.attachImu(prediction.id,imu,error)) return false;
    window_=std::move(staged); return true;
}
bool FixedLagBackend::initializeCalibration(const std::vector<ParameterBlock>& blocks, std::string& error) {
    if (window_.size()) { error="calibration must be initialized before any FusionEpoch"; return false; }
    auto candidate=window_;
    // Validate the whole calibration insertion on a copy; a later invalid block must
    // not leave only half of the camera/LiDAR calibration installed.
    for (auto block:blocks) {
        if (!block.key.isCalibration()) { error="initialization accepts calibration blocks only"; return false; }
        block.constant=true;
        if (!candidate.addParameter(std::move(block),error)) return false;
    }
    window_=std::move(candidate); return true;
}
bool FixedLagBackend::applyFrontendUpdate(StateId id,const std::optional<kinematics::Transformation>& T_WB,
    const std::vector<KeypointIdentifier>& rejected,std::string& error) {
    if(window_.order_.empty() || window_.order_.back()!=id) {
        error="frontend consensus may only update the newest active epoch"; return false;
    }
    if(T_WB) {
        const auto published=snapshot();
        if(published) for(const auto& state:published->states) if(state.id==id) {
            error="frontend reseeding cannot overwrite an already published pose"; return false;
        }
        if(!ParameterBlock::pose(BlockKey::pose(id),*T_WB,id).valid(&error)) return false;
    }
    auto staged=window_;
    for(const auto& key:rejected) {
        const auto observation=staged.landmarks_.find(key);
        if(key.frameId!=id.value() || !observation || observation->use==ObservationUse::Absorbed) {
            error="frontend cannot reject missing, historical or other-epoch evidence"; return false;
        }
        if(!staged.removeObservationImpl(key,error)) return false;
    }
    if(T_WB) staged.graph_.blocks.at(BlockKey::pose(id)).value=ParameterBlock::pose(BlockKey::pose(id),*T_WB,id).value;
    if(!staged.valid(error)) return false;
    window_=std::move(staged); return true;
}
bool FixedLagBackend::applyFrontendBatch(StateId id,const VisualUpdateBatch& batch,
    const std::optional<kinematics::Transformation>& T_WB,SensorHealth visual,bool keyframe,std::string& error) {
    if(window_.order_.empty() || window_.order_.back()!=id) {
        error="frontend batch requires the newest active epoch"; return false;
    }
    const auto published=snapshot();
    if(published) for(const auto& state:published->states) if(state.id==id) {
        error="frontend batch must precede publication of its epoch"; return false;
    }
    if(!batch.removedObservations.empty()) {
        error="frontend association cannot rewrite historical evidence"; return false;
    }
    // Only tentative, unused seeds may be reclaimed by admission control. They
    // have no rho, live residual or absorbed information; mature points still
    // require the explicit backend Schur-retirement API.
    for(auto seed:batch.retiredLandmarks) {
        const auto point=window_.landmarks().find(seed);
        if(!point || point->initialized || window_.graph().blocks.count({BlockKind::Landmark,seed.value()})) {
            error="frontend can retire only uninitialized seeds"; return false;
        }
        for(const auto& key:window_.landmarks().ofLandmark(seed)) {
            const auto observation=window_.landmarks().find(key);
            if(key.frameId==id.value() || !observation || observation->use!=ObservationUse::TrackingOnly
                || observation->factorId || window_.landmarks().consumed(key)) {
                error="frontend seed retirement would discard active/absorbed evidence"; return false;
            }
        }
    }
    for(const auto& observation:batch.observations) if(observation.key.frameId!=id.value()) {
        error="frontend extra observations must belong to the current epoch"; return false;
    }
    // A malformed depth initialization must not affect an unrelated track. Each
    // promoted seed must already have, or receive in this batch, a current view.
    std::set<LandmarkId> currentViews;
    for(const auto& observation:batch.observations) currentViews.insert(observation.landmark);
    for(const auto& key:window_.landmarks().inFrame(id)) currentViews.insert(window_.landmarks().find(key)->landmark);
    for(const auto& initialization:batch.initializations) {
        const bool current=currentViews.count(initialization.id)!=0;
        if(!current) { error="frontend depth initialization has no current observation"; return false; }
    }
    for(const auto& reference:batch.seedAnchors) {
        if(!currentViews.count(reference.id)) { error="frontend seed reference change has no current support"; return false; }
    }
    for(const auto& landmark:batch.landmarks) {
        const bool current=landmark.anchor.frameId==id.value() || currentViews.count(landmark.id)!=0;
        if(!current) { error="frontend track has no current observation"; return false; }
    }
    if(T_WB && !ParameterBlock::pose(BlockKey::pose(id),*T_WB,id).valid(&error)) return false;
    const auto copyStarted=std::chrono::steady_clock::now();
    auto staged=window_;
    const auto applyStarted=std::chrono::steady_clock::now();
    if(T_WB) staged.graph_.blocks.at(BlockKey::pose(id)).value=ParameterBlock::pose(BlockKey::pose(id),*T_WB,id).value;
    const auto lidar=staged.epochs().at(id).lidar;
    // Health changes and all promoted/added tracks share one activation sweep.
    // This is still a private candidate; every error discards all its mutations.
    staged.deferPendingActivation_=true;
    if(!staged.setHealth(id,visual,lidar) || !staged.applyVisualUpdatesImpl(batch,error)
        || !staged.setKeyframe(id,keyframe)) return false;
    staged.deferPendingActivation_=false; staged.activatePending();
    const auto validationStarted=std::chrono::steady_clock::now();
    if(!staged.valid(error)) return false;
    const auto complete=std::chrono::steady_clock::now();
    size_t rematured=0;
    for(const auto& initialization:batch.initializations)
        if(staged.landmarks().find(initialization.id)->initializationCount>1) ++rematured;
    const auto ms=[](auto a,auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    if(VLOG_IS_ON(1) || id.value()%30==0 || ms(copyStarted,complete)>20)
        LOG(INFO)<<"[Timing][FrontendCommit] frame="<<id.value()<<" copy_ms="<<ms(copyStarted,applyStarted)
            <<" apply_ms="<<ms(applyStarted,validationStarted)<<" validate_ms="<<ms(validationStarted,complete)
            <<" landmarks="<<staged.landmarks().size()<<" observations="<<staged.landmarks().observations().size()
            <<" promoted="<<batch.initializations.size()<<" rematured="<<rematured
            <<" seed_anchor_updates="<<batch.seedAnchors.size()<<" transaction=accepted";
    window_=std::move(staged);
    if(VLOG_IS_ON(1)) {
      const auto logPromotion=[&](LandmarkId landmarkId) {
        const auto& point=*window_.landmarks().find(landmarkId);
        LOG(INFO)<<"[LandmarkLifecycle] frame="<<id.value()<<" landmark="<<landmarkId.value()
            <<" action="<<(point.initializationCount>1?"remature":"mature")<<" anchor="<<point.anchor.frameId
            <<" camera="<<point.anchor.cameraIndex<<" initializations="<<point.initializationCount
            <<" resets="<<point.resetCount<<" demotions="<<point.demotionCount<<" transaction=frontend_accepted";
      };
      for(const auto& initialization:batch.initializations) logPromotion(initialization.id);
      for(const auto& landmark:batch.landmarks) if(landmark.inverseDepth) logPromotion(landmark.id);
    }
    return true;
}
std::shared_ptr<const BackendSnapshot> FixedLagBackend::makeSnapshot(uint64_t revision) const {
    // Copy accepted surviving states and matching calibration. No raw optimization
    // pointers escape, so readers cannot observe writes during the next transaction.
    auto snapshot=std::make_shared<BackendSnapshot>(); snapshot->revision=revision;
    for (auto id:window_.order()) snapshot->states.push_back(*window_.estimate(id));
    for (const auto& p:window_.graph().blocks) if (p.first.isCalibration()) snapshot->calibration.emplace(p.first,p.second.value);
    for(const auto& item:window_.landmarks().landmarks()) {
        auto estimate=window_.landmark(item.first);
        if(!estimate) throw std::runtime_error("failed to export a managed landmark");
        snapshot->landmarks.push_back(std::move(*estimate));
    }
    if (!window_.order().empty()) {
        const auto& e=window_.epochs().at(window_.order().back());
        snapshot->imuOnly=e.visual==SensorHealth::Invalid && e.lidar==SensorHealth::Invalid;
    }
    return snapshot;
}
StepReport FixedLagBackend::step(std::optional<MarginalizationPolicy> policy,
    const CalibrationUpdatePolicy& calibration, const SolutionValidator& validator, const MapCommitGates& gates,
    const LandmarkReanchorHook& reanchor, const SolveBudget& budget) {
    StepReport out; out.snapshot=snapshot();
    if (!window_.size()) { out.message="empty backend window"; return out; }
    const auto inputValidationStarted=std::chrono::steady_clock::now();
    if(!window_.valid(out.message)) return out;
    const auto checkpointStarted=std::chrono::steady_clock::now();
    out.inputValidationMs=std::chrono::duration<double,std::milli>(checkpointStarted-inputValidationStarted).count();
    auto checkpoint=window_;
    out.checkpointMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-checkpointStarted).count();
    // The checkpoint includes already-staged measurements/predictions. Failure rolls
    // back this attempt, not input submission; the caller can repair/retry the same epoch.
    const auto rollback=[&](const std::string& text) {
        window_=std::move(checkpoint); out.message=text; out.solve.accepted=false;
        out.solve.termination=SolveTermination::Failure; out.solve.unrefittedOutliers=0; out.mapCandidate.reset();
        out.retiredState.reset(); out.retiredVisualPoints.clear(); out.optimizedVisualHistory.clear(); out.rejectedVisualObservations.clear(); return out;
    };
    const auto& newest=window_.epochs().at(window_.order().back());
    bool connected=false;
    const BlockKey newestPose=BlockKey::pose(newest.id);
    // Sensor packets may arrive independently. Do not publish an unconstrained latest
    // prediction just because older epochs still have enough residuals to run Ceres.
    for (const auto& item:window_.graph_.factors) if (item.second.participates())
        for (auto k:item.second.keys) connected=connected || k==newestPose;
    if (window_.graph_.prior) for (auto k:window_.graph_.prior->keys()) connected=connected || k==newestPose;
    if (!connected) { out.message="latest epoch pose has no active constraint; awaiting measurements"; return out; }
    try {
        // Compute permission decisions from one immutable graph view, before changing any constant flags.
        std::map<BlockKey,bool> allow;
        for (const auto& item:window_.graph_.blocks) if (item.first.isCalibration()) {
            const bool camera=item.first.kind==BlockKind::CameraExtrinsic || item.first.kind==BlockKind::CameraTimeOffset;
            const bool healthy=(camera?newest.visual:newest.lidar)==SensorHealth::Valid;
            allow[item.first]=healthy && calibration && calibration(item.first,window_.graph_);
        }
        for (const auto& item:allow) window_.graph_.blocks.at(item.first).constant=!item.second;
        // NIS gating may remove the latest pose's last constraint. Check connectivity
        // again at the final candidate, then invoke the application's stronger quality gate.
        const SolutionValidator validateConnected=[&](const BackendGraph& graph,std::string& reason) {
            bool constrained=false;
            for (const auto& item:graph.factors) if (item.second.participates())
                for (auto k:item.second.keys) constrained=constrained || k==newestPose;
            if (graph.prior) for (auto k:graph.prior->keys()) constrained=constrained || k==newestPose;
            if (!constrained) { reason="latest epoch lost all pose constraints after NIS gating"; return false; }
            return !validator || validator(graph,reason);
        };
        const auto imuDiagnostics=[&](const char* phase) {
            if(!VLOG_IS_ON(1) && newest.id.value()%30!=0) return;
            if(std::string(phase)=="post_solve" && window_.graph_.prior) {
                const auto& prior=*window_.graph_.prior; std::vector<const double*> parameters;
                for(auto key:prior.keys()) parameters.push_back(window_.graph_.blocks.at(key).value.data());
                std::vector<MarginalizationPrior::BlockDiagnostic> blocks; double frozen=0,total=0;
                if(prior.diagnose(parameters.data(),blocks,frozen,total)) {
                    LOG(INFO)<<"[PriorResidual] frame="<<newest.id.value()<<" frozen_norm="<<frozen
                        <<" total_norm="<<total<<" components="<<prior.num_residuals()
                        <<" cross_terms_retained=1";
                    for(const auto& d:blocks) LOG(INFO)<<"[PriorBlock] frame="<<newest.id.value()
                        <<" kind="<<int(d.key.kind)<<" id="<<d.key.id<<" contribution_norm="<<d.contributionNorm
                        <<" reference_gradient_norm="<<d.referenceGradientNorm
                        <<" translation_delta_m="<<d.translationDelta<<" rotation_delta_rad="<<d.rotationDelta
                        <<" velocity_delta_mps="<<d.velocityDelta<<" gyro_bias_delta="<<d.gyroBiasDelta
                        <<" accel_bias_delta="<<d.accelBiasDelta;
                }
            }
            FactorId worst=0; double largest=-1; factor::BodyImuDiagnostics original;
            for(const auto& item:window_.graph_.factors) {
                const auto& f=item.second;
                if(f.kind!=FactorKind::Imu || f.lifecycle!=FactorLifecycle::Active || !f.imuReplay) continue;
                if(!f.solveEnabled) {
                    LOG(INFO)<<"[ImuIntervalSuppressed] frame="<<newest.id.value()<<" phase="<<phase<<" factor="<<f.id
                        <<" from="<<f.keys[0].id<<" to="<<f.keys[2].id
                        <<" dt_s="<<(f.imuReplay->end-f.imuReplay->start).toSec()
                        <<" samples="<<f.imuReplay->samples.size()<<" raw_complete="<<f.imuReplay->completeSamples
                        <<" max_duration_s="<<options_.maxImuIntervalSeconds;
                    continue;
                }
                std::vector<const double*> x;
                for(auto key:f.keys) x.push_back(window_.graph_.blocks.at(key).value.data());
                factor::BodyImuDiagnostics d;
                if(!factor::diagnoseBodyImu(*f.cost,x.data(),d)) continue;
                if(d.nis>largest) { worst=f.id; largest=d.nis; original=d; }
                LOG(INFO)<<"[ImuResidual] frame="<<newest.id.value()<<" phase="<<phase<<" factor="<<f.id
                    <<" from="<<f.keys.front().id<<" to="<<f.keys[2].id<<" dt_s="<<d.duration<<" nis="<<d.nis
                    <<" position_m="<<d.physical.segment<3>(0).norm()<<" rotation_2qvec="<<d.physical.segment<3>(3).norm()
                    <<" velocity_mps="<<d.physical.segment<3>(6).norm()<<" gyro_bias_difference="<<d.physical.segment<3>(9).norm()
                    <<" accel_bias_difference="<<d.physical.segment<3>(12).norm()
                    <<" reference_delta_bg="<<d.deltaGyroBias<<" reference_delta_ba="<<d.deltaAccelBias;
                VLOG(1)<<"[ImuCovariance] frame="<<newest.id.value()<<" factor="<<f.id<<" dt_s="<<d.duration
                    <<" position_sigma_norm_m="<<std::sqrt(std::max(0.0,d.covariance.block<3,3>(0,0).trace()))
                    <<" rotation_sigma_norm_rad="<<std::sqrt(std::max(0.0,d.covariance.block<3,3>(3,3).trace()))
                    <<" velocity_sigma_norm_mps="<<std::sqrt(std::max(0.0,d.covariance.block<3,3>(6,6).trace()));
            }
            if(worst && std::string(phase)=="post_solve") try {
                // Independent full replay of the worst retained interval at the
                // accepted left bias. Diagnostic only: never alter the objective
                // after acceptance or mutate a cost shared with a checkpoint.
                const auto& f=window_.graph_.factors.at(worst);
                const auto first=window_.estimate(StateId(f.keys[0].id)),last=window_.estimate(StateId(f.keys[2].id));
                if(!first || !last) return;
                const auto replay=factor::makeBodyImuFactor(f.imuReplay->samples,*f.imuReplay->parameters,*first,*last);
                std::vector<const double*> x;
                for(auto key:f.keys) x.push_back(window_.graph_.blocks.at(key).value.data());
                factor::BodyImuDiagnostics fresh;
                if(!factor::diagnoseBodyImu(*replay,x.data(),fresh)) return;
                LOG(INFO)<<"[ImuReplayCheck] frame="<<newest.id.value()<<" factor="<<worst
                    <<" from="<<first->id.value()<<" to="<<last->id.value()<<" dt_s="<<fresh.duration
                    <<" stored_nis="<<original.nis<<" fresh_nis="<<fresh.nis
                    <<" position_difference_m="<<(fresh.physical.head<3>()-original.physical.head<3>()).norm()
                    <<" velocity_difference_mps="<<(fresh.physical.segment<3>(6)-original.physical.segment<3>(6)).norm()
                    <<" covariance_relative_change="<<(fresh.covariance-original.covariance).norm()/std::max(1e-30,original.covariance.norm())
                    <<" velocity_sigma_norm_mps="<<std::sqrt(std::max(0.0,fresh.covariance.block<3,3>(6,6).trace()));
                // Sample expensive derivative/time-sensitivity audits only in the
                // diagnostic region or on a statistically strained retained edge.
                // No accepted factor, state, calibration or FEJ reference is edited.
                const bool watched=newest.id.value()==840 || newest.id.value()==1920;
                if(newest.id.value()%30==0 && (newest.id.value()==300 || watched || original.nis>30)) {
                    const auto auditStarted=std::chrono::steady_clock::now();
                    factor::BodyImuJacobianCheck check;
                    const bool evaluated=factor::checkBodyImuJacobians(*f.cost,x.data(),check);
                    LOG(INFO)<<"[ImuJacobianCheck] frame="<<newest.id.value()<<" factor="<<worst
                        <<" evaluated="<<evaluated<<" columns="<<check.columns<<" failed_columns="<<check.failedColumns
                        <<" max_scaled_error="<<check.maxScaledError<<" max_absolute_error="<<check.maxAbsoluteError
                        <<" worst_block="<<check.worstBlock<<" worst_column="<<check.worstColumn
                        <<" atol=1e-6 rtol=1e-6";
                    const auto& samples=f.imuReplay->samples;
                    double maxGap=0;
                    for(size_t i=1;i<samples.size();++i)
                        if(samples[i-1].timeStamp<last->timestamp && samples[i].timeStamp>first->timestamp)
                            maxGap=std::max(maxGap,(samples[i].timeStamp-samples[i-1].timeStamp).toSec());
                    LOG(INFO)<<"[ImuIntervalCheck] frame="<<newest.id.value()<<" factor="<<worst
                        <<" start_ns="<<first->timestamp.toNSec()<<" end_ns="<<last->timestamp.toNSec()
                        <<" samples="<<samples.size()<<" max_gap_s="<<maxGap
                        <<" left_bracket_s="<<(first->timestamp-samples.front().timeStamp).toSec()
                        <<" right_bracket_s="<<(samples.back().timeStamp-last->timestamp).toSec();
                    for(const auto& endpoint:{*first,*last}) {
                        const auto frame=window_.frames().measurements(endpoint.id); if(!frame) continue;
                        for(size_t camera=0;camera<frame->numFrames();++camera)
                            if(const auto& timing=frame->imageTiming(camera))
                                LOG(INFO)<<"[ImuImageTiming] frame="<<newest.id.value()<<" factor="<<worst
                                    <<" endpoint="<<endpoint.id.value()<<" camera="<<camera
                                    <<" node_ns="<<endpoint.timestamp.toNSec()<<" image_ns="<<timing->originalTimestamp.toNSec()
                                    <<" image_offset_ns="<<timing->offsetNs<<" intra_bundle_compensation=0";
                    }
                    for(int sign:{-1,1}) {
                        auto a=*first,b=*last;
                        const Duration shift(0.001);
                        if(sign<0) { a.timestamp=a.timestamp-shift; b.timestamp=b.timestamp-shift; }
                        else { a.timestamp=a.timestamp+shift; b.timestamp=b.timestamp+shift; }
                        std::string reason;
                        if(!factor::validImuInterval(samples,a.timestamp,b.timestamp,reason,f.imuReplay->parameters->max_sample_gap)) {
                            LOG(INFO)<<"[ImuTimeSensitivity] frame="<<newest.id.value()<<" factor="<<worst
                                <<" shift_s="<<sign*0.001<<" evaluated=0 reason=insufficient_raw_bracket"; continue;
                        }
                        const auto shifted=factor::makeBodyImuFactor(samples,*f.imuReplay->parameters,a,b);
                        factor::BodyImuDiagnostics d;
                        const bool ok=factor::diagnoseBodyImu(*shifted,x.data(),d);
                        LOG(INFO)<<"[ImuTimeSensitivity] frame="<<newest.id.value()<<" factor="<<worst
                            <<" shift_s="<<sign*0.001<<" evaluated="<<ok<<" shifted_nis="<<d.nis
                            <<" reference_nis="<<fresh.nis<<" fixed_states=1";
                    }
                    if(watched) {
                        for(const auto& state:{*first,*last}) {
                            const auto& q=state.T_WB.q(); const auto& p=state.T_WB.r(); const auto& sb=state.speedBias;
                            LOG(INFO)<<std::setprecision(17)<<"[ImuAuditState] frame="<<newest.id.value()<<" id="<<state.id.value()
                                <<" timestamp_ns="<<state.timestamp.toNSec()<<" px="<<p.x()<<" py="<<p.y()<<" pz="<<p.z()
                                <<" qx="<<q.x()<<" qy="<<q.y()<<" qz="<<q.z()<<" qw="<<q.w()
                                <<" vx="<<sb[0]<<" vy="<<sb[1]<<" vz="<<sb[2]
                                <<" bgx="<<sb[3]<<" bgy="<<sb[4]<<" bgz="<<sb[5]
                                <<" bax="<<sb[6]<<" bay="<<sb[7]<<" baz="<<sb[8]<<" stage=post_solve_pre_schur";
                        }
                        LOG(INFO)<<std::setprecision(17)<<"[ImuAuditReference] frame="<<newest.id.value()<<" factor="<<worst
                            <<" bgx="<<original.referenceBias[0]<<" bgy="<<original.referenceBias[1]<<" bgz="<<original.referenceBias[2]
                            <<" bax="<<original.referenceBias[3]<<" bay="<<original.referenceBias[4]<<" baz="<<original.referenceBias[5];
                        for(int row=0;row<15;++row) {
                            std::ostringstream values; values<<std::setprecision(17);
                            for(int column=0;column<15;++column) { if(column) values<<','; values<<original.covariance(row,column); }
                            LOG(INFO)<<"[ImuAuditCovariance] frame="<<newest.id.value()<<" factor="<<worst<<" row="<<row<<" values="<<values.str();
                        }
                        // Marginal 3D NIS values use covariance SUBMATRICES.
                        // They cannot be summed to obtain the correlated 15D NIS.
                        const char* groups[]{"position","rotation_2qvec","velocity","gyro_bias","accel_bias"};
                        for(int i=0;i<5;++i) {
                            const Eigen::Matrix3d covariance=fresh.covariance.block<3,3>(3*i,3*i);
                            const Eigen::Vector3d error=fresh.physical.segment<3>(3*i);
                            Eigen::LDLT<Eigen::Matrix3d> ldlt(covariance);
                            const bool valid=ldlt.info()==Eigen::Success && (ldlt.vectorD().array()>0).all();
                            const double marginal=valid?error.dot(ldlt.solve(error)):std::numeric_limits<double>::quiet_NaN();
                            LOG(INFO)<<"[ImuConflict] frame="<<newest.id.value()<<" factor="<<worst<<" group="<<groups[i]
                                <<" evaluated="<<valid<<" marginal_nis="<<marginal<<" sigma_norm="<<std::sqrt(std::max(0.0,covariance.trace()))
                                <<" error_x="<<error.x()<<" error_y="<<error.y()<<" error_z="<<error.z()<<" additive=0";
                        }
                        const auto balance=optimizer_.constraintGradients(window_.graph_,f.keys);
                        if(!balance.success) LOG(WARNING)<<"[ConstraintBalance] frame="<<newest.id.value()<<" evaluated=0 reason="<<balance.message;
                        else {
                            const char* kinds[]{"visual","imu","lidar","prior","other"};
                            for(size_t block=0;block<balance.keys.size();++block) {
                                const int offset=balance.offsets[block],size=balance.sizes[block];
                                const char* poseGroups[]{"translation","rotation"};
                                const char* sbGroups[]{"velocity","gyro_bias","accel_bias"};
                                for(int component=0;component<size/3;++component) {
                                    const Eigen::Vector3d imuGradient=balance.gradients[1].segment<3>(offset+3*component);
                                    Eigen::Vector3d sum=Eigen::Vector3d::Zero();
                                    for(const auto& g:balance.gradients) sum+=g.segment<3>(offset+3*component);
                                    for(size_t category=0;category<balance.gradients.size();++category) if(balance.factors[category]) {
                                        const Eigen::Vector3d g=balance.gradients[category].segment<3>(offset+3*component);
                                        const double scale=g.norm()*imuGradient.norm();
                                        LOG(INFO)<<"[ConstraintBalance] frame="<<newest.id.value()<<" evaluated=1 factor="<<worst
                                            <<" block_kind="<<int(balance.keys[block].kind)<<" id="<<balance.keys[block].id
                                            <<" group="<<(size==6?poseGroups[component]:sbGroups[component])<<" kind="<<kinds[category]
                                            <<" incident_factors_total="<<balance.factors[category]<<" gradient_norm="<<g.norm()
                                            <<" cosine_with_imu="<<(scale>1e-30?g.dot(imuGradient)/scale:0)
                                            <<" cosine_defined="<<(scale>1e-30)<<" total_gradient_norm="<<sum.norm()
                                            <<" chart=current_left_tangent robust_weighted=1";
                                    }
                                }
                            }
                        }
                    }
                    LOG(INFO)<<"[Timing][ImuAudit] frame="<<newest.id.value()<<" elapsed_ms="
                        <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-auditStarted).count();
                }
            } catch(const std::exception& e) {
                LOG(WARNING)<<"[ImuReplayCheck] diagnostic replay failed; factor="<<worst<<" reason="<<e.what();
            }
        };
        imuDiagnostics("pre_refresh");
        const auto refreshStarted=std::chrono::steady_clock::now();
        size_t refreshed=0;
        for(auto& item:window_.graph_.factors) {
            auto& f=item.second;
            if(f.kind!=FactorKind::Imu || !f.participates() || !f.imuReplay) continue;
            const auto first=window_.estimate(StateId(f.keys.front().id)),last=window_.estimate(StateId(f.keys[2].id));
            if(!first || !last) return rollback("native IMU refresh endpoint missing");
            // Match the legacy gyro repropagation threshold. Accelerometer drift
            // also invalidates a first-order bias approximation on long-lived edges.
            if(factor::bodyImuNeedsRefresh(*f.cost,first->speedBias,0.0003,0.02)) {
                f.cost=factor::makeBodyImuFactor(f.imuReplay->samples,*f.imuReplay->parameters,*first,*last); ++refreshed;
            }
        }
        const double refreshMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-refreshStarted).count();
        const auto* visualObservations=budget.collectCameraLandmarkCounts?&window_.landmarks():nullptr;
        out.solve=optimizer_.optimizeImpl(window_.graph_,validateConnected,budget,false,visualObservations,newest.id);
        out.solve.refreshedImuFactors=refreshed;
        out.solve.imuRefreshMs=refreshMs;
        if (!out.solve.accepted) return rollback(out.solve.message);
        imuDiagnostics("post_solve");
        const auto maintenanceStarted=std::chrono::steady_clock::now();
        for(const auto& item:window_.landmarks().observations()) if(item.second.factorId) {
            const auto factor=window_.graph_.factors.find(item.second.factorId);
            if(factor!=window_.graph_.factors.end() && factor->second.lifecycle==FactorLifecycle::Rejected)
                out.rejectedVisualObservations.push_back(item.first);
        }
        window_.collectTerminalFactors();
        // Eviction performs the final full domain validation on its private
        // candidate. Without eviction this is the publication boundary check.
        if(!window_.needsMarginalization() && !window_.valid(out.message)) return rollback(out.message);
        if(options_.exportVisualHistory) for(const auto& item:window_.landmarks().landmarks()) {
            const auto point=window_.landmark(item.first);
            if(point) out.optimizedVisualHistory.push_back(*point);
        }
        const auto outgoingStarted=std::chrono::steady_clock::now();
        out.maintenanceMs=std::chrono::duration<double,std::milli>(outgoingStarted-maintenanceStarted).count();
        std::optional<StateEstimate> retired;
        std::map<LandmarkId,MapPoint> outgoingVisual;
        // Topology changes follow numeric acceptance. Save the outgoing final estimate
        // before deleting its buffers; a failed Schur step rolls back the numeric solve too.
        if (window_.needsMarginalization()) {
            const auto selected=policy.value_or(window_.choosePolicy());
            const auto id=selected==MarginalizationPolicy::MARGIN_OLD?window_.order().front():window_.order()[window_.size()-2];
            retired=window_.estimate(id); // Final accepted values, before erasing buffers.
            for(const auto& item:window_.landmarks().landmarks()) if(item.second.anchor.frameId==id.value()) {
                const auto point=window_.landmark(item.first);
                if(!point || !point->pointWorld) continue;
                uint64_t latest=0;
                for(const auto& observation:point->observations) latest=std::max(latest,observation.key.frameId);
                const auto& p=*point->pointWorld;
                outgoingVisual.emplace(item.first,MapPoint(item.first.value(),Eigen::Vector4d(p.x(),p.y(),p.z(),1),item.second.quality,latest));
            }
            const auto marginalizationStarted=std::chrono::steady_clock::now();
            out.outgoingMs=std::chrono::duration<double,std::milli>(marginalizationStarted-outgoingStarted).count();
            out.marginalization=window_.marginalize(selected,reanchor);
            out.marginalizationMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-marginalizationStarted).count();
            if (!out.marginalization->success) return rollback("marginalization failed: "+out.marginalization->message);
            // SECOND_NEW is a redundancy decision and never emits a normal map candidate.
            // OLD still requires valid LiDAR, keyframe status, and every external map gate.
            if (selected==MarginalizationPolicy::MARGIN_OLD && retired->keyframe
                && retired->lidar==SensorHealth::Valid && gates.passed()) out.mapCandidate=retired;
        }
        // Build the immutable result before publishing a revision; never expose a half-committed state.
        if (revision_==std::numeric_limits<uint64_t>::max()) return rollback("snapshot revision exhausted");
        const auto snapshotStarted=std::chrono::steady_clock::now();
        auto published=makeSnapshot(revision_+1);
        out.snapshotMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-snapshotStarted).count();
        const auto receiptStarted=std::chrono::steady_clock::now();
        if(out.snapshot) for(const auto& previous:out.snapshot->landmarks) {
            if(!window_.landmarks().find(previous.metadata.id)) out.retiredLandmarks.push_back(previous.metadata.id);
            for(const auto& observation:previous.observations) {
                const auto current=window_.landmarks().find(observation.key);
                if(!current || current->landmark!=observation.landmark) out.removedObservations.push_back(observation.key);
            }
        }
        out.retiredState=retired;
        if(out.marginalization) for(auto id:out.marginalization->retiredLandmarks) {
            const auto point=outgoingVisual.find(id);
            if(point!=outgoingVisual.end()) out.retiredVisualPoints.push_back(point->second);
        }
        out.snapshot=published; out.accepted=true; out.message="accepted";
        out.receiptMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-receiptStarted).count();
        ++revision_;
        // Release/acquire shared_ptr publication makes the fully built immutable value
        // visible to readers without allowing them to traverse the mutable window graph.
        std::atomic_store_explicit(&snapshot_,std::move(published),std::memory_order_release);
        return out;
    } catch (const std::exception& e) {
        return rollback(std::string("backend transaction failed: ")+e.what());
    }
}
CalibrationInformation FixedLagBackend::calibrationInformation(bool prior) {
    return optimizer_.calibrationInformation(window_.graph_,prior);
}
} }
