param()
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$checks=[Collections.Generic.List[object]]::new()
function Check([string]$name,[bool]$passed) { $checks.Add([pscustomobject]@{name=$name;passed=$passed}) }
function Read([string]$path) { [IO.File]::ReadAllText((Join-Path $root $path)) }
$strip='(?s)/\*.*?\*/|//[^\r\n]*|"(?:\\.|[^"\\])*"|''(?:\\.|[^''\\])*'''
$paths=@('src/pipeline/ThreadedSlam.cpp','src/pipeline/ThreadedSlam.hpp','src/pipeline/StateOutput.cpp','src/pipeline/StateOutput.hpp',
    'src/backend/ViSlamBackend.hpp','src/backend/ViSlamBackend.cpp','src/backend/GlobalSlamGraph.hpp','src/backend/GlobalSlamGraph.cpp',
    'src/backend/FixedLagBackend.cpp','src/backend/BackendTypes.hpp','src/factor/BodyImuFactor.hpp','src/factor/BodyImuFactor.cpp',
    'src/factor/ImuError.hpp','src/factor/ImuError.cpp','src/factor/ErrorInterface.hpp','src/factor/ErrorInterface.cpp',
    'src/factor/PoseLocalParameterization.hpp','src/factor/PoseLocalParameterization.cpp',
    'src/frontend/Frontend.cpp','src/frontend/Frontend.hpp','src/frontend/FrontendPoseAdapter.cpp','src/frontend/FrontendPoseAdapter.hpp',
    'src/common/Parameters.hpp','src/common/ViParametersReader.cpp','src/cameras/EucmCamera.hpp','src/cameras/PinholeCamera.hpp','src/cameras/NCameraSystem.cpp','src/cameras/NCameraSystem.hpp',
    'src/pipeline/tests/NativeSlamRegression.cpp','src/app/CsvOffline.cpp','src/visualization/ViVisualizer.cpp','src/visualization/ViVisualizer.hpp')
$raw=@{}; $code=@{}; $compact=@{}
foreach($path in $paths) {
    $source=Read $path; $clean=[regex]::Replace($source,$strip,' ')
    $raw[$path]=$source; $code[$path]=$clean; $compact[$path]=[regex]::Replace($clean,'\s+','')
    $stack=[Collections.Generic.Stack[char]]::new(); $balanced=$true; $pairs=@{')'='(';']'='[';'}'='{'}
    foreach($character in $clean.ToCharArray()) {
        if($character -in '(','[','{') { $stack.Push($character) }
        elseif($pairs.ContainsKey([string]$character)) {
            if(!$stack.Count -or $stack.Pop() -ne $pairs[[string]$character]) { $balanced=$false; break }
        }
    }
    Check ($path+': delimiters') ($balanced -and !$stack.Count)
    Check ($path+': no conflict markers') ($source -notmatch '(?m)^\s*(<<<<<<<|=======|>>>>>>>)')
    foreach($include in [regex]::Matches($source,'(?m)^\s*#include\s*[<"]((?:pipeline|frontend|visual|visualization|backend|common|factor|cameras|kinematics|utils)/[^>"]+)[>"]')) {
        Check ($path+': include '+$include.Groups[1].Value) (Test-Path -LiteralPath (Join-Path $root ('src/'+$include.Groups[1].Value)))
    }
    if($path.EndsWith('.hpp')) {
        $guard=[regex]::Match($source,'(?m)^#ifndef\s+(\w+)'); $definition=[regex]::Match($source,'(?m)^#define\s+(\w+)')
        if($guard.Success) { Check ($path+': include guard matches') ($guard.Groups[1].Value -eq $definition.Groups[1].Value) }
    }
}
$backend=$code['src/backend/ViSlamBackend.hpp']; $source=$code['src/pipeline/ThreadedSlam.cpp']
Check 'ORB calibration caches prepared at startup' ($compact['src/pipeline/ThreadedSlam.cpp'].Contains('frontend_.prepareOrb(parameters_.nCameraSystem,parameters_.frontend.parallelise_detection);'))
$prepareOffset=$source.IndexOf('frontend_.prepareOrb(')
Check 'ORB preparation completes before publisher starts' ($prepareOffset -ge 0 -and $prepareOffset -lt $source.IndexOf('publishingThread_ = std::thread'))
Check 'ORB preparation follows final extraction options' ($prepareOffset -gt $source.IndexOf('frontend_.setOrbOptions(orbOptions)'))
foreach($call in @([regex]::Matches($source,'\b(?:estimator_|candidate)\.([A-Za-z_]\w*)\s*\(') |
    ForEach-Object {$_.Groups[1].Value} | Sort-Object -Unique)) {
    Check ('native facade call declared: '+$call) ($backend -match ('\b'+$call+'\s*\('))
}
$methods=@{
    'ThreadedSlam'=@('initialize','setError','lastError','addImages','addImagesWithTimestamps','addImuMeasurement','addLidarMeasurement','addDepthMeasurement',
        'writeDebugStatisticsCsv','collectImu','predict','extract','stage','processFrame','launchOptimization','finishOptimization',
        'waitForOptimization','optimizeAndPublish','validateSolution','rollbackPendingFrame','setFatalError','publishStep','queuePublication','prepareVisualization','publishingLoop','visualizationLoop','display',
        'stopThreading','writeFinalTrajectoryCsv','saveMap','updateGlobalHistory','doFinalBa','cacheFeatureDebug','prepareFeatureDebug');
    'GlobalSlamGraph'=@('~GlobalSlamGraph','loadVocabulary','canonical','globalPose','findLoop','optimizePoseGraph','ingest','estimates','estimate','correctPoint','mapPoints','cameraExtrinsics','bundleAdjust');
    'ImuError'=@('append','redoPreintegration','syncFrom','setImuParameters','clone','propagation','initPose','Evaluate','EvaluateWithMinimalJacobians','EvaluateWithSigmaGradientAndHessian')
}
foreach($class in $methods.Keys) {
    $path=switch($class){ 'ThreadedSlam' {'src/pipeline/ThreadedSlam.cpp'} 'GlobalSlamGraph' {'src/backend/GlobalSlamGraph.cpp'} 'ImuError' {'src/factor/ImuError.cpp'} }
    foreach($method in $methods[$class]) {
        Check ('one real definition: '+$class+'::'+$method) ([regex]::Matches($code[$path],('\b'+$class+'::'+$method+'\s*\(')).Count -eq 1)
    }
}
$contracts=@(
 @('rig/body initialization','src/pipeline/ThreadedSlam.cpp','estimator_.initializeRig\(parameters_.nCameraSystem'),
 @('same camera models used for extraction','src/pipeline/ThreadedSlam.cpp','frontend_.detectAndDescribe\(c,frame,state.T_WS\*extrinsic,nullptr\)'),
 @('conditional queue reservation','src/pipeline/ThreadedSlam.cpp','cameraQueue_.PopFrontIf'),
 @('image input owns pixels','src/pipeline/ThreadedSlam.cpp','m.measurement.image=item.second.clone\(\)'),
 @('original image times survive input queue','src/pipeline/ThreadedSlam.cpp','m.measurement.originalImageTimestamp=source==originalTimes.end\(\)\?stamp:source->second;'),
 @('original image times survive frame creation','src/pipeline/ThreadedSlam.cpp','frame->setImageTiming\(c,\*packet\[c\].measurement.originalImageTimestamp,packet\[c\].measurement.imageTimeOffsetNs\);'),
 @('timed input regression called','src/pipeline/tests/NativeSlamRegression.cpp','averagedImageInputPreservesTimes\(\);modelSpecificCameraCalibration\(\);'),
 @('feature snapshots captured without backend reads','src/pipeline/ThreadedSlam.cpp','captured=ViVisualizer::captureFeatures\(\*frame\);'),
 @('debug snapshots scheduled before backend result','src/pipeline/ThreadedSlam.cpp','prepareFeatureDebug\(frame,pendingAssociation_\)'),
 @('rendered feature views stay separate from reprojection','src/pipeline/ThreadedSlam.cpp','ViVisualizer::drawFeatureDebug\(\*data->featureDebug\)'),
 @('feature debug rendering regression called','src/pipeline/tests/NativeSlamRegression.cpp','try\{featureDebugSnapshotsAndRendering\(\);'),
 @('debug drawing uses actual endpoint feature IDs','src/visualization/ViVisualizer.cpp','match->first.keypointIndex>=left.keypoints.size\(\)'),
 @('debug drawing scales original keypoints','src/visualization/ViVisualizer.cpp','a.x\*left.scaleX\*scaleA'),
 @('bounded lines preserve both diagnostic classes','src/visualization/ViVisualizer.cpp','keptStride=std::max<size_t>\(1,\(retained\+59\)/60\)'),
 @('real IMU initializer','src/pipeline/ThreadedSlam.cpp','factor::initializeImu\(imuBuffer_,parameters_.imu'),
 @('explicit predicted-state insertion','src/pipeline/ThreadedSlam.cpp','candidate.addStates\(\*frame,state,imuBuffer_,error\)'),
 @('real whitened IMU replay record created','src/pipeline/ThreadedSlam.cpp','factor::makeBodyImuRecord\(imuBuffer_,parameters_.imu,\*previous,state\)'),
 @('real IMU record submitted','src/pipeline/ThreadedSlam.cpp','candidate.addFactor\(factor::makeBodyImuRecord'),
 @('eligible replay record wraps real native preintegration','src/factor/BodyImuFactor.cpp','record.cost=record.solveEnabled\?makeBodyImuFactor\(replay->samples,p,first,second\)'),
 @('replay interval retains explicit calibration','src/factor/BodyImuFactor.cpp','replay->parameters=std::make_shared<constImuParameters>\(p\)'),
 @('merged brackets detect conflicting samples','src/factor/BodyImuFactor.cpp','i->timeStamp==j->timeStamp'),
 @('merged interval uses accepted endpoints','src/factor/BodyImuFactor.cpp','std::make_shared<BodyImuCost>\(replay->samples,a,first,last,old,&appended\)'),
 @('merged native IMU regression registered','src/pipeline/tests/NativeSlamRegression.cpp','nativeSecondNewestImuReplay\(\);'),
 @('frontend inputs share one checkpoint','src/pipeline/ThreadedSlam.cpp','frontend_.dataAssociationAndInitialization\(candidate,parameters_,frame,false,&keyframe\)'),
 @('native full solve and marginalization with quality validator','src/pipeline/ThreadedSlam.cpp','estimator_.optimize\(std::nullopt,calibration,quality,\{\},fullBudget\)'),
 @('failure keeps staged input','src/pipeline/ThreadedSlam.cpp','if\(!step.accepted\)\{setError\('),
 @('one pending insertion retried after failed join','src/pipeline/ThreadedSlam.cpp','if\(staged_&&!optimizationThread_.joinable\(\)\)\{launchOptimization\(\);returntrue;\}'),
 @('extraction reads accepted control-thread calibration cache','src/pipeline/ThreadedSlam.cpp','constautosnapshot=extractionSnapshot_;'),
 @('accepted calibration cache refreshed at finish','src/pipeline/ThreadedSlam.cpp','extractionSnapshot_=snapshot;'),
 @('pending work drains worker but quarantines terminal input','src/pipeline/ThreadedSlam.hpp','returnoptimizationThread_.joinable\(\)\|\|\(!faulted_&&\(staged_\|\|preparedFrame_\|\|!cameraQueue_.Empty\(\)\)\)'),
 @('replay input diagnostic excludes active worker and terminal input','src/pipeline/ThreadedSlam.hpp','boolhasPendingImages\(\)const\{return!faulted_&&\(preparedFrame_\|\|!cameraQueue_.Empty\(\)\);\}'),
 @('bootstrap alone waits before prediction','src/pipeline/ThreadedSlam.cpp','if\(!latestLocal_.id.isInitialised\(\)&&!waitForOptimization\(\)\)returnfalse;'),
 @('post-join prediction uses updated accepted state','src/pipeline/ThreadedSlam.cpp','if\(preparedFromState_!=latestLocal_.id\)\{if\(!predict\(\*frame,preparedPrediction_\)\)returnfalse;'),
 @('separate next frame retained across failed solve','src/pipeline/ThreadedSlam.cpp','if\(!waitForOptimization\(\)\)returnfalse;constdoublewaitMs=milliseconds\(waitStarted\);constautoframe=preparedFrame_;'),
 @('new timestamp checked again after join','src/pipeline/ThreadedSlam.cpp','frame->timestamp\(\)<=latestLocal_.timestamp'),
 @('historical outgoing value included','src/pipeline/ThreadedSlam.cpp','if\(step.retiredState\)remember\(\*step.retiredState\)'),
 @('CV predecessor comes from history','src/pipeline/ThreadedSlam.cpp','localHistory_.lower_bound\(id\)'),
 @('publication queue drained on stop','src/pipeline/ThreadedSlam.cpp','while\(publicationQueue_.PopNonBlocking\(&p\)\)publish\(p\)'),
 @('trajectory export uses accepted history','src/pipeline/ThreadedSlam.cpp','pipeline::writeTrajectory\(trajectoryFile_,outputHistory_,rpg_\)'),
 @('sensor/body prediction conversion','src/pipeline/StateOutput.cpp','body.T_WB=sensor.T_WS\*T_BS.inverse\(\)'),
 @('body/sensor publication conversion','src/pipeline/StateOutput.cpp','state.T_WS=body.T_WB\*T_BS'),
 @('sensor velocity lever arm','src/pipeline/StateOutput.cpp','state.v_W=body.speedBias.head<3>\(\)\+\(state.T_WS.C\(\)\*state.omega_S\).cross'),
 @('stationary startup uses guarded pose entry','src/factor/BodyImuFactor.cpp','ImuError::initPose\(interval,initialized,stationary,&statistics\)'),
 @('pose entry checks acceleration covariance','src/factor/ImuError.cpp','s.accelStd>options.maximumAccelStd'),
 @('pose entry checks gyroscope covariance','src/factor/ImuError.cpp','s.gyroStd>options.maximumGyroStd'),
 @('pose entry rejects constant angular motion','src/factor/ImuError.cpp','s.meanAngularSpeed>options.maximumMeanAngularSpeed'),
 @('pose entry enforces sample duration','src/factor/ImuError.cpp','s.duration\+1e-9<options.minimumDuration'),
 @('direct pose entry has a numerical motion regression','src/pipeline/tests/NativeSlamRegression.cpp','stationaryPoseEntryRejectsMotion\(\);'),
 @('sample interval and configured gap checked before integration','src/factor/BodyImuFactor.cpp','validImuInterval\(samples,start,end,error,p.max_sample_gap\)'),
 @('IMU factor uses the same configured gap bound','src/factor/BodyImuFactor.cpp','validImuInterval\(samples,first.timestamp,second.timestamp,error,p.max_sample_gap\)'),
 @('gap test applies only to requested integration support','src/factor/BodyImuFactor.cpp','samples\[i-1\].timeStamp<end&&samples\[i\].timeStamp>start'),
 @('physical IMU gap is bounded','src/factor/BodyImuFactor.cpp','\(samples\[i\].timeStamp-samples\[i-1\].timeStamp\).toSec\(\)>maximumGapSeconds\+1e-9'),
 @('pipeline enables finite configured visual NIS gate','src/pipeline/ThreadedSlam.cpp','options.visualNisThreshold=p.estimator.visual_nis_threshold;'),
 @('backend default visual gate is finite','src/backend/BackendTypes.hpp','doublevisualNisThreshold=9.210340371976184;'),
 @('pipeline rejects disabled visual threshold','src/pipeline/ThreadedSlam.cpp','(?:!std::isfinite\(health.visual_nis_threshold\))'),
 @('camera model required before model-specific validation','src/common/ViParametersReader.cpp','node\[\].isString\(\)'),
 @('EUCM validates its own two model parameters','src/common/ViParametersReader.cpp','numericSequence\(node\[\],2\)'),
 @('camera extrinsic validates complete homogeneous matrix','src/common/ViParametersReader.cpp','numericSequence\(node\[\],16\)'),
 @('invalid reload cannot expose previous rig','src/common/ViParametersReader.cpp','readConfigFile_=false;viParameters_=ViParameters\{\};'),
 @('eight-term distortion validates all eight coefficients','src/common/ViParametersReader.cpp','numericSequence\(coefficients,8\)'),
 @('plain pinhole NoDistortion factory exists','src/common/ViParametersReader.cpp','make_shared<cameras::PinholeCamera<cameras::NoDistortion>>'),
 @('absolute consensus checks inertial agreement','src/frontend/Frontend.cpp','if\(runRansac3d2d\(estimator,plan,decision\)\)\{if\(consistent\(decision\)\)returntrue;'),
 @('relative fallback checks inertial agreement','src/frontend/Frontend.cpp','runRansac2d2d\(estimator,plan,decision\)&&consistent\(decision\)'),
 @('rejected visual pose cannot leak into triangulation','src/frontend/Frontend.cpp','if\(accepted\)frontend_association::reject\(plan,decision.rejected\);else\{decision=PoseDecision\{\};decision.T_WB=predicted;'),
 @('post-solve checks use original inertial prediction','src/pipeline/ThreadedSlam.cpp','inertialPoseConsistent\(pendingPrediction_.T_WB\*parameters_.imu.T_BS'),
 @('post-solve checks surviving independent visual support','src/pipeline/ThreadedSlam.cpp','surviving.size\(\)<10'),
 @('retry budget is bounded','src/pipeline/ThreadedSlam.cpp','optimizationAttempts_<=parameters_.estimator.backend_max_retries'),
 @('frontend stage does not reduce Ceres time allowance','src/pipeline/ThreadedSlam.cpp','pendingBudget_.timeSeconds=options_.maxSolverTimeSeconds;'),
 @('Ceres allowance is visible in timing log','src/pipeline/ThreadedSlam.cpp','step.solve.ceresTimeLimitMs'),
 @('Ceres allowance exhaustion is visible in timing log','src/pipeline/ThreadedSlam.cpp','step.solve.ceresTimeBudgetReached'),
 @('limited acceptance records its stop reason','src/pipeline/ThreadedSlam.cpp','backend::solveTerminationName\(step.solve.termination\)'),
 @('unrefitted live outliers are explicitly reported','src/pipeline/ThreadedSlam.cpp','step.solve.unrefittedOutliers'),
 @('local window uses one frame count','src/pipeline/ThreadedSlam.cpp','options.windowSize=static_cast<size_t>\(p.estimator.window_size\)'),
 @('single frame count defaults to ten','src/common/Parameters.hpp','intwindow_size=10;'),
 @('invalid frame count rejected before unsigned conversion','src/pipeline/ThreadedSlam.cpp','if\(p.estimator.window_size<2\)throw'),
 @('staging moves the exact pre-insertion backend checkpoint','src/pipeline/ThreadedSlam.cpp','stagedCheckpoint_.emplace\(std::move\(estimator_\)\)'),
 @('overlong raw replay is explicitly marked incomplete','src/factor/BodyImuFactor.cpp','replay->completeSamples=left.imuReplay->completeSamples'),
 @('enabled raw replay enforces storage budget','src/factor/BodyImuFactor.cpp','if\(!suppressed&&merged.size\(\)>limit\)'),
 @('partial image-time tails never become append prefixes','src/factor/BodyImuFactor.cpp','constTimeprefixEnd=boundary->timeStamp'),
 @('incremental fractional replay regression invoked','src/pipeline/tests/NativeSlamRegression.cpp','fractionalIncrementalImuMatchesFullReplay\(\);'),
 @('disabled interval regression invoked','src/pipeline/tests/NativeSlamRegression.cpp','longImuIntervalsAreStorageOnly\(\);'),
 @('watched IMU conflict frames are always diagnosed','src/backend/FixedLagBackend.cpp','newest.id.value\(\)==840\|\|newest.id.value\(\)==1920'),
 @('conflict statistics use covariance marginal blocks','src/backend/FixedLagBackend.cpp','fresh.covariance.block<3,3>\(3\*i,3\*i\)'),
 @('constraint balance inspects actual accepted endpoint blocks','src/backend/FixedLagBackend.cpp','optimizer_.constraintGradients\(window_.graph_,f.keys\)'),
 @('prior formation receipt is logged','src/pipeline/ThreadedSlam.cpp','marginal.formation.evaluated'),
 @('false incidence and same-frame elimination counters logged','src/pipeline/ThreadedSlam.cpp','marginal.sameFrameRangeEliminations'),
 @('rollback restores graph prior and managed data together','src/pipeline/ThreadedSlam.cpp','estimator_=std::move\(\*stagedCheckpoint_\)'),
 @('terminal segment fault is exposed to replay','src/app/CsvOffline.cpp','if\(slam.hasFatalError\(\)\)throwstd::runtime_error\(slam.lastError\(\)\)'),
 @('real legacy integrator reused','src/factor/BodyImuFactor.cpp','cgmli::ceres::ImuError::propagation'),
 @('frozen integration on shared cost','src/factor/BodyImuFactor.cpp','fixedLinearization_=true;redo_=false;redoCounter_=1;'),
 @('frozen cache cannot repropagate','src/factor/ImuError.cpp','if\(!fixedLinearization_&&'),
 @('parallel gravity robust initialization','src/factor/ImuError.cpp','Eigen::Quaterniond::FromTwoVectors\(acc_S,Eigen::Vector3d::UnitZ\(\)\)'),
 @('body factor has actual native layout','src/factor/BodyImuFactor.cpp','classBodyImuCostfinal:public::ceres::SizedCostFunction<15,7,9,7,9>'),
 @('chain includes bias-dependent lever arm','src/factor/BodyImuFactor.cpp','omegaS=rawGyro-x.segment<3>\(10\)'),
 @('analytic preintegration Jacobian reused','src/factor/BodyImuFactor.cpp','result=sensorDerivative\*chain'),
 @('EUCM has analytic normalized bearing derivative','src/cameras/EucmCamera.hpp','jacobian=\(Eigen::Matrix3d::Identity\(\)-unit\*unit.transpose\(\)\)\*derivative/ray.norm\(\)'),
 @('EUCM valid domain checked before square root','src/cameras/EucmCamera.hpp','if\(w<0\|\|\(pointJacobian&&w<=1e-12\)\)returnfalse;'),
 @('overlap masks reject invalid rays','src/cameras/NCameraSystem.cpp','if\(!camera->backProject'),
 @('camera insertion defers overlaps by default','src/cameras/NCameraSystem.hpp','boolcomputeOverlaps=false'),
 @('self visibility returns before cross-camera validation','src/cameras/NCameraSystem.hpp','if\(cameraIndexSeenBy==cameraIndex\)returntrue;CGMLI_ASSERT_TRUE\(Exception,overlapComputationValid\('),
 @('self masks generated on demand','src/cameras/NCameraSystem.hpp','if\(cameraIndexSeenBy==cameraIndex\)returnselfOverlapMask\(cameraIndex\);'),
 @('lazy self-mask allocation is synchronized','src/cameras/NCameraSystem.cpp','std::lock_guard<std::mutex>lock\(selfOverlapCache_->mutex\);'),
 @('self pairs skipped before image allocation','src/cameras/NCameraSystem.cpp','visible\[source\]\[source\]=true;.*?if\(target==source\)continue;auto&mask=masks\[target\]\[source\];mask=cv::Mat::zeros'),
 @('recomputed overlap flags start false','src/cameras/NCameraSystem.cpp','std::vector<std::vector<bool>>visible\(count,std::vector<bool>\(count,false\)\);'),
 @('cross masks published from fresh storage','src/cameras/NCameraSystem.cpp','overlapMats_=std::move\(masks\);overlaps_=std::move\(visible\);'),
 @('historical measurements before Schur','src/backend/FixedLagBackend.cpp','out.optimizedVisualHistory.push_back\(\*point\)'),
 @('failed transaction returns no history','src/backend/FixedLagBackend.cpp','out.optimizedVisualHistory.clear\(\);out.rejectedVisualObservations.clear\(\)'),
 @('global loops exclude active window','src/backend/GlobalSlamGraph.cpp','if\(active.count\(id\)'),
 @('global proposals require metric geometry','src/backend/GlobalSlamGraph.cpp','pose_solver::AbsolutePoseSolver\(options\).solve\(input.data,&seed\)'),
 @('trained global vocabulary persists historical entries','src/backend/GlobalSlamGraph.cpp','places_->identities.push_back\(current\)'),
 @('vocabulary checks descriptor format','src/backend/GlobalSlamGraph.cpp','profile.empty\(\)\|\|static_cast<std::string>'),
 @('new keyframes inherit global correction','src/backend/GlobalSlamGraph.cpp','node.global=globalPose\(reference\)\*nodes_.at\(reference\).local.T_WB.inverse\(\)\*state.T_WB'),
 @('global loops require spatial coverage','src/backend/GlobalSlamGraph.cpp','if\(coverage.size\(\)<6\)continue;'),
 @('loop information checked','src/backend/GlobalSlamGraph.cpp','frontend_pose::absoluteInformation'),
 @('loop temporal consistency','src/backend/GlobalSlamGraph.cpp','if\(candidateStreak_<2\)returntrue;'),
 @('inertial pose graph preserves gravity','src/backend/GlobalSlamGraph.cpp','newGravityPoseManifold'),
 @('global corrections never enter local prior','src/backend/GlobalSlamGraph.hpp','!synchroniseRealtimeAndFullGraph'),
 @('global BA rebuilds raw IMU factors','src/backend/GlobalSlamGraph.cpp','factor::makeBodyImuFactor\(it->second.imu,parameters_.imu,a,b\)'),
 @('global BA uses spherical raw measurements','src/backend/GlobalSlamGraph.cpp','NumericDiffCostFunction<BearingCost,::ceres::CENTRAL,2,7,3,7>'),
 @('global BA excludes local marginal priors','src/backend/GlobalSlamGraph.cpp','!graph\.prior|MarginalizationPrior\('),
 @('global BA refines real 3D points','src/backend/GlobalSlamGraph.cpp','problem.AddParameterBlock\(point.second.data\(\),3\)'),
 @('global BA only uses actual multi-view support','src/backend/GlobalSlamGraph.cpp','if\(actualViews\[input.point\].size\(\)<2\)continue;'),
 @('global solver selects an available sparse backend','src/backend/GlobalSlamGraph.cpp','IsSparseLinearAlgebraLibraryTypeAvailable\(::ceres::EIGEN_SPARSE\)'),
 @('final BA publishes calibrated sensor extrinsics','src/pipeline/ThreadedSlam.cpp','state.extrinsics\[c\]=parameters_.imu.T_BS.inverse\(\)\*extrinsics\[c\]'),
 @('main replays sensor measurements','src/app/CsvOffline.cpp','slam.addImuMeasurement'),
 @('main exercises actual SLAM pipeline','src/app/CsvOffline.cpp','slam.processFrame\(\)'),
 @('main per-image waits only for input consumption','src/app/CsvOffline.cpp','for\(intattempts=0;slam.hasPendingImages\(\)&&attempts<maxProcessingAttempts;\+\+attempts\)slam.processFrame\(\);'),
 @('replay processing bound accounts for configured retries','src/app/CsvOffline.cpp','maxProcessingAttempts=2\*\(parameters.estimator.backend_max_retries\+1\)\+4'),
 @('main checks successful visual tracking','src/app/CsvOffline.cpp','if\(visualUpdates.load\(\)==0\)throw'),
 @('main runs configured historical BA','src/app/CsvOffline.cpp','if\(parameters.estimator.do_final_ba\)slam.doFinalBa\(\)')
)
foreach($entry in $contracts) {
    $pattern=$entry[2]; $negative=$pattern.StartsWith('!'); if($negative) { $pattern=$pattern.Substring(1) }
    $ok=$compact[$entry[1]] -match $pattern; if($negative) { $ok=!$ok }; Check $entry[0] $ok
}
$c=$compact['src/pipeline/ThreadedSlam.cpp']; $f=$compact['src/backend/FixedLagBackend.cpp']
# Inspect complete method bodies, excluding comments and literals. These are
# source-order/ownership checks, not a substitute for a runtime race detector.
function MethodBody([string]$name) {
    $start=$c.IndexOf('ThreadedSlam::'+$name+'(')
    if($start -lt 0) { throw ('Missing method: '+$name) }
    $open=$c.IndexOf('{',$start); $depth=1; $end=$open+1
    while($end -lt $c.Length -and $depth -gt 0) {
        if($c[$end] -eq '{') { ++$depth }
        elseif($c[$end] -eq '}') { --$depth }
        ++$end
    }
    return $c.Substring($open+1,$end-$open-2)
}
$process=MethodBody 'processFrame'; $extract=MethodBody 'extract'; $predict=MethodBody 'predict'
$wait=MethodBody 'waitForOptimization'; $finish=MethodBody 'finishOptimization'
$barrier=$process.LastIndexOf('if(!waitForOptimization())returnfalse;')
$extraction=$process.IndexOf('extract(frame,prediction);')
$insertion=$process.IndexOf('stage(frame,preparedPrediction_)')
Check 'normal-path extraction precedes writer barrier' ($extraction -ge 0 -and $barrier -gt $extraction)
Check 'association/insertion follows writer barrier' ($insertion -gt $barrier -and $barrier -ge 0)
Check 'no mutable estimator/history/global access before barrier' ($barrier -ge 0 -and $process.Substring(0,$barrier) -notmatch 'estimator_|localHistory_|outputHistory_|global_->')
Check 'prediction reads no mutable estimator/history' ($predict -notmatch 'estimator_|localHistory_|outputHistory_|global_->')
Check 'extraction reads no live backend snapshot' ($extract -notmatch 'estimator_|localHistory_|global_->')
Check 'one cache captured for every camera in an extraction' ([regex]::Matches($extract,'snapshot=extractionSnapshot_').Count -eq 1 -and $extract -match 'snapshot\?live.at\(c\)')
Check 'worker result inspected only after thread join' ($wait.IndexOf('optimizationThread_.join();') -ge 0 -and $wait.IndexOf('finishOptimization()') -gt $wait.IndexOf('optimizationThread_.join();'))
Check 'failed finish retries before touching accepted caches' ($finish -match 'if\(!accepted_\)\{if\(optimizationRetryable_&&optimizationAttempts_<=parameters_.estimator.backend_max_retries\)\{LOG\(WARNING\)[^;]*;returnfalse;\}rollbackPendingFrame\(\);returntrue;\}' -and $finish.IndexOf('if(!accepted_){') -lt $finish.IndexOf('extractionSnapshot_=snapshot;'))
Check 'worker forwards bounded-refit retry classification' ($compact['src/pipeline/ThreadedSlam.cpp'] -match 'optimizationRetryable_=step.solve.retryable')
$rollback=MethodBody 'rollbackPendingFrame'; $launch=MethodBody 'launchOptimization'
Check 'rollback requires a joined worker' ($rollback -match 'if\(optimizationThread_.joinable\(\)\|\|!stagedCheckpoint_\|\|!pendingFrame_\)throw')
Check 'rollback leaves prepared next frame and IMU interval intact' ($rollback -notmatch 'preparedFrame_.reset|imuBuffer_\.(?:clear|pop)|extractionSnapshot_=|latestLocal_=|publishStep\(')
Check 'retry count changes only on control thread' ($launch -match '\+\+optimizationAttempts_' -and (MethodBody 'optimizeAndPublish') -notmatch '(?:\+\+|--)optimizationAttempts_|optimizationAttempts_(?:\+\+|--|[+*/-]?=(?!=))')
Check 'thread creation failure obeys bounded recovery' ($launch -match 'catch\(\.\.\.\)\{optimizationError_=std::current_exception\(\);finishOptimization\(\);throw;\}')
Check 'faulted processing still drains previous writer' ($process -match 'if\(faulted_\)\{waitForOptimization\(\);if\(staged_\)rollbackPendingFrame\(\);returnfalse;\}')
Check 'no frame input still drains last worker' ($process -match 'if\(!cameraQueue_.getCopyOfFront\(&packet\)\)\{waitForOptimization\(\);returnfalse;\}')
Check 'missing IMU leaves backend running until stop' ($process -match 'if\(!collectImu\(time\)\)\{\+\+imuCoverageWaits_;VLOG_IF\(1,imuCoverageWaits_==1\|\|imuCoverageWaits_%1000==0\)[^;]*;if\(stopping_\)waitForOptimization\(\);returnfalse;\}')
Check 'prepared next frame saved before waiting' ($process.IndexOf('preparedFrame_=frame;') -gt $extraction -and $process.IndexOf('preparedFrame_=frame;') -lt $barrier)
$main=$compact['src/app/CsvOffline.cpp']
Check 'main drains outstanding worker only after image loop' ($main.IndexOf('slam.hasPendingWork()') -gt $main.IndexOf('slam.display(display);') -and $main.IndexOf('slam.hasPendingWork()') -lt $main.IndexOf('slam.stopThreading();'))
foreach($method in @('optimizeAndPublish','publishStep','prepareVisualization')) {
    $body=MethodBody $method
    Check ($method+': no control-thread extraction state access') ($body -notmatch 'preparedFrame_|preparedPrediction_|preparedFromState_|extractionSnapshot_|latestLocal_|previousLocal_|imuBuffer_|frontend_\.(?:detectAndDescribe|propagation)')
}
Check 'raw history captured before marginalization' ($f.IndexOf('optimizedVisualHistory.push_back') -lt $f.IndexOf('window_.marginalize('))
Check 'publication follows accepted solve' ($c.IndexOf('if(!step.accepted)') -lt $c.IndexOf('accepted_=true;publishStep(step)'))
Check 'safe interval loops do not dereference end' ([regex]::Matches($code['src/factor/ImuError.cpp'],'\(it\s*\+\s*1\)\s*!=\s*imuMeasurements_?\.end\(\)').Count -eq 3)
Check 'no obsolete backend declarations' ($backend -notmatch '\b(addImu|addCamera|optimiseRealtimeGraph|applyStrategy|poseSensor|optimiseFullGraph|synchroniseRealtimeAndFullGraph)\s*\(')
Check 'no old pipeline call methods' ($source -notmatch '\b(optimiseRealtimeGraph|applyStrategy|poseSensor|synchroniseRealtimeAndFullGraph)\s*\(')
Check 'global descriptor_profile metadata is read' ($raw['src/backend/GlobalSlamGraph.cpp'].Contains('file["descriptor_profile"]'))
foreach($test in @('modelSpecificCameraCalibration','imuGapAndAtomicPropagation','inertialPoseAcceptanceBounds','failedSolveRollbackPreservesNextFrame','imuGapStopsCurrentSegment','eucmBackProjectionJacobian','stationaryInitializationAndAtomicFailure','bodyImuJacobianAndFrozenCost','nativePipelineImuWindowAndHistory','historicalSphericalImuBundleAdjustment','historicalLoopClosesMetricDrift')) {
    Check ('numerical regression source exists: '+$test) ($code['src/pipeline/tests/NativeSlamRegression.cpp'] -match ('void\s+'+$test+'\('))
    Check ('regression invoked by standalone entry: '+$test) ($compact['src/pipeline/tests/NativeSlamRegression.cpp'].Substring($compact['src/pipeline/tests/NativeSlamRegression.cpp'].IndexOf('intmain()')) -match ($test+'\(\)'))
}
$active=Get-ChildItem -LiteralPath (Join-Path $root 'src') -Recurse -File | Where-Object { $_.Extension -in '.cpp','.hpp','.h','.cc' -and $_.FullName -notmatch '\\legacy\\' }
foreach($file in $active) {
    $text=[IO.File]::ReadAllText($file.FullName); $clean=[regex]::Replace($text,$strip,' ')
    Check ('no satellite dependency: '+$file.Name) ($text -notmatch '(?i)\b\w*(?:gps|gnss)\w*\b|\bT_GW\b')
    Check ('no service registry: '+$file.Name) ($text -notmatch '\bstruct\s+Services\b|pipeline::Services|PipelineInterfaces\.hpp')
    if($file.DirectoryName -notmatch '\\factor$') { Check ('Ceres namespace explicit: '+$file.Name) ($clean -notmatch '(?<!::)\bceres::') }
}
[xml]$project=Read 'CGMLI-SLAM/CGMLI-SLAM.vcxproj'; [xml]$filters=Read 'CGMLI-SLAM/CGMLI-SLAM.vcxproj.filters'
$ns=[Xml.XmlNamespaceManager]::new($project.NameTable); $ns.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
$fn=[Xml.XmlNamespaceManager]::new($filters.NameTable); $fn.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
foreach($path in $paths | Where-Object { $_.StartsWith('src/') -and $_ -notmatch '/tests/' }) {
    $tag=if($path.EndsWith('.cpp')){'ClCompile'}else{'ClInclude'}; $include='..\'+$path.Replace('/','\')
    Check ('project registered: '+$path) (@($project.SelectNodes('//m:'+$tag,$ns) | Where-Object { $_.GetAttribute('Include') -eq $include }).Count -eq 1)
    Check ('filter registered: '+$path) (@($filters.SelectNodes('//m:'+$tag,$fn) | Where-Object { $_.GetAttribute('Include') -eq $include }).Count -eq 1)
}
Check 'standalone regression not application main' (@($project.SelectNodes('//m:ClCompile',$ns) | Where-Object { $_.GetAttribute('Include') -like '*NativeSlamRegression.cpp' }).Count -eq 0)
Check 'removed projection not registered' ($project.OuterXml -notmatch 'original_pipeline_projection')
foreach($entry in $project.SelectNodes('//*[@Include]')) {
    $path=$entry.GetAttribute('Include'); if($path.StartsWith('..\src\')) {
        Check ('registered source path exists: '+$path) (Test-Path -LiteralPath (Join-Path (Join-Path $root 'CGMLI-SLAM') $path))
    }
}
$rigReader=$compact['src/common/ViParametersReader.cpp']
Check 'finite gyro bias derivative uses exponential right Jacobian' ($compact['src/factor/ImuError.cpp'].Contains('kinematics::rightJacobian(phi)'))
Check 'both IMU evaluation paths use finite bias derivative' ([regex]::Matches($compact['src/factor/ImuError.cpp'],'F0.block<3,3>\(3,9\)=finiteGyroBiasJacobian').Count -eq 2)
Check 'dynamic IMU derivative regression registered' ($compact['src/pipeline/tests/NativeSlamRegression.cpp'].Contains('dynamicBodyImuJacobiansAndIntervals();'))
Check 'continuous density rate invariance regression registered' ($compact['src/pipeline/tests/NativeSlamRegression.cpp'].Contains('imuNoiseDensityUnits();'))
Check 'worst retained IMU edge is replayed read-only' ($raw['src/backend/FixedLagBackend.cpp'].Contains('[ImuReplayCheck]'))
Check 'IMU seed diagnostics compare prediction and visual update' ($raw['src/pipeline/ThreadedSlam.cpp'].Contains('prediction_nis=') -and $raw['src/pipeline/ThreadedSlam.cpp'].Contains('after_visual_nis='))
Check 'IMU refresh preserves checkpoint cost regression' ($raw['src/pipeline/tests/NativeSlamRegression.cpp'].Contains('replacement preserves old checkpoint cost'))
foreach($field in @('imuMode','replacedImuFactors','activeImuFactors','droppedVisualFactors','resetLandmarks','demotedLandmarks','movedSeeds','priorReused','priorLandmarks')) {
    Check ('window lifecycle diagnostics logged: '+$field) ($compact['src/pipeline/ThreadedSlam.cpp'].Contains('marginal.'+$field) -and $code['src/backend/BackendTypes.hpp'] -match ('\b'+$field+'\b'))
}
foreach($field in @('lineSearchInterpolation','jacobianEvaluations','residualEvaluations','lineSearchCostMs','lineSearchGradientMs','lineSearchMs')) {
    Check ('solve diagnostics logged: '+$field) ($compact['src/pipeline/ThreadedSlam.cpp'].Contains('step.solve.'+$field) -and $code['src/backend/BackendTypes.hpp'] -match ('\b'+$field+'\b'))
}
Check 'all five camera model paths defer overlap computation' ([regex]::Matches($rigReader,'nCameraSystem.addCamera\([^;]*,false,calibrations\[i\].cameraType\);').Count -eq 5)
Check 'one overlap finalization after the camera loading loop' ([regex]::Matches($rigReader,'nCameraSystem.computeOverlaps\(\)').Count -eq 1 -and $rigReader -match '\+\+camIdx;\}viParameters_.nCameraSystem.computeOverlaps\(\);')
Check 'insertion and extrinsic updates invalidate overlap results' ([regex]::Matches($compact['src/cameras/NCameraSystem.hpp'],'invalidateOverlaps\(\);').Count -eq 2)
Check 'self masks are not allocated in computeOverlaps' ($compact['src/cameras/NCameraSystem.cpp'].Substring($compact['src/cameras/NCameraSystem.cpp'].IndexOf('voidNCameraSystem::computeOverlaps()')) -notmatch 'Mat::ones|selfOverlapMask\(')
$failures=@($checks|Where-Object { !$_.passed })
$report=[ordered]@{checkedAtUtc=[DateTime]::UtcNow.ToString('o');scope='Native SLAM source/API/project integration inspection only';
    compiled=$false;executedCppTests=$false;datasetValidated=$false;missingPipelineDefinitions=@();
    implemented=@('model-specific camera configuration','stationary IMU initialization and gap validation','body/sensor IMU factor','bounded solve retry and full pre-insertion rollback','finite visual NIS and inertial pose quality gates','native local solve and marginalization','historical geometric loops and pose graph','raw visual/IMU historical BA','CSV replay entry point');
    deferred=@('LiDAR/depth submapping and SE2 export','automatic dynamic reinitialization after a terminal IMU gap');passed=$checks.Count-$failures.Count;total=$checks.Count;checks=$checks}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'pipeline_static_report.json'),($report|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
Write-Output ('Native pipeline static checks: {0}/{1}; no compilation or C++ execution' -f ($checks.Count-$failures.Count),$checks.Count)
foreach($failure in $failures) { Write-Output ('FAIL: '+$failure.name) }
if($failures.Count) { exit 1 }
