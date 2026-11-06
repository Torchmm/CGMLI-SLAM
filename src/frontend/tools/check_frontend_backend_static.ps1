param()
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$checks=[Collections.Generic.List[object]]::new()
function Check([string]$name,[bool]$passed) { $checks.Add([pscustomobject]@{name=$name;passed=$passed}) }
function Read([string]$path) { [IO.File]::ReadAllText((Join-Path $root $path)) }
$strip='(?s)/\*.*?\*/|//[^\r\n]*|"(?:\\.|[^"\\])*"|''(?:\\.|[^''\\])*'''
$paths=@('src/frontend/Frontend.cpp','src/frontend/Frontend.hpp','src/frontend/FrontendAssociation.cpp',
    'src/frontend/FrontendAssociation.hpp','src/frontend/tests/FrontendBackendRegression.cpp',
    'src/backend/ViSlamBackend.hpp','src/backend/FixedLagBackend.cpp','src/backend/FixedLagBackend.hpp',
    'src/backend/SlidingWindowManager.hpp','src/backend/LandmarkStore.hpp','src/backend/tests/ManagedWindowRegression.cpp',
    'src/common/ViFrontendInterface.hpp','src/cameras/NCameraSystem.cpp','src/cameras/NCameraSystem.hpp',
    'src/visual/MultiFrame.hpp','src/backend/SlidingWindowData.cpp')
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
    foreach($include in [regex]::Matches($source,'(?m)^\s*#include\s*[<"]((?:frontend|visual|backend|common|cameras|kinematics|utils)/[^>"]+)[>"]')) {
        Check ($path+': include '+$include.Groups[1].Value) (Test-Path -LiteralPath (Join-Path $root ('src/'+$include.Groups[1].Value)))
    }
}
Check 'Estimator is ViSlamBackend' ($compact['src/common/ViFrontendInterface.hpp'].Contains('usingEstimator=ViSlamBackend;'))
Check 'transient anchors defer depth promotion' ($compact['src/frontend/FrontendAssociation.cpp'].Contains('if(transient){++plan.deferredPromotions;continue;}'))
Check 'recovery stereo retains promotion eligibility' ($compact['src/frontend/Frontend.cpp'].Contains('keyframe||!accepted)'))
Check 'keyframe declaration uses a header-visible pose type' ($compact['src/frontend/FrontendAssociation.hpp'].Contains('decideKeyframe(constEstimator&estimator,constMultiFrame&current,constkinematics::Transformation&currentPose,'))
Check 'keyframe definition uses the same qualified pose type' ($compact['src/frontend/FrontendAssociation.cpp'].Contains('decideKeyframe(constEstimator&estimator,constMultiFrame&current,constkinematics::Transformation&currentPose,'))
$backend=$code['src/backend/ViSlamBackend.hpp']
foreach($path in @('src/frontend/Frontend.cpp','src/frontend/FrontendAssociation.cpp')) {
    # Exclude params.estimator: it is configuration, not the backend facade.
    $calls=@([regex]::Matches($code[$path],'(?<![\w.])estimator\.([A-Za-z_]\w*)\s*\(') |
        ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
    foreach($call in $calls) { Check ($path+': backend API '+$call) ($backend -match ('\b'+$call+'\s*\(')) }
    Check ($path+': no templated legacy observation calls') ($code[$path] -notmatch 'estimator\.(addObservation|removeObservation)\s*<')
    Check ($path+': no mutable casts of backend frames') ($code[$path] -notmatch 'const_cast|const_pointer_cast|MultiFramePtr')
    Check ($path+': no implicit optimization') ($code[$path] -notmatch '\b(estimator\.(optimize|step)|optimiseRealtimeGraph|fullGraph_|realtimeGraph_)')
    Check ($path+': no homogeneous camera dispatch') ($code[$path] -notmatch 'CAMERA_GEOMETRY|geometryAs<|distortionType\(')
}
[xml]$project=Read 'CGMLI-SLAM/CGMLI-SLAM.vcxproj'; [xml]$filters=Read 'CGMLI-SLAM/CGMLI-SLAM.vcxproj.filters'
$ns=[Xml.XmlNamespaceManager]::new($project.NameTable); $ns.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
$fn=[Xml.XmlNamespaceManager]::new($filters.NameTable); $fn.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
foreach($entry in @(@('ClCompile','FrontendAssociation.cpp'),@('ClInclude','FrontendAssociation.hpp'),
    @('None','tests\FrontendBackendRegression.cpp'),@('None','tools\check_frontend_backend_static.ps1'))) {
    $tag=$entry[0]; $path='..\src\frontend\'+$entry[1]
    Check ('project registration: '+$entry[1]) (@($project.SelectNodes('//m:'+$tag,$ns) | Where-Object { $_.GetAttribute('Include') -eq $path }).Count -eq 1)
    Check ('filter registration: '+$entry[1]) (@($filters.SelectNodes('//m:'+$tag,$fn) | Where-Object { $_.GetAttribute('Include') -eq $path }).Count -eq 1)
}
Check 'regression excluded from application build' (@($project.SelectNodes('//m:ClCompile',$ns) | Where-Object { $_.GetAttribute('Include') -like '*FrontendBackendRegression.cpp' }).Count -eq 0)
$contracts=@(
    @('frozen measurement identity checked','src/frontend/Frontend.cpp','frontend_association::sameMeasurements\(\*framesInOut,\*frozen\)'),
    @('descriptor bytes checked without flattening row stride','src/frontend/FrontendAssociation.cpp','cv::norm\(a,b,cv::NORM_INF\)'),
    @('keypoint identity includes octave and geometry','src/frontend/FrontendAssociation.cpp','x.pt!=y.pt\|\|x.size!=y.size\|\|x.angle!=y.angle\|\|x.octave!=y.octave'),
    @('explicit detached proposal avoids incomplete Frame copy','src/frontend/FrontendAssociation.cpp','current=estimator.associationFrame\(currentId\)'),
    @('frozen feature comparison avoids exporting a second copy','src/frontend/Frontend.cpp','estimator.window\(\).frames\(\).measurements\(id\)'),
    @('existing identity reaches matcher before ownership selection','src/frontend/FrontendAssociation.cpp','guidance.queryPriorities=&queryCache.queryPriorities'),
    @('assignment visits selected frame indexes only','src/frontend/FrontendAssociation.cpp','indexFrame\(currentId\);for\(constauto&source:plan.sources\)indexFrame\(source.first\)'),
    @('healthy admission replenishes spatial gaps','src/frontend/FrontendAssociation.cpp','load\[cell\]>=\(cellFeatures\[cell\]\+1\)/2'),
    @('new texture cannot alone certify identity support','src/frontend/FrontendAssociation.cpp','existingCounts\[c\]<std::min<size_t>\(10,options.minimumStageMatchesPerCamera\)'),
    @('seed continuity diagnostics distinguish filtered extensions','src/frontend/Frontend.cpp','committedSeedExtensions.size\(\)'),
    @('rotation dominance avoids an unused relative solve','src/frontend/Frontend.cpp','if\(!rotationDominant\)relative=pose_solver::RelativePoseSolver'),
    @('matcher uses virtual per-camera models','src/frontend/FrontendAssociation.cpp','guidance.queryCamera=source.geometry\(q\);guidance.trainCamera=current->geometry\(t\)'),
    @('frontend does not call protected camera mask helper','src/frontend/FrontendAssociation.cpp','!\bisMasked\s*\('),
    @('mask check uses public camera interface','src/frontend/FrontendAssociation.cpp','return!camera.hasMask\(\)\|\|camera.mask\(\).at<uchar>'),
    @('bearing validates pixel before back-projection','src/frontend/FrontendAssociation.cpp','(?:!camera\|\|!usablePixel\(\*camera,pixel\)\|\|\(!frame.getBackProjection)'),
    @('projected map points use same public validation','src/frontend/FrontendAssociation.cpp','(?:!usablePixel\(\*current->geometry\(t\),projected\))'),
    @('guided ORB handles actual rows','src/frontend/FrontendAssociation.cpp','matcher.matchGuided\(source.descriptors\(q\),current->descriptors\(t\)'),
    @('world camera predictions use live calibration','src/frontend/FrontendAssociation.cpp','value.T_WC=pose\*estimator.extrinsics\(StateId\(frame.id\(\)\),c\)'),
    @('absorbed appearance cannot become a new anchor','src/frontend/FrontendAssociation.cpp','value.queryEnabled\[k\]=!consumed\|\|observed'),
    @('landmark camera epoch uniqueness','src/frontend/FrontendAssociation.cpp','occupied.count\(\{id,key.frameId,key.cameraIndex\}\)'),
    @('new anchors use monotonic managed IDs','src/frontend/FrontendAssociation.cpp','estimator.nextLandmarkId\(\).value\(\)'),
    @('anchor insertion does not duplicate observation','src/frontend/FrontendAssociation.cpp','plan.batch.landmarks.push_back\(\{id,candidate.first,std::nullopt,noiseAt\(candidate.first\)\}\)'),
    @('observation noise scales with actual octave','src/frontend/FrontendAssociation.cpp','options.pixelSigma\*std::pow\(options.pyramidScale'),
    @('range point lies on original anchor ray','src/frontend/FrontendAssociation.cpp','point=origin\+depth\*direction'),
    @('full sphere positive half-ray check','src/frontend/FrontendAssociation.cpp','delta.dot\(ray\)<=0.0'),
    @('temporal range requires metric motion','src/frontend/FrontendAssociation.cpp','!allowTemporal&&key.frameId!=anchorKey.frameId'),
    @('unverified current view cannot be promoted through historical stereo','src/frontend/FrontendAssociation.cpp','if\(!allowTemporal&&std::any_of\(all.begin\(\),all.end\(\)'),
    @('all retained seed views checked','src/frontend/FrontendAssociation.cpp','for\(constauto&view:all\)'),
    @('no unit baseline temporal triangulation','src/frontend/Frontend.cpp','decision.T_WB,accepted&&\(decision.metric\|\|inertialMotion\),keyframe\|\|!accepted\)'),
    @('metric IMU requires a numerically eligible factor','src/frontend/Frontend.cpp','factor.kind==backend::FactorKind::Imu&&factor.participates\(\)'),
    @('non-keyframe seed preservation regression invoked','src/frontend/tests/FrontendBackendRegression.cpp','nonKeyframeRetainsSeedsWithoutRho\(\);'),
    @('seed capacity and expiry regression invoked','src/frontend/tests/FrontendBackendRegression.cpp','seedAdmissionIsBoundedAndExpiresAtomically\(\);'),
    @('seed reference safety regression invoked','src/frontend/tests/FrontendBackendRegression.cpp','seedReferenceUpdateCannotRewriteOptimizedGeometry\(\);'),
    @('promotion checks live outgoing support after seed reference selection','src/frontend/FrontendAssociation.cpp','estimator.window\(\).promotionWouldBeMarginalized\(anchorKey,all\)'),
    @('seed capacity protects existing IDs before new admissions','src/frontend/FrontendAssociation.cpp','plan.trackingSeeds>=options.maxTrackingSeeds\?size_t\(0\)'),
    @('expired source IDs excluded from tentative assignment','src/frontend/FrontendAssociation.cpp','if\(retiredSeedIds.count\(observation.landmark\)\)continue;'),
    @('new tracks are spatially balanced after managed extensions','src/frontend/FrontendAssociation.cpp','usingCellEntry=std::tuple<size_t,float,size_t>;'),
    @('promotion deferral has a distinct support reason','src/frontend/Frontend.cpp','plan.deferredEvictionSupport'),
    @('retained anchor with outgoing support is regression covered','src/frontend/tests/FrontendBackendRegression.cpp','plan.deferredEvictionSupport==1'),
    @('metric IMU edge must touch the current state','src/frontend/Frontend.cpp','std::find\(factor.epochs.begin\(\),factor.epochs.end\(\),id\)!=factor.epochs.end\(\)'),
    @('keyframe uses rotation compensated full-sphere parallax','src/frontend/FrontendAssociation.cpp','std::atan2\(rotated.cross\(b\).norm\(\),rotated.dot\(b\)\)'),
    @('low motion needs independent previous tracks','src/frontend/FrontendAssociation.cpp','result.previousSharedTracks>=minimum&&result.previousTranslation<settings.keyframe_low_motion_translation'),
    @('slow accumulated motion can still create a keyframe','src/frontend/FrontendAssociation.cpp','result.translation>=settings.keyframe_translation\|\|result.rotation>=settings.keyframe_rotation'),
    @('static mature anchors survive penultimate eviction regression','src/frontend/tests/FrontendBackendRegression.cpp','stationaryKeyframesPreserveMatureAnchors\(\);'),
    @('one frontend commit after consensus','src/frontend/Frontend.cpp','estimator.applyFrontendBatch\(id,plan.batch,seed,health,keyframe,error\)'),
    @('no legacy unknown propagation implementation','src/frontend/Frontend.cpp','!cgmli::ceres::ImuError'),
    @('IMU propagation has a concrete default','src/frontend/Frontend.cpp','imuPropagator_\?imuPropagator_:ImuPropagator\(factor::propagateImu\)'),
    @('IMU propagation forwards covariance and Jacobian','src/frontend/Frontend.cpp','integrate\(measurements,parameters,propagated,sb,start,end,covariance\?&cov:nullptr,jacobian\?&jac:nullptr\)'),
    @('old loops are candidates only','src/frontend/Frontend.cpp','lastReport_.localLoopCandidates.push_back'),
    @('local database purged of evicted epochs','src/frontend/Frontend.cpp','dBow_->database.clear\(\);dBow_->poseIds.clear\(\)'),
    @('backend newest guard','src/backend/FixedLagBackend.cpp','window_.order_.back\(\)!=id'),
    @('backend forbids published batch','src/backend/FixedLagBackend.cpp','if\(state.id==id\)'),
    @('backend disallows historical removal','src/backend/FixedLagBackend.cpp','!batch.removedObservations.empty\(\)'),
    @('backend retirement forbids optimized seeds','src/backend/FixedLagBackend.cpp','point->initialized\|\|window_.graph\(\).blocks.count'),
    @('backend retirement protects bound and absorbed observations','src/backend/FixedLagBackend.cpp','observation->use!=ObservationUse::TrackingOnly\|\|observation->factorId\|\|window_.landmarks\(\).consumed\(key\)'),
    @('backend current observation guard','src/backend/FixedLagBackend.cpp','observation.key.frameId!=id.value\(\)'),
    @('backend preserves lidar health','src/backend/FixedLagBackend.cpp','constautolidar=staged.epochs\(\).at\(id\).lidar'),
    @('backend validates complete candidate','src/backend/FixedLagBackend.cpp','if\(!staged.valid\(error\)\)returnfalse;'),
    @('frontend batch activates before full validation','src/backend/FixedLagBackend.cpp','staged.deferPendingActivation_=false;staged.activatePending\(\);constautovalidationStarted='),
    @('admission uses backend landmark and observation capacities','src/frontend/FrontendAssociation.cpp','remainingObservationCapacity\(\)'),
    @('new-track limit is bounded by available landmarks','src/frontend/FrontendAssociation.cpp','plan.admittedTrackLimit=std::min\(options.maxNewLandmarks,landmarkSlots\)'),
    @('new anchors reserve at least two observation slots','src/frontend/FrontendAssociation.cpp','observationSlots<2'),
    @('anchor consumes an observation slot','src/frontend/FrontendAssociation.cpp','plan.batch.landmarks.push_back[^;]+;--observationSlots'),
    @('finite points are evaluated once without observation copies','src/frontend/FrontendAssociation.cpp','window\(\).landmarkPoint\(item.first,world\)'),
    @('triangulation caches world rays by measurement key','src/frontend/FrontendAssociation.cpp','std::map<KeypointIdentifier,WorldRay>worldRays'),
    @('all cached seed rays still checked for every hypothesis','src/frontend/FrontendAssociation.cpp','for\(constauto\*view:retainedRays\)'),
    @('map representatives retain landmark identities','src/frontend/FrontendAssociation.cpp','guidance.queryIdentities=&identities'),
    @('staged map claims exclude later train reuse','src/frontend/FrontendAssociation.cpp','if\(mapClaimed\[t\]\[k\]\)guidance.trainEnabled\[k\]=0'),
    @('staged selection retains coverage and motion fallback','src/frontend/FrontendAssociation.cpp','if\(rotation>0.10\)'),
    @('coverage fallback is target-camera local','src/frontend/FrontendAssociation.cpp','if\(weak\[jobs\[i\].t\]\)broad.push_back\(i\)'),
    @('expansion prioritizes history and newest camera handoffs','src/frontend/FrontendAssociation.cpp','weak\[job.t\]&&\(job.q==job.t\|\|StateId\(job.source->id\(\)\)==latestSource\)'),
    @('map reliability is target-camera local','src/frontend/FrontendAssociation.cpp','if\(mapReady\[t\]\)'),
    @('partial camera coverage has regression source','src/frontend/tests/FrontendBackendRegression.cpp','local.seedJobs<complete.seedJobs'),
    @('failed staged consensus retries full matching before commit','src/frontend/Frontend.cpp','if\(!accepted&&plan.stagedUsed\)'),
    @('pose retry reconstructs an isolated proposal','src/frontend/Frontend.cpp','options.stagedMatching=false;constautoretryStarted='),
    @('frontend returns committed status separately from tracking','src/frontend/Frontend.hpp','boolcommitted=false,trackingAccepted=false,metricPose=false;')
)
foreach($entry in $contracts) {
    $pattern=$entry[2]; $negative=$pattern.StartsWith('!'); if($negative) { $pattern=$pattern.Substring(1) }
    # Literal C++ negations must be escaped in contract patterns.
    if($entry[0] -in @('temporal range requires metric motion','backend disallows historical removal')) { $negative=$false; $pattern=$entry[2] }
    $passed=$compact[$entry[1]] -match $pattern; if($negative) { $passed=!$passed }
    Check $entry[0] $passed
}
foreach($name in @('exportedRingRigKeepsStereoGate','proposalOwnershipAndRange','frontendCallOrderAndFrozenIndices','capacityPreservesExistingObservations','staleSeedRetirementIsAtomic','immutableRayCacheSurvivesCopies','stagedMapMatchingAndCoverageFallback','missingOrFailedPropagation')) {
    Check ('regression source: '+$name) ($code['src/frontend/tests/FrontendBackendRegression.cpp'] -match ('void\s+'+$name+'\('))
}
Check 'backend export applies common rig reference transform' ($compact['src/backend/SlidingWindowData.cpp'].Contains('result->transformCameraSystemReferenceFrame(T_new_old)') -and
    !$compact['src/backend/SlidingWindowData.cpp'].Contains('result->setExtrinsics('))
Check 'frontend rejects invalid overlap graph before proposing matches' ($compact['src/frontend/FrontendAssociation.cpp'].Contains('if(current->numFrames()>1&&!current->cameraSystem().overlapComputationValid())throwstd::logic_error('))
Check 'same-epoch mask gate never silently falls back to all pairs' ($compact['src/frontend/FrontendAssociation.cpp'].Contains('if(stereo){if(!current->hasOverlap(t,q)||!current->hasOverlap(q,t))') -and
    !$compact['src/frontend/FrontendAssociation.cpp'].Contains('if(stereo&&current->cameraSystem().overlapComputationValid())'))
Check 'common rig transform updates primary virtual and rectified cameras' ($compact['src/cameras/NCameraSystem.cpp'].Contains('for(auto&pose:cameras)pose=transform(pose)') -and
    $compact['src/cameras/NCameraSystem.cpp'].Contains('for(auto&pose:virtualCameras)pose=transform(pose)') -and
    $compact['src/cameras/NCameraSystem.cpp'].Contains('for(auto&entry:rectifiedCameras)entry.second=transform(entry.second)'))
Check 'backend whole frontend batch rollback regression' ($code['src/backend/tests/ManagedWindowRegression.cpp'] -match 'void\s+frontendBatchAtomicity\(')
Check 'caller sync happens after successful commit' ($compact['src/frontend/Frontend.cpp'].IndexOf('estimator.synchronizeFrame(id,*framesInOut)') -gt
    $compact['src/frontend/Frontend.cpp'].IndexOf('estimator.applyFrontendBatch('))
Check 'healthy promotion budget preserves recovery bypass' (([IO.File]::ReadAllText((Join-Path $root 'src/frontend/Frontend.cpp'))) -match 'trackingLost_ \|\| !accepted \|\| !decision.metric \|\| plan.poseFallback')
$failures=@($checks|Where-Object { !$_.passed })
$report=[ordered]@{checkedAtUtc=[DateTime]::UtcNow.ToString('o');scope='Frontend to managed ViSlamBackend source/API/project inspection only; no C++ compilation or execution';
    passed=$checks.Count-$failures.Count;total=$checks.Count;checks=$checks}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'frontend_backend_static_report.json'),($report|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
Write-Output ('Frontend/backend static checks: {0}/{1}; no compilation or runtime tests' -f ($checks.Count-$failures.Count),$checks.Count)
foreach($failure in $failures) { Write-Output ('FAIL: '+$failure.name) }
if($failures.Count) { exit 1 }
