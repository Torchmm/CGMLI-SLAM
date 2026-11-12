param()
$ErrorActionPreference='Stop'
$backendRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$root=[IO.Path]::GetFullPath((Join-Path $backendRoot '..\..'))
$checks=[Collections.Generic.List[object]]::new()
function Check([string]$name,[bool]$ok) { $checks.Add([pscustomobject]@{name=$name;passed=$ok}) }
function Read([string]$path) { [IO.File]::ReadAllText($path) }
$strip='(?s)/\*.*?\*/|//[^\r\n]*|"(?:\\.|[^"\\])*"|''(?:\\.|[^''\\])*'''
$production=@(Get-ChildItem -LiteralPath $backendRoot -File | Where-Object Extension -in '.cpp','.hpp')
$tests=@(Get-ChildItem -LiteralPath (Join-Path $backendRoot 'tests') -Filter '*.cpp' -File)
[xml]$project=Read (Join-Path $root 'CGMLI-SLAM\CGMLI-SLAM.vcxproj')
[xml]$filters=Read (Join-Path $root 'CGMLI-SLAM\CGMLI-SLAM.vcxproj.filters')
$ns=[Xml.XmlNamespaceManager]::new($project.NameTable); $ns.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
$fn=[Xml.XmlNamespaceManager]::new($filters.NameTable); $fn.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
Check 'application project XML parses' ($null -ne $project.DocumentElement)
Check 'filters XML parses' ($null -ne $filters.DocumentElement)
$raw=@{}; $code=@{}; $compact=@{}
foreach($file in @($production)+@($tests)) {
    $source=Read $file.FullName
    $clean=[regex]::Replace($source,$strip,' ')
    $raw[$file.Name]=$source; $code[$file.Name]=$clean; $compact[$file.Name]=[regex]::Replace($clean,'\s+','')
    $stack=[Collections.Generic.Stack[char]]::new(); $balanced=$true
    $pairs=@{')'='(';']'='[';'}'='{'}
    foreach($character in $clean.ToCharArray()) {
        if($character -in '(','[','{') { $stack.Push($character) }
        elseif($pairs.ContainsKey([string]$character)) {
            if(!$stack.Count -or $stack.Pop() -ne $pairs[[string]$character]) { $balanced=$false; break }
        }
    }
    Check ($file.Name+': delimiters') ($balanced -and !$stack.Count)
    Check ($file.Name+': no implementation stubs') ($clean -notmatch '\b(TODO|FIXME|NotImplemented)\b')
    foreach($include in [regex]::Matches($source,'(?m)^\s*#include\s*"([^"]+)"')) {
        Check ($file.Name+': include '+$include.Groups[1].Value) (Test-Path -LiteralPath (Join-Path $file.DirectoryName $include.Groups[1].Value))
    }
    foreach($include in [regex]::Matches($source,'(?m)^\s*#include\s*<((?:backend|common|visual|cameras|kinematics|utils)/[^>]+)>')) {
        Check ($file.Name+': project include '+$include.Groups[1].Value) (Test-Path -LiteralPath (Join-Path (Join-Path $root 'src') $include.Groups[1].Value))
    }
}
foreach($file in $production) {
    $relative='..\src\backend\'+$file.Name
    $tag=if($file.Extension -eq '.cpp'){'ClCompile'}else{'ClInclude'}
    $entries=@($project.SelectNodes('//m:'+$tag,$ns) | Where-Object { $_.GetAttribute('Include') -eq $relative })
    $filterEntries=@($filters.SelectNodes('//m:'+$tag,$fn) | Where-Object { $_.GetAttribute('Include') -eq $relative })
    Check ($file.Name+': project registration once') ($entries.Count -eq 1)
    Check ($file.Name+': filter registration once') ($filterEntries.Count -eq 1)
}
foreach($file in $tests) {
    $entries=@($project.SelectNodes('//m:ClCompile',$ns) | Where-Object { $_.GetAttribute('Include') -eq '..\src\backend\tests\'+$file.Name })
    Check ($file.Name+': standalone test excluded from application') ($entries.Count -eq 0)
}
foreach($entry in $project.SelectNodes('//*[@Include]')) {
    $path=$entry.GetAttribute('Include')
    if($path.StartsWith('..\src\backend\')) {
        Check ('registered backend path exists: '+$path) (Test-Path -LiteralPath (Join-Path (Join-Path $root 'CGMLI-SLAM') $path))
    }
}
$definitions=($production | Where-Object Extension -eq '.cpp' | ForEach-Object { $code[$_.Name] }) -join "`n"
$methods=@{
    'SlidingWindowManager'=@('append:2','attachFrame:1','attachImu:1','addLandmark:1','initializeLandmark:1','addObservation:1',
        'removeObservation:1','retireLandmark:1','applyVisualUpdates:1','valid:1','reanchorOutgoing:1','marginalize:1',
        'pointWorld:1','landmark:1','getLandmark:1','exportFrame:1','submitFactor:1');
    'FrameStore'=@('attach:1','attachImu:1','copyMeasurements:1','cameraPose:1','observation:1','contains:1');
    'LandmarkStore'=@('find:2','insert:2','bind:1','factorRemoved:1','eraseFrame:1','eraseLandmark:1','valid:1');
    'ViSlamBackend'=@('initializeRig:1','addStates:1','addImuFactor:1','addLandmark:1','getLandmark:1','getLandmarks:1','multiFrame:1','optimize:1');
    'FixedLagBackend'=@('append:1','step:1','makeSnapshot:1','applyFrontendUpdate:1');
    'Marginalizer'=@('compute:1','linearize:1','psdSolve:1');
    'LeftSE3Manifold'=@('Plus:1','Minus:1','PlusJacobian:1','MinusJacobian:1')
}
foreach($class in $methods.Keys) { foreach($method in $methods[$class]) {
    $parts=$method.Split(':')
    # Require a definition body and exclude assignment/call statements. Static helper
    # calls such as auto view=FrameStore::copyMeasurements(...) are not definitions.
    $definitionPattern='(?m)^[^\r\n;{}=]*\b'+$class+'::'+$parts[0]+'\s*\([^;{}]*\)\s*(?:const\s*)?\{'
    $count=[regex]::Matches($definitions,$definitionPattern).Count
    Check ($class+'::'+$parts[0]+': definition count') ($count -eq [int]$parts[1])
} }
$contracts=@(
    @('single facade backend','ViSlamBackend.hpp','backend::FixedLagBackendbackend_;'),
    @('no old graph forwarding','ViSlamBackend.hpp','!\b(realtimeGraph_|fullGraph_|multiFrames_)\b'),
    @('window owns frame store','SlidingWindowManager.hpp','FrameStoreframes_;'),
    @('window owns landmark store','SlidingWindowManager.hpp','LandmarkStorelandmarks_;'),
    @('frame store mutation is manager-only','FrameStore.hpp','friendclassSlidingWindowManager;'),
    @('landmark store mutation is manager-only','LandmarkStore.hpp','friendclassSlidingWindowManager;'),
    @('explicit feature reconstruction','FrameStore.cpp','resetKeypoints\(c,source.keypoints\(c\)\)'),
    @('descriptor deep copy','FrameStore.cpp','source.descriptors\(c\).clone\(\)'),
    @('rig identity persists without visual frames','FrameStore.hpp','rigBaselines_;'),
    @('pixel covariance transported to sphere','FrameStore.cpp','candidate.sqrtInformation=llt.matrixL\(\).solve'),
    @('epoch td reference frozen on creation','SlidingWindowManager.cpp','referenceCameraTimeOffset='),
    @('late observation uses epoch td reference','SlidingWindowData.cpp','frames_.find\(StateId\(key.frameId\)\)->referenceCameraTimeOffset'),
    @('second newest selects penultimate','SlidingWindowManager.cpp','order_\[size\(\)-2\]'),
    @('late epoch rejection','SlidingWindowManager.cpp','id.value\(\)<=largestEpochId_'),
    @('raw landmark block bypass prevented','SlidingWindowManager.cpp','if\(p.key.kind==BlockKind::Landmark\)'),
    @('raw visual factor bypass prevented','SlidingWindowManager.cpp','f.kind==FactorKind::Visual&&!managedObservation'),
    @('unique camera/epoch association','LandmarkStore.cpp','inCamera\(observation.landmark,StateId\(observation.key.frameId\),observation.key.cameraIndex\)'),
    @('contiguous sorted observation reverse indexes','LandmarkStore.hpp','usingObservationKeys=std::vector<KeypointIdentifier>'),
    @('camera index uses binary range lookup','LandmarkStore.cpp','std::lower_bound\(keys.begin\(\),keys.end\(\),KeypointIdentifier\(frame.value\(\),camera,0\)\)'),
    @('bulk frame erasure removes its vector bucket once','LandmarkStore.cpp','byFrame_.erase\(id\);for\(constauto&key:keys\)eraseObservation'),
    @('tombstone erase is restricted to one frame range','LandmarkStore.cpp','absorbedKeys_.erase\(first,last\)'),
    @('consumed key rejects reinsertion','LandmarkStore.cpp','consumed\(observation.key\)'),
    @('terminal factors synchronize observations','SlidingWindowManager.cpp','landmarks_.factorRemoved\(it->first'),
    @('production defaults retire anchored points','BackendTypes.hpp','boolautoReanchorLandmarks=false;'),
    @('prior guards automatic reanchor','LandmarkReanchoring.cpp','priorKeys.count\(rho\)'),
    @('time-velocity reanchor safely excluded','LandmarkReanchoring.cpp','observation->bearingVelocity.norm\(\)>1e-12'),
    @('reanchor world point equivalence','LandmarkReanchoring.cpp','\(reconstructed-point\).norm\(\)>'),
    @('old anchor measurement retained','LandmarkReanchoring.cpp','store.bind\(original.anchor,id\)'),
    @('automatic chart changes require information equivalence','LandmarkReanchoring.cpp','if\(!audit.equivalent\)'),
    @('scalar chart audits physical point perturbations','LandmarkReanchoring.cpp','report.oldPointJacobian\*report.oldFromNew-report.newPointJacobian'),
    @('information transported in common tangent chart','LandmarkReanchoring.cpp','report.oldFromNew.transpose\(\)\*report.oldInformation\*report.oldFromNew'),
    @('gradient transported with the same chart','LandmarkReanchoring.cpp','report.oldFromNew.transpose\(\)\*report.oldGradient'),
    @('reanchor information regression invoked','ManagedWindowRegression.cpp','reanchorInformationAudit\(\);'),
    @('unsafe fixed-bearing chart regression invoked','ManagedWindowRegression.cpp','reanchorFixedBearingChangeIsRejected\(\);'),
    @('coupled prior transport reference regression invoked','ManagedWindowRegression.cpp','reanchorFullCoordinateTransportReference\(\);'),
    @('all domain stores staged together','SlidingWindowManager.cpp','SlidingWindowManagerstaged\(\*this,std::move\(candidate\),std::move\(candidateLandmarks\),std::move\(candidateFrames\)\)'),
    @('staged frame records adopted without duplicate shell copy','SlidingWindowManager.cpp','frames_\(std::move\(frames\)\)'),
    @('detached association frame omits images','SlidingWindowData.cpp','returncopyFrame\(id,T_SB,false\)'),
    @('observation validation resolves layout once per frame','SlidingWindowData.cpp','if\(item.first.frameId!=checkedFrame\)'),
    @('marginalization preserves event-only history','SlidingWindowManager.cpp','staged.recordEvent\(std::move\(event\)\)'),
    @('adopted window preserves ID high-water marks','SlidingWindowManager.cpp','nextFactorId_\(source.nextFactorId_\)'),
    @('gate parameter pointers are prepared once per solve','BackendOptimizer.cpp','gateEntries.push_back\(std::move\(entry\)\)'),
    @('gate buffers are allocated before worker launch','BackendOptimizer.cpp','buffers.emplace_back\(residualSize\)'),
    @('parallel gate has deterministic rollback regression','ManagedWindowRegression.cpp','parallelNisGateAgreementAndRollback\(\);'),
    @('ten-frame capacity and temporary slot regression is invoked','ManagedWindowRegression.cpp','tenFrameWindowUsesOneInsertionSlot\(\);'),
    @('gate decisions apply only after worker join','BackendOptimizer.cpp','for\(auto&worker:workers\)worker.join\(\);constautostatisticsStarted='),
    @('Schur timings propagate into window receipt','SlidingWindowManager.cpp','out.schurMs=result.schurMs'),
    @('cross-store validation before commit','SlidingWindowManager.cpp','if\(!staged.valid\(out.message\)\)returnout;'),
    @('seed tracks retire by anchor metadata','SlidingWindowManager.cpp','item.second.anchor.frameId!=out.removed.value\(\)'),
    @('batch uses one staging copy','SlidingWindowData.cpp','applyVisualUpdates\([^\{]+\)\{autostaged=\*this;'),
    @('outer frontend checkpoint uses non-copying batch helper','FixedLagBackend.cpp','staged.applyVisualUpdatesImpl\(batch,error\)'),
    @('batch defers repeated global pending activation','SlidingWindowData.cpp','PendingGuardguard\(deferPendingActivation_\)'),
    @('seed reference rejects an existing optimized coordinate','SlidingWindowData.cpp','point->second.initialized\|\|graph_.blocks.count\(rho\)\|\|priorCoupled'),
    @('seed reference cannot reactivate an absorbed observation','SlidingWindowData.cpp','view->use!=ObservationUse::TrackingOnly\|\|view->factorId\|\|landmarks_.consumed\(anchor\)'),
    @('seed anchor metadata uses the actual measured ray','SlidingWindowData.cpp','point->second.anchor=anchor;point->second.anchorBearing=view->bearing;'),
    @('new views precede seed selection and promotion','SlidingWindowData.cpp','for\(constauto&point:batch.seedAnchors\)'),
    @('seed eviction regression covers SECOND_NEW','ManagedWindowRegression.cpp','second.marginalization->movedSeeds==1'),
    @('health changes honor batch activation deferral','SlidingWindowManager.cpp','if\(!deferPendingActivation_\)\{collectTerminalFactors\(\);activatePending\(\);\}'),
    @('budget-limited acceptance has a separate status','BackendOptimizer.cpp','out.accepted=true;out.termination=termination;'),
    @('limited result requires normal application quality checks','BackendOptimizer.cpp','if\(validator&&!validator\(graph,out.message\)\)'),
    @('limited result cannot publish an unconstrained graph','BackendOptimizer.cpp','if\(!out.passes\|\|!constrained\)returnrollback\('),
    @('accepted statistics use surviving observations','BackendOptimizer.cpp','out.postSolveResiduals=survivingResiduals;'),
    @('quality failure still rolls back at solve limits','ManagedWindowRegression.cpp','rejected.termination==SolveTermination::Failure&&failed.blocks.at\(x\).value\[0\]==0'),
    @('limited solve prior-cleanup regression is invoked','ManagedWindowRegression.cpp','budgetLimitedSolvePreservesCleanPrior\(\);'),
    @('only completed Ceres time is charged before a refit','BackendOptimizer.cpp','constdoubletimeRemaining=timeLimit-out.ceresMs\*1e-3;'),
    @('minimum-iteration callback shares previous Ceres calls','BackendOptimizer.cpp','returnout.ceresMs\*1e-3\+std::chrono::duration<double>'),
    @('Solver stopwatch spans online Schur and Ceres fallback','BackendOptimizer.cpp','solveStarted=std::chrono::steady_clock::now\(\);boolnativeSchur=false;[\s\S]*?if\(!nativeSchur\)\{[\s\S]*?::ceres::Solve\('),
    @('Ceres budget reach diagnostic is updated after Solve','BackendOptimizer.cpp','out.ceresTimeBudgetReached=out.ceresMs\*1e-3>=timeLimit;'),
    @('slow gate does not consume Ceres time regression','ManagedWindowRegression.cpp','afterSlowGate.accepted&&afterSlowGate.passes==1'),
    @('Schur eligibility checks the historical prior','BackendOptimizer.cpp','\(!graph.prior\|\|independent\(graph.prior->keys\(\)\)'),
    @('Schur eligibility checks every active factor','BackendOptimizer.cpp','schur=schur&&independent\(item.second.keys\)'),
    @('Schur ordering separates independent landmarks','BackendOptimizer.cpp','key.kind==BlockKind::Landmark\?0:1'),
    @('actual Ceres solver timing is exported','BackendOptimizer.cpp','summary.linear_solver_time_in_seconds'),
    @('bounded LM uses quadratic projected line search','BackendOptimizer.cpp','solver.line_search_interpolation_type=::ceres::QUADRATIC;'),
    @('actual interpolation is exported in solve receipt','BackendOptimizer.cpp','out.lineSearchInterpolation=::ceres::LineSearchInterpolationTypeToString\(solver.line_search_interpolation_type\)'),
    @('Ceres Jacobian evaluations accumulate across refits','BackendOptimizer.cpp','out.jacobianEvaluations\+=std::max\(0,summary.num_jacobian_evaluations\)'),
    @('Ceres residual evaluations accumulate across refits','BackendOptimizer.cpp','out.residualEvaluations\+=std::max\(0,summary.num_residual_evaluations\)'),
    @('line search gradient cost is exported','BackendOptimizer.cpp','out.lineSearchGradientMs\+=std::max\(0.0,summary.line_search_gradient_evaluation_time_in_seconds\)\*1000'),
    @('positive inverse-range bound is retained','SlidingWindowData.cpp','block.lower=Eigen::VectorXd::Constant\(1,options_.minInverseDepth\)'),
    @('eviction prepares native redundancy rules before Schur','SlidingWindowManager.cpp','prepareEviction\(out,candidate,candidateLandmarks,candidateFrames,discarded,reanchorReasons,out.message\)'),
    @('merged IMU factors are erased before marginalization','SlidingWindowManager.cpp','graph.factors.erase\(a\);graph.factors.erase\(b\)'),
    @('old IMU bridge prevents replay duplication','SlidingWindowManager.cpp','(?:!priorKeys.count\(BlockKey::speedBias\(out.removed\)\))'),
    @('external state constraints retain Schur bridge','SlidingWindowManager.cpp','incident.size\(\)==2&&!externalStateConstraint'),
    @('replay merge receives the storage budget','SlidingWindowManager.cpp','options_.maxImuSamplesPerEpoch,merged'),
    @('reset skips historical landmark coordinates','SlidingWindowManager.cpp','if\(priorKeys.count\(rho\)\)'),
    @('reset prefers retained keyframe references','SlidingWindowManager.cpp','if\(ka!=kb\)returnka>kb;'),
    @('reset keeps original whitened residual gate','SlidingWindowManager.cpp','residual.squaredNorm\(\)>f.nisThreshold'),
    @('seed reference keeps surviving measurement','SlidingWindowManager.cpp','landmark.anchor=surviving.front\(\)'),
    @('single-ray mature track keeps its ID as a seed','SlidingWindowManager.cpp','graph.blocks.erase\(rho\);landmark.initialized=false;'),
    @('oldest model reset enabled by default','BackendTypes.hpp','boolresetOldestLandmarks=true;'),
    @('strict exact reanchor mode keeps priority','SlidingWindowManager.cpp','options_.resetOldestLandmarks\|\|options_.autoReanchorLandmarks'),
    @('oldest ID retention regression invoked','ManagedWindowRegression.cpp','oldestFrameRetainsPriorFreeTracks\(\);'),
    @('bias refresh replaces immutable active IMU costs','FixedLagBackend.cpp','f.cost=factor::makeBodyImuFactor\(f.imuReplay->samples'),
    @('bias refresh occurs before numeric solving','FixedLagBackend.cpp','out.solve=optimizer_.optimize'),
    @('IMU physical diagnostics retain interval endpoints','FixedLagBackend.cpp','d.physical.segment<3>\(6\).norm\(\)'),
    @('pure residual path skips visual derivative matrices','WindowVisualFactor.cpp','if\(!derivatives\)returntrue;'),
    @('residual-only numerical agreement regression','ManagedWindowRegression.cpp','\(residualOnly-ra\).norm\(\)'),
    @('unaffected prior is reused without recentering','Marginalizer.cpp','out.consumed.empty\(\)&&!priorTouches'),
    @('diagonal PSD path uses exact structural check','Marginalizer.cpp','H.isDiagonal\(0.0\)'),
    @('separator product reuses partition storage','Marginalizer.cpp','subtract\(Hmr,solved,Hrr,br\)'),
    @('prior reference and rank fast path regression invoked','ManagedWindowRegression.cpp','untouchedPriorAndDiagonalKernel\(\);'),
    @('repeated redundancy lifecycle regression invoked','ManagedWindowRegression.cpp','redundantFrameKeepsTracksSparse\(\);'),
    @('refit rollback and coupled-prior regression is called','ManagedWindowRegression.cpp','refitRecoveryAndSchurSafety\(\);'),
    @('analytic visual derivative is the default','BackendTypes.hpp','boolanalyticVisualJacobians=true;'),
    @('visual derivatives retain a numerical reference switch','WindowVisualFactor.cpp','if\(!analytic_\)returnevaluateNumerical'),
    @('sphere analytic derivative includes basis transport','WindowVisualFactor.cpp','measurement_\*dR.transpose\(\)\*logarithm'),
    @('stereo kernel cancels common poses before evaluation','WindowVisualFactor.cpp','pc=Rcb_.transpose\(\)\*\(local-tcb_\)'),
    @('fixed observer transport reuses its prepared measurement','WindowVisualFactor.cpp','P=cachedObserver\?observerTime->projection:measurement_;if\(movingObservation_&&!cachedObserver\)'),
    @('moving anchor cache promotes the same point to time derivatives','WindowVisualFactor.cpp','evaluateJacobians&&!value.derivativeReady'),
    @('refits remove rejected residuals from the same problem','BackendOptimizer.cpp','problem.RemoveResidualBlock\(it->second\)'),
    @('refits remove now-unused Ceres parameter buffers','BackendOptimizer.cpp','if\(!stillUsed.count\(key\)\)problem.RemoveParameterBlock'),
    @('rollback releases Ceres views before restoring graph','BackendOptimizer.cpp','workspace.reset\(\);checkpoint.restore\(\);'),
    @('numeric rollback restores original buffers in place','BackendOptimizer.cpp','p.parameter->value.data\(\)'),
    @('OLD preserves visual information before scalar retirement','SlidingWindowManager.cpp','policy==MarginalizationPolicy::MARGIN_OLD&&options_.preserveOldestVisualInformation'),
    @('OLD consumes implicit anchor measurements too','SlidingWindowManager.cpp','candidateLandmarks.absorbedKeys_.insert\(view\)'),
    @('Schur identity survival regression invoked','ManagedWindowRegression.cpp','oldestSchurKeepsIdentityWithoutReusingEvidence\(\);'),
    @('scalar range Schur keeps coupled prior fallback','Marginalizer.cpp','nl&&Hll.isDiagonal\(0.0\)'),
    @('Schur rejects disabled measurements','Marginalizer.cpp','!f.participates\(\)'),
    @('Ceres adapter rejects disabled measurements','BackendProblem.hpp','if\(!f.participates\(\)\)'),
    @('Ceres cache updates before parallel costs','BackendProblem.hpp','options.evaluation_callback=&visualCache'),
    @('cached trial derivatives have a reference regression','ManagedWindowRegression.cpp','visualPoseCacheMatchesUncachedTrials\(\);'),
    @('explicit dense Schur solver is available','BackendOptimizer.cpp','::ceres::DENSE_SCHUR'),
    @('OLD ignores consumed incidence before inspecting new rho','SlidingWindowManager.cpp','observation->use==ObservationUse::Absorbed\|\|candidateLandmarks.consumed\(key\)'),
    @('OLD requires an actual live observation factor','SlidingWindowManager.cpp','observation->use!=ObservationUse::LiveFactor'),
    @('OLD requires the bound factor to participate','SlidingWindowManager.cpp','factor==candidate.factors.end\(\)\|\|!factor->second.participates\(\)'),
    @('promotion prognosis excludes consumed measurements','SlidingWindowManager.cpp','key.frameId==outgoing.value\(\)&&!landmarks_.consumed\(key\)'),
    @('promotion records a dedicated initialization epoch','SlidingWindowData.cpp','point->second.lastInitializedFrame=order_.back\(\).value\(\)'),
    @('false incidence regression retains the same live cost','ManagedWindowRegression.cpp','window.graph\(\).factors.at\(retained\).cost==retainedCost'),
    @('constraint gradient audit retains robust first derivative','BackendOptimizer.cpp','rho\[1\]\*g'),
    @('constraint gradient audit uses current manifold tangent','BackendOptimizer.cpp','block.manifold->PlusJacobian'),
    @('gradient balance numerical regression invoked','ManagedWindowRegression.cpp','constraintGradientBalanceIsRobustAndReadOnly\(\);'),
    @('formation distinguishes robust cost and returned quadratic','Marginalizer.cpp','out.robustCost=cost;out.residualSquaredNorm=r.squaredNorm\(\)'),
    @('formation accounts for eliminated gradient energy','Marginalizer.cpp','formation.eliminatedReduction\+=0.5\*bm.dot\(solved.col\(nr\)\)'),
    @('formation accounts for omitted objective constant','Marginalizer.cpp','formation.quadraticConstantOffset=0.5\*lin.residualSquaredNorm'),
    @('profiling stripes avoid one shared cache line','BackendProblem.hpp','structalignas\(64\)Slot'),
    @('stats reuse pre-loss gate residual evaluations','BackendOptimizer.cpp','energies\[i\]=buffer.head\(n\).squaredNorm\(\)'),
    @('prior diagnostics do not apply another robust loss','BackendOptimizer.cpp','s.robustCost=0.5\*s.squaredNorm'),
    @('retirement support recorded before prior absorption','SlidingWindowManager.cpp','survivingLive'),
    @('residual statistics regression invoked','ManagedWindowRegression.cpp','realWindowResidualStatisticsContract\(\);'),
    @('sphere analytic derivative coalesces same-epoch pose','WindowVisualFactor.cpp','if\(observerPose_==0\)'),
    @('analytic versus numeric derivative regression is invoked','ManagedWindowRegression.cpp','analyticSphereDerivativeAgreement\(\);'),
    @('factor profiling is sampled by epoch','BackendOptimizer.cpp','newest&&newest%30==0'),
    @('profiled factors preserve evaluation and request types','ManagedWindowRegression.cpp','factorProfilingPreservesEvaluation\(\);'),
    @('index cardinality proof remains in validation','LandmarkStore.cpp','frameCount!=observations_.size\(\)\|\|landmarkCount!=observations_.size\(\)'),
    @('single-pass visual dependency validation stays complete','SlidingWindowData.cpp','dependencies!=15'),
    @('promotion honors pending batch deferral','SlidingWindowData.cpp','if\(!deferPendingActivation_\)activatePending\(\)'),
    @('coordinate-only map access is available','SlidingWindowManager.hpp','boollandmarkPoint\(LandmarkIdid,Eigen::Vector3d&point\)const'),
    @('landmark retirement is joint Schur','SlidingWindowData.cpp','marginalizer_.compute\(graph_,eliminated\)'),
    @('world point derived from graph rho','SlidingWindowData.cpp','direction/block->second.value\[0\]'),
    @('same epoch pose coalesced','WindowVisualFactor.cpp','if\(!sameEpoch\)mutable_parameter_block_sizes\(\)->push_back\(7\)'),
    @('S2 full sphere angular residual','WindowVisualFactor.cpp','std::atan2\(s,c\)'),
    @('visual derivative lifted with left manifold','WindowVisualFactor.cpp','manifold.MinusJacobian'),
    @('optional clock derivative uses bearing velocity','WindowVisualFactor.cpp','observation_.referenceTimeOffset-td'),
    @('solve plus domain rollback','FixedLagBackend.cpp','window_=std::move\(checkpoint\);'),
    @('snapshot includes landmark values','FixedLagBackend.cpp','snapshot->landmarks.push_back'),
    @('atomic publication','FixedLagBackend.cpp','std::atomic_store_explicit'),
    @('map gates still explicit','FixedLagBackend.cpp','gates.passed\(\)'),
    @('no constant quaternion bounds','BackendTypes.cpp','poseType&&\(lower.size\(\)\|\|upper.size\(\)\)'),
    @('historical prior included once','Marginalizer.cpp','linearize\(graph,out.consumed,true,true'),
    @('no explicit inverse in Schur','Marginalizer.cpp','!\.inverse\s*\(')
)
foreach($item in $contracts) {
    $pattern=$item[2]; $negative=$pattern.StartsWith('!'); if($negative){$pattern=$pattern.Substring(1)}
    $ok=$compact[$item[1]] -match $pattern; if($negative){$ok=!$ok}
    Check ('contract: '+$item[0]) $ok
}
$data=$code['WindowDataTypes.hpp']
$metadata=[regex]::Match($data,'(?s)struct WindowLandmark\s*\{(.*?)\};').Groups[1].Value
Check 'landmark metadata has no second mutable world/depth estimate' ($metadata -notmatch '\b(pointWorld|inverseDepth|T_WB)\b')
foreach($test in @('ownershipAndIndices','factorGeometryAndJacobians','windowPoliciesAndRollback','seedRetirementAndFacade','priorCoupledRetirement')) {
    Check ('regression source: '+$test) ($code['ManagedWindowRegression.cpp'] -match ('void\s+'+$test+'\('))
}
$ceres=[IO.Path]::GetFullPath((Join-Path $root '..\vs2022\ceres-solver2.2.0\include\ceres'))
foreach($api in @(@('problem.h','manifold_ownership'),@('problem.h','apply_loss_function'),@('manifold.h','MinusJacobian'),
    @('types.h','IsSparseLinearAlgebraLibraryTypeAvailable'),@('cost_function.h','parameter_block_sizes'),
    @('types.h','QUADRATIC'),@('types.h','LineSearchInterpolationTypeToString'),
    @('solver.h','line_search_interpolation_type'),@('solver.h','num_jacobian_evaluations'),@('solver.h','num_residual_evaluations'),
    @('solver.h','line_search_cost_evaluation_time_in_seconds'),@('solver.h','line_search_gradient_evaluation_time_in_seconds'),
    @('solver.h','line_search_total_time_in_seconds'))) {
    $path=Join-Path $ceres $api[0]; Check ('installed Ceres API: '+$api[1]) ((Test-Path -LiteralPath $path) -and (Read $path) -match $api[1])
}
Check 'visual pair key includes three bound states and full fixed rig' ($code['WindowVisualFactor.cpp'] -match 'reinterpret_cast<std::uintptr_t>\(a\)' -and $code['WindowVisualFactor.hpp'] -match 'std::array<double,24>')
Check 'pair refresh follows new-point callback' ($code['WindowVisualFactor.cpp'] -match 'for\(auto& entry:pairs_\)' -and $code['WindowVisualFactor.cpp'] -match '!ready_ \|\| newPoint')
Check 'pair residual reuses relative camera geometry' ($code['WindowVisualFactor.cpp'] -match 'pair->cameraRotation\*\(ba/rho\)\+pair->offset')
Check 'visual cached quaternion lift is available' ($code['WindowVisualFactor.cpp'] -match 'qm=\(\*cache\)\[block\]->quaternionLift')
Check 'rho-only requests skip pose derivative chains' ($code['WindowVisualFactor.cpp'] -match 'if\(!pair \|\| poseDerivatives\)')
Check 'visual pair cache has a per-pose-only reference switch' ($code['BackendOptimizer.cpp'] -match 'workspace->visualCache,options_\.cacheVisualPairs')
Check 'visual kernel avoids Windows RPC small macro identifier' ($code['WindowVisualFactor.cpp'] -notmatch '\bsmall\b')
Check 'visual series uses a descriptive nonmacro identifier' ($code['WindowVisualFactor.cpp'] -match 'const bool useSmallAngleSeries=' -and [regex]::Matches($code['WindowVisualFactor.cpp'],'\buseSmallAngleSeries\b').Count -eq 4)
Check 'pair-cache partial derivative regression retained' ((Read (Join-Path $root 'src/backend/tests/ManagedWindowRegression.cpp')) -match 'Cached partial derivatives including rho-only must evaluate')
Check 'NIS gate uses the cached visual adapter when available' ($code['BackendOptimizer.cpp'] -match 'entry.evaluationCost\?entry.evaluationCost:f.cost.get' -and $code['BackendOptimizer.cpp'] -match 'gateVisualCache.PrepareForEvaluation\(false,true\)')
Check 'same-point derivative preparation is separate from geometry readiness' ($code['WindowVisualFactor.cpp'] -match 'evaluateJacobians && !derivativesReady_' -and $code['WindowVisualFactor.cpp'] -match 'derivativesReady_=false')
Check 'pair computation shares state-camera chains' ($code['WindowVisualFactor.cpp'] -match 'p.cameraRotation=p.W\*a.worldRotation')
Check 'retirement candidates include both indexed outgoing views and eliminated ranges' ($code['SlidingWindowManager.cpp'] -match 'retirementCandidates=outgoingTracks' -and $code['SlidingWindowManager.cpp'] -match 'retirementCandidates.insert\(LandmarkId\(key.id\)\)')
$failures=@($checks | Where-Object { !$_.passed })
$report=[ordered]@{checkedAtUtc=[DateTime]::UtcNow.ToString('o');kind='static source/project inspection; no build or C++ runtime tests';
    productionFiles=$production.Count;regressionSources=$tests.Count;passed=$checks.Count-$failures.Count;total=$checks.Count;checks=$checks}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'backend_static_report.json'),($report | ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
Write-Output ('Backend static checks: {0}/{1}; no compilation or runtime tests' -f ($checks.Count-$failures.Count),$checks.Count)
foreach($failure in $failures) { Write-Output ('FAIL: '+$failure.name) }
if($failures.Count) { exit 1 }
