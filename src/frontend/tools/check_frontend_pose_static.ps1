param()
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$checks=[Collections.Generic.List[object]]::new()
function Check([string]$name,[bool]$passed) { $checks.Add([pscustomobject]@{name=$name;passed=$passed}) }
function Read([string]$path) { [IO.File]::ReadAllText((Join-Path $root $path)) }
$strip='(?s)/\*.*?\*/|//[^\r\n]*|"(?:\\.|[^"\\])*"|''(?:\\.|[^''\\])*'''
$paths=@('src/frontend/Frontend.cpp','src/frontend/Frontend.hpp','src/frontend/FrontendPoseAdapter.cpp',
    'src/frontend/FrontendPoseAdapter.hpp','src/frontend/tests/FrontendPoseRegression.cpp',
    'src/visual/pose_solver/RotationPoseSolver.cpp','src/visual/pose_solver/RotationPoseSolver.hpp',
    'src/visual/pose_solver/PoseEstimatorAdapter.hpp','src/visual/pose_solver/AbsolutePoseSolver.cpp',
    'src/visual/pose_solver/RelativePoseSolver.cpp','src/visual/pose_solver/PoseSolverInternal.hpp','src/visual/pose_solver/PoseSolverTypes.hpp',
    'src/backend/FixedLagBackend.cpp','src/backend/FixedLagBackend.hpp','src/backend/ViSlamBackend.hpp',
    'src/backend/tests/ManagedWindowRegression.cpp')
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
[xml]$project=Read 'CGMLI-SLAM/CGMLI-SLAM.vcxproj'
[xml]$filters=Read 'CGMLI-SLAM/CGMLI-SLAM.vcxproj.filters'
$ns=[Xml.XmlNamespaceManager]::new($project.NameTable); $ns.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
$fn=[Xml.XmlNamespaceManager]::new($filters.NameTable); $fn.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
foreach($entry in @(@('ClCompile','frontend\FrontendPoseAdapter.cpp'),@('ClInclude','frontend\FrontendPoseAdapter.hpp'),
    @('ClCompile','visual\pose_solver\RotationPoseSolver.cpp'),@('ClInclude','visual\pose_solver\RotationPoseSolver.hpp'),
    @('None','frontend\tests\FrontendPoseRegression.cpp'),@('None','frontend\tools\check_frontend_pose_static.ps1'))) {
    $tag=$entry[0]; $path='..\src\'+$entry[1]
    Check ('project registration: '+$entry[1]) (@($project.SelectNodes('//m:'+$tag,$ns) | Where-Object { $_.GetAttribute('Include') -eq $path }).Count -eq 1)
    Check ('filter registration: '+$entry[1]) (@($filters.SelectNodes('//m:'+$tag,$fn) | Where-Object { $_.GetAttribute('Include') -eq $path }).Count -eq 1)
    Check ('registered path exists: '+$entry[1]) (Test-Path -LiteralPath (Join-Path $root ('src/'+$entry[1])))
}
Check 'pose regression excluded from application build' (@($project.SelectNodes('//m:ClCompile',$ns) | Where-Object { $_.GetAttribute('Include') -like '*FrontendPoseRegression.cpp' }).Count -eq 0)
$contracts=@(
    @('dominant accepted rotation short circuits unused relative solve','src/frontend/Frontend.cpp','rotationDominant=accepted\(rotation\)&&rotation.inlierRatio>poseOptions_.rotationDominanceRatio'),
    @('other fallback cases retain generalized relative solver','src/frontend/Frontend.cpp','if\(!rotationDominant\)relative=pose_solver::RelativePoseSolver\(options\).solve\(input.data,&initial\)'),
    @('rotation-only analytic chart has zero translation columns','src/visual/pose_solver/RotationPoseSolver.cpp','J.leftCols<3>\(\).setZero\(\)'),
    @('rotation-only analytic chart uses spherical derivative','src/visual/pose_solver/RotationPoseSolver.cpp','J.rightCols<3>\(\)=-D\*detail::crossMatrix\(predicted\)'),
    @('rotation numeric reference remains selectable','src/visual/pose_solver/PoseSolverTypes.hpp','analyticRotationJacobians=true'),
    @('OpenGV removed from Frontend','src/frontend/Frontend.cpp','!opengv::|OpenGV|OpenGv|FrameNoncentralAbsoluteAdapter|FrameRelativeAdapter'),
    @('old model matrix assumptions removed','src/frontend/Frontend.cpp','!model_coefficients_|ransac\.inliers_|adapter\.camIndex|adapter\.keypointIndex'),
    @('old Ceres loop implementation removed','src/frontend/Frontend.cpp','!PoseManifold|HomogeneousPointManifold|ReprojectionError|quickSolver'),
    @('absolute solver used for tracking and loop','src/frontend/Frontend.cpp','AbsolutePoseSolver\(options\).solve\(input.data,&initial\)'),
    @('relative solver uses pooled input','src/frontend/Frontend.cpp','RelativePoseSolver\(options\).solve\(input.data,&initial\)'),
    @('rotation-only solver used as separate hypothesis','src/frontend/Frontend.cpp','RotationPoseSolver\(options\).solve\(input.data,&initial\)'),
    @('live body-camera calibration used','src/frontend/Frontend.cpp','estimator.extrinsics\(StateId\(frame.id\(\)\),camera\)'),
    @('self-anchored points excluded from absolute reseeding','src/frontend/Frontend.cpp','metadata->anchor.frameId==currentFrame->id\(\)'),
    @('absolute inliers map to original key','src/frontend/Frontend.cpp','candidate.rejected.insert\(input.observations\[i\].key\)'),
    @('relative inliers map to current key','src/frontend/Frontend.cpp','candidate.rejected.insert\(input.observations\[i\].second\)'),
    @('relative input pools all camera pairs','src/frontend/Frontend.cpp','frontend_pose::relativeInput\(\*it->second,\*currentFrame'),
    @('scale observability controls initialization','src/frontend/Frontend.cpp','constboolmetric=!useRotation&&result.translationScaleObservable'),
    @('relative pose composes old and new in correct direction','src/frontend/FrontendPoseAdapter.cpp','candidate=firstPose\*result.pose'),
    @('nonmetric seed keeps predicted position','src/frontend/FrontendPoseAdapter.cpp','kinematics::Transformation\(predicted.r\(\),firstPose.q\(\)\*result.pose.q\(\)\)'),
    @('failed relative solve does not claim metric initialization','src/frontend/Frontend.cpp','if\(!useRotation&&\!relativeAccepted\)returnfalse;'),
    @('old-frame loop points transformed explicitly','src/frontend/Frontend.cpp','constEigen::Vector3dpoint=T_old_W\*world'),
    @('loop outputs committed only after information acceptance','src/frontend/Frontend.cpp','T_Sold_Snew=result.pose;H=information;'),
    @('loop correspondence conflict resolution','src/frontend/Frontend.cpp','if\(used.count\(candidate.key\)\)continue;'),
    @('loop minimum distinct landmarks enforced','src/frontend/Frontend.cpp','uniquePoints.size\(\)<poseOptions_.minimumLoopLandmarks'),
    @('pixel covariance uses unprojection derivative','src/frontend/FrontendPoseAdapter.cpp','pixelToTangent\*pixelToTangent.transpose\(\)'),
    @('octave noise uses scale, not patch diameter','src/frontend/FrontendPoseAdapter.cpp','pixelSigma\*std::pow\(pyramidScale,point.octave\)'),
    @('left SE3 information Jacobian','src/frontend/FrontendPoseAdapter.cpp','backend::LeftSE3Manifoldmanifold;'),
    @('information is conditional fixed-point JtJ','src/frontend/FrontendPoseAdapter.cpp','total\+=J.transpose\(\)\*J'),
    @('rank deficient information is rejected','src/frontend/FrontendPoseAdapter.cpp','spectrum.eigenvalues\(\).minCoeff\(\)<=1e-9\*spectrum.eigenvalues\(\).maxCoeff\(\)'),
    @('information cannot double count inlier index','src/frontend/FrontendPoseAdapter.cpp','\!unique.insert\(index\).second'),
    @('absolute correspondence mapping appended atomically','src/frontend/FrontendPoseAdapter.cpp','output.data.push_back\(match\);output.observations.push_back\(\{key,id\}\)'),
    @('relative tracks selected once per landmark','src/frontend/FrontendPoseAdapter.cpp','for\(constauto&item:a\)'),
    @('relative camera-pair sampling balanced','src/frontend/FrontendPoseAdapter.cpp','\+\+used\[\{bestA->first.cameraIndex,bestB->first.cameraIndex\}\]'),
    @('ray direction built by generic project interface','src/frontend/FrontendPoseAdapter.cpp','pose_solver::makeGeneralizedRay\(extrinsics\[key.cameraIndex\],bearing,ray\)'),
    @('rotation estimator inherits common LO adapter','src/visual/pose_solver/RotationPoseSolver.cpp',':publicdetail::PoseEstimatorAdapter'),
    @('rotation RANSAC remains owned by PoseLib','src/visual/pose_solver/RotationPoseSolver.cpp','poselib::ransac<RotationEstimator>'),
    @('rotation hypothesis uses Wahba SVD','src/visual/pose_solver/RotationPoseSolver.cpp','Eigen::JacobiSVD<Eigen::Matrix3d>svd'),
    @('rotation observability has rank three only','src/visual/pose_solver/RotationPoseSolver.cpp','estimator.finish\(best,statistics,3,false\)'),
    @('rotation translation fixed to zero','src/visual/pose_solver/RotationPoseSolver.cpp','pose.set\(Eigen::Vector3d::Zero\(\),rotation\)'),
    @('sampler output has allocated size','src/visual/pose_solver/PoseEstimatorAdapter.hpp','sample_\(sample_sz\)'),
    @('sampler cardinality validated before construction','src/visual/pose_solver/PoseEstimatorAdapter.hpp','num_data\(checkedDataSize\(count,sampleSize\)\)'),
    @('sampler rejects impossible samples','src/visual/pose_solver/PoseEstimatorAdapter.hpp','if\(sampleSize==0\|\|count<sampleSize\)'),
    @('shared adapter owns PoseLib sampling','src/visual/pose_solver/PoseEstimatorAdapter.hpp','sampler_.generate_sample\(&sample_\)'),
    @('absolute pose uses sized shared buffer','src/visual/pose_solver/AbsolutePoseSolver.cpp','constauto&sample=drawMinimalSample\(\)'),
    @('relative pose uses sized shared buffer','src/visual/pose_solver/RelativePoseSolver.cpp','constauto&sample=drawMinimalSample\(\)'),
    @('rotation pose uses sized shared buffer','src/visual/pose_solver/RotationPoseSolver.cpp','constauto&sample=drawMinimalSample\(\)'),
    @('frontend cannot reseed historical epoch','src/backend/FixedLagBackend.cpp','window_.order_.back\(\)!=id'),
    @('frontend cannot reseed published pose','src/backend/FixedLagBackend.cpp','for\(constauto&state:published->states\)if\(state.id==id\)'),
    @('frontend transaction uses private staged window','src/backend/FixedLagBackend.cpp','auto staged=window_;'),
    @('frontend cannot subtract absorbed evidence','src/backend/FixedLagBackend.cpp','observation->use==ObservationUse::Absorbed'),
    @('frontend transaction validates before commit','src/backend/FixedLagBackend.cpp','if\(!staged.valid\(error\)\)returnfalse;window_=std::move\(staged\)'),
    @('frontend uses explicit transaction API','src/frontend/Frontend.cpp','estimator.applyFrontendBatch'),
    @('outlier identity agrees with authoritative association','src/frontend/Frontend.cpp','managed->landmark!=observation.landmark'),
    @('identity ordering cannot silently enable PROSAC','src/frontend/FrontendPoseAdapter.cpp','\!solver.ransac.progressive_sampling'),
    @('frontend no raw setPose bypass','src/frontend/Frontend.cpp','!estimator.setPose\(')
)
foreach($entry in $contracts) {
    $pattern=$entry[2]; $negative=$pattern.StartsWith('!'); if($negative) { $pattern=$pattern.Substring(1) }
    if($entry[0] -eq 'frontend transaction uses private staged window') { $pattern=$pattern.Replace(' ','') }
    $passed=$compact[$entry[1]] -match $pattern; if($negative) { $passed=!$passed }
    Check $entry[0] $passed
}
Check 'absolute consensus loop/tracking/full-budget fallback calls' ([regex]::Matches($code['src/frontend/Frontend.cpp'],'AbsolutePoseSolver\(options\)\.solve').Count -eq 3)
Check 'one final frontend input transaction' ([regex]::Matches($code['src/frontend/Frontend.cpp'],'estimator\.applyFrontendBatch\(').Count -eq 1)
Check 'absolute consensus reads const backend' ($code['src/frontend/Frontend.cpp'] -match 'runRansac3d2d\(const Estimator&')
Check 'absolute input reuses proposal coordinates' ($compact['src/frontend/Frontend.cpp'].Contains('absoluteInput(*currentFrame,plan.finitePoints,'))
Check 'absolute permissions use a proposal index' ($compact['src/frontend/Frontend.cpp'].Contains('proposedObservations.count({observation.key,observation.landmark})'))
Check 'cached absolute geometry has regression source' ($raw['src/frontend/tests/FrontendPoseRegression.cpp'].Contains('Cached coordinates/rays changed solver geometry'))
Check 'relative consensus reads const backend' ($code['src/frontend/Frontend.cpp'] -match 'runRansac2d2d\(const Estimator&')
foreach($name in @('minimalSamplingBuffers','absoluteMappingAndBodyConvention','pooledRelativeAndLoopFrames','centralScaleAndRotationFallback')) {
    Check ('regression source: '+$name) ($code['src/frontend/tests/FrontendPoseRegression.cpp'] -match ('void\s+'+$name+'\('))
}
Check 'sampling regression invoked by standalone main' (($compact['src/frontend/tests/FrontendPoseRegression.cpp'] -split '#elseintmain\(\)')[1].Contains('minimalSamplingBuffers();'))
Check 'uniform and PROSAC sampling regression' ($compact['src/frontend/tests/FrontendPoseRegression.cpp'] -match 'for\(boolprogressive:\{false,true\}\)')
$unsafeSamplerCalls=@(@('src/visual/pose_solver/AbsolutePoseSolver.cpp','src/visual/pose_solver/RelativePoseSolver.cpp','src/visual/pose_solver/RotationPoseSolver.cpp') | Where-Object { $code[$_] -match '\.generate_sample\(' })
Check 'no solver calls PoseLib with its own empty sample' ($unsafeSamplerCalls.Count -eq 0)
Check 'backend consensus rollback regression present' ($code['src/backend/tests/ManagedWindowRegression.cpp'] -match 'void\s+frontendConsensusTransaction\(')
Check 'actual PoseLib generic RANSAC header installed' (Test-Path -LiteralPath (Join-Path $root '..\vs2022\poselib2.0.5\include\PoseLib\robust\ransac_impl.h'))
Check 'actual PoseLib sampling header installed' (Test-Path -LiteralPath (Join-Path $root '..\vs2022\poselib2.0.5\include\PoseLib\robust\sampling.h'))
$backend=$code['src/backend/ViSlamBackend.hpp']
foreach($name in @('extrinsics','pose','multiFrame','getLandmarks','applyFrontendBatch')) {
    Check ('consensus backend API exists: '+$name) ($backend -match ('\b'+$name+'\s*\('))
}
Check 'absolute analytic derivative uses solver translation-addition chart' ($compact['src/visual/pose_solver/AbsolutePoseSolver.cpp'].Contains('j.rightCols<3>()=-D*detail::crossMatrix(rotated)'))
Check 'absolute numerical reference remains selectable' ($code['src/visual/pose_solver/AbsolutePoseSolver.cpp'] -match 'return options_\.analyticAbsoluteJacobians')
Check 'LO accumulates fixed six-dimensional normal equations' ($compact['src/visual/pose_solver/PoseSolverInternal.hpp'].Contains('normal.noalias()+=w*j.transpose()*j'))
Check 'LO rejected damping trials reuse normal equations' ($code['src/visual/pose_solver/PoseSolverInternal.hpp'] -match 'if\(!linearizationValid\)' -and $code['src/visual/pose_solver/PoseSolverInternal.hpp'] -match 'linearizationValid=false')
Check 'observability keeps direct Jacobian SVD' ($code['src/visual/pose_solver/PoseSolverInternal.hpp'] -match 'JacobiSVD<Eigen::MatrixXd> svd\(j\)')
Check 'healthy budget checks independent prediction' ($code['src/frontend/Frontend.cpp'] -match 'absolutePredictionHealthy\(input.data,initial,poseOptions_\)' -and $code['src/frontend/FrontendPoseAdapter.cpp'] -match 'data.size\(\)<o.minimumBudgetCorrespondences')
Check 'reduced budget has periodic and geometric fallbacks' ($code['src/frontend/Frontend.cpp'] -match 'fullRansacPeriod' -and $code['src/frontend/Frontend.cpp'] -match 'budgetFallback=true')
Check 'IMU gate rejection retries a full absolute budget' ($compact['src/frontend/Frontend.cpp'].Contains('decision.reducedBudget&&runRansac3d2d(estimator,plan,decision,true)&&consistent(decision)'))
Check 'new analytic and prediction guard regressions registered' ($raw['src/frontend/tests/FrontendPoseRegression.cpp'].Contains('"absoluteAnalyticLinearizationAgreement"') -and $raw['src/frontend/tests/FrontendPoseRegression.cpp'].Contains('"absoluteAnalyticRecoveryAndBudgetGuard"'))
$failures=@($checks|Where-Object { !$_.passed })
$report=[ordered]@{checkedAtUtc=[DateTime]::UtcNow.ToString('o');scope='Frontend OpenGV to project PoseLib adapter migration; heuristic static source/project checks only; no C++ compilation or execution';
    passed=$checks.Count-$failures.Count;total=$checks.Count;checks=$checks}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'frontend_pose_static_report.json'),($report|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
Write-Output ('Frontend pose static checks: {0}/{1}; no compilation or runtime tests' -f ($checks.Count-$failures.Count),$checks.Count)
foreach($failure in $failures) { Write-Output ('FAIL: '+$failure.name) }
if($failures.Count) { exit 1 }
