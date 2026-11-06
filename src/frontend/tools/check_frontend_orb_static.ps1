param()
$ErrorActionPreference='Stop'
$frontendRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$root=[IO.Path]::GetFullPath((Join-Path $frontendRoot '..\..'))
$checks=[Collections.Generic.List[object]]::new()
function Check([string]$name,[bool]$passed) { $checks.Add([pscustomobject]@{name=$name;passed=$passed}) }
function Read([string]$relative) { [IO.File]::ReadAllText((Join-Path $root $relative)) }
$strip='(?s)/\*.*?\*/|//[^\r\n]*|"(?:\\.|[^"\\])*"|''(?:\\.|[^''\\])*'''
$files=@('src/frontend/Frontend.cpp','src/frontend/Frontend.hpp','src/frontend/OrbFrontendFeatures.cpp',
    'src/frontend/OrbFrontendFeatures.hpp','src/frontend/FrontendAssociation.cpp','src/frontend/FrontendAssociation.hpp','src/frontend/tests/FrontendOrbRegression.cpp','src/visual/Frame.hpp',
    'src/visual/MultiFrame.hpp','src/backend/FrameStore.cpp','src/common/Parameters.hpp',
    'src/common/ViParametersReader.cpp','src/pipeline/ThreadedSlam.cpp','src/visual/features/OrbMatcher.cpp','src/visual/features/OrbMatcher.hpp')
$raw=@{}; $code=@{}; $compact=@{}
foreach($path in $files) {
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
    foreach($include in [regex]::Matches($source,'(?m)^\s*#include\s*[<"]((?:frontend|visual|common|backend|cameras|kinematics|utils)/[^>"]+)[>"]')) {
        Check ($path+': include '+$include.Groups[1].Value) (Test-Path -LiteralPath (Join-Path $root ('src/'+$include.Groups[1].Value)))
    }
}
[xml]$project=Read 'CGMLI-SLAM/CGMLI-SLAM.vcxproj'
[xml]$filters=Read 'CGMLI-SLAM/CGMLI-SLAM.vcxproj.filters'
$ns=[Xml.XmlNamespaceManager]::new($project.NameTable); $ns.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
$fn=[Xml.XmlNamespaceManager]::new($filters.NameTable); $fn.AddNamespace('m','http://schemas.microsoft.com/developer/msbuild/2003')
foreach($entry in @(@('ClCompile','OrbFrontendFeatures.cpp'),@('ClInclude','OrbFrontendFeatures.hpp'),
    @('None','tests\FrontendOrbRegression.cpp'),@('None','README.md'),@('None','tools\check_frontend_orb_static.ps1'))) {
    $relative='..\src\frontend\'+$entry[1]; $tag=$entry[0]
    Check ('project registration: '+$entry[1]) (@($project.SelectNodes('//m:'+$tag,$ns) | Where-Object { $_.GetAttribute('Include') -eq $relative }).Count -eq 1)
    Check ('filter registration: '+$entry[1]) (@($filters.SelectNodes('//m:'+$tag,$fn) | Where-Object { $_.GetAttribute('Include') -eq $relative }).Count -eq 1)
    Check ('registered path exists: '+$entry[1]) (Test-Path -LiteralPath (Join-Path $frontendRoot $entry[1]))
}
Check 'standalone regression not in application build' (@($project.SelectNodes('//m:ClCompile',$ns) | Where-Object { $_.GetAttribute('Include') -like '*FrontendOrbRegression.cpp' }).Count -eq 0)
Check 'deferred overlap and concurrent lazy mask regression present' ($compact['src/frontend/tests/FrontendOrbRegression.cpp'] -match 'voiddeferredRigOverlaps\(\)' -and $compact['src/frontend/tests/FrontendOrbRegression.cpp'] -match 'try\{deferredRigOverlaps\(\);')
Check 'sparse matching compared with dense reference in regression source' ($code['src/frontend/tests/FrontendOrbRegression.cpp'] -match 'void\s+sparseGuidedMatchingEquivalence\(' -and $code['src/frontend/tests/FrontendOrbRegression.cpp'] -match 'matcher.matchFiltered\(query,train,expected,permitted\)')
Check 'octave partition retains deterministic exact neighbor semantics' ($compact['src/visual/features/OrbMatcher.cpp'].Contains('trainOctaves.lower_bound(first)') -and $compact['src/frontend/tests/FrontendOrbRegression.cpp'].Contains('for(intdifference:{0,1,3})'))
Check 'matching reports whole traversal interval without per-pair timers' ($compact['src/visual/features/OrbMatcher.cpp'].Contains('statistics->traversalMs=milliseconds(traversalStarted)'))
Check 'detection internal timings logged' ($raw['src/frontend/OrbFrontendFeatures.cpp'].Contains('candidate_filter_ms=') -and $raw['src/frontend/OrbFrontendFeatures.cpp'].Contains('balance_ms='))
Check 'FAST dispatch warmup runs once before camera workers' ($raw['src/frontend/OrbFrontendFeatures.cpp'].Contains('std::call_once(fastWarmup,') -and $raw['src/frontend/OrbFrontendFeatures.cpp'].Contains('[Timing][FastWarmup]'))
Check 'frontend finite-point prediction includes translation' ($compact['src/frontend/FrontendAssociation.cpp'].Contains('pc=T_CW.C()*world+T_CW.r()'))
Check 'frontend no longer allocates a dense pair mask' (!$compact['src/frontend/FrontendAssociation.cpp'].Contains('guidance.allowedPairs=cv::Mat::zeros'))
Check 'guided matching computes both neighbor directions from one distance' ($compact['src/visual/features/OrbMatcher.cpp'].Contains('best.add(j,distance)') -and $compact['src/visual/features/OrbMatcher.cpp'].Contains('reverse[j].add(group,distance)'))
Check 'map identities preserve distinct reverse neighbors' ($compact['src/visual/features/OrbMatcher.cpp'].Contains('guidance.queryIdentities') -and $compact['src/frontend/tests/FrontendOrbRegression.cpp'].Contains('groupedMapNeighborsAndPreparedLayouts();'))
Check 'existing identity priority retains full reverse fallback' ($compact['src/visual/features/OrbMatcher.cpp'].Contains('!reciprocal(reverse[j])&&!(tier&&reciprocal(ownedReverse[j]))'))
Check 'priority matching does not add a second Hamming traversal' ($compact['src/frontend/tests/FrontendOrbRegression.cpp'].Contains('prioritized.descriptorComparisons==baseline.descriptorComparisons'))
Check 'priority matching retains strict sparse-neighbor regression' ($compact['src/frontend/tests/FrontendOrbRegression.cpp'].Contains('guidance.queryEnabled={1,0,1}'))
Check 'training indexes are shared without per-job feature gates' ($compact['src/visual/features/OrbMatcher.hpp'].Contains('constOrbTrainIndex*trainIndex=nullptr;') -and $compact['src/frontend/FrontendAssociation.cpp'].Contains('guidance.trainIndex=&trainCache.trainIndex;'))
Check 'job-specific train gates compact octave and pixel buckets' ($compact['src/visual/features/OrbMatcher.cpp'].Contains('if(validT[j])filteredGrid[cell].push_back(j);') -and $compact['src/visual/features/OrbMatcher.cpp'].Contains('filterTrain?filteredOctaves:index.octaves'))
Check 'filtered buckets have neighbor and immutability regression coverage' ($raw['src/frontend/tests/FrontendOrbRegression.cpp'].Contains('Disabled/overlap-excluded rows must not enter candidate traversal') -and $raw['src/frontend/tests/FrontendOrbRegression.cpp'].Contains('A job must not mutate the shared training index'))
Check 'immutable ray cache is shared after explicit feature copy' ($compact['src/backend/FrameStore.cpp'].Contains('result->shareBackProjections(c,source);') -and $compact['src/visual/Frame.hpp'].Contains('std::shared_ptr<constBackProjectionCache>backProjectionCache_;'))
$contracts=@(
    @('frontend startup preparation is exposed','src/frontend/Frontend.hpp','voidprepareOrb\(constcameras::NCameraSystem&rig,boolparallel=true\)'),
    @('startup shares actual camera geometry without images','src/frontend/Frontend.cpp','constMultiFramecalibration\(rig,Time\(0,0\)\);orbFeatures_.prepare\(calibration,parallel\)'),
    @('per-camera startup builds static masks','src/frontend/OrbFrontendFeatures.cpp','extractors_\[camera\]->prepareStaticCaches\(\)'),
    @('startup async futures explicitly launched','src/frontend/OrbFrontendFeatures.cpp','std::async\(std::launch::async,prepareCamera,camera\)'),
    @('startup workers complete before returning','src/frontend/OrbFrontendFeatures.cpp','for\(auto&worker:workers\)worker.get\(\)'),
    @('prepared first extraction still logs timing','src/frontend/OrbFrontendFeatures.cpp','VLOG_IS_ON\(1\)\|\|firstExtraction\|\|cold'),
    @('first-use logging uses independent camera bytes','src/frontend/OrbFrontendFeatures.hpp','std::vector<unsignedchar>extracted_'),
    @('startup regression is invoked','src/frontend/tests/FrontendOrbRegression.cpp','startupCameraPreparation\(\);'),
    @('Frontend delegates actual extraction','src/frontend/Frontend.cpp','returnorbFeatures_.detectAndDescribe\(cameraIndex,\*frameOut,T_WC,keypoints\)'),
    @('no BRISK runtime dependencies','src/frontend/Frontend.cpp','!brisk::|cv::Brisk|initialiseBrisk|setCameraProperties'),
    @('no fixed 48-byte reads','src/frontend/Frontend.cpp','!\b48\b'),
    @('32-byte descriptor contract','src/frontend/OrbFrontendFeatures.hpp','kDescriptorBytes=32;'),
    @('portable unaligned Hamming reads','src/frontend/OrbFrontendFeatures.cpp','std::memcpy\(&a,first\+offset,4\)'),
    @('one lock for requested camera','src/frontend/OrbFrontendFeatures.cpp','lock\(\*mutexes_\[camera\]\)'),
    @('one extractor for requested camera','src/frontend/OrbFrontendFeatures.cpp','auto&extractor=extractors_\[camera\]'),
    @('per-camera model is passed to ORB','src/frontend/OrbFrontendFeatures.cpp','make_shared<OrbFeatureExtractor>\(frame.geometry\(camera\),options\)'),
    @('common rig focal independent of execution order','src/frontend/OrbFrontendFeatures.cpp','frame.geometry\(0\)->getIntrinsics'),
    @('gravity uses supplied camera pose','src/frontend/OrbFrontendFeatures.cpp','updateOrbGravity\(\*extractor,T_WC\)'),
    @('shared pyramid detectAndCompute call','src/frontend/OrbFrontendFeatures.cpp','extractor->detectAndCompute\(frame.image\(camera\),cv::noArray\(\),points,descriptors,supplied!=nullptr\)'),
    @('indices replaced after descriptor filtering','src/frontend/OrbFrontendFeatures.cpp','frame.resetKeypoints\(camera,points\);frame.resetDescriptors\(camera,descriptors\)'),
    @('backprojections follow final indices','src/frontend/OrbFrontendFeatures.cpp','frame.computeBackProjections\(camera\)'),
    @('profile stores full precision','src/frontend/OrbFrontendFeatures.cpp','max_digits10'),
    @('profile includes reference focal','src/frontend/OrbFrontendFeatures.cpp','extractor.virtualFocalLength\(\)'),
    @('profile validated before association','src/frontend/Frontend.cpp','orbFeatures_.validateFrame\(\*framesInOut\)'),
    @('old descriptor views cached and validated','src/frontend/Frontend.cpp','for\(constauto&item:plan.sources\)orbFeatures_.validateFrame\(\*item.second\)'),
    @('matching uses actual feature rows without uninitialized pools','src/frontend/FrontendAssociation.cpp','matcher.matchGuided\(source.descriptors\(q\),current->descriptors\(t\)'),
    @('current Mat row stride respected','src/frontend/Frontend.cpp','framesInOut->descriptors\(camera\).ptr<uchar>'),
    @('profiles survive backend copies','src/backend/FrameStore.cpp','result->setDescriptorProfile\(c,source.descriptorProfile\(c\)\)'),
    @('source image timing survives frozen copies','src/backend/FrameStore.cpp','result->setImageTiming\(c,timing->originalTimestamp,timing->offsetNs\)'),
    @('source image timing is checked before association','src/frontend/FrontendAssociation.cpp','inputTiming.has_value\(\)!=frozenTiming.has_value\(\)'),
    @('debug captures the actual matcher candidates','src/frontend/FrontendAssociation.cpp','if\(options.collectDebugMatches\)plan.debugMatches=std::move\(candidates\)'),
    @('debug pair capture gated by output option','src/frontend/Frontend.cpp','options.collectDebugMatches=params.output.display_matches'),
    @('debug colors derive from surviving track endpoints','src/frontend/Frontend.cpp','match.retained=a!=0&&a==second.landmarkId'),
    @('FORB database implementation','src/frontend/Frontend.cpp','OrbDatabasedatabase;'),
    @('FORB features are Mat rows','src/frontend/Frontend.cpp','std::vector<cv::Mat>features;'),
    @('vocabulary profile guard','src/frontend/Frontend.cpp','dBow_->descriptorProfile!=framesInOut->descriptorProfile\(0\)'),
    @('unloaded vocabulary never queried','src/frontend/Frontend.cpp','if\(!dBow\|\|features.empty\(\)\)return0;'),
    @('vocabulary load is atomic','src/frontend/Frontend.cpp','dBow_=std::move\(candidate\)'),
    @('sample variance avoids single-point division','src/frontend/Frontend.cpp','row<2'),
    @('Hamming threshold bounded to 256 bits','src/frontend/Frontend.cpp','threshold<0\|\|threshold>kOrbDescriptorBits'),
    @('old BRISK setters removed from pipeline','src/pipeline/ThreadedSlam.cpp','!setBrisk'),
    @('native options reach frontend','src/pipeline/ThreadedSlam.cpp','frontend_.setOrbOptions\(orbOptions\)')
)
foreach($entry in $contracts) {
    $pattern=$entry[2]; $negative=$pattern.StartsWith('!'); if($negative) { $pattern=$pattern.Substring(1) }
    $ok=$compact[$entry[1]] -match $pattern; if($negative) { $ok=!$ok }
    Check $entry[0] $ok
}
Check 'loop verification uses fixed 256-bit Hamming' ([regex]::Matches($code['src/frontend/Frontend.cpp'],'OrbFeatures::hammingDistance\(').Count -eq 1)
Check 'legacy guided loops replaced by OrbMatcher' ($code['src/frontend/Frontend.cpp'] -notmatch 'matchToMapByThread|matchMotionStereo|template\s*<\s*class\s+CAMERA_GEOMETRY')
foreach($name in @('detectAndDescribe','options','setOptions','clear','validateFrame','validateDescriptors','hammingDistance','lockAll','referenceFocal','profile')) {
    Check ('feature component definition: '+$name) ([regex]::Matches($code['src/frontend/OrbFrontendFeatures.cpp'],('OrbFrontendFeatures::'+$name+'\(')).Count -eq 1)
}
foreach($key in @('orb_pyramid_levels','orb_scale_factor','orb_fast_threshold','orb_min_fast_threshold','orb_virtual_focal_length','orb_matching_threshold')) {
    foreach($path in @('src/common/Parameters.hpp','src/common/ViParametersReader.cpp','src/pipeline/ThreadedSlam.cpp')) {
        Check ($path+': native option '+$key) ($raw[$path].Contains($key))
    }
}
foreach($test in @('hammingAndRowLayout','mixedCamerasReuseAndProfiles','parallelCameraInitialization')) {
    Check ('regression source: '+$test) ($code['src/frontend/tests/FrontendOrbRegression.cpp'] -match ('void\s+'+$test+'\('))
}
# Report existing integration debt honestly. Passing feature structure checks cannot
# prove that legacy full-graph/OpenGV/IMU/pipeline code matches the current backend.
$backendCode=[regex]::Replace((Read 'src/backend/ViSlamBackend.hpp'),$strip,' ')
$legacyCalls=@([regex]::Matches($code['src/frontend/Frontend.cpp'],'(?<![\w.])estimator\.([A-Za-z_]\w*)\s*(?:<[^>]+>)?\s*\(') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique | Where-Object { $backendCode -notmatch ('\b'+$_+'\s*\(') })
$gaps=[Collections.Generic.List[string]]::new()
if($legacyCalls.Count) { $gaps.Add('Legacy Frontend calls absent from current ViSlamBackend: '+($legacyCalls -join ', ')) }
if($code['src/frontend/Frontend.cpp'] -match 'opengv::') { $gaps.Add('Legacy association/RANSAC still references OpenGV adapters; current project pose solvers use PoseLib.') }
if($code['src/frontend/Frontend.cpp'] -match 'cgmli::ceres::ImuError') { $gaps.Add('IMU propagation still depends on the unported cgmli::ceres::ImuError implementation.') }
if($code['src/frontend/Frontend.cpp'] -match 'geometryAs<CAMERA_GEOMETRY>') { $gaps.Add('Legacy data association still uses homogeneous camera templates; mixed-model support here is limited to the new extraction component and existing OrbMatcher APIs.') }
Check 'all Frontend backend calls exist' ($legacyCalls.Count -eq 0)
if($code['src/pipeline/ThreadedSlam.cpp'] -match 'optimiseRealtimeGraph|optimiseFullGraph') { $gaps.Add('Original ThreadedSlam graph/IMU call sites are restored; their owning-class declarations intentionally await implementations. Pipeline is not link-ready.') }
$failures=@($checks | Where-Object { !$_.passed })
$report=[ordered]@{checkedAtUtc=[DateTime]::UtcNow.ToString('o');scope='ORB feature integration: heuristic static source/project inspection; no compilation or C++ execution';
    passed=$checks.Count-$failures.Count;total=$checks.Count;existingIntegrationGaps=$gaps;checks=$checks}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'frontend_orb_static_report.json'),($report|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
Write-Output ('Frontend ORB static checks: {0}/{1}; no compilation or runtime tests' -f ($checks.Count-$failures.Count),$checks.Count)
Write-Output ('Existing full-frontend integration gaps recorded: '+$gaps.Count)
foreach($failure in $failures) { Write-Output ('FAIL: '+$failure.name) }
if($failures.Count) { exit 1 }
