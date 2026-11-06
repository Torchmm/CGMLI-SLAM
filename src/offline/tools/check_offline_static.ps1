param()
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$checks=[Collections.Generic.List[object]]::new()
function Check([string]$name,[bool]$passed) { $checks.Add([pscustomobject]@{name=$name;passed=$passed}) }
$files=@('src/offline/HiltiBinReader.hpp','src/offline/HiltiBinReader.cpp','src/offline/HiltiDatasetReader.hpp',
    'src/offline/HiltiDatasetReader.cpp','src/app/offline_main.cpp','src/app/CsvOffline.cpp','src/app/CsvOffline.hpp',
    'src/offline/tests/HiltiOfflineRegression.cpp')
$source=@{}
$strip='(?s)/\*.*?\*/|//[^\r\n]*|"(?:\\.|[^"\\])*"|''(?:\\.|[^''\\])*'''
foreach($file in $files) {
    $raw=[IO.File]::ReadAllText((Join-Path $root $file)); $source[$file]=$raw
    $clean=[regex]::Replace($raw,$strip,' ')
    $stack=[Collections.Generic.Stack[char]]::new(); $balanced=$true; $pairs=@{')'='(';']'='[';'}'='{'}
    foreach($character in $clean.ToCharArray()) {
        if($character -in '(','[','{') { $stack.Push($character) }
        elseif($pairs.ContainsKey([string]$character)) {
            if(!$stack.Count -or $stack.Pop() -ne $pairs[[string]$character]) { $balanced=$false; break }
        }
    }
    Check ($file+': balanced delimiters') ($balanced -and !$stack.Count)
    Check ($file+': no conflict markers') ($raw -notmatch '(?m)^\s*(<<<<<<<|=======|>>>>>>>)')
    foreach($include in [regex]::Matches($raw,'(?m)^\s*#include\s*[<"]((?:offline|pipeline|common)/[^>"]+)[>"]')) {
        Check ($file+': include '+$include.Groups[1].Value) (Test-Path -LiteralPath (Join-Path $root ('src/'+$include.Groups[1].Value)))
    }
}
$reader=$source['src/offline/HiltiBinReader.cpp']; $stream=$source['src/offline/HiltiDatasetReader.cpp']; $app=$source['src/app/offline_main.cpp']
$conditions=@(
    @('file logs use a native path stream',$app.Contains('stream_.open(file,std::ios::binary)')),
    @('logging keeps console output',$app.Contains('FLAGS_logtostderr=true')),
    @('logs include a latest-run pointer',$app.Contains('std::ofstream latest(root/"latest.txt"')),
    @('logging uses a glog sink',$app.Contains('google::AddLogSink(this)')),
    @('sink detached before stream destruction',$app.Contains('google::RemoveLogSink(this)')),
    @('logging uses official glog prefix formatting',$app.Contains('google::LogSink::ToString(')),
    @('normal log flushing is batched',$app.Contains('now-lastFlush_>=std::chrono::seconds(1)')),
    @('warning and error logs flush immediately',$app.Contains('severity>=google::GLOG_WARNING')),
    @('source checkout gets a stable log directory',$app.Contains('return directory/"logs"')),
    @('log directory has an environment override',$app.Contains('std::getenv("CGMLI_LOG_DIR")')),
    @('glog is shut down on every exit',$app.Contains('~LoggingShutdown() { google::ShutdownGoogleLogging(); }')),
    @('32-byte explicit record header',$reader.Contains('std::array<uint8_t,32>')),
    @('directory manifest maps topics',$reader.Contains('summary["topic_files"][topic]')),
    @('directory file identity retained',$reader.Contains('item.fileId=fileId')),
    @('manifest cannot escape dataset',$reader.Contains('target.parent_path()!=root')),
    @('incomplete conversion rejected',$reader.Contains('summary["complete"].as<bool>()')),
    @('partial conversion rejected',$reader.Contains('path.extension()==".partial"')),
    @('truncation validated before allocation',$reader.Contains('item.payloadBytes>remaining-item.metadataBytes')),
    @('source headers sorted',$reader.Contains('a.sensorNs<b.sensorNs')),
    @('complete four-camera grouping',$reader.Contains('if(exhausted) break;')),
    @('integer ns mean avoids epoch precision loss',$reader.Contains('packet.timestampNs=packet.firstImageNs+(offsetSum+2)/4;')),
    @('original camera times retained',$reader.Contains('packet.imageTimestampNs[c]=records[cameras[c][cursors[c]]].sensorNs;')),
    @('ambiguous frame windows rejected',$reader.Contains('if(2*toleranceNs>=records[camera[i]].sensorNs-records[camera[i-1]].sensorNs)')),
    @('unmatched images counted',$reader.Contains('result.unmatchedImages=totalImages-4*packets.size();')),
    @('reader refuses partial image bundles',$stream.Contains('if(images.size()!=4 || originalTimes.size()!=4)')),
    @('source timestamps delivered with images',$stream.Contains('imagesCallback_(projectTime(packet.timestampNs),images,originalTimes)')),
    @('application preserves source image times',$app.Contains('estimator.addImagesWithTimestamps(t,images,originalTimes)')),
    @('YAML controls grouping span',$app.Contains('parameters.camera.timestamp_tolerance*1e9')),
    @('bag time retained',$reader.Contains('result.bagNs=raw.index.bagNs')),
    @('no bag-time fallback',($reader -notmatch 'sensorNs\s*=\s*[^;]*bagNs')),
    @('RGB converted for native pipeline',$reader.Contains('cv::COLOR_RGB2BGR')),
    @('raw image row stride checked',$reader.Contains('uint64_t(height)*step!=raw.payload.size()')),
    @('IMU gyro index 13',$reader.Contains('result.original[13+i]')),
    @('IMU acceleration index 25',$reader.Contains('result.original[25+i]')),
    @('cloud row stride honored',$reader.Contains('uint64_t(row)*result.rowStep')),
    @('cloud double timestamp required',$reader.Contains('timestamp.datatype!=8')),
    @('raw cloud preserved',$reader.Contains('result.originalData=std::move(raw.payload)')),
    @('calibrated camera geometry reused',$stream.Contains('parameters_.nCameraSystem.cameraGeometry(image.camera)')),
    @('reader does not reconstruct or modify calibration',($stream -notmatch '\.addCamera\(')),
    @('image dimensions checked',$stream.Contains('camera->imageWidth()')),
    @('all callbacks required',$stream.Contains('!imuCallback_ || !imagesCallback_ || !lidarCallback_')),
    @('not calling unavailable LiDAR point interface',($app -notmatch '\.addLidarMeasurement\(')),
    @('20ms future IMU support',$stream.Contains('corrected(packet)+Duration(0.02)')),
    @('physical IMU gap checked',$stream.Contains('parameters_.imu.max_sample_gap')),
    @('reader has no estimator process loop',($stream -notmatch '\.processFrame\(')),
    @('reader has no estimator dependency',($source['src/offline/HiltiDatasetReader.hpp'] -notmatch '#include.*ThreadedSlam')),
    @('EOF drains remaining worker',$app.Contains('while(datasetReader.isStreaming() || estimator.hasPendingWork())')),
    @('terminal fault propagated',$app.Contains('estimator.hasFatalError()')),
    @('no duplicate acceleration scaling',($stream -notmatch 's_a\.cwiseProduct')),
    @('Unicode Windows entry',($app -match 'int\s+wmain\s*\(\s*int\s+argc\s*,\s*wchar_t\s*\*\s*\*\s*argv\s*\)')),
    @('inspection entry provided',$app.Contains('"--inspect"')),
    @('reader worker owns streaming',$stream.Contains('std::thread(&HiltiDatasetReader::processing,this)')),
    @('reader callback exception retained',$stream.Contains('failure_=std::current_exception()')),
    @('worker failure rethrown',$stream.Contains('std::rethrow_exception(failure_)')),
    @('blocking receiver shutdown precedes reader join',($app -match 'requestStop\(\);\s*estimator.stopThreading\(\);\s*datasetReader.stopStreaming\(\);')),
    @('application connects image callback',$app.Contains('datasetReader.setImagesCallback(')),
    @('application connects IMU callback',$app.Contains('datasetReader.setImuCallback(')),
    @('application connects LiDAR callback',$app.Contains('datasetReader.setLidarCallback(')),
    @('application starts streaming',$app.Contains('datasetReader.startStreaming()')),
    @('debug GUI runs on application thread',$app.Contains('showDebugViews(images,latestViews,windows)')),
    @('debug GUI can pause processing',$app.Contains('const bool processed=!paused && estimator.processFrame()')),
    @('debug screenshots support Unicode paths',$app.Contains('std::ofstream file(folder/(name+".png"),std::ios::binary)')),
    @('single estimator config and dataset entry',$app.Contains('runHilti(fs::u8path(argv[1]),dataset,output)')),
    @('no extra offline YAML parsing',($app -notmatch 'YAML::|struct Configuration|readConfiguration')),
    @('real calibration required',$app.Contains('ViParametersReader parameterReader(config.u8string())')),
    @('visual success checked',$app.Contains('if(visualUpdates.load()==0)')),
    @('CSV compatibility dispatched',$app.Contains('return runCsvOffline(argc,argv)'))
)
foreach($condition in $conditions) { Check $condition[0] $condition[1] }
Check 'CSV compatibility entry implemented' ($source['src/app/CsvOffline.cpp'] -match '\bint\s+runCsvOffline\([^;{}]*\)\s*\{')
Check 'CSV entry signature shared with dispatcher' ($app.Contains('#include "CsvOffline.hpp"') -and $source['src/app/CsvOffline.cpp'].Contains('#include "CsvOffline.hpp"'))
Check 'CSV source does not introduce a second main' ($source['src/app/CsvOffline.cpp'] -notmatch '\bint\s+(?:wmain|main)\s*\(')
Check 'application stops before final export' ($app.IndexOf('estimator.stopThreading();') -lt $app.IndexOf('estimator.writeFinalTrajectoryCsv();'))
Check 'callback regression present' ($source['src/offline/tests/HiltiOfflineRegression.cpp'].Contains('streamingCallbacks(directory);'))
Check 'mean/grouping/drop recovery regression present' ($source['src/offline/tests/HiltiOfflineRegression.cpp'].Contains('meanTimeBundles(path);'))
Check 'blocking cancellation regression present' ($source['src/offline/tests/HiltiOfflineRegression.cpp'].Contains('stopBlockedCallback(directory);'))
[xml]$project=[IO.File]::ReadAllText((Join-Path $root 'CGMLI-SLAM/CGMLI-SLAM.vcxproj'))
[xml]$filters=[IO.File]::ReadAllText((Join-Path $root 'CGMLI-SLAM/CGMLI-SLAM.vcxproj.filters'))
$ns=[Xml.XmlNamespaceManager]::new($project.NameTable); $ns.AddNamespace('m',$project.DocumentElement.NamespaceURI)
$fn=[Xml.XmlNamespaceManager]::new($filters.NameTable); $fn.AddNamespace('m',$filters.DocumentElement.NamespaceURI)
$csvDefinitions=0
foreach($entry in $project.SelectNodes('//m:ClCompile[@Include]',$ns)) {
    $path=Join-Path $root ('CGMLI-SLAM/'+$entry.Include)
    if(Test-Path -LiteralPath $path) {
        $text=[IO.File]::ReadAllText($path)
        $csvDefinitions+=[regex]::Matches($text,'(?m)^\s*int\s+runCsvOffline\([^;{}]*\)\s*\{').Count
    }
}
Check 'exactly one CSV definition in compiled project sources' ($csvDefinitions -eq 1)
foreach($file in $files) {
    $include='..\'+$file.Replace('/','\'); $kind=if($file.Contains('/tests/')) {'None'} elseif($file.EndsWith('.cpp')) {'ClCompile'} else {'ClInclude'}
    Check ($file+': project entry') (@($project.SelectNodes('//m:'+$kind,$ns) | Where-Object { $_.Include -eq $include }).Count -eq 1)
    Check ($file+': filter entry') (@($filters.SelectNodes('//m:'+$kind,$fn) | Where-Object { $_.Include -eq $include }).Count -eq 1)
}
foreach($node in $project.SelectNodes('//m:ClCompile|//m:ClInclude|//m:None',$ns)) {
    Check ('project file exists: '+$node.Include) (Test-Path -LiteralPath (Join-Path $root ('CGMLI-SLAM/'+$node.Include)))
}
Check 'standalone regression excluded from app' (@($project.SelectNodes('//m:ClCompile',$ns) | Where-Object { $_.Include -like '*HiltiOfflineRegression.cpp' }).Count -eq 0)
Check 'CSV fast path has record bounds and complete-file coverage' ($reader.Contains('item.offset!=ends[fileId]') -and $reader.Contains('ends[f]!=lengths[f]'))
Check 'deferred CSV header is verified on record consumption' ($reader.Contains('expected.recordBytes!=32+uint64_t(item.metadataBytes)+item.payloadBytes'))
Check 'VIO app explicitly skips unused LiDAR points' ($app.Contains('readerOptions.decodeLidarPoints=false') -and $stream.Contains('options_.decodeLidarPoints'))
$failed=@($checks | Where-Object {!$_.passed})
if($failed.Count) { $failed | Format-Table -AutoSize; throw "$($failed.Count) offline static checks failed" }
Write-Output "Offline static checks passed: $($checks.Count). C++ was not compiled or executed."
