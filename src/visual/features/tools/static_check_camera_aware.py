"""Source/API/configuration audit only; does not compile or execute C++ tests."""
from pathlib import Path
import hashlib
import re
import sys
import xml.etree.ElementTree as ET

FEATURES = Path(__file__).resolve().parents[1]
ROOT = FEATURES.parents[2]
results = []


def check(name, condition):
    results.append((name, bool(condition)))


def uncomment(source):
    # Retain newlines and blank out strings/comments for delimiter checking.
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: '\n' * m.group().count('\n'), source, flags=re.S)


def balanced(source):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for token in uncomment(source):
        if token in '([{':
            stack.append(token)
        elif token in pairs:
            if not stack or stack.pop() != pairs[token]:
                return False
    return not stack


files = [*FEATURES.glob('*.hpp'), *FEATURES.glob('*.cpp'),
         FEATURES / 'tests/OrbCameraAwareRegression.cpp']
sources = {path.name: path.read_text(encoding='utf-8-sig') for path in files}
extractor = sources['OrbFeatureExtractor.cpp']
header = sources['OrbFeatureExtractor.hpp']
adapter = sources['OrbFrameAdapter.cpp']
adapter_header = sources['OrbFrameAdapter.hpp']
pattern = sources['OrbBriefPattern.hpp']
tests = sources['OrbCameraAwareRegression.cpp']
sampling = sources['OrbCameraAwareSampling.hpp']
sparse = sources['OrbSparseSampling.hpp']

project_path = ROOT / 'CGMLI-SLAM/CGMLI-SLAM.vcxproj'
project = ET.parse(project_path)
namespace = {'ms': 'http://schemas.microsoft.com/developer/msbuild/2003'}
include_roots = [ROOT / 'src']
for item in project.findall('.//ms:AdditionalIncludeDirectories', namespace):
    for directory in (item.text or '').split(';'):
        if directory and '$(' not in directory and '%(' not in directory:
            path = Path(directory)
            include_roots.append(path if path.is_absolute() else (project_path.parent / path).resolve())

for path in files:
    check(f'{path.name}: balanced delimiters', balanced(sources[path.name]))
    check(f'{path.name}: no conflict markers', not re.search(r'^\s*(<<<<<<<|=======|>>>>>>>)', sources[path.name], re.M))
    for include in re.findall(r'^\s*#include [<"]([^>"]+)[>"]', sources[path.name], re.M):
        if include.startswith(('visual/', 'cameras/', 'kinematics/', 'Eigen/', 'opencv2/')):
            check(f'{path.name}: include {include}', any((directory / include).is_file() for directory in include_roots))

pattern_body = pattern.split('kOrbPattern[256][4] = {', 1)[1].split('};', 1)[0]
pairs = [tuple(map(int, row)) for row in re.findall(
    r'\{\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\}', pattern_body)]
check('ORB learned pattern: exactly 256 point pairs', len(pairs) == 256)
unique_positions = {(row[i], row[i + 1]) for row in pairs for i in (0, 2)}
check('ORB sparse pattern: exactly 375 unique source positions', len(unique_positions) == 375)
check('ORB learned pattern: positions inside 31x31', all(-15 <= n <= 15 for row in pairs for n in row))
check('ORB learned pattern: upstream first/last pair', pairs[0] == (8, -3, 9, 5) and pairs[-1] == (-1, -6, 0, -11))
check('ORB learned pattern: copyright/license retained', 'Willow Garage' in pattern and 'Redistribution' in pattern and 'AS IS' in pattern)
pattern_digest = hashlib.sha256(','.join(str(n) for row in pairs for n in row).encode()).hexdigest()
check('ORB learned pattern: all upstream numeric pairs unchanged',
      pattern_digest == '88df8ca875cc8db56799edd57bb914edad8acb2d48c202b7a464a575b55dbdb8')

for method in ('setExtractionDirection', 'clearExtractionDirection', 'initialiseGeometry', 'geometryAt', 'samplingWarp', 'levelToBase', 'baseToLevel'):
    check(f'{method}: declaration and definition', method in header and f'OrbFeatureExtractor::{method}(' in extractor)
for method in ('configureOrb', 'bindOrb', 'updateOrbGravity'):
    check(f'{method}: Frame/rig overloads', adapter.count(f'{method}(') == 2 and adapter_header.count(f'{method}(') == 2)
for name, text, fragment in (
    ('Feature2D interface retained', header, ': public cv::Feature2D'),
    ('descriptor remains 32 bytes', header, 'descriptorSize() const override { return 32; }'),
    ('descriptor remains Hamming', header, 'cv::NORM_HAMMING'),
    ('camera-aware gravity enabled by default', header, 'descriptorMode = DescriptorMode::GravityCameraAware'),
    ('new descriptor profile version', header, 'cgmli.ORB.GravityCameraAware.v2'),
    ('sparse backend is the default', header, 'samplingMode = SamplingMode::SparseBox'),
    ('sparse descriptor profile version', header, 'cgmli.ORB.GravityCameraAware.SparseBox.v3'),
    ('linear-time compile-time deduplication', sparse, 'int indices[31][31]{}'),
    ('375 shared gray samples', sparse, 'kSparseOrbPattern.count == 375'),
    ('reference-plane box scale', sparse, '3.4641016151377546 * warp.row(0).norm()'),
    ('fractional-area pixel-cell convention', sparse, 'pixel.x()+0.5-half_.x()'),
    ('sparse gravity-pole probes', sparse, 'if (x * x + y * y > 9) continue'),
    ('sparse comparisons reuse cached gray values', sparse, 'values[kSparseOrbPattern.pairs[bit][0]] < values[kSparseOrbPattern.pairs[bit][1]]'),
    ('integral built once per used layer', extractor, 'if (!intensityReady[point.octave])'),
    ('gray integral readiness resets for every image', extractor, 'std::vector<bool> intensityReady(images.size(),false)'),
    ('external masks bypass the static cache', extractor, 'const bool reusable=externalMask.empty()'),
    ('geometry reuse stays within a combined call', extractor, 'useProvidedKeypoints?nullptr:&geometry'),
    ('candidate geometry follows response ordering', extractor, 'std::sort(points.begin(),points.end(),betterResponse)'),
    ('candidate search replenishes until valid quota', extractor, 'if(valid.size()>=size_t(geometryQuota)) break'),
    ('prepared warp and geometry travel together', header, 'std::vector<KeypointGeometry>* detectedGeometry'),
    ('default branch skips tile work', extractor, 'continue; // Default path never allocates/filters a per-feature tile.'),
    ('candidate support uses sparse backend', extractor, 'detail::sparseSupportValid(image.size(), invalidIntegral, centre, warp, pole)'),
    ('zero-initialized unit-bearing LUT', extractor, 'rayDirections_ = cv::Mat::zeros(size, CV_32FC3)'),
    ('zero-initialized projection-Jacobian LUT', extractor, 'projectionJacobians_ = cv::Mat::zeros(size, CV_32FC(6))'),
    ('generic CameraBase projection Jacobian', extractor, 'camera_->project(bearing, &projected, &jacobian)'),
    ('projection round-trip validation', extractor, '(projected - Eigen::Vector2d(x, y)).norm() > 0.25'),
    ('gravity tangent basis', extractor, 'gravityCamera_.cross(bearing)'),
    ('finite pole fallback', extractor, 'eu -= bearing * bearing.dot(eu)'),
    ('camera-aware warp uses virtual focal', extractor, 'jacobian * ev / virtualFocalLength_'),
    ('warp singular-value checks', extractor, 'detail::singularValues2x2(warp)'),
    ('analytic maximum singular value', sampling, 'std::hypot(a-c,2*b)'),
    ('stable minimum singular value', sampling, 'std::abs(determinant)/largest'),
    ('missing gravity explicit failure', extractor, 'Set ORB camera gravity before describing each frame'),
    ('affine Gaussian padding', sampling, 'kOrbTileRadius = kOrbPatchRadius + kOrbBlurRadius'),
    ('scaled Gaussian reference support', sampling, '2 * kOrbBlurRadius * scale + 1'),
    ('scaled Gaussian reference sigma', sampling, 'kOrbSigma * scale, kOrbSigma * scale'),
    ('warp transforms filter support', sampling, 'warp * Eigen::Vector2d(x * radius, y * radius)'),
    ('supersampling prevents stretched-source gaps', sampling, 'std::ceil(stretch - 1e-6)'),
    ('supersampled source coordinates', sampling, 'double(x) / scale, double(y) / scale'),
    ('floating point rectification tile', sampling, '2 * radius + 1, 2 * radius + 1, CV_32FC1'),
    ('canonical Gaussian instead of image preblur', extractor, 'detail::smoothRectifiedPatch(patch, filtered)'),
    ('canonical texture orientation at poles', extractor, 'detail::rectifiedTextureOrientation(patch, textureAngle)'),
    ('resampling source after pole rotation', extractor, 'warp = (warp * rotation(textureAngle)).eval()'),
    ('neutral pole basis avoids gravity normalization', extractor, 'gram.inverse() * Eigen::Vector2d::UnitX()'),
    ('unobservable pole direction rejected', sampling, 'std::hypot(momentX, momentY) <= 1e-6'),
    ('candidate masks validated before quotas', extractor, 'detail::warpedSupportValid(image.size(), invalidIntegral'),
    ('Jacobian accounts for actual resized width', extractor, 'jacobian.row(0) *= scale * levelWidth / width'),
    ('Jacobian accounts for actual resized height', extractor, 'jacobian.row(1) *= scale * levelHeight / height'),
    ('original-image pixel centre convention', extractor, '(pixel.x + 0.5) * sx - 0.5'),
    ('pyramid pixel centre convention', extractor, '(pixel.x + 0.5) / sx - 0.5'),
    ('compressed patch avoids fixed large erosion', extractor, 'const int radius = isCameraAware() ? 3 : kMaskRadius'),
    ('mask interpolation avoids uint8 validity rounding', extractor, 'previousMask.convertTo(floatingMask, CV_32F, 1.0 / 255.0)'),
    ('full patch mask query', sampling, 'cv::integral(invalid, integral, CV_64F)'),
    ('binary pair sampling', extractor, 'detail::kOrbPattern[bit]'),
    ('OpenCV ORB bit order', extractor, 'bits[bit / 8] |= static_cast<uchar>(1u << (bit % 8))'),
    ('retained feature/row alignment', extractor, 'result.rowRange(0, static_cast<int>(keypoints.size())).copyTo(descriptors)'),
    ('standard descriptor backend retained', extractor, 'orb->compute(gray, keypoints, descriptors)'),
    ('world/rig/camera rotation ordering', adapter, 'frame.T_SC(camera)->C().transpose() * T_WS.C().transpose() * gravityWorld'),
    ('reuse validates calibration pointers', adapter, 'extractors[camera]->camera().get() != frame.geometry(camera).get()'),
):
    check(name, fragment in text)
check('no concrete camera downcast in geometry', not re.search(r'(static|dynamic)_pointer_cast', extractor))
check('no positive-z ray restriction', not re.search(r'(bearing|ray)\.z\(\)\s*[<>]', extractor))
for test in ('missingAndInvalidGravity', 'gravityRollInvariance', 'poleFallbackAndOctaveScale',
             'projectionDerivativeAgreement', 'fullWarpedMaskSupport', 'rejectExcessiveStretch',
             'rearHemisphereEucm', 'cameraVariant', 'frameAdapterAndRepeatability', 'multiCameraGravityAndReuse',
             'defaultModeAndStandardOptOut', 'affineSmoothingAgreement', 'smoothingMaskAndCompressedSupport', 'canonicalPoleOrientation',
             'sparsePatternAndFractionalArea', 'sparseAndReferenceProfiles',
             'cachedExtractionAndDynamicMasks', 'sparseSamplerAndAnalyticWarpAgreement'):
    check(f'regression source: {test}', test in tests)
check('validity integrals have one shared construction site', extractor.count('detail::invalidMaskIntegral(') == 1)
check('public calibration-only preparation entry', 'void prepareStaticCaches();' in header)
prepare_body = extractor.split('void OrbFeatureExtractor::prepareStaticCaches() {', 1)[1].split('void OrbFeatureExtractor::buildPyramid', 1)[0]
check('startup builds masks from calibration domain', 'buildMasks(cameraValidMask_,pyramid)' in prepare_body)
check('startup static preparation is idempotent', 'if(staticMasksReady_) return;' in prepare_body)
check('startup does not build image pyramid or descriptors', not re.search(r'\b(buildPyramid|detectAndCompute|setExtractionDirection|integral)\s*\(', prepare_body))
check('prepared first-image cache hit has regression coverage', 'Prepared first image must reuse static masks' in tests)
check('detection reports FAST and geometric-filter timings', 'lastTiming_.fastMs+=' in extractor and 'lastTiming_.candidateFilterMs+=' in extractor)
check('FAST output allocations are reused between cells', 'auto& points=fastScratch; points.clear();' in extractor)
check('shared FAST retains ROI reference switch', 'bool layerFast = true;' in header and 'const bool sharedCandidates=options_.layerFast && threshold==options_.initialFastThreshold;' in extractor)
check('shared scan uses high threshold', 'cv::FAST(image,rawFastScratch_,options_.initialFastThreshold,false' in extractor and 'cv::FAST(image,maximaFastScratch_,options_.initialFastThreshold,true' in extractor)
check('low-threshold ROI work is counted separately', '++lastTiming_.fastFallbackCalls;' in extractor and 'fastFallbackMs' in header)
check('low contrast fallback has regression coverage', 'Low-contrast cells must retain the low-threshold ROI fallback' in tests)
check('empty grid mask cache follows static and external masks', 'pyramid.cellMasks=staticCellMasks_;' in extractor and 'pyramid.cellMasks.push_back(cells);' in extractor)
check('empty safe grids skip candidate work', 'if(!pyramid.cellMasks[level].at<uchar>(row,col))' in extractor)
check('empty-mask shortcut has regression coverage', 'Empty safe-mask cells must never request low-threshold FAST' in tests)
check('coarse fallback retains fine-grid high detection', 'processCell(row,col,false);' in extractor and 'coarseSupport[region(row,col)]==0' in extractor)
check('budget fallback counts per-cell selectable capacity', 'if(retained<options_.maxFeaturesPerCell) ++selectableCapacity;' in extractor)
check('legacy per-cell fallback remains selectable', 'bool budgetedFastFallback = true;' in header and 'if(options_.budgetedFastFallback) deferred.push_back(cell);' in extractor)
check('budget fallback has count and occupancy regression', 'budgetDrivenFallbackPreservesCoverage();' in tests and 'Budgeted fallback must retain the fine-cell occupancy limit' in tests)
check('shared FAST only reconstructs scores near cell boundaries', 'x-left<=1 || right-1-x<=1' in extractor)
check('cell-local strict NMS remains explicit', 'score<=scores.at<uchar>(yy,xx)' in extractor)
check('compact integral has an exact overflow guard', 'image.total()<=size_t(std::numeric_limits<int>::max()/255)' in extractor)
check('compact integral keeps double interpolation and typed loads', 'template<class Pixel> class BasicSparseBoxSampler' in sources['OrbSparseSampling.hpp'] and 'const double dx0=' in sources['OrbSparseSampling.hpp'])
check('layer FAST and integral reference regression is invoked', 'layerFastAndCompactIntegralAgreement();' in tests)

check('raw FAST boundary strips include the circle halo', 'core.x-3,core.y-3,core.width+6,core.height+6' in extractor and 'options_.boundaryFast' in extractor)
check('boundary intersections are deduplicated before local NMS', 'rawFastScratch_.resize(uniqueBoundary)' in extractor)
check('geometry batches retain a bounded replenishment path', 'geometryQuota<options_.candidatesPerCell' in extractor and '2*geometryQuota' in extractor)
check('geometry results and failures share pixel-indexed memo storage', 'geometryMemo.emplace_back()' in extractor and 'if(index>=0)' in extractor)
check('grouped and flat balancing have selectable reference paths', 'options_.groupedBalancing' in extractor and 'next[previous]=index' in extractor)
check('exact boundary FAST and grouped heap comparisons are present', 'Boundary-strip FAST or grouped heap changed exact selection' in tests)
check('geometry batch support and coverage regression is registered', '"adaptiveGeometryBatchesPreserveSupport"' in tests and 'Adaptive batches must retain spatial and spherical coverage' in tests)
check('VS project is valid XML', project.getroot().tag.endswith('Project'))
filters = ET.parse(project_path.with_suffix('.vcxproj.filters'))
check('VS filters are valid XML', filters.getroot().tag.endswith('Project'))
for path in files:
    if path.parent != FEATURES:
        continue
    tag = 'ClCompile' if path.suffix == '.cpp' else 'ClInclude'
    registered = [item.attrib.get('Include', '') for item in project.findall(f'.//ms:{tag}', namespace)]
    check(f'VS project registers {path.name}', any(item.endswith('\\' + path.name) for item in registered))
check('README documents gravity transform and default mode', 'R_SCi^T R_WS^T g_W' in (FEATURES / 'README.md').read_text(encoding='utf-8'))

failed = [name for name, passed in results if not passed]
report = [f'Static audit: {len(results) - len(failed)}/{len(results)} checks passed.',
          'Scope: source/API/configuration only. No C++ compilation or runtime tests.',
          f'ORB numeric point-pair SHA256: {pattern_digest}', '']
report += [f'{"PASS" if passed else "FAIL"}: {name}' for name, passed in results]
(FEATURES / 'static_check_camera_aware_report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
print('\n'.join(report[:3]))
for name in failed:
    print('FAIL:', name)
sys.exit(bool(failed))
