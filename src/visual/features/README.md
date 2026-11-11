# ORB 特征与重力 / camera-aware 描述子

`OrbFeatureExtractor` 继承 `cv::Feature2D`，兼容 OpenCV 4 的 `FeatureDetector` / `DescriptorExtractor` 别名和本项目 `Frame`。检测仍为每层网格 FAST、高低阈值补点、网格配额与 S² 分布均衡；新增的是描述子几何与采样。`OrbMatcher` 的 32 字节 Hamming、IMU 预测窗口、ratio/mutual 和球面几何接口保持可用。

## 模式与调用

默认启用 `GravityCameraAware`。构造时需要相机标定，每帧检测/描述前需要更新重力。`Standard` 需显式选择，可用于无重力初始化或对照；`GravityAligned` 可用于单独评估重力旋转对齐的贡献。

默认采样后端为 `SamplingMode::SparseBox`，优先控制描述子提取成本。逐特征整块 tile 的二维 Gaussian 仅保留为 `SamplingMode::AffineGaussianReference` 对照模式，需要时显式设置 `options.samplingMode`；它不是默认实时路径。

```cpp
#include <visual/features/OrbFrameAdapter.hpp>
#include <visual/Frame.hpp>

cgmli::features::OrbExtractorOptions options;
// 默认 options.descriptorMode 为 GravityCameraAware。
options.virtualFocalLength = 400.0; // 示例值：根据标定与期望角度支撑范围选择。
auto orb = cgmli::features::configureOrb(frame, options); // frame 已设置图像和 geometry。

// 每帧：T_WC 把相机坐标变换到世界坐标，gravityWorld 使用同一世界坐标约定。
cgmli::features::updateOrbGravity(*orb, T_WC, gravityWorld);
frame.detect();
frame.describe();
frame.computeBackProjections();

// 下一帧复用 LUT；newFrame 使用同一个标定对象。
cgmli::features::bindOrb(newFrame, orb);
cgmli::features::updateOrbGravity(*orb, newT_WC, gravityWorld);
newFrame.detect();
newFrame.describe();
```

多相机使用每相机独立的实例：

```cpp
#include <visual/MultiFrame.hpp>

auto extractors = cgmli::features::configureOrb(multiFrame, options); // 初始化一次。

// 每个新 MultiFrame：图像已经设置，geometry 指针和初始化时一致。
cgmli::features::bindOrb(multiFrame, extractors);
cgmli::features::updateOrbGravity(multiFrame, extractors, T_WS, gravityWorld);
for (std::size_t camera = 0; camera < multiFrame.numFrames(); ++camera) {
    multiFrame.detect(camera);
    multiFrame.describe(camera);
    multiFrame.computeBackProjections(camera);
}
```

`T_WS` 将机体/相机 rig 的 S 坐标变换到 W；`MultiFrame::T_SC(i)` 将相机 C_i 变换到 S。因此 `g_Ci = R_SCi^T R_WS^T g_W`，平移不参与。默认 `g_W=(0,0,-1)`，仅在世界坐标确实采用这个方向时使用默认值。传入 IMU 融合后的姿态/重力估计，不要直接把含运动加速度的加速度计测量当作重力。初始化阶段若无可靠方向，可显式使用 Standard；切换模式时需同步切换参与匹配的帧/地图描述子。

也可以直接调用 `orb->setExtractionDirection(gravityCamera)`。非普通模式没有设置有效方向时，检测和描述子计算都会抛出异常；不会注入一个假定的相机重力或静默退回普通 ORB。近重力极点的纹理方向回退仅处理方向不可观的局部特征。

## 几何与描述子约定

参考 [ethz-mrl BRISK 的切平面采样几何](https://github.com/ethz-mrl/brisk/blob/main/src/brisk-descriptor-extractor.cc#L602-L667)，使用单位射线 `b`、归一化重力 `g_C` 构造 `e_u=normalize(g_C×b)`、`e_v=normalize(b×e_u)`。由投影 Jacobian `J = ∂pixel/∂bearing` 得到 `W=J[e_u e_v]/f_virtual`，直接变形 ORB 点对。`keypoint.angle` 仅记录变形后 x 轴的图像方向，描述子不会再次应用该角度。

LUT 在构造时通过 `CameraBase::backProject()` 和 `project(..., &J)` 生成，并检查有限值、投影状态、反投影/投影一致性。无须具体相机类转换，也无须事先调用相机的 `initialiseCameraAwarenessMaps()`。兼容本项目 EUCM、Pinhole + NoDistortion / Equidistant / RadTan / RadTan8；有效域由相机模型决定，不额外要求 `bearing.z()>0`。射线和 J 进行四邻点插值；射线重新归一化。

重力与视线的叉乘模长小于 `gravityParallelSinThreshold`（默认 0.1）时，camera-aware 模式先用图像 x 轴建立有效中性切平面。默认稀疏路径使用该平面圆盘上的 29 个固定探针估计质心方向，再旋转采样矩阵计算描述子；不会为极点生成 tile。Gaussian 参考模式则在规范化 patch 上估计质心方向，再从原金字塔图像重新采样，避免二次插值。纹理方向不可观时丢弃该点。`samplingWarp()` 在极点只返回中性基矩阵，最终纹理旋转由 `compute()` 完成；普通重力对齐模式仍使用图像质心角度。

退化 Jacobian、过大的最大奇异值（默认 4，可配置范围 1～8）或条件数（默认 12）会使该点被丢弃。使用奇异值检查一般二维矩阵，不把包含旋转/剪切的 W 当作对称矩阵求特征值。

每个金字塔层的角度支撑按名义尺度 s 增长。原图 Jacobian 转到该层时，使用实际舍入后的宽高比例 `s_x=width/levelWidth`、`s_y=height/levelHeight`：`W_level=diag(s/s_x,s/s_y) J[e_u e_v]/f_virtual`，不再次除以 octave scale。重力模式的原图/层坐标遵循 resize 的像素中心约定 `(x+0.5)s_x-0.5`，修正了粗层的半像素和宽高舍入误差。普通模式保留 OpenCV ORB 的原有坐标约定。关键点仍以原图像素存储。`virtualFocalLength=0` 时读取本项目相机第一个内参 `fu`；建议参与跨相机匹配的实例显式使用同一个参考焦距，以统一同层采样的角度尺度。其他 CameraBase 若内参顺序不同，必须显式传值。

ORB 点对来自 [OpenCV 4.8.0 的 31×31 学习模式](https://github.com/opencv/opencv/blob/4.8.0/modules/features2d/src/orb.cpp#L346-L604)，保留上游 BSD 声明。默认稀疏描述子的流程为：

1. 保留完整 W 对采样位置的旋转、拉伸和剪切。将 512 个点对端点去重为 375 个位置；固定映射在编译期以线性查表方式生成。
2. 仅对本帧实际用到的金字塔层各生成一次灰度积分图，所有关键点共享。像素总数不超过 `INT_MAX/255` 时默认用精确的 CV_32S 前缀和，插值继续使用 double；更大图像自动回退 CV_64F。`compactGrayIntegral=false` 可强制 double 参考路径。不为每个特征生成 tile 或执行滤波。
3. 为每个唯一位置查询一次局部矩形平均灰度；支持亚像素边界的精确面积权重。矩形半宽为 `h_x=max(0.5,sqrt(3)*2*norm(W.row(0)))`，h_y 同理。均匀矩形的方差 h²/3 对应参考 sigma=2 经 W 变形后协方差的对角项。滤波保留两个图像轴方向的尺度，**近似掉非对角相关项，并用 box 代替 Gaussian**；不是完整各向异性 Gaussian。
4. 缓存 375 个 float 灰度值（1500 字节），复用这些值完成 256 个比较，低位优先打包 32 字节。极点只额外计算 29 个探针。

每个灰度查询为固定次数积分图访问，局部拉伸增大时只扩大矩形边界，不增加采样个数。代价转移为每层共享积分图和固定数量的查询，接近 BRISK 的稀疏平滑思路。BRISK 的位置数仍是 66，因此不能仅凭取消 tile 就断言提取比 BRISK 更快；ORB 的确定优势仍包括更短的 256-bit 匹配和存储格式。目标硬件尚未实测。

可选 `AffineGaussianReference` 保留较重的完整二维平滑流程：

1. 在切平面上定义 31×31 规范化 patch，增加参考 Gaussian 半径 3 的 padding，单位采样率时为 37×37。
2. 由 W 最大奇异值选择超采样倍率 `q=max(1,ceil(s_max))`（整数边界有 1e-6 浮点容差）。tile 大小为 `(36q+1)×(36q+1)`，原金字塔图像双线性采样为 CV_32FC1。默认最大拉伸 4 对应 q≤4，配置上限 q≤8。
3. 在规范化 tile 上执行 Gaussian 滤波，核宽 `6q+1`、sigma=`2q`。图像空间的名义平滑协方差为 `sigma_ref² W Wᵀ`；核有有限截断，并以离散超采样近似。位置和滤波偏移都经历同一个 W，包含局部旋转、拉伸和剪切。
4. 从平滑 tile 的整数参考坐标读取 ORB 点对，比较 `I(first)<I(second)`，低位优先打包为 32 字节 CV_8UC1。不重新应用关键点角度。

参考模式的平滑支撑采用完整二维变形，用来评估稀疏近似带来的质量差异；它不是默认实时方案。两个后端都不是 BRISK 描述子的逐位复现，相机几何仍为局部一阶模型。

默认路径的 mask 包围范围覆盖 31×31 参考区域经 W 变形后的采样位置，以及局部矩形平滑支撑；参考模式则覆盖变形 Gaussian padding。两者的积分 mask 查询都会拒绝包围范围内的无效像素。检测候选按所选后端检查几何和支撑，再计入 Top-K/配额和高低阈值补点。极点候选使用能覆盖任意纹理旋转的保守包围范围；描述时再检查最终变形。camera-aware 检测仅固定侵蚀 FAST 所需的 3 像素。被拒绝的关键点和描述子行同时移除，最终索引以 describe 结果为准。

默认配置由 `getDefaultName()` 标识为 `cgmli.ORB.GravityCameraAware.SparseBox.v3`（纯重力模式为 `cgmli.ORB.Gravity.SparseBox.v3`）。Gaussian 参考后端保留 v2 标识。不能把不同后端、v1、普通 ORB 和 v3 描述子混用；已有地图观测/词袋词典需按当前配置重新生成。参与匹配的观测应使用同一模式和一致的参考尺度配置。`Frontend` 的实际提取入口及描述子消费格式现已改为 ORB，具体接入和旧关联模块边界见 `src/frontend/README.md`。

## 缓存与 CUDA

标定和 camera mask 在构造时冻结；改变后应重新 configure。`rayDirections()`、`projectionJacobians()`、`geometryValidMask()` 返回可供上传 GPU 的只读 LUT，`samplingWarp()` 可供比较 CPU/GPU 每点基础几何；原始点对在 `OrbBriefPattern.hpp`，默认稀疏路径与去重表在 `OrbSparseSampling.hpp`，可选 tile 参考流程在 `OrbCameraAwareSampling.hpp`。同一相机的方向更新与提取不可并发，多个相机使用独立实例；使用新帧前更新方向。

构造提取器时生成相机 LUT；可在读取首张图像之前调用 `orb->prepareStaticCaches()`，生成各层静态检测掩码、采样掩码和无效像素积分图。此调用只需要标定，不需要图像或重力，可重复调用。工程的 `ThreadedSlam` 已通过 `Frontend::prepareOrb()` 在启动阶段自动完成这两部分工作。直接使用 OpenCV 风格接口时，也可以不显式预计算，保留首次提取时建立静态掩码的行为。

CPU 提取路径采用以下复用边界：

- 构造阶段合并相机有效域与几何 LUT 的反投影遍历；保留投影、Jacobian 和回投一致性检查。
- 无外部 mask 时，金字塔各层的静态 mask、侵蚀结果和无效像素积分图只生成一次。传入外部 mask 时逐次重建，不覆盖静态缓存。图像尺寸变化会使层布局缓存失效；有标定的实例仍要求图像尺寸与标定一致。
- 每格 FAST 候选先按响应排序，再从强到弱检查几何。默认先取得 `min(candidatesPerCell, max(2, targetPerCell))` 个有效候选；容量不足时按倍数补到原来的 `candidatesPerCell` 上限，或遍历完毕。无效点会继续补位，高低阈值补点、网格配额和 S² 均衡继续生效。
- 同层的 FAST 输出缓冲区在不同网格和高低阈值之间复用，候选容器预留空间；`fast_ms / candidate_sort_ms / candidate_filter_ms / balance_ms` 用于进一步定位检测开销，均包含在 `detect_ms` 内。
- 默认 `layerFast=true`、`boundaryFast=true`：每层保留一次整图高阈值 FAST+NMS，用于网格内部；无 NMS 的原始角点仅在网格边界及其邻点的条带上扫描。条带带有 3 像素 FAST 圆周 halo，交叉位置去重，重新计算边界 FAST-9 分数并恢复逐格严格 NMS。`boundaryFast=false` 回到以前的“整图 raw + 整图 NMS”两次扫描；`layerFast=false` 是逐格 ROI 参考。低阈值 ROI 仍按几何配额与覆盖缺口请求。条带版本调用次数可能更多，需看扫描像素量和时间，不能按 FAST 调用次数判断工作量。
- 默认 `adaptiveGeometryCandidates=true`：各格只准备一小批已经通过完整几何与支撑检查的候选，容量不足时逐批扩大，再按需补低阈值。以每层像素索引和连续数组复用本次成功/失败结果，进入下一层或下一帧会失效；低阈值 FAST 结果也在本层复用。所有最终点仍通过原门限，但候选池缩小可能改变最终位置和 S² 覆盖，需独立回放检验。设为 false 恢复原每格完整候选池，便于单变量比较。
- 默认 `groupedBalancing=true`：同一 `(bearing bin, image cell)` 仅把响应最强的未选候选放进堆，选中后推进该组。组内候选的占用键相同，保留原 bin/cell/response/index 比较顺序；减少同组全部候选的重复堆刷新。设为 false 使用原 flat heap，回归源码对照点序、响应、octave、角度和描述子位。
- 同一次 `detectAndCompute()` 将候选的 warp、插值单位射线的方向分桶和层坐标传给描述阶段，避免重复计算。分桶与 camera-aware 采样使用同一 LUT 射线；相较逐点精确反投影，极少数分桶边界点的选择可能变化。单独调用 `compute()` 会根据当前重力重新计算，不复用前次检测的 warp。
- 二阶矩阵的奇异值改为解析计算，仍检查最大拉伸与条件数。稀疏 box 查询复用半宽、面积和行指针，四个边界的 floor 各计算一次；保持原分数面积插值和位比较方式。
- 图像金字塔与灰度积分图复用内存；每张新图像仍重新生成所用层的灰度积分值，不能跨帧缓存灰度内容。
- 静态掩码预计算同时生成各层网格的合法中心标志，完全无合法中心的网格跳过检测和低阈值补点；整层为空时连共享 FAST 扫描也跳过。外部动态掩码生成独立标志，不污染静态缓存。不会根据高阈值无点就判定弱纹理区域无效。
- 默认 `budgetedFastFallback=true`：所有细网格仍检测高阈值候选，低阈值先补充没有候选的粗分区；粗分区数量由该层最终特征预算决定。若按细格上限计算的可选容量仍不足预算，再按固定顺序补点，直到容量足够或候选网格耗尽。细格上限、几何/支撑检查、S² 均衡和最终预算继续生效。该策略会改变特征选择，需要检验覆盖与跟踪质量；`budgetedFastFallback=false` 保留逐细格补点策略，与 `layerFast` 的共享扫描开关相互独立。

这些改动保留当前 8 层配置和默认 `GravityCameraAware`，也保留 375 个唯一采样位置、256-bit 格式及 v3 描述子定义。没有通过减少层数或采样位数获得加速；实际耗时需要运行日志验证。

前端 `[Timing][Extract]` 在 `pyramid_ms / detect_ms / describe_ms` 外增加以下计数：

| 字段 | 含义 |
| --- | --- |
| `mask_cache_hit` | 启动预计算完成后首帧即为 1；未预计算实例的首次提取及外部 mask 路径为 0 |
| `fast_candidates` | 各格高低阈值 FAST 返回的原始候选总数，包含重复和后续拒绝点 |
| `geometry_evals` | 实际执行采样几何计算的次数，包含最终未入选的候选 |
| `warp_reused` | 描述阶段复用本次检测几何的次数，可能大于最终描述子行数；分开 detect/compute 为 0 |
| `gray_integrals` | 本次重新构建灰度积分图的层数；SparseBox 模式通常为有描述子的层数 |
| `fast_calls / fast_fallback_calls` | OpenCV FAST 总调用数 / 共享模式下低阈值 ROI 补点调用数；二者之差为共享高阈值调用数；条带版本不再固定为两次 |
| `fast_raw_scan_ms / fast_nms_scan_ms` | 共享高阈值原始角点扫描 / 带 NMS 扫描的耗时；后者包含扫描和 OpenCV 内部 NMS |
| `fast_boundary_ms / fast_bucket_ms / fast_fallback_ms` | 网格边界分数恢复 / 分桶及边界 NMS / 按需低阈值 ROI 扫描耗时；均计入 `fast_ms`，清零与管理开销也包含在 `fast_ms` |
| `fast_raw_corners` | 共享高阈值扫描的未抑制角点数，区别于高低阈值各格候选累计数 `fast_candidates` |
| `empty_mask_cells / quota_fallback_cells` | 因没有合法中心而跳过的网格数 / 实际进入低阈值补点的网格数 |
| `zero_valid_with_candidates_cells` | 进入补点且高阈值有候选、但没有候选通过掩码/几何/支撑筛选的网格数；不等同于整格必定无效 |
| `budget_skipped_cells` | 原本细格配额不足，但因粗分区已有支持且全层可选容量已足够而未执行低阈值补点的网格数 |
| `gray_integrals32` | 使用精确 int32 前缀和的层数，计入 `gray_integrals`；1280×800 全白图的最大前缀和为 261120000，可安全使用 |

VS x64 工程检测到 CUDA 12.6 构建组件后启用 CUDA 提取。默认的 `GravityCameraAware + SparseBox` 联合提取可在设备端生成灰度金字塔、FAST-9 分数和逐格 NMS 候选，并计算积分图、重力极点纹理方向及描述子。每相机复用独立 stream、显存和页锁定缓冲。CPU 复用原有几何、网格配额与球面均衡规则，最终关键点和 32 字节描述子按原顺序写入 Frame。动态外部掩码、其他提取模式或 CUDA 初始化及执行失败时使用 CPU 路径。

可在启动程序前用 PowerShell 设置 `$env:CGMLI_ORB_BACKEND = 'cpu'` 运行同一二进制的 CPU 对照组，恢复默认用 `Remove-Item Env:CGMLI_ORB_BACKEND`。只构建 CPU 版本时使用 MSBuild 参数 `/p:CgmliCudaOrb=false`。`[Timing][Extract]` 的 `backend` 和 `gpu_*_ms` 分别记录实际执行路径和设备阶段耗时，四目总耗时看 `[Timing][ExtractWall]`。

CUDA 提取默认在候选回传后立即提交积分图，四路使用各自的 stream，CPU 同时执行几何筛选和配额分配。将 `frontend_parameters.orb_cuda_early_integral` 设为 `false` 可在描述阶段按实际使用的金字塔层提交积分图，用于对照。

将 `frontend_parameters.orb_cuda_matcher` 设为 `true` 可试用时序与地图引导匹配的 GPU 路径。GPU 遍历候选、检查几何、计算 Hamming 距离并归约双向前两名，CPU 应用原有比例检验和轨迹归属规则。GPU 不可用、计算失败或候选接近几何门限时回到 CPU。`[Timing][Frontend]` 的 `cuda_jobs` 记录实际在 GPU 完成的任务数。

CUDA 匹配将每个任务的引导数据打包到一块页锁定缓冲，跨帧按描述子内容复用显存。通过几何筛选的候选进入有容量上限的稀疏列表，超过容量时由 CPU 完成该任务。双向归约先找最优，再排除同一身份找次优。健康跟踪中，可在两路相机的地图及时序支持均充足时延后其同帧匹配，覆盖不足、恢复和关键帧仍可执行完整搜索。

## 检查

静态检查命令：`python src/visual/features/tools/static_check_camera_aware.py`（Windows 可使用本机 Python 的绝对路径）。它检查源码结构、局部 include、上游采样表、接口声明/实现和 VS 工程 XML；不会编译或运行 C++。

`tests/OrbCameraAwareRegression.cpp` 提供后续可构建的回归用例源码，覆盖默认稀疏/显式参考模式、375 点映射、分数面积积分与逐像素求和一致性、重力缺失/非法输入、90°旋转、极点方向、金字塔尺度、投影数值导数、参考模式仿射滤波、mask、异常拉伸、EUCM 后半球、畸变、Frame 和多相机接口。缓存回归另覆盖冷/热提取一致性、动态 mask 与静态缓存隔离、分开 detect/compute、重力更新、图像更新、带行跨度的积分图采样及解析奇异值与 SVD 对照。按要求仅做静态检查，运行时用例未执行。


### 2026-11-04 v11 CPU 对照

Hilti YAML 新增可选 `orb_boundary_fast / orb_adaptive_geometry / orb_grouped_balancing`，均默认 true；旧 YAML 缺少这些字段可继续读取。`[Timing][OrbPrepare]` 记录实际选择。保持 8 层、800 点和当前 GravityCameraAware/SparseBox 描述子定义。

新增日志：`fast_raw_pixels` 是 raw 扫描面积（含 halo 和条带交叉重复），`fast_raw_corners` 是去重前枚举数量，`geometry_memo_hits` 是本层几何成功/失败复用次数，`geometry_expansion_rounds` 是容量补充轮数，`balance_heap_pushes` 包括初始堆条目和后续更新。`fast_candidates` 可包含批量补充时重新遍历的候选，不能当作唯一角点数。

`layerFastAndCompactIntegralAgreement` 扩展比较逐格参考、整图 raw、条带 raw、flat/grouped heap 的完整输出；`adaptiveGeometryBatchesPreserveSupport` 单独比较点数、空间/S² 覆盖和动态掩码，并输出 `adaptive_geometry_batches.csv`。这些是新增回归源码，尚未编译或执行。


### 2026-11-05 v12 描述子坐标批处理

`batchDescriptorCoordinates=true`（YAML `orb_batch_descriptor_coordinates`）使用固定2×375坐标矩阵批量生成采样像素，复用原375个灰度值和256bit比较规则；积分查询/支撑/重力与camera-aware几何保持原接口。设false使用原逐坐标路径。新 `batchedDescriptorCoordinatesMatchScalar` 对 int32/double 积分图及旋转/压缩warp逐字节比较描述子，源码已接线，尚未执行。

Hilti配置将 `orb_boundary_fast` 改为false，恢复整图raw+NMS；保留几何分批和分组堆，用于检查上次条带扫描的耗时回退。8层、800点及采样模式保持原设置。
