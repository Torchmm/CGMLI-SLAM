# Frontend ORB integration

`Frontend::detectAndDescribe()` 已接入 `OrbFrontendFeatures`，默认使用 `GravityCameraAware + SparseBox`。每相机独立实例、独立互斥锁。`ThreadedSlam` 在启动阶段预先创建 LUT 和各层静态掩码，全部完成后再报告 `VIO Ready`；每帧提取时更新该相机的重力方向。独立调用 Frontend 时也保留首次使用时创建缓存的路径。

## 调用

```cpp
#include <frontend/Frontend.hpp>

cgmli::features::OrbExtractorOptions options;
options.maxFeatures = 1000;
options.virtualFocalLength = 0.0; // Frontend: one shared reference, selected from camera-0 fu.
cgmli::Frontend frontend(cameraSystem.numCameras(), options);
// Optional for standalone callers; ThreadedSlam already does this at startup.
// No image, gravity direction or backend state is needed.
frontend.prepareOrb(cameraSystem, true);

// frame has calibrated geometry and images; T_WCi maps camera i into the world.
// The world convention must use gravity direction (0,0,-1).
for (size_t camera = 0; camera < frame->numFrames(); ++camera) {
    frontend.detectAndDescribe(camera, frame, T_WCi[camera], nullptr);
}
// Rows, keypoints and cached backprojections now use the FINAL retained indices.
```

不同相机可以由不同工作线程调用，但一次提取过程中不能修改 Rig 标定、相机 mask、图像或配置。`T_WC` 已含相机外参，不可再乘一次 `T_SC`。有外部关键点时，第 4 个参数可传关键点列表；仍会检查几何、mask 和采样支撑，并同步删除不合法的点及描述子行。

内部直接调用一次 `detectAndCompute()`，检测与描述共享金字塔；不再顺序调用 `Frame::detect()`、`describe()` 重建两次金字塔。提取完成后才写入最终关键点、描述子和 backprojection。

## 容量与视觉更新连续性

新种子上限为 `min(maxNewLandmarks, 后端剩余地标名额 + 本批可回收种子数, 种子接纳剩余名额)`，观测预算还要计入每个新种子的锚观测及新增视图。前端先给已有轨迹分配当前观测，再创建新种子；地标名额为零时仍可提交已有轨迹观测，不会因固定新增 200 点超过容量而回滚整批视觉更新。

回收不在选定来源帧中的未初始化种子，或按数据时间戳已失跟超时的纯种子。后端独立检查它没有优化参数、live factor、已吸收观测或当前帧观测；成熟地图点仍须走后端 Schur 退役流程。种子回收与新关联在同一事务内完成，后续错误会一起回滚。观测总预算耗尽时也不会越界添加；这不替代正常滑窗和地图生命周期管理。

事务失败会打印 `[Frontend] Commit rejected`，包含真正的 `pose_consensus`、`pose_inliers`、批次大小和拒绝原因；上层转为 IMU-only 时也会打印原因。不能将未提交报告中的默认 `inliers=0` 直接当作 RANSAC 零内点。

## 关联耗时与复用

数据准备只为已初始化地图点计算一次世界坐标，不复制整条观测集合；导出 Frame 时也避免构造用不到的地标观测列表。三角化在本次调用内缓存每个观测的世界射线、每台相机的位姿，仍保留所有视图的半射线、深度、视差和角度检查。前端原子提交保留外层窗口副本与最终一致性验证，内部直接更新该私有副本；多个深度初始化只在批末统一扫描 pending factors。

无深度的球面匹配按已有 octave 门限建立训练点分桶，省去必然会被尺度门限拒绝的候选访问；不缩小球面几何域、不修改 ratio/mutual 或第二邻居规则。有限深度预测仍使用像素网格。

特征射线现在由 `Frame` 的只读共享缓存持有；后端冻结、导出及关联工作副本保留该缓存，重设关键点或替换相机对象只使当前副本失效。缓存不保存会随优化变化的位姿或世界射线。每次 proposal 为各相机建立一次连续描述子布局，并为当前图像建立一次像素网格、尺度索引；匹配任务共享这些只读数据，观测/已吸收状态及重叠 mask 仍逐任务检查。

默认 `orb_staged_matching: true` 使用以下流程：先按地标向各当前相机投影，并从所选历史来源中按视角与距离变化选最多三份描述子；多个描述子共享地标身份，forward/reverse ratio 的第二邻居采用不同当前特征/不同地标。描述子歧义在完整投影搜索窗口内判断，选中的地图观测再检查半射线和角度误差。地图支持不足时保留完整旧相机对搜索，不屏蔽成熟点查询。

地图可靠性逐目标相机判断，默认该相机至少 10 个有效地图匹配才跳过其成熟点的重复历史搜索。先匹配最近有效帧的同相机轨迹与同帧重叠相机；某目标相机初步支持不足 40 点或不足两个图像分区时，只为该目标追加同相机历史及最新来源的跨相机交接任务；仍不足才追加该目标的其他历史跨相机任务。旋转超过 0.10 rad 时仍恢复全部剩余任务。不使用同帧 FOV 表裁剪跨时刻相机组合。分阶段方案若未通过位姿共识/IMU 检查，Frontend 在提交前用 `stagedMatching=false` 重建完整参考候选集再求解，后端始终只接收一次事务。此策略属于关联调度调整，需要实测跟踪质量。

各任务复用不可变的训练索引，再按该任务的重叠掩码、启用标志和有效射线生成局部过滤视图，避免每个 query 遍历无效训练点。原始行号和第二近邻/反向比值检验保留，共享索引不被修改。`candidate_tests` 现在排除了这些无效访问，因此与旧版本计数比较时应同时看 Hamming 数和最终内点。

`orb_staged_matching: false` 可用于完整相机对参考路径的对照；`Options::mapDescriptorsPerLandmark` 和 `minimumStageMatchesPerCamera` 可通过关联配置接口调整。地图优先和来源调度属于关联策略调整，需要数据集回归；这两个路径保持同一描述子格式。

`[Timing][Extract]` 新增 `fast_ms`、`candidate_sort_ms`、`candidate_filter_ms`、`balance_ms`，分别覆盖 FAST、原始候选排序、候选有效域/几何/采样支撑筛选、全层网格与球面均衡选择。未单列的逐格合并、去重与管理开销仍包含在 `detect_ms`。

`[Timing][Frontend]` 新增以下字段：

| 字段 | 含义 |
| --- | --- |
| `prepare_frames_ms / prepare_map_ms / prepare_rays_ms` | 来源帧准备、有限地图点及种子预算准备、射线/相机位姿缓存准备 |
| `match_guidance_sum_ms` | 各匹配任务建立引导及有限点投影的耗时之和 |
| `match_setup_sum_ms` | 各任务匹配器校验、射线、描述子布局、像素网格及尺度分桶准备耗时之和 |
| `match_traverse_sum_ms` | 各任务候选遍历、门限及融合 Hamming 比较耗时之和 |
| `match_select_sum_ms` | 各任务 ratio/mutual/唯一性及结果整理耗时之和 |
| `predicted_queries / unpredicted_queries` | 有像素预测和无深度球面搜索的有效 query 任务数；同一特征可能参与多个任务 |
| `new_track_limit / capacity_skipped / retired_seeds` | 本批新种子上限、因批次/总预算不足跳过的创建候选数、回收的未初始化种子数 |
| `staged / broad_fallback / pose_fallback` | 地图优先路径是否参与、是否扩大搜索、是否因共识失败重跑完整参考匹配 |
| `map_jobs / seed_jobs` | 有效地图投影任务数、实际调度的帧间/同帧相机对任务数；位姿回退时累计两次尝试 |
| `expansion_jobs / broad_jobs / fallback_reason` | 相机局部扩展任务数 / 广搜任务数 / 原因：支持充分、相机扩展、相机覆盖不足、旋转或位姿共识失败 |

`[Match][Stages]` 另外记录旋转角和 `camera_map_ready_primary_sectors_expanded`：分号分隔每台相机，格式 `相机号:地图匹配数/地图可靠标志/初步支持数/初步分区位掩码/扩展后支持数`。支持数为候选端点去重数量，尚未经最终位姿验证；转弯直接广搜时扩展后计数显示 `-`。原因标签分别为 `sufficient_support / camera_expansion / camera_coverage / rotation / pose_consensus`。

`*_sum_ms` 是各任务的经过时间之和；并行时可超过 `match_ms` 墙钟时间，不能作为顺序步骤相加。候选门限和 Hamming 在同一循环执行，遍历计时包含二者，未引入逐候选时钟查询。

绝对位姿输入直接复用关联提案已求出的有限三维点与帧的反投影缓存，避免为 PnP 再导出整张地图及观测列表；提议观测的权限检查使用键/地图点索引，保持原始对应点顺序。RANSAC、内点映射、独立地图点门限和 IMU 一致性检查保留。`[Timing][Pose] kind=absolute` 分别记录 `prepare_ms / solve_ms`，并输出迭代、精化及生成模型数量，用于区分数据准备和 PoseLib 求解本身的波动。

启动准备在并行相机工作之前，通过 `std::call_once` 用局部合成图像调用三次 OpenCV FAST，覆盖高阈值原始/抑制及低阈值路径，并输出 `[Timing][FastWarmup]`。该图像不会进入 Frame、生成描述子或消耗状态 ID；校准与掩码准备保持原职责。预热针对首次库调用开销，不能保证消除所有调度、调试器或缺页引起的首帧波动。

启动预计算使用与真实帧相同的相机对象和共同参考焦距；`parallelise_detection` 同时控制各相机预计算是否并行。启动时不生成特征、不构造灰度金字塔、不估计重力，也不消耗帧编号或向后端插入状态。重复调用 `prepareOrb()` 可复用已有缓存；配置须在预计算之前完成，变更配置仍需先清理前端与地图。

日志 `[Timing][OrbPrepare]` 逐相机记录 `geometry_ms`（提取器构造及 LUT）、`masks_ms`（各层静态掩码与无效像素积分图）和 `total_ms`；`[Timing][OrbPrepareWall]` 记录全部相机的墙钟时间。这些耗时计入应用的 `[Timing][Startup]`。首张真实图像仍打印 `[Timing][Extract]`，无外部动态 mask 时应看到 `cold_ms=0 mask_cache_hit=1`。图像金字塔、灰度积分图和描述子仍随真实图像生成，因此首帧并非完全没有内存分配开销。

## 不同相机模型与参考尺度

每个相机传入自己的 `geometry(camera)`；射线和 Jacobian 由其虚接口生成，兼容本项目 Pinhole/畸变模型及 EUCM。无 BRISK 类型转换，也不依赖 `getCameraAwarenessMaps()` 的预初始化。

独立使用 `OrbFeatureExtractor` 时，`virtualFocalLength=0` 表示该相机自身 `fu`；通过 **Frontend** 使用时，0 表示整个 Rig 统一取 C0 的 `fu`，显式传给所有相机。相机 1 先于相机 0 初始化，也会取同一个参考。正值表示调用者指定的共同参考焦距。这样可以统一跨相机同层描述子的角度支撑，同时保留各自 Jacobian 引起的局部采样形变。

内参顺序不同于本项目 Pinhole/EUCM 的外部模型，应显式设置参考焦距。活动会话不允许悄悄替换相机模型对象；变更标定、mask 或描述子采样配置时，停止工作线程，清理 frontend 和后端/地图中的旧描述子，再重新配置。

## 描述子、匹配及词袋

- 活动前端输出 `CV_8UC1`、每点 32 字节的 256-bit ORB。地图描述子池、跨帧/跨相机匹配及回环验证的旧 48 字节拷贝、步长和 Hamming 调用均已改成 ORB 长度。
- `OrbFrontendFeatures::hammingDistance()` 固定读取 32 字节，使用无对齐要求的拷贝和位计数；不再调用 BRISK 的 `PopcntofXORed(...,3)`。输入图像描述子使用 Mat 行指针，允许非连续矩阵。
- `setOrbMatchingThreshold()` 单位为 256-bit Hamming 距离，默认 64，合法范围 `[0,256]`。地图、时间及同帧跨相机关联统一调用 `OrbMatcher::matchGuided()`：几何候选过滤后执行 Hamming、ratio（默认 0.8）、mutual 和反向 ratio。默认拒绝仅一个候选、无法计算 ratio 的情况；有可靠窄搜索区域时可通过 `setAssociationOptions()` 显式启用 `matcher.allowSingleCandidate`。本地地点验证仍使用描述子最优候选、唯一键冲突消解及 PoseLib 几何验证。
- 回环描述子离散度按位数归一化，避免直接对 256-bit ORB 使用原来 384-bit 的累加阈值；该启发式仍需数据集验证。

每帧记录 `descriptorProfile(camera)`，包括描述子版本/模式、采样后端、实际共同参考焦距、金字塔 scale 和重力极点阈值。旧帧/地图进入匹配前，检查行数、类型、长度、相机对象和 profile；仅有 32 字节长度相同还不足以混用普通 ORB、旧版本和 camera-aware ORB。

`Frame::resetDescriptors()`、`resetKeypoints()` 会清除旧 profile。Frontend 写入描述子后重新设置，后端 `FrameStore` 的冻结副本和导出帧保留它。外部替换描述子必须由生产者重新提供真实 profile，不能只修改标签来冒充兼容。

DBoW 适配改为 `OrbDatabase` / `DBoW2::FORB`，特征类型为 `vector<cv::Mat>` 的单行副本。默认没有词袋时跳过相关查询。可在关联线程调用：

```cpp
std::string error;
bool loaded = frontend.loadOrbVocabulary(filename, frame->descriptorProfile(0), error);
```

文件采用本机 DBoW2 支持的 OpenCV FileStorage vocabulary 格式，并在根节点增加 `descriptor_profile` 字符串，与实际训练描述子的完整 profile 一致。加载时检查每个节点包含 32 字节；失败不替换当前数据库。需要用相同采样配置训练词袋；旧 BRISK 和普通 ORB 词袋不自动复用，也不自动下载替代文件。旧 pipeline 构造函数的目录参数不再自动加载 BRISK 词袋。

## 配置

`FrontendParameters` 与配置读取器增加以下可选项，未提供时使用下面的默认值。已有配置中的其他必要字段仍需保留。

```yaml
frontend_parameters:
  orb_pyramid_levels: 8
  orb_scale_factor: 1.2
  orb_fast_threshold: 20
  orb_layer_fast: true
  orb_staged_matching: true
  orb_cuda_matcher: false
  orb_cuda_early_integral: true
  orb_min_fast_threshold: 7
  orb_virtual_focal_length: 0.0
  orb_matching_threshold: 64
```

`max_num_keypoints` 继续控制每相机特征数。pipeline 的前端配置入口已改为 `setOrbOptions()` 和 `setOrbMatchingThreshold()`；旧 `detection_threshold`、`absolute_threshold`、`octaves`、`matching_threshold` 不再映射成 FAST/Hamming 参数。避免将 Harris 分数或均匀化半径当作 FAST 阈值。

## 位姿求解接入

`Frontend.cpp` 的 OpenGV adapter、RANSAC 和旧回环 Ceres 精化均已替换为 `visual/pose_solver` 接口：

| 前端入口 | 当前算法 / 返回值 |
| --- | --- |
| `runRansac3d2d()` | `AbsolutePoseSolver`：GP3P，同中心最小样本回退 P3P；返回 `T_WB` |
| `runRansac2d2d()` | `RelativePoseSolver`：广义六点，同中心数据回退五点；转换为当前 `T_WB` 提案 |
| 纯旋转备选 | `RotationPoseSolver`：Wahba SVD 三对方向假设，平行方向退化检查；仅返回旋转 |
| `verifyRecognisedPlace()` | 旧帧世界点转换到旧 Body 坐标，再调用 `AbsolutePoseSolver`；得到 `T_Bold_Bnew` 和条件信息 |

三个求解器都使用 PoseLib 2.0.5 的通用 LO-RANSAC 主循环、局部优化调度和停止条件，适配器执行球面角评分、多内点 Huber LM 精化及秩检查。前端不另写 RANSAC 循环。几何门限现在是弧度，不能直接沿用旧 OpenGV 的 9/16 数值。

射线原点/方向由实时 `T_BC` 与各相机虚接口产生，支持 Pinhole、EUCM 和有效后半球方向。求解器接口中的 S 是通用射线参考系名称；本适配层传入 Body B 数据，输出因此为 Body 位姿，支持 `B != S`。回环函数保留历史参数名 `T_Sold_Snew`，实际输出约定为 `T_Bold_Bnew`；外部使用 IMU S 坐标的图模块需相应变换位姿及信息矩阵。

每条对应关系同时存储原始 `{frameId,cameraIndex,keypointIndex}` 与 landmark ID，内点索引严格映射回这张表。无效射线、未初始化点和无穷远 3D 点不会进入绝对位姿问题。以当前帧为锚的地标不用于重置它自己的位姿，避免把随该位姿变化的世界点当成独立地图约束。外部帧的关联还必须与后端权威观测表一致。

相对位姿按 landmark ID 汇总所有相机的公共轨迹，每条轨迹选择一对独立观测，并均衡可用相机组合，包含跨相机轨迹；不再按相机分别跑单目 RANSAC，也不把同一轨迹展开成多个独立点增加内点数。只对实际参与求解的观测进行外点判断；重复相机观测不凭空标记为内点或外点。

单中心五点结果的平移为单位相机中心基线，尺度不可观。只有非旋转模型且 `translationScaleObservable=true` 时才用 `T_WBold*T_BoldBnew` 初始化完整位姿；否则保留预测的世界位置，仅更新旋转。`lastAssociationReport().metricPose` 对纯旋转和非米制中心模型均为 false，不能据此完成米制初始化或时间三角化。纯旋转与相对位姿均失败时不产生位姿初值。

默认 3D–2D 至少 10 个内点且内点率不小于 0.7；相对/旋转至少 11 个内点且内点率不小于 0.8。旋转模型在内点数占优或内点率超过 `rotationDominanceRatio` 时优先，用于保守处理低视差。可在关联线程配置：

```cpp
cgmli::frontend_pose::Options pose;
pose.solver.angularThreshold = 0.01; // radians
pose.solver.ransac.max_iterations = 1000;
pose.solver.ransac.min_iterations = 50;
pose.minimumAbsoluteInliers = 10;
pose.minimumRelativeInliers = 11;
pose.loopPixelSigma = 1.0; // octave-zero pixel noise, not ORB patch size
frontend.setPoseSolverOptions(pose);
```

前端对应关系按身份排列，没有质量排序，因此这条接入路径禁止 `progressive_sampling`（PROSAC）；默认使用 PoseLib 的普通随机采样。IMU/预测位姿只作为评分候选，不是额外硬约束。

回环匹配先解决“同一当前关键点匹配多个旧地标”的冲突，要求足够的不同地标，并检查 ORB 离散度。PoseLib 精化后，通过每相机反投影 Jacobian 传播像素噪声，计算球面切空间白化残差及左 SE(3) 数值 Jacobian，输出 `H=ΣJᵀJ`。金字塔噪声按 `loopPixelSigma*scaleFactor^octave` 增长。信息矩阵坐标为输出位姿的 LEFT `[rho,theta]`；这是固定地标/固定外参下的条件信息，不含地图点、标定及旧帧位姿不确定性。退化时拒绝，不向 H 添加人为阻尼；失败不覆盖调用者原来的位姿和 H。

当前完整关联路径使用 `ViSlamBackend::applyFrontendBatch(id,batch,optional<T_WB>,visualHealth,keyframe,error)`：在候选窗口内设置位姿和视觉健康、批量创建地标/观测、初始化逆深度、设置关键帧，再校验提交；LiDAR 健康保持原值。拒绝历史帧或已发布帧，以及历史观测删除/合并。共识函数只读后端，在尚未提交的计划上剔除候选外点，避免先污染图再清理。失败不修改后端和调用者的关联。旧的 `applyFrontendUpdate()` 仍保留给显式管理已有 live 观测的调用者；它同样不能覆盖历史状态或删除已吸收证据。

旧帧位姿/特征必须仍可从后端取得。当前滑窗不提供全局归档或完整回环图；已滑出帧的验证会失败，不用单位变换冒充缺失旧位姿。
## ViSlamBackend 调用顺序

`Estimator` 是 `ViSlamBackend` 的别名。`Frontend.cpp` 已移除旧 `keyFrames()`、`imuFrames()`、`mergeLandmarks()`、`setLandmark()`、`optimiseRealtimeGraph()` 及 full-graph 调用，使用当前滑窗和批量观测 API。实际顺序如下；示例中的 prediction、imu、params、T_SL 和 T_BS 由上层提供：

```cpp
cgmli::ViSlamBackend estimator;
std::string error;
if (!estimator.initializeRig(cameraSystem, T_SL, error, 0.0, 0.0, T_BS))
    throw std::runtime_error(error);

// For each frame: prediction.id/time must equal frame.id/time.
for (size_t c = 0; c < frame->numFrames(); ++c) {
    const auto T_BC = estimator.numFrames()
        ? estimator.extrinsics(estimator.currentStateId(), c)
        : T_BS * (*cameraSystem.T_SC(c));
    frontend.detectAndDescribe(c, frame, prediction.T_WB * T_BC, nullptr);
}
if (!estimator.addStates(*frame, prediction, imu, error))
    throw std::runtime_error(error);
bool asKeyframe = false;
if (!frontend.dataAssociationAndInitialization(estimator, params, frame, false, &asKeyframe))
    throw std::runtime_error(frontend.lastAssociationReport().message);
const auto& visual = frontend.lastAssociationReport();
// visual.committed: ingestion succeeded, including bootstrap/seed-only frames.
// visual.trackingAccepted: geometric pose consensus succeeded.
// visual.metricPose: the accepted visual pose has metric translation.
// Supply actual IMU/LiDAR factors here before optimizing.
const auto step = estimator.optimize();
// Only step.accepted publishes a snapshot and may marginalize an epoch.
```

必须先完成提取再 `addStates()`，后端冻结关键点和描述子。关联时检查 ID、时间戳、相机对象、profile、每个关键点属性及描述子内容；不能在冻结后过滤/重排/重算特征。成功提交后，调用者 `frame` 的 ID/世界点缓存从后端只读输出同步；这些缓存不拥有后端状态，优化后需要重新取得 `multiFrame()` 或快照。

`FrontendAssociation` 在最多 4 个最近有效视觉帧和当前 Rig 内生成候选，跨全部相机组合匹配，利用在线 `T_BC`、成熟点重投影窗口与球面几何门限。候选统一按距离消解冲突；每关键点只能属于一个地标，每地标在同帧同相机只能有一个观测。不同已有地标不自动合并；已吸收的观测可以提供已有地标的外观，但失去地标所有权的 consumed 关键点不能作为新锚。默认每帧最多创建 200 个轨迹，后端仍负责总容量检查。

候选观测先保存在隔离计划中。绝对位姿共识优先，失败后从保留帧中选择公共独立轨迹最多的帧进行广义相对位姿/纯旋转共识。没有共识时，删除对已初始化地图点的新增观测，仅保留 tentative seed 和同次采样的 Rig 立体几何；视觉状态为 Degraded。返回 true 表示输入事务提交，不代表跟踪健康；上层应读 `trackingAccepted`、内点数和其他传感器状态。`isInitialized()` 表示曾通过米制视觉共识并拥有足够有限距离轨迹，不代替 VIO 的 IMU 偏置/重力/尺度初始化。

三角化保留原锚射线，`rho=1/range`，不是 `1/z`；检查视差、正射线距离、距离范围及全部已有/新增 seed 观测的角残差。像素噪声按 `pixelSigma*scaleFactor^octave` 传入后端。没有米制时间运动时，只有完全来自同一次采样的已知 Rig 基线可初始化距离；不会用单目五点的单位平移制造米制深度。单相机且没有已知地图/外部米制初始化的配置仍需专门的 VIO/深度初始化模块。

关键帧依据当前特征覆盖网格与最近有效关键帧共享轨迹计算，同时接受 `kfPrior`。宽高取相机模型，兼容默认不保存原图的 FrameStore。Frontend 只提交关键帧标记；后端成功优化后根据次新帧标记执行 OLD / SECOND_NEW。

Frontend 内部的可选 DBoW 仍只提供活动窗口候选，不重复进入局部因子。pipeline 的历史检索和几何回环现由独立 `GlobalSlamGraph` 执行，使用归档三维点、ORB 和广义 GP3P；提供词袋时采用独立全局 DBoW2 索引。旧组件导入 `loadComponent()` 仍不支持。

`propagation()` 默认直接调用 `factor::propagateImu()`，复用工程中的 IMU 积分实现；保留可选覆盖用于独立测试，失败时输出不变。接口保持 `T_WS` 及 Sensor 速度约定；`BodyImuFactor` 负责原生 `T_WB` 状态与 Sensor 方程之间的位姿、速度杆臂和 Jacobian 转换，并通过现有 `addImuFactor()` 提交真实残差。

`ThreadedSlam.cpp` 现直接使用原生预测插入、真实 IMU 因子、`optimize()` 事务及成功快照，旧空声明和 Services 已移除。真实 IMU 边允许时间距离初始化，仍须经过共识、视差和全观测检查；无米制来源的单位基线仍不能使用。全局归档、几何回环、位姿图及历史视觉/IMU BA 已接入，详见 `../pipeline/README.md`；LiDAR/深度子图暂缓。当前只做静态检查，未编译或用真实数据集验证。

## 静态检查

关键帧改用旋转补偿后的 S² 视差与相对保留关键帧的累计运动量。每个 Landmark 只计一条独立的同相机对应，不能把静止 rig 的跨相机空间基线当作时间视差。视差采用中位数以抑制少量错配；有足够最近帧跟踪且运动/视差很小时，当前帧标记为非关键帧，之后由滑窗的倒数第二帧标记选择 SECOND_NEW。窗口填充及第一次状态切换允许一个标记延迟，日志分别显示当前决策和实际移除帧。

`keyframe_parallax`（rad）、`keyframe_translation`（m）、`keyframe_rotation`（rad）、`keyframe_min_shared_tracks`、`keyframe_low_motion_translation` / `rotation` 可在 `frontend_parameters` 配置，未填写时使用默认值。相对参考关键帧的累计运动仍可触发关键帧，避免慢速移动被无限当作静止。视觉位姿未验证时不正常升级关键帧；共享点不足会明确记录原因。v14已删除无作用的旧 `keyframe_overlap` 配置和访问接口。

`[Keyframe]` 每帧记录 selected、reason、reference、shared_tracks、parallax_rad、translation_m、rotation_rad 和最近帧对应指标。新增 `stationaryKeyframesPreserveMatureAnchors` 使用合成双相机特征/描述子，调用工程三角化、视觉因子和优化，连续 35 帧越过 10 帧窗口，检查 SECOND_NEW 和原始成熟点保留，并有累计运动触发关键帧的正向对照；它不替代真实图像回放。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File src/frontend/tools/check_frontend_orb_static.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File src/frontend/tools/check_frontend_pose_static.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File src/frontend/tools/check_frontend_backend_static.ps1
```

报告为 `tools/frontend_orb_static_report.json`，分开记录 ORB 接入结构检查结果和现有整机集成缺口。检查 include、分隔符、工程注册、入口调用、字节长度、几何绑定、并发所有权、profile、词袋及配置，不证明类型检查或数值正确。

`tests/FrontendOrbRegression.cpp` 是独立回归源码，覆盖非对齐 256-bit Hamming、非连续行、错误长度拒绝、混合相机提取、统一焦距、缓存复用、profile 防混用、外部关键点过滤及并行首次初始化。未加入应用编译目标，本轮没有编译或运行。

`tools/frontend_pose_static_report.json` 额外检查 OpenGV/旧回环 Ceres 的清除、求解器入口、Body 坐标方向、尺度可观性、内点映射、像素噪声信息矩阵与原子提交。`tests/FrontendPoseRegression.cpp` 补充混合模型、Body/IMU 不同坐标、索引过滤、异常地图点、后半球、广义米制相对位姿、中心尺度与纯旋转退化的回归源码；后端回归增加前端初始化事务及失败回滚用例。均未编译或执行。


`tests/FrontendBackendRegression.cpp` 覆盖混合相机和 Body/IMU 非同坐标、提案隔离、锚射线逆距离、后半球、consumed 防复用、完整 Frontend 调用顺序、冻结索引/描述子拒绝及 IMU 输出回滚；后端回归补充整批位姿/观测/健康/关键帧回滚和发布保护。均只提供源码，未编译或运行。

## v9：参考 OKVIS2-X 控制视觉优化规模

核查了本机 OKVIS2-X 和官方源码。它的控制点主要在观测生命周期，而不只是检测数量：

- [Frontend.cpp](https://github.com/ethz-mrl/OKVIS2-X/blob/main/okvis_frontend/src/Frontend.cpp) 只在关键帧执行新地标 stereo 创建；普通帧主要延续已有点。其 stereo 内部也拒绝非关键帧两端均无 ID 的新点。
- [ViSlamBackend.cpp](https://github.com/ethz-mrl/OKVIS2-X/blob/main/okvis_ceres/src/ViSlamBackend.cpp) 先剔除过量非关键帧，再按与当前帧/当前关键帧的共视关系选择旧关键帧，转成 pose graph/MST 并移除相应原始视觉观测；更老的状态冻结。实时图和完整图分开管理。
- `ViGraph::cleanUnobservedLandmarks()` 清理只剩不超过一个观测的地标；Ceres Problem 随图增删 block/residual，不是每帧完整重建。
- 官方 [Hilti22 示例配置](https://github.com/ethz-mrl/OKVIS2-X/blob/main/config/hilti22/okvis2.yaml) 每图最多 700 点、5 个关键帧与 3 个近端 IMU 连接帧；这是特定配置，不是整个优化器仅有 700 点，也不是本项目四目 Hilti2023 的实测性能对照。

本项目保留VINS风格统一10帧窗口。v15移除旧的keyframe_only_new_landmarks策略：非关键帧保留新的TrackingOnly轨迹，稳定参考和几何支撑决定是否创建rho，跟踪身份接纳与优化变量规模分别控制。检测点数仍由max_num_keypoints决定，不随种子策略修改。

v10：三角化前向窗口查询 `promotionWouldBeMarginalized(anchor,support)`。除了出窗锚点，还要检查未消费的出窗支撑是否会让新 rho 在本次 OLD 中立即被吸收；此时保留种子并记录 `deferred_eviction_support`。Absorbed 历史观测不参与三角化，也不触发该延迟或后台附加 rho 消元。对应正向/反向回归见 `promotionSurvivesImpendingEviction` 与 `oldestSchurKeepsIdentityWithoutReusingEvidence`。

OLD 的旧证据先进入先验，存活 ID 变为仅供跟踪的种子，新深度必须来自之后的未消费观测；SECOND_NEW 仍按专用冗余帧路径丢弃其视觉观测、合并 IMU。独立标量地标适用 Dense Schur，保留先验耦合时退回通用求解。当前只在同一求解事务的 refit 内复用 Problem；跨帧仍重建，完整图仍会在事务边界复制，这是与 OKVIS2-X 在管理成本上的剩余区别。

运行比较应关注 `active_landmarks / residual_blocks / prior_columns`、`deferred_new_tracks`、`schur_preserved_tracks`、`jacobian_ms / residual_ms / linear_solver_ms / solve_checkpoint_ms / window_checkpoint_ms / commit_ms`，同时检查持续接受率、S²/IMU/prior 分项残差、轨迹和成熟点覆盖。不能只比较元数据中的地标总数；控制变量减少后也不能单凭耗时变小认定定位精度提高。本轮没有执行新增数值用例或数据集。


### 2026-11-04 v11：绝对位姿 CPU 热路径

AbsolutePose 的多内点 LO 与最终可观性使用解析 S² log 导数。内部 `T_SW` 更新为平移直接相加、旋转左乘，几何导数为 `[I, -skew(R_SW * p_W)]`，不同于后端完整左 SE3。LO 直接累积 6×6 H/g，拒绝的阻尼试探复用同一线性化；可观性仍对列缩放后的 J 直接做 SVD。`analyticAbsoluteJacobians=false` 保留中心差分对照。相对/旋转恢复仍采用原导数路径，GP3P/P3P 和 PoseLib LO-RANSAC 主循环继续复用。

MSAC 对明显超过角阈值的半射线用叉积/点积判定，内点仍计算原 atan2 角度平方。分类一次取得内点、评分和 RMS，LO 复用下一轮分类；最小求解器数组在估计器内复用。

新增可选 YAML `pose_analytic_absolute_jacobians`、`pose_adaptive_ransac`，默认 true。预算策略仅作用于已初始化且未丢失的绝对位姿跟踪：至少 60 条对应、预测角阈值内点率 ≥0.9、预测内点 RMS ≤0.5×角阈值、且非每 30 帧的完整检查帧，才将最低采样从 50 降为 20。最大 1000 和 PoseLib 置信度动态停止保留；其停止判断为严格大于下限，健康情况下通常对应 21 次而非恰好 20 次。恢复、回环和相对位姿保持原预算。

缩减预算候选失败、支持下降、RMS 增大、偏离预测超过 0.1m/0.03rad 时重跑完整预算；若通过本地检查但被项目现有 IMU 门控拒绝，也先完整重跑绝对位姿。0.1m/0.03rad 仅触发增加工作量，不拒绝完整求解结果。原独立地标数、内点、IMU 和上层提交门限继续生效。

`[Timing][Pose]` 新增 `generation_ms / scoring_ms / lo_ms / observability_ms`、`scoring_passes / residual_point_evals / jacobian_point_evals / lo_iterations`，以及 `reduced_budget / budget_fallback / forced_full_budget / attempted_iterations / attempted_models`。LO 含其内部评分时间，不能把 scoring 与 LO 相加；point eval 计数覆盖 LO/可观性残差及其导数，不含角度评分遍历。内部预算回退的时间与 attempted 计数累积两次尝试，iterations/generated_models 仍描述最终返回的那次求解。上层 IMU 门控触发的第二次调用会有单独日志。

新增解析导数与无真值初值的含噪外点对照源码，导数 CSV 为 `absolute_analytic_jacobians.csv`。本轮仅静态检查，没有运行这些用例或数据集。


### 2026-11-05 v12 晋升预算与精化停止

`frontend_parameters.max_promotions_per_frame: 40`、`max_active_initialized: 1000` 是健康 metric 跟踪时的新 rho 晋升预算：需 allowTemporal 且当前可用有限地图点至少60；初始化/支撑不足的恢复保留原策略。可分别设0关闭对应预算。达到上限会保留原始种子/观测并延迟晋升，不删除成熟点或历史因子，因此1000为软门槛，初始化/恢复及已有窗口可能超过它。预算内按32个世界射线等立体角分区覆盖、视差和ID确定性选择。`deferred_promotion_budget` 只计本次被预算推迟的合格候选。预算不限制三角化候选的几何验证，不将不合法点计作有效支撑。

`PoseSolverOptions.refinementRelativeCostTolerance=1e-10` 控制LO接受步的相对cost停止（设0关闭）；阻尼步小于1e-9时在额外残差遍历前停止。保留最佳模型评分、可观性和上层门限。新日志 `lo_accepted / lo_rejected / lo_small_step_stops / lo_relative_cost_stops` 为一次solve多轮LO的累计；内部完整预算重试会合并计数。

本轮仅静态检查；晋升预算改变新约束的激活时机，需用同GUI设置、同数据段比较追踪覆盖、窗口规模、残差与轨迹。


v14已从参数结构和读取器移除 `detection_threshold`、`absolute_threshold`、`matching_threshold`、`octaves`、`keyframe_overlap`、`use_cnn`。保留 `max_num_keypoints` 与实际使用的ORB/关键帧运动参数。Frame/MultiFrame不再声明CNN分类、分类概率缓存及分类状态；CNN线程空接口和条件编译过滤分支同步移除。Frame仅有内联实现，其旧分类空实现Frame.cpp及两个VS目标中的编译项已删除。


## 2026-11-05 v15：持久种子与优化地标分开管理

非关键帧不再统一撤销新轨迹。通过现有引导匹配、关联及几何筛选的候选可获得持久ID及TrackingOnly观测，保留它们不会创建rho或Ceres视觉因子。已有轨迹先接纳当前观测，再将新身份按每相机4x4单元均衡选择；单元内仍按描述子距离排序。每次最多200个新身份，另外受后端地标/观测容量约束。种子预算满时仍可延续已有轨迹。

配置：

- `max_tracking_seeds: 1000` 是全rig未初始化身份的新增接纳上限；0关闭此额外上限。它不强制删除仍在跟踪的种子，已有成熟点降级等情况可以暂时超过名额，后端总容量仍有效。
- `seed_max_untracked_seconds: 0.5` 使用数据时间戳距最后有效观测的年龄，0关闭超时清理；不是种子总寿命，更不是CPU处理耗时。仍持续观测的静止种子不会因年龄大被删除。只回收无rho、无live factor、无Absorbed测量的纯种子。
- `require_keyframe_anchor_for_promotion: true` 默认只从未消费、非本次删帧目标的关键帧观测创建rho。没有稳定参考时继续跟踪。启动/恢复仍保留已知rig stereo的原有入口。

纯种子可先选择已有或本次关键帧的新鲜实测观测作参考，`batch.seedAnchors`在同一私有事务内应用；已有rho、prior-coupled坐标、被消费的锚观测均禁止使用此入口。再按原有正逆距离、全视图角度、视差及出窗信息检查，受40点晋升/1000成熟点预算约束。预算延迟只延迟rho，不抹掉种子ID。

删除旧帧或次新帧时，现有窗口逻辑只删除该帧观测；有存活观测的种子保留ID并选择存活参考。已吸收图像的tombstone不恢复，不重复三角化旧信息。

`[Keyframe]`新增 `new_seed_tracks / active_seeds / seed_anchor_updates / seed_pool_before / seed_budget_skipped / expired_seeds / new_seed_cells`。`deferred_new_tracks`不再表示非关键帧的统一撤销数，已移除该字段。绿色匹配线包括保留种子和成熟点，图例已明确；06 Feature Tracks继续区分成熟/种子。红线仍可能含关联冲突、容量和几何拒绝，不是纯外点统计。

该策略改善身份连续性，可能增加元数据、关联及提交成本，不保证实际帧率提高；下一次实测应联合观察active_seeds、活跃rho/因子数、内点、晋升/退休及细分耗时。本轮仅静态检查，未运行C++或数据集。

## 2026-11-05 v16：短轨迹延续与启动恢复成本

源码确认了三个周转机制：描述子互检先于“已有ID优先”的关联分配，未分配特征可能先抢走已有种子的互检名额；原分阶段覆盖率把未获得ID的新纹理候选也算作已有跟踪支持；新身份接纳只看总名额，即使已有足够空间覆盖仍不断补入。它们能解释短种子的产生，但旧日志不能逐ID证明每一次退休来自哪个机制。

`OrbMatchGuidance.queryPriorities`允许已有身份在已有身份集合内建立反向最近邻及ratio。原全候选反向检验同时保留，已有身份稀疏时不会凭空制造第二邻居。通过原正向ratio、Hamming和几何门限的已有身份先接纳，再接纳新纹理；同一Hamming遍历同时生成两组反向邻居。未提供该字段的普通匹配保持原门限和语义。不同已有ID不会被自动合并；最终仍经过位姿、地图几何及后端检查。

分阶段恢复现在要求每相机至少`min(10, minimumStageMatchesPerCamera)`个已有ID候选支持，不能仅凭新纹理数量跳过历史恢复。已有地图支持可靠且跟踪正常时，新增身份按每相机4x4单元补至该单元检测点数的一半（向上取整）；已有身份的延续不受此额度限制。双当前图像的stereo候选只要任一端存在空间缺口仍可接纳。该密度是补点目标，不会删除超额的已有轨迹，也不是每帧必须补满的数量。未初始化、失跟及完整候选恢复绕过此补点控制；`Options.replenishSeedGaps=false`保留参考接纳路径。检测200点/相机、晋升40/1000及种子TTL配置未更改。

`[SeedContinuity]`每30帧及完整候选恢复时输出：`source_seed_ids`（选中历史帧里的可查询种子身份）、`matched_seed_ids`（匹配候选涉及的种子身份）、`proposed_extensions`（分配后延续的不同ID）、`committed_extensions`（筛选后提交的不同ID）、`filtered_extensions`、`slot_conflict_attempts`、`identity_conflicts`、`coverage_skipped`及所有权互检统计。前两者之差不能直接等同于几何外点；历史帧中不是每个种子在当前帧都可见。slot计数是尝试次数，身份冲突是候选事件数。`[Keyframe]`逐帧额外记录`seed_coverage_skipped`。

启动/恢复的2D-2D路径先完成旋转求解。只有旋转已通过原有接纳条件且`inlierRatio > rotationDominanceRatio`时才跳过六点求解：这是原最终选择规则必然选旋转的分支短路，不新增“静止”假设，不修改平移或发明尺度。其余情况仍运行原PoseLib通用六点/中心五点LO-RANSAC。旋转LO与可观测性检查使用解析S2雅可比，`analyticRotationJacobians=false`保留数值参考。fallback日志新增`relative_attempted / relative_skip`、旋转内点/迭代，以及实际相对求解的生成、评分、LO、可观测性计时，便于独立比较启动22次慢调用。

关联从后端`associationFrame()`直接获得一个无图像、描述子独立的可写提案；避免先导出再复制。冻结测量一致性直接比较原只读payload，三角化补读历史射线也不再导出整帧。此处没有让前端共享可写后端描述子。新增回归源码覆盖所有权竞争、单邻居/距离门限、补点不影响旧地图关联、脱离副本及旋转解析/数值恢复；本轮仅静态检查。
