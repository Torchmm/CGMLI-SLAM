# Sliding-window backend

本目录使用 C++17、Eigen 3.4、Ceres 2.2，实现滑窗数据管理、因子图优化和边缘化。当前窗口已经管理帧、地标、观测与因子，项目入口为 `cgmli::ViSlamBackend`。传感器预积分、LiDAR 因子及 pipeline 调度仍需接入。

## CUDA 视觉批量求值

`estimator_parameters.backend_visual_batch` 默认是 `cpu`。在 CUDA 构建中设为 `cuda` 后，Ceres 的求值回调批量计算双精度球面残差和解析雅可比。窗口拓扑在一次求解的多次试探点之间复用。GPU 仅回传可变参数块的雅可比，`[VisualBatchSolve]` 记录回传字节数。Ceres 管理鲁棒损失、流形、线性求解和步长，GPU 求值失败时回到 CPU 因子。

`hybrid` 模式保留 Ceres 优化器和 CUDA 雅可比批处理。仅计算残差的试探点由 Ceres 并行求值 CPU 球面因子。`[VisualBatchSolve]` 的 `cpu_trials` 记录走 CPU 的试探点。

`schur` 是实验模式。首轮完整求解用 GPU 累计视觉正规方程并消去独立逆距离，CPU 合入 IMU、标定和边缘化先验后求解约化系统。压缩后的逆距离块随约化矩阵一并回传，CPU 据此回代并用带损失的 CPU 因子验收试探步。遇到非视觉因子连接自由逆距离、无效步或数值失败时回退 Ceres。外点 refit 和整帧重试也使用 Ceres 的混合路径。`[OnlineSchur]` 记录尝试耗时和回退原因。

将 `estimator_parameters.backend_visual_batch` 设为 `audit` 后，每 30 帧对同一窗口比较原有 Ceres 求值、主机批量求值和 CUDA 批量求值。`[VisualBatchAudit]` 记录白化残差、切空间雅可比、完整窗口的 `JᵀJ` 和 `Jᵀr`，以及独立逆距离参数可消元时的阻尼 Schur 约化。审计运行仍使用 Ceres 提交估计。

## 职责与数据所有权

| 类 / 文件 | 负责内容 |
| --- | --- |
| `ViSlamBackend` | 工程入口：Rig 初始化、帧/地标/观测提交、项目类型查询、优化与快照 |
| `FixedLagBackend` | 输入暂存，组织求解、边缘化、失败回滚和原子发布 |
| `SlidingWindowManager` | 统一拥有 `FrameStore`、`LandmarkStore`、`BackendGraph`；维护窗口成员和两种删帧策略 |
| `FrameStore` | Epoch、冻结的 `MultiFrame` 特征、可选图像、原始 IMU 样本、固定相机基线 |
| `LandmarkStore` | 地标身份/锚点/质量、观测表、按帧/地标/因子反向索引、历史测量消费标记 |
| `SlidingWindowData.cpp` | 帧关联、批量视觉更新、深度初始化、地标退休、项目数据导出及跨存储一致性检查 |
| `BackendGraph` | 唯一可写的位姿、速度、bias、逆深度、在线标定数值，以及因子和边缘化先验 |
| `WindowVisualFactor` | 多相机逆深度 S² 视觉残差、像素噪声白化、可选时间偏移修正 |
| `BackendOptimizer` | Ceres Problem、NIS 门控、LM、外点重拟合、标定范围检查及条件信息 |

局部窗口只使用一个容量：应用配置 `estimator_parameters.window_size` 对应 `BackendOptions::windowSize`，两层默认均为 10。每个四目图像组对应一个状态，关键帧是该状态的属性；IMU 提供相邻状态之间的预积分约束，不占用单独配额。已有 10 帧时允许插入第 11 帧，优化后移除最老帧或次新帧，再保留 10 帧。优化/边缘化失败仍按原事务回滚；没有通过删除 IMU 因子来缩小窗口。
| `Marginalizer` / `MarginalizationPrior` | 切空间 Schur 消元、秩处理、固定线性化平方根先验 |

`SlidingWindowManager` 是这些存储的唯一修改者。优化器只求解传入图，不另建帧或地图点容器。地标元数据不保存第二份可写 rho 或世界点；世界点由当前锚帧、rho、相机外参计算。`MapPoint2`、`MultiFrame` 查询结果均为派生视图。

接收帧时显式重建特征数组并克隆描述子；默认不保留图像，`retainFrameImages=true` 时克隆图像。这样前端复用原始 `cv::Mat` 不会改写后端输入。不能直接依赖现有 `Frame` 的拷贝构造：该构造没有复制完整特征数组。相机模型指针共享，调用者必须保持其标定与 mask 不变。

`FrameStore::measurements()` 是冻结的测量载荷，不包含后续关联更新。需要当前关联时调用 `ViSlamBackend::multiFrame()`；返回独立只读视图，重建 landmark ID、世界点和当前外参。频繁获取完整视图会复制描述子，前端热路径宜使用单点查询、反向索引或已发布快照。

## 工程约定与多相机支持

- `T_AB` 将 B 中的坐标变换到 A。默认 Body B = 工程 IMU S；不同时显式提供固定 `T_BS`。
- 状态 Pose7 为 `[p_WB,qx,qy,qz,qw]`，SpeedBias9 为 `[v_W,bg,ba]`。预测输入必须使用这些坐标约定。
- 位姿采用左 SE(3) 扰动 `Plus(T,delta)=Exp(delta)T`，切空间次序 `[rho,theta]`。
- 以 C0 为 Rig 坐标 R：`T_BR=T_BS*T_SC0`，`T_RCi=T_SC0.inverse()*T_SCi`。只优化一个共享 `T_BR`，各相机固定基线不独立漂移。
- 相机位姿为 `T_WCi=T_WB*T_BR*T_RCi`；LiDAR 外参为 `T_BL=T_BS*T_SL`。
- 任意非空相机数量均可；通过 `CameraBase::backProject()` 及其 Jacobian 接入项目已有 Pinhole、EUCM 等模型，不依赖 ORB/BRISK 的具体描述子算法。
- 同一次 Rig 使用同一组相机模型对象和固定相对基线；模型/基线变更需要重新初始化系统。输入 `geometry()` 应与 `NCameraSystem` 中的模型一致。
- 描述子当前按项目二进制特征接口接收 `CV_8UC1`，非空时行数必须匹配关键点数量。
- Epoch ID 与时间严格递增；`MultiFrame` ID/时间必须与其 Epoch 一致。时间统一到 IMU 参考轴，传感器原始时间转换由上层完成，不能在求解中改变 Epoch 时间或 ID。

时间偏移约定为 `t_IMU=t_sensor+td`。每个 Epoch 创建时冻结当时的相机 `td_ref`；迟到匹配默认仍使用该值，避免把新标定当作旧测量的参考。若前端有自己的预修正参考，可以逐观测显式提供 `referenceTimeOffset`。

`extrinsics(id,camera)` 返回当前 `T_BCi`；导出的 `MultiFrame` 则转换为工程使用的 `T_SCi=T_SB*T_BCi`。

## 地标与观测生命周期

观测唯一键为 `{frameId,cameraIndex,keypointIndex}`。一个关键点只能属于一个地标；同一地标在同一帧的同一相机最多一条观测，允许同帧跨相机观测。

1. 创建 seed：登记锚观测和轨迹，`inverseDepth=std::nullopt`，不创建虚假的优化深度。
2. 上层三角化/深度初始化成功后调用 `initializeLandmark()`，创建正值 rho 参数。rho 表示单位 bearing 上的逆距离，并非 `1/z`；初始值不是额外深度测量先验。
3. 已初始化轨迹的非锚观测自动生成 S² 因子；seed 的已有观测也在初始化事务中一起转成因子。锚 bearing 是逆深度参数化的坐标参考。
4. 外点剔除同步清理关联、反向索引和因子。不可单独删除几何锚点；应退休该地标。
5. 成熟地标退休时，将其全部相关 live 因子及历史先验中的相关信息一起做 Schur 消元，然后清理 rho、元数据和关联。未初始化 seed 没有优化信息，直接清理。

观测状态为 `TrackingOnly`、`LiveFactor`、`Absorbed`。已经进入先验的观测可以用于外观关联，但不能再次当作独立测量加入优化。即使原地标被退休，也保留当前活动帧内的消费标记，阻止改用新 landmark ID 重复使用同一个图像测量；对应帧滑出时清除标记。

Landmark ID 必须严格递增且不复用；一批新地标可从 `nextLandmarkId().value()` 起连续分配。窗口、地标、观测与 IMU 样本均有容量约束。

## 两种滑窗与原子提交

默认保留 10 个 Epoch，允许一个额外插入槽。第 11 个状态求解成功后：

- `MARGIN_OLD` 删除最老帧；
- `MARGIN_SECOND_NEW` 删除倒数第二帧，保留最新帧和较早关键帧。

自动策略检查倒数第二帧的关键帧标记：关键帧选 OLD，否则选 SECOND_NEW；也可显式指定。关键帧判断由前端通过视差、运动与健康状态决定。

`MARGIN_OLD` 默认启用 `preserveOldestVisualInformation=true`：不主动丢弃最老帧的视觉证据，将出窗状态、出窗锚点的 rho，以及通过实际参与求解的 live 视觉因子关联到它的成熟点 rho 一起 Schur 消去。已消费的 Absorbed 观测只有身份意义，不能触发新 rho 消元；没有参与因子的 TrackingOnly/Pending/Rejected 记录也不触发附加消元。消去 live 相关 rho 可避免仍在优化的地标通过新先验相互耦合。`preserveSchurTrackIds=true` 时，有存活观测的地标保留原 ID，成为仅供外观关联的种子；旧观测连同没有独立 residual 的隐式锚点都标记为 Absorbed。再次成熟必须使用新的未消费观测，不能复用已入 prior 的几何信息。这是消元后保留跟踪身份，不是 rho 的精确换锚运输。

`resetOldestLandmarks` 只在关闭信息保留模式时启用此前的近似测量模型重设，用于对照；严格 `autoReanchorLandmarks` 的等价审计仍然保留。`preserveSchurTrackIds=false` 可对照完整几何与身份退休，包括锚点仍在窗内但 rho 已参与消元的点。

`MARGIN_SECOND_NEW` 使用专用冗余帧路径。生产 IMU 由 `factor::makeBodyImuRecord()` 提供原始采样与固定配置，删除前后两条原因子并合并区间；相同时间戳的边界采样去重且必须数值一致。`incrementalImu=true` 时复用左端冻结预积分的完整采样区间，只追加新增部分；图像时刻落在采样间时，从完整采样前缀重建末尾的分数区间，避免反复拆分 midpoint 积分。参考 bias 超过 gyro 0.0003 rad/s、accel 0.02 m/s² 阈值则整段重积分。两条原因子不再进入 prior，避免重复计数。没有 replay 数据的外部 IMU cost、次新帧 SB 已在 prior 或其他状态约束仍使用 Schur bridge；阈值内原始采样超过硬预算时拒绝本事务，不改变为重复的信息桥。

`maxImuIntervalSeconds=10` 参考 VINS-Fusion 的长区间筛选：超过阈值的原生 IMU 记录可以继续合并和保存，但 `solveEnabled=false`，门控、求解、重积分刷新和 Schur 都跳过它。超过原始采样预算后只留端点包围样本并标明 `completeSamples=false`，不允许重新启用为数值因子。长边不再进入 prior 不等于已有 prior 会被重置，也不能保证 prior 残差立即降低；需要用新的完整运行检查约束变化后的轨迹。

帧滑出时同步处理位姿、速度/bias、锚定地标、图像观测、反向索引、因子、描述子/可选图像及 IMU 样本。未初始化种子只要有安全的存活观测，就保留 ID 并迁移参考；没有安全存活观测时才退休。次新帧上的成熟点允许在保留观测中重设测量锚点：优先较老关键帧，使用其实际单位 bearing，按正射线投影距离初始化 rho，并重建其他存活视觉因子；要求至少两个存活视图、无 prior/已消费观测耦合、零 bearing 速度、无外部 rho 因子，且所有重建白化残差通过原 NIS 门限。

该操作属于 VINS 风格的冗余观测丢弃和测量模型重设，日志为 `reset_tracks`；它不声称保留旧锚的全部信息，也不是可逆换锚或历史先验坐标运输。无法安全重设的成熟点继续联合 Schur 退休，历史消费标记保留。`autoReanchorLandmarks` 的严格等价审计仍用于 OLD 的可选换锚，与默认 SECOND_NEW 测量重设是不同操作。

没有 prior/已消费/外部因子耦合、但只剩一条安全观测射线的成熟点，转回未初始化种子，日志为 `demoted_tracks`：保留 ID 和存活测量，删除无足够视图支持的 rho 与活跃成本，后续可由真实多视图重新初始化。降级不等于退休，也不把旧深度包装成新的先验。

`[Window]` 新增 `imu_mode / replaced_imu / active_imu / dropped_visual / reset_tracks / demoted_tracks / moved_seeds / prior_reused / prior_landmarks`。纯本地原生 VIO 中，期望保留地标不再因冗余图像进入稠密 prior，`prior_landmarks` 通常为 0，Schur 独立性能够保持。外部因子确实耦合地标时仍按实际结构回退，不强制使用 SPARSE_SCHUR。

`step()` / `optimize()` 将求解、外点清理、边缘化和快照构造放在一个事务中。失败恢复整个窗口的数据和数值，已提交输入仍保留以便修复后重试；旧快照不变。成功才发布新的 `shared_ptr<const BackendSnapshot>`，包含同一版本的状态、标定和地标估计。

`StepReport.retiredLandmarks` 和 `removedObservations` 表示相对上一份成功发布快照的变化，供前端清理缓存；本轮边缘化即时清理的对象同时记录在 `MarginalizationReport`。首次发布前创建又退休的对象不出现在快照差量中。

所有修改、普通查询及观测性计算都在同一后端线程调用。只有 `snapshot()` 支持跨线程读取；窗口内部的引用/指针不能跨事务长期缓存。

## 换锚边界

每次成功边缘化在 `[Window]` 输出实际 OLD/SECOND_NEW、被删帧和关键帧标记、地标前后数量、成熟/未初始化退休数量、仍有 surviving live observation 的成熟退休数量。`[LandmarkRetireSummary]` 默认聚合原因；`GLOG_v=1` 时的 `[LandmarkRetire]` 包含每个 ID、anchor 和 live/absorbed/tracking 支撑。支撑数量在 Schur 消费观测之前采集，避免把“有可用观测但旧锚退休”误报为“没有观测”。原因区分默认关闭换锚、历史 prior 耦合、信息检查不通过、没有安全新锚、缺少 surviving observation 等；显式新种子退休继续由前端的 retired_seeds 记录。

视觉因子默认为解析雅可比，`[VIO] Ready` 和 `[Timing][Solve] visual_jacobian` 明确打印路径。解析内核预计算测量白化矩阵和静态 bearing；同帧 stereo 在 rig 中直接消去公共位姿/外参。跨帧使用相对位置及符号化消去的 ambient lift，减少重复 SE3 构造/归一化和矩阵乘法；数值参考路径保留不变。Ceres 的 jacobian_ms 是整个 solve/refit 的累计求值时间，不是单次单个因子的耗时。

同一优化事务的 refit 复用一个 Ceres Problem，只移除新判定的外点 residual block，保留共同参数缓冲和同一先验。Problem 在 rollback 恢复图缓冲之前销毁，且不跨帧缓存。总迭代预算、NIS 和最终重拟合要求没有降低。`problem_builds` / `problem_reuses` 可核对是否真正复用。性能采样的计数器按 cache line 分条，减少多个 worker 争用同一分类原子变量；`factor_profiled` 标明带采样开销的帧。

逆距离仍为 `rho=1/||p_C||`，单位 bearing 可有负 z；正参数下界保持不变。有界 LM 使用的投影 Armijo 线搜索改为 `QUADRATIC` 插值，复用起点斜率，避免默认 `CUBIC` 在试探点额外请求梯度；没有关闭线搜索或放宽接受门限。每次 refit 都使用同一设置。`[Timing][Solve] line_search_interpolation=QUADRATIC` 可确认实际配置；新增 `jacobian_evaluations` / `residual_evaluations` 是 Ceres evaluator 的整次求值计数，累计所有 refit，与按因子统计的 `jacobian_calls` / `residual_calls` 单位不同。`line_search_cost_ms` / `line_search_gradient_ms` / `line_search_ms` 是 Ceres 内含诊断时间，与 `jacobian_ms` / `residual_ms` / `ceres_ms` 存在重叠，不能相加。

运行同一段数据后，在窗口大小、活跃视觉因子数、迭代次数和 refit 次数相近的帧上比较雅可比计数及耗时；二次插值的线搜索梯度耗时应为零或接近零，但试探次数、残差求值次数和收敛轨迹可能改变。还需同时检查 accepted、post_rejected 和分项残差，不能只凭单帧耗时判断优化效果。当前变更仅完成静态检查，尚无新数据集性能结果。

真实窗口残差诊断复用 NIS gate 的实际白化残差，不额外重算全部视觉/IMU。`[Residual]` 默认每 30 帧及失败时输出，`GLOG_v=1` 每帧输出；分 pre_gate、pre_solve、post_solve 三组，并按 visual/imu/lidar/prior/other 列出因子数、残差维度、白化分量 RMS、NIS 中位数/P95/最大值、超门限数、robust_cost 和 worst_factor。历史先验单独求值一次且不再套鲁棒核。前后集合不同不能直接比较总 cost；prior 的 NIS 是整条动态维度向量的平方范数，不与单个视觉 2DOF 门限比较。residual_stats_ms 是 gate_ms 内含的统计工作及单独 prior 统计，不能再次累加到总步骤耗时。

`autoReanchorLandmarks` 默认 **false**。显式开启后，候选必须通过世界点、世界点扰动、信息矩阵和梯度一致性检查。`checkReanchorInformation()` 使用实际视觉因子和历史先验各一次，遵循后端相同的白化、鲁棒核和左扰动约定，在共同切空间中比较 `J_point_old * G`、`Gᵀ H_old G`、`Gᵀ b_old` 与候选结果。逐块检查 H/b，避免强先验掩盖弱方向的信息丢失。

仅尝试不在历史先验中、没有已消费观测、bearing 速度为零且没有未知外部 rho 因子的轨迹；新锚还需满足视差与重投影检查。候选在私有图中重建视觉因子并把旧锚测量显式化，信息检查通过后才提交。普通的“固定 bearing + 一个逆深度”换锚，在有基线时通常不是等价的坐标变换：名义点相同，但可扰动的方向不同。检查不通过时保留原参数化进入正常 Schur 退休，不写入候选，不消耗新因子 ID。

模块测试输出三组证据：原图不变的正向检查与损坏历史先验的反向检查；实际固定 bearing 候选的切空间/H/b 差异及生产拒绝行为；真正可逆的三维地标坐标参考变换下，解析/数值变换 Jacobian、耦合历史先验和最终 Schur 的一致性。第三组是数学参考，**不代表当前单逆深度生产路径已实现完整换锚运输**。历史 prior 耦合的 rho 仍保守退休；不应直接打开开关声称已完成需求中的 Landmark 持续换锚。

## 调用示例

```cpp
#include <backend/ViSlamBackend.hpp>
#include <stdexcept>

using namespace cgmli;
using namespace cgmli::backend;

BackendOptions options = BackendOptions::realtime();
ViSlamBackend estimator(options);
std::string error;
if (!estimator.initializeRig(cameraSystem, T_SL, error, tdCamera, tdLidar)) {
    throw std::runtime_error(error);
}

// Per epoch: the caller supplies an IMU/body prediction in the common time frame.
auto prediction = makeEpoch(frame, T_WB_prediction, speedAndBias,
    SensorHealth::Valid, SensorHealth::Degraded, isKeyframe);
if (!estimator.addStates(frame, prediction, rawImuSamples, error)) {
    throw std::runtime_error(error);
}

// Batch all matches from one image into ONE transaction.
VisualUpdateBatch batch;
const LandmarkId point = estimator.nextLandmarkId();
const KeypointIdentifier anchor{frame.id(), anchorCamera, anchorKeypoint};
batch.landmarks.push_back({point, anchor, triangulatedInverseDistance, {}});
batch.observations.push_back({point,
    {frame.id(), otherCamera, otherKeypoint}, {}});
// Existing tracks: batch.observations.push_back({existingId, key, options});
// Seeds: use std::nullopt on insertion, then batch.initializations when triangulated.
if (!estimator.applyVisualUpdates(batch, error)) throw std::runtime_error(error);

// The IMU preintegrator supplies a real whitened cost, when available.
// auto submission = estimator.addImuFactor(previousId, prediction.id, imuCost);
// if (!submission.accepted()) handle(submission.message);

// Initial gauge/velocity/bias constraints and all relevant sensor costs must have
// been supplied by the initializer/factor modules before accepting a fused result.
auto result = estimator.optimize();
if (result.accepted) {
    auto stateAndLandmarks = result.snapshot;
    // Publish accepted state; propagate high-rate IMU outside this window.
}
```

示例中的图像、匹配索引、三角化结果、预测和初始约束来自现有前端/初始化器；此接口不凭图像自行产生初始位姿或 IMU 预积分。只有预测和 raw IMU 数据时，后端不会伪造约束；最新位姿没有 active 约束则等待测量。

`applyVisualUpdates()` 每批只复制一次窗口元数据，并做一次最终一致性验证；不可变特征载荷共享。顺序为：移除观测、联合退休地标、创建轨迹、初始化深度、追加观测。任一操作失败整批回滚。逐点 API 仍可使用，但一帧数百个特征宜使用批量入口，避免逐点复制整个窗口。

工程适配入口 `ViSlamBackend` 直接提供状态、帧、地标和事务接口，不再保留旧双图后端的空声明。ThreadedSlam 使用显式 Body 预测、真实 IMU 因子和 `optimize()`，由现有窗口管理器完成 OLD/SECOND_NEW；成功后才发布。历史回环和最终 BA 由独立 `GlobalSlamGraph` 承担，详见 `../pipeline/README.md`。原文件保留在 legacy，LiDAR 暂缓。

## 视觉残差与优化

原生视觉因子使用单位球面 S² 对数误差，不做 `x/z`，也不要求 camera-z 为正；合法鱼眼后半球方向可参与。相同 Epoch 的跨相机观测合并重复的 Pose 参数，避免 Ceres 一个 residual 重复传同一参数指针。

像素噪声经过相机反投影 Jacobian、单位向量归一化及球面切基变换得到二维 bearing 协方差，再做白化。`ObservationOptions.pixelSigma` 默认 1 像素，需按检测器和金字塔层级配置，不能把 ORB patch 直径当作 sigma。

可选 `bearingVelocity` 是每秒 bearing 切向速度；因子按 `td_ref-td` 沿球面修正并运输测量切基。速度为零时不提供有效视觉时间偏移信息；该模型不包含 rolling shutter 或任意运动插值。

视觉因子默认使用解析雅可比，参考了本机 VINS-Fusion `vins_estimator/src/factor/projection_factor.cpp` 的坐标链式求导方法，但保留本项目的完整 S² 对数残差、逆距离和左 SE(3) 扰动，未替换为归一化平面残差或切平面弦长误差。`BackendOptions::analyticVisualJacobians=false` 保留原数值左切空间差分及其残差实现，供独立对照。两条路径输出相同物理残差和 whitening 定义；解析路径需要数据集回归，静态检查不能证明数值正确。

具体链路：一次计算相机变换、世界点、单位预测射线、球面对数与测量基底的最短弧运输。以世界点 `pW` 的左扰动作用 `[I,-skew(pW)]` 分别生成锚点/观测位姿的相反导数；共享外参同时累计两侧变换贡献，同帧相机的共享位姿和共享外参严格消去。球面 `theta/sin(theta)` 在近零处使用稳定导数展开；时间偏移同时求导锚点射线、观测射线和基底运输。最后乘左流形 `MinusJacobian` 提升到 Ceres ambient 坐标。主路径使用固定大小矩阵，不再逐维扰动或重复调用完整残差。

`visualNisThreshold` 默认 9.210340371976184，作用于二维白化球面残差的平方；`visualHuberScale` 默认 1，单位为白化残差。`ThreadedSlam` 通过 `estimator_parameters.visual_nis_threshold` 传入有限正门限；通用后端的外部调用者仍可显式传 infinity 禁用统计剔除。实际门限需要结合噪声模型评估。先白化、再 NIS、最后 robust loss。

外部 IMU 因子需提供 15D 白化残差和参数布局 `[Pose_i(7),SB_i(9),Pose_j(7),SB_j(9)]`，`addImuFactor()` 检查布局。其他 `FactorRecord.keys` 也必须严格对应 cost 的参数次序；视觉/LiDAR 因子声明自身 `sensorEpoch` 以应用健康门控。托管 rho 及其视觉因子必须通过地标/观测 API 创建，不能绕过关联表直接插入。

cost/loss/manifold 在提交后保持只读，由共享所有权存活；临时 Ceres Problem 不取得其所有权。默认 LM 优先使用可用的 SuiteSparse，其次 Eigen Sparse。`preferLandmarkSchur=true` 时检查每个活动残差和历史先验，只有每个残差最多包含一个待消元的非固定地图点、且存在保留变量时，才使用 Sparse Schur 和显式地图点优先排序。先验耦合多个待消元地图点时保持 Sparse Normal Cholesky；无可用稀疏库则保持 Dense QR。显式选择 Dense QR 或关闭 `preferLandmarkSchur` 可作对照；不会拆分或丢弃历史先验。

默认 `maxOutlierRefits=3`，初次求解及重拟合共享迭代和Ceres时间额度。v14将时间、迭代和refit次数耗尽与估计失败分开：已有可用Ceres状态时，清理新外点后执行参数、标定、当前节点连接性及应用质量检查，通过则接受并记录限制原因与未重优化外点数；数值异常、质量失败或边缘化失败仍整次回滚。边缘化使用清理后图在接受状态处重新求值，保留非零梯度，不把LM阻尼写入先验。限额接受不是删后目标函数已收敛的声明。

`[Backend][Refit]` 记录每轮求解预算、剩余迭代、剔除数和同轮代价；`[Timing][Solve]` 在失败、慢帧及周期摘要中记录实际求解器、稀疏库、活动地图点、问题参数/残差规模、先验行列数以及线性求解、雅可比和残差评估耗时。各轮计时累计，规模字段来自最后一轮；不跨不同残差集合比较代价下降比例，也不能把含地图点所有权的元数据总数当作优化变量数。

前端事务仍一次复制候选并完整验证，健康变更与观测批处理只在末尾共同激活一次。`[Timing][FrontendCommit]` 分别报告复制、应用及验证耗时，便于进一步定位成本；保留所有历史观测与吸收证据保护。

`[Timing][Factors]` 每 30 帧采样，`GLOG_v=1` 时逐帧记录 `visual/imu/lidar/prior/other` 的雅可比与仅残差调用数、调用耗时之和。适配器仅存在于该次 Ceres Problem，不写回图或历史先验；其所有权在 Problem 之外，计数为原子变量。计时不包含 NIS 门控和边缘化，且为并行调用经过时间之和，可能超过墙钟时间；采样帧会有时钟及计数开销。

完整验证继续检查每条正反索引、因子绑定和基数。反索引先逐项查找观测并验证归属，基数再证明无遗漏，因此观测遍历不再重复做反向集合查找；因子的四类依赖改为单次遍历检查。未删减一致性条件或改为只验证新数据。

NIS 门控在一次 `optimize()` 内复用参数指针，仅在该事务稳定的图缓冲区上有效。较大问题按本次求解线程预算分块评估，工作线程复用残差缓冲，所有线程结束后才按因子 ID 顺序更新生命周期。仍评估白化、未施加 robust loss 的残差，包括无限门限因子的非有限值检查；评估失败或异常仍完整回滚。`gate_setup_ms` 是一次性缓存准备时间，`gate_ms` 是各轮门控累计墙钟时间，两者需合并比较旧版门控成本。

边缘化只暂存被消费/被退休/被替换因子的轻量事件，不为历史事件复制完整因子表。提交候选同时采用私有图、地图点和 IMU frame payload；失败恢复因子 ID 高水位并丢弃候选。只有确实被 Schur 吸收的观测才写消费标记；主动丢弃的冗余观测与替换的 IMU 边不会被标记为已吸收。

如果出窗变量既不涉及旧 prior，也没有待吸收因子，直接共享原先验对象，保持原参考值，不做线性化、稠密特征分解或重定参考。其他情况仍按相同秩阈值计算 Schur；精确对角 PSD 系统使用逐行秩截断除法，其他系统保留特征分解，separator 更新复用 Hrr/br 缓冲。没有添加阻尼或把先验强行对角化。

`[Timing][Backend]` 新增输入验证、窗口 checkpoint、求解器内部验证/checkpoint、终态清理、出窗导出及快照差异回执耗时。`marginalize_linearize_ms / marginalize_schur_ms / marginalize_prior_ms` 是边缘化内部线性化、Schur 消元与先验构造耗时，包含在 `marginalize_ms` 中，不能重复相加。未归入这些区段的连接检查、对象销毁等仍包含在后端总耗时中。

先验使用固定参考值和平方根矩阵 `r0+A*boxminus(x,x_ref)`，历史测量不重新线性化。当前 Schur 和特征分解使用 dense 信息矩阵，大量地标/稠密先验仍有成本；不能由窗口帧数推断满足 Orin 实时目标。批量更新减少管理成本，视觉 Jacobian、边缘化和快照复制仍需后续实测。

四个全局块 `T_BR,td_camera,T_BL,td_lidar` 默认冻结。每轮外部 `CalibrationUpdatePolicy` 允许且相应最新传感器为 `VALID` 时才解锁。求解器报告超出物理范围的标定块，流水线用原来的值固定这些块后重算同一帧。`calibrationInformation()` 返回消去轨迹、速度、偏置和地标后的条件信息。报告次序为 `[theta_BR,p_BR,td_camera,theta_BL,p_BL,td_lidar]`。默认排除显式标定先验和历史先验，后者可以显式纳入。

地图候选只在成功 OLD、LiDAR VALID 关键帧且全部 `MapCommitGates` 通过时输出；返回候选不执行 Stable Map 写入。

## 当前边界与验证

2026-11-04 v10：修复已消费观测触发新 rho 消元，并在前端晋升前调用 `promotionWouldBeMarginalized()` 检查本次真实删帧策略。该查询同时检查锚点所有权与未消费的出窗支撑；不会把旧 Absorbed 视图当作新测量。延迟晋升仍保留 ID 和实际观测。真正参与旧帧约束的成熟点仍按现有模型进入 Schur；这次修复不代表已经实现长期成熟点的精确换锚，也不保证 schur_to_seed 归零。

新增验收字段：`[Window] ignored_absorbed_incident` 是被忽略的已消费观测数；`live_incident_ranges` 是除所有权选择之外、由有效出窗因子新增的 rho 数；`same_frame_range_eliminations` 检查本图像帧刚成熟又消元的范围坐标。`[Keyframe] deferred_eviction_support` 统计因有效出窗支撑而延迟的晋升，包含于 deferred_promotions。应同时看 schur_preserved_tracks、同一 ID 的 initializations/demotions、成熟覆盖与跟踪接受率，不以两个旧 reset/demoted 计数归零作为成功标准。

v10 的第 840/1920 帧及 prior 核查：

- 已有日志的第 840 帧最差边为 821→823，0.2 s，两端四相机时间完全一致，IMU 最大间隔 5 ms；NIS 67.3764 与独立重积分 67.3774 接近，±1 ms 反而升至 1059.7/289.841。这条边没有证据支持用组内不同步或一个简单全局 ±1 ms 修正解释异常。
- 第 1920 帧最差边为 1821→1919，9.79996 s，NIS 109.712，独立重积分 109.681；两端组内偏差约 -0.81～+1.40 ms，±1 ms 后 NIS 117.356/103.676。长区间模型误差、视觉/先验约束和时间近似需要区分；目前不能据此估计时延。
- 读到的 `/imu/data`、连续噪声密度和随机游走与本地官方 calibration_imu.yaml 一致。协方差传播的连续密度 sigma²*dt 及分块形式与 OKVIS2-X 源码相符。VINS 的离散 midpoint 注入形式与状态扰动不同，不能把其数值或矩阵直接替换进来。尚无证据要求修改噪声单位或放大噪声。
- 新 `[ImuConflict]` 输出 p/rotation/v/bg/ba 的各 3×3 边缘协方差 NIS；这些值有相关性，不能相加为 15D NIS。`[ConstraintBalance]` 输出 visual/IMU/prior 在同一当前左扰动切空间内、计入 robust 一阶权重的分组梯度。cosine_with_imu 接近 -1 表示局部方向相反；仍不是判定哪一种传感器错误的充分条件。该审计在指定 840/1920 帧只读运行，其成本包含于 Timing/ImuAudit。
- `[ImuAuditState] / [ImuAuditReference] / [ImuAuditCovariance]` 保存这些目标帧在求解后、Schur 前的实际两端状态、冻结 bias 和完整 15×15 协方差。目标帧选取当时最差的有效 IMU 边，修复后拓扑或最差边可能改变，必须同时对照 from/to 和 dt。最终 trajectory.csv 中历史端点可能已经在更晚时刻更新，不能直接拿它代替异常时刻的窗口状态重放。
- `[PriorFormation]` 每次新建窗口先验记录参与 visual/IMU/other 数、输入旧 prior 的参考/当前范数、输入 robust cost 与线性化残差平方和、消元降低量、新 r0 范数、丢弃梯度投影及秩阈值。`quadratic_constant_offset` 属于局部二次模型中没有写入 prior 的常数，不应与 robust cost 或真值误差混为一谈。这里只计算诊断量，不改 H、b、FEJ 参考、秩阈值或约束权重。

### 对照工程及后续性能方案

[VINS-Fusion 的 OLD](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/vins_estimator/src/estimator/estimator.cpp) 选择锚定在最老帧的有效特征，将该帧 pose/SB、这些特征的深度和旧 prior 一起消元；SECOND_NEW 处理旧 prior 的次新 pose，并在滑窗中拼接预积分。它采用 Dense Schur 和求解时间预算。本项目四目允许跨相机/跨帧关联，保留锚点的点也可能有最老帧观测，不能假设它们都符合其 start_frame=0 拓扑。精确保留这些 rho 时可能生成耦合先验，必须保留交叉项并选择通用求解器。

[OKVIS2-X 前端](https://github.com/ethz-mrl/OKVIS2-X/blob/main/okvis_frontend/src/Frontend.cpp) 先关联/估计位姿/判关键帧，再在关键帧做新点 stereo；[后端](https://github.com/ethz-mrl/OKVIS2-X/blob/main/okvis_ceres/src/ViSlamBackend.cpp) 按共视关系管理关键帧、移除视觉观测或转换成 pose graph/MST。本工程继续采用统一 10 帧 OLD/SECOND_NEW，不恢复 num_imu_frames 配额。

以 15:32 日志的 60～2250 帧为基线，建议按以下顺序验证后推进，第 2～6 项为方案，未在 v10 实现：

| 优先级 | 工作 | 基线与验收 |
|---|---|---|
| 1（本轮已修改源码） | 修正消费状态选择和晋升预判 | 观察同帧消元计数、单 ID 成熟次数；验收不能只看地标总数减少 |
| 2 | 延后新 ID 分配及新点 stereo 创建 | 对照 OKVIS 的阶段划分；非关键帧累计丢弃 393000 个提案，匹配候选仍需供共识/恢复使用，不能直接删掉恢复路径 |
| 3 | 跨帧视觉因子缓存相机对的相对变换及共同链式矩阵 | 当前 Ceres 52.66 ms，其中 Jacobian 22.84、residual 13.51；视觉 worker 求值合计 65.07/42.08 ms，远大于 prior 的 0.807/0.142 ms。沿现有 EvaluationCallback 扩展，必须对试探点、变量外参、不同 x 地址做残差/J 对照 |
| 4 | 位姿恢复的最小采样次数及 LO 预算分阶段 | 当前绝对位姿平均约 16 ms，min_iterations=50，记录的迭代均为 51。优先预测模型评分、保持置信度及几何接受门限，先 A/B 再减少最低采样，恢复时保留完整预算 |
| 5 | 稳定参数槽、增量维护 Ceres Problem 和事务变更日志 | 参考 OKVIS 的图维护方式，减少跨帧重建/复制；必须保持失败时数值、因子生命周期、观测消费与先验的整体回滚，先验证槽地址和所有权，再替换现实现 |
| 6 | 继续 CPU 检测优化，必要时再迁移 CUDA | 四目提取约 47 ms，单相机检测均值20.6、描述12.4 ms；先看 FAST fallback、候选几何筛选和配额分配。GPU 迁移需包含相机感知采样与传输成本，暂不承诺速度 |

当前边缘化细分的 Schur 核仅约 0.23 ms，边缘化总计约 11.17 ms；优先减少管理和重复初始化，不优先重写特征分解。对照性能时关闭 matches/trajectory 两个视图，固定帧段、窗口、预算和健康门限。当前来源是多项修改后的联合运行，不可用它单独量化 Dense Schur 或缓存收益。

2026-11-04 v9 的性能与一致性调整：

- Ceres 工作区通过 `EvaluationCallback` 在每个新求值点更新共享位姿旋转缓存；视觉因子绑定原参数缓冲的缓存条目，不能依赖 Ceres 传入 x 指针与用户地址相同。试探点和 refit 都刷新；离线 NIS/边缘化继续用原始因子。
- 优化器只备份数值、因子状态和标定参考，不再复制整张图；回滚前释放 Problem，再原地恢复缓冲。上层已经完整验证的私有图不重复做同一输入验证，边缘化发布边界仍完整验证。流水线把插入前原后端移动到 checkpoint，减少另一份复制。
- 可选 `DenseSchur` 用于独立地标、小规模状态 separator；任何先验或因子耦合多个地标时退回通用求解，保留完整先验。Hilti 配置 `backend_dense_schur: true`，设为 false 可对比旧稀疏路径。仍采用 LM 和正逆距离下界，未改变鲁棒门控与最终重拟合要求。
- OLD 新增独立标量 rho 先消元、再对状态块做 PSD 求解；耦合地标先验保留联合路径。仍按配置丢弃不可观方向，不加入人工阻尼。两个阶段的秩截断在病态问题上可能与联合特征分解有数值差异，新增独立 H/b 对照源码待运行。
- `[Window]` 增加 `suppressed_imu / incremental_imu_merges / full_imu_merges / schur_preserved_tracks / scalar_schur_landmarks`。`[PriorResidual]` 和 `[PriorBlock]` 显示冻结项、当前总残差及各参考坐标偏移/贡献，用于定位 prior 的增长来源。各块 contribution_norm 含交叉作用，不能直接相加为总残差。

与 OKVIS2-X 的视觉规模管理对照见 `../frontend/README.md`。本轮只做静态检查；没有编译、数值执行或新数据集速度结果。保留真实窗口差分诊断，不能用被筛掉的长边导致的残差降低代替精度验证。

2026-11-04 v7：修正 IMU 旋转残差对有限 gyro bias 修正的导数。使用 Exp(phi) 的右 Jacobian，普通求值与 sigma-gradient 求值共用同一实现。噪声四项与 config/reference/hilti2023_robot/calibration_imu.yaml 一致，保持连续密度及 sigma²*dt 的协方差注入，不放大官方值。新增动态解析运动夹具（平移、转动、非零速度、非单位杆臂、非零 bias 差），对全部左 SE3/SB 列做数值差分，并检查传播、完整动态区间与两段 raw replay 的一致性。噪声夹具检查 std 翻倍使 covariance 变为 4 倍、bias 随机游走方差为 sigma²*T，以及 200/400 Hz 下同一连续密度的速度方差一致。新增 C++ 用例本轮未执行，静态检查不证明这些数值断言已通过。

每个诊断周期新增 [ImuReplayCheck]，以当前接受的左端状态重新积分最差旧 IMU 边，比较 stored_nis、fresh_nis、位置/速度物理差异和 covariance_relative_change，操作为只读，不在求解接受后改变目标函数。若二者接近且 NIS 仍高，应继续排查模型、标定、时间近似与测量冲突，不能将其直接归因于旧 bias 参考。

静止初始化检查统一在 ImuError::initPose() 内执行，上层 initializeImu() 传入配置中的 g、a0、g0 和最大采样间隔。直接两参数调用同样受检查，失败不改变输出姿态。默认至少 10 个严格递增样本、覆盖 0.4 秒、间隔不超过 0.05 秒；Welford 计算无偏 3x3 样本协方差，以 sqrt(trace(C)) 检查加速度/角速度波动，门限分别为 0.5 m/s²、0.05 rad/s。平均角速度相对 bias prior 不超过 0.05 rad/s，校正后的平均加速度模长与 g 偏差不超过 0.7 m/s²。常量转动不能只靠协方差判断；IMU 自身也无法区分静止与匀速直线运动。参数可通过 ImuStationarityOptions 调整，统计和失败原因由 ImuStationarityStatistics 返回。

2026-11-04：活跃原生 IMU 边在求解边界检查左端 bias 与预积分参考的偏移（gyro 0.0003 rad/s、accel 0.02 m/s²），超过阈值时用保存的原始采样创建新 cost。已有 cost 和历史 prior 不被修改；整步失败仍恢复旧对象。日志 `refreshed_imu / imu_refresh_ms` 记录额外工作；没有通过更改噪声或降低 IMU 权重来压低残差。

每 30 帧（或 VLOG1 每帧）的 `[ImuSeed]` 对比新边在 IMU 预测、视觉位姿写入后的 NIS；`[ImuResidual] phase=pre_refresh/post_solve` 按 factor ID 和 from/to 区间输出原始物理 p、2*quaternion-vector、v、bg、ba 范数，以及参考 bias 偏移。旋转项为原因子的 2*q.vec()，小误差下近似弧度；完整 NIS 才是整体白化统计量，不能将混合协方差下的白化行直接解释为物理分量。日志用于定位剩余问题；旧日志不能证明参考 bias 偏移是残差上升的唯一原因。

健康视觉跟踪下，前端在三角化前确定关键帧，暂缓初始化当前非关键帧或即将删除的非关键帧所锚定的种子，记录 `deferred_promotions`，等待锚点稳定后再利用真实多视图初始化。恢复阶段仍允许已知 rig baseline 的 stereo。球面因子 residual-only 路径不再计算仅为雅可比使用的相机导数旋转矩阵和 bearing 时间导数；对应残差/雅可比一致性回归保留。

已实现局部 VIO 数据通路、真实 IMU 预积分及冗余帧区间重积分、S² 视觉因子、窗口策略、事务和只读快照。point-time LiDAR、Stable Map 和完整换锚先验运输仍未完成；本次未运行数据集，不能由静态检查认定 VIO 数值正确或满足实时要求。

按本次要求，仅做静态检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File src/backend/tools/check_backend_static.ps1
```

脚本检查源文件分隔符/include、VS 工程与 filters 注册、关键接口定义、存储/事务约束以及本机 Ceres API，输出 `tools/backend_static_report.json`。这些是静态结构检查，不证明 C++ 编译或数值正确。

`tests/ManagedWindowRegression.cpp` 是独立 main 的回归源码，覆盖特征所有权、多相机索引、seed 初始化、批量回滚、S² 几何/Jacobian、两种删帧、历史测量防重用、先验换锚边界、整窗回滚和工程入口。未加入主程序编译目标，本轮没有编译或运行；后续验证需链接 backend 与项目相机/时间等依赖后执行。
前端共识初始化入口 `applyFrontendUpdate(id, optional<T_WB>, rejectedKeys, error)` 用于最新未发布帧的位姿初值及 live 观测外点清理，统一在暂存窗口上执行并校验后提交。拒绝历史帧/已发布位姿的重置，以及删除已进入先验的证据或几何锚点；不修改已发布快照，也不替代正常后端优化。`Frontend` 的 OpenGV 位姿调用已迁移到 `visual/pose_solver`，完整关联通过 `applyFrontendBatch()` 原子提交：最新未发布帧的可选位姿初值、新地标/观测、深度初始化、视觉健康和关键帧标记统一生效；保留 LiDAR 健康，不允许删除或追加历史观测。Frontend 不在内部优化；上层补齐真实 IMU/LiDAR 因子后调用 `optimize()`。ThreadedSlam 已接入真实 `BodyImuFactor` 与原生事务；历史全局模块有具体实现，不向局部先验重复加入旧观测，LiDAR 暂缓。
本地求解新增可选 SolveBudget 参数，以值传入每帧时间/最大最小迭代/线程预算，不在检测与优化重叠时修改共享选项。StepReport 的 retiredState、retiredVisualPoints 保存两种策略下优化后、Schur 前的历史输出；失败回滚不输出这些回执，不改变原 mapCandidate 的 Stable Map 门控。

2026-11-05 v13：`SolveBudget.timeSeconds` / `BackendOptions.maxSolverTimeSeconds` 仅计 `ceres::Solve()` 内部的累计墙钟时间，初次求解与外点 refit 共享额度。前端、NIS、Problem 构建、校验和边缘化不扣除该额度。最低迭代回调使用“已完成 Ceres 时间 + 当前 Solve 时间”，后续 pass 只扣已完成 Ceres 时间。v14起预算耗尽转为清理后质量验收，详见下文；迭代边界/最低迭代允许软超时，数值收敛允许提前退出。`SolveReport` 导出 Ceres 额度与是否到达额度。现有 `perStepBudgetAndHistory` 回归增加慢前置 NIS 检查（60 ms）大于 Ceres 额度（20 ms）后仍能进入求解的案例；不对机器相关的精确耗时作断言。本轮仅静态检查，未编译或执行该案例。


启用 BackendOptions.exportVisualHistory 时，成功事务在 Schur 前输出 optimizedVisualHistory，并记录真实拒绝的视觉观测，供独立全局归档使用；失败会清空这些回执。默认关闭，以保留局部窗口的内存开销。全局原始 BA 不重复加入局部 marginal prior 或位姿图里程计。

前端容量管理使用窗口的 `remainingLandmarkCapacity()` 和 `remainingObservationCapacity()`；后者包含已吸收观测的防重用记录。前端可在同一批次回收失去来源帧支持的未初始化种子，后端会再次检查它没有 rho 参数、绑定因子、已吸收测量或当前帧观测。这是对纯跟踪元数据的回收，成熟点和历史优化证据仍禁止由前端直接删除。批次任一步失败时，回收、位姿初值、新轨迹和观测一起回滚。

`applyFrontendBatch()` 使用一个私有窗口副本，通过 `applyVisualUpdatesImpl()` 完成批量更新后统一检查 `valid()` 再提交，避免内层再次复制窗口。独立公共入口 `applyVisualUpdates()` 仍自行复制并验证。批量深度初始化将全图 pending-factor 扫描推迟到批末；每个新视觉因子的正常校验和激活判断保持执行。导出图像帧和前端几何缓存可通过坐标接口获取世界点，不必同时复制其所有观测。


### 2026-11-04 v11：视觉相机对求值缓存

原每位姿 rotation cache 继续使用。新增 solver-local 相机对缓存，键含 anchor/observer/全局外参的绑定参数地址及两端完整固定 rig 变换；同一相机对共享 W、cameraRotation 和平移 offset。每个新 Ceres 求值点统一刷新，试探步、refit 和变量外参变化均生效；同一点 residual→Jacobian 请求可复用。每位姿 quaternion ambient lift 同时缓存。rho、观测基底和时间修正 bearing 仍逐观测计算，保留每观测鲁棒核、白化、NIS 和 S² geodesic log。

仅请求 rho/td 时跳过不需要的位姿链；小角度采用 theta/sin(theta) 及其导数的一致级数，大角度仍用 atan2。保持球面误差定义、正逆距离和反向几何检查。

YAML `estimator_parameters.cache_visual_pairs: false` 可关闭 pair cache，同时保留原 per-pose cache，供单变量对照；默认 true。`[Timing][Solve] visual_cached_pairs` 是当前 Problem 已绑定条目数（refit 移除残差后不立即压缩），`visual_cache_refreshes` 是该次 Problem 的刷新次数。`visualPoseCacheMatchesUncachedTrials` 增加共享条目、外参/位姿试探点、不同输入地址及 partial-J/rho-only 对照，原解析/数值混合相机与 td 测试保留。新增内容未编译或执行。


### 2026-11-05 v12 CPU 求值路径

NIS 门控使用与 Ceres 相同的 CachedSphereAdapter，在每次门控前按当前图参数显式刷新独立缓存；不使用鲁棒核修正后的残差、不假定最后一个试探点已经接受。保留原门限、求解后检查、refit 和回滚。缓存按状态/相机共享世界旋转、逆旋转和位置偏移，再生成跨帧 pair；geometry 与 quaternion lift 分级准备，支持同一点 residual→Jacobian 的补齐。

新增 `[Timing][Solve] visual_pose_prepare_ms / visual_camera_prepare_ms / visual_pair_prepare_ms / gate_visual_prepare_ms / visual_cached_cameras`。前三项是该 Ceres Problem 的累计准备墙钟，包含在其求值计时内；gate_visual_prepare_ms 包含在 gate_ms 内。不能重复相加。

边缘化的退休扫描改由出窗观测索引与消元 rho 的并集驱动；继续保留私有候选、全量提交校验与原子回滚。`[Timing][Backend] marginalize_copy_ms / marginalize_validate_ms` 分别记录候选拓扑复制和最终校验，均属于 marginalize_ms；本轮没有移除窗口 checkpoint 或引入未经验证的共享可变拓扑。

`cachedVisualGateMatchesReference` 对比缓存/参考求解与 NIS 决策；原相机缓存测试继续覆盖相同点由 residual 升级到 Jacobian、partial-J、输入地址复制及变量外参。新增回归源码尚未编译或执行。


### 2026-11-05 v14：限额接受与估计失败分离

`SolveReport.termination` 为 `SolverFinished`、`TimeBudget`、`IterationBudget`、`RefitBudget` 或 `Failure`。三种额度结束均不会自动撤销最后可用解；先删除后验外点，再经过原有质量检查。`unrefittedOutliers` 说明接受状态有多少个刚删除的外点尚未通过另一轮求解消除其影响，0不表示全局最优。`postSolveResiduals` 对应清理后仍参与估计的因子。`[Backend][Limited]` 明确输出 `accepted=1`、停止原因、未重优化外点数及Ceres耗时。

FixedLagBackend先收集并归档Rejected因子，然后执行滑窗及新的线性化；这些外点不进入本次历史先验。既有非零 `J^T r` / Schur梯度传递保留。质量、标定、数值及先验形成失败仍回滚，限额并不绕过这些检查。扩展 `refitRecoveryAndSchurSafety` 覆盖时间、迭代、refit上限的接受和质量失败回滚；新增 `budgetLimitedSolvePreservesCleanPrior` 验证限额接受后正常发布、边缘化，且被拒绝的旧帧关联不把辅助变量信息写进先验。仅静态检查，未运行这些C++回归。


### 2026-11-05 v15：TrackingOnly种子参考与事务顺序

`VisualUpdateBatch.seedAnchors`只允许未初始化且没有rho/prior坐标的身份，选择该身份已有的新鲜TrackingOnly观测为参考。它不搬运已优化的逆距离或先验；已消费视图保持Absorbed。批次顺序改为回收、创建纯种子、加入观测、选择种子参考、初始化所有新/已有种子的rho，然后统一激活和校验。外部公共addLandmark(...,rho)仍在单次事务内完成初始化。非法参考、重复参考更新或稍后的初始化失败会整批回滚。

该顺序保证新锚观测先存在，且视觉成本只按最终参考构造一次，不在历史先验上作近似运输。前端参考更新还必须有当前观测支撑。现有prepareEviction继续保留具有存活观测的种子；seedRetirementAndFacade补齐SECOND_NEW移动种子但不创建rho/消费当前测量的回归。

### 2026-11-05 v16：连续索引与窗口复制/验证

LandmarkStore的`inFrame()`和`ofLandmark()`返回`ObservationKeys`（有序唯一vector）只读范围，主观测表和factor反向表仍是权威map。每地标的观测数受10帧及相机数约束，连续索引减少窗口checkpoint中每个观测的两份树节点分配。`inCamera(id, frame, camera)`用二分查找检查每相机每时刻唯一性。删除整帧先一次移除frame索引桶，再维护各小地标桶，避免对大frame向量逐点移动；tombstone仅删除对应frame的有序范围。有效性检查仍覆盖严格有序、相机/时刻重复、索引覆盖数、payload key、factor绑定和消费生命周期。

完整窗口验证没有关闭。对按frame排列的观测，特征payload只在换frame时查找一次；每条观测仍检查相机/关键点边界和因子依赖。边缘化直接接管已经私有的FrameStore，不再先复制一份随后覆盖；失败时从checkpoint移动恢复，避免失败路径额外拷贝整窗。原事务隔离、图检查、prior一致性和失败回滚均保留。

`associationFrame()`返回独立可写提案，描述子复制一次且省略图像；普通`exportFrame()`继续遵循retainFrameImages。在线外参参考变换和权威ID/坐标同步使用同一实现。`seedRetirementAndFacade`扩展同相机重复观测的整批回滚、索引查询与整帧删除后索引完整性检查。

`Retirement.observedEpochs`记录退休前仍保留的不同观测epoch数量（同epoch多个相机只计一次，不是完整历史寿命）。`[SeedRetire]`每30帧/VLOG输出无剩余观测、仅剩无效帧观测、至多两个保留epoch及平均保留epoch数，均按退休种子身份计数。有存活可用观测的种子仍由原prepareEviction迁移，未改动OLD/SECOND_NEW信息策略。

本轮仅静态检查，尚无v16运行时正确性或加速测量。
