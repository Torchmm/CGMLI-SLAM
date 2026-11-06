# Native visual/inertial SLAM pipeline

`ThreadedSlam` 直接调用当前 `ViSlamBackend`，不再保留旧双图后端的空声明，也不使用 Services 装配。目标是完成实际估计的数据链，允许为当前后端调整函数组织和执行次序。

## 数据链与模块职责

```text
图像/IMU 队列
  -> 静止初始化或 IMU 传播
  -> 各相机 ORB 提取（默认 GravityCameraAware + SparseBox）
  -> 显式 Body 状态、特征帧和真实 IMU 因子提交
  -> Frontend 几何共识、关联及多视图距离初始化
  -> FixedLagBackend 原子求解、OLD/SECOND_NEW 边缘化与成功快照
  -> 独立历史归档、可选全局回环
  -> 状态/地图点发布、可视化及 CSV
```

- `ViSlamBackend` 提供原生状态/地标接口；`SlidingWindowManager` 管理窗口、帧、观测和地图点；`BackendOptimizer` 执行求解。
- `estimator_parameters.window_size` 是唯一的局部窗口容量参数，默认保留 10 个图像组状态。新增帧临时占用第 11 个槽，优化后按 OLD/SECOND_NEW 边缘化回到 10 帧。IMU 测量作为帧间约束，不另建 IMU 帧配额；视觉退化时保留同一图像节点上的惯性预测和已有约束。
- `factor/ImuError` 是工程已有的 IMU 初始化、传播及预积分残差实现。
- `factor/BodyImuFactor` 将这些方程接入当前 Body 状态布局，检查原始时间覆盖、处理固定 `T_BS` 和速度杆臂，并在共享代价前固定预积分与白化。代价的数值定义不会因一次求解失败或历史先验线性化而改变。
- `StateOutput` 只处理坐标转换、覆盖率与文件输出。
- `GlobalSlamGraph` 拥有独立的历史状态、原始观测和关键帧特征；不向已经边缘化的局部图重复提交旧观测或覆盖其先验。

不再声明 `addImu/addCamera`、原始 IMU 版 `addStates`、`optimiseRealtimeGraph/applyStrategy`、`poseSensor` 或双图同步等旧兼容接口。当前原生接口已有相应职责。

## 初始化与支持范围

启用 IMU 时，启动需要最近 0.5 秒中至少 0.4 秒、至少 10 个有效样本。检查加速度/角速度方差、重力模长与平均角速度，满足静止条件后初始化姿态、零速度及陀螺偏置；加速度偏置使用配置先验。运动启动会继续等待，不会伪装成已经完成 VIO 初始化。当前没有动态启动的重力/尺度/偏置联合初始化器。

没有 IMU 时，需要参与跟踪的相机具有非零已标定基线。纯单目、无 IMU/深度的米制 SLAM 不在当前入口支持范围，会明确拒绝；不会使用五点算法的单位平移生成米制深度。带真实 IMU 边时，经过共识的时间观测可使用 IMU 的米制预测进行距离初始化，仍检查正射线距离、视差与全部观测一致性。

IMU 必须覆盖图像时间之后至少 20 ms；时间戳必须严格递增，输入单位为 rad/s 和 m/s²。比例校正在 `addImuMeasurement` 只执行一次。异常和有限性检查不会把损坏数据变成成功观测。

多种相机通过各自虚拟几何接口接入。EUCM 的单位射线解析 Jacobian 已补齐，两个反投影重载采用一致有效域；反投影失败不会继续进入重叠图或噪声计算。Pinhole/EUCM 的齐次 Jacobian 行赋值也已修正。

## 坐标与发布

后端状态为 `T_WB`、Body 原点的世界速度、Sensor S 坐标的偏置。`T_BS` 把 S 映射到 B。IMU 方程和公共回调使用 `T_WS=T_WB*T_BS`，速度同时转换 `omega × lever_arm`，不是只转换位姿。

成功快照中的公共 Rig 外参与固定相机基线恢复当前 `T_SC`；提取、关联和优化使用相同相机模型。在线外参估计必须有正先验，并且仅在窗口条件信息通过门限后解冻公共 Rig 外参；时间偏移保持固定。

后台线程负责一次完整后端事务。稳定运行时，控制线程先用上一次已接受的状态缓存传播到当前图像时间，并提取当前帧 ORB；这一步与上一帧的后端求解、边缘化及历史图处理重叠。提取结束后才 join，刷新已接受状态，再从最新状态重新传播，用于当前帧的数据关联、真实 IMU 因子与状态插入。随后启动当前帧的优化。

```text
控制线程：提取 N（缓存预测） -> join N-1 -> 最新预测 / 关联 / 插入 N -> 提取 N+1 ...
后端线程：        优化 N-1 --------┘                         优化 N --------...
```

位姿缓存与不可变外参快照在 join 后一起更新；提取不读取正在写入的后端，不会将旧状态预测与刚发布的新外参混用。同一帧的所有相机使用同一缓存版本。GravityCameraAware 使用提取时的预测重力方向，join 后更新关联预测不会重新生成描述子；因此并行提取的重力方向仍受此前状态与偏置预测误差影响。

首次优化尚未产生可用状态缓存时提前等待。图像已经排队但 IMU 未到齐时直接返回，让生产者继续供给 IMU，同时后端继续运行；没有新图像或停止时则会收取上一帧结果，保证回放结尾能正常排空。`preparedFrame_` 单独保存已提取、尚未插入的下一帧；前一帧求解失败时先保留它，在配置次数内重试原 staged packet，不追加第二个后端插入槽，也不重复提取或关联。重试耗尽后恢复插入前的完整后端检查点，移除失败帧的活动 IMU/角速度缓存，并重置前端跟踪诊断；保留下一帧和原始 IMU，允许下一帧从上一个已接受状态继续传播。失败帧不会发布为估计，也不会把失败的 Schur 先验带入下一帧。

这里恢复的是特征提取与后端的重叠；关联、插入和后端写操作仍串行。实际帧率取决于提取、局部求解和历史图处理的耗时，尚未做运行计时。

## 输入完整性、门控与失败恢复

相机配置按 `cam_model` 校验。EUCM 要求 `eucm_parameters: [alpha, beta]`（alpha 属于 [0,1]，beta > 0），不要求针孔畸变字段。针孔相机允许 `distortion_type: none/None/no_distortion/NoDistortion` 或省略该字段，畸变向量可省略或全为零；equidistant/radialtangential/plumb_bob 必须提供四项，radialtangential8/plumb_bob8 必须提供八项。所有相机都检查完整有限的 4×4 刚体外参、正焦距和正图像尺寸。未知模型或不合法标定会使整个配置失败，不能跳过某相机后继续改变图像索引。重新读取配置前先清空旧 rig，读取失败时 `getParameters()` 返回 false。

参考 [OKVIS2-X 的 Frontend.cpp](https://github.com/ethz-mrl/OKVIS2-X/blob/main/okvis_frontend/src/Frontend.cpp) 的 RANSAC、优化后剔除和跟踪质量分层处理。本项目继续使用 PoseLib 和球面残差；有限重试、事务回滚、以下物理门限以及按模型修复配置校验属于本项目的实现，不能视为原样复制了上游全部行为。上游配置入口也有旧分支限制，因此没有直接照搬其读取代码。

默认参数及可选 YAML 字段：

```yaml
imu_parameters:
  max_sample_gap: 0.05                 # 相邻物理 IMU 样本允许的最大间隔，秒
estimator_parameters:
  backend_max_retries: 1                # 首次求解失败后最多额外尝试一次，允许 0..10
  visual_nis_threshold: 9.210340371976184 # 二维白化球面残差的平方门限
  max_visual_imu_rotation_error: 0.2     # 与 IMU 预测的最大姿态差，弧度
  max_visual_imu_position_error: 0.5     # Sensor 原点位置差的固定容限，米
  max_visual_imu_position_error_rate: 2.0 # 位置容限随间隔增加的速率，米/秒
```

视觉绝对位姿和相对位姿分别经过与真实 IMU 预测的一致性检查，绝对位姿不通过后仍会尝试相对位姿。比较使用 `T_WS=T_WB*T_BS`，避免把非零杆臂当成位置误差。拒绝的视觉位姿不能继续用于过滤、三角化或初始化。上述位置/旋转门限是可配置的工程包络，不是传播协方差的卡方检验。

后端在初始与拟合后两个阶段检查原始白化视觉残差；NIS 不会被 Huber 损失衰减后再计算。求解后质量验证器再次比较原始 IMU 预测，限制单次速度变化为 5 m/s、陀螺偏置变化为 max(0.1 rad/s, 5×sigma_bg)、加计偏置变化为 max(1 m/s², 5×sigma_ba)。若输入时已有至少十个独立有限视觉地标且视觉共识成功，剔除后也必须保留至少十个；IMU-only 和尚无有限深度的合法启动帧不受这一地图支持门限限制。门限是初始默认值，需结合实际噪声和运动范围评估。

初始化、传播和 IMU 因子构造统一检查物理样本的时间间隔，只检查与本次积分区间相交的相邻样本。已接受状态之后出现不可跨越的样本缺口，会令当前实例进入终止错误状态：`hasFatalError()` 为 true，`lastError()` 保留首次原因，后续输入返回 false，旧优化线程仍会安全收取。必须停止该实例并显式建立新的初始化段；当前未实现动态重初始化，不会把跨缺失区间的预测当作连续 VIO。回放入口会明确报错退出。

回滚仅丢弃未提交帧；连续丢弃可能使有效帧率下降，它不等同于已经实现完整的丢失重定位。IMU 缺口引起的分段也不自动声明两个世界坐标系连续。

两种边缘化策略都保存删帧前最终状态。`localHistory` 保留局部世界坐标的预测依据，`outputHistory` 保留输出世界坐标的轨迹，兼容次新帧移除后的非连续 ID。历史 raw IMU 仅在启用全局归档/最终 BA 时保留；活动缓存随窗口释放。

## 全局回环与最终 BA

`estimator.do_loop_closures` 启用历史回环。只查询活动窗口以外、至少间隔 15 个关键帧及 5 秒的历史关键帧。提供 profile 兼容的 ORB vocabulary 文件时使用独立 DBoW2 全局索引及 `p_dbow` 门限；未提供时使用确定性哈希词频产生候选。后者是基本候选检索，其召回率和性能尚未通过数据集评估。

候选还必须通过双向 ORB 过滤、至少 20 个独立三维点的广义 GP3P/LO-RANSAC、内点比例、图像覆盖、条件信息和连续两次候选一致性检查。启用 IMU 时，全局位姿图只调整世界平移及 yaw，保留重力方向；视觉模式使用 SE(3)。回环在独立图求解成功后修正输出轨迹和地图，不把局部候选重复当作独立约束。全局处理目前在同一个估计 writer 顺序执行，尚无实时耗时保证。

`estimator.do_final_ba` 在采集前启用原始历史归档。`doFinalBa()` 在停止输入和局部优化后，从原始球面视觉观测及真实 IMU 因子重建 BA，联合优化历史位姿、速度偏置和三维点；不会再次加入局部边缘化先验或位姿图里程计。回环建立的轨迹别名统一三维点，每个点/相机/采样时刻仅保留一个观测。可按配置优化一个公共 Rig 外参，固定各相机基线。

位姿图里程计采用明确的近似噪声模型；回环信息由固定地图条件信息保守缩放，不宣称等同完整联合协方差。最终 BA 与局部估计都有实际实现，但当前验证范围仍是静态检查。

## 调用与回放

v13：`estimator_parameters.enforce_realtime` 单独决定是否启用 Ceres 软求解时间预算，与离线队列的 `setBlocking(true)` 无关。设为 true 后，`realtime_time_limit: 0.04` 完整传给后端，表示同一次优化事务内所有 `ceres::Solve()` 调用累计共享 40 ms。不扣除前端 stage、NIS 门控、问题构建、事务、诊断、边缘化和发布时间；这些分别记录墙钟耗时。`[Timing][Solve] ceres_budget_ms` 报告额度（0 表示不限制），`ceres_budget_reached` 报告 Ceres 累计耗时是否到达额度。最低迭代数与不可中断的单次迭代可能使求解超时，数值收敛也可能提前退出，因此不是整帧 40 ms 硬期限。v14中，有剩余额度时继续refit；额度结束时，清理后图通过原有质量检查则接受最后可用状态，并输出 `termination` / `unrefitted_outliers`，不再仅因额度耗尽而丢帧。真正估计失败仍走回滚。外层可重试失败仍按每次调用领取额度，并非跨重试的硬截止时间。

启动日志 `[BackendPolicy]` 输出长 IMU 区间门限、增量积分、OLD 信息保留、Dense Schur、关键帧新点和求解预算设置。长边跳过后用 `[ImuIntervalSuppressed]` 确认，比较性能时以同一数据区间、同一可视化开关和同一预算为准。

```cpp
cgmli::ThreadedSlam slam(parameters, orbVocabularyFile); // 可传空字符串
slam.setOptimisedGraphCallback(callback);              // 在 processFrame 前配置
slam.addImuMeasurement(timestamp, acceleration, gyroscope);
slam.addImages(timestamp, imagesByCamera);
slam.processFrame();                                 // 在同一控制线程持续调用
```

`processFrame()` 的 false 可以表示等待数据或求解失败；用 `lastError()` 读取原因。单线程回放使用 `hasPendingImages()` 检查当前图像是否尚未消费，消费后即可输入下一帧；使用 `hasPendingWork()` 在 EOF 排空全部任务。若每帧都排空全部后端任务，会失去跨帧的提取/优化重叠。停止、导出和最终 BA 在控制线程调用，不从发布回调调用。

Hilti 离线应用采用完整四目平均时间节点，调用 `addImagesWithTimestamps(meanTime, imagesByCamera, originalTimes)`。四目节点时间为平均曝光中点加上 `image_delay`。`MultiFrame::imageTiming(camera)` 保存各路原始曝光中点及其相对平均值的偏移。原来的 `addImages()` 接口将组内偏移设为零。

在 `camera_parameters.online_calibration` 中，`do_extrinsics` 控制刚性相机组外参，`estimate_td_camera` 控制公共时偏及球面运动补偿。打开时偏估计后，有可信同相机跨帧轨迹的观测按 `-(imageTimeOffset + td_camera)` 修正到固定节点，三角化与视觉因子使用同一模型。`td_camera` 是在线残余量，原始相机时间到 IMU 时间的总修正为 `image_delay + td_camera`。日志 `[Calibration]` 分别用 `td_camera_s` 和 `td_camera_total_s` 输出残余量和总修正量。

标定使用相对初值的绝对范围。时偏范围直接作为 Ceres 参数边界。如果外参候选越界，流水线在重试时固定对应标定块，继续估计其余已释放的变量，并重新检查质量。日志以 `retry_with_violating_blocks_fixed` 标记该路径。

在线标定根据最近窗口的运动和当前活跃跨帧轨迹释放参数。至少需要 10 条独立跨帧地标轨迹。旋转要求多轴陀螺激励，时偏还要求 10 条具有球面切向速度的轨迹。时偏的运动条件可以由旋转激励或带速度变化的平移满足。平移在旋转和已启用的时偏各积累 3 次有效更新后释放，并要求足够的位移和运动方向变化。`[Calibration]` 的 `temporal_tracks` 和 `timed_tracks` 计到释放门限为止，`gate_ms` 记录判断耗时。条件信息矩阵仍可通过后端诊断接口显式计算。

`skip_pre_nis_when_tracking_healthy` 只在健康的首次求解跳过优化前扫描，求解后仍在边缘化前检查。`parallel_marginalization_accumulation` 允许较大问题按线程累积正规方程。日志分别记录前后 NIS 及边缘化装配、因子求值、正规方程累积的时间。

`output.display_matches` 默认关闭；设为 `true` 时统一启用四目特征、同组匹配、跨帧匹配和后端重投影四个命名视图。关联诊断保存实际 guided matcher 的端点索引，并区分前端保留/丢弃候选，不将全部绿色关联宣称为 RANSAC 内点。控制线程在提取结束后缓存独立缩略图，绘图线程只读快照；核心后端仍可以关闭原图保留。`display()` 返回命名图像，由应用主线程调用 HighGUI；Hilti 入口支持暂停及 PNG 截图，详见 `src/offline/README.md`。

应用入口已替换原 Hello World：

```text
CGMLI-SLAM.exe config.yaml images.csv imu.csv|- [trajectory.csv] [map.csv] [orb_vocabulary.yml]
```

`images.csv`：

```csv
timestamp_ns,path_camera0,path_camera1,...
1000000000,cam0/0001.png,cam1/0001.png,...
```

相机列数必须与标定一致；路径相对于 images.csv，空列或 `-` 表示该相机本次没有图像。路径含逗号时使用 CSV 双引号。

`imu.csv`：

```csv
timestamp_ns,wx,wy,wz,ax,ay,az
500000000,0,0,0,0,0,9.81
```

输入 CSV 按时间递增。驱动先排入图像，再逐个补充 IMU 至正确的后向覆盖，避免阻塞模式的生产者死锁；图像消费后继续读取下一帧，EOF 才排空全部后端任务。仅得到惯性/种子状态、没有成功视觉跟踪时，应用返回错误，不宣称已经完成视觉 SLAM。

## 范围与验证

GPS 已从活动代码移除。LiDAR/深度子图和 SE2 导出仍按此前约定暂缓，相关入口明确返回 unavailable/false。历史视觉点 CSV 不代表 LiDAR Stable Map；Stable Map 的原有门控不变。`legacy/*.reference` 仅作为原始对照。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File src/pipeline/tools/check_pipeline_static.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File src/backend/tools/check_backend_static.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File src/frontend/tools/check_frontend_backend_static.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File src/frontend/tools/check_frontend_orb_static.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File src/frontend/tools/check_frontend_pose_static.ps1
```

`tests/NativeSlamRegression.cpp` 提供模型配置校验、IMU 缺口/原子传播、视觉惯性包络、确定性求解失败后的有限重试/下一帧保留，以及终止错误状态的回归源码。原有 EUCM Jacobian、静止初始化、非零杆臂 IMU Jacobian、跨窗口发布、历史球面/IMU BA 和历史回环回归仍保留。它作为 None 注册，未编译或执行。静态检查覆盖提取/等待顺序、写入边界、检查点回滚、终止错误和这些回归的入口调用；这些检查不能替代运行时竞争检测。源码连通性不等同于成功编译、真实闭环召回、精度或实时性验证；目前没有真实数据集 ATE/RPE 结果。


v14配置清理：移除CNN开关、Frame/MultiFrame分类接口及空CNN线程接口；移除旧BRISK/Harris阈值、无作用的keyframe_overlap访问接口、同步列表、深度网络校正入口、旧回环帧配额/漂移启发式和未实现的topview/submapping输出开关。实际在线/最终外参标定、回环、最终BA及IMU/LiDAR标定路径仍保留。已有四相机200点、最大8次迭代、GUI和时间限制开关的用户设置不重置。


v15：正常非关键帧也接纳有界TrackingOnly新轨迹；稳定参考后再创建rho。启动日志BackendPolicy输出keyframe_anchor_promotion、max_tracking_seeds及seed_max_untracked_s，旧keyframe_only_new_landmarks参数已由稳定锚点晋升策略替换。默认种子接纳1000、失跟0.5秒，40/1000优化地标晋升预算继续有效。配置不改变已有200点/相机、8次最大迭代、Ceres时间限制及GUI开关。匹配图绿色包含种子和成熟点，不能作为成熟地标数量。

v16增加SeedContinuity、SeedRetire分项诊断。前者区分已有种子候选、分配延续、筛选后提交和空间补点限额；后者区分无剩余观测退休与仅剩无效帧支持，并统计退休时保留的观测epoch数量。fallback日志的relative_attempted=0、relative_skip=rotation_dominance表示原选择规则必然采用旋转而省去未使用的六点解；不是初始化成功或尺度可观测的标记。40/1000晋升配置、Ceres预算和10帧窗口未调整。本轮只做静态检查。
