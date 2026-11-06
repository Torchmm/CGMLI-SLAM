# 分模块正确性调试

目标是用已知输入、明确真值和数值对照定位问题，再用真实 Hilti 数据观察模块组合后的行为。不能用“Ceres 接受了”“代价下降”“quality=good”替代模块正确性验证。

## 独立程序

解决方案新增 `CGMLI-ModuleDebug` 项目，生产程序及其正常构建保持独立。调试项目使用当前生产源文件、同一套依赖与六份现有回归源码；没有另写一套算法来通过测试。

1. 在 VS 打开 `CGMLI-SLAM/CGMLI-SLAM.sln`，选择 **Release / x64**。
2. 右键 `CGMLI-ModuleDebug` → **生成**，需要 F5 时再设为启动项目。该项目没有加入普通解决方案构建的默认勾选项。
3. 输出位于 `CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe`，与正常程序共用 DLL 搜索目录；中间文件使用独立目录。

在工程根目录执行：

```powershell
# 查看可选模块和全部用例
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --list

# 优先检查当前最需要确认的数学环节
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module jacobian
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module marginalization

# 单独检查一个用例，便于断点调试
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module jacobian --case analyticSphereDerivativeAgreement

# 其他模块，或完整受控回归
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module extraction
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module matching
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module initialization
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module backend
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module tracking
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module all
```

VS 的调试命令参数可填写 `--module jacobian --case analyticSphereDerivativeAgreement`。模块参数后可加自定义输出根目录，例如 `--module jacobian "H:/BaiduSyncdisk/CGMLI-SLAM/debug_results"`。调试程序不要求 Hilti 数据目录，不会自动运行完整 VIO。

退出码：`0` 为选中用例全部通过，`1` 为有用例失败，`2` 为参数/报告写入等运行错误。普通异常会记录当前用例失败并继续；断言终止、崩溃或强制结束可能留下不完整报告，不能视为通过。

## 模块与证据

| 模块 | 受控检查 | 数值或可视化证据 |
|---|---|---|
| `extraction` | 掩码、金字塔、重力滚转、极点、相机模型、描述子重复性、预算/网格约束、共享 FAST 与 ROI 对照 | `adaptive_features.png`、`reference_features.png`、关键点 CSV；均标注为合成图像 |
| `matching` | 稀疏引导与稠密 BF 对照、ratio/mutual、地图点身份去重、相机重叠约束 | 合成匹配图、对应索引和 Hamming 距离 CSV；不是同名真实场景匹配截图 |
| `initialization` | 理想静止 IMU 的重力姿态、运动启动拒绝、缺测时的原子性、相机和时间约定 | 初始位姿/速度/bias CSV、重力旋转误差 |
| `tracking` | 已知几何的绝对/相对位姿、内点映射、前后端对接、失败帧回滚、下一帧保留 | 位姿和关联断言；静止空图流水线输出轨迹 CSV，不代表已覆盖真实纹理运动跟踪 |
| `backend` | 已知标量解、剔除与重拟合、串行/并行 NIS 一致性、非有限值/异常回滚、观测所有权 | 解误差、PASS/FAIL 与用例日志 |
| `jacobian` | 球面视觉解析/数值雅可比、同帧共享消项、后半球、非零外参与射线速度、时间偏移、IMU 杆臂 | 每个 ambient/tangent 分量的解析值、差分值、绝对误差 CSV；误差与明确上限 |
| `marginalization` | 线性问题的独立 Schur 真值、条件最小二乘二次型一致性、先验耦合、最老/次新帧、10+1 槽 | 信息矩阵/梯度误差、不同保留变量扰动下的完整/边缘化代价 CSV |

注册了 **93 个受控用例**。成功只说明这些用例覆盖的条件成立，未覆盖的运动、退化、标定误差和真实数据问题仍需验证。原 standalone main 通过宏守卫保留；各模块有独立夹具，避免复用上一个模块的后端状态。

## 2026 年 11 月 3 日加强后的验证

v10 源码验证版本为 `2026-11-04-v10`，共 93 个用例。v10 加强 `oldestSchurKeepsIdentityWithoutReusingEvidence`：同一 ID 用新 stereo 成熟后，再删除含已消费观测的旧帧，新 rho 和原 live cost 必须保留。`promotionSurvivesImpendingEviction` 同时检查出窗锚点、存活锚点加有效出窗支撑、可正常晋升的正向对照。新增 `constraintGradientBalanceIsRobustAndReadOnly` 对照 robust 目标中心差分及 Ceres 当前流形切空间梯度；原线性 Schur 用例检查输入二次项、消元能量与常数项。数值误差写入 `derivative_checks.csv`，本轮没有编译或执行 C++；旧 PASS 不覆盖 v10。

v9 的超长 IMU 排除、非整采样时间增量积分、OLD H/b 与联合 Schur 对照、隐式锚点防重用、试探点位姿缓存、非关键帧已有 ID 延续、Dense Schur 耦合先验回退等用例全部保留。下文 v4～v8 为验证历史。

v10 数值附件新增 `prior_formation.csv` 和 `constraint_gradients.csv`。真实数据的 `[PriorFormation]` 区分旧 prior 在当前点的值与新 prior 的参考值；`[ConstraintBalance]` 在 840/1920 帧按同一切空间比较 visual/IMU/prior 梯度。先检查各范数和 cosine_defined，只有定义成立时才解释 cosine_with_imu 的正负；接近 -1 只表示局部约束方向相反，不足以认定某传感器错误。`[ImuConflict]` 的各三维 marginal_nis 不能相加。完整定义和性能方案见 `../backend/README.md` 的 v10 部分。

报告 HTML、status.json 和 glog 标注 `validation_policy=2026-11-03-v4`。请重新生成调试程序；旧报告不会被修改，旧的 PASS 不代表已通过新策略。v3 的 35 帧静止视觉、真实窗口残差统计和 refit 检查保留；v4 新增 `redundantFrameKeepsTracksSparse`、`untouchedPriorAndDiagonalKernel`、`nativeSecondNewestImuReplay`。对应 CSV 为 `redundant_window.csv` 和 `second_newest_imu_replay.csv`：检查多次次新帧删除后的成熟点/种子 ID、prior 地标数；合并 IMU 与独立整段积分的残差/雅可比、重复边界采样与冲突回滚；原 prior 指针/冻结参考和秩截断。新增用例本轮未编译或执行。

- 匹配：将“结果必须非空”与“候选遍历必须减少”分开。严格双向 ratio 下允许因邻居不足正确拒绝；开启 single-candidate 的组合必须恢复非空匹配作为正向控制。`matching_policy_combinations.csv` 列出全部八种策略组合的参考/实际数量和遍历次数；`single_candidate_matches.png` 展示正向控制，`guided_matches.png` 展示严格策略结果。
- 调度：检查扩展后的任务数量、成熟点关联与完整参考一致，不再要求扩展阶段已经完成全部工作时仍进入 broad 阶段。`coverage_fallback.csv` 给出实际阶段与任务数。
- 雅可比：普通块使用 `1e-7 + 1e-6 * reference_norm`，视觉 CameraExtrinsic/CameraTimeOffset 复杂块使用 `1e-7 + 1e-5 * reference_norm`。EUCM 像素反投影绝对底限为 `1e-10`。`derivative_checks.csv` 明列参考范数、相对容差和绝对底限；零导数检查依赖绝对底限，不用除以接近零的范数。
- 位姿：新增绝对/相对位姿各三个用例：无初值无噪声、无初值带噪声/外点、错误初值带噪声/外点。每个用例三个固定种子，每次 160 条具有四个不同相机中心的射线；污染场景约 `1e-4 rad` 切向噪声、20% 大角度外点。噪声为切向高斯分量尺度，不等于每条射线角误差严格固定为该值。输出 `pose_recovery.csv` 与逐行射线/外点真值/内点掩码 CSV。成功还必须有最小模型生成和 RANSAC 迭代，不能只接收真值初始模型。
- 位姿验收：无噪声平移误差 ≤ `1e-5 m`、旋转误差 ≤ `1e-6 rad`；污染场景绝对位姿平移 ≤ `0.01 m`、相对位姿 ≤ `0.03 m`、旋转 ≤ `0.003 rad`；内点 precision ≥ 98%、recall ≥ 90%。这些是固定受控场景的验收值，不是对真实 Hilti 数据的精度承诺。
- 换锚：`reanchorInformationAudit` 检查不变图通过、损坏历史 prior 被识别；`reanchorFixedBearingChangeIsRejected` 检查名义点相同但信息不等价的候选被拒绝，`equivalent=0` 是此反向用例的正确结果；`reanchorFullCoordinateTransportReference` 验证可逆三维参考变换的 Jacobian、冻结 prior 与 Schur 一致性。这不意味着当前单逆深度生产表示已完成完整换锚；默认仍关闭，显式开启也必须通过信息门控，否则安全退休。

推荐重新运行：

```powershell
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module matching
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module tracking
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module jacobian
& './CGMLI-SLAM/x64/Release/CGMLI-ModuleDebug.exe' --module marginalization
```

`--case Recovery` 可仅运行六个新增位姿用例，`--case reanchor` 可仅运行三项换锚验证。函数内普通异常会结束该用例，所以同时检查测试覆盖是否完整，不能把后续未执行的子场景视为通过。

## 报告与本地读取

默认保存在工程 `debug_results/run_日期_时间_序号/`：

- `report.html`：模块状态、异常信息、用例耗时、图片预览、数据链接，可勾选只看失败。
- `cases.csv`：逐用例 PASS/FAIL、耗时和原因。
- `status.json`：选中数量、完成数量、失败数量及 `complete`；不完整运行明确标注。
- `module.log`、`arguments.txt`：该次调试日志和参数。
- 每个用例的子目录：`metrics.csv`、逐项数值和 PNG；没有数值导出的用例以源码断言和日志为证据。

`debug_results/latest.txt` 指向最新报告。之后可直接要求“分析最新模块调试报告”，无需上传。模块日志与正常 VIO 的 `logs/latest.txt` 分开，避免把负向用例预期触发的警告当成真实 SLAM 故障。

```powershell
$report = (Get-Content './debug_results/latest.txt' -Encoding UTF8 -TotalCount 1).Trim()
Start-Process -FilePath $report
```

测试计时包含数据准备和证据写入，不用于衡量生产算法性能；大量雅可比 CSV 也会有开销。

## 与真实数据调试结合

受控模块验证后，用现有离线程序回放 Hilti 数据。需要图像证据时在 `config/hilti_robot_estimator.yaml` 设置 `output_parameters.display_matches: true`，通过特征、时序、同帧相机匹配和后端重投影窗口观察，并用 `S` 保存 PNG；说明见 `src/offline/README.md`。几何结果应同时检查内点、残差、深度/视差和跟踪连续性，不能只凭连线看起来整齐就判断正确。

早期日志曾在第 218～251 帧出现连续拒绝；2026-11-04 的后续运行已经持续提交，因此不能将旧故障描述视作当前状态。仍需确认动态雅可比、后期 IMU/prior 残差和地标生命周期。真实失败窗口的完整序列化重放尚未在此入口实现。

## 静态检查和项目同步

```powershell
python src/debug/tools/check_module_debug_static.py
```

检查项目注册、main 宏守卫、模块函数和证据通道，不编译、不执行 C++，不能替代上述数值运行。现有正常项目仍不编译测试源码。若更改了生产项目的库路径或编译配置，执行 `python src/debug/tools/generate_module_debug_project.py` 同步独立调试项目，再手动生成；脚本不执行编译。

## 真实数据轨迹、跟踪与 IMU 联动调试（v8）

在 `config/hilti_robot_estimator.yaml` 的现有 `output_parameters` 中选择：

```yaml
output_parameters:
  display_matches: true    # 01/02/03/04/06：提取、匹配、重投影、已接受的 ID 轨迹
  display_overhead: true   # 05：已接受的完整局部轨迹 XY/XZ + 当前滑窗
```

独立开关可以只启用轨迹，不复制四路图像。代码默认值均为 false；本轮保留用户 YAML 中已有的选择。`display_topview` 不控制新增轨迹窗口，使用 `display_overhead`。

| 窗口 | 数据阶段与读图方法 |
|---|---|
| 01 ORB Features | 提取结果，检查点数、空间覆盖、方向及边缘分布 |
| 02 Stereo Matches / 03 Temporal Matches | 前端关联候选与保留结果，标题中的 committed 是前端事务提交，不等于后端接受 |
| 04 Backend Reprojection | 后端接受后状态的投影残差，检查残差方向是否系统一致、边缘是否明显更差 |
| 05 Trajectory | 后端接受的局部 S 原点轨迹；XY 检查平面路径，XZ 检查高度漂移；灰线是历史、黄色是当前窗口、黄环是窗口关键帧、绿色是当前帧 |
| 06 Feature Tracks | 后端接受后的关联 ID；绿点为成熟点、蓝点为种子、橙点为最近重置；每相机显示最长的最多 64 条短轨迹和最多 12 个 ID 标签 |

06 标签 `123 i2 r3 d1` 表示地标 123 累计初始化 2 次、重置 3 次、降级 1 次。换锚只增加 r；真正降级后再次成熟才增加 i。先观察同一 ID 是否稳定，再观察轨迹是否突然跨物体、跳变。短轨迹保存最近 12 个**实际渲染**位置；显示队列允许丢旧包，所以不能把它当逐帧完整光流。30 个帧 ID 未再见到的轨迹会释放。05 的历史最多均匀采样约 8000 个位置，当前滑窗独立保留。

绘图在可视化线程，队列满时丢旧显示包，不阻塞 Ceres；开启图像视图仍会增加图像复制、投影和 CPU 开销，因此性能基准应关闭两个开关。01～03 可以领先于后端，04～06 的接受帧号用于判别是否同一时刻。按 SPACE 暂停/继续，S 保存当前窗口 PNG 到输出目录的 `debug/snapshot_*`，ESC 正常停止并导出轨迹。`trajectory.csv` 保存完整接受历史；实时视图使用局部坐标，不做真值对齐，也不将稀疏 Hilti 控制点当连续轨迹真值。

运行日志新增：

- `[Window] repeated_resets/repeated_demotions`：本次删帧中，生命周期里已经发生过同类操作的点数。
- `[LandmarkLifecycle]`：ID、旧/新锚点、支撑观测数、累计次数、上次变化帧、点位移动和最大保留 NIS。默认每 30 帧最多采样两个重复 ID；完整身份事件需设置环境变量 `GLOG_v=1` 后运行。`transaction=frontend_accepted` 是前端暂存成功，仍需该帧后端接受；`transaction=accepted` 是已接受的滑窗处理。
- `[Window] reset_prior_unchanged=1`：锚点重置准备阶段没有更换输入冻结先验、没有改变其引用坐标；随后正常边缘化仍可生成新先验。这不是“重置前后完整信息等价”的证明，因为被丢弃的旧图像测量没有保留。
- `[ImuJacobianCheck]`：对实际旧 IMU 边，比较当前解析/链式因子雅可比与中心差分；覆盖两个左 SE3 位姿和两个 SB 共 30 列。`evaluated=1 failed_columns=0 max_scaled_error<=1` 才表示本次差分通过，容差为每列 `1e-6 + 1e-6 * numeric_norm`。
- `[ImuIntervalCheck] / [ImuImageTiming]`：原始 IMU 包围、区间最大缺口、各相机原始时间和节点偏移。
- `[ImuTimeSensitivity]`：固定状态，只将积分区间整体移动 ±1 ms。只在原始样本覆盖时评估；NIS 降低表示该方向敏感，不是标定结果，不能据此自动修改时间偏移。
- `[ImuCalibration]`：本轮实际使用的连续噪声密度、随机游走、重力和 T_BS。组内相机时间补偿仍按此前选择关闭。

实际 IMU 差分/时间敏感性在第 300 帧，以及其他每 30 帧采样点中最差旧边 NIS > 30 时执行，保持只读。`[Timing][ImuAudit]` 单列额外耗时。用 `GLOG_v=1` 获取完整生命周期时，不应将该轮耗时与普通运行直接比较。

建议分两轮验证：先关闭图像视图检查差分结果、prior 保护、重复生命周期计数、分项耗时；再开启 05/06，在异常帧附近暂停保存图像，对照同帧的 IMU、prior 和地标事件。后续可再增加轨迹点选跳帧、按 ID 筛选、日志曲线和失败窗口重放；这些交互功能本轮未实现。


当前新增验证版本为 `2026-11-04-v11`，共 96 个源码用例。新增绝对位姿解析/中心差分导数对照、错误预测下含噪外点恢复与预算健康门控、候选分批点数与空间/S² 覆盖检查；原缓存试探点与 FAST 用例扩展 partial-J、缓存共享、边界条带及 flat/grouped heap 比较。新增 CSV 为 `absolute_analytic_jacobians.csv`、`adaptive_geometry_batches.csv`。本轮仅通过静态接线检查，未编译/运行 C++；原有 PASS 不覆盖 v11。


当前新增验证版本 `2026-11-05-v12`，99个源码用例。增加 `cachedVisualGateMatchesReference`、`healthyPromotionBudgetPreservesSeeds`、`batchedDescriptorCoordinatesMatchScalar`；扩展缓存同点分级刷新与LO微小步退出对照。静态接线通过不代表这些数值用例已通过，本轮没有编译或执行C++。

当前验证版本更新为 `2026-11-05-v13`，仍为99个源码用例。`perStepBudgetAndHistory` 增加“60 ms前置NIS耗时不扣除20 ms Ceres额度”的回归子场景。该检查只断言能够进入并接受求解，不断言精确运行耗时；本轮未编译或执行C++，已有v12报告不覆盖此更改。


当前版本 `2026-11-05-v14`，100个源码用例。扩展refitRecoveryAndSchurSafety检查限额接受、清理后的统计、时间/迭代上限和质量失败回滚；新增marginalization/budgetLimitedSolvePreservesCleanPrior检查限额接受后Rejected因子不进入新先验。源码静态接线检查通过不等于数值回归通过，本轮未编译或执行C++。


v15历史版本 `2026-11-05-v15`，102个源码用例。nonKeyframeRetainsSeedsWithoutRho替换旧的“非关键帧只留已有轨迹”用例，覆盖非关键帧创建种子、下一帧复用ID、关键帧选择参考后成熟。新增seedAdmissionIsBoundedAndExpiresAtomically、seedReferenceUpdateCannotRewriteOptimizedGeometry，覆盖独立种子容量、空间均衡、满容量仍延续已有ID、超时回收与整批回滚、成熟坐标/Absorbed测量禁止改作新seed锚点；seedRetirementAndFacade扩展SECOND_NEW保留种子ID。几何参考fixture每帧相隔1秒，显式关闭其失跟计时，超时用例单独打开0.5秒阈值。本轮仅静态接线检查，未编译或执行C++，旧v14数值报告不覆盖这些变更。

v16更新：当前验证策略为`2026-11-05-v16`，105个源码用例。新增existingIdentityWinsBeforeNewTexture、seedReplenishmentPreservesExistingCoverage、rotationAnalyticAgreementAndRecovery，分别覆盖同次Hamming遍历下的旧ID延续与严格门限、空间补点及独立提案、旋转S2解析导数与带外点数值参考恢复。seedRetirementAndFacade补齐连续索引的重复观测回滚和删帧一致性。晋升/身份生命周期的既有fixture显式关闭空间补点，避免不同策略互相掩盖。未编译或执行这些用例；旧报告的PASS不能验证v16。
