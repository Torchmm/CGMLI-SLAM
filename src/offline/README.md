# Hilti 同步离线应用

入口参考用户提供的 `okvis2x_app_synchronous.cpp` 的组织方式：

```text
读取估计器配置 → 创建数据读取器与 ThreadedSlam → 连接回调
→ startStreaming → 主线程 processFrame/display → 停止线程 → 导出
```

只需要一个估计器 YAML 和转换数据目录，不再需要单独的 offline YAML、`--hilti` 模式或额外的标定索引配置。现有 `config/hilti_robot_estimator.yaml` 已按四个左目排序。

## 运行

在 exe 所在目录：

```powershell
.\CGMLI-SLAM.exe "H:/BaiduSyncdisk/CGMLI-SLAM/config/hilti_robot_estimator.yaml" "H:/数据集/hilti2023/floor1_large_room/converted" "H:/BaiduSyncdisk/CGMLI-SLAM/results/site2_robot_2"
```

参数依次为 **估计器配置、数据目录、可选输出目录**。不传输出目录时，写入数据目录下的 `results`。VS 的“调试 → 命令参数”填写同样的三个路径，不填写 exe。

运行结束输出 `trajectory.csv` 和 `map.csv`。轨迹沿用 `T_WS`（IMU 传感器系到世界系）与整数纳秒时间；和 ground truth 比较前需处理参考坐标系及轨迹对齐。可视化调试默认关闭，由估计器 YAML 中的统一开关控制：

```yaml
output_parameters:
  display_matches: false  # Set true to enable all four feature/match/reprojection views.
  display_overhead: false # Optional independent overhead trajectory/map view.
```

主线程负责 GUI。回环及 final BA 沿用估计器配置，当前默认关闭。仅检查数据、不运行 SLAM：

```powershell
.\CGMLI-SLAM.exe --inspect "H:/数据集/hilti2023/floor1_large_room/converted"
```

CSV 回放由 `src/app/CsvOffline.cpp` 保留，原 `config.yaml images.csv imu.csv ...` 参数仍可使用，也可以加显式 `--csv` 前缀。

## 特征和匹配图像调试

Hilti 应用仍使用原来的启动参数。将 `output_parameters.display_matches` 设为 `true` 后，正常完成 IMU 初始化并开始处理图像时，会显示四个可缩放窗口。默认 `false` 时不收集调试匹配、不缓存调试缩略图，也不启动图像绘制线程；未启用其他显示时，回放结束后直接退出，不等待按键。

| 窗口 | 显示内容 |
|---|---|
| `01 ORB Features` | 四目特征点拼图；黄色圆圈为提取点，短线使用存储的 keypoint angle，每相机标注特征数和相对节点的时间偏移 |
| `02 Stereo Matches` | 同一组四目图像的六个相机对，各面板左右拼接两张原图并连线；没有候选时仍显示该相机对 |
| `03 Temporal Matches` | 最近一个有候选且缓存图像可用的参考帧到当前帧，显示候选数量最多的六组相机对，包含同相机和跨相机跟踪；标题注明两端帧号和相机号 |
| `04 Backend Reprojection` | 原有重投影画面的四目拼图，仅在后端接受状态后更新；青色投影点到实测点的线段表示重投影误差 |

匹配图使用实际 `OrbMatcher::matchGuided()` 输出，不单独运行另一套匹配器。绿色表示两端仍归属同一个前端保留轨迹，红色表示该候选关联未被保留；红色可能来自几何过滤、轨迹占用冲突或数量限制，不能一概视为描述子匹配错误。前端绿色连线也不是每条独立通过 RANSAC 或后端 NIS 门控的证明。窗口标题另外显示整体 `visual_consensus` 和位姿求解器 `pose_inliers`。

每对图像最多各抽样绘制 60 条保留、60 条丢弃连线，避免过密；面板分别显示全部候选数、保留数、丢弃数和实际绘制数。跨帧图只展示上述最近参考帧和前六组相机对，不代表所有历史参考帧。前端图像在关联后即可显示，即使后端随后拒绝该帧仍能用于排查；前端与后端视图允许处于不同帧，请按标题帧号核对。

- `空格`：暂停/继续输入处理。已经启动的后端优化可以完成一次。
- `S`：保存当前所有窗口为 PNG，路径为 `输出目录/debug/snapshot_000001/`，每次新建编号目录，支持中文路径。
- `Esc` 或 `Q`：停止回放，排空线程并导出。
- 正常回放结束后保留最后画面，可以继续按 `S` 保存；按 `Esc`、`Enter` 或 `Q` 关闭。即使没有成功视觉跟踪，也允许先查看最后的调试图，再返回对应错误状态。

绘图在线程中使用独立缩略图和特征快照，历史图像缓存按活动滑窗状态加待处理帧保留；不要求后端长期保存全分辨率原图。截图和显示会增加开销，测试实时性能时将 `display_matches` 设为 `false`。以上交互针对 Hilti 离线入口；CSV 回放的原有处理循环未增加 GUI 交互。

## 运行日志

应用使用 glog，默认同时输出到控制台和工程 `logs/run_日期_时间_毫秒_序号/slam.log`。每次运行保留独立目录，`logs/latest.txt` 的第一行指向最新启动的日志绝对路径；包含启动失败和异常退出前的错误消息。正常启动后无需增加参数即可查看以下信息：

- `[Rig]`：相机模型、分辨率，以及相机两两视野重叠比例和计算耗时。比例是各自整幅图像中重叠像素的占比，采用无穷远方向近似。
- `[Dataset]` / `[App]`：正在索引的文件、相机与传感器话题、记录数量、有效时间范围、IMU 覆盖不足的丢帧、回放进度与结束统计。
- `[Init]`：等待静止 IMU 初始化，以及第一条后端成功提交状态的时间、初始位姿、速度和偏置。第一条后端状态成功不代表视觉跟踪已经成功，应结合 `[Tracking]` 查看。
- `[Tracking]`：后端提交后的跟踪质量变化；丢失立即警告，恢复立即提示。
- `[Backend]` / `[VIO]`：第一次及每 30 次后端成功提交的摘要，包括帧号、窗口状态数、地图点数、内点数、优化代价、迭代次数、耗时和当前位置。
- `[Output]`：轨迹、地图导出路径和数量。输入队列溢出、求解拒绝/重试、终止性 IMU 故障分别输出警告或错误。

需要逐帧诊断时，通过环境变量开启 `VLOG(1)`，保持原有位置参数不变：

```powershell
$env:GLOG_v = '1'
.\CGMLI-SLAM.exe "H:/BaiduSyncdisk/CGMLI-SLAM/config/hilti_robot_estimator.yaml" "H:/数据集/hilti2023/floor1_large_room/converted" "H:/BaiduSyncdisk/CGMLI-SLAM/results/site2_robot_2"
```

VS 的“项目属性 → 调试 → 环境”中填写 `GLOG_v=1` 也可以。级别 `0` 只显示默认信息，`1` 增加逐帧提取耗时、每相机特征数、候选匹配数量、关联耗时/内点/新增轨迹、关键帧选择、后端求解和 `MARGIN_OLD` / `MARGIN_SECOND_NEW` 边缘化详情；`2` 进一步显示每对相机的匹配数量、位姿求解器的状态、对应点数量、RANSAC 迭代/模型数和每个数据 epoch 的传感器覆盖。`[Frontend] Staged` 表示待后端验证的关联结果，最终是否接受以 `[Backend]` 和 `[Tracking]` 为准。

逐帧日志会增加输出开销，平时使用级别 `0`；恢复默认可设置 `$env:GLOG_v='0'`。文件记录自动启用，无需重定向。后续分析直接读取本地 `logs/latest.txt` 及其指向的文件，或者在工程根目录执行 `& ./src/offline/tools/read_latest_log.ps1 -Full`（省略 `-Full` 读取末尾 200 行）。环境变量 `CGMLI_LOG_DIR` 可覆盖目录，读取脚本也支持 `-LogRoot`。VS 工作目录变化不会影响源码树内的默认位置；独立部署回退到程序目录。

文件通过 glog 的线程安全 sink 写入，使用原生文件路径兼容中文目录，并保留 glog 前缀格式。INFO 在下一条日志到达且距上次刷新至少一秒时刷新，WARNING/ERROR 立即刷新，正常退出统一刷新；强制终止可能损失最后一小段缓冲。`arguments.txt` 保存本次启动参数，历史日志不会覆盖。

相机重叠在全部标定加载完毕后统一计算一次。四台相机的 `overlapMats_` 是 4×4 的索引表，其中只分配 12 张有方向的跨相机掩码，对角线保持为空；`hasOverlap(i,i)` 直接返回 `true`，`overlap(i,i)` 在首次请求时才生成并缓存全 `1` 掩码，不影响跨相机计算是否有效。添加相机或修改外参后应重新调用 `computeOverlaps()`；`ThreadedSlam` 初始化时也会检查并补算尚未完成的结果。

## 文件职责

默认日志现在还包含 `[Keyframe]` 的逐帧原因和球面视差/累计运动、`[Window]` 的实际边缘化与成熟点退休数量、`[LandmarkRetireSummary]` 的退休原因。`[Residual]` 每 30 帧及失败时记录真实窗口的 visual/imu/prior 等白化 RMS、NIS 分位数与 robust cost；`GLOG_v=1` 可逐帧记录，并输出每个退休 Landmark 的 anchor 与剩余支撑。`[Timing][Solve]` 标明解析/数值视觉雅可比、Ceres Problem 构建/复用次数和是否启用逐因子性能采样。

静止片段预期看到 `selected=0 reason=low_motion` 或 `low_parallax`，随后实际 `policy=MARGIN_SECOND_NEW`；关键帧标记属于当前输入，而策略使用倒数第二帧标记，状态切换可能有一帧延迟。先核对 `retired_mature_with_live_views`，区分新种子清理和仍有观测的成熟点退休。性能对照宜保持 `GLOG_v=0`、关闭显示，并单独观察 `factor_profiled=0` 的帧；采样帧的时钟/计数开销不可直接当作纯因子耗时。

- `src/app/offline_main.cpp`：配置、估计器、回调连接、处理/显示循环和最终导出。
- `HiltiDatasetReader.hpp/.cpp`：后台数据生产线程，按图像 epoch 供给测量；不包含 `ThreadedSlam`，也不调用 `processFrame()`。
- `HiltiBinReader.hpp/.cpp`：用户 `hilti_bag_to_bin.py` 的 HILTBIN1 解码、文件索引及图像/IMU/点云转换。

核心调用结构如下：

```cpp
HiltiDatasetReader reader(datasetPath, parameters);
ThreadedSlam estimator(parameters, "");
estimator.setBlocking(true);
reader.setImuCallback(/* estimator.addImuMeasurement */);
reader.setImagesCallback(/* estimator.addImagesWithTimestamps(meanTime, images, originalTimes) */);
reader.setLidarCallback(/* complete scan consumer, currently read-only */);
reader.startStreaming();
while (reader.isStreaming() || estimator.hasPendingWork()) {
    reader.rethrowIfFailed();
    estimator.processFrame();
    estimator.display(images);
}
```

主线程在 EOF 后继续处理尚未提交的后端任务。提前结束或异常时，实际代码先 `reader.requestStop()`，然后 `estimator.stopThreading()` 释放阻塞输入队列，再 `reader.stopStreaming()` 回收读取线程；不能先 join 一个仍阻塞在 `addImages()` 的读取线程。最后收完发布线程，再进行 final BA 和导出。读取异常会传回主线程，不能当成正常 EOF。

## 输入和时间

| 相机 ID | 方向/左目 | 默认话题 | 对应官方标定 |
|---|---|---|---|
| 0 | 前 | `/oak_cam_front/left/image_raw` | cam0 |
| 1 | 左 | `/oak_cam_left/left/image_raw` | cam2 |
| 2 | 后 | `/oak_cam_rear/left/image_raw` | cam4 |
| 3 | 右 | `/oak_cam_right/left/image_raw` | cam6 |

另外读取 `/imu/data` 和 `/rslidar_points`。这四路原图实际是 1280×800 mono8；需要与 YAML 的四相机顺序及原图标定一致。不同模型仍由各自 `CameraBase` 实现处理，读取器不修改内外参、图像尺寸或畸变模型。

数据目录通过已完成的 `summary.json` 中 `topic_files` 打开六个 `.bin`，不猜文件名，也不加载其它传感器。兼容旧版混合话题的单个 `data.bin`；中文路径通过原生文件系统路径打开。读取过程不依赖 ROS，不修改 bag、转换脚本或数据。

内部按传感器头时间排序，bag 时间仅用于诊断。当前采用**完整四目组包、平均曝光中点作为节点时间、同步近似**：每个包从四路各取一张尚未使用的图像，最大、最小曝光中点之差不能超过 `camera_parameters.timestamp_tolerance`，本配置为 `0.025` 秒。组包时间使用整数纳秒计算四路平均值并舍入到最近纳秒，避免绝对时间转成 double 后损失精度。Hilti robot 同时开始曝光，但各组自动曝光时长不同，图像头表示曝光中点，参见 [官方 robot 文档](https://github.com/Hilti-Research/hilti-slam-challenge-2023/blob/main/documentation/hardware/Robot.md)。当前没有读取 `exposure_info` 来验证触发 ID，因此这是有界时间邻近配对。

只提交包含四个相机的完整包；时间跨度过大时跳过无法匹配的较旧记录，恢复后继续配对。缺失相机或文件尾部的剩余记录会计入 `unmatched_images`，不复用上一张图补齐，也不再提交只有两张图的部分相机包。为避免匹配窗口覆盖同一路的相邻帧，容差必须小于每路最小帧间隔的一半；不满足时明确报错。

平均时间是优化节点的时间，原始图像时间不会被覆盖：读取回调传入 `originalTimes`，应用调用 `ThreadedSlam::addImagesWithTimestamps()`，将原始时间和相对平均值的有符号偏移随队列、`MultiFrame::imageTiming(camera)` 和后端冻结帧保留。局部视觉因子、匹配、三角化以及历史 BA 仍把四目当作平均时刻观测，暂不使用这些偏移做运动补偿，也不增加图优化节点。

对当前 `floor1_large_room/converted/index.csv` 的时间索引进行了静态核对：四路各 3059 张，同序号四目最大时间跨度为 10.92 ms，默认 25 ms 容差可覆盖。首组平均时间为 `1674222195036093000 ns`，前/左偏移 `+5.001 ms`、后/右偏移 `-5.001 ms`。这项核对未解码整个图像数据集或执行 SLAM。

读取线程先通过回调排入完整四目包，再供给修正后的平均节点时间后 20 ms 的 IMU 支持，与本项目 `collectImu()` 一致。`image_delay` 在入队时加到平均值上，原始图像时间保持不变。估计线程可同时消费有界队列，保持特征提取和后端优化的并行。默认保留首节点前最多 600 ms 的 IMU 支持；没有前后支持的两端完整包跳过并计入 `uncovered_epochs`，与未匹配图像分开统计。物理 IMU 间断超过 `imu.max_sample_gap` 会报错，停止该估计段。现有初始化仍要求静止区间。

`HiltiReaderOptions` 提供 C++ 层的自定义话题、合帧容差和 IMU 预读长度。默认合帧容差为 25 ms，离线 main 从估计器 YAML 的 `camera_parameters.timestamp_tolerance` 覆盖此值，无需额外配置文件。零容差只接受四路曝光中点完全相等的完整包；不能恢复旧的部分相机组包模式。

## LiDAR 和解码边界

应用当前为 **四目 + IMU VIO，LiDAR 扫描元数据计数**。离线入口设置 `HiltiReaderOptions::decodeLidarPoints=false`，回调保留扫描时间、布局和字段表，跳过未使用的点云载荷读取及逐点解码；此时 `points` 和 `originalData` 为空，逐点时间不可用。读取器默认选项仍为完整解码，`readLidar(record)` 和 `--inspect` 保留逐点检查；后续接入 LIO 时应显式启用完整扫描。

## 索引与耗时分析

目录模式优先复用转换器的 `index.csv`，顺序读取 CSV，检查版本、文件名映射、记录数量、文件大小、偏移连续性和完整覆盖，并在消费记录时检查二进制头及元数据。缺失、过期或格式错误时回退到原二进制扫描；日志注明 `source=index.csv` 或 `source=binary_scan` 及索引耗时。该快速路径不生成新的索引缓存，也不修改数据集文件。

`num_matching_threads` 已连接到相机对任务。任务使用只读帧/射线快照，按固定顺序合并并分配轨迹；未初始化深度的轨迹仍保留球面几何搜索。成熟点采用完整位姿投影和像素网格查询；双向最近邻及 ratio 的第二近邻来自同一允许候选集合。

设置 `GLOG_v=1` 可逐帧查看 `[Timing]`：

| 日志 | 计时内容 |
|---|---|
| `[Startup]`、`[Replay]` | 启动总时间和独立的回放时间 |
| `[Input]` | 图像、IMU、LiDAR 读取，以及回调阻塞时间；阻塞通常包含估计器反压 |
| `[Extract]`、`[ExtractWall]` | 每相机冷启动、金字塔、检测、描述子及四路提取墙钟时间 |
| `[Frontend]`、`[Pose]` | 快照准备、匹配、轨迹分配、位姿、过滤、三角化、提交；另有实际候选和 Hamming 比较计数 |
| `[Pipeline]` | 等待前一个后端、状态插入和前端处理、单次 processFrame 调用时间 |
| `[Backend]` | NIS 门控、Ceres 问题构建、求解、边缘化、快照和发布 |

默认级别每 30 帧及冷启动/明显慢阶段输出摘要。并行相机时间不能相加当作提取延迟；输入回调阻塞不等于磁盘读取慢；后端异步执行，所以各线程耗时不能简单相加。计时不改变现有优化迭代预算、几何门限或可视化默认关闭设置。

- HILTBIN1 文件头为 12 字节，记录头为 `<IIqqQ` 32 字节，小端、无结构体填充。
- 原始图像按 encoding、step 和行填充转换；RGB/RGBA 转为工程需要的 BGR/BGRA，支持 mono8、常见八位彩色/Bayer 及压缩图像。
- IMU 为 37 个小端 double，gyro 从索引 13 开始、acc 从索引 25 开始，保持 SI 单位和原轴向，不将原始 orientation 当成位姿先验。
- PointCloud2 按字段偏移、类型、大小端、point_step/row_step 解码。robot 的 FLOAT64 `timestamp` 为绝对秒，头时间为末点，转换时不经过 float32。坐标仍在 LiDAR 系，未应用外参或去畸变。
- 保留截断、未完成转换、错误话题映射、非法步长、缺失传感器、重复时间戳、非有限测量及大小上限检查。

## 检查

```powershell
powershell -ExecutionPolicy Bypass -File src/offline/tools/check_offline_static.ps1
python config/tools/check_hilti_config.py
```

独立回归源 `tests/HiltiOfflineRegression.cpp` 覆盖二进制布局、跨文件索引、时间分组，以及读取回调、异常传递和阻塞回调取消。它以 `None` 注册，不引入第二个应用入口。本轮仅做静态检查，不编译或执行 C++ 回归和真实 SLAM。
