# Hilti 2023 robot 估计器配置

布尔开关的说明注释放在独立行。OpenCV FileStorage 对未加引号的字符串及行尾注释的处理有边界情况：例如 `orb_layer_fast: true # false: ROI reference` 可能被读成映射节点而非布尔字符串。配置检查脚本会核对所有布尔节点的类型和值，不能只检查键是否存在。

`hilti_robot_estimator.yaml` 已填入官方 robot 平台的实际标定，适用于 `site2_robot_2.bag` 使用的传感器组合。原始标定来自 [Hilti 官方数据集页面](https://hilti-challenge.com/dataset-2023) 指向的 [calibration.zip](https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2023/blob/main/calibration/calibration.zip)。这里保留了归档中的五个 robot 文本文件，未下载 PDF 报告；各条目的 ZIP CRC 已校验，见 `reference/hilti2023_robot/SOURCE.json`。

| 配置部分 | 必需信息 | 本次填写方式 |
|---|---|---|
| `cameras` | 模型、畸变、分辨率、焦距、主点、`T_SC`、使用标记 | 官方 cam0/2/4/6，按前、左、后、右左目重排 |
| `camera_parameters` | 时间偏移、组时间容差、在线外参优化及先验宽度 | 官方报告时间偏移 0；初次调试关闭在线外参优化 |
| `imu_parameters` | IMU 使用开关、`T_BS`、连续噪声密度、偏置随机游走、初始偏置先验、重力及缩放 | 四个噪声参数取官方值；其余工程设定有单独注释 |
| `frontend_parameters` | ORB 数量、金字塔、FAST/Hamming 门限、关键帧视差/运动门限及线程设置 | 填写可供调试的初值，保留默认 GravityCameraAware/SparseBox 路径 |
| `estimator_parameters` | 滑窗规模、优化预算、健康检查、重试、回环和最终 BA 开关 | 默认保留 10 帧，先运行局部 VIO，关闭回环和最终 BA |
| `output_parameters` | 可视化开关 | 结构体默认关闭，当前用户调试配置已开启；`display_matches: true` 统一开启特征、同帧匹配、时序匹配和重投影调试图 |
| `lidar_reference` | LiDAR 到 IMU 的 `T_SL` 及时间约定 | 官方外参，仅留参考，不启用点云融合 |

## 相机编号和坐标约定

| 本工程 ID | 官方标定编号 | 话题 |
|---|---|---|
| 0 | cam0 | `/oak_cam_front/left/image_raw` |
| 1 | cam2 | `/oak_cam_left/left/image_raw` |
| 2 | cam4 | `/oak_cam_rear/left/image_raw` |
| 3 | cam6 | `/oak_cam_right/left/image_raw` |

已读取你转换完成的六个 `.bin` 文件的首条记录头和 JSON 元数据，四路左目均为 **1280×800、mono8**，IMU 坐标系为 `xsens_imu_link`，点云坐标系为 `rslidar`。配置按实际灰度左目填写；没有把中央 RGB 相机参数套在双目左目上。这次只核对元数据，未运行 VIO。

四目均采用官方 **pinhole + equidistant** 标定，不改为无畸变或 EUCM；原图不需要预先去畸变。`T_AB` 的含义统一为 `p_A = R_AB p_B + t_AB`。

官方 `calibration_cams.yaml` 的 `T_here_imu0` 是 **IMU → 相机**。本工程 `T_SC` 是 **相机 → IMU**，因此必须取刚体逆：`R_SC = R_CS.transpose()`、`t_SC = -R_CS.transpose() * t_CS`。生成时使用完整精度，且与官方 `calibration.yaml` 中的 `T_imu0_camN` 四元数和平移交叉核对。

为了先调通 VIO，这份配置将机体参考系 **B 定义为 IMU 系 S**，所以 `T_BS` 为单位矩阵；它不是声称物理机器人 `base_link` 与 IMU 重合。以后若要在真实 `base_link` 系输出或接入机体传感器，需要填写对应外参，并保持管线的坐标转换一致。当前导出轨迹仍是 `T_WS`。

## 噪声和调试初值

以下四项直接取自官方 `calibration_imu.yaml`，是连续时间参数：

- `sigma_g_c = 0.00013548849713483453`：陀螺仪白噪声密度。
- `sigma_a_c = 0.0006771784583884886`：加速度计白噪声密度。
- `sigma_gw_c = 2.6867875817855036e-6`：陀螺仪偏置随机游走。
- `sigma_aw_c = 4.855592390728982e-6`：加速度计偏置随机游走。

不要把报告中的 discrete noise 填到这里，也不要在配置层再乘 `sqrt(200)`。`sigma_bg`、`sigma_ba` 是初始化偏置先验宽度，不是上述随机游走；零偏先验均值、单位加速度缩放、`g=9.81`、输入限幅以及优化门限都在文件中标为工程初值，尚未根据真实轨迹调优。

原始图像时间为曝光中点。官方报告的 camera-to-IMU shift 为 0，当前 `image_delay: 0.0`。离线读取器按 `timestamp_tolerance: 0.025` 秒配成完整四目包，以四路曝光中点的平均值建立状态节点，无法形成完整包的记录会跳过并统计。默认采用同步近似。`estimate_td_camera: true` 启用公共时偏估计，并对有可信同相机跨帧轨迹的观测补偿组内曝光偏移。`do_extrinsics: true` 启用相机组到 IMU 的在线外参估计，平移还需开启 `estimate_camera_rig_translation` 并通过可观性检查。

`image_delay` 是加到原始图像时间戳上的固定修正，单位为秒。在线标定的总修正为 `image_delay + td_camera`。从旧版减法约定迁移时，将已有的非零 `image_delay` 取反。

窗口容量只由 `estimator_parameters.window_size` 决定，省略时默认 10，本配置也为 10。四目同组图像对应一个帧状态；IMU 通过帧间预积分因子接入，不存在单独的 IMU 帧配额。关键帧标记用于决定边缘化最老帧还是次新帧，不额外占用另一类容量。常态保留 10 帧，新帧插入后临时为 11 帧，求解和边缘化成功后恢复为 10 帧。旧的两类帧数配置不再支持，读取到旧键会提示替换为 `window_size`，避免静默使用错误容量。

当前初始化仍要求静止段，配置不会自动增加动态初始化能力。关闭回环、最终 BA 和在线外参是初次数据调试的运行设置，后续可按跟踪效果和耗时调整。

## LiDAR

官方 `calibration_lidar.yaml` 的 `T_here_imu0` 同样需要取逆才能得到 `T_SL`。该值已存入 `lidar_reference`。此节不被 `ViParametersReader` 当成活动的 `lidar:` 节读取；当前 VIO 应用只读取 LiDAR 扫描元数据并计数，跳过未使用的点云载荷，尚未接入 LIO。读取器默认完整解码及 `--inspect` 仍保留；未来接入 LIO 时需启用完整点云。非均匀垂直扫描的角分辨率没有编造数值。

## 调用和静态核对

入口直接接收估计器配置、数据目录和可选输出目录，不再需要额外的离线 YAML。相机 0/1/2/3 对应前/左/后/右左目。在可执行文件所在目录：

```powershell
.\CGMLI-SLAM.exe "H:/BaiduSyncdisk/CGMLI-SLAM/config/hilti_robot_estimator.yaml" "H:/数据集/hilti2023/floor1_large_room/converted" "H:/BaiduSyncdisk/CGMLI-SLAM/results/site2_robot_2"
```

静态核对脚本需要 Python、NumPy 和 OpenCV Python：

```powershell
python config/tools/check_hilti_config.py
```

脚本检查解析、C++ 读取器必填键、相机映射、标定数值、外参方向和矩阵合法性；不会编译或运行 SLAM。独立的 Python OpenCV 解析通过不能替代本工程的编译和实际数据调试。

官方标定及派生标定遵循 **CC BY-NC-SA 3.0**，归属 Hilti SLAM Challenge 2023 的发布方；保留原始来源及许可。算法背景引用：Nair 等，*Hilti SLAM Challenge 2023: Benchmarking Single + Multi-session SLAM across Sensor Constellations in Construction*，IEEE RA-L，2024，DOI `10.1109/LRA.2024.3421791`。


## v14清理后的配置

已删除没有当前算法消费者的CNN、旧BRISK/Harris阈值、旧keyframe_overlap、sync_cameras、深度网络双目索引和校正FOV、每相机mapping/mapping_rectification、num_loop_closure_frames、drift_percentage_heuristic、display_topview和enable_submapping，并删除对应参数存储、读取、空接口和神经网络校正代码。测量得到的相机/IMU/LiDAR外参与噪声资料保留。在线标定、回环及最终BA参数仍有实际调用，关闭开关并不代表参数无用，因此保留。

`enforce_realtime: true` 启用40 ms Ceres各轮共享额度，false关闭时间限制。v14在额度耗尽后优先验收最后有效状态，已删除的外点不参与边缘化；未重优化外点数及停止原因会记录。保留本次用户设置的200点/相机、8次最大迭代和已开启的可视化。静态配置检查同时检查已删除字段没有重新进入YAML/参数结构，以及CNN接口没有残留消费者。


## v15种子接纳与晋升

`keyframe_only_new_landmarks`已删除。使用 `require_keyframe_anchor_for_promotion: true` 限制rho的参考，非关键帧仍保留种子。`max_tracking_seeds: 1000` 控制全rig新种子接纳，`seed_max_untracked_seconds: 0.5`清理失跟纯种子；两者独立于成熟点预算，设0分别关闭额外上限/超时。超时按数据时间而非运行耗时，持续跟踪的种子不按出生年龄过期。已消费或有活跃优化证据的身份不被此清理路径删除。
