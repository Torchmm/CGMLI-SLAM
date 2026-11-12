# ROS 2 相机和 IMU 输入

ROS 2 可执行程序 `cgmli_slam_node` 和离线程序共用 `ThreadedSlam`、估计器 YAML、相机模型和滑窗后端。Windows 可继续用原 Visual Studio 工程执行 Hilti 离线回放。ROS 2 包名是 `cgmli_slam`。

## 构建和运行

在已初始化的 ROS 2 Linux 环境中，将仓库放到 colcon 工作区的 `src` 下。需要 Eigen、Boost、OpenCV 4、Ceres 2.2 或更新版本、glog、yaml-cpp、PoseLib 2.0.5 和 DBoW2。先安装 PoseLib 2.0.5 和 DBoW2，并把自定义安装前缀加入 `CMAKE_PREFIX_PATH`。以下以 ROS 2 Jazzy 为例，在工作区根目录执行。

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select cgmli_slam --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

pkg_share="$(ros2 pkg prefix --share cgmli_slam)"
ros2 run cgmli_slam cgmli_slam_node --ros-args \
  --params-file "$pkg_share/config/hilti_robot_ros2.yaml" \
  -p estimator_config:="$pkg_share/config/hilti_robot_estimator.yaml"
```

可通过相机驱动发布原始图像和 IMU，也可以先启动节点再用 `ros2 bag play` 回放相同话题。打开 `ros2 topic echo /odom` 可检查估计器输出。若打开估计器回环功能，额外传入 `-p vocabulary_path:=/absolute/path/to/vocabulary`。

CMake 默认构建 CPU 版。CUDA 构建时加上 `-DCGMLI_ENABLE_CUDA=ON`，并用 `CMAKE_CUDA_ARCHITECTURES` 选择目标 GPU 架构。前端 CUDA 开关和后端 `backend_visual_batch` 沿用估计器 YAML。仅构建 CMake 离线程序时使用 `-DCGMLI_BUILD_ROS2=OFF`。

## 输入和输出

- `camera_topics` 与估计器 YAML 的相机索引一一对应。已禁用的相机可使用空字符串。示例为 Hilti 四组双目的左目，顺序为前、左、后、右。
- 相机订阅 `sensor_msgs/msg/Image`，通过 cv_bridge 转为 `mono8`，图像大小必须和标定一致。IMU 订阅 `sensor_msgs/msg/Imu`，使用加速度（m/s²）和角速度（rad/s）。订阅使用 ROS 2 的 SensorDataQoS。
- 相机和 IMU 的 `header.stamp` 必须来自同一时间基准，每个话题按时间递增。将相机时间戳设置为曝光中点。各路时间跨度在估计器 YAML 的 `timestamp_tolerance` 内才组成一帧，整数纳秒平均值作为 VIO 节点时间，各相机原始时间同时传入。固定 `image_delay` 仍由现有估计器应用。
- 沿用现有静止初始化和 IMU 时间覆盖要求。启动时让 IMU 先到达，并保持静止一小段时间。话题中断、严重不同步或队列溢出会产生丢弃日志。若 bag 循环造成时间戳回跳，重启节点开始新的估计段。
- `odom_topic` 默认是 `odom`，类型为 `nav_msgs/msg/Odometry`。输出 IMU 传感器坐标系在 `world_frame` 中的位姿，以及以 `imu_frame` 表示的速度。协方差未提供。可在 RViz 中通过 Odometry 显示查看结果。该节点只接相机和 IMU。
