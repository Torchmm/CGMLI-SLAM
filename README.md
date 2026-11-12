# CGMLI-SLAM

CGMLI-SLAM 是一个 C++17 多相机视觉惯性 SLAM 工程。前端使用重力对齐和相机感知的 ORB 描述子，支持跨相机和跨帧匹配，并通过 PoseLib 求解广义相机位姿。后端将视觉观测、IMU 预积分和边缘化先验放入固定大小的滑窗，按关键帧策略移除最老帧或次新帧。工程也提供在线相机组外参和相机到 IMU 的时偏估计。

- **Windows 离线回放**读取转换后的 Hilti 2023 四路左目图像和 IMU，输出轨迹 CSV 与视觉地图 CSV。读取器也可读取 LiDAR 数据，当前估计器以 VIO 为主，离线应用仅统计扫描。
- **ROS 2 节点**订阅相机与 IMU 话题，复用同一估计器并发布 `nav_msgs/msg/Odometry`。ROS 2 构建和数据回放尚待在 Linux 上验证。
- **可选 CUDA**覆盖 ORB 提取和匹配，以及后端批量因子求值、原生 Schur 求解、双帧轻量求解和边缘化。Windows x64 CPU 和 CUDA 路径均已做过 Hilti 回放。

## 依赖与版本

下表列出 CMake 的版本要求和已用于 Windows 构建的参考版本。建议同一次构建使用同一架构和工具链编译的依赖。

| 依赖 | 用途 | 版本要求或参考版本 |
| --- | --- | --- |
| C++ 编译器 | 核心工程 | C++17，Windows 使用 Visual Studio 2022（MSVC v143）和 Windows 10/11 SDK |
| CMake | 跨平台构建 | 3.18 或更新 |
| Eigen | 矩阵运算、几何 | 3.4.0 |
| OpenCV | 图像、标定、ORB、调试显示 | CMake 要求 4.x，Windows 验证 4.8.0 |
| Ceres Solver | 滑窗非线性优化和全局 BA | CMake 请求 2.2，Windows 验证 2.2.0 |
| [PoseLib v2.0.5](https://github.com/PoseLib/PoseLib/releases/tag/v2.0.5) | 广义绝对位姿、相对位姿和 LO-RANSAC | 2.0.5，使用 C++ 安装包 |
| DBoW2 | ORB 词袋与回环 | Windows 使用 1.1 构建，需提供 `DBoW2/DBoW2.h` 和 DBoW2 库 |
| Boost | 计时统计 | Windows 验证 1.87.0 |
| glog | 运行日志 | Windows 验证 0.3.5，CMake 适配 0.6 及之后版本的日志回调接口 |
| yaml-cpp | Hilti 二进制数据清单 | Windows 验证 0.7.0 |
| SuiteSparse、BLAS、LAPACK | Ceres 稀疏求解依赖，随 Ceres 的构建配置安装 | Windows 使用 SuiteSparse 7.5.1；BLAS/LAPACK 版本与该构建匹配 |
| gflags | Windows 参考构建中的日志依赖 | 2.2.0 |
| CUDA Toolkit | 可选 GPU 路径，包含 CUDA Runtime、cuBLAS 和 cuSOLVER | Visual Studio 工程使用 12.6，验证设备为 GTX 1050 Ti（SM 6.1） |
| ROS 2 | 实时话题输入和里程计输出 | 构建示例使用 Jazzy，依赖 `ament_cmake`、`rclcpp`、`builtin_interfaces`、`sensor_msgs`、`nav_msgs` 和 `cv_bridge` |

原 Visual Studio 工程还链接了 PCL 1.15.0、oneTBB 2023.0 和 `slam_visualization_tool.lib`，工程属性也保留了 FLANN、GLM 0.9.3.2 的头文件路径。这些是原工程的本机配置项，CMake 路径无需这些额外库。

## Windows 离线构建

### Visual Studio 2022

1. 打开 [CGMLI-SLAM.sln](CGMLI-SLAM/CGMLI-SLAM.sln)，选择 `Release | x64`。
2. 在 [主工程](CGMLI-SLAM/CGMLI-SLAM.vcxproj) 的属性页核对头文件目录、库目录和附加依赖项。工程中的 `vs2022` 目录及绝对路径需对应本机依赖安装位置。运行前把依赖的 DLL 所在目录加入 `PATH`。
3. 从 Visual Studio 生成，或在 Developer PowerShell 中执行以下命令。CUDA 版需要安装 CUDA 12.6 的 Visual Studio 构建集成。

```powershell
msbuild .\CGMLI-SLAM\CGMLI-SLAM.sln /p:Configuration=Release /p:Platform=x64
```

若遇到 MSVC 全程序优化的编译器内部错误，可用 `/p:WholeProgramOptimization=false` 重新生成。只关闭 CUDA 内核编译可传入 `/p:CgmliCudaOrb=false`。原工程仍配置了 `cudart_static.lib`，无需 CUDA Toolkit 的 CPU 构建使用下面的 CMake 入口。

### CMake CPU 构建

在仓库根目录执行。将示例中的依赖安装前缀换成本机路径，可以用分号分隔多个前缀。如果 OpenCV 发行包有多套工具链，`OpenCV_DIR` 应指向对应工具链的 `OpenCVConfig.cmake` 目录。

```powershell
$deps = "C:/deps;C:/deps/poselib;C:/deps/dbow2"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCGMLI_BUILD_ROS2=OFF -DCGMLI_ENABLE_CUDA=OFF "-DCMAKE_PREFIX_PATH=$deps"
cmake --build build --config Release --target cgmli_offline cgmli_image_synchronizer_test
ctest --test-dir build -C Release --output-on-failure
```

### 回放 Hilti 数据

```powershell
& .\build\Release\cgmli_offline.exe --inspect "D:/datasets/hilti2023/converted"
& .\build\Release\cgmli_offline.exe .\config\hilti_robot_estimator.yaml "D:/datasets/hilti2023/converted" .\results\hilti
```

Visual Studio 生成的 `CGMLI-SLAM.exe` 接收同样的参数。数据目录应包含转换脚本生成的 `summary.json` 和对应的 `.bin` 文件，`index.csv` 可加快索引。Hilti 示例按前、左、后、右的左目顺序组包，启动时需要一段静止 IMU 数据。输出目录包含 `trajectory.csv` 和 `map.csv`。无界面回放时，将估计器 YAML 中的 `display_matches` 和 `display_overhead` 均设为 `false`。完整数据格式见[离线回放文档](src/offline/README.md)。

## Linux / ROS 2 构建

以下命令在包含本仓库的 colcon 工作区根目录执行。先安装 PoseLib 2.0.5 和 DBoW2 的 C++ 库，将自定义安装前缀加入 `CMAKE_PREFIX_PATH`。其他 ROS 依赖可通过 rosdep 安装。

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select cgmli_slam --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

share="$(ros2 pkg prefix --share cgmli_slam)"
ros2 run cgmli_slam cgmli_slam_node --ros-args \
  --params-file "$share/config/hilti_robot_ros2.yaml" \
  -p estimator_config:="$share/config/hilti_robot_estimator.yaml"
```

在 [ROS 2 参数文件](config/hilti_robot_ros2.yaml) 中修改 `camera_topics` 和 `imu_topic`。图像与 IMU 的 `header.stamp` 应使用同一时钟，话题顺序与估计器 YAML 的相机标定一致。节点默认发布 `/odom`。更多配置见 [ROS 2 接口说明](src/ros2/README.md)。

CMake 默认构建 CPU 版。启用 CUDA 时加上 `-DCGMLI_ENABLE_CUDA=ON`，并将 `CMAKE_CUDA_ARCHITECTURES` 设为目标 GPU 的计算能力。使用 GPU Schur 后端时，将估计器 YAML 的 `backend_visual_batch` 设为 `schur`，CPU 对照组设为 `cpu`。Jetson ARM64 的 CUDA 构建和运行尚待验证。

## 模块文档

- [配置与 Hilti 标定](config/README.md)
- [前端跟踪与位姿求解](src/frontend/README.md)
- [ORB 特征与 CUDA 路径](src/visual/features/README.md)
- [滑窗后端与边缘化](src/backend/README.md)
- [模块调试与回归](src/debug/README.md)
