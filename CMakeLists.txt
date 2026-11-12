cmake_minimum_required(VERSION 3.18)
project(cgmli_slam VERSION 0.1.0 LANGUAGES CXX)

option(CGMLI_BUILD_ROS2 "Build the ROS 2 VIO node with ament" ON)
option(CGMLI_ENABLE_CUDA "Build the CUDA ORB and backend paths" OFF)

if(CGMLI_BUILD_ROS2)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(builtin_interfaces REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(cv_bridge REQUIRED)
endif()

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core calib3d features2d imgproc imgcodecs highgui)
find_package(Ceres 2.2 REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(PoseLib CONFIG REQUIRED)
find_path(DBoW2_INCLUDE_DIR NAMES DBoW2/DBoW2.h REQUIRED)
find_library(DBoW2_LIBRARY NAMES DBoW2 dbow2 REQUIRED)

add_library(cgmli_core STATIC
    src/backend/BackendFactors.cpp
    src/backend/BackendOptimizer.cpp
    src/backend/BackendTypes.cpp
    src/backend/FixedLagBackend.cpp
    src/backend/FrameStore.cpp
    src/backend/GlobalSlamGraph.cpp
    src/backend/LandmarkReanchoring.cpp
    src/backend/LandmarkStore.cpp
    src/backend/LeftSE3Manifold.cpp
    src/backend/MarginalizationPrior.cpp
    src/backend/Marginalizer.cpp
    src/backend/ProjectBackendAdapter.cpp
    src/backend/SlidingWindowData.cpp
    src/backend/SlidingWindowManager.cpp
    src/backend/ViSlamBackend.cpp
    src/backend/VisualBatchAudit.cpp
    src/backend/WindowVisualFactor.cpp
    src/cameras/CameraBase.cpp
    src/cameras/NCameraSystem.cpp
    src/common/ViInterface.cpp
    src/common/ViParametersReader.cpp
    src/factor/BodyImuFactor.cpp
    src/factor/ErrorInterface.cpp
    src/factor/ImuError.cpp
    src/factor/PoseLocalParameterization.cpp
    src/frontend/Frontend.cpp
    src/frontend/FrontendAssociation.cpp
    src/frontend/FrontendPoseAdapter.cpp
    src/frontend/OrbFrontendFeatures.cpp
    src/frontend/stereo_triangulation.cpp
    src/pipeline/StateOutput.cpp
    src/pipeline/ThreadedSlam.cpp
    src/utils/Duration.cpp
    src/utils/Time.cpp
    src/utils/Timer.cpp
    src/visual/features/OrbFeatureExtractor.cpp
    src/visual/features/OrbFrameAdapter.cpp
    src/visual/features/OrbMatcher.cpp
    src/visual/pose_solver/AbsolutePoseSolver.cpp
    src/visual/pose_solver/PoseSolverTypes.cpp
    src/visual/pose_solver/RelativePoseSolver.cpp
    src/visual/pose_solver/RotationPoseSolver.cpp
    src/visualization/ViVisualizer.cpp)
target_compile_features(cgmli_core PUBLIC cxx_std_17)
target_compile_definitions(cgmli_core PUBLIC GLOG_NO_ABBREVIATED_SEVERITIES)
target_include_directories(cgmli_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" ${OpenCV_INCLUDE_DIRS}
    PRIVATE ${DBoW2_INCLUDE_DIR})
target_link_libraries(cgmli_core PUBLIC Eigen3::Eigen Ceres::ceres glog::glog PoseLib::PoseLib
    Boost::boost Threads::Threads ${OpenCV_LIBS} PRIVATE "${DBoW2_LIBRARY}")

if(MSVC)
    target_compile_definitions(cgmli_core PUBLIC NOMINMAX _USE_MATH_DEFINES)
endif()

if(CGMLI_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(cgmli_core PRIVATE src/backend/VisualCudaBatch.cu
        src/visual/features/OrbCuda.cu src/visual/features/OrbMatchCuda.cu)
    set_target_properties(cgmli_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    target_compile_definitions(cgmli_core PUBLIC CGMLI_ENABLE_CUDA_ORB)
    target_compile_options(cgmli_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(cgmli_core PRIVATE CUDA::cudart CUDA::cusolver CUDA::cublas)
endif()

add_executable(cgmli_offline src/app/offline_main.cpp src/app/CsvOffline.cpp
    src/offline/HiltiBinReader.cpp src/offline/HiltiDatasetReader.cpp)
target_link_libraries(cgmli_offline PRIVATE cgmli_core)
if(glog_VERSION VERSION_GREATER_EQUAL "0.6")
    target_compile_definitions(cgmli_offline PRIVATE CGMLI_GLOG_MESSAGE_TIME)
endif()
if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(cgmli_offline PRIVATE yaml-cpp::yaml-cpp)
else()
    target_link_libraries(cgmli_offline PRIVATE yaml-cpp)
endif()
if(MINGW)
    target_link_options(cgmli_offline PRIVATE -municode)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(cgmli_image_synchronizer_test src/ros2/tests/ImageSynchronizerTest.cpp)
    target_compile_features(cgmli_image_synchronizer_test PRIVATE cxx_std_17)
    add_test(NAME cgmli_image_synchronizer COMMAND cgmli_image_synchronizer_test)
endif()

if(CGMLI_BUILD_ROS2)
    add_executable(cgmli_slam_node src/ros2/SlamNode.cpp)
    ament_target_dependencies(cgmli_slam_node PUBLIC rclcpp builtin_interfaces sensor_msgs nav_msgs cv_bridge)
    target_link_libraries(cgmli_slam_node PRIVATE cgmli_core)
    install(TARGETS cgmli_slam_node cgmli_offline DESTINATION lib/${PROJECT_NAME})
    install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
    ament_package()
else()
    install(TARGETS cgmli_offline RUNTIME DESTINATION bin)
endif()
