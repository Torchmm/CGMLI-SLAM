#include "ImageSynchronizer.hpp"
#include <common/ViParametersReader.hpp>
#include <pipeline/ThreadedSlam.hpp>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <builtin_interfaces/msg/time.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <glog/logging.h>

#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

namespace cgmli::ros2 {
namespace {
using Image = sensor_msgs::msg::Image;
using Imu = sensor_msgs::msg::Imu;
using Synchronizer = ImageSynchronizer<Image::ConstSharedPtr>;
constexpr uint64_t nsPerSecond = 1000000000ULL;
constexpr size_t pendingBundleLimit = 2;
constexpr size_t pendingImuLimit = 5000;

std::optional<uint64_t> nanoseconds(const builtin_interfaces::msg::Time& stamp) {
    if (stamp.sec < 0 || stamp.nanosec >= nsPerSecond) return std::nullopt;
    return uint64_t(stamp.sec) * nsPerSecond + stamp.nanosec;
}

Time projectTime(uint64_t stamp) {
    Time result;
    result.fromNSec(stamp);
    return result;
}

class SlamNode final : public rclcpp::Node {
public:
    SlamNode() : rclcpp::Node("cgmli_slam") {
        const auto config = declare_parameter<std::string>("estimator_config", "");
        const auto topics = declare_parameter<std::vector<std::string>>("camera_topics", std::vector<std::string>{});
        const auto imuTopic = declare_parameter<std::string>("imu_topic", "/imu/data");
        const auto odomTopic = declare_parameter<std::string>("odom_topic", "odom");
        const auto vocabulary = declare_parameter<std::string>("vocabulary_path", "");
        worldFrame_ = declare_parameter<std::string>("world_frame", "odom");
        imuFrame_ = declare_parameter<std::string>("imu_frame", "imu_link");
        const auto cameraDepth = declare_parameter<int64_t>("camera_queue_size", 8);
        if (config.empty() || imuTopic.empty() || odomTopic.empty() || worldFrame_.empty() ||
            imuFrame_.empty() || cameraDepth < 1 || cameraDepth > 100)
            throw std::invalid_argument("Set estimator_config and valid topics, frames and camera_queue_size (1..100)");

        ViParametersReader reader(config);
        if (!reader.getParameters(parameters_) || !parameters_.imu.use ||
            topics.size() != parameters_.nCameraSystem.numCameras())
            throw std::invalid_argument("Estimator YAML must enable IMU and camera_topics must follow its camera order");
        std::set<std::string> uniqueTopics;
        for (size_t i = 0; i < topics.size(); ++i) {
            if (!parameters_.nCameraSystem.cameraType(i).isUsed) continue;
            if (topics[i].empty() || !uniqueTopics.insert(topics[i]).second)
                throw std::invalid_argument("Enabled cameras require distinct nonempty topics");
            cameraIndices_.push_back(i);
        }
        const double tolerance = parameters_.camera.timestamp_tolerance;
        if (cameraIndices_.empty() || !std::isfinite(tolerance) || tolerance < 0 || tolerance > 0.05)
            throw std::invalid_argument("Enable at least one camera and set timestamp_tolerance in [0, 0.05] seconds");
        synchronizer_.emplace(cameraIndices_.size(), uint64_t(std::llround(tolerance * nsPerSecond)), size_t(cameraDepth));

        parameters_.output.display_matches = false;
        parameters_.output.display_overhead = false;
        estimator_ = std::make_unique<ThreadedSlam>(parameters_, vocabulary);
        odometry_ = create_publisher<nav_msgs::msg::Odometry>(odomTopic, rclcpp::QoS(10));
        estimator_->setOptimisedGraphCallback([this](const State& state, const TrackingState&, auto, auto) {
            publishOdometry(state);
        });
        auto imageQos = rclcpp::SensorDataQoS();
        imageQos.keep_last(size_t(cameraDepth));
        for (size_t i = 0; i < cameraIndices_.size(); ++i) {
            images_.push_back(create_subscription<Image>(topics[cameraIndices_[i]], imageQos,
                [this, i](Image::ConstSharedPtr image) { receiveImage(i, std::move(image)); }));
            RCLCPP_INFO(get_logger(), "Camera %zu <- %s", cameraIndices_[i], images_.back()->get_topic_name());
        }
        auto imuQos = rclcpp::SensorDataQoS();
        imuQos.keep_last(1000);
        imu_ = create_subscription<Imu>(imuTopic, imuQos, [this](Imu::ConstSharedPtr imu) { receiveImu(*imu); });
        RCLCPP_INFO(get_logger(), "IMU <- %s, odometry -> %s, image synchronization tolerance %.3f ms",
            imu_->get_topic_name(), odometry_->get_topic_name(), tolerance * 1000);
        worker_ = std::thread(&SlamNode::process, this);
    }

    ~SlamNode() override { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    bool failed() const { return failed_; }

private:
    struct ImuSample {
        uint64_t stamp;
        Eigen::Vector3d acceleration, gyro;
    };

    void receiveImage(size_t camera, Image::ConstSharedPtr image) {
        const auto stamp = nanoseconds(image->header.stamp);
        const auto geometry = parameters_.nCameraSystem.cameraGeometry(cameraIndices_[camera]);
        if (!stamp || image->width != geometry->imageWidth() || image->height != geometry->imageHeight()) {
            RCLCPP_WARN_THROTTLE(get_logger(), throttleClock_, 5000,
                "Invalid camera %zu header stamp or image size does not match calibration", cameraIndices_[camera]);
            return;
        }
        bool ready = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return;
            const auto before = synchronizer_->discarded();
            synchronizer_->push(camera, *stamp, std::move(image));
            while (auto bundle = synchronizer_->pop()) {
                if (bundles_.size() == pendingBundleLimit) {
                    bundles_.pop_front();
                    RCLCPP_WARN_THROTTLE(get_logger(), throttleClock_, 5000, "VIO input backlog; dropped oldest complete image bundle");
                }
                bundles_.push_back(std::move(*bundle));
                ready = dirty_ = true;
            }
            if (synchronizer_->discarded() != before)
                RCLCPP_WARN_THROTTLE(get_logger(), throttleClock_, 5000, "Unmatched or unordered images dropped; total=%zu",
                    synchronizer_->discarded());
        }
        if (ready) wake_.notify_one();
    }

    void receiveImu(const Imu& imu) {
        const auto stamp = nanoseconds(imu.header.stamp);
        const Eigen::Vector3d acceleration(imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z);
        const Eigen::Vector3d gyro(imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
        if (!stamp || !acceleration.allFinite() || !gyro.allFinite() ||
            imu.linear_acceleration_covariance[0] == -1 || imu.angular_velocity_covariance[0] == -1) {
            RCLCPP_WARN_THROTTLE(get_logger(), throttleClock_, 5000, "Discarded invalid IMU sample");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return;
            if (latestImu_ && *stamp <= *latestImu_) {
                RCLCPP_WARN_THROTTLE(get_logger(), throttleClock_, 5000, "Discarded non-increasing IMU timestamp");
                return;
            }
            latestImu_ = *stamp;
            if (pendingImu_.size() == pendingImuLimit) {
                pendingImu_.pop_front();
                RCLCPP_WARN_THROTTLE(get_logger(), throttleClock_, 5000, "VIO IMU backlog; dropped oldest sample");
            }
            pendingImu_.push_back({*stamp, acceleration, gyro});
            dirty_ = true;
        }
        wake_.notify_one();
    }

    void submit(const Synchronizer::Bundle& bundle) {
        // Keep ROS image storage alive until ThreadedSlam has copied the Mats.
        std::vector<cv_bridge::CvImageConstPtr> decoded;
        std::map<size_t, cv::Mat> images;
        std::map<size_t, Time> stamps;
        try {
            for (size_t i = 0; i < bundle.observations.size(); ++i) {
                const auto& entry = bundle.observations[i];
                decoded.push_back(cv_bridge::toCvShare(entry.image, sensor_msgs::image_encodings::MONO8));
                images.emplace(cameraIndices_[i], decoded.back()->image);
                stamps.emplace(cameraIndices_[i], projectTime(entry.stamp));
            }
        } catch (const std::exception& error) {
            RCLCPP_WARN_THROTTLE(get_logger(), throttleClock_, 5000, "Image conversion failed: %s", error.what());
            return;
        }
        if (!estimator_->addImagesWithTimestamps(projectTime(bundle.stamp), images, stamps))
            RCLCPP_WARN_THROTTLE(get_logger(), throttleClock_, 5000, "Estimator rejected image bundle: %s", estimator_->lastError().c_str());
    }

    void process() {
        try {
            bool advanced = false;
            for (;;) {
                std::deque<ImuSample> samples;
                std::optional<Synchronizer::Bundle> bundle;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    if (!advanced) wake_.wait(lock, [this] { return dirty_ || stopping_; });
                    if (stopping_) break;
                    dirty_ = false;
                    samples.swap(pendingImu_);
                    if (!bundles_.empty() && !estimator_->hasPendingImages()) {
                        bundle = std::move(bundles_.front());
                        bundles_.pop_front();
                    }
                }
                for (const auto& sample : samples)
                    estimator_->addImuMeasurement(projectTime(sample.stamp), sample.acceleration, sample.gyro);
                if (bundle) submit(*bundle);
                advanced = estimator_->processFrame() || (bundle && !estimator_->hasPendingImages());
                if (estimator_->hasFatalError()) throw std::runtime_error(estimator_->lastError());
            }
        } catch (const std::exception& error) {
            failed_ = true;
            RCLCPP_ERROR(get_logger(), "VIO stopped: %s", error.what());
        }
        estimator_->stopThreading();
        if (failed_ && rclcpp::ok()) rclcpp::shutdown();
    }

    void publishOdometry(const State& state) {
        if (!rclcpp::ok()) return;
        const uint64_t stamp = state.timestamp.toNSec();
        if (stamp / nsPerSecond > uint64_t(std::numeric_limits<int32_t>::max())) return;
        nav_msgs::msg::Odometry msg;
        msg.header.stamp.sec = int32_t(stamp / nsPerSecond);
        msg.header.stamp.nanosec = uint32_t(stamp % nsPerSecond);
        msg.header.frame_id = worldFrame_;
        msg.child_frame_id = imuFrame_;
        const auto& p = state.T_WS.r();
        const auto q = state.T_WS.q();
        msg.pose.pose.position.x = p.x();
        msg.pose.pose.position.y = p.y();
        msg.pose.pose.position.z = p.z();
        msg.pose.pose.orientation.x = q.x();
        msg.pose.pose.orientation.y = q.y();
        msg.pose.pose.orientation.z = q.z();
        msg.pose.pose.orientation.w = q.w();
        const Eigen::Vector3d velocity = q.conjugate() * state.v_W;
        msg.twist.twist.linear.x = velocity.x();
        msg.twist.twist.linear.y = velocity.y();
        msg.twist.twist.linear.z = velocity.z();
        msg.twist.twist.angular.x = state.omega_S.x();
        msg.twist.twist.angular.y = state.omega_S.y();
        msg.twist.twist.angular.z = state.omega_S.z();
        odometry_->publish(msg);
    }

    ViParameters parameters_;
    std::vector<size_t> cameraIndices_;
    std::optional<Synchronizer> synchronizer_;
    std::string worldFrame_, imuFrame_;
    rclcpp::Clock throttleClock_{RCL_STEADY_TIME};
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_;
    rclcpp::Subscription<Imu>::SharedPtr imu_;
    std::vector<rclcpp::Subscription<Image>::SharedPtr> images_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Synchronizer::Bundle> bundles_;
    std::deque<ImuSample> pendingImu_;
    std::optional<uint64_t> latestImu_;
    bool dirty_ = false, stopping_ = false, failed_ = false;
    std::unique_ptr<ThreadedSlam> estimator_;
    std::thread worker_;
};
}
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    int status = 1;
    try {
        rclcpp::init(argc, argv);
        auto node = std::make_shared<cgmli::ros2::SlamNode>();
        rclcpp::spin(node);
        node->stop();
        status = node->failed() ? 1 : 0;
    } catch (const std::exception& error) {
        RCLCPP_ERROR(rclcpp::get_logger("cgmli_slam"), "%s", error.what());
    }
    if (rclcpp::ok()) rclcpp::shutdown();
    google::ShutdownGoogleLogging();
    return status;
}
