#pragma once

// Include before Windows.h to avoid its ERROR macro colliding with glog.
#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
#include <glog/logging.h>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>

namespace cgmli {

// glog is process-global. Initialize on the main thread before starting workers;
// join all workers before Shutdown. Do not initialize glog elsewhere when using
// this wrapper. Flags must not be modified concurrently with logging.
class GlogLogger final {
public:
    enum class Level { Info = 0, Warning = 1, Error = 2, Fatal = 3 };

    struct Options {
        std::string directory = "logs";
        Level minimum = Level::Info;
        bool console = false; // true: copy all enabled records to stderr
        int verbosity = 0;    // 1 enables ALG_LOG_DEBUG, including Release builds
        int max_file_size_mb = 64; // per severity file; not total disk quota
        int buffer_seconds = 1;
    };

    GlogLogger() = delete;

    static void Init(const char* program_name) {
        Init(program_name, Options{});
    }

    static void Init(const char* program_name, const Options& options) {
        std::lock_guard<std::mutex> lock(ControlMutex());
        if (Owned())
            throw std::logic_error("glog is already initialized; initialize once per process");
        if (!program_name || !*program_name || options.directory.empty() ||
            options.verbosity < 0 || options.max_file_size_mb <= 0 ||
            options.buffer_seconds < 0 || static_cast<int>(options.minimum) < 0 ||
            static_cast<int>(options.minimum) > 3)
            throw std::invalid_argument("Invalid logger options");

        // Use an absolute path so later working-directory changes do not redirect logs.
        const auto directory = std::filesystem::absolute(
            std::filesystem::path(options.directory));
        std::filesystem::create_directories(directory);
        if (!std::filesystem::is_directory(directory))
            throw std::runtime_error("Log path is not a directory");

        FLAGS_log_dir = directory.string();
        FLAGS_logtostderr = false;
        FLAGS_alsologtostderr = options.console;
        FLAGS_stderrthreshold = options.console ? 0 : 2;
        FLAGS_minloglevel = static_cast<int>(options.minimum);
        FLAGS_v = options.verbosity;
        FLAGS_max_log_size = options.max_file_size_mb;
        FLAGS_logbufsecs = options.buffer_seconds;
        FLAGS_logbuflevel = 0; // buffer INFO; WARNING and above flush immediately
        FLAGS_stop_logging_if_full_disk = true;
        google::InitGoogleLogging(program_name);
        Owned() = true;
    }

    static void Flush() {
        std::lock_guard<std::mutex> lock(ControlMutex());
        if (!Owned()) throw std::logic_error("Logger has not been initialized");
        google::FlushLogFiles(google::GLOG_INFO);
    }

    // Call after joining worker threads. Repeated Shutdown calls are harmless.
    static void Shutdown() {
        std::lock_guard<std::mutex> lock(ControlMutex());
        if (!Owned()) return;
        google::FlushLogFiles(google::GLOG_INFO);
        google::ShutdownGoogleLogging();
        Owned() = false;
    }

private:
    static std::mutex& ControlMutex() { static std::mutex mutex; return mutex; }
    static bool& Owned() { static bool owned = false; return owned; }
};

} // namespace algorithm
