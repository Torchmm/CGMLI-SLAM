#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cgmli::ros2 {

// Assemble one observation from each enabled camera, in calibration order.
// Per-camera messages must be time ordered. When all fronts are present, a
// front older than the allowed span cannot match a later observation.
template<class Image>
class ImageSynchronizer {
public:
    struct Observation { uint64_t stamp; Image image; };
    struct Bundle { uint64_t stamp; std::vector<Observation> observations; };

    ImageSynchronizer(size_t cameras, uint64_t tolerance, size_t capacity)
        : queues_(cameras), latest_(cameras), tolerance_(tolerance), capacity_(capacity) {
        if (!cameras || !capacity) throw std::invalid_argument("Image synchronizer requires cameras and queue capacity");
    }

    void push(size_t camera, uint64_t stamp, Image image) {
        auto& latest = latest_.at(camera);
        if (latest && stamp <= *latest) {
            ++discarded_;
            return;
        }
        latest = stamp;
        auto& queue = queues_[camera];
        if (queue.size() == capacity_) {
            queue.pop_front();
            ++discarded_;
        }
        queue.push_back({stamp, std::move(image)});
    }

    std::optional<Bundle> pop() {
        while (std::none_of(queues_.begin(), queues_.end(), [](const auto& queue) { return queue.empty(); })) {
            const auto [oldest, newest] = std::minmax_element(queues_.begin(), queues_.end(),
                [](const auto& a, const auto& b) { return a.front().stamp < b.front().stamp; });
            const uint64_t first = oldest->front().stamp;
            if (newest->front().stamp - first > tolerance_) {
                oldest->pop_front();
                ++discarded_;
                continue;
            }
            Bundle result{first, {}};
            result.observations.reserve(queues_.size());
            uint64_t offsets = 0;
            for (auto& queue : queues_) {
                offsets += queue.front().stamp - first;
                result.observations.push_back(std::move(queue.front()));
                queue.pop_front();
            }
            result.stamp += (offsets + queues_.size() / 2) / queues_.size();
            return result;
        }
        return std::nullopt;
    }

    size_t discarded() const { return discarded_; }

private:
    std::vector<std::deque<Observation>> queues_;
    std::vector<std::optional<uint64_t>> latest_;
    uint64_t tolerance_;
    size_t capacity_;
    size_t discarded_ = 0;
};

}
