#include "../ImageSynchronizer.hpp"
#include <iostream>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void fourCameraMean() {
    constexpr uint64_t start = 1674222195036093000ULL;
    cgmli::ros2::ImageSynchronizer<int> sync(4, 10000001, 3);
    sync.push(1, start, 1);
    sync.push(3, start, 3);
    sync.push(0, start + 10000001, 0);
    require(!sync.pop(), "wait for the fourth camera");
    sync.push(2, start + 10000001, 2);
    const auto bundle = sync.pop();
    require(bundle && bundle->stamp == start + 5000001, "rounded mean in integer nanoseconds");
    require(bundle->observations[0].image == 0 && bundle->observations[0].stamp == start + 10000001 &&
        bundle->observations[1].image == 1 && bundle->observations[1].stamp == start, "preserve camera order and source stamps");
    require(!sync.pop(), "consume each observation once");
}

void dropsAndRecovery() {
    cgmli::ros2::ImageSynchronizer<int> sync(2, 10, 2);
    sync.push(0, 1, 1);
    sync.push(0, 30, 30);
    sync.push(0, 29, 29);
    sync.push(1, 20, 20);
    auto bundle = sync.pop();
    require(bundle && bundle->stamp == 25 && sync.discarded() == 2, "drop unmatched and out-of-order data");
    sync.push(0, 31, 31);
    sync.push(0, 40, 40);
    sync.push(0, 50, 50);
    sync.push(1, 49, 49);
    bundle = sync.pop();
    require(bundle && bundle->stamp == 45 && sync.discarded() == 3, "recover after missing-camera queue overflow");
}
}

int main() {
    try {
        fourCameraMean();
        dropsAndRecovery();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
