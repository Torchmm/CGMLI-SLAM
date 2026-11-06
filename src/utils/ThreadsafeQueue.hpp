
#ifndef INCLUDE_CGMLI_THREADSAFE_THREADSAFEQUEUE_HPP_
#define INCLUDE_CGMLI_THREADSAFE_THREADSAFEQUEUE_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <queue>

#include <glog/logging.h>

/// \brief cgmli Main namespace of this package.
namespace cgmli {

/// \brief Namespace for helper classes for threadsafe operation.
namespace threadsafe {

/**
 * @brief Class that implements a threadsafe FIFO queue.
 * @tparam QueueType Datatype that is safed in the queue.
 */
template<typename QueueType, class Allocator = std::allocator<QueueType>>
class Queue {

 public:

  /// \brief Notify all waiting threads. Only used in destructor and when shutting down.
  virtual void NotifyAll() const final {
    condition_empty_.notify_all();
    condition_full_.notify_all();
  }

  /// \brief Constructor.
  Queue() {
    shutdown_ = false;
  }

  /// \brief Destructor. All threads using this queue must have been joined.
  virtual ~Queue() {
    // The owner must Shutdown() and join all users before destruction.
    Shutdown();
  }

  /// \brief Tell the queue shut down. This will notify all threads to wake up.
  virtual void Shutdown() final {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    NotifyAll();
  }

  /// \brief Resume after shutdown. Join previous waiting users before resuming.
  virtual void Resume() final {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = false;
    }
    NotifyAll();
  }

  /// \brief Push non-blocking to the queue.
  void Push(const QueueType& value) {
    PushNonBlocking(value);
  }

  /// \brief Push to the queue.
  void PushNonBlocking(const QueueType& value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(value);
    }
    condition_empty_.notify_one();
  }

  /// \brief Return the size of the queue.
  virtual size_t Size() const final {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  /// \brief Return true if the queue is empty.
  virtual bool Empty() const final {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  /// \brief Push to the queue if the size is less than max_queue_size, else block.
  /// \param[in] value New entry in queue.
  /// \param[in] max_queue_size Maximum queue size.
  /// \return False if shutdown is requested.
  bool PushBlockingIfFull(const QueueType& value, size_t max_queue_size) {
    if (max_queue_size == 0) {
      throw std::invalid_argument("max_queue_size must be greater than zero");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    condition_full_.wait(lock, [this, max_queue_size] {
      return shutdown_.load() || queue_.size() < max_queue_size;
    });
    if (shutdown_) {
      return false;
    }
    queue_.push(value);
    lock.unlock();
    condition_empty_.notify_one();
    return true;
  }

  /// \brief Push to the queue. If full, drop the oldest entry.
  /// \param[in] value New entry in queue.
  /// \param[in] max_queue_size Maximum queue size.
  /// \return True if oldest was dropped because queue was full.
  bool PushNonBlockingDroppingIfFull(const QueueType& value,
                                     size_t max_queue_size) {
    if (max_queue_size == 0) {
      throw std::invalid_argument("max_queue_size must be greater than zero");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    bool result = false;
    if (queue_.size() >= max_queue_size) {
      queue_.pop();
      result = true;
    }
    queue_.push(value);
    lock.unlock();
    condition_empty_.notify_one();
    return result;
  }

  /**
   * @brief Get the oldest entry still in the queue. Blocking if queue is empty.
   * @param[out] value Oldest entry in queue.
   * @return False if shutdown is requested.
   */
  bool Pop(QueueType* value) {
    return PopBlocking(value);
  }

  /**
   * @brief Get the oldest entry still in the queue. Blocking if queue is empty.
   * @param[out] value Oldest entry in queue.
   * @return False if shutdown is requested.
   */
  bool PopBlocking(QueueType* value) {
    CHECK_NOTNULL(value);
    std::unique_lock<std::mutex> lock(mutex_);
    condition_empty_.wait(lock, [this] {
      return shutdown_.load() || !queue_.empty();
    });
    if (shutdown_) {
      return false;
    }
    QueueType result = queue_.front();
    queue_.pop();
    lock.unlock();
    condition_full_.notify_one();
    *value = result;
    return true;
  }

  /**
   * @brief Get the oldest entry still in the queue. If queue is empty value is not altered.
   * @param[out] value Oldest entry in queue if queue was not empty.
   * @return True if queue was not empty.
   */
  /// Atomically reserve the previously peeked front item. Producers may drop an
  /// old front on overflow, so a separate peek followed by Pop is not sufficient.
  /// The predicate runs under the queue lock and must not reenter this queue.
  template<class Predicate>
  bool PopFrontIf(QueueType* value, Predicate matches) {
    CHECK_NOTNULL(value);
    std::unique_lock<std::mutex> lock(mutex_);
    if(queue_.empty() || !matches(queue_.front())) return false;
    *value=queue_.front(); queue_.pop();
    lock.unlock(); condition_full_.notify_one(); return true;
  }

  bool PopNonBlocking(QueueType* value) {
    CHECK_NOTNULL(value);
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    *value = queue_.front();
    queue_.pop();
    lock.unlock();
    condition_full_.notify_one();
    return true;
  }

  /**
   * @brief Get the oldest entry still in the queue. If the queue is empty wait for a given
   *        amount of time. If during this time an entry was pushed alter the value. If the
   *        queue is still empty, the value is not altered and it will return false
   * @param[out] value Oldest entry in queue if queue was not empty.
   * @param timeout_nanoseconds Maximum amount of time to wait for an entry if queue is empty.
   * @return True if value was updated. False on timeout or shutdown.
   *         Nonpositive timeouts perform an immediate check.
   */
  bool PopTimeout(QueueType* value, int64_t timeout_nanoseconds) {
    CHECK_NOTNULL(value);
    std::unique_lock<std::mutex> lock(mutex_);
    if (shutdown_) {
      return false;
    }
    if (queue_.empty()) {
      // A monotonic clock avoids changes caused by system clock adjustment.
      // wait_for uses a steady-clock deadline and a predicate for spurious wakes.
      if (timeout_nanoseconds <= 0 ||
          !condition_empty_.wait_for(
              lock, std::chrono::nanoseconds(timeout_nanoseconds), [this] {
                return shutdown_.load() || !queue_.empty();
              })) {
        return false;
      }
    }
    if (shutdown_) {
      return false;
    }
    QueueType result = queue_.front();
    queue_.pop();
    lock.unlock();
    condition_full_.notify_one();
    *value = result;
    return true;
  }

  /**
   * @brief Get a copy of the front / oldest element in the queue. If queue is empty
   *        value is not altered. The queue itself is not changed, i.e. the returned
   *        element is still in the queue.
   * @param[out] value Oldest entry in queue if queue was not empty.
   * @return True if queue was not empty and value was updated.
   */
  bool getCopyOfFront(QueueType* value) {
    CHECK_NOTNULL(value);
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    *value = queue_.front();
    return true;
  }

  /**
   * @brief Get a copy of the front / oldest element in the queue. Blocking if queue is empty.
   *        The queue itself is not changed, i.e. the returned element is still in the queue.
   * @param value Oldest entry in the queue.
   * @return False if shutdown is requested.
   */
  bool getCopyOfFrontBlocking(QueueType* value) {
    CHECK_NOTNULL(value);
    std::unique_lock<std::mutex> lock(mutex_);
    condition_empty_.wait(lock, [this] {
      return shutdown_.load() || !queue_.empty();
    });
    if (shutdown_) {
      return false;
    }
    *value = queue_.front();
    return true;
  }

  /**
   * @brief Get a copy of the back / newest element in the queue. If queue is empty
   *        value is not altered. The queue itself is not changed, i.e. the returned
   *        element is still in the queue.
   * @param[out] value Newest entry in queue if queue was not empty.
   * @return True if queue was not empty and value was updated.
   */
  bool getCopyOfBack(QueueType* value) {
    CHECK_NOTNULL(value);
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    *value = queue_.back();
    return true;
  }


  mutable std::mutex mutex_;                ///< The queue mutex.

  /// \brief Condition variable to wait and signal that queue is not empty.
  mutable std::condition_variable condition_empty_;

  /// \brief Condition variable to wait and signal when an element is popped.
  mutable std::condition_variable condition_full_;

  std::queue<QueueType,
      std::deque<QueueType, Allocator>> queue_;  ///< Actual queue.
  std::atomic_bool shutdown_;                    ///< Flag if shutdown is requested.

};

}  // namespace threadsafe

}  // namespace cgmli

#endif  // INCLUDE_CGMLI_THREADSAFE_THREADSAFEQUEUE_HPP_
