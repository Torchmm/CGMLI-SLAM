
#ifndef INCLUDE_CGMLI_TIMING_TIMER_HPP_
#define INCLUDE_CGMLI_TIMING_TIMER_HPP_

#include <chrono>
#include <mutex>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#pragma GCC diagnostic pop

#include <utils/assert_macros.hpp>


namespace cgmli {
namespace timing {
  
  CGMLI_DEFINE_EXCEPTION(TimerException, std::runtime_error)
  
  /// \brief Helper struct to map timer values.
  struct TimerMapValue {
    /// \brief Initialize the window size for the rolling mean.
    TimerMapValue() : m_acc(boost::accumulators::tag::rolling_window::window_size = 50){}
    boost::accumulators::accumulator_set<
      double, 
      boost::accumulators::features<
	boost::accumulators::tag::lazy_variance,
	boost::accumulators::tag::sum,
	boost::accumulators::tag::min,
	boost::accumulators::tag::max,
	boost::accumulators::tag::rolling_mean,
	boost::accumulators::tag::mean
	>
      > m_acc; ///< accumulator.
  };
  
  /// \brief A class that has the timer interface but does nothing.
  /// Swapping this in in place of the Timer class (say with a
  /// typedef) should allow one to disable timing. Because all
  /// of the functions are inline, they should just disappear.
  class DummyTimer {
  public:
    DummyTimer(size_t /* handle */, bool /* constructStopped */ ){}
    DummyTimer(size_t /* handle */){}
    DummyTimer(std::string const & /* tag */){}
    DummyTimer(std::string const & /* tag */, bool /* constructStopped */){}
    ~DummyTimer(){}
    
    void start(){}
    void stop(){}
    void discardTiming(){}
    bool isTiming(){ return false; }
  };
  
  /// \brief Timer class to get timings.
  class Timer {
  public:
    Timer(size_t handle, bool constructStopped = false);
    Timer(std::string const & tag, bool constructStopped = false);
    ~Timer();
    
    void start();
    void stop();
    bool isTiming();
    void discardTiming();
  private:
    std::chrono::steady_clock::time_point m_time;
    bool m_timing;
    size_t m_handle;
  };
  
  /// \brief Timing class.
  class Timing{
  public:
    friend class Timer;
    // Static funcitons to query the timers:
    static  size_t getHandle(std::string const & tag);
    static  std::string getTag(size_t handle);
    static  double getTotalSeconds(size_t handle);
    static  double getTotalSeconds(std::string const & tag);
    static  double getMeanSeconds(size_t handle);
    static  double getMeanSeconds(std::string const & tag);
    static  size_t getNumSamples(size_t handle);
    static  size_t getNumSamples(std::string const & tag);
    static  double getVarianceSeconds(size_t handle);
    static  double getVarianceSeconds(std::string const & tag);
    static  double getMinSeconds(size_t handle);
    static  double getMinSeconds(std::string const & tag);
    static  double getMaxSeconds(size_t handle);
    static  double getMaxSeconds(std::string const & tag);
    static  double getHz(size_t handle);
    static  double getHz(std::string const & tag);
    static  void print(std::ostream & out);
    static  void reset(size_t handle);
    static  void reset(std::string const & tag);
    static  void resetAll();
    static  std::string print();
    static  std::string secondsToTimeString(double seconds);
    
  private:
    void addTime(size_t handle, double seconds);
    
    static Timing & instance();
    
    // Singleton design pattern
    Timing();
    ~Timing();
    
    typedef std::unordered_map<std::string,size_t> map_t;
    typedef std::vector<TimerMapValue> list_t;
    
    std::mutex addNewHandleMutex_;

    // Static members
    list_t m_timers;
    map_t m_tagMap;
    size_t m_maxTagLength;
    
  }; // end class timer
  
#ifdef NDEBUG
  typedef DummyTimer DebugTimer;
#else
  typedef Timer DebugTimer;
#endif
  
} // namespace timing
} // end namespace cgmli

namespace cgmli {
#ifdef DEACTIVATE_TIMERS
  typedef timing::DummyTimer TimerSwitchable;
#else
  typedef timing::Timer TimerSwitchable;
#endif
} // end namespace cgmli


#endif // INCLUDE_CGMLI_TIMING_TIMER_HPP_
