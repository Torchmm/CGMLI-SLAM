
#ifndef INCLUDE_CGMLI_TIMING_NSECTIMEUTILITIES_HPP_
#define INCLUDE_CGMLI_TIMING_NSECTIMEUTILITIES_HPP_
#include <chrono>
#include <cstdint>

namespace cgmli {

/// \brief The namespace for timing-related things.
namespace timing {

/// \brief Nanoseconds since the epoch.
typedef std::int64_t NsecTime;

/// \brief Convert nanoseconds since the epoch to std::chrono
std::chrono::system_clock::time_point nsecToChrono( const NsecTime & time );

/// \brief Convert std::chrono to nanoseconds since the epoch.
NsecTime chronoToNsec( const std::chrono::system_clock::time_point & time );

/// \brief Get the epoch time as nanoseconds since the epoch.
NsecTime nsecNow();

/// \brief Convert the time (in integer nanoseconds) to decimal seconds.
double nsecToSec( const NsecTime & time );

/// \brief Convert the time (in seconds) to integer nanoseconds
NsecTime secToNsec( const double & time );

/// \brief return a magic number representing an invalid timestamp
constexpr NsecTime getInvalidTime();

/// \brief Is the time valid? This uses a magic number
///        std::numeric_limits<NsecTime>::min() to represent an invalid time
bool isValid(const NsecTime & time);

} // namespace timing
} // namespace cgmli

#endif /* INCLUDE_CGMLI_TIMING_NSECTIMEUTILITIES_HPP_ */
