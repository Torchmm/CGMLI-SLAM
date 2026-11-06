
#ifndef INCLUDE_CGMLI_CERES_ODE_ODE_HPP_
#define INCLUDE_CGMLI_CERES_ODE_ODE_HPP_

#include <math.h>

/// \brief cgmli Main namespace of this package.
namespace cgmli {
/// \brief ode Namespace for functionality related to ODE integration implemented in cgmli.
namespace ode {

/// \brief sin(x)/x: to make things a bit faster than using angle-axis conversion.
/// \param x Input x.
/// \return Result of sin(x)/x.
inline double sinc(double x) {
  if (fabs(x) > 1e-6) {
   return sin(x) / x;
   } else{
    static const double c_2 = 1.0 / 6.0;
    static const double c_4 = 1.0 / 120.0;
    static const double c_6 = 1.0 / 5040.0;
    const double x_2 = x * x;
    const double x_4 = x_2 * x_2;
    const double x_6 = x_2 * x_2 * x_2;
    return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
  }
}

}
}  // namespace cgmli

#endif /* INCLUDE_CGMLI_CERES_ODE_ODE_HPP_ */
