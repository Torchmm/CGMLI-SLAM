
#ifndef CGMLI_ASSERT_MACROS_HPP
#define CGMLI_ASSERT_MACROS_HPP

#include <sstream>
#include "utils/source_file_pos.hpp"

//! Macro for defining an exception with a given parent
//  (std::runtime_error should be top parent)
// adapted from ros/drivers/laser/hokuyo_driver/hokuyo.h
//#define CGMLI_DEFINE_EXCEPTION(exceptionName, exceptionParent)				\
//  class exceptionName : public exceptionParent {						\
//  public:																\
//  typedef __typeof__(exceptionParent) exceptionParent_t;  \
//  using exceptionParent_t::exceptionParent_t; \
//};

#define CGMLI_DEFINE_EXCEPTION(exceptionName, exceptionParent) \
class exceptionName : public exceptionParent {               \
public:                                                      \
    using exceptionParent_t = exceptionParent;                \
    using exceptionParent_t::exceptionParent_t;               \
};

/// \brief cgmli Main namespace of this package.
namespace cgmli {

  namespace detail {

    template<typename CGMLI_EXCEPTION_T>
    inline void CGMLI_throw_exception(std::string const & exceptionType, cgmli::source_file_pos sfp, std::string const & message)
    {
      std::stringstream cgmli_assert_stringstream;
      // I have no idea what broke doesn't work with the << operator. sleutenegger: not just Windows, but in general...???
      cgmli_assert_stringstream << exceptionType <<  sfp.toString() << " " << message;
      throw(CGMLI_EXCEPTION_T(cgmli_assert_stringstream.str()));
    }

    template<typename CGMLI_EXCEPTION_T>
    inline void CGMLI_throw_exception(std::string const & exceptionType, std::string const & function, std::string const & file,
								   int line, std::string const & message)
    {
      CGMLI_throw_exception<CGMLI_EXCEPTION_T>(exceptionType, cgmli::source_file_pos(function,file,line),message);
    }


  } // namespace cgmli::detail

  template<typename CGMLI_EXCEPTION_T>
  inline void cgmli_assert_throw(bool assert_condition, std::string message, cgmli::source_file_pos sfp) {
    if(!assert_condition)
      {
		detail::CGMLI_throw_exception<CGMLI_EXCEPTION_T>("", sfp,message);
      }
  }



} // namespace cgmli

#define CGMLI_CHECK_MAP(obj, id)                                                                                          	\
  if(!(obj.count(id)))															\
  {																	\
      std::stringstream cgmli_assert_stringstream;											\
      cgmli_assert_stringstream << #obj << ".at(" << #id <<") failed! "; 						\
      cgmli::detail::CGMLI_throw_exception<std::runtime_error>("[CGMLI_AT_CHECKED] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
  }

#define CGMLI_THROW(exceptionType, message) {								\
    std::stringstream cgmli_assert_stringstream;							\
    cgmli_assert_stringstream << message;									\
    cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, cgmli_assert_stringstream.str()); \
  }


#define CGMLI_THROW_SFP(exceptionType, SourceFilePos, message){			\
    std::stringstream cgmli_assert_stringstream;							\
    cgmli_assert_stringstream << message;									\
    cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", SourceFilePos, cgmli_assert_stringstream.str()); \
  }

#define CGMLI_ASSERT_TRUE(exceptionType, condition, message)				\
  if(!(condition))														\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert(" << #condition << ") failed: " << message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, cgmli_assert_stringstream.str()); \
    }

#define CGMLI_ASSERT_FALSE(exceptionType, condition, message)				\
  if((condition))														\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert( not " << #condition << ") failed: " << message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, cgmli_assert_stringstream.str()); \
    }



#define CGMLI_ASSERT_GE_LT(exceptionType, value, lowerBound, upperBound, message) \
  if((value) < (lowerBound) || (value) >= (upperBound))							\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert(" << #lowerBound << " <= " << #value << " < " << #upperBound << ") failed [" << (lowerBound) << " <= " << (value) << " < " << (upperBound) << "]: " << message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }



#define CGMLI_ASSERT_LT(exceptionType, value, upperBound, message)			\
  if((value) >= (upperBound))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert(" << #value << " < " << #upperBound << ") failed [" << (value) << " < " << (upperBound) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }

#define CGMLI_ASSERT_GE(exceptionType, value, lowerBound, message)			\
  if((value) < (lowerBound))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert(" << #value << " >= " << #lowerBound << ") failed [" << (value) << " >= " << (lowerBound) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }



#define CGMLI_ASSERT_LE(exceptionType, value, upperBound, message)			\
  if((value) > (upperBound))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert(" << #value << " <= " << #upperBound << ") failed [" << (value) << " <= " << (upperBound) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }

#define CGMLI_ASSERT_GT(exceptionType, value, lowerBound, message)			\
  if((value) <= (lowerBound))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert(" << #value << " > " << #lowerBound << ") failed [" << (value) << " > " << (lowerBound) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }



#define CGMLI_ASSERT_EQ(exceptionType, value, testValue, message)			\
  if((value) != (testValue))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert(" << #value << " == " << #testValue << ") failed [" << (value) << " == " << (testValue) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }

#define CGMLI_ASSERT_NE(exceptionType, value, testValue, message)			\
  if((value) == (testValue))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert(" << #value << " != " << #testValue << ") failed [" << (value) << " != " << (testValue) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }

#define CGMLI_ASSERT_NEAR(exceptionType, value, testValue, abs_error, message) \
  if(!(fabs((testValue) - (value)) <= fabs(abs_error)))						\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "assert(" << #value << " == " << #testValue << ") failed [" << (value) << " == " << (testValue) << " (" << fabs((testValue) - (value)) << " > " << fabs(abs_error) << ")]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }



#ifndef NDEBUG

#define CGMLI_THROW_DBG(exceptionType, message){							\
    std::stringstream cgmli_assert_stringstream;							\
    cgmli_assert_stringstream << message;									\
    cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, cgmli_assert_stringstream.str()); \
  }



#define CGMLI_ASSERT_TRUE_DBG(exceptionType, condition, message)			\
  if(!(condition))														\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert(" << #condition << ") failed: " << message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, cgmli_assert_stringstream.str()); \
    }

#define CGMLI_ASSERT_FALSE_DBG(exceptionType, condition, message)			\
  if((condition))														\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert( not " << #condition << ") failed: " << message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, cgmli_assert_stringstream.str()); \
    }


#define CGMLI_ASSERT_DBG_RE( condition, message) CGMLI_ASSERT_DBG(std::runtime_error, condition, message)

#define CGMLI_ASSERT_GE_LT_DBG(exceptionType, value, lowerBound, upperBound, message) \
  if((value) < (lowerBound) || (value) >= (upperBound))							\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert(" << #lowerBound << " <= " << #value << " < " << #upperBound << ") failed [" << (lowerBound) << " <= " << (value) << " < " << (upperBound) << "]: " << message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }



#define CGMLI_ASSERT_LT_DBG(exceptionType, value, upperBound, message)		\
  if((value) >= (upperBound))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert(" << #value << " < " << #upperBound << ") failed [" << (value) << " < " << (upperBound) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }



#define CGMLI_ASSERT_GE_DBG(exceptionType, value, lowerBound, message)		\
  if((value) < (lowerBound))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert(" << #value << " >= " << #lowerBound << ") failed [" << (value) << " >= " << (lowerBound) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }



#define CGMLI_ASSERT_LE_DBG(exceptionType, value, upperBound, message)		\
  if((value) > (upperBound))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert(" << #value << " <= " << #upperBound << ") failed [" << (value) << " <= " << (upperBound) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }

#define CGMLI_ASSERT_GT_DBG(exceptionType, value, lowerBound, message)		\
  if((value) <= (lowerBound))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert(" << #value << " > " << #lowerBound << ") failed [" << (value) << " > " << (lowerBound) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }



#define CGMLI_ASSERT_EQ_DBG(exceptionType, value, testValue, message)		\
  if((value) != (testValue))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert(" << #value << " == " << #testValue << ") failed [" << (value) << " == " << (testValue) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }


#define CGMLI_ASSERT_NE_DBG(exceptionType, value, testValue, message)		\
  if((value) == (testValue))												\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert(" << #value << " != " << #testValue << ") failed [" << (value) << " != " << (testValue) << "]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }



#define CGMLI_ASSERT_NEAR_DBG(exceptionType, value, testValue, abs_error, message) \
  if(!(fabs((testValue) - (value)) <= fabs(abs_error)))						\
    {																	\
      std::stringstream cgmli_assert_stringstream;							\
      cgmli_assert_stringstream << "debug assert(" << #value << " == " << #testValue << ") failed [" << (value) << " == " << (testValue) << " (" << fabs((testValue) - (value)) << " > " << fabs(abs_error) << ")]: " <<  message; \
      cgmli::detail::CGMLI_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,cgmli_assert_stringstream.str()); \
    }


#define CGMLI_OUT(X) std::cout << #X << ": " << (X) << std::endl

#else

#define CGMLI_OUT(X)
#define CGMLI_THROW_DBG(exceptionType, message)
#define CGMLI_ASSERT_TRUE_DBG(exceptionType, condition, message)
#define CGMLI_ASSERT_FALSE_DBG(exceptionType, condition, message)
#define CGMLI_ASSERT_GE_LT_DBG(exceptionType, value, lowerBound, upperBound, message)
#define CGMLI_ASSERT_LT_DBG(exceptionType, value, upperBound, message)
#define CGMLI_ASSERT_GT_DBG(exceptionType, value, lowerBound, message)
#define CGMLI_ASSERT_LE_DBG(exceptionType, value, upperBound, message)
#define CGMLI_ASSERT_GE_DBG(exceptionType, value, lowerBound, message)
#define CGMLI_ASSERT_NE_DBG(exceptionType, value, testValue, message)
#define CGMLI_ASSERT_EQ_DBG(exceptionType, value, testValue, message)
#define CGMLI_ASSERT_NEAR_DBG(exceptionType, value, testValue, abs_error, message)	
#endif



#endif // CGMLI_ASSERT_MACROS_HPP

