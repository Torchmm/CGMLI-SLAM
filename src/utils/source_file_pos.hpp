
#ifndef CGMLI_SOURCE_FILE_POS_HPP
#define CGMLI_SOURCE_FILE_POS_HPP

#include <string>
#include <iostream>
#include <sstream>
// A class and macro that gives you the current file position.

/// \brief cgmli Main namespace of this package.
namespace cgmli {

/// \brief Helper class to get source file positions.
class source_file_pos {
public:
  std::string function; ///< The function.
  std::string file; ///< The file.
  int line; ///< The line.

  /// \brief Constructor.
  /// \param function The function.
  /// \param file The file.
  /// \param line The line.
  source_file_pos(std::string function, std::string file, int line) :
    function(function), file(file), line(line) {}

  /// \brief Return as string.
  /// \return String message.
  operator std::string()
  {
    return toString();
  }

  /// \brief Return as string.
  /// \return String message.
  std::string toString() const
  {
    std::stringstream s;
    s << file << ":" << line << ": " << function << "()";
    return s.str();
  }

};

}// namespace cgmli

/// \brief Print to stream.
/// \param out Out stream buffer.
/// \param sfp cgmli::source_file_pos object to print.
/// \return The stream object.
inline std::ostream & operator<<(std::ostream & out, const cgmli::source_file_pos & sfp)
{
  out << sfp.file << ":" << sfp.line << ": " << sfp.function << "()";
  return out;
}


#define CGMLI_SOURCE_FILE_POS cgmli::source_file_pos(__FUNCTION__,__FILE__,__LINE__)

#endif // CGMLI_SOURCE_FILE_POS_HPP
