// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_BASE_LOGGING_H_
#define IREE_BASE_LOGGING_H_

// IREE_LOG(severity) << ...;
//   Logs a message at the given severity.
//   Severity:
//     INFO    Logs information text.
//     WARNING Logs a warning.
//     ERROR   Logs an error.
//     FATAL   Logs an error and exit(1).
//
// IREE_DLOG(severity) << ...;
//   Behaves like `IREE_LOG` in debug mode (i.e. `#ifndef NDEBUG`).
//   Otherwise, it compiles away and does nothing.
//
// IREE_VLOG(level) << ...;
//   Logs a verbose message at the given verbosity level.
//
// IREE_DVLOG(level) << ...;
//   Behaves like `IREE_VLOG` in debug mode (i.e. `#ifndef NDEBUG`).
//   Otherwise, it compiles away and does nothing.
//
// IREE_CHECK(condition) << ...;
//   Runtime asserts that the given condition is true even in release builds.
//   It's recommended that IREE_DCHECK is used instead as too many CHECKs
//   can impact performance.
//
// IREE_CHECK_EQ|NE|LT|GT|LE|GE(val1, val2) << ...;
//   Runtime assert the specified operation with the given values.
//
// IREE_DCHECK(condition) << ...;
//   Runtime asserts that the given condition is true only in non-opt builds.
//
// IREE_DCHECK_EQ|NE|LT|GT|LE|GE(val1, val2) << ...;
//   Runtime assert the specified operation with the given values in non-opt
//   builds.
//
// IREE_QCHECK(condition) << ...;
// IREE_QCHECK_EQ|NE|LT|GT|LE|GE(val1, val2) << ...;
//   These behave like `IREE_CHECK` but do not print a full stack trace.
//   They are useful when problems are definitely unrelated to program flow,
//   e.g. when validating user input.

#include <cstdint>
#include <limits>
#include <sstream>

#include "iree/base/attributes.h"

namespace iree {

// ------------------------------------------------------------------------- //
// |                               IREE_LOG                                | //
// ------------------------------------------------------------------------- //

// Severity levels for IREE_LOG().
const int INFO = 0;     // absl::LogSeverity::kInfo
const int WARNING = 1;  // absl::LogSeverity::kWarning
const int ERROR = 2;    // absl::LogSeverity::kError
const int FATAL = 3;    // absl::LogSeverity::kFatal

namespace internal {

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* file_name, int line, int severity);
  ~LogMessage();

  const char* file_name() const { return file_name_; }
  int line() const { return line_; }
  int severity() const { return severity_; }

  // Returns the minimum log level for IREE_VLOG statements.
  // E.g., if MinVLogLevel() is 2, then IREE_VLOG(2) statements will produce
  // output, but IREE_VLOG(3) will not. Defaults to 0.
  static int64_t MinVLogLevel();

 protected:
  void EmitLogMessage();

 private:
  const char* file_name_;
  int line_;
  int severity_;
};

// LogMessageFatal ensures the process exits in failure after logging a message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) IREE_ATTRIBUTE_COLD;
  IREE_ATTRIBUTE_NORETURN ~LogMessageFatal();
};

// NullStream implements operator<< but does nothing.
class NullStream {
 public:
  NullStream& stream() { return *this; }
};
template <typename T>
inline NullStream& operator<<(NullStream& str, const T&) {
  return str;
}
inline NullStream& operator<<(NullStream& str,
                              std::ostream& (*)(std::ostream& os)) {
  return str;
}
inline NullStream& operator<<(NullStream& str,
                              std::ios_base& (*)(std::ios_base& os)) {
  return str;
}

#define _IREE_LOG_INFO \
  ::iree::internal::LogMessage(__FILE__, __LINE__, ::iree::INFO)
#define _IREE_LOG_WARNING \
  ::iree::internal::LogMessage(__FILE__, __LINE__, ::iree::WARNING)
#define _IREE_LOG_ERROR \
  ::iree::internal::LogMessage(__FILE__, __LINE__, ::iree::ERROR)
#define _IREE_LOG_FATAL ::iree::internal::LogMessageFatal(__FILE__, __LINE__)

#define IREE_LOG(severity) _IREE_LOG_##severity

#ifndef NDEBUG
#define IREE_DLOG IREE_LOG
#else
#define IREE_DLOG(severity) \
  switch (0)                \
  default:                  \
    ::iree::internal::NullStream().stream()
#endif

#define IREE_VLOG_IS_ON(lvl) \
  ((lvl) <= ::iree::internal::LogMessage::MinVLogLevel())

#define IREE_VLOG(lvl)                     \
  if (IREE_UNLIKELY(IREE_VLOG_IS_ON(lvl))) \
  ::iree::internal::LogMessage(__FILE__, __LINE__, ::iree::INFO)

// `IREE_DVLOG` behaves like `IREE_VLOG` in debug mode (i.e. `#ifndef NDEBUG`).
// Otherwise, it compiles away and does nothing.
#ifndef NDEBUG
#define IREE_DVLOG IREE_VLOG
#else
#define IREE_DVLOG(verbose_level) \
  while (false && (verbose_level) > 0) ::iree::internal::NullStream().stream()
#endif  // !NDEBUG

// ------------------------------------------------------------------------- //
// |                              IREE_CHECK                               | //
// ------------------------------------------------------------------------- //

// IREE_CHECK dies with a fatal error if condition is not true.  It is *not*
// controlled by NDEBUG, so the check will be executed regardless of
// compilation mode.  Therefore, it is safe to do things like:
//    IREE_CHECK(fp->Write(x) == 4)
#define IREE_CHECK(condition)      \
  if (IREE_UNLIKELY(!(condition))) \
  IREE_LOG(FATAL) << "Check failed: " #condition " "

// Function is overloaded for integral types to allow static const
// integrals declared in classes and not defined to be used as arguments to
// IREE_CHECK* macros. It's not encouraged though.
template <typename T>
inline const T& GetReferenceableValue(const T& t) {
  return t;
}
inline char GetReferenceableValue(char t) { return t; }
inline int8_t GetReferenceableValue(int8_t t) { return t; }
inline uint8_t GetReferenceableValue(uint8_t t) { return t; }
inline int16_t GetReferenceableValue(int16_t t) { return t; }
inline uint16_t GetReferenceableValue(uint16_t t) { return t; }
inline int32_t GetReferenceableValue(int32_t t) { return t; }
inline uint32_t GetReferenceableValue(uint32_t t) { return t; }
inline int64_t GetReferenceableValue(int64_t t) { return t; }
inline uint64_t GetReferenceableValue(uint64_t t) { return t; }

// This formats a value for a failing IREE_CHECK_XX statement.  Ordinarily,
// it uses the definition for operator<<, with a few special cases below.
template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T& v) {
  (*os) << v;
}

// Overrides for char types provide readable values for unprintable
// characters.
template <>
void MakeCheckOpValueString(std::ostream* os, const char& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const int8_t& v);
template <>
void MakeCheckOpValueString(std::ostream* os, const uint8_t& v);
// We need an explicit specialization for std::nullptr_t.
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v);

// A container for a string pointer which can be evaluated to a bool -
// true iff the pointer is non-NULL.
struct CheckOpString {
  CheckOpString(std::string* str) : str_(str) {}  // NOLINT
  // No destructor: if str_ is non-NULL, we're about to IREE_LOG(FATAL),
  // so there's no point in cleaning up str_.
  operator bool() const { return IREE_UNLIKELY(str_ != NULL); }
  std::string* str_;
};

// Build the error message string. Specify no inlining for code size.
template <typename T1, typename T2>
std::string* MakeCheckOpString(const T1& v1, const T2& v2,
                               const char* exprtext) IREE_ATTRIBUTE_NOINLINE;

// A helper class for formatting "expr (V1 vs. V2)" in a IREE_CHECK_XX
// statement. See MakeCheckOpString for sample usage.
class CheckOpMessageBuilder {
 public:
  // Inserts "exprtext" and " (" to the stream.
  explicit CheckOpMessageBuilder(const char* exprtext);
  // Deletes "stream_".
  ~CheckOpMessageBuilder();
  // For inserting the first variable.
  std::ostream* ForVar1() { return stream_; }
  // For inserting the second variable (adds an intermediate " vs. ").
  std::ostream* ForVar2();
  // Get the result (inserts the closing ")").
  std::string* NewString();

 private:
  std::ostringstream* stream_;
};

template <typename T1, typename T2>
std::string* MakeCheckOpString(const T1& v1, const T2& v2,
                               const char* exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

// Helper functions for IREE_CHECK_OP macro.
// The (int, int) specialization works around the issue that the compiler
// will not instantiate the template version of the function on values of
// unnamed enum type - see comment below.
// The (size_t, int) and (int, size_t) specialization are to handle unsigned
// comparison errors while still being thorough with the comparison.
#define _IREE_DEFINE_CHECK_OP_IMPL(name, op)                             \
  template <typename T1, typename T2>                                    \
  inline std::string* name##Impl(const T1& v1, const T2& v2,             \
                                 const char* exprtext) {                 \
    if (IREE_LIKELY(v1 op v2))                                           \
      return NULL;                                                       \
    else                                                                 \
      return ::iree::internal::MakeCheckOpString(v1, v2, exprtext);      \
  }                                                                      \
  inline std::string* name##Impl(int v1, int v2, const char* exprtext) { \
    return name##Impl<int, int>(v1, v2, exprtext);                       \
  }                                                                      \
  inline std::string* name##Impl(const size_t v1, const int v2,          \
                                 const char* exprtext) {                 \
    if (IREE_UNLIKELY(v2 < 0)) {                                         \
      return ::iree::internal::MakeCheckOpString(v1, v2, exprtext);      \
    }                                                                    \
    const size_t uval = (size_t)((unsigned)v1);                          \
    return name##Impl<size_t, size_t>(uval, v2, exprtext);               \
  }                                                                      \
  inline std::string* name##Impl(const int v1, const size_t v2,          \
                                 const char* exprtext) {                 \
    if (IREE_UNLIKELY(v2 >= std::numeric_limits<int>::max())) {          \
      return ::iree::internal::MakeCheckOpString(v1, v2, exprtext);      \
    }                                                                    \
    const size_t uval = (size_t)((unsigned)v2);                          \
    return name##Impl<size_t, size_t>(v1, uval, exprtext);               \
  }

_IREE_DEFINE_CHECK_OP_IMPL(Check_EQ, ==)
_IREE_DEFINE_CHECK_OP_IMPL(Check_NE, !=)
_IREE_DEFINE_CHECK_OP_IMPL(Check_LE, <=)
_IREE_DEFINE_CHECK_OP_IMPL(Check_LT, <)
_IREE_DEFINE_CHECK_OP_IMPL(Check_GE, >=)
_IREE_DEFINE_CHECK_OP_IMPL(Check_GT, >)
#undef _IREE_DEFINE_CHECK_OP_IMPL

// In optimized mode, use CheckOpString to hint to compiler that
// the while condition is unlikely.
#define IREE_CHECK_OP_LOG(name, op, val1, val2)                 \
  while (::iree::internal::CheckOpString _result =              \
             ::iree::internal::name##Impl(                      \
                 ::iree::internal::GetReferenceableValue(val1), \
                 ::iree::internal::GetReferenceableValue(val2), \
                 #val1 " " #op " " #val2))                      \
  ::iree::internal::LogMessageFatal(__FILE__, __LINE__) << *(_result.str_)

#define IREE_CHECK_OP(name, op, val1, val2) \
  IREE_CHECK_OP_LOG(name, op, val1, val2)

// IREE_CHECK_EQ/NE/...
#define IREE_CHECK_EQ(val1, val2) IREE_CHECK_OP(Check_EQ, ==, val1, val2)
#define IREE_CHECK_NE(val1, val2) IREE_CHECK_OP(Check_NE, !=, val1, val2)
#define IREE_CHECK_LE(val1, val2) IREE_CHECK_OP(Check_LE, <=, val1, val2)
#define IREE_CHECK_LT(val1, val2) IREE_CHECK_OP(Check_LT, <, val1, val2)
#define IREE_CHECK_GE(val1, val2) IREE_CHECK_OP(Check_GE, >=, val1, val2)
#define IREE_CHECK_GT(val1, val2) IREE_CHECK_OP(Check_GT, >, val1, val2)

#ifndef NDEBUG
#define IREE_DCHECK(condition) IREE_CHECK(condition)
#define IREE_DCHECK_EQ(val1, val2) IREE_CHECK_EQ(val1, val2)
#define IREE_DCHECK_NE(val1, val2) IREE_CHECK_NE(val1, val2)
#define IREE_DCHECK_LE(val1, val2) IREE_CHECK_LE(val1, val2)
#define IREE_DCHECK_LT(val1, val2) IREE_CHECK_LT(val1, val2)
#define IREE_DCHECK_GE(val1, val2) IREE_CHECK_GE(val1, val2)
#define IREE_DCHECK_GT(val1, val2) IREE_CHECK_GT(val1, val2)

#else

#define IREE_DCHECK(condition) \
  while (false && (condition)) IREE_LOG(FATAL)

// NDEBUG is defined, so IREE_DCHECK_EQ(x, y) and so on do nothing.
// However, we still want the compiler to parse x and y, because
// we don't want to lose potentially useful errors and warnings.
// _IREE_DCHECK_NOP is a helper, and should not be used outside of this file.
#define _IREE_DCHECK_NOP(x, y) \
  while (false && ((void)(x), (void)(y), 0)) IREE_LOG(FATAL)

#define IREE_DCHECK_EQ(x, y) _IREE_DCHECK_NOP(x, y)
#define IREE_DCHECK_NE(x, y) _IREE_DCHECK_NOP(x, y)
#define IREE_DCHECK_LE(x, y) _IREE_DCHECK_NOP(x, y)
#define IREE_DCHECK_LT(x, y) _IREE_DCHECK_NOP(x, y)
#define IREE_DCHECK_GE(x, y) _IREE_DCHECK_NOP(x, y)
#define IREE_DCHECK_GT(x, y) _IREE_DCHECK_NOP(x, y)

#endif  // !NDEBUG

// These are for when you don't want a IREE_CHECK failure to print a verbose
// stack trace.  The implementation of IREE_CHECK* in this file already doesn't.
#define IREE_QCHECK(condition) IREE_CHECK(condition)
#define IREE_QCHECK_EQ(x, y) IREE_CHECK_EQ(x, y)
#define IREE_QCHECK_NE(x, y) IREE_CHECK_NE(x, y)
#define IREE_QCHECK_LE(x, y) IREE_CHECK_LE(x, y)
#define IREE_QCHECK_LT(x, y) IREE_CHECK_LT(x, y)
#define IREE_QCHECK_GE(x, y) IREE_CHECK_GE(x, y)
#define IREE_QCHECK_GT(x, y) IREE_CHECK_GT(x, y)

}  // namespace internal
}  // namespace iree

#endif  // IREE_BASE_LOGGING_H_
