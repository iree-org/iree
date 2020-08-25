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

#ifndef IREE_BASE_INTERNAL_LOGGING_H_
#define IREE_BASE_INTERNAL_LOGGING_H_

#include <cstdint>
#include <limits>
#include <sstream>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "iree/base/target_platform.h"

namespace iree {

// ------------------------------------------------------------------------- //
// |                                 LOG                                   | //
// ------------------------------------------------------------------------- //

// Severity levels for LOG().
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

  // Returns the minimum log level for VLOG statements.
  // E.g., if MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
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
  LogMessageFatal(const char* file, int line) ABSL_ATTRIBUTE_COLD;
  ABSL_ATTRIBUTE_NORETURN ~LogMessageFatal();
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

#define LOG(severity) _IREE_LOG_##severity

#ifndef NDEBUG
#define DLOG LOG
#else
#define DLOG(severity) \
  switch (0)           \
  default:             \
    ::iree::internal::NullStream().stream()
#endif

#define VLOG_IS_ON(lvl) ((lvl) <= ::iree::internal::LogMessage::MinVLogLevel())

#define VLOG(lvl)                          \
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(lvl))) \
  ::iree::internal::LogMessage(__FILE__, __LINE__, ::iree::INFO)

// `DVLOG` behaves like `VLOG` in debug mode (i.e. `#ifndef NDEBUG`).
// Otherwise, it compiles away and does nothing.
#ifndef NDEBUG
#define DVLOG VLOG
#else
#define DVLOG(verbose_level) \
  while (false && (verbose_level) > 0) ::iree::internal::NullStream().stream()
#endif  // !NDEBUG

// ------------------------------------------------------------------------- //
// |                                CHECK                                  | //
// ------------------------------------------------------------------------- //

// CHECK dies with a fatal error if condition is not true.  It is *not*
// controlled by NDEBUG, so the check will be executed regardless of
// compilation mode.  Therefore, it is safe to do things like:
//    CHECK(fp->Write(x) == 4)
#define CHECK(condition)                \
  if (ABSL_PREDICT_FALSE(!(condition))) \
  LOG(FATAL) << "Check failed: " #condition " "

// Function is overloaded for integral types to allow static const
// integrals declared in classes and not defined to be used as arguments to
// CHECK* macros. It's not encouraged though.
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

// This formats a value for a failing CHECK_XX statement.  Ordinarily,
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
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& p);

// A container for a string pointer which can be evaluated to a bool -
// true iff the pointer is non-NULL.
struct CheckOpString {
  CheckOpString(std::string* str) : str_(str) {}  // NOLINT
  // No destructor: if str_ is non-NULL, we're about to LOG(FATAL),
  // so there's no point in cleaning up str_.
  operator bool() const { return ABSL_PREDICT_FALSE(str_ != NULL); }
  std::string* str_;
};

// Build the error message string. Specify no inlining for code size.
template <typename T1, typename T2>
std::string* MakeCheckOpString(const T1& v1, const T2& v2,
                               const char* exprtext) ABSL_ATTRIBUTE_NOINLINE;

// A helper class for formatting "expr (V1 vs. V2)" in a CHECK_XX
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

// Helper functions for CHECK_OP macro.
// The (int, int) specialization works around the issue that the compiler
// will not instantiate the template version of the function on values of
// unnamed enum type - see comment below.
// The (size_t, int) and (int, size_t) specialization are to handle unsigned
// comparison errors while still being thorough with the comparison.
#define _IREE_DEFINE_CHECK_OP_IMPL(name, op)                             \
  template <typename T1, typename T2>                                    \
  inline std::string* name##Impl(const T1& v1, const T2& v2,             \
                                 const char* exprtext) {                 \
    if (ABSL_PREDICT_TRUE(v1 op v2))                                     \
      return NULL;                                                       \
    else                                                                 \
      return ::iree::internal::MakeCheckOpString(v1, v2, exprtext);      \
  }                                                                      \
  inline std::string* name##Impl(int v1, int v2, const char* exprtext) { \
    return name##Impl<int, int>(v1, v2, exprtext);                       \
  }                                                                      \
  inline std::string* name##Impl(const size_t v1, const int v2,          \
                                 const char* exprtext) {                 \
    if (ABSL_PREDICT_FALSE(v2 < 0)) {                                    \
      return ::iree::internal::MakeCheckOpString(v1, v2, exprtext);      \
    }                                                                    \
    const size_t uval = (size_t)((unsigned)v1);                          \
    return name##Impl<size_t, size_t>(uval, v2, exprtext);               \
  }                                                                      \
  inline std::string* name##Impl(const int v1, const size_t v2,          \
                                 const char* exprtext) {                 \
    if (ABSL_PREDICT_FALSE(v2 >= std::numeric_limits<int>::max())) {     \
      return ::iree::internal::MakeCheckOpString(v1, v2, exprtext);      \
    }                                                                    \
    const size_t uval = (size_t)((unsigned)v2);                          \
    return name##Impl<size_t, size_t>(v1, uval, exprtext);               \
  }

// We use the full name Check_EQ, Check_NE, etc. in case the file including
// base/logging.h provides its own #defines for the simpler names EQ, NE, etc.
_IREE_DEFINE_CHECK_OP_IMPL(Check_EQ,
                           ==)  // Compilation error with CHECK_EQ(NULL, x)?
_IREE_DEFINE_CHECK_OP_IMPL(Check_NE, !=)  // Use CHECK(x == NULL) instead.
_IREE_DEFINE_CHECK_OP_IMPL(Check_LE, <=)
_IREE_DEFINE_CHECK_OP_IMPL(Check_LT, <)
_IREE_DEFINE_CHECK_OP_IMPL(Check_GE, >=)
_IREE_DEFINE_CHECK_OP_IMPL(Check_GT, >)
#undef _IREE_DEFINE_CHECK_OP_IMPL

// In optimized mode, use CheckOpString to hint to compiler that
// the while condition is unlikely.
#define CHECK_OP_LOG(name, op, val1, val2)                      \
  while (::iree::internal::CheckOpString _result =              \
             ::iree::internal::name##Impl(                      \
                 ::iree::internal::GetReferenceableValue(val1), \
                 ::iree::internal::GetReferenceableValue(val2), \
                 #val1 " " #op " " #val2))                      \
  ::iree::internal::LogMessageFatal(__FILE__, __LINE__) << *(_result.str_)

#define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2)

// CHECK_EQ/NE/...
#define CHECK_EQ(val1, val2) CHECK_OP(Check_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(Check_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(Check_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(Check_LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(Check_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(Check_GT, >, val1, val2)

#ifndef NDEBUG
#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)

#else

#define DCHECK(condition) \
  while (false && (condition)) LOG(FATAL)

// NDEBUG is defined, so DCHECK_EQ(x, y) and so on do nothing.
// However, we still want the compiler to parse x and y, because
// we don't want to lose potentially useful errors and warnings.
// _DCHECK_NOP is a helper, and should not be used outside of this file.
#define _IREE_DCHECK_NOP(x, y) \
  while (false && ((void)(x), (void)(y), 0)) LOG(FATAL)

#define DCHECK_EQ(x, y) _IREE_DCHECK_NOP(x, y)
#define DCHECK_NE(x, y) _IREE_DCHECK_NOP(x, y)
#define DCHECK_LE(x, y) _IREE_DCHECK_NOP(x, y)
#define DCHECK_LT(x, y) _IREE_DCHECK_NOP(x, y)
#define DCHECK_GE(x, y) _IREE_DCHECK_NOP(x, y)
#define DCHECK_GT(x, y) _IREE_DCHECK_NOP(x, y)

#endif  // !NDEBUG

// These are for when you don't want a CHECK failure to print a verbose
// stack trace.  The implementation of CHECK* in this file already doesn't.
#define QCHECK(condition) CHECK(condition)
#define QCHECK_EQ(x, y) CHECK_EQ(x, y)
#define QCHECK_NE(x, y) CHECK_NE(x, y)
#define QCHECK_LE(x, y) CHECK_LE(x, y)
#define QCHECK_LT(x, y) CHECK_LT(x, y)
#define QCHECK_GE(x, y) CHECK_GE(x, y)
#define QCHECK_GT(x, y) CHECK_GT(x, y)

}  // namespace internal
}  // namespace iree

#endif  // IREE_BASE_INTERNAL_LOGGING_H_
