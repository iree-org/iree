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
#include <sstream>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "iree/base/platform_headers.h"

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

// TODO(scotttodd): Log information about the check failure
#define CHECK_OP_LOG(name, op, val1, val2)   \
  if (ABSL_PREDICT_FALSE(!((val1)op(val2)))) \
  LOG(FATAL) << "Check " #name " failed: " #val1 " " #op " " #val2

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
