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

#include "iree/base/logging.h"

#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"
#include "iree/base/tracing.h"

ABSL_FLAG(int, iree_minloglevel, 0,
          "Minimum logging level. 0 = INFO and above.");
ABSL_FLAG(int, iree_v, 0,
          "Verbosity level maximum. 1 = IREE_VLOG(0-1), 2 = IREE_VLOG(0-2).");

namespace iree {
namespace internal {

namespace {

// Parse log level (int64_t) from environment variable (char*).
// Returns true if the value was present and parsed successfully.
bool LogLevelStrToInt(const char* iree_env_var_val, int64_t* out_level) {
  *out_level = 0;
  if (iree_env_var_val == nullptr) {
    return false;
  }

  std::string min_log_level(iree_env_var_val);
  std::istringstream ss(min_log_level);
  int64_t level;
  if (!(ss >> level)) {
    // Invalid vlog level setting, set level to default (0).
    return false;
  }

  *out_level = level;
  return true;
}

int64_t MinLogLevelFromEnv() {
  const char* iree_env_var_val = getenv("IREE_MIN_LOG_LEVEL");
  int64_t level = 0;
  if (LogLevelStrToInt(iree_env_var_val, &level)) {
    return level;
  }
  return absl::GetFlag(FLAGS_iree_minloglevel);
}

int64_t MinVLogLevelFromEnv() {
  const char* iree_env_var_val = getenv("IREE_MIN_VLOG_LEVEL");
  int64_t level = 0;
  if (LogLevelStrToInt(iree_env_var_val, &level)) {
    return level;
  }
  return absl::GetFlag(FLAGS_iree_v);
}

}  // namespace

LogMessage::LogMessage(const char* file_name, int line, int severity)
    : file_name_(file_name), line_(line), severity_(severity) {}

LogMessage::~LogMessage() {
  // Read the min log level once during the first call to logging.
  static int64_t min_log_level = MinLogLevelFromEnv();
  if (IREE_LIKELY(severity_ >= min_log_level)) {
    EmitLogMessage();
  }
}

int64_t LogMessage::MinVLogLevel() {
  static int64_t min_vlog_level = MinVLogLevelFromEnv();
  return min_vlog_level;
}

void LogMessage::EmitLogMessage() {
  // TODO(scotttodd): Include current system time
  fprintf(stderr, "%c %s:%d] %s\n", "IWEF"[severity_], file_name_, line_,
          str().c_str());

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_LOG_MESSAGES
  constexpr int kLevelColors[4] = {
      IREE_TRACING_MESSAGE_LEVEL_INFO,     // INFO
      IREE_TRACING_MESSAGE_LEVEL_WARNING,  // WARNING
      IREE_TRACING_MESSAGE_LEVEL_ERROR,    // ERROR
      IREE_TRACING_MESSAGE_LEVEL_ERROR,    // FATAL
  };
  std::string message =
      absl::StrFormat("%s:%d] %s\n", file_name_, line_, str().c_str());
  IREE_TRACE_MESSAGE_DYNAMIC_COLORED(kLevelColors[severity_], message.c_str(),
                                     message.size());
#endif  // IREE_TRACING_FEATURES& IREE_TRACING_FEATURE_LOG_MESSAGES
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  EmitLogMessage();

  // abort() ensures we don't return (as promised via ATTRIBUTE_NORETURN).
  abort();
}

template <>
void MakeCheckOpValueString(std::ostream* os, const char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<int16_t>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const int8_t& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<int16_t>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const uint8_t& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<uint16_t>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v) {
  (*os) << "nullptr";
}

CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << "Check failed: " << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete stream_; }

std::ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}

std::string* CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new std::string(stream_->str());
}

}  // namespace internal
}  // namespace iree
