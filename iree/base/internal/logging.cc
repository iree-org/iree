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

#include "iree/base/internal/logging.h"

#include <string>

#include "absl/flags/flag.h"

ABSL_FLAG(int, iree_minloglevel, 0,
          "Minimum logging level. 0 = INFO and above.");
ABSL_FLAG(int, iree_v, 0,
          "Verbosity level maximum. 1 = VLOG(0-1), 2 = VLOG(0-2).");
ABSL_FLAG(bool, iree_logtostderr, false, "Logs to stderr instead of stdout");

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
  if (ABSL_PREDICT_TRUE(severity_ >= min_log_level)) {
    EmitLogMessage();
  }
}

int64_t LogMessage::MinVLogLevel() {
  static int64_t min_vlog_level = MinVLogLevelFromEnv();
  return min_vlog_level;
}

void LogMessage::EmitLogMessage() {
  // TODO(scotttodd): Include current system time
  fprintf(absl::GetFlag(FLAGS_iree_logtostderr) ? stderr : stdout,
          "%c %s:%d] %s\n", "IWEF"[severity_], file_name_, line_,
          str().c_str());
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  EmitLogMessage();

  // abort() ensures we don't return (as promised via ATTRIBUTE_NORETURN).
  abort();
}

}  // namespace internal
}  // namespace iree
