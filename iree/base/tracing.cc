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

// Force the header to detect WTF_ENABLE so that this library builds
// (for when building recursively).
#if !defined(WTF_ENABLE)
#define WTF_ENABLE
#endif

#include "iree/base/tracing.h"

#include <thread>  // NOLINT

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "iree/base/file_io.h"
#include "iree/base/file_path.h"
#include "iree/base/initializer.h"
#include "iree/base/logging.h"
#include "iree/base/status.h"

ABSL_FLAG(int32_t, iree_trace_file_period, 5,
          "Seconds between automatic flushing of WTF trace files. 0 to "
          "disable auto-flush.");
ABSL_FLAG(std::string, iree_trace_file, "",
          "wtf-trace file to save if --define=GLOBAL_WTF_ENABLE=1 was used "
          "when building.");

namespace iree {
namespace {

// Guards global WTF state (like the flush fiber and IO).
ABSL_CONST_INIT absl::Mutex global_tracing_mutex(absl::kConstInit);

// True when tracing has been enabled and initialized.
bool global_tracing_initialized ABSL_GUARDED_BY(global_tracing_mutex) = false;

// If there is an existing file at the given path back it up by moving it aside.
// Only kMaxBackups will be kept to avoid unbounded growth.
void RollTraceFiles(const std::string& path) {
  std::string path_stem = file_path::JoinPaths(file_path::DirectoryName(path),
                                               file_path::Stem(path));
  const int kMaxBackups = 5;
  for (int i = kMaxBackups; i >= 0; i--) {
    std::string source_name;
    if (i > 0) {
      source_name = absl::StrCat(path_stem, ".", i, ".wtf-trace");
    } else {
      source_name = path;
    }
    if (!file_io::FileExists(source_name).ok()) {
      continue;
    }

    Status status;
    if (i == kMaxBackups) {
      status = file_io::DeleteFile(source_name);
    } else {
      std::string backup_name =
          absl::StrCat(path_stem, ".", (i + 1), ".wtf-trace");
      status = file_io::MoveFile(source_name, backup_name);
    }
    if (!status.ok()) {
      LOG(WARNING) << "Could not remove backup trace file " << source_name
                   << ": " << status;
    }
  }
}

// Flushes all recorded trace data since the last flush.
void FlushTraceFile(absl::optional<absl::string_view> explicit_trace_path)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(global_tracing_mutex) {
  if (!global_tracing_initialized) return;

  static std::string* current_trace_path = nullptr;
  static ::wtf::Runtime::SaveCheckpoint checkpoint;
  static bool is_first_flush = false;

  // Detect whether explicitly overriding the trace file.
  if (explicit_trace_path) {
    if (!current_trace_path || *current_trace_path != *explicit_trace_path) {
      // Reset.
      delete current_trace_path;
      current_trace_path = new std::string(*explicit_trace_path);

      // Trigger first flush semantics.
      is_first_flush = true;
      checkpoint = ::wtf::Runtime::SaveCheckpoint();
    }
  } else if (!current_trace_path) {
    // Resolve implicitly from flags.
    const auto& implicit_trace_path = absl::GetFlag(FLAGS_iree_trace_file);
    if (!implicit_trace_path.empty()) {
      current_trace_path = new std::string(implicit_trace_path);
      // Trigger first flush semantics.
      is_first_flush = true;
      checkpoint = ::wtf::Runtime::SaveCheckpoint();
    }
  }

  if (!current_trace_path) {
    return;
  }

  if (is_first_flush) {
    // Backup existing any existing trace files at the specified path.
    RollTraceFiles(*current_trace_path);
  }

  auto save_options =
      ::wtf::Runtime::SaveOptions::ForStreamingFile(&checkpoint);
  if (is_first_flush) {
    // On the first time, truncate the file. All subsequent flushes append.
    save_options.open_mode = std::ios_base::trunc;
  }

  is_first_flush = false;

  auto* runtime = ::wtf::Runtime::GetInstance();
  if (!runtime->SaveToFile(*current_trace_path, save_options)) {
    LOG(ERROR) << "Error saving WTF file: " << *current_trace_path;
    return;
  }

  VLOG(1) << "Flushed WTF trace to: " << *current_trace_path;
}

}  // namespace

void InitializeTracing() {
  if (!::wtf::kMasterEnable) {
    if (!absl::GetFlag(FLAGS_iree_trace_file).empty()) {
      LOG(WARNING) << "WTF trace save requested but WTF is not compiled in. "
                   << "Enable by building with --define=GLOBAL_WTF_ENABLE=1.";
    }
    return;
  }

  absl::MutexLock lock(&global_tracing_mutex);
  if (global_tracing_initialized) return;
  global_tracing_initialized = true;

  // Enable tracing on this thread, which we know is main.
  IREE_TRACE_THREAD_ENABLE("main");

  // Register atexit callback to stop tracking.
  atexit(StopTracing);

  LOG(INFO) << "Tracing enabled and streaming to: "
            << absl::GetFlag(FLAGS_iree_trace_file);

  // Launch a thread to periodically flush the trace.
  if (absl::GetFlag(FLAGS_iree_trace_file_period) > 0 &&
      !absl::GetFlag(FLAGS_iree_trace_file).empty()) {
    absl::Duration period =
        absl::Seconds(absl::GetFlag(FLAGS_iree_trace_file_period));
    StartTracingAutoFlush(period);
  }
}

bool IsTracingAvailable() { return ::wtf::kMasterEnable; }

void StartTracingAutoFlush(absl::Duration period) {
  static std::thread flush_thread = ([period]() -> std::thread {
    flush_thread = std::thread([period]() {
      while (true) {
        absl::SleepFor(period);
        absl::MutexLock lock(&global_tracing_mutex);
        if (!global_tracing_initialized) {
          return;
        }
        FlushTraceFile(absl::optional<absl::string_view>());
      }
    });
    flush_thread.detach();
    return std::move(flush_thread);
  })();
}

// Stops tracing if currently initialized.
void StopTracing() {
  if (!::wtf::kMasterEnable) return;
  absl::MutexLock lock(&global_tracing_mutex);
  if (!global_tracing_initialized) return;

  // Flush any pending trace data.
  FlushTraceFile(absl::optional<absl::string_view>());

  // Mark WTF as uninitialized to kill the flush thread.
  global_tracing_initialized = false;

  LOG(INFO) << "Tracing stopped and flushed to file: "
            << absl::GetFlag(FLAGS_iree_trace_file);
}

void FlushTrace(absl::optional<absl::string_view> explicit_trace_path) {
  if (!::wtf::kMasterEnable) return;
  absl::MutexLock lock(&global_tracing_mutex);
  if (!global_tracing_initialized) return;
  FlushTraceFile(explicit_trace_path);
}

}  // namespace iree

IREE_DECLARE_MODULE_INITIALIZER(iree_tracing);

IREE_REGISTER_MODULE_INITIALIZER(iree_tracing, ::iree::InitializeTracing());
