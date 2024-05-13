// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pthread.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <ratio>
#include <string>
#include <vector>

#include "experimental/hal_executable_library_call_hooks/perf_event_linux.h"
#include "experimental/hal_executable_library_call_hooks/stats.h"
#include "iree/base/api.h"
#include "iree/base/string_view.h"
#include "iree/hal/local/executable_library.h"

namespace {

using Clock = std::chrono::steady_clock;

// Records the data obtained about one instrumented call. Each ThreadInfo
// records essentially a big vector<CallInfo>.
struct CallInfo {
  float duration_us = 0;
  unsigned int cpu = -1;
  unsigned int node = -1;
  std::vector<int64_t> perf_event_counts;
};

// Records all the data obtained about all the calls made in one thread.
// Outputs statistics and/or CSV files on destruction.
class ThreadInfo {
 public:
  // Constructor: should only be called once per thread to create a
  // thread-specific singleton.
  ThreadInfo();
  // Destructor: this is what prints out all the output.
  ~ThreadInfo();

  // Public hook: start of a new call.
  void beginCall(const char *name);

  // Public hook: end of call started by the last beginCall().
  void endCall();

 private:
  // Whether we are between a beginCall and an endCall.
  bool recording_call_ = false;
  // Timepoint of the current beginCall.
  Clock::time_point current_call_start_time_;
  // If not empty, will filter only calls to the specific named function.
  std::string name_filter_;
  // The entire record of calls made on this thread.
  std::vector<CallInfo> entries_;
  // How many milliseconds to skip initially before actually recording calls.
  float skip_start_ms_ = 0;
  // Timepoint of initialization of this class instance.
  Clock::time_point start_time_;
  // perf_event_open file descriptors used to query event counts.
  std::vector<PerfEventFd> perf_event_fds_;
  // Types of events to query.
  std::vector<PerfEventType> perf_event_types_;
  // If not null, will dump CSV there.
  const char *output_csv_ = nullptr;

  // Helper: show stats.
  void printStats();
  // Helper: dump CSV data file.
  void printCsv(const char *path_base);
};

// Returns the number of microseconds between `start` and `end`.
float elapsedMicroseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<float, std::micro>(end - start).count();
}

// Returns the calling thread's name (platform-specific).
std::string getThreadName() {
  char buf[64];
  pthread_getname_np(pthread_self(), buf, sizeof buf);
  return buf;
}

ThreadInfo::ThreadInfo() {
  const char *name_filter_env = getenv("IREE_HOOK_FILTER_NAME");
  if (name_filter_env) {
    name_filter_ = name_filter_env;
  }
  const char *skip_start_ms_env = getenv("IREE_HOOK_SKIP_START_MS");
  if (skip_start_ms_env) {
    skip_start_ms_ = strtof(skip_start_ms_env, nullptr);
  }
  const char *perf_event_types_env = getenv("IREE_HOOK_PERF_EVENT_TYPES");
  if (perf_event_types_env) {
    perf_event_types_ = parsePerfEventTypes(perf_event_types_env);
    for (PerfEventType perf_event_type : perf_event_types_) {
      perf_event_fds_.emplace_back(perf_event_type);
    }
  }
  output_csv_ = getenv("IREE_HOOK_OUTPUT_CSV");
  start_time_ = Clock::now();
}
ThreadInfo::~ThreadInfo() {
  if (entries_.empty()) {
    return;
  }
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  if (getenv("IREE_HOOK_LIST_EVENT_TYPES")) {
    printAllEventTypesAndDescriptions(stderr);
    exit(0);
  }
  std::string thread_name = getThreadName();
  printStats();
  if (output_csv_) {
    printCsv(output_csv_);
  } else {
    fprintf(stderr,
            "Note: to get the whole CSV data, set "
            "IREE_HOOK_OUTPUT_CSV=/tmp/path\n");
  }
  fprintf(stderr, "\n");
}

void ThreadInfo::printStats() {
  // duration_us is the 0-th data_vector.
  const size_t data_vectors_count = perf_event_types_.size() + 1;
  std::vector<const char *> data_vector_names(data_vectors_count);
  std::vector<std::vector<float>> data_vectors(data_vectors_count);
  for (const auto &entry : entries_) {
    data_vector_names[0] = "duration_ms";
    data_vectors[0].push_back(entry.duration_us);
    for (size_t i = 0; i < perf_event_types_.size(); ++i) {
      data_vector_names[i + 1] = perf_event_types_[i].name;
      data_vectors[i + 1].push_back(entry.perf_event_counts[i]);
    }
  }
  const char *bucket_count_env = getenv("IREE_HOOK_BUCKET_COUNT");
  int bucket_count =
      bucket_count_env ? strtol(bucket_count_env, nullptr, 10) : 16;
  std::vector<std::vector<float>> data_vector_bucket_means(data_vectors_count);
  std::vector<std::vector<int>> data_vector_bucket_indices(data_vectors_count);
  for (size_t i = 0; i < data_vectors_count; ++i) {
    splitIntoBuckets(data_vectors[i], bucket_count,
                     &data_vector_bucket_means[i],
                     &data_vector_bucket_indices[i]);
  }

  std::string thread_name = getThreadName();
  printf("Statistics for thread %s:\n", thread_name.c_str());
  printf("  %zu matching calls, of which:\n", entries_.size());
  std::map<unsigned int, int> cpu_counts;
  for (const auto &entry : entries_) {
    if (cpu_counts.count(entry.cpu))
      cpu_counts[entry.cpu]++;
    else
      cpu_counts[entry.cpu] = 1;
  }
  for (auto p : cpu_counts) {
    printf("    %d calls on cpu %u\n", p.second, p.first);
  }

  for (size_t i = 0; i < data_vectors_count; ++i) {
    printf("  %s:\n", data_vector_names[i]);
    printf("    mean: %.3g\n", mean(data_vectors[i]));
    printf("    %d-ile means:", bucket_count);
    for (float m : data_vector_bucket_means[i]) {
      printf(" %.3g", m);
    }
    printf("\n");

    for (size_t j = 0; j < i; ++j) {
      printf("  correlation of %s vs. %s: %.2g\n", data_vector_names[j],
             data_vector_names[i],
             correlation(data_vectors[j], data_vectors[i]));

      if (getenv("IREE_HOOK_NO_PROBABILITY_TABLE")) {
        continue;
      }
      std::vector<float> cond_proba_table;
      computeConditionalProbabilityTable(
          bucket_count, data_vector_bucket_indices[i],
          data_vector_bucket_indices[j], &cond_proba_table);
      printf(
          "  conditional probability of %s %d-ile (↓) given "
          "%s %d-ile (→):\n",
          data_vector_names[j], bucket_count, data_vector_names[i],
          bucket_count);
      printConditionalProbabilityTable(stdout, bucket_count, cond_proba_table);
    }
  }
}

void ThreadInfo::printCsv(const char *path_base) {
  char path[256];
  std::string thread_name = getThreadName();
  snprintf(path, sizeof path, "%s/iree_hook_%s.csv", path_base,
           thread_name.c_str());
  FILE *file = fopen(path, "w");
  if (!file) {
    fprintf(stderr, "Failed to open %s for write.\n", path);
    exit(1);
  }
  fprintf(file, "thread,cpu,node,duration_us");
  for (auto perf_event_types : perf_event_types_) {
    fprintf(file, ",%s", perf_event_types.name);
  }
  fprintf(file, "\n");
  for (const auto &entry : entries_) {
    fprintf(file, "%s,%u,%u,%g", thread_name.c_str(), entry.cpu, entry.node,
            entry.duration_us);
    for (int64_t event_count : entry.perf_event_counts) {
      fprintf(file, ",%ld", event_count);
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

void ThreadInfo::beginCall(const char *name) {
  if (!name_filter_.empty() && name_filter_ != std::string(name)) {
    return;
  }
  Clock::time_point call_start_time = Clock::now();
  if (elapsedMicroseconds(start_time_, call_start_time) <
      skip_start_ms_ * 1000.f) {
    return;
  }
  for (auto &perf_event_fd : perf_event_fds_) {
    perf_event_fd.reset();
  }
  for (auto &perf_event_fd : perf_event_fds_) {
    perf_event_fd.enable();
  }
  recording_call_ = true;
  current_call_start_time_ = call_start_time;
}

void ThreadInfo::endCall() {
  if (!recording_call_) {
    return;
  }
  CallInfo entry;
  entry.duration_us =
      elapsedMicroseconds(current_call_start_time_, Clock::now());
  for (auto &perf_event_fd : perf_event_fds_) {
    perf_event_fd.disable();
  }
  for (size_t i = 0; i < perf_event_fds_.size(); ++i) {
    entry.perf_event_counts.push_back(perf_event_fds_[i].read());
  }
  getcpu(&entry.cpu, &entry.node);
  entries_.push_back(entry);
  recording_call_ = false;
}

// Returns a thread-specific singleton object.
ThreadInfo &ThreadInfoSingleton() {
  static thread_local ThreadInfo singleton;
  return singleton;
}

}  // namespace

#ifdef __GNUC__
#define IREE_HOOK_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define IREE_HOOK_EXPORT extern "C"
#endif

IREE_HOOK_EXPORT void iree_hal_executable_library_call_hook_begin(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t *library, iree_host_size_t ordinal) {
  ThreadInfoSingleton().beginCall(library->exports.names[ordinal]);
}

IREE_HOOK_EXPORT void iree_hal_executable_library_call_hook_end(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t *library, iree_host_size_t ordinal) {
  ThreadInfoSingleton().endCall();
}
