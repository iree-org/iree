// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_HAL_EXECUTABLE_LIBRARY_CALL_HOOKS_PERF_EVENT_LINUX_H_
#define EXPERIMENTAL_HAL_EXECUTABLE_LIBRARY_CALL_HOOKS_PERF_EVENT_LINUX_H_

#include <cstdint>
#include <cstdio>
#include <vector>

// Describes a perf event type. Matches the corresponding data structure in
// the linux perf source code.
struct PerfEventType {
  const char *name;
  uint32_t type;
  uint64_t config;
  const char *target;
  const char *description;
};

// A perf-event file-descriptor for querying a specific event count.
class PerfEventFd {
 public:
  PerfEventFd(PerfEventType perf_event_type);
  ~PerfEventFd();

  // Resets the event counter.
  void reset();
  // Enables the event counter.
  void enable();
  // Disables the event counter.
  void disable();
  // Queries the current value of the event counter.
  int64_t read() const;

 private:
  int fd_ = 0;
};

// Parses a string as a comma-separated list of event types.
std::vector<PerfEventType> parsePerfEventTypes(const char *types_str);

// Prints all event types and their descriptions.
void printAllEventTypesAndDescriptions(FILE *file);

#endif  // EXPERIMENTAL_HAL_EXECUTABLE_LIBRARY_CALL_HOOKS_PERF_EVENT_LINUX_H_
