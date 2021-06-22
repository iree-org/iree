// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/time.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "iree/base/target_platform.h"

IREE_API_EXPORT iree_time_t iree_time_now(void) {
#if defined(IREE_TIME_NOW_FN)
  IREE_TIME_NOW_FN
#elif defined(IREE_PLATFORM_WINDOWS)
  // GetSystemTimePreciseAsFileTime requires Windows 8, add a fallback
  // (such as using std::chrono) if older support is needed.
  FILETIME system_time;
  GetSystemTimePreciseAsFileTime(&system_time);

  const int64_t kUnixEpochStartTicks = 116444736000000000i64;
  const int64_t kFtToMicroSec = 10;
  LARGE_INTEGER li;
  li.LowPart = system_time.dwLowDateTime;
  li.HighPart = system_time.dwHighDateTime;
  li.QuadPart -= kUnixEpochStartTicks;
  li.QuadPart /= kFtToMicroSec;
  return li.QuadPart;
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)
  struct timespec clock_time;
  clock_gettime(CLOCK_REALTIME, &clock_time);
  return clock_time.tv_nsec;
#else
#error "IREE system clock needs to be set up for your platform"
#endif  // IREE_PLATFORM_*
}

IREE_API_EXPORT iree_time_t
iree_relative_timeout_to_deadline_ns(iree_duration_t timeout_ns) {
  if (timeout_ns == IREE_DURATION_ZERO) {
    return IREE_TIME_INFINITE_PAST;
  } else if (timeout_ns == IREE_DURATION_INFINITE) {
    return IREE_TIME_INFINITE_FUTURE;
  }
  return iree_time_now() + timeout_ns;
}

IREE_API_EXPORT iree_duration_t
iree_absolute_deadline_to_timeout_ns(iree_time_t deadline_ns) {
  if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    return IREE_DURATION_ZERO;
  } else if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    return IREE_DURATION_INFINITE;
  } else {
    // We have either already passed the deadline (and can turn this into a
    // poll) or want to do nanos->millis. We round up so that a deadline of 1ns
    // results in 1ms as it should still wait, vs. if it was actually 0ns
    // indicating the user intended a poll.
    iree_time_t now_ns = iree_time_now();
    return deadline_ns < now_ns ? IREE_DURATION_ZERO : deadline_ns - now_ns;
  }
}
