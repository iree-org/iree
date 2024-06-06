// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/time.h"

#include <time.h>

#include "iree/base/config.h"
#include "iree/base/target_platform.h"

int64_t iree_platform_time_now(void) {
#if defined(IREE_TIME_NOW_FN)
  IREE_TIME_NOW_FN
#elif defined(IREE_PLATFORM_WINDOWS)
  // GetSystemTimePreciseAsFileTime requires Windows 8, add a fallback
  // (such as using std::chrono) if older support is needed.
  FILETIME system_time;
  GetSystemTimePreciseAsFileTime(&system_time);
  const int64_t kUnixEpochStartTicks = 116444736000000000i64;
  const int64_t kFtToNanoSec = 100;
  LARGE_INTEGER li;
  li.LowPart = system_time.dwLowDateTime;
  li.HighPart = system_time.dwHighDateTime;
  li.QuadPart -= kUnixEpochStartTicks;
  li.QuadPart *= kFtToNanoSec;
  return li.QuadPart;
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_EMSCRIPTEN)
  struct timespec clock_time;
  clock_gettime(CLOCK_REALTIME, &clock_time);
  return clock_time.tv_sec * 1000000000ull + clock_time.tv_nsec;
#else
#error "IREE system clock needs to be set up for your platform"
#endif  // IREE_PLATFORM_*
}
