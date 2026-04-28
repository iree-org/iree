// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/time.h"

#include "iree/base/config.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WASM) && !defined(IREE_PLATFORM_WASI)

// Freestanding wasm (browser/Node.js) uses a direct JS import that returns
// nanoseconds as int64_t. This avoids the POSIX clock_gettime abstraction
// (struct timespec, clock_id dispatch) and maps directly to performance.now()
// in the JS host. Resolution is microsecond-level when cross-origin isolated,
// millisecond otherwise.
__attribute__((import_module("iree_syscall"),
               import_name("clock_now_ns"))) extern int64_t
iree_wasm_clock_now_ns(void);

int64_t iree_platform_time_now(void) { return iree_wasm_clock_now_ns(); }

#else

#include <time.h>

int64_t iree_platform_time_now(void) {
#if defined(IREE_TIME_NOW_FN)
  IREE_TIME_NOW_FN
#elif defined(IREE_PLATFORM_WINDOWS)
  // QueryPerformanceCounter provides a high-resolution monotonic timer.
  // QPC frequency is fixed at boot and doesn't change.
  static LARGE_INTEGER frequency = {0};
  if (frequency.QuadPart == 0) {
    QueryPerformanceFrequency(&frequency);
  }
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  // Convert to nanoseconds: counter * 1e9 / frequency.
  // Split into seconds and remainder to avoid overflow without 128-bit math.
  int64_t seconds = counter.QuadPart / frequency.QuadPart;
  int64_t remainder = counter.QuadPart % frequency.QuadPart;
  return seconds * 1000000000ll +
         (remainder * 1000000000ll) / frequency.QuadPart;
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_WASI)
  // CLOCK_MONOTONIC is used across all POSIX-like platforms for consistency.
  // WASI libc provides clock_gettime backed by __wasi_clock_time_get.
  // CLOCK_MONOTONIC_RAW exists on Linux but can't be used with
  // pthread_cond_timedwait (only CLOCK_MONOTONIC and CLOCK_REALTIME are
  // supported). The difference is minimal: RAW excludes NTP rate adjustments
  // which are typically <0.05%.
  struct timespec clock_time;
  clock_gettime(CLOCK_MONOTONIC, &clock_time);
  return clock_time.tv_sec * 1000000000ull + clock_time.tv_nsec;
#else
#error "IREE monotonic clock needs to be set up for your platform"
#endif  // IREE_PLATFORM_*
}

#endif  // IREE_PLATFORM_WASM && !IREE_PLATFORM_WASI
