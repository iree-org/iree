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
#include "iree/base/tracing.h"

IREE_API_EXPORT iree_time_t iree_time_now(void) {
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
    iree_time_t now_ns = iree_time_now();
    return deadline_ns < now_ns ? IREE_DURATION_ZERO : deadline_ns - now_ns;
  }
}

IREE_API_EXPORT uint32_t
iree_absolute_deadline_to_timeout_ms(iree_time_t deadline_ns) {
  if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    return IREE_DURATION_ZERO;
  } else if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    return UINT32_MAX;
  } else {
    // We have either already passed the deadline (and can turn this into a
    // poll) or want to do nanos->millis. We round up so that a deadline of 1ns
    // results in 1ms as it should still wait, vs. if it was actually 0ns
    // indicating the user intended a poll.
    iree_time_t now_ns = iree_time_now();
    return deadline_ns < now_ns
               ? IREE_DURATION_ZERO
               : (deadline_ns - now_ns + 1000000 - 1) / 1000000ull;
  }
}

#if defined(IREE_WAIT_UNTIL_FN)

// Define IREE_WAIT_UNTIL_FN to call out to a user-configured function.
static bool iree_wait_until_impl(iree_time_t deadline_ns) {
  return IREE_WAIT_UNTIL_FN(deadline_ns);
}

#elif defined(IREE_PLATFORM_WINDOWS)

// No good sleep APIs on Windows; we need to accumulate low-precision relative
// waits to reach the absolute time. Lots of slop here, but we primarily use
// nanoseconds as a uniform time API and don't guarantee that precision. Note
// that we try to round up to ensure we wait until at least the requested time.
static bool iree_wait_until_impl(iree_time_t deadline_ns) {
  iree_time_t now_ns = iree_time_now();
  while (now_ns < deadline_ns) {
    iree_time_t delta_ns = deadline_ns - now_ns;
    uint32_t delta_ms = (uint32_t)((delta_ns + 1000000 - 1) / 1000000ull);
    if (delta_ms == 0) {
      // Sleep(0) doesn't actually sleep and instead acts as a yield; instead of
      // potentially spilling in a tight loop when we get down near the end of
      // the wait we bail a bit early. We don't guarantee the precision of the
      // waits so this is fine.
      break;
    }
    Sleep(delta_ms);
    now_ns = iree_time_now();
  }
  return true;
}

#elif (_POSIX_C_SOURCE >= 200112L) && defined(TIMER_ABSTIME)

// This is widely available on *nix-like systems (linux/bsd/etc) and in
// most libc implementations (glibc/musl/etc). It's the best as we get to
// tell the system the exact time we want to sleep until.
//
// https://man7.org/linux/man-pages/man2/clock_nanosleep.2.html
//
// NOTE: we could save a syscall in many cases if we returned the time upon wake
// from the API.
static bool iree_wait_until_impl(iree_time_t deadline_ns) {
  struct timespec ts = {
      .tv_sec = (time_t)(deadline_ns / 1000000000ull),
      .tv_nsec = (long)(deadline_ns % 1000000000ull),
  };
  int ret = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts, NULL);
  return ret == 0;
}

#elif (_POSIX_C_SOURCE >= 199309L) || defined(IREE_PLATFORM_APPLE)

// Apple doesn't have clock_nanosleep. We could use the Mach APIs on darwin to
// do this but they require initialization and potential updates during
// execution as clock frequencies change. Instead we use the relative nanosleep
// and accumulate until the deadline, which is a good fallback for some other
// platforms as well.
//
// https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/nanosleep.2.html
static bool iree_wait_until_impl(iree_time_t deadline_ns) {
  iree_time_t now_ns = iree_time_now();
  while (now_ns < deadline_ns) {
    iree_time_t delta_ns = deadline_ns - now_ns;
    struct timespec abs_ts = {
        .tv_sec = (time_t)(delta_ns / 1000000000ull),
        .tv_nsec = (long)(delta_ns % 1000000000ull),
    };
    int ret = nanosleep(&abs_ts, NULL);
    if (ret != 0) return false;
    now_ns = iree_time_now();
  }
  return true;
}

#else

// No waiting available; just pretend like we did. This will cause programs
// using timers to run as fast as possible but without having a way to delay
// time there's not much else they could do.
static bool iree_wait_until_impl(iree_time_t deadline_ns) { return true; }

#endif  // (platforms)

bool iree_wait_until(iree_time_t deadline_ns) {
  // Can't wait forever - or for the past.
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) return false;
  if (deadline_ns == IREE_TIME_INFINITE_PAST) return true;

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(
      z0, (uint64_t)iree_absolute_deadline_to_timeout_ns(deadline_ns));

  // NOTE: we want to use sleep APIs with absolute times as that makes retrying
  // on spurious wakes easier; if we using relative timeouts we need to ensure
  // we don't drift.
  bool did_wait = iree_wait_until_impl(deadline_ns);

  IREE_TRACE_ZONE_END(z0);
  return did_wait;
}
