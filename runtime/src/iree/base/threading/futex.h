// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-platform futex primitives for low-level synchronization.
//
// This provides a 32-bit futex API that works across Linux, Windows, and
// Emscripten. These are the building blocks for higher-level synchronization
// primitives like iree_slim_mutex_t and iree_notification_t.
//
// The API is 32-bit only because that's the lowest common denominator:
//   - Linux futex syscall: 32-bit
//   - Windows WaitOnAddress: variable size, but we use 32-bit
//   - Emscripten: 32-bit (JavaScript Atomics.wait)
//
// For 64-bit futex support (Linux kernel 6.7+ futex2), see
// iree/async/operations/futex.h which provides async io_uring operations
// with size flags.
//
// Note: Futex operations are disabled under ThreadSanitizer because TSan
// doesn't instrument futex syscalls. When IREE_SANITIZER_THREAD is defined,
// IREE_PLATFORM_HAS_FUTEX will be defined but IREE_RUNTIME_USE_FUTEX will not,
// causing higher-level primitives to fall back to pthread-based
// implementations.

#ifndef IREE_BASE_THREADING_FUTEX_H_
#define IREE_BASE_THREADING_FUTEX_H_

#include <stdint.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/base/time.h"

//===----------------------------------------------------------------------===//
// Platform detection
//===----------------------------------------------------------------------===//

// Allow users to fully disable all synchronization for systems that are known
// to never need it. This removes our dependency on pthreads.
#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_EMSCRIPTEN) || \
    defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_WINDOWS)
#define IREE_PLATFORM_HAS_FUTEX 1
#endif  // IREE_PLATFORM_*

#if defined(IREE_PLATFORM_HAS_FUTEX) && !defined(IREE_SANITIZER_THREAD)
// TODO: If we have TSan instrumentation for futexes we can enabled them when
// compiling with TSan.
#define IREE_RUNTIME_USE_FUTEX 1
#endif  // IREE_PLATFORM_HAS_FUTEX

#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

//===----------------------------------------------------------------------===//
// Platform headers
//===----------------------------------------------------------------------===//

#if defined(IREE_RUNTIME_USE_FUTEX)

#if defined(IREE_PLATFORM_EMSCRIPTEN)
#include <emscripten/threading.h>
#include <errno.h>
#elif defined(IREE_PLATFORM_WINDOWS)
// Windows headers included via iree/base/target_platform.h
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
#include <errno.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(IREE_ARCH_RISCV_32) && defined(__NR_futex_time64) && \
    !defined(__NR_futex)
// RV32 uses 64-bit times by default (unlike other 32-bit archs).
#define __NR_futex __NR_futex_time64
#endif  // IREE_ARCH_RISCV_32

// Oh Android...
#ifndef SYS_futex
#define SYS_futex __NR_futex
#endif  // !SYS_futex
#ifndef FUTEX_PRIVATE_FLAG
#define FUTEX_PRIVATE_FLAG 128
#endif  // !FUTEX_PRIVATE_FLAG

#endif  // IREE_PLATFORM_*

#endif  // IREE_RUNTIME_USE_FUTEX

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Sentinel value to wake all waiters.
#define IREE_ALL_WAITERS INT32_MAX

// Infinite timeout value in milliseconds for internal use.
#define IREE_INFINITE_TIMEOUT_MS UINT32_MAX

//===----------------------------------------------------------------------===//
// Futex API
//===----------------------------------------------------------------------===//

#if defined(IREE_RUNTIME_USE_FUTEX)

// Waits in the OS for the value at the specified |address| to change.
// If the contents of |address| do not match |expected_value| the wait will
// fail and return IREE_STATUS_UNAVAILABLE and should be retried.
//
// |deadline_ns| can be either IREE_TIME_INFINITE_FUTURE to wait forever or an
// absolute time to wait until prior to returning early with
// IREE_STATUS_DEADLINE_EXCEEDED.
//
// Returns:
//   IREE_STATUS_OK: Woken by another thread or value changed.
//   IREE_STATUS_DEADLINE_EXCEEDED: Timeout reached before wake.
//   IREE_STATUS_UNAVAILABLE: Value at address != expected_value (retry needed).
static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns);

// Wakes at most |count| threads waiting for the |address| to change.
// Use IREE_ALL_WAITERS to wake all waiters. Which waiters are woken is
// undefined and it is not guaranteed that higher priority waiters will be woken
// over lower priority waiters.
static inline void iree_futex_wake(void* address, int32_t count);

//===----------------------------------------------------------------------===//
// Platform implementations
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_EMSCRIPTEN)

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  uint32_t timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);
  int rc = emscripten_futex_wait(address, expected_value, (double)timeout_ms);
  switch (rc) {
    default:
      return IREE_STATUS_OK;
    case -ETIMEDOUT:
      return IREE_STATUS_DEADLINE_EXCEEDED;
    case -EWOULDBLOCK:
      return IREE_STATUS_UNAVAILABLE;
  }
}

static inline void iree_futex_wake(void* address, int32_t count) {
  emscripten_futex_wake(address, count);
}

#elif defined(IREE_PLATFORM_WINDOWS)

#pragma comment(lib, "Synchronization.lib")

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  uint32_t timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);
  if (IREE_LIKELY(WaitOnAddress(address, &expected_value,
                                sizeof(expected_value), timeout_ms) == TRUE)) {
    return IREE_STATUS_OK;
  }
  if (GetLastError() == ERROR_TIMEOUT) {
    return IREE_STATUS_DEADLINE_EXCEEDED;
  }
  return IREE_STATUS_UNAVAILABLE;
}

static inline void iree_futex_wake(void* address, int32_t count) {
  if (count == INT32_MAX) {
    WakeByAddressAll(address);
    return;
  }
  for (; count > 0; --count) {
    WakeByAddressSingle(address);
  }
}

#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  uint32_t timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);
  struct timespec timeout = {
      .tv_sec = timeout_ms / 1000,
      .tv_nsec = (timeout_ms % 1000) * 1000000,
  };
  int rc = syscall(
      SYS_futex, address, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, expected_value,
      timeout_ms == IREE_INFINITE_TIMEOUT_MS ? NULL : &timeout, NULL, 0);
  if (IREE_LIKELY(rc == 0) || errno == EAGAIN || errno == EINTR) {
    return IREE_STATUS_OK;
  } else if (errno == ETIMEDOUT) {
    return IREE_STATUS_DEADLINE_EXCEEDED;
  }
  return IREE_STATUS_UNAVAILABLE;
}

static inline void iree_futex_wake(void* address, int32_t count) {
  syscall(SYS_futex, address, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, count, NULL,
          NULL, 0);
}

#endif  // IREE_PLATFORM_*

#endif  // IREE_RUNTIME_USE_FUTEX

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_FUTEX_H_
