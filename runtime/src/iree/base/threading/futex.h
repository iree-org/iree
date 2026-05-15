// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-platform futex primitives for low-level synchronization.
//
// This provides a 32-bit futex API that works across Linux, Windows, and
// Wasm. These are the building blocks for higher-level synchronization
// primitives like iree_slim_mutex_t and iree_notification_t.
//
// The API is 32-bit only because that's the lowest common denominator:
//   - Linux futex syscall: 32-bit
//   - Windows WaitOnAddress: variable size, but we use 32-bit
//   - Wasm: 32-bit (memory.atomic.wait32 / memory.atomic.notify)
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

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX) || \
    defined(IREE_PLATFORM_WASM) || defined(IREE_PLATFORM_WINDOWS)
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

#if defined(IREE_PLATFORM_WASM)
// Wasm atomic wait/notify are compiler builtins — no headers needed.
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
// Shared (cross-process) futex API
//===----------------------------------------------------------------------===//

// Cross-process variants that operate on physical page addresses instead of
// virtual addresses. On Linux, this omits FUTEX_PRIVATE_FLAG, causing the
// kernel to hash by physical page rather than {mm, virtual address}. This
// allows futex operations to work across processes sharing the same physical
// page (e.g., via mmap MAP_SHARED or shm_open).
//
// On Windows, WaitOnAddress/WakeByAddress already hash by physical page, so
// these are identical to the private variants.
//
// On Wasm, cross-process shared memory is not meaningful, so these are
// identical to the private variants.
//
// Performance: ~20ns slower per operation on Linux due to the kernel page table
// walk. Use the private variants (iree_futex_wait/wake) when cross-process
// semantics are not needed.
static inline iree_status_code_t iree_futex_wait_shared(
    void* address, uint32_t expected_value, iree_time_t deadline_ns);
static inline void iree_futex_wake_shared(void* address, int32_t count);

//===----------------------------------------------------------------------===//
// Platform implementations
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WASM)

// Wasm provides memory.atomic.wait32 and memory.atomic.notify instructions
// via compiler builtins. These map directly to Atomics.wait/Atomics.notify
// in the JS host. memory.atomic.wait32 is only allowed on non-main threads
// (Web Workers); on the main thread it traps. This is fine because IREE's
// futex-based synchronization primitives are only used from worker threads
// in cooperative mode (the main thread uses the proactor poll loop instead).
//
// The timeout is specified in nanoseconds (-1 for infinite).

static inline iree_status_code_t iree_futex_wait(void* address,
                                                 uint32_t expected_value,
                                                 iree_time_t deadline_ns) {
  int64_t timeout_ns = -1;
  if (deadline_ns != IREE_TIME_INFINITE_FUTURE) {
    iree_time_t now_ns = iree_time_now();
    timeout_ns = deadline_ns > now_ns ? (int64_t)(deadline_ns - now_ns) : 0;
  }
  // Returns: 0 = "ok" (woken), 1 = "not-equal", 2 = "timed-out".
  int rc = __builtin_wasm_memory_atomic_wait32(
      (int32_t*)address, (int32_t)expected_value, timeout_ns);
  switch (rc) {
    case 0:
      return IREE_STATUS_OK;
    case 1:
      return IREE_STATUS_UNAVAILABLE;
    case 2:
      return IREE_STATUS_DEADLINE_EXCEEDED;
    default:
      return IREE_STATUS_OK;
  }
}

static inline void iree_futex_wake(void* address, int32_t count) {
  __builtin_wasm_memory_atomic_notify((int32_t*)address, (uint32_t)count);
}

// Wasm has no cross-process shared memory — alias to private variants.
static inline iree_status_code_t iree_futex_wait_shared(
    void* address, uint32_t expected_value, iree_time_t deadline_ns) {
  return iree_futex_wait(address, expected_value, deadline_ns);
}
static inline void iree_futex_wake_shared(void* address, int32_t count) {
  iree_futex_wake(address, count);
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

// WaitOnAddress/WakeByAddress already hash by physical page on Windows.
static inline iree_status_code_t iree_futex_wait_shared(
    void* address, uint32_t expected_value, iree_time_t deadline_ns) {
  return iree_futex_wait(address, expected_value, deadline_ns);
}
static inline void iree_futex_wake_shared(void* address, int32_t count) {
  iree_futex_wake(address, count);
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

// Cross-process variants: omit FUTEX_PRIVATE_FLAG so the kernel hashes by
// physical page instead of {mm, virtual address}. This allows futex operations
// across processes sharing the same physical page via mmap MAP_SHARED.
static inline iree_status_code_t iree_futex_wait_shared(
    void* address, uint32_t expected_value, iree_time_t deadline_ns) {
  uint32_t timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);
  struct timespec timeout = {
      .tv_sec = timeout_ms / 1000,
      .tv_nsec = (timeout_ms % 1000) * 1000000,
  };
  int rc = syscall(SYS_futex, address, FUTEX_WAIT, expected_value,
                   timeout_ms == IREE_INFINITE_TIMEOUT_MS ? NULL : &timeout,
                   NULL, 0);
  if (IREE_LIKELY(rc == 0) || errno == EAGAIN || errno == EINTR) {
    return IREE_STATUS_OK;
  } else if (errno == ETIMEDOUT) {
    return IREE_STATUS_DEADLINE_EXCEEDED;
  }
  return IREE_STATUS_UNAVAILABLE;
}

static inline void iree_futex_wake_shared(void* address, int32_t count) {
  syscall(SYS_futex, address, FUTEX_WAKE, count, NULL, NULL, 0);
}

#endif  // IREE_PLATFORM_*

#endif  // IREE_RUNTIME_USE_FUTEX

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_FUTEX_H_
