// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Futex operations for kernel-side wait/wake via io_uring.
//
// These operations enable futex wait/wake to be performed as part of io_uring
// submission chains, eliminating userspace round-trips for synchronization
// patterns that combine I/O with futex operations.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   n/a     | 6.7+     | n/a  | n/a
//
// These operations require IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS.
// Check this capability before using—other backends will return
// IREE_STATUS_UNAVAILABLE.
//
// Use cases:
//   - LINK chains: POLL_ADD(eventfd) → FUTEX_WAKE(addr, count)
//     Allows semaphore timepoint callbacks to write to an eventfd (~200ns
//     under lock) while the heavy futex_wake happens in kernel space.
//   - Async wait: Submit FUTEX_WAIT and continue with other I/O, getting
//     notified via poll() when the futex is woken.
//
// Fallback for unsupported platforms:
//   On kernels without futex operation support (pre-6.7, or non-Linux),
//   use direct futex syscalls via iree/base/threading/futex.h.
//
// Relationship to iree/base/threading/futex.h:
//   - iree/base/threading/futex.h: Cross-platform 32-bit futex syscall wrappers
//     (iree_futex_wait, iree_futex_wake). Works on Linux, Windows, Emscripten.
//     Use this for direct synchronous futex operations.
//
//   - iree/async/operations/futex.h (this file): io_uring async operations with
//     size flags for 8/16/32/64-bit futex words. Only available on Linux 6.7+.
//     Use this for async I/O integration where the kernel should perform the
//     futex operation as part of an io_uring completion chain.
//
// Future direction:
//   For cross-platform notification semantics, use iree_async_notification_t
//   (when available) which provides a higher-level abstraction that uses
//   futexes where available and events elsewhere.

#ifndef IREE_ASYNC_OPERATIONS_FUTEX_H_
#define IREE_ASYNC_OPERATIONS_FUTEX_H_

#include "iree/async/operation.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Futex flags
//===----------------------------------------------------------------------===//

// Size flags for futex operations (matches kernel FUTEX2_SIZE_* flags).
// Exactly one size flag must be set.
enum iree_async_futex_size_bits_e {
  IREE_ASYNC_FUTEX_SIZE_U8 = 0x00,
  IREE_ASYNC_FUTEX_SIZE_U16 = 0x01,
  IREE_ASYNC_FUTEX_SIZE_U32 = 0x02,
  IREE_ASYNC_FUTEX_SIZE_U64 = 0x03,
};

enum iree_async_futex_flag_bits_e {
  IREE_ASYNC_FUTEX_FLAG_NONE = 0u,

  // Futex is process-private (not shared across processes).
  // Enables kernel fast-path optimizations.
  IREE_ASYNC_FUTEX_FLAG_PRIVATE = 1u << 7,
};
typedef uint32_t iree_async_futex_flags_t;

//===----------------------------------------------------------------------===//
// Futex wait
//===----------------------------------------------------------------------===//

// Waits on a futex word until it changes from an expected value.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   n/a     | 6.7+     | n/a  | n/a
//
// Completes when:
//   - Another thread/operation wakes this address (OK status).
//   - The value at futex_address != expected_value at submission (OK,
//     immediate completion—the wait "succeeded" because the value changed).
//   - The operation is cancelled (CANCELLED status).
//
// Threading model:
//   Callback fires on the poll thread when the wait completes.
//   The futex may be woken by any thread via FUTEX_WAKE syscall,
//   iree_futex_wake(), or another iree_async_futex_wake_operation_t.
//
// Memory ordering:
//   The futex operations provide acquire/release semantics matching the
//   Linux kernel futex implementation. Waiters see stores that happened
//   before the corresponding wake.
typedef struct iree_async_futex_wait_operation_t {
  iree_async_operation_t base;

  // Address of the futex word to wait on. Must remain valid and at a stable
  // address until the operation completes. The memory must be naturally aligned
  // for the futex size (e.g., 4-byte aligned for U32).
  void* futex_address;

  // Expected value. The wait only blocks if *futex_address == expected_value
  // at the time the kernel processes the operation. If the value has already
  // changed, the operation completes immediately with OK status.
  uint64_t expected_value;

  // Futex flags: size (IREE_ASYNC_FUTEX_SIZE_*) OR'd with optional flags
  // (IREE_ASYNC_FUTEX_FLAG_PRIVATE). Size must match the actual futex word
  // size and expected_value width.
  iree_async_futex_flags_t futex_flags;
} iree_async_futex_wait_operation_t;

//===----------------------------------------------------------------------===//
// Futex wake
//===----------------------------------------------------------------------===//

// Wakes threads waiting on a futex address.
//
// Completes immediately after the kernel processes the wake request.
// The completion reports how many waiters were actually woken.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   n/a     | 6.7+     | n/a  | n/a
//
// Threading model:
//   Callback fires on the poll thread after the wake is processed.
//   The wake itself happens synchronously in kernel space before the
//   CQE is posted, so woken waiters may begin running before the
//   wake operation's callback fires.
//
// Use in LINK chains:
//   FUTEX_WAKE is commonly used as the final step in a linked sequence:
//     RECV → PROCESS → FUTEX_WAKE
//   This wakes waiting consumer threads without returning to userspace
//   between the I/O completion and the wake.
typedef struct iree_async_futex_wake_operation_t {
  iree_async_operation_t base;

  // Address of the futex word to wake waiters on. Must match the address
  // used by waiting threads/operations.
  void* futex_address;

  // Maximum number of waiters to wake. Common values:
  //   1: Wake a single waiter (e.g., mutex unlock)
  //   INT32_MAX: Wake all waiters (e.g., broadcast/barrier)
  int32_t wake_count;

  // Futex flags: size (IREE_ASYNC_FUTEX_SIZE_*) OR'd with optional flags.
  // Must match the flags used by waiters.
  iree_async_futex_flags_t futex_flags;

  // Result: actual number of waiters woken. Populated on completion.
  // May be less than wake_count if fewer threads were waiting.
  int32_t woken_count;
} iree_async_futex_wake_operation_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_OPERATIONS_FUTEX_H_
