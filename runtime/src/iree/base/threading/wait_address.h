// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Zero-storage waits on atomic address changes.

#ifndef IREE_BASE_THREADING_WAIT_ADDRESS_H_
#define IREE_BASE_THREADING_WAIT_ADDRESS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/attributes.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/status.h"
#include "iree/base/threading/futex.h"
#include "iree/base/time.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

// Waits until the 32-bit atomic value at |address| no longer equals
// |expected_value| or |deadline_ns| is reached.
//
// Callers must always re-check their full predicate after this returns:
// wakeups may be spurious and timeout does not imply the predicate is still
// false.
static inline iree_status_code_t iree_wait_address_wait_int32(
    iree_atomic_int32_t* address, int32_t expected_value,
    iree_time_t deadline_ns) {
  (void)address;
  (void)expected_value;
  (void)deadline_ns;
  return IREE_STATUS_OK;
}

// Wakes all threads waiting on |address| through iree_wait_address_wait_int32.
static inline void iree_wait_address_wake_all(iree_atomic_int32_t* address) {
  (void)address;
}

#elif defined(IREE_RUNTIME_USE_FUTEX)

// Waits until the 32-bit atomic value at |address| no longer equals
// |expected_value| or |deadline_ns| is reached.
//
// Callers must always re-check their full predicate after this returns:
// wakeups may be spurious and timeout does not imply the predicate is still
// false.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE iree_status_code_t
iree_wait_address_wait_int32(iree_atomic_int32_t* address,
                             int32_t expected_value, iree_time_t deadline_ns) {
  return iree_futex_wait((void*)address, (uint32_t)expected_value, deadline_ns);
}

// Wakes all threads waiting on |address| through iree_wait_address_wait_int32.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE void iree_wait_address_wake_all(
    iree_atomic_int32_t* address) {
  iree_futex_wake((void*)address, IREE_ALL_WAITERS);
}

#else

// Waits until the 32-bit atomic value at |address| no longer equals
// |expected_value| or |deadline_ns| is reached.
//
// This is a low-level predicate wait for lock-free state machines that already
// have an atomic epoch/state word and cannot afford per-object initialization
// or teardown for a notification object. Callers must always re-check their
// full predicate after this returns: wakeups may be spurious, non-futex
// fallbacks may wake unrelated waiters sharing the same internal bucket, and
// timeout does not imply the predicate is still false.
//
// Returns:
//   IREE_STATUS_OK: value changed, a wake was observed, or a spurious wake
//     occurred before the deadline.
//   IREE_STATUS_DEADLINE_EXCEEDED: deadline reached before any wake/change was
//     observed.
//   IREE_STATUS_UNAVAILABLE: platform wait failed in a retryable way.
IREE_API_EXPORT iree_status_code_t
iree_wait_address_wait_int32(iree_atomic_int32_t* address,
                             int32_t expected_value, iree_time_t deadline_ns);

// Wakes all threads waiting on |address| through iree_wait_address_wait_int32.
//
// This may wake more threads than are waiting on the exact address when running
// on a platform fallback. Callers must encode correctness in the atomic
// predicate, not in exact wake cardinality.
IREE_API_EXPORT void iree_wait_address_wake_all(iree_atomic_int32_t* address);

#endif  // IREE_SYNCHRONIZATION_DISABLE_UNSAFE / IREE_RUNTIME_USE_FUTEX

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_THREADING_WAIT_ADDRESS_H_
