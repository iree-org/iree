// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_THREADING_NOTIFICATION_H_
#define IREE_BASE_THREADING_NOTIFICATION_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/futex.h"

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !defined(IREE_PLATFORM_WINDOWS)
#include <pthread.h>
#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !IREE_PLATFORM_WINDOWS

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// iree_notification_t
//==============================================================================

// A lightweight wait-free cross-thread notification mechanism.
// Classically called an 'event counter', these replace the use of condvars in
// lock-free code where you wouldn't want to guard a lock-free data structure
// with a lock.
//
// See:
// http://www.1024cores.net/home/lock-free-algorithms/eventcounts
// https://software.intel.com/en-us/forums/intel-threading-building-blocks/topic/299245
// https://github.com/r10a/Event-Counts
// https://github.com/facebook/folly/blob/main/folly/experimental/EventCount.h
// https://github.com/concurrencykit/ck/blob/master/include/ck_ec.h
typedef struct iree_notification_t {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  // Nothing required. Unused field to make compilers happy.
  int reserved;
#elif !defined(IREE_RUNTIME_USE_FUTEX)
  // No futex on darwin/when using TSAN, so use mutex/condvar instead.
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint32_t epoch;
  uint32_t waiters;
#else
  iree_atomic_int64_t value;
#endif  // IREE_PLATFORM_*
} iree_notification_t;

// Initializes a notification to no waiters and an initial epoch of 0.
IREE_API_EXPORT void iree_notification_initialize(
    iree_notification_t* out_notification);

// Deinitializes |notification| (after a prior call to
// iree_notification_initialize). No threads may be waiting on the notification.
IREE_API_EXPORT void iree_notification_deinitialize(
    iree_notification_t* notification);

// Notifies up to |count| waiters of a change. Each waiter will wake and can
// check to see if they need to do any additional work.
// To notify all potential waiters pass IREE_ALL_WAITERS.
//
// Acts as (at least) a memory_order_release operation on the
// notification object. See the comment on iree_notification_commit_wait, which
// is the memory_order_acquire operation that is meant to pair with that.
IREE_API_EXPORT void iree_notification_post(iree_notification_t* notification,
                                            int32_t count);

typedef uint32_t iree_wait_token_t;  // opaque

// Prepares for a wait operation, returning a token that must be passed to
// iree_notification_commit_wait to perform the actual wait.
//
// Acts as a memory_order_acq_rel read-modify-write operation on the
// notification object. See the comment on iree_notification_commit_wait for a
// general explanation of acquire/release semantics in this context.
IREE_API_EXPORT iree_wait_token_t
iree_notification_prepare_wait(iree_notification_t* notification);

// Commits a pending wait operation when the caller has ensured it must wait.
// Waiting will continue until a notification has been posted or |deadline_ns|
// is reached. Returns false if the deadline is reached before a notification is
// posted.
//
// If |spin_ns| is not IREE_DURATION_ZERO the wait _may_ spin for at least the
// specified duration before entering the system wait API.
//
// Acts as (at least) a memory_order_acquire operation on the notification
// object. This is meant to be paired with iree_notification_post, which is a
// memory_order_release operation. This means the following guarantee:
// When iree_notification_commit_wait returns on this thread T1 from having
// waited on another thread T2 calling iree_notification_post, all memory read
// and write operations performed on thread T2 prior to calling
// iree_notification_post are guaranteed to "happen-before" any subsequent
// memory read or write on thread T1.
IREE_API_EXPORT bool iree_notification_commit_wait(
    iree_notification_t* notification, iree_wait_token_t wait_token,
    iree_duration_t spin_ns, iree_time_t deadline_ns);

// Cancels a pending wait operation without blocking.
//
// Acts as (at least) a memory_order_relaxed barrier:
//   Relaxed operation: there are no synchronization or ordering constraints
//   imposed on other reads or writes, only this operation's atomicity is
//   guaranteed.
IREE_API_EXPORT void iree_notification_cancel_wait(
    iree_notification_t* notification);

// Returns true if the condition is true.
// |arg| is the |condition_arg| passed to the await function.
// Implementations must ensure they are coherent with their state values.
typedef bool (*iree_condition_fn_t)(void* arg);

// Blocks and waits until |condition_fn| returns true. Other threads must modify
// state checked by the |condition_fn| and post the notification.
// Returns true if the condition is true before |timeout| is reached. If the
// timeout is infinite then the return will always be true.
//
// Example:
//  thread 1:
//   bool check_flag_pred(void* arg) {
//     return iree_atomic_int32_load((iree_atomic_int32_t*)arg,
//                                   iree_memory_order_acquire) == 1;
//   }
//   iree_atomic_int32_t* flag = ...;
//   iree_notification_await(&notification, check_flag_pred, flag);
//  thread 2:
//   iree_atomic_int32_store(flag, 1, iree_memory_order_release);
//   iree_notification_post(&notification, IREE_ALL_WAITERS);
IREE_API_EXPORT bool iree_notification_await(iree_notification_t* notification,
                                             iree_condition_fn_t condition_fn,
                                             void* condition_arg,
                                             iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_NOTIFICATION_H_
