// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_SEMAPHORE_BASE_H_
#define IREE_HAL_UTILS_SEMAPHORE_BASE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Callback handler for semaphore timepoints.
// Handlers receive the semaphore, the current value, and the status code.
//
// The |value| is only valid if |status_code| is IREE_STATUS_OK.
// In error cases handlers can query the status of the |semaphore| to receive
// the full status if desired.
//
// Handlers run outside the semaphore lock but under the timepoint list lock
// and may re-entrantly use the semaphore to query but not manage timepoints.
typedef iree_status_t(IREE_API_PTR* iree_hal_semaphore_callback_fn_t)(
    void* user_data, iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_status_code_t status_code);

typedef struct iree_hal_semaphore_callback_t {
  // Callback function pointer.
  iree_hal_semaphore_callback_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_hal_semaphore_callback_t;

// Storage for a semaphore timepoint.
// Each semaphore manages a list of active timepoints and issues their specified
// callback when the semaphore is signaled to or beyond a given value.
typedef struct iree_hal_semaphore_timepoint_t {
  // Intrusive doubly-linked list next entry pointer.
  // Guarded by the semaphore mutex.
  struct iree_hal_semaphore_timepoint_t* next;
  // Intrusive doubly-linked list previous entry pointer.
  // Guarded by the semaphore mutex.
  struct iree_hal_semaphore_timepoint_t* prev;

  // Retained semaphore; this ensures the semaphore remains valid for the
  // lifetime of the timepoint. The semaphore must be released by the underlying
  // implementation or by the user with iree_hal_semaphore_release_timepoint.
  struct iree_hal_semaphore_t* semaphore;

  // Target value the semaphore must reach or exceed to trigger the timepoint.
  uint64_t minimum_value;

  // Absolute deadline after which the timepoint will expire if the semaphore
  // has not reached the target value.
  iree_time_t deadline_ns;

  // Callback to issue when the timepoint is reached, the deadline is exceeded,
  // or the semaphore fails.
  iree_hal_semaphore_callback_t callback;
} iree_hal_semaphore_timepoint_t;

// A doubly-linked FIFO list of timepoints.
// The order of the timepoints does *not* match increasing payload values but
// instead the order they were added to the list.
//
// Note that the timepoints are not owned by the list - this just nicely
// stitches together timepoints for easier management.
typedef struct iree_hal_semaphore_timepoint_list_t {
  iree_hal_semaphore_timepoint_t* head;
  iree_hal_semaphore_timepoint_t* tail;
} iree_hal_semaphore_timepoint_list_t;

// Abstract base implementation of semaphores that perform timepoint tracking.
//
// Device implementations can acquire timepoints that provide low-latency
// directed notification of when a semaphore timeline reaches a certain point
// (or fails). The storage for the timepoints is managed by the requester and
// can be allocation-free making the timepoint operations safe to perform from
// driver threads/callbacks.
//
// Semaphore implementations need to notify the tracking semaphore of signal and
// failure events using the iree_hal_semaphore_notify method. Any satisfied
// timepoints will have their callback made immediately from the notifying
// thread.
struct iree_hal_semaphore_t {
  iree_hal_resource_t resource;  // must be at 0

  // Non-recursive mutex guarding access to the timepoint list.
  iree_slim_mutex_t timepoint_mutex;

  // Timepoint list in insertion order.
  // There are probably better orderings we could use here that allow us to
  // walk the entire list less frequently, though target payload value is not
  // enough as deadlines still require the scan. We could sort by non-infinite
  // deadlines first and then infinite ones last but given the common timepoint
  // counts (0..1) it's not worth the complexity today.
  iree_hal_semaphore_timepoint_list_t timepoint_list
      IREE_GUARDED_BY(timepoint_mutex);
};

// Initializes the base |out_semaphore| resource.
IREE_API_EXPORT void iree_hal_semaphore_initialize(
    const iree_hal_semaphore_vtable_t* vtable,
    iree_hal_semaphore_t* out_semaphore);

// Deinitializes the |semaphore|.
// Because timepoints retain their semaphore the timepoint list is known empty.
IREE_API_EXPORT void iree_hal_semaphore_deinitialize(
    iree_hal_semaphore_t* semaphore);

// Acquires a timepoint on the semaphore timeline that issues the given
// |callback| when the semaphore payload reaches or exceeds |minimum_value|. The
// callback may be made from a random external thread and must avoid recursive
// locks (such as managing timepoints).
//
// The caller provides storage in |out_timepoint| and it must remain valid until
// either the callback is made or the timepoint is cancelled via
// iree_hal_semaphore_cancel_timepoint.
//
// If the timepoint has already been reached the callback _may_ be issued prior
// to the function returning. If the |timeout| has already been reached then the
// callback _may_ be issued with IREE_STATUS_DEADLINE_EXCEEDED.
//
// NOTE: this behavior is due to racy multi-threaded behavior and not a
// guarantee of the API: if there's only ever a single thread then acquiring a
// timepoint that has been satisfied will *NOT* callback here.
// iree_hal_semaphore_poll can be used to force a flush of any resolved
// timepoints on-demand.
//
// Must not be called from a timepoint callback.
IREE_API_EXPORT void iree_hal_semaphore_acquire_timepoint(
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value,
    iree_timeout_t timeout, iree_hal_semaphore_callback_t callback,
    iree_hal_semaphore_timepoint_t* out_timepoint);

// Cancels a |timepoint| and prevents any future callbacks.
// The timepoint is only considered cancelled once execution returns
// to the caller; due to races it's possible for a callback to be made with
// a different status code while releasing.
//
// Only the owner of the timepoint (whatever is listening for the callback)
// should use this as otherwise the program may become desynchronized.
//
// Must not be called from a timepoint callback.
IREE_API_EXPORT void iree_hal_semaphore_cancel_timepoint(
    iree_hal_semaphore_t* semaphore, iree_hal_semaphore_timepoint_t* timepoint);

// Used by implementations to notify when a new timepoint is reached.
// Implementations must call this when they observe changes.
// Calling this incorrectly will result in undefined behavior.
//
// Must not be called from a timepoint callback.
// Must not be called with a semaphore lock held as notifications may
// re-entrantly use the semaphore.
IREE_API_EXPORT void iree_hal_semaphore_notify(
    iree_hal_semaphore_t* semaphore, uint64_t new_value,
    iree_status_code_t new_status_code);

// Polls timepoints and issues callbacks for those already resolved.
// This polling is performed internally on user calls such as signal and wait
// but can be made more frequently to reduce latency in cases where users are
// not making calls frequently enough. Implementations should always prefer to
// notify directly with iree_hal_semaphore_notify to avoid additional
// synchronization overheads.
//
// Must not be called from a timepoint callback.
IREE_API_EXPORT void iree_hal_semaphore_poll(iree_hal_semaphore_t* semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_SEMAPHORE_BASE_H_
