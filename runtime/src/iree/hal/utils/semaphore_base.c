// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/semaphore_base.h"

#include <stddef.h>

#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// Timepoint utilities
//===----------------------------------------------------------------------===//

// Returns true if the timepoint list is empty.
static inline bool iree_hal_semaphore_timepoint_list_is_empty(
    const iree_hal_semaphore_timepoint_list_t* list) {
  return list->head == NULL;
}

// Pushes |timepoint| on to the end of the given timepoint |list|.
static void iree_hal_semaphore_timepoint_list_push_back(
    iree_hal_semaphore_timepoint_list_t* list,
    iree_hal_semaphore_timepoint_t* timepoint) {
  if (list->tail) {
    list->tail->next = timepoint;
  } else {
    list->head = timepoint;
  }
  timepoint->next = NULL;
  timepoint->prev = list->tail;
  list->tail = timepoint;
}

// Erases |timepoint| from |list|.
static void iree_hal_semaphore_timepoint_list_erase(
    iree_hal_semaphore_timepoint_list_t* list,
    iree_hal_semaphore_timepoint_t* timepoint) {
  iree_hal_semaphore_timepoint_t* next = timepoint->next;
  iree_hal_semaphore_timepoint_t* prev = timepoint->prev;
  if (prev) {
    prev->next = next;
    timepoint->prev = NULL;
  } else {
    list->head = next;
  }
  if (next) {
    next->prev = prev;
    timepoint->next = NULL;
  } else {
    list->tail = prev;
  }
}

// Takes all timepoints from |available_list| and moves them into |ready_list|.
static void iree_hal_semaphore_timepoint_list_take_all(
    iree_hal_semaphore_timepoint_list_t* available_list,
    iree_hal_semaphore_timepoint_list_t* ready_list) {
  IREE_ASSERT(available_list != ready_list);
  ready_list->head = available_list->head;
  ready_list->tail = available_list->tail;
  available_list->head = NULL;
  available_list->tail = NULL;
}

// Issues the callback for the given |timepoint| and resets it.
static void iree_hal_semaphore_issue_timepoint_callback(
    iree_hal_semaphore_t* semaphore, uint64_t new_value,
    iree_status_code_t new_status_code,
    iree_hal_semaphore_timepoint_t* timepoint) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Clean up the timepoint.
  // We do this before the callback so that the handler can reuse the
  // timepoint storage if it wants. After this we can't rely on it being
  // valid.
  iree_hal_semaphore_callback_t callback = timepoint->callback;
  memset(timepoint, 0, sizeof(*timepoint));

  // Issue the callback.
  iree_status_ignore(
      callback.fn(callback.user_data, semaphore, new_value, new_status_code));

  // Release semaphore that was retained by the timepoint.
  // This _shouldn't_ be the last owner as the caller has to have a reference.
  iree_hal_semaphore_release(semaphore);

  IREE_TRACE_ZONE_END(z0);
}

// Issues callbacks for all timepoints in the |list|.
// Timepoints are released and the list is emptied upon return.
static void iree_hal_semaphore_issue_timepoint_callbacks(
    iree_hal_semaphore_t* semaphore, uint64_t new_value,
    iree_status_code_t new_status_code,
    iree_hal_semaphore_timepoint_list_t* list) {
  if (iree_hal_semaphore_timepoint_list_is_empty(list)) return;
  for (iree_hal_semaphore_timepoint_t* timepoint = list->head;
       timepoint != NULL;) {
    list->head = timepoint->next;
    timepoint->next = NULL;
    timepoint->prev = NULL;
    iree_hal_semaphore_issue_timepoint_callback(semaphore, new_value,
                                                new_status_code, timepoint);
    timepoint = list->head;
  }
  list->tail = NULL;
}

// NOTE: semaphore timepoint lock must not be held.
static void iree_hal_semaphore_resolve_timepoints(
    iree_hal_semaphore_t* semaphore, uint64_t new_value) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_timepoint_list_t pending_list = {NULL, NULL};
  iree_hal_semaphore_timepoint_list_t ready_list = {NULL, NULL};
  iree_hal_semaphore_timepoint_list_t expired_list = {NULL, NULL};

  iree_slim_mutex_lock(&semaphore->timepoint_mutex);

  if (iree_hal_semaphore_timepoint_list_is_empty(&semaphore->timepoint_list)) {
    iree_slim_mutex_unlock(&semaphore->timepoint_mutex);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Scan through the list and divvy up timepoints into still-pending, ready to
  // resolve, and expired buckets.
  iree_time_t now_ns = iree_time_now();
  for (iree_hal_semaphore_timepoint_t* timepoint =
           semaphore->timepoint_list.head;
       timepoint != NULL;) {
    iree_hal_semaphore_timepoint_t* next_timepoint = timepoint->next;
    timepoint->next = NULL;

    if (timepoint->minimum_value <= new_value) {
      // Reached the timepoint; even if the deadline has been reached we'll
      // still consider this a hit.
      iree_hal_semaphore_timepoint_list_push_back(&ready_list, timepoint);
    } else if (timepoint->deadline_ns <= now_ns) {
      // Deadline expired before the timepoint was reached.
      iree_hal_semaphore_timepoint_list_push_back(&expired_list, timepoint);
    } else {
      // Still pending.
      iree_hal_semaphore_timepoint_list_push_back(&pending_list, timepoint);
    }

    timepoint = next_timepoint;
  }

  // Preserve pending timepoints.
  semaphore->timepoint_list = pending_list;

  // Issue callbacks for all successes and failures.
  iree_hal_semaphore_issue_timepoint_callbacks(semaphore, new_value,
                                               IREE_STATUS_OK, &ready_list);
  iree_hal_semaphore_issue_timepoint_callbacks(
      semaphore, new_value, IREE_STATUS_DEADLINE_EXCEEDED, &expired_list);

  iree_slim_mutex_unlock(&semaphore->timepoint_mutex);

  IREE_TRACE_ZONE_END(z0);
}

// NOTE: semaphore timepoint lock must not be held.
static void iree_hal_semaphore_reject_timepoints(
    iree_hal_semaphore_t* semaphore, iree_status_code_t new_status_code) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&semaphore->timepoint_mutex);

  // Take the entire timepoint list from the semaphore.
  iree_hal_semaphore_timepoint_list_t failed_list = {NULL, NULL};
  iree_hal_semaphore_timepoint_list_take_all(&semaphore->timepoint_list,
                                             &failed_list);

  // Issue failure callbacks for all timepoints.
  iree_hal_semaphore_issue_timepoint_callbacks(semaphore, UINT64_MAX,
                                               new_status_code, &failed_list);

  iree_slim_mutex_unlock(&semaphore->timepoint_mutex);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_semaphore_initialize(
    const iree_hal_semaphore_vtable_t* vtable,
    iree_hal_semaphore_t* out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  iree_hal_resource_initialize(vtable, &out_semaphore->resource);
  iree_slim_mutex_initialize(&out_semaphore->timepoint_mutex);
  memset(&out_semaphore->timepoint_list, 0,
         sizeof(out_semaphore->timepoint_list));
}

IREE_API_EXPORT void iree_hal_semaphore_deinitialize(
    iree_hal_semaphore_t* semaphore) {
  IREE_ASSERT_ARGUMENT(semaphore);
  iree_slim_mutex_deinitialize(&semaphore->timepoint_mutex);
}

IREE_API_EXPORT void iree_hal_semaphore_acquire_timepoint(
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value,
    iree_timeout_t timeout, iree_hal_semaphore_callback_t callback,
    iree_hal_semaphore_timepoint_t* out_timepoint) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_timepoint);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Prepare timepoint structure.
  // Note that we capture the timeout as an absolute deadline as we don't know
  // how long it'll take to acquire the lock and how long it'll be pending.
  out_timepoint->next = NULL;
  out_timepoint->prev = NULL;
  out_timepoint->semaphore = semaphore;
  iree_hal_semaphore_retain(semaphore);
  out_timepoint->minimum_value = minimum_value;
  out_timepoint->deadline_ns = iree_timeout_as_deadline_ns(timeout);
  out_timepoint->callback = callback;

  // Insert into timepoint list.
  // After we release the lock the callback may be issued immediately as another
  // thread may be waiting to signal the timepoint.
  iree_slim_mutex_lock(&semaphore->timepoint_mutex);
  iree_hal_semaphore_timepoint_list_push_back(&semaphore->timepoint_list,
                                              out_timepoint);
  iree_slim_mutex_unlock(&semaphore->timepoint_mutex);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_semaphore_cancel_timepoint(
    iree_hal_semaphore_t* semaphore,
    iree_hal_semaphore_timepoint_t* timepoint) {
  if (!semaphore || !timepoint) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&semaphore->timepoint_mutex);

  // NOTE: if the semaphore is NULL then the timepoint has already been issued.
  // The caller is expected to know it's safe to still use the timepoint struct
  // even if such a race is possible.
  const bool needs_release = timepoint->semaphore != NULL;
  if (needs_release) {
    // Remove the timepoint from the list to ensure no other code can issue its
    // callback.
    iree_hal_semaphore_timepoint_list_erase(&semaphore->timepoint_list,
                                            timepoint);

    // Neuter the timepoint so that it is never called.
    // Other threads may be sitting and waiting for the lock and we need to
    // ensure that as soon as we unlock if they then try to operate on the
    // timepoint it's easy to bail.
    memset(timepoint, 0, sizeof(*timepoint));
  }

  iree_slim_mutex_unlock(&semaphore->timepoint_mutex);

  // Release the semaphore outside of the lock as it may recursively release
  // resources.
  if (needs_release) {
    iree_hal_semaphore_release(semaphore);
  }

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_semaphore_notify(
    iree_hal_semaphore_t* semaphore, uint64_t new_value,
    iree_status_code_t new_status_code) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: because the timepoints may be the last owners of the semaphore we
  // retain it so we can safely manage the lock.
  iree_hal_semaphore_retain(semaphore);

  if (new_status_code == IREE_STATUS_OK) {
    // Semaphore is in a valid state and has reached some value.
    // Resolve timepoints that have been hit (or have expired).
    iree_hal_semaphore_resolve_timepoints(semaphore, new_value);
  } else {
    // Semaphore has failed and we need to reject all timepoints.
    iree_hal_semaphore_reject_timepoints(semaphore, new_status_code);
  }

  // NOTE: semaphore may be destroyed after this!
  iree_hal_semaphore_release(semaphore);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_semaphore_poll(iree_hal_semaphore_t* semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Query the current value of the semaphore and notify with the results.
  uint64_t value = 0;
  iree_status_t status = iree_hal_semaphore_query(semaphore, &value);
  iree_hal_semaphore_notify(semaphore, value, iree_status_consume_code(status));

  IREE_TRACE_ZONE_END(z0);
}
