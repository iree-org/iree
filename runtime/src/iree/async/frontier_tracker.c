// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier_tracker.h"

#include "iree/async/semaphore.h"
#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns the index of |axis| in |frontier|->entries, or -1 if not found.
static int32_t iree_async_frontier_find_axis(
    const iree_async_frontier_t* frontier, iree_async_axis_t axis) {
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    if (frontier->entries[i].axis == axis) return (int32_t)i;
  }
  return -1;
}

// Result of checking a waiter's satisfaction state.
typedef enum iree_async_waiter_check_result_e {
  IREE_ASYNC_WAITER_CHECK_PENDING = 0,    // Not yet satisfied.
  IREE_ASYNC_WAITER_CHECK_SATISFIED = 1,  // All entries satisfied.
  IREE_ASYNC_WAITER_CHECK_FAILED = 2,     // At least one axis failed.
} iree_async_waiter_check_result_t;

// Checks whether a waiter is satisfied or failed.
// If FAILED, |out_failure_status| is set to the cloned failure status (caller
// owns). Must be called under the waiters_mutex.
static iree_async_waiter_check_result_t
iree_async_frontier_tracker_check_waiter(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_waiter_t* waiter,
    iree_status_t* out_failure_status) {
  const iree_async_frontier_t* frontier = waiter->frontier;
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    int32_t index = iree_async_axis_table_find(&tracker->axis_table,
                                               frontier->entries[i].axis);
    if (index < 0) {
      // Axis not in table — should not happen if wait() validated correctly.
      // Treat as unsatisfied (will never satisfy → stuck waiter).
      return IREE_ASYNC_WAITER_CHECK_PENDING;
    }

    // Check for axis failure.
    iree_status_t failure = tracker->axis_failure_statuses[index];
    if (!iree_status_is_ok(failure)) {
      *out_failure_status = iree_status_clone(failure);
      return IREE_ASYNC_WAITER_CHECK_FAILED;
    }

    // Check epoch.
    int64_t current_epoch =
        iree_atomic_load(&tracker->axis_table.entries[index].current_epoch,
                         iree_memory_order_acquire);
    if ((uint64_t)current_epoch < frontier->entries[i].epoch) {
      return IREE_ASYNC_WAITER_CHECK_PENDING;
    }
  }
  return IREE_ASYNC_WAITER_CHECK_SATISFIED;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_async_frontier_tracker_initialize(
    iree_async_frontier_tracker_t* tracker,
    iree_async_axis_table_entry_t* axis_table_entries,
    uint32_t axis_table_capacity, iree_allocator_t allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Initialize axis table.
  iree_async_axis_table_initialize(&tracker->axis_table, axis_table_entries,
                                   axis_table_capacity);

  // Initialize waiters list.
  iree_slim_mutex_initialize(&tracker->waiters_mutex);
  tracker->waiters_head = NULL;

  // Store allocator.
  tracker->allocator = allocator;

  // Allocate failure status array.
  iree_status_t status = iree_ok_status();
  if (axis_table_capacity > 0) {
    status = iree_allocator_malloc_array(
        allocator, axis_table_capacity, sizeof(iree_status_t),
        (void**)&tracker->axis_failure_statuses);
    if (iree_status_is_ok(status)) {
      memset(tracker->axis_failure_statuses, 0,
             axis_table_capacity * sizeof(iree_status_t));
    }
  } else {
    tracker->axis_failure_statuses = NULL;
  }

  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_deinitialize(&tracker->waiters_mutex);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_frontier_tracker_deinitialize(
    iree_async_frontier_tracker_t* tracker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Cancel all pending waiters under the waiters lock.
  iree_slim_mutex_lock(&tracker->waiters_mutex);
  iree_async_frontier_waiter_t* waiter = tracker->waiters_head;
  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;
    waiter->callback(
        waiter->user_data,
        iree_make_status(IREE_STATUS_CANCELLED, "tracker deinitialized"));
    waiter = next;
  }
  tracker->waiters_head = NULL;
  iree_slim_mutex_unlock(&tracker->waiters_mutex);

  // Deinitialize mutex.
  iree_slim_mutex_deinitialize(&tracker->waiters_mutex);

  // Free failure statuses.
  if (tracker->axis_failure_statuses != NULL) {
    for (uint32_t i = 0; i < tracker->axis_table.capacity; ++i) {
      if (!iree_status_is_ok(tracker->axis_failure_statuses[i])) {
        iree_status_free(tracker->axis_failure_statuses[i]);
      }
    }
    iree_allocator_free(tracker->allocator, tracker->axis_failure_statuses);
    tracker->axis_failure_statuses = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Tracker operations
//===----------------------------------------------------------------------===//

iree_host_size_t iree_async_frontier_tracker_advance(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    uint64_t epoch) {
  // Phase 1: Find axis and update epoch (lock-free).
  int32_t axis_index = iree_async_axis_table_find(&tracker->axis_table, axis);
  if (axis_index < 0) {
    return 0;  // Unknown axis — no-op.
  }

  iree_async_axis_table_entry_t* entry =
      &tracker->axis_table.entries[axis_index];

  // CAS loop to update epoch if advancing. Uses acq_rel ordering so that the
  // subsequent load of waiters_head (lock-free fast path) cannot be reordered
  // before the store. Without this, on weakly-ordered architectures (ARM),
  // advance() could store the epoch and read a stale waiters_head (NULL) while
  // wait() inserts a waiter and reads the old epoch — a classic lost wakeup.
  int64_t current_epoch;
  do {
    current_epoch =
        iree_atomic_load(&entry->current_epoch, iree_memory_order_acquire);
    if (epoch <= (uint64_t)current_epoch) {
      return 0;  // Monotonic — epoch not advancing.
    }
  } while (!iree_atomic_compare_exchange_weak(
      &entry->current_epoch, &current_epoch, (int64_t)epoch,
      iree_memory_order_acq_rel, iree_memory_order_relaxed));

  // Phase 2: Signal semaphore if present.
  if (entry->semaphore != NULL) {
    // Each timeline value must be signaled exactly once. If the signal fails
    // (e.g. semaphore already past this epoch), it indicates a structural
    // error — fail the semaphore so waiters get a proper diagnostic.
    iree_status_t signal_status =
        iree_async_semaphore_signal(entry->semaphore, epoch, NULL);
    if (IREE_UNLIKELY(!iree_status_is_ok(signal_status))) {
      iree_async_semaphore_fail(entry->semaphore, signal_status);
    }
  }

  // Phase 3: Collect satisfied waiters under lock, dispatch after unlock.
  // Callbacks must fire without the mutex held to prevent deadlock if a
  // callback reenters the tracker (e.g., registering a new waiter or
  // failing another axis).

  // Lock-free quick check: if no waiters, skip the mutex entirely.
  // This is safe because wait() re-checks satisfaction after insertion:
  //   - advance() reads waiters_head (sees NULL, stale due to concurrent
  //     insert)
  //   - advance() returns early (never takes lock)
  //   - wait() inserts waiter, re-checks epoch (sees new value from our CAS)
  //   - wait() dispatches immediately
  // The acq_rel CAS on current_epoch ensures this load cannot be reordered
  // before the epoch store, preventing lost wakeups on weakly-ordered
  // architectures.
  if (*(volatile iree_async_frontier_waiter_t**)&tracker->waiters_head ==
      NULL) {
    return 0;
  }

  iree_slim_mutex_lock(&tracker->waiters_mutex);

  // Re-check under lock: another thread may have emptied the list.
  if (tracker->waiters_head == NULL) {
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    return 0;
  }

  // Collect satisfied/failed waiters into a local dispatch list.
  iree_async_frontier_waiter_t* dispatch_head = NULL;
  iree_async_frontier_waiter_t** dispatch_tail = &dispatch_head;
  iree_host_size_t dispatched_count = 0;

  iree_async_frontier_waiter_t** prev_ptr = &tracker->waiters_head;
  iree_async_frontier_waiter_t* waiter = tracker->waiters_head;

  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;

    // Quick check: does this waiter care about the axis we just advanced?
    if (iree_async_frontier_find_axis(waiter->frontier, axis) < 0) {
      // This waiter doesn't reference the advanced axis — skip.
      prev_ptr = &waiter->next;
      waiter = next;
      continue;
    }

    // Full satisfaction check.
    iree_status_t failure_status = iree_ok_status();
    iree_async_waiter_check_result_t result =
        iree_async_frontier_tracker_check_waiter(tracker, waiter,
                                                 &failure_status);

    if (result == IREE_ASYNC_WAITER_CHECK_SATISFIED ||
        result == IREE_ASYNC_WAITER_CHECK_FAILED) {
      // Unlink from tracker list and append to dispatch list.
      *prev_ptr = next;
      waiter->next = NULL;
      *dispatch_tail = waiter;
      dispatch_tail = &waiter->next;
      // Stash failure status in user_data temporarily — we restore it
      // during dispatch below. For satisfied waiters, failure_status is
      // iree_ok_status() so this is a no-op.
      waiter->dispatch_status = failure_status;
      ++dispatched_count;
    } else {
      // Still pending — keep in list.
      prev_ptr = &waiter->next;
    }

    waiter = next;
  }

  iree_slim_mutex_unlock(&tracker->waiters_mutex);

  // Dispatch all collected waiters outside the lock.
  waiter = dispatch_head;
  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;
    waiter->callback(waiter->user_data, waiter->dispatch_status);
    waiter = next;
  }

  return dispatched_count;
}

iree_status_t iree_async_frontier_tracker_wait(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier,
    iree_async_frontier_waiter_fn_t callback, void* user_data,
    iree_async_frontier_waiter_t* waiter) {
  // Validate: all axes must be in the table.
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    if (iree_async_axis_table_find(&tracker->axis_table,
                                   frontier->entries[i].axis) < 0) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "frontier entry %" PRIu8
                              " references unknown axis 0x%016" PRIX64,
                              i, frontier->entries[i].axis);
    }
  }

  // Populate waiter fields (needed even for immediate dispatch, for
  // consistency).
  waiter->frontier = frontier;
  waiter->callback = callback;
  waiter->user_data = user_data;
  waiter->next = NULL;

  // Take lock and check satisfaction.
  iree_slim_mutex_lock(&tracker->waiters_mutex);

  iree_status_t failure_status = iree_ok_status();
  iree_async_waiter_check_result_t result =
      iree_async_frontier_tracker_check_waiter(tracker, waiter,
                                               &failure_status);

  if (result == IREE_ASYNC_WAITER_CHECK_SATISFIED) {
    // Already satisfied — dispatch immediately under lock.
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    callback(user_data, iree_ok_status());
    return iree_ok_status();
  }

  if (result == IREE_ASYNC_WAITER_CHECK_FAILED) {
    // Axis failed — dispatch immediately with failure.
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    callback(user_data, failure_status);
    return iree_ok_status();
  }

  // Not yet satisfied — insert into waiter list.
  waiter->next = tracker->waiters_head;
  tracker->waiters_head = waiter;

  // Re-check satisfaction after insertion. This handles the race where:
  //   1. wait() checks epoch (old value), decides to insert
  //   2. advance() updates epoch, does lock-free check (sees NULL), returns
  //   3. wait() inserts waiter (would be stuck without this re-check)
  // The re-check sees the new epoch and dispatches immediately.
  result = iree_async_frontier_tracker_check_waiter(tracker, waiter,
                                                    &failure_status);
  if (result == IREE_ASYNC_WAITER_CHECK_SATISFIED) {
    // Epoch advanced while we were inserting — remove and dispatch.
    tracker->waiters_head = waiter->next;
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    callback(user_data, iree_ok_status());
    return iree_ok_status();
  }
  if (result == IREE_ASYNC_WAITER_CHECK_FAILED) {
    // Axis failed while we were inserting — remove and dispatch with failure.
    tracker->waiters_head = waiter->next;
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    callback(user_data, failure_status);
    return iree_ok_status();
  }

  iree_slim_mutex_unlock(&tracker->waiters_mutex);
  return iree_ok_status();
}

bool iree_async_frontier_tracker_cancel_wait(
    iree_async_frontier_tracker_t* tracker,
    iree_async_frontier_waiter_t* waiter) {
  iree_slim_mutex_lock(&tracker->waiters_mutex);

  // Search for waiter in list.
  iree_async_frontier_waiter_t** prev_ptr = &tracker->waiters_head;
  iree_async_frontier_waiter_t* current = tracker->waiters_head;

  while (current != NULL) {
    if (current == waiter) {
      // Found — unlink and we're done. Callback will never fire.
      *prev_ptr = current->next;
      iree_slim_mutex_unlock(&tracker->waiters_mutex);
      return true;
    }
    prev_ptr = &current->next;
    current = current->next;
  }

  // Not found — waiter was already dispatched (callback completed or
  // in-flight). This is a no-op.
  iree_slim_mutex_unlock(&tracker->waiters_mutex);
  return false;
}

void iree_async_frontier_tracker_fail_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status) {
  // Find axis.
  int32_t axis_index = iree_async_axis_table_find(&tracker->axis_table, axis);
  if (axis_index < 0) {
    // Unknown axis — free status and return.
    iree_status_free(status);
    return;
  }

  // Take lock before checking/storing failure status.
  // This protects against:
  //   - Write-write race: two concurrent fail_axis() calls for the same axis
  //   - Read-write race: check_waiter() reads axis_failure_statuses under lock
  iree_slim_mutex_lock(&tracker->waiters_mutex);

  // Check for existing failure (first-failure-wins).
  if (!iree_status_is_ok(tracker->axis_failure_statuses[axis_index])) {
    // Already failed — ignore this failure.
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    iree_status_free(status);
    return;
  }

  // Store failure status (take ownership).
  tracker->axis_failure_statuses[axis_index] = status;

  // Fail the associated semaphore (if any). This dispatches all pending
  // semaphore timepoints with the failure status, bridging axis failure to
  // the proactor's semaphore-based wait infrastructure. Without this, proxy
  // semaphore timepoints would be orphaned when a remote axis fails.
  iree_async_semaphore_t* semaphore =
      tracker->axis_table.entries[axis_index].semaphore;
  if (semaphore != NULL) {
    iree_async_semaphore_fail(semaphore, iree_status_clone(status));
  }

  // Collect all waiters that reference this axis into a local dispatch list.
  // Callbacks must fire outside the lock to prevent deadlock if a callback
  // reenters the tracker.
  iree_async_frontier_waiter_t* dispatch_head = NULL;
  iree_async_frontier_waiter_t** dispatch_tail = &dispatch_head;

  iree_async_frontier_waiter_t** prev_ptr = &tracker->waiters_head;
  iree_async_frontier_waiter_t* waiter = tracker->waiters_head;

  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;

    // Check if this waiter references the failed axis.
    if (iree_async_frontier_find_axis(waiter->frontier, axis) >= 0) {
      // Unlink from tracker list and append to dispatch list.
      *prev_ptr = next;
      waiter->next = NULL;
      *dispatch_tail = waiter;
      dispatch_tail = &waiter->next;
    } else {
      prev_ptr = &waiter->next;
    }

    waiter = next;
  }

  iree_slim_mutex_unlock(&tracker->waiters_mutex);

  // Dispatch all collected waiters outside the lock.
  waiter = dispatch_head;
  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;
    waiter->callback(waiter->user_data, iree_status_clone(status));
    waiter = next;
  }
}
