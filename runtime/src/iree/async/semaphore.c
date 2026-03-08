// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/semaphore.h"

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"

//===----------------------------------------------------------------------===//
// Semaphore implementation
//===----------------------------------------------------------------------===//

// Tentative definition; full definition at end of section.
static const iree_async_semaphore_vtable_t iree_async_semaphore_default_vtable;

IREE_API_EXPORT void iree_async_semaphore_initialize(
    const iree_async_semaphore_vtable_t* vtable,
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_host_size_t frontier_offset, uint8_t frontier_capacity,
    iree_async_semaphore_t* out_semaphore) {
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  iree_atomic_ref_count_init(&out_semaphore->ref_count);
  out_semaphore->vtable = vtable;
  out_semaphore->proactor = proactor;
  iree_atomic_store(&out_semaphore->timeline_value, (int64_t)initial_value,
                    iree_memory_order_release);
  iree_atomic_store(&out_semaphore->last_untainted_value,
                    (int64_t)initial_value, iree_memory_order_release);
  iree_slim_mutex_initialize(&out_semaphore->mutex);
  iree_atomic_store(&out_semaphore->failure_status, 0,
                    iree_memory_order_release);
  out_semaphore->timepoints_head = NULL;
  out_semaphore->frontier_capacity = frontier_capacity;
  out_semaphore->frontier =
      (iree_async_frontier_t*)((uint8_t*)out_semaphore + frontier_offset);
  iree_async_frontier_initialize(out_semaphore->frontier, 0);
}

IREE_API_EXPORT void iree_async_semaphore_retain(
    iree_async_semaphore_t* semaphore) {
  if (semaphore) {
    iree_atomic_ref_count_inc(&semaphore->ref_count);
  }
}

IREE_API_EXPORT void iree_async_semaphore_release(
    iree_async_semaphore_t* semaphore) {
  if (semaphore && iree_atomic_ref_count_dec(&semaphore->ref_count) == 1) {
    semaphore->vtable->destroy(semaphore);
  }
}

IREE_API_EXPORT void iree_async_semaphore_deinitialize(
    iree_async_semaphore_t* semaphore) {
  IREE_ASSERT_ARGUMENT(semaphore);

  // Dispatch all pending timepoints with CANCELLED status.
  iree_slim_mutex_lock(&semaphore->mutex);
  iree_async_semaphore_timepoint_t* timepoint = semaphore->timepoints_head;
  while (timepoint != NULL) {
    iree_async_semaphore_timepoint_t* next = timepoint->next;
    timepoint->callback(
        timepoint->user_data, timepoint,
        iree_make_status(IREE_STATUS_CANCELLED, "semaphore destroyed"));
    timepoint = next;
  }
  semaphore->timepoints_head = NULL;
  iree_slim_mutex_unlock(&semaphore->mutex);

  // Free failure status if set. Deinitialize has exclusive access — no
  // concurrent operations — so relaxed ordering is sufficient.
  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &semaphore->failure_status, iree_memory_order_relaxed);
  if (!iree_status_is_ok(failure_status)) {
    iree_status_free(failure_status);
  }

  iree_slim_mutex_deinitialize(&semaphore->mutex);
}

IREE_API_EXPORT iree_status_t iree_async_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    uint8_t frontier_capacity, iree_allocator_t allocator,
    iree_async_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  iree_host_size_t frontier_offset = 0, total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_semaphore_layout(sizeof(iree_async_semaphore_t),
                                      frontier_capacity, &frontier_offset,
                                      &total_size));
  iree_async_semaphore_t* semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&semaphore));
  iree_async_semaphore_initialize(&iree_async_semaphore_default_vtable,
                                  proactor, initial_value, frontier_offset,
                                  frontier_capacity, semaphore);
  semaphore->allocator = allocator;

  *out_semaphore = semaphore;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_semaphore_default_destroy(
    iree_async_semaphore_t* semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = semaphore->allocator;
  iree_async_semaphore_deinitialize(semaphore);
  iree_allocator_free(allocator, semaphore);
  IREE_TRACE_ZONE_END(z0);
}

static uint64_t iree_async_semaphore_default_query(
    iree_async_semaphore_t* semaphore) {
  return (uint64_t)iree_atomic_load(&semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_async_semaphore_default_signal(
    iree_async_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  // CAS loop to advance the timeline value monotonically.
  // We store uint64 values as bit patterns in int64_t atomics. Comparisons
  // must cast BOTH sides to uint64_t to handle the full range correctly.
  int64_t current_raw = 0;
  do {
    current_raw =
        iree_atomic_load(&semaphore->timeline_value, iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) {
      // Timeline is already at or past the requested value. Each timeline
      // value must be signaled exactly once — concurrent races are not valid
      // (they indicate duplicate signals or non-monotonic scheduling).
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "semaphore already signaled past %" PRIu64,
                              value);
    }
  } while (!iree_atomic_compare_exchange_weak(
      &semaphore->timeline_value, &current_raw, (int64_t)value,
      iree_memory_order_release, iree_memory_order_relaxed));

  // Merge frontier if provided.
  iree_status_t merge_status = iree_ok_status();
  if (frontier != NULL && frontier->entry_count > 0) {
    // We need to hold the mutex during frontier merge to avoid concurrent
    // merges corrupting the frontier. This is acceptable because frontier
    // merge is fast (O(n) where n is entry count).
    iree_slim_mutex_lock(&semaphore->mutex);
    merge_status = iree_async_frontier_merge(
        semaphore->frontier, semaphore->frontier_capacity, frontier);
    iree_slim_mutex_unlock(&semaphore->mutex);
    // Note: we continue to dispatch even on merge failure because the timeline
    // has already been advanced. Waiters must be woken to avoid hangs.
  }

  // Dispatch satisfied timepoints. This must happen even if merge failed
  // because the timeline value has already been published via CAS.
  iree_async_semaphore_dispatch_timepoints(semaphore, value);

  return merge_status;
}

static uint8_t iree_async_semaphore_default_query_frontier(
    iree_async_semaphore_t* semaphore, iree_async_frontier_t* out_frontier,
    uint8_t capacity) {
  iree_slim_mutex_lock(&semaphore->mutex);
  uint8_t actual_count = semaphore->frontier->entry_count;
  if (out_frontier != NULL) {
    uint8_t copy_count = actual_count < capacity ? actual_count : capacity;
    iree_async_frontier_initialize(out_frontier, copy_count);
    memcpy(out_frontier->entries, semaphore->frontier->entries,
           copy_count * sizeof(iree_async_frontier_entry_t));
  }
  iree_slim_mutex_unlock(&semaphore->mutex);

  return actual_count;
}

static void iree_async_semaphore_default_fail(iree_async_semaphore_t* semaphore,
                                              iree_status_t status) {
  // First failure wins via CAS. Release semantics ensure the status payload
  // is visible to any thread that loads the pointer with acquire.
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &semaphore->failure_status, &expected, (intptr_t)status,
          iree_memory_order_release, iree_memory_order_acquire)) {
    iree_status_free(status);
    return;
  }

  // Dispatch all pending timepoints with the failure status.
  iree_slim_mutex_lock(&semaphore->mutex);
  iree_async_semaphore_timepoint_t* timepoint = semaphore->timepoints_head;
  while (timepoint != NULL) {
    iree_async_semaphore_timepoint_t* next = timepoint->next;
    timepoint->callback(timepoint->user_data, timepoint,
                        iree_status_clone(status));
    timepoint = next;
  }
  semaphore->timepoints_head = NULL;

  iree_slim_mutex_unlock(&semaphore->mutex);
}

IREE_API_EXPORT iree_status_t iree_async_semaphore_acquire_timepoint(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_async_semaphore_timepoint_t* timepoint) {
  // Initialize timepoint fields.
  timepoint->semaphore = semaphore;
  timepoint->minimum_value = minimum_value;
  timepoint->next = NULL;
  timepoint->prev = NULL;

  // Fast path: check for immediate failure or satisfaction without the lock.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    timepoint->callback(timepoint->user_data, timepoint,
                        iree_status_clone(failure));
    return iree_ok_status();
  }
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &semaphore->timeline_value, iree_memory_order_acquire);
  if (current_value >= minimum_value) {
    timepoint->callback(timepoint->user_data, timepoint, iree_ok_status());
    return iree_ok_status();
  }

  // Acquire the lock for list mutation and re-check under lock.
  iree_slim_mutex_lock(&semaphore->mutex);

  // Insert into the doubly-linked list (prepend for O(1)).
  timepoint->next = semaphore->timepoints_head;
  if (semaphore->timepoints_head != NULL) {
    semaphore->timepoints_head->prev = timepoint;
  }
  semaphore->timepoints_head = timepoint;

  // Re-check after insertion (handles signal-during-insertion race).
  // This matches the pattern used in frontier_tracker.c.
  current_value = (uint64_t)iree_atomic_load(&semaphore->timeline_value,
                                             iree_memory_order_acquire);
  if (current_value >= minimum_value) {
    // Unlink and dispatch immediately.
    if (timepoint->next != NULL) {
      timepoint->next->prev = timepoint->prev;
    }
    if (timepoint->prev != NULL) {
      timepoint->prev->next = timepoint->next;
    } else {
      semaphore->timepoints_head = timepoint->next;
    }
    iree_slim_mutex_unlock(&semaphore->mutex);
    timepoint->callback(timepoint->user_data, timepoint, iree_ok_status());
    return iree_ok_status();
  }

  // Also re-check failure status (set via CAS outside the mutex).
  failure = (iree_status_t)iree_atomic_load(&semaphore->failure_status,
                                            iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    // Unlink and dispatch with failure.
    if (timepoint->next != NULL) {
      timepoint->next->prev = timepoint->prev;
    }
    if (timepoint->prev != NULL) {
      timepoint->prev->next = timepoint->next;
    } else {
      semaphore->timepoints_head = timepoint->next;
    }
    iree_slim_mutex_unlock(&semaphore->mutex);
    timepoint->callback(timepoint->user_data, timepoint,
                        iree_status_clone(failure));
    return iree_ok_status();
  }

  iree_slim_mutex_unlock(&semaphore->mutex);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_semaphore_cancel_timepoint(
    iree_async_semaphore_t* semaphore,
    iree_async_semaphore_timepoint_t* timepoint) {
  iree_slim_mutex_lock(&semaphore->mutex);

  // Search for the timepoint in the list.
  iree_async_semaphore_timepoint_t* current = semaphore->timepoints_head;
  while (current != NULL) {
    if (current == timepoint) {
      // Found — unlink it.
      if (current->next != NULL) {
        current->next->prev = current->prev;
      }
      if (current->prev != NULL) {
        current->prev->next = current->next;
      } else {
        semaphore->timepoints_head = current->next;
      }
      iree_slim_mutex_unlock(&semaphore->mutex);
      return;  // Callback will not fire.
    }
    current = current->next;
  }

  // Not found — callback already completed (or never registered).
  // This is a no-op. With dispatch-under-lock, acquiring the lock guarantees
  // the callback has finished if the timepoint isn't in the list.
  iree_slim_mutex_unlock(&semaphore->mutex);
}

static iree_status_t iree_async_semaphore_default_export_primitive(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_async_primitive_t* out_primitive) {
  // Default semaphores cannot export pollable primitives.
  // Callers should use the timepoint callback mechanism instead.
  (void)semaphore;
  (void)minimum_value;
  (void)out_primitive;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "default semaphores do not support primitive export");
}

static const iree_async_semaphore_vtable_t iree_async_semaphore_default_vtable =
    {
        .destroy = iree_async_semaphore_default_destroy,
        .query = iree_async_semaphore_default_query,
        .signal = iree_async_semaphore_default_signal,
        .query_frontier = iree_async_semaphore_default_query_frontier,
        .fail = iree_async_semaphore_default_fail,
        .export_primitive = iree_async_semaphore_default_export_primitive,
};

//===----------------------------------------------------------------------===//
// Tainting API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT bool iree_async_semaphore_is_value_tainted(
    iree_async_semaphore_t* semaphore, uint64_t value) {
  uint64_t untainted = (uint64_t)iree_atomic_load(
      &semaphore->last_untainted_value, iree_memory_order_acquire);
  return value > untainted;
}

IREE_API_EXPORT void iree_async_semaphore_mark_tainted_above(
    iree_async_semaphore_t* semaphore, uint64_t threshold) {
  // The watermark only decreases (marking more values as tainted).
  // Use CAS loop to ensure we never increase the watermark.
  // We store uint64 values as bit patterns in int64_t atomics.
  int64_t current_raw = 0;
  do {
    current_raw = iree_atomic_load(&semaphore->last_untainted_value,
                                   iree_memory_order_acquire);
    if (threshold >= (uint64_t)current_raw) {
      return;  // Already at or below threshold, nothing to do.
    }
  } while (!iree_atomic_compare_exchange_weak(
      &semaphore->last_untainted_value, &current_raw, (int64_t)threshold,
      iree_memory_order_release, iree_memory_order_relaxed));
}

IREE_API_EXPORT iree_status_t iree_async_semaphore_signal_untainted(
    iree_async_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  // Signal the semaphore normally.
  IREE_RETURN_IF_ERROR(iree_async_semaphore_signal(semaphore, value, frontier));

  // Advance the untainted watermark.
  // Use CAS loop to ensure monotonic advancement.
  // We store uint64 values as bit patterns in int64_t atomics.
  int64_t current_raw = 0;
  do {
    current_raw = iree_atomic_load(&semaphore->last_untainted_value,
                                   iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) {
      return iree_ok_status();  // Already at or past this value.
    }
  } while (!iree_atomic_compare_exchange_weak(
      &semaphore->last_untainted_value, &current_raw, (int64_t)value,
      iree_memory_order_release, iree_memory_order_relaxed));

  return iree_ok_status();
}

IREE_API_EXPORT uint64_t
iree_async_semaphore_query_untainted_value(iree_async_semaphore_t* semaphore) {
  return (uint64_t)iree_atomic_load(&semaphore->last_untainted_value,
                                    iree_memory_order_acquire);
}

//===----------------------------------------------------------------------===//
// HAL composition helpers
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_semaphore_advance_timeline(
    iree_async_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  // Advances the timeline value and merges the frontier, but does NOT dispatch
  // timepoints. HAL implementations call this first, then signal their native
  // primitive, then call dispatch_timepoints.
  //
  // Returns INVALID_ARGUMENT if the timeline is already at or past |value| —
  // each timeline value must be signaled exactly once and concurrent races are
  // not valid (they indicate duplicate signals or non-monotonic scheduling).
  // Returns frontier merge errors if a frontier is provided and the merge
  // fails; in that case the timeline HAS been advanced and callers MUST still
  // call dispatch_timepoints.

  // CAS loop to advance the timeline value monotonically.
  // We store uint64 values as bit patterns in int64_t atomics. Comparisons
  // must cast BOTH sides to uint64_t to handle the full range correctly.
  int64_t current_raw = 0;
  do {
    current_raw =
        iree_atomic_load(&semaphore->timeline_value, iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) {
      // Timeline is already at or past the requested value. Each timeline
      // value must be signaled exactly once — concurrent races are not valid
      // (they indicate duplicate signals or non-monotonic scheduling).
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "semaphore already signaled past %" PRIu64,
                              value);
    }
  } while (!iree_atomic_compare_exchange_weak(
      &semaphore->timeline_value, &current_raw, (int64_t)value,
      iree_memory_order_release, iree_memory_order_relaxed));

  // Merge frontier if provided. Note: merge failure does not prevent the
  // signal from taking effect — the timeline has already been advanced.
  // Callers should still dispatch timepoints even if this returns an error.
  if (frontier != NULL && frontier->entry_count > 0) {
    iree_slim_mutex_lock(&semaphore->mutex);
    iree_status_t merge_status = iree_async_frontier_merge(
        semaphore->frontier, semaphore->frontier_capacity, frontier);
    iree_slim_mutex_unlock(&semaphore->mutex);
    return merge_status;  // Return merge status but signal has taken effect.
  }

  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_semaphore_dispatch_timepoints(
    iree_async_semaphore_t* semaphore, uint64_t value) {
  iree_slim_mutex_lock(&semaphore->mutex);

  // Walk the list and dispatch all satisfied timepoints.
  iree_async_semaphore_timepoint_t** prev_ptr = &semaphore->timepoints_head;
  iree_async_semaphore_timepoint_t* timepoint = semaphore->timepoints_head;

  while (timepoint != NULL) {
    iree_async_semaphore_timepoint_t* next = timepoint->next;

    if (timepoint->minimum_value <= value) {
      // Unlink.
      *prev_ptr = next;
      if (next != NULL) {
        next->prev = timepoint->prev;
      }
      // Dispatch under lock.
      timepoint->callback(timepoint->user_data, timepoint, iree_ok_status());
    } else {
      prev_ptr = &timepoint->next;
    }

    timepoint = next;
  }

  iree_slim_mutex_unlock(&semaphore->mutex);
}

IREE_API_EXPORT void iree_async_semaphore_dispatch_timepoints_failed(
    iree_async_semaphore_t* semaphore, iree_status_t status) {
  iree_slim_mutex_lock(&semaphore->mutex);

  // Walk the list and dispatch all timepoints with the failure status.
  iree_async_semaphore_timepoint_t* timepoint = semaphore->timepoints_head;
  while (timepoint != NULL) {
    iree_async_semaphore_timepoint_t* next = timepoint->next;
    timepoint->callback(timepoint->user_data, timepoint,
                        iree_status_clone(status));
    timepoint = next;
  }
  semaphore->timepoints_head = NULL;

  iree_slim_mutex_unlock(&semaphore->mutex);

  // Free the original status.
  iree_status_free(status);
}

//===----------------------------------------------------------------------===//
// Multi-wait
//===----------------------------------------------------------------------===//

// Shared state for all timepoints in a multi-wait operation. A single
// notification is used to wake the blocking thread when any timepoint fires,
// instead of one event per semaphore.
typedef struct iree_async_multi_wait_state_t {
  // Number of semaphores not yet satisfied. Decremented atomically by each
  // timepoint callback. For ALL mode, the wait completes when this reaches 0.
  // For ANY mode, the wait completes when this drops below the initial count.
  iree_atomic_int32_t remaining_count;

  // First failure status captured from any semaphore (CAS-guarded).
  // Non-zero indicates at least one semaphore failed.
  iree_atomic_intptr_t failure_status;

  // Notification used to wake the blocking thread. Posted by every timepoint
  // callback (satisfied or failed), allowing the wait loop to re-check.
  iree_notification_t notification;
} iree_async_multi_wait_state_t;

// Per-semaphore timepoint that references the shared multi-wait state.
// Extends iree_async_semaphore_timepoint_t at offset 0 for intrusive linkage.
typedef struct iree_async_multi_wait_timepoint_t {
  iree_async_semaphore_timepoint_t base;
  iree_async_multi_wait_state_t* state;
} iree_async_multi_wait_timepoint_t;

// Timepoint callback shared by all semaphores in a multi-wait.
// Fires under each semaphore's internal lock. Only performs atomic ops and a
// notification post — both safe under slim_mutex.
static void iree_async_multi_wait_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* base_timepoint,
    iree_status_t status) {
  iree_async_multi_wait_timepoint_t* timepoint =
      (iree_async_multi_wait_timepoint_t*)base_timepoint;
  iree_async_multi_wait_state_t* state = timepoint->state;

  if (!iree_status_is_ok(status)) {
    // Record first failure via CAS. Subsequent failures are freed.
    intptr_t expected = 0;
    if (!iree_atomic_compare_exchange_strong(
            &state->failure_status, &expected, (intptr_t)status,
            iree_memory_order_release, iree_memory_order_acquire)) {
      iree_status_free(status);
    }
  }

  // Decrement remaining count (even on failure — the semaphore is "resolved").
  iree_atomic_fetch_sub(&state->remaining_count, 1, iree_memory_order_acq_rel);

  // Wake the blocking thread unconditionally. The wait loop re-checks the
  // actual condition (ALL vs ANY) after waking.
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
}

// Condition predicate for ALL mode: all semaphores satisfied or any failed.
typedef struct iree_async_multi_wait_condition_t {
  iree_async_multi_wait_state_t* state;
  iree_async_wait_mode_t wait_mode;
  int32_t initial_count;
} iree_async_multi_wait_condition_t;

static bool iree_async_multi_wait_condition_fn(void* arg) {
  iree_async_multi_wait_condition_t* condition =
      (iree_async_multi_wait_condition_t*)arg;
  iree_async_multi_wait_state_t* state = condition->state;

  // Any failure wakes immediately.
  if (iree_atomic_load(&state->failure_status, iree_memory_order_acquire) !=
      0) {
    return true;
  }

  int32_t remaining =
      iree_atomic_load(&state->remaining_count, iree_memory_order_acquire);
  if (condition->wait_mode == IREE_ASYNC_WAIT_MODE_ALL) {
    return remaining <= 0;
  } else {
    // ANY: at least one satisfied.
    return remaining < condition->initial_count;
  }
}

// Maximum number of semaphores for stack-allocated timepoint storage.
// Covers typical multi-GPU workloads (2-8 devices) without heap allocation.
#define IREE_ASYNC_MULTI_WAIT_INLINE_CAPACITY 8

IREE_API_EXPORT iree_status_t iree_async_semaphore_multi_wait(
    iree_async_wait_mode_t wait_mode, iree_async_semaphore_t** semaphores,
    const uint64_t* minimum_values, iree_host_size_t count,
    iree_timeout_t timeout, iree_allocator_t allocator) {
  if (count == 0) return iree_ok_status();

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)count);

  // Fast path: single semaphore. The timepoint machinery is overkill.
  if (count == 1) {
    // Check failure before value — the vtable query may encode failure as a
    // high sentinel value that would falsely satisfy the >= check.
    iree_status_t failure = (iree_status_t)iree_atomic_load(
        &semaphores[0]->failure_status, iree_memory_order_acquire);
    if (!iree_status_is_ok(failure)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_status_from_code(iree_status_code(failure));
    }

    // Atomic check for immediate satisfaction.
    uint64_t current_value = (uint64_t)iree_atomic_load(
        &semaphores[0]->timeline_value, iree_memory_order_acquire);
    if (current_value >= minimum_values[0]) {
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }

    if (iree_timeout_is_immediate(timeout)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }

    // Fall through to the general path for single-semaphore blocking wait.
  }

  // Check for immediate timeout after fast-path checks.
  if (iree_timeout_is_immediate(timeout)) {
    // Poll all semaphores without blocking. Uses raw timeline_value rather
    // than the vtable query to avoid failure sentinel encoding that could
    // falsely satisfy comparisons.
    bool any_satisfied = false;
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_status_t failure = (iree_status_t)iree_atomic_load(
          &semaphores[i]->failure_status, iree_memory_order_acquire);
      if (!iree_status_is_ok(failure)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_status_from_code(iree_status_code(failure));
      }
      uint64_t current_value = (uint64_t)iree_atomic_load(
          &semaphores[i]->timeline_value, iree_memory_order_acquire);
      if (current_value >= minimum_values[i]) {
        any_satisfied = true;
        if (wait_mode == IREE_ASYNC_WAIT_MODE_ANY) {
          IREE_TRACE_ZONE_END(z0);
          return iree_ok_status();
        }
      }
    }
    IREE_TRACE_ZONE_END(z0);
    if (wait_mode == IREE_ASYNC_WAIT_MODE_ALL && any_satisfied) {
      // Some satisfied but not all — recheck all for ALL mode.
      for (iree_host_size_t i = 0; i < count; ++i) {
        iree_status_t failure = (iree_status_t)iree_atomic_load(
            &semaphores[i]->failure_status, iree_memory_order_acquire);
        if (!iree_status_is_ok(failure)) {
          return iree_status_from_code(iree_status_code(failure));
        }
        uint64_t current_value = (uint64_t)iree_atomic_load(
            &semaphores[i]->timeline_value, iree_memory_order_acquire);
        if (current_value < minimum_values[i]) {
          return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
        }
      }
      return iree_ok_status();
    }
    return iree_status_from_code(any_satisfied ? IREE_STATUS_OK
                                               : IREE_STATUS_DEADLINE_EXCEEDED);
  }

  // Allocate timepoint storage. Use stack for small counts.
  iree_async_multi_wait_timepoint_t
      inline_timepoints[IREE_ASYNC_MULTI_WAIT_INLINE_CAPACITY];
  iree_async_multi_wait_timepoint_t* timepoints = inline_timepoints;
  if (count > IREE_ARRAYSIZE(inline_timepoints)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(allocator, count * sizeof(timepoints[0]),
                                  (void**)&timepoints));
  }
  memset(timepoints, 0, count * sizeof(timepoints[0]));

  // Initialize shared wait state.
  iree_async_multi_wait_state_t state;
  iree_atomic_store(&state.remaining_count, (int32_t)count,
                    iree_memory_order_release);
  iree_atomic_store(&state.failure_status, 0, iree_memory_order_release);
  iree_notification_initialize(&state.notification);

  // Register timepoints on all semaphores. Each callback shares the state and
  // will wake the blocking thread via the notification.
  iree_host_size_t registered_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count; ++i) {
    timepoints[i].state = &state;
    timepoints[i].base.callback = iree_async_multi_wait_timepoint_callback;
    timepoints[i].base.user_data = NULL;
    status = iree_async_semaphore_acquire_timepoint(
        semaphores[i], minimum_values[i], &timepoints[i].base);
    if (!iree_status_is_ok(status)) break;
    ++registered_count;
  }

  // Block until the wait condition is met. The notification-based await handles
  // spurious wakes correctly: it re-evaluates the predicate after each wake.
  if (iree_status_is_ok(status)) {
    iree_async_multi_wait_condition_t condition = {
        .state = &state,
        .wait_mode = wait_mode,
        .initial_count = (int32_t)count,
    };

    // Check if the condition is already met (timepoints may have fired
    // synchronously during acquire_timepoint above).
    if (!iree_async_multi_wait_condition_fn(&condition)) {
      bool satisfied = iree_notification_await(
          &state.notification, iree_async_multi_wait_condition_fn, &condition,
          timeout);
      if (!satisfied) {
        status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
    }

    // Check for failure status even if we didn't time out. Returns the failure
    // status code (not the full status) to match the fast-path behavior and the
    // HAL convention of following up with a query to get the full status.
    if (iree_status_is_ok(status)) {
      iree_status_t failure = (iree_status_t)iree_atomic_load(
          &state.failure_status, iree_memory_order_acquire);
      if (!iree_status_is_ok(failure)) {
        status = iree_status_from_code(iree_status_code(failure));
      }
    }
  }

  // Cancel any timepoints that haven't fired yet. After cancel returns, the
  // callback is guaranteed not to fire (dispatch-under-lock semantics).
  for (iree_host_size_t i = 0; i < registered_count; ++i) {
    if (timepoints[i].base.semaphore != NULL) {
      iree_async_semaphore_cancel_timepoint(timepoints[i].base.semaphore,
                                            &timepoints[i].base);
    }
  }

  // Free captured failure status clone (we extracted the code above).
  iree_status_t captured_failure = (iree_status_t)iree_atomic_load(
      &state.failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(captured_failure)) {
    iree_status_free(captured_failure);
  }

  iree_notification_deinitialize(&state.notification);

  if (timepoints != inline_timepoints) {
    iree_allocator_free(allocator, timepoints);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Device fence bridging (via proactor)
//===----------------------------------------------------------------------===//

// NOTE: These functions require a proactor but are declared in semaphore.h to
// keep all semaphore-related APIs together. The proactor.h vtable provides the
// implementations, so we include proactor.h here for the vtable access.
#include "iree/async/proactor.h"

IREE_API_EXPORT iree_status_t iree_async_semaphore_import_fence(
    iree_async_proactor_t* proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      proactor->vtable->import_fence(proactor, fence, semaphore, signal_value);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_async_semaphore_export_fence(
    iree_async_proactor_t* proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_fence);
  memset(out_fence, 0, sizeof(*out_fence));
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = proactor->vtable->export_fence(proactor, semaphore,
                                                        wait_value, out_fence);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
