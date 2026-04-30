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
#include "iree/base/threading/processor.h"

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

// Detaches all timepoints from the semaphore's pending list, returning them
// as a singly-linked list (via ->next). The semaphore's list is left empty.
// Must be called with semaphore->mutex held.
static iree_async_semaphore_timepoint_t* iree_async_semaphore_detach_all_locked(
    iree_async_semaphore_t* semaphore) {
  iree_async_semaphore_timepoint_t* head = semaphore->timepoints_head;
  semaphore->timepoints_head = NULL;
  return head;
}

// Dispatches a detached timepoint list, firing each callback with the given
// |status|. For OK status, passes iree_ok_status() directly. For failure
// status, clones it for each callback (each callback takes ownership of its
// clone). Does NOT take ownership of |status|.
// Must be called WITHOUT the semaphore's lock held.
static void iree_async_semaphore_dispatch_detached(
    iree_async_semaphore_timepoint_t* head, iree_status_t status) {
  while (head != NULL) {
    iree_async_semaphore_timepoint_t* next = head->next;
    head->next = NULL;
    head->prev = NULL;
    head->callback(head->user_data, head,
                   iree_status_is_ok(status) ? iree_ok_status()
                                             : iree_status_clone(status));
    head = next;
  }
}

IREE_API_EXPORT void iree_async_semaphore_deinitialize(
    iree_async_semaphore_t* semaphore) {
  IREE_ASSERT_ARGUMENT(semaphore);

  // Detach all pending timepoints under lock.
  iree_slim_mutex_lock(&semaphore->mutex);
  iree_async_semaphore_timepoint_t* pending =
      iree_async_semaphore_detach_all_locked(semaphore);
  iree_slim_mutex_unlock(&semaphore->mutex);

  // Dispatch outside lock with CANCELLED status.
  iree_status_t cancelled =
      iree_make_status(IREE_STATUS_CANCELLED, "semaphore destroyed");
  iree_async_semaphore_dispatch_detached(pending, cancelled);
  iree_status_free(cancelled);

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
  // No native signaling for the default implementation — just advance the
  // timeline, merge the frontier, and dispatch timepoints.
  IREE_RETURN_IF_ERROR(
      iree_async_semaphore_advance_timeline(semaphore, value, frontier));
  iree_async_semaphore_dispatch_timepoints(semaphore, value);
  return iree_ok_status();
}

IREE_API_EXPORT uint8_t iree_async_semaphore_query_frontier(
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

IREE_API_EXPORT void iree_async_semaphore_fail(
    iree_async_semaphore_t* semaphore, iree_status_t status) {
  // First failure wins. Store the original directly — no clone needed.
  // Release semantics ensure the status payload is visible to any thread
  // that loads the pointer with acquire.
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &semaphore->failure_status, &expected, (intptr_t)status,
          iree_memory_order_release, iree_memory_order_acquire)) {
    iree_status_free(status);
    return;
  }

  // |status| is now stored in failure_status (owned by the semaphore, freed
  // at deinitialize). dispatch_timepoints_failed borrows it as a template
  // for per-timepoint clones without taking ownership.
  iree_status_code_t status_code = iree_status_code(status);
  iree_async_semaphore_dispatch_timepoints_failed(semaphore, status);

  // Notify hardware backend for device-side failure signaling.
  if (semaphore->vtable->on_fail) {
    semaphore->vtable->on_fail(semaphore, status_code);
  }
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

IREE_API_EXPORT bool iree_async_semaphore_cancel_timepoint(
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
      return true;  // Cancelled — callback will not fire.
    }
    current = current->next;
  }

  // Not found — already dispatched (or dispatching). The callback has been
  // collected for dispatch and may be executing or may have completed.
  // Callers that need to wait for completion must use their own mechanism.
  iree_slim_mutex_unlock(&semaphore->mutex);
  return false;
}

//===----------------------------------------------------------------------===//
// Resolve (wait_source callback bridge)
//===----------------------------------------------------------------------===//

// Heap-allocated wrapper bridging a timepoint callback to a wait_source
// resolve callback. Freed when the timepoint fires.
typedef struct iree_async_semaphore_resolve_state_t {
  iree_async_semaphore_timepoint_t timepoint;
  iree_wait_source_resolve_callback_t callback;
  void* user_data;
} iree_async_semaphore_resolve_state_t;

static void iree_async_semaphore_resolve_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_async_semaphore_resolve_state_t* state =
      (iree_async_semaphore_resolve_state_t*)user_data;
  iree_wait_source_resolve_callback_t callback = state->callback;
  void* callback_user_data = state->user_data;
  iree_allocator_free(iree_allocator_system(), state);
  callback(callback_user_data, status);
}

IREE_API_EXPORT iree_status_t iree_async_semaphore_resolve(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_timeout_t timeout, iree_wait_source_resolve_callback_t callback,
    void* user_data) {
  IREE_ASSERT_ARGUMENT(callback);
  (void)timeout;  // Async — caller manages deadlines externally.

  // Fast path: check for immediate failure without allocation.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    callback(user_data, iree_status_clone(failure));
    return iree_ok_status();
  }

  // Fast path: check for immediate satisfaction without allocation.
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &semaphore->timeline_value, iree_memory_order_acquire);
  if (current_value >= minimum_value) {
    callback(user_data, iree_ok_status());
    return iree_ok_status();
  }

  // Slow path: allocate wrapper and register timepoint.
  iree_async_semaphore_resolve_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(),
                                             sizeof(*state), (void**)&state));
  state->callback = callback;
  state->user_data = user_data;
  state->timepoint.callback = iree_async_semaphore_resolve_timepoint_callback;
  state->timepoint.user_data = state;
  return iree_async_semaphore_acquire_timepoint(semaphore, minimum_value,
                                                &state->timepoint);
}

static const iree_async_semaphore_vtable_t iree_async_semaphore_default_vtable =
    {
        .destroy = iree_async_semaphore_default_destroy,
        .query = iree_async_semaphore_default_query,
        .signal = iree_async_semaphore_default_signal,
        // on_fail: NULL — no hardware cleanup needed for software semaphores.
};

//===----------------------------------------------------------------------===//
// Semaphore linking (zero-allocation relay)
//===----------------------------------------------------------------------===//

// Built-in timepoint callback for semaphore links. Signals the target on
// success or propagates the failure status to the target.
static void iree_async_semaphore_link_callback(
    void* user_data, iree_async_semaphore_timepoint_t* base_timepoint,
    iree_status_t status) {
  (void)user_data;
  iree_async_semaphore_link_t* link =
      (iree_async_semaphore_link_t*)base_timepoint;
  if (iree_status_is_ok(status)) {
    // Relay signal to target. Each timeline value must be signaled exactly
    // once — if the target is already at or past signal_value, it indicates a
    // structural error (duplicate signal or misconfigured link). Propagate the
    // error by failing the target semaphore so waiters get a proper diagnostic.
    iree_status_t signal_status =
        iree_async_semaphore_signal(link->target, link->signal_value, NULL);
    if (IREE_UNLIKELY(!iree_status_is_ok(signal_status))) {
      iree_async_semaphore_fail(link->target, signal_status);
    }
  } else {
    // Source failed or was cancelled — propagate to target. Takes ownership of
    // status (stores via CAS or frees if target already failed).
    iree_async_semaphore_fail(link->target, status);
  }
}

IREE_API_EXPORT iree_status_t
iree_async_semaphore_link(iree_async_semaphore_t* source, uint64_t source_value,
                          iree_async_semaphore_t* target, uint64_t target_value,
                          iree_async_semaphore_link_t* out_link) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_link);
  out_link->target = target;
  out_link->signal_value = target_value;
  out_link->timepoint.callback = iree_async_semaphore_link_callback;
  out_link->timepoint.user_data = NULL;
  return iree_async_semaphore_acquire_timepoint(source, source_value,
                                                &out_link->timepoint);
}

IREE_API_EXPORT bool iree_async_semaphore_unlink(
    iree_async_semaphore_link_t* link) {
  IREE_ASSERT_ARGUMENT(link);
  return iree_async_semaphore_cancel_timepoint(link->timepoint.semaphore,
                                               &link->timepoint);
}

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
  // CAS loop to advance the timeline value monotonically.
  int64_t current_raw = 0;
  do {
    current_raw =
        iree_atomic_load(&semaphore->timeline_value, iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "semaphore already signaled past %" PRIu64,
                              value);
    }
  } while (!iree_atomic_compare_exchange_weak(
      &semaphore->timeline_value, &current_raw, (int64_t)value,
      iree_memory_order_release, iree_memory_order_relaxed));

  // Merge frontier and conditionally advance the untainted watermark.
  // The watermark is only advanced if the merge succeeds — on overflow the
  // frontier is left unchanged (valid lower bound) but the value stays
  // tainted so consumers fall back to device-side synchronization.
  bool frontier_merged = true;
  if (frontier != NULL && frontier->entry_count > 0) {
    iree_slim_mutex_lock(&semaphore->mutex);
    frontier_merged = iree_async_frontier_merge(
        semaphore->frontier, semaphore->frontier_capacity, frontier);
    iree_slim_mutex_unlock(&semaphore->mutex);
  }
  if (frontier_merged) {
    int64_t watermark_raw = 0;
    do {
      watermark_raw = iree_atomic_load(&semaphore->last_untainted_value,
                                       iree_memory_order_acquire);
      if (value <= (uint64_t)watermark_raw) break;
    } while (!iree_atomic_compare_exchange_weak(
        &semaphore->last_untainted_value, &watermark_raw, (int64_t)value,
        iree_memory_order_release, iree_memory_order_relaxed));
  }

  iree_async_semaphore_dispatch_timepoints(semaphore, value);

  return iree_ok_status();
}

IREE_API_EXPORT bool iree_async_semaphore_merge_frontier(
    iree_async_semaphore_t* semaphore, const iree_async_frontier_t* frontier) {
  IREE_ASSERT_ARGUMENT(frontier);
  if (frontier->entry_count == 0) return true;
  iree_slim_mutex_lock(&semaphore->mutex);
  bool merged = iree_async_frontier_merge(
      semaphore->frontier, semaphore->frontier_capacity, frontier);
  iree_slim_mutex_unlock(&semaphore->mutex);
  return merged;
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

  // CAS loop to advance the timeline value monotonically.
  // We store uint64 values as bit patterns in int64_t atomics. Comparisons
  // must cast BOTH sides to uint64_t to handle the full range correctly.
  int64_t current_raw = 0;
  do {
    current_raw =
        iree_atomic_load(&semaphore->timeline_value, iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "semaphore already signaled past %" PRIu64,
                              value);
    }
  } while (!iree_atomic_compare_exchange_weak(
      &semaphore->timeline_value, &current_raw, (int64_t)value,
      iree_memory_order_release, iree_memory_order_relaxed));

  // Merge frontier if provided. On overflow the frontier is left unchanged
  // (still a valid lower bound) — this is not an error because the frontier
  // is an acceleration structure, not a correctness requirement.
  if (frontier != NULL && frontier->entry_count > 0) {
    iree_slim_mutex_lock(&semaphore->mutex);
    iree_async_frontier_merge(semaphore->frontier, semaphore->frontier_capacity,
                              frontier);
    iree_slim_mutex_unlock(&semaphore->mutex);
  }

  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_semaphore_dispatch_timepoints(
    iree_async_semaphore_t* semaphore, uint64_t value) {
  // Phase 1: collect satisfied timepoints under lock.
  iree_async_semaphore_timepoint_t* satisfied_head = NULL;
  iree_async_semaphore_timepoint_t** satisfied_tail = &satisfied_head;

  iree_slim_mutex_lock(&semaphore->mutex);

  iree_async_semaphore_timepoint_t** prev_ptr = &semaphore->timepoints_head;
  iree_async_semaphore_timepoint_t* timepoint = semaphore->timepoints_head;

  while (timepoint != NULL) {
    iree_async_semaphore_timepoint_t* next = timepoint->next;

    if (timepoint->minimum_value <= value) {
      // Unlink from semaphore list.
      *prev_ptr = next;
      if (next != NULL) {
        next->prev = timepoint->prev;
      }
      // Append to satisfied list (preserves dispatch order).
      timepoint->next = NULL;
      timepoint->prev = NULL;
      *satisfied_tail = timepoint;
      satisfied_tail = &timepoint->next;
    } else {
      prev_ptr = &timepoint->next;
    }

    timepoint = next;
  }

  iree_slim_mutex_unlock(&semaphore->mutex);

  // Phase 2: dispatch without any lock held. Callbacks may signal other
  // semaphores, perform work, or do anything that would deadlock under lock.
  iree_async_semaphore_dispatch_detached(satisfied_head, iree_ok_status());
}

IREE_API_EXPORT void iree_async_semaphore_dispatch_timepoints_failed(
    iree_async_semaphore_t* semaphore, iree_status_t status) {
  // Borrows |status| as a template for per-timepoint clones; does NOT take
  // ownership (caller retains or has already stored it in failure_status).
  iree_slim_mutex_lock(&semaphore->mutex);
  iree_async_semaphore_timepoint_t* pending =
      iree_async_semaphore_detach_all_locked(semaphore);
  iree_slim_mutex_unlock(&semaphore->mutex);

  // Dispatch outside lock with failure status (cloned for each callback).
  iree_async_semaphore_dispatch_detached(pending, status);
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
  // Set to 1 after the callback completes its last touch of shared state.
  // The cleanup path spins on this flag for timepoints whose cancel failed
  // (already dispatched) to ensure the callback has finished before the
  // stack-allocated state is destroyed.
  iree_atomic_int32_t completed;
} iree_async_multi_wait_timepoint_t;

// Timepoint callback shared by all semaphores in a multi-wait.
// Performs atomic ops and a notification post, then marks itself completed
// so the cleanup path knows it is safe to destroy state.
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
  // actual condition (ALL vs ANY) after waking. This is the last touch of
  // shared state — after this point, only timepoint->completed is written.
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);

  // Mark completed so the cleanup path can safely destroy state. The cleanup
  // path spins on this flag for timepoints where cancel returned false
  // (already dispatched). We must not touch state after this store.
  iree_atomic_store(&timepoint->completed, 1, iree_memory_order_release);
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
    iree_timeout_t timeout, iree_async_wait_flags_t flags,
    iree_allocator_t allocator) {
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

  // Block until the wait condition is met.
  if (iree_status_is_ok(status)) {
    iree_async_multi_wait_condition_t condition = {
        .state = &state,
        .wait_mode = wait_mode,
        .initial_count = (int32_t)count,
    };

    // Check if the condition is already met (timepoints may have fired
    // synchronously during acquire_timepoint above).
    if (!iree_async_multi_wait_condition_fn(&condition)) {
      // Map wait flags to spin duration:
      //   ACTIVE: full spin (never enter kernel wait)
      //   YIELD:  brief spin to catch fast signals, then block
      //   NONE:   straight to futex (no spin)
      // ACTIVE takes precedence if both ACTIVE and YIELD are set.
      iree_duration_t spin_ns;
      if (iree_any_bit_set(flags, IREE_ASYNC_WAIT_FLAG_ACTIVE)) {
        spin_ns = IREE_DURATION_INFINITE;
      } else if (iree_any_bit_set(flags, IREE_ASYNC_WAIT_FLAG_YIELD)) {
        spin_ns = 500;  // ~500ns; tuned to avoid futex overhead on fast signals
      } else {
        spin_ns = IREE_DURATION_ZERO;
      }
      const iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
      bool satisfied = false;
      while (!satisfied) {
        iree_wait_token_t wait_token =
            iree_notification_prepare_wait(&state.notification);
        if (iree_async_multi_wait_condition_fn(&condition)) {
          iree_notification_cancel_wait(&state.notification);
          satisfied = true;
        } else if (!iree_notification_commit_wait(
                       &state.notification, wait_token, spin_ns, deadline_ns)) {
          break;  // Deadline expired.
        }
      }
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

  // Cancel any timepoints that haven't been dispatched yet. For timepoints
  // where cancel returns false (already dispatched), wait for the callback to
  // complete before destroying the stack-allocated state. The spin is bounded
  // by callback execution time (a few atomic ops + notification_post) and only
  // occurs when timeout races with dispatch.
  for (iree_host_size_t i = 0; i < registered_count; ++i) {
    if (timepoints[i].base.semaphore != NULL) {
      if (!iree_async_semaphore_cancel_timepoint(timepoints[i].base.semaphore,
                                                 &timepoints[i].base)) {
        // Callback dispatched — wait for it to finish.
        while (!iree_atomic_load(&timepoints[i].completed,
                                 iree_memory_order_acquire)) {
          iree_processor_yield();
        }
      }
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
