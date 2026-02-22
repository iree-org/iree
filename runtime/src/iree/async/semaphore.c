// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/semaphore.h"

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

//===----------------------------------------------------------------------===//
// Software semaphore
//===----------------------------------------------------------------------===//

// Software-only semaphore with inline frontier storage.
// Allocated as a single block: struct + frontier entries.
typedef struct iree_async_semaphore_software_t {
  // Base must be first for toll-free casting.
  iree_async_semaphore_t base;

  // Allocator used for this semaphore (stored for deallocation).
  iree_allocator_t allocator;

  // Current timeline value. Monotonically increasing.
  // Uses int64_t for safe CAS operations (negative values not used).
  iree_atomic_int64_t timeline_value;

  // Untainted watermark. Values <= this are known to have been signaled by
  // witnessed work. Values > this may be tainted (from imports/IPC).
  iree_atomic_int64_t last_untainted_value;

  // Protects failure_status and timepoints_head.
  // Also used for dispatch-under-lock timepoint callbacks.
  iree_slim_mutex_t mutex;

  // Sticky failure status. Once set, all waiters receive this status.
  // Owned by the semaphore (freed on deinitialize).
  iree_status_t failure_status;

  // Doubly-linked list of pending timepoints.
  iree_async_semaphore_timepoint_t* timepoints_head;

  // Maximum number of frontier entries the trailing storage can hold.
  uint8_t frontier_capacity;

  // Trailing storage: iree_async_frontier_t header + entries.
  // Access via iree_async_semaphore_software_frontier().
  // The frontier header (entry_count + reserved) is followed by entries[].
} iree_async_semaphore_software_t;

// Returns a pointer to the inline frontier (header + entries as trailing
// storage). The frontier is valid for up to frontier_capacity entries.
static inline iree_async_frontier_t* iree_async_semaphore_software_frontier(
    iree_async_semaphore_software_t* semaphore) {
  return (iree_async_frontier_t*)(semaphore + 1);
}

// Computes the total allocation size for a software semaphore with the given
// frontier capacity using overflow-checked arithmetic.
// Trailing storage: iree_async_frontier_t header + entries[frontier_capacity].
static inline iree_status_t iree_async_semaphore_software_size(
    uint8_t frontier_capacity, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_semaphore_software_t), out_size,
      IREE_STRUCT_FIELD_FAM(1, iree_async_frontier_t),
      IREE_STRUCT_FIELD_FAM(frontier_capacity, iree_async_frontier_entry_t));
}

// Tentative definition; full definition at end of software semaphore section.
static const iree_async_semaphore_vtable_t iree_async_semaphore_software_vtable;

IREE_API_EXPORT void iree_async_semaphore_initialize(
    const iree_async_semaphore_vtable_t* vtable,
    iree_async_semaphore_t* out_semaphore) {
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  iree_atomic_ref_count_init(&out_semaphore->ref_count);
  out_semaphore->vtable = vtable;
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

IREE_API_EXPORT iree_status_t iree_async_semaphore_create_software(
    uint64_t initial_value, uint8_t frontier_capacity,
    iree_allocator_t allocator, iree_async_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_semaphore_software_size(frontier_capacity, &total_size));
  iree_async_semaphore_software_t* semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&semaphore));
  memset(semaphore, 0, total_size);

  iree_async_semaphore_initialize(&iree_async_semaphore_software_vtable,
                                  &semaphore->base);
  semaphore->allocator = allocator;
  iree_atomic_store(&semaphore->timeline_value, (int64_t)initial_value,
                    iree_memory_order_release);
  iree_atomic_store(&semaphore->last_untainted_value, (int64_t)initial_value,
                    iree_memory_order_release);
  iree_slim_mutex_initialize(&semaphore->mutex);
  semaphore->failure_status = iree_ok_status();
  semaphore->timepoints_head = NULL;
  semaphore->frontier_capacity = frontier_capacity;

  // Initialize the inline frontier (trailing storage).
  iree_async_frontier_t* frontier =
      iree_async_semaphore_software_frontier(semaphore);
  iree_async_frontier_initialize(frontier, 0);

  *out_semaphore = &semaphore->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_semaphore_software_destroy(
    iree_async_semaphore_t* base_semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_semaphore_software_t* semaphore =
      (iree_async_semaphore_software_t*)base_semaphore;

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

  // Free failure status if set.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    iree_status_free(semaphore->failure_status);
  }

  iree_slim_mutex_deinitialize(&semaphore->mutex);

  iree_allocator_t allocator = semaphore->allocator;
  iree_allocator_free(allocator, semaphore);
  IREE_TRACE_ZONE_END(z0);
}

static uint64_t iree_async_semaphore_software_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_async_semaphore_software_t* semaphore =
      (iree_async_semaphore_software_t*)base_semaphore;
  return (uint64_t)iree_atomic_load(&semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_async_semaphore_software_signal_impl(
    iree_async_semaphore_t* base_semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  iree_async_semaphore_software_t* semaphore =
      (iree_async_semaphore_software_t*)base_semaphore;

  // CAS loop to advance the timeline value monotonically.
  // We store uint64 values as bit patterns in int64_t atomics. Comparisons
  // must cast BOTH sides to uint64_t to handle the full range correctly.
  int64_t current_raw = 0;
  do {
    current_raw =
        iree_atomic_load(&semaphore->timeline_value, iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "signal value %" PRIu64
                              " must be greater than current value %" PRIu64,
                              value, (uint64_t)current_raw);
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
    iree_async_frontier_t* accumulated_frontier =
        iree_async_semaphore_software_frontier(semaphore);
    merge_status = iree_async_frontier_merge(
        accumulated_frontier, semaphore->frontier_capacity, frontier);
    iree_slim_mutex_unlock(&semaphore->mutex);
    // Note: we continue to dispatch even on merge failure because the timeline
    // has already been advanced. Waiters must be woken to avoid hangs.
  }

  // Dispatch satisfied timepoints. This must happen even if merge failed
  // because the timeline value has already been published via CAS.
  iree_async_semaphore_dispatch_timepoints(base_semaphore, value);

  return merge_status;
}

static uint8_t iree_async_semaphore_software_query_frontier(
    iree_async_semaphore_t* base_semaphore, iree_async_frontier_t* out_frontier,
    uint8_t capacity) {
  iree_async_semaphore_software_t* semaphore =
      (iree_async_semaphore_software_t*)base_semaphore;

  iree_slim_mutex_lock(&semaphore->mutex);
  iree_async_frontier_t* accumulated_frontier =
      iree_async_semaphore_software_frontier(semaphore);
  uint8_t actual_count = accumulated_frontier->entry_count;
  if (out_frontier != NULL) {
    uint8_t copy_count = actual_count < capacity ? actual_count : capacity;
    iree_async_frontier_initialize(out_frontier, copy_count);
    memcpy(out_frontier->entries, accumulated_frontier->entries,
           copy_count * sizeof(iree_async_frontier_entry_t));
  }
  iree_slim_mutex_unlock(&semaphore->mutex);

  return actual_count;
}

static void iree_async_semaphore_software_fail(
    iree_async_semaphore_t* base_semaphore, iree_status_t status) {
  iree_async_semaphore_software_t* semaphore =
      (iree_async_semaphore_software_t*)base_semaphore;

  iree_slim_mutex_lock(&semaphore->mutex);

  // First failure wins.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    iree_status_free(status);
    return;
  }

  // Store the failure status.
  semaphore->failure_status = status;

  // Dispatch all pending timepoints with the failure status.
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

static iree_status_t iree_async_semaphore_software_acquire_timepoint(
    iree_async_semaphore_t* base_semaphore, uint64_t minimum_value,
    iree_async_semaphore_timepoint_t* timepoint) {
  iree_async_semaphore_software_t* semaphore =
      (iree_async_semaphore_software_t*)base_semaphore;

  // Initialize timepoint fields.
  timepoint->semaphore = base_semaphore;
  timepoint->minimum_value = minimum_value;
  timepoint->next = NULL;
  timepoint->prev = NULL;

  iree_slim_mutex_lock(&semaphore->mutex);

  // Check for immediate failure.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    iree_status_t failure = iree_status_clone(semaphore->failure_status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    timepoint->callback(timepoint->user_data, timepoint, failure);
    return iree_ok_status();
  }

  // Check for immediate satisfaction.
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &semaphore->timeline_value, iree_memory_order_acquire);
  if (current_value >= minimum_value) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    timepoint->callback(timepoint->user_data, timepoint, iree_ok_status());
    return iree_ok_status();
  }

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

  // Also re-check failure status.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Unlink and dispatch with failure.
    if (timepoint->next != NULL) {
      timepoint->next->prev = timepoint->prev;
    }
    if (timepoint->prev != NULL) {
      timepoint->prev->next = timepoint->next;
    } else {
      semaphore->timepoints_head = timepoint->next;
    }
    iree_status_t failure = iree_status_clone(semaphore->failure_status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    timepoint->callback(timepoint->user_data, timepoint, failure);
    return iree_ok_status();
  }

  iree_slim_mutex_unlock(&semaphore->mutex);
  return iree_ok_status();
}

static void iree_async_semaphore_software_cancel_timepoint(
    iree_async_semaphore_t* base_semaphore,
    iree_async_semaphore_timepoint_t* timepoint) {
  iree_async_semaphore_software_t* semaphore =
      (iree_async_semaphore_software_t*)base_semaphore;

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

static iree_status_t iree_async_semaphore_software_export_primitive(
    iree_async_semaphore_t* base_semaphore, uint64_t minimum_value,
    iree_async_primitive_t* out_primitive) {
  // Software semaphores cannot export pollable primitives.
  // Callers should use the timepoint callback mechanism instead.
  (void)base_semaphore;
  (void)minimum_value;
  (void)out_primitive;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "software semaphores do not support primitive export");
}

static const iree_async_semaphore_vtable_t
    iree_async_semaphore_software_vtable = {
        .destroy = iree_async_semaphore_software_destroy,
        .query = iree_async_semaphore_software_query,
        .signal = iree_async_semaphore_software_signal_impl,
        .query_frontier = iree_async_semaphore_software_query_frontier,
        .fail = iree_async_semaphore_software_fail,
        .acquire_timepoint = iree_async_semaphore_software_acquire_timepoint,
        .cancel_timepoint = iree_async_semaphore_software_cancel_timepoint,
        .export_primitive = iree_async_semaphore_software_export_primitive,
};

//===----------------------------------------------------------------------===//
// Tainting API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT bool iree_async_semaphore_is_value_tainted(
    iree_async_semaphore_t* semaphore, uint64_t value) {
  // For now, only software semaphores are supported.
  // HAL semaphores will also embed this state and can use the same logic.
  iree_async_semaphore_software_t* sw =
      (iree_async_semaphore_software_t*)semaphore;
  uint64_t untainted = (uint64_t)iree_atomic_load(&sw->last_untainted_value,
                                                  iree_memory_order_acquire);
  return value > untainted;
}

IREE_API_EXPORT void iree_async_semaphore_mark_tainted_above(
    iree_async_semaphore_t* semaphore, uint64_t threshold) {
  iree_async_semaphore_software_t* sw =
      (iree_async_semaphore_software_t*)semaphore;
  // The watermark only decreases (marking more values as tainted).
  // Use CAS loop to ensure we never increase the watermark.
  // We store uint64 values as bit patterns in int64_t atomics.
  int64_t current_raw = 0;
  do {
    current_raw =
        iree_atomic_load(&sw->last_untainted_value, iree_memory_order_acquire);
    if (threshold >= (uint64_t)current_raw) {
      return;  // Already at or below threshold, nothing to do.
    }
  } while (!iree_atomic_compare_exchange_weak(
      &sw->last_untainted_value, &current_raw, (int64_t)threshold,
      iree_memory_order_release, iree_memory_order_relaxed));
}

IREE_API_EXPORT iree_status_t iree_async_semaphore_signal_untainted(
    iree_async_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  // Signal the semaphore normally.
  IREE_RETURN_IF_ERROR(iree_async_semaphore_signal(semaphore, value, frontier));

  // Advance the untainted watermark.
  iree_async_semaphore_software_t* sw =
      (iree_async_semaphore_software_t*)semaphore;
  // Use CAS loop to ensure monotonic advancement.
  // We store uint64 values as bit patterns in int64_t atomics.
  int64_t current_raw = 0;
  do {
    current_raw =
        iree_atomic_load(&sw->last_untainted_value, iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) {
      return iree_ok_status();  // Already at or past this value.
    }
  } while (!iree_atomic_compare_exchange_weak(
      &sw->last_untainted_value, &current_raw, (int64_t)value,
      iree_memory_order_release, iree_memory_order_relaxed));

  return iree_ok_status();
}

IREE_API_EXPORT uint64_t
iree_async_semaphore_query_untainted_value(iree_async_semaphore_t* semaphore) {
  iree_async_semaphore_software_t* sw =
      (iree_async_semaphore_software_t*)semaphore;
  return (uint64_t)iree_atomic_load(&sw->last_untainted_value,
                                    iree_memory_order_acquire);
}

//===----------------------------------------------------------------------===//
// HAL composition helpers
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_semaphore_software_signal(
    iree_async_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  // This is the same as the software signal implementation, but does NOT
  // dispatch timepoints. HAL implementations call this first, then signal
  // their native primitive, then call dispatch_timepoints.
  //
  // IMPORTANT: Callers MUST call dispatch_timepoints() regardless of whether
  // this function returns success or failure, because the timeline value is
  // updated before the frontier merge is attempted. Failure to dispatch would
  // leave waiters stuck even though the timeline has advanced.
  iree_async_semaphore_software_t* sw =
      (iree_async_semaphore_software_t*)semaphore;

  // CAS loop to advance the timeline value monotonically.
  // We store uint64 values as bit patterns in int64_t atomics. Comparisons
  // must cast BOTH sides to uint64_t to handle the full range correctly.
  int64_t current_raw = 0;
  do {
    current_raw =
        iree_atomic_load(&sw->timeline_value, iree_memory_order_acquire);
    if (value <= (uint64_t)current_raw) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "signal value %" PRIu64
                              " must be greater than current value %" PRIu64,
                              value, (uint64_t)current_raw);
    }
  } while (!iree_atomic_compare_exchange_weak(
      &sw->timeline_value, &current_raw, (int64_t)value,
      iree_memory_order_release, iree_memory_order_relaxed));

  // Merge frontier if provided. Note: merge failure does not prevent the
  // signal from taking effect - the timeline has already been advanced.
  // Callers should still dispatch timepoints even if this returns an error.
  if (frontier != NULL && frontier->entry_count > 0) {
    iree_slim_mutex_lock(&sw->mutex);
    iree_async_frontier_t* accumulated_frontier =
        iree_async_semaphore_software_frontier(sw);
    iree_status_t merge_status = iree_async_frontier_merge(
        accumulated_frontier, sw->frontier_capacity, frontier);
    iree_slim_mutex_unlock(&sw->mutex);
    return merge_status;  // Return merge status but signal has taken effect.
  }

  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_semaphore_dispatch_timepoints(
    iree_async_semaphore_t* semaphore, uint64_t value) {
  iree_async_semaphore_software_t* sw =
      (iree_async_semaphore_software_t*)semaphore;

  iree_slim_mutex_lock(&sw->mutex);

  // Walk the list and dispatch all satisfied timepoints.
  iree_async_semaphore_timepoint_t** prev_ptr = &sw->timepoints_head;
  iree_async_semaphore_timepoint_t* timepoint = sw->timepoints_head;

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

  iree_slim_mutex_unlock(&sw->mutex);
}

IREE_API_EXPORT void iree_async_semaphore_dispatch_timepoints_failed(
    iree_async_semaphore_t* semaphore, iree_status_t status) {
  iree_async_semaphore_software_t* sw =
      (iree_async_semaphore_software_t*)semaphore;

  iree_slim_mutex_lock(&sw->mutex);

  // Walk the list and dispatch all timepoints with the failure status.
  iree_async_semaphore_timepoint_t* timepoint = sw->timepoints_head;
  while (timepoint != NULL) {
    iree_async_semaphore_timepoint_t* next = timepoint->next;
    timepoint->callback(timepoint->user_data, timepoint,
                        iree_status_clone(status));
    timepoint = next;
  }
  sw->timepoints_head = NULL;

  iree_slim_mutex_unlock(&sw->mutex);

  // Free the original status.
  iree_status_free(status);
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
