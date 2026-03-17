// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/semaphore.h"

#include "iree/base/threading/mutex.h"

static const iree_hal_semaphore_vtable_t
    iree_hal_remote_client_semaphore_vtable;

typedef struct iree_hal_remote_client_semaphore_t {
  // Embedded at offset 0 for toll-free bridging with iree_async_semaphore_t.
  iree_async_semaphore_t async;
  iree_allocator_t host_allocator;
  // Maps signal values to submission epochs for wait frontier encoding.
  // Sorted by value (monotonically increasing — HAL semaphore values are
  // monotonic). Populated during signal waiter registration; queried when
  // building wait frontiers for subsequent queue operations.
  //
  // Protected by epoch_map_mutex: record_epoch (write) and lookup_epoch
  // (read) can race between the app thread (immediate submits) and the
  // proactor thread (deferred submit callbacks).
  iree_slim_mutex_t epoch_map_mutex;
  struct {
    uint64_t* values;
    uint64_t* epochs;
    iree_async_axis_t* axes;
    iree_host_size_t count;
    iree_host_size_t capacity;
  } epoch_map;
} iree_hal_remote_client_semaphore_t;

static iree_hal_remote_client_semaphore_t*
iree_hal_remote_client_semaphore_cast(iree_hal_semaphore_t* base_semaphore) {
  IREE_HAL_ASSERT_TYPE(base_semaphore,
                       &iree_hal_remote_client_semaphore_vtable);
  return (iree_hal_remote_client_semaphore_t*)base_semaphore;
}

iree_status_t iree_hal_remote_client_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  // Compute allocation layout: struct + trailing frontier storage.
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_semaphore_layout(
              sizeof(iree_hal_remote_client_semaphore_t),
              /*frontier_capacity=*/0, &frontier_offset, &total_size));

  iree_hal_remote_client_semaphore_t* semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore));
  memset(semaphore, 0, total_size);

  // Initialize the embedded async semaphore (sets ref_count, vtable, etc.).
  iree_async_semaphore_initialize(
      (const iree_async_semaphore_vtable_t*)&iree_hal_remote_client_semaphore_vtable,
      proactor, initial_value, frontier_offset, /*frontier_capacity=*/0,
      &semaphore->async);

  semaphore->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&semaphore->epoch_map_mutex);

  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_remote_client_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_remote_client_semaphore_t* semaphore =
      iree_hal_remote_client_semaphore_cast(
          iree_hal_semaphore_cast(base_semaphore));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = semaphore->host_allocator;

  // Deinitialize the async semaphore FIRST — it flushes/fails any pending
  // timepoints, which may fire callbacks that access other semaphores'
  // epoch_maps. This must complete before we tear down our own subclass state.
  iree_async_semaphore_deinitialize(&semaphore->async);

  // Now safe to tear down subclass state: no callbacks can reach us.
  iree_slim_mutex_deinitialize(&semaphore->epoch_map_mutex);
  iree_allocator_free(host_allocator, semaphore->epoch_map.values);
  iree_allocator_free(host_allocator, semaphore->epoch_map.epochs);
  iree_allocator_free(host_allocator, semaphore->epoch_map.axes);

  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static uint64_t iree_hal_remote_client_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_remote_client_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  // Advance the timeline and merge the frontier. This is the shared helper
  // that handles the CAS loop, monotonicity check, and frontier merge.
  IREE_RETURN_IF_ERROR(iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier));

  // Dispatch any timepoints that are now satisfied.
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);

  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  // Delegate to the centralized async semaphore wait which uses a stack-local
  // futex-based notification. Same pattern as task_semaphore_wait.
  iree_async_semaphore_t* async_semaphore =
      (iree_async_semaphore_t*)base_semaphore;
  return iree_async_semaphore_multi_wait(IREE_ASYNC_WAIT_MODE_ALL,
                                         &async_semaphore, &value, 1, timeout,
                                         flags, iree_allocator_system());
}

static iree_status_t iree_hal_remote_client_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import not supported on remote client "
                          "semaphore");
}

static iree_status_t iree_hal_remote_client_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export not supported on remote client "
                          "semaphore");
}

//===----------------------------------------------------------------------===//
// Epoch mapping
//===----------------------------------------------------------------------===//

void iree_hal_remote_client_semaphore_record_epoch(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_async_axis_t axis, uint64_t epoch) {
  iree_hal_remote_client_semaphore_t* semaphore =
      iree_hal_remote_client_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->epoch_map_mutex);

  // Grow all three parallel arrays together when capacity is exceeded.
  iree_host_size_t minimum_capacity = semaphore->epoch_map.count + 1;
  if (minimum_capacity > semaphore->epoch_map.capacity) {
    iree_host_size_t values_capacity = semaphore->epoch_map.capacity;
    iree_host_size_t epochs_capacity = semaphore->epoch_map.capacity;
    iree_host_size_t axes_capacity = semaphore->epoch_map.capacity;
    iree_status_t status = iree_allocator_grow_array(
        semaphore->host_allocator, minimum_capacity, sizeof(uint64_t),
        &values_capacity, (void**)&semaphore->epoch_map.values);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_grow_array(
          semaphore->host_allocator, minimum_capacity, sizeof(uint64_t),
          &epochs_capacity, (void**)&semaphore->epoch_map.epochs);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_grow_array(semaphore->host_allocator, minimum_capacity,
                                    sizeof(iree_async_axis_t), &axes_capacity,
                                    (void**)&semaphore->epoch_map.axes);
    }
    if (!iree_status_is_ok(status)) {
      // Allocation failure during epoch recording is non-fatal: the
      // semaphore will still work, but wait frontier encoding for this
      // value will fall back to the deferred send path.
      iree_status_ignore(status);
      iree_slim_mutex_unlock(&semaphore->epoch_map_mutex);
      return;
    }
    // Use the minimum capacity across all three arrays.
    semaphore->epoch_map.capacity =
        iree_min(values_capacity, iree_min(epochs_capacity, axes_capacity));
  }

  // Append at end. Values are monotonically increasing (HAL invariant).
  iree_host_size_t index = semaphore->epoch_map.count++;
  semaphore->epoch_map.values[index] = value;
  semaphore->epoch_map.epochs[index] = epoch;
  semaphore->epoch_map.axes[index] = axis;

  iree_slim_mutex_unlock(&semaphore->epoch_map_mutex);
}

bool iree_hal_remote_client_semaphore_lookup_epoch(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_async_axis_t* out_axis, uint64_t* out_epoch) {
  iree_hal_remote_client_semaphore_t* semaphore =
      iree_hal_remote_client_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->epoch_map_mutex);

  if (semaphore->epoch_map.count == 0) {
    iree_slim_mutex_unlock(&semaphore->epoch_map_mutex);
    return false;
  }

  // Binary search for the first recorded signal value >= |value|.
  // The values array is sorted (monotonically increasing).
  iree_host_size_t low = 0;
  iree_host_size_t high = semaphore->epoch_map.count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    if (semaphore->epoch_map.values[mid] < value) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  if (low >= semaphore->epoch_map.count) {
    iree_slim_mutex_unlock(&semaphore->epoch_map_mutex);
    return false;
  }

  *out_axis = semaphore->epoch_map.axes[low];
  *out_epoch = semaphore->epoch_map.epochs[low];
  iree_slim_mutex_unlock(&semaphore->epoch_map_mutex);
  return true;
}

static const iree_hal_semaphore_vtable_t
    iree_hal_remote_client_semaphore_vtable = {
        .async =
            {
                .destroy = iree_hal_remote_client_semaphore_destroy,
                .query = iree_hal_remote_client_semaphore_query,
                .signal = iree_hal_remote_client_semaphore_signal,
            },
        .wait = iree_hal_remote_client_semaphore_wait,
        .import_timepoint = iree_hal_remote_client_semaphore_import_timepoint,
        .export_timepoint = iree_hal_remote_client_semaphore_export_timepoint,
};
