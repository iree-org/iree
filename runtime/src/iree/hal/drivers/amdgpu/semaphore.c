// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/semaphore.h"

#include "iree/hal/drivers/amdgpu/util/notification_ring.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_semaphore_t {
  // Embedded async semaphore at offset 0 for toll-free bridging.
  iree_async_semaphore_t async;

  // Allocator used to free this semaphore.
  iree_allocator_t host_allocator;

  // Back-pointer to the logical device that created this semaphore.
  // Used for type discrimination (is_local check). Not retained.
  iree_hal_amdgpu_logical_device_t* device;

  // Creation flags controlling synchronization behavior.
  iree_hal_semaphore_flags_t flags;

  // Seqlock-protected cache of the most recent signal from a queue.
  // Updated by the submission path when queue_execute signals this semaphore.
  // Read by the submission path for same-queue FIFO elision, cross-queue
  // epoch lookup, and by the host-wait fast path for direct signal waits.
  // Initialized to zero (flags=0) — no valid signal has been recorded yet.
  iree_hal_amdgpu_last_signal_t last_signal;
} iree_hal_amdgpu_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_amdgpu_semaphore_vtable;

static iree_hal_amdgpu_semaphore_t* iree_hal_amdgpu_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_semaphore_vtable);
  return (iree_hal_amdgpu_semaphore_t*)base_value;
}

iree_status_t iree_hal_amdgpu_semaphore_create(
    iree_hal_amdgpu_logical_device_t* device, iree_async_proactor_t* proactor,
    iree_hal_queue_affinity_t queue_affinity, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0, total_size = 0;
  // Match the queue frontier/snapshot capacity so publishing a full queue
  // frontier into a semaphore does not overflow just because the semaphore was
  // allocated with a narrower async default.
  iree_status_t status = iree_async_semaphore_layout(
      sizeof(*semaphore), IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT,
      &frontier_offset, &total_size);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_async_semaphore_initialize(
        (const iree_async_semaphore_vtable_t*)&iree_hal_amdgpu_semaphore_vtable,
        proactor, initial_value, frontier_offset,
        IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT, &semaphore->async);
    semaphore->host_allocator = host_allocator;
    semaphore->device = device;
    semaphore->flags = flags;
    memset(&semaphore->last_signal, 0, sizeof(semaphore->last_signal));
    *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_amdgpu_semaphore_t* semaphore =
      iree_hal_amdgpu_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_amdgpu_semaphore_vtable);
}

bool iree_hal_amdgpu_semaphore_is_local(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_amdgpu_logical_device_t* device) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_amdgpu_semaphore_vtable) &&
         ((const iree_hal_amdgpu_semaphore_t*)semaphore)->device == device;
}

bool iree_hal_amdgpu_semaphore_has_private_stream_semantics(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_amdgpu_logical_device_t* device) {
  if (!iree_hal_amdgpu_semaphore_is_local(semaphore, device)) return false;

  const iree_hal_semaphore_flags_t flags =
      ((const iree_hal_amdgpu_semaphore_t*)semaphore)->flags;
  const iree_hal_semaphore_flags_t required_flags =
      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
      IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER;
  const iree_hal_semaphore_flags_t public_flags =
      IREE_HAL_SEMAPHORE_FLAG_HOST_INTERRUPT |
      IREE_HAL_SEMAPHORE_FLAG_EXPORTABLE |
      IREE_HAL_SEMAPHORE_FLAG_EXPORTABLE_TIMEPOINTS;
  return iree_all_bits_set(flags, required_flags) &&
         !iree_any_bit_set(flags, public_flags);
}

iree_hal_amdgpu_last_signal_t* iree_hal_amdgpu_semaphore_last_signal(
    iree_hal_semaphore_t* semaphore) {
  return &((iree_hal_amdgpu_semaphore_t*)semaphore)->last_signal;
}

bool iree_hal_amdgpu_semaphore_publish_signal(
    iree_hal_semaphore_t* base_semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier, uint64_t producer_epoch,
    uint64_t producer_value) {
  IREE_ASSERT_ARGUMENT(producer_frontier);
  iree_hal_amdgpu_semaphore_t* semaphore =
      iree_hal_amdgpu_semaphore_cast(base_semaphore);

  iree_hal_amdgpu_last_signal_flags_t flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID;
  bool source_dominates_frontier = false;
  iree_slim_mutex_lock(&semaphore->async.mutex);
  bool merged = iree_async_frontier_merge_and_test_source_dominance(
      semaphore->async.frontier, semaphore->async.frontier_capacity,
      producer_frontier, &source_dominates_frontier);
  if (merged && source_dominates_frontier) {
    flags |= IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT;
  }
  iree_hal_amdgpu_last_signal_store(
      &semaphore->last_signal, merged ? flags : 0,
      merged ? producer_axis : (iree_async_axis_t)0,
      merged ? producer_epoch : 0, merged ? producer_value : 0);
  iree_slim_mutex_unlock(&semaphore->async.mutex);

  return merged;
}

void iree_hal_amdgpu_semaphore_publish_private_stream_signal(
    iree_hal_semaphore_t* base_semaphore, iree_async_axis_t producer_axis,
    uint64_t producer_epoch, uint64_t producer_value) {
  iree_hal_amdgpu_semaphore_t* semaphore =
      iree_hal_amdgpu_semaphore_cast(base_semaphore);
  iree_hal_amdgpu_last_signal_store(
      &semaphore->last_signal,
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID |
          IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT,
      producer_axis, producer_epoch, producer_value);
}

void iree_hal_amdgpu_semaphore_clear_last_signal(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_amdgpu_semaphore_t* semaphore =
      iree_hal_amdgpu_semaphore_cast(base_semaphore);
  iree_slim_mutex_lock(&semaphore->async.mutex);
  iree_hal_amdgpu_last_signal_store(&semaphore->last_signal,
                                    IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE,
                                    (iree_async_axis_t)0, 0, 0);
  iree_slim_mutex_unlock(&semaphore->async.mutex);
}

static uint64_t iree_hal_amdgpu_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  // Both fields are atomic — fully lock-free query.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_amdgpu_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  // Advance the timeline (CAS) and merge frontier.
  iree_status_t status = iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier);
  if (!iree_status_is_ok(status)) return status;

  // Dispatch satisfied timepoints.
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  iree_hal_amdgpu_semaphore_t* semaphore =
      iree_hal_amdgpu_semaphore_cast(base_semaphore);

  // Fast check: already reached or failed? Lock-free atomic load.
  // Failure check must come first: failure values are numerically larger than
  // any valid timeline value and would falsely satisfy the >= check.
  uint64_t current = iree_async_semaphore_query(&semaphore->async);
  if (current >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    return iree_hal_semaphore_failure_as_status(current);
  }
  if (current >= value) return iree_ok_status();

  // Epoch fast path: if we have a cached epoch for a value >= target, we can
  // wait directly on the queue's epoch signal instead of going through the
  // timepoint/notification machinery. This is the minimum-latency path for
  // "submit work, wait for result."
  //
  // The epoch signal and hsa_signal_wait_scacquire call will be wired here
  // when the per-queue epoch signal is implemented. Until then, we fall
  // through to the software path which is functionally identical.
  //
  // iree_async_axis_t producer_axis = 0;
  // iree_hal_amdgpu_last_signal_flags_t last_signal_flags = 0;
  // uint64_t epoch = 0, cached_value = 0;
  // if (iree_hal_amdgpu_last_signal_load(&semaphore->last_signal,
  //                                       &last_signal_flags, &producer_axis,
  //                                       &epoch, &cached_value)) {
  //   if (cached_value >= value) {
  //     // Wait directly on producer_axis' epoch signal with LT condition.
  //   }
  // }

  // Software fallback: timepoint-based blocking wait.
  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t**)&base_semaphore,
      &value, 1, timeout, flags, iree_allocator_system());
}

static iree_status_t iree_hal_amdgpu_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU timepoint import not yet implemented");
}

static iree_status_t iree_hal_amdgpu_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU timepoint export not yet implemented");
}

static const iree_hal_semaphore_vtable_t iree_hal_amdgpu_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_amdgpu_semaphore_destroy,
            .query = iree_hal_amdgpu_semaphore_query,
            .signal = iree_hal_amdgpu_semaphore_signal,
        },
    .wait = iree_hal_amdgpu_semaphore_wait,
    .import_timepoint = iree_hal_amdgpu_semaphore_import_timepoint,
    .export_timepoint = iree_hal_amdgpu_semaphore_export_timepoint,
};
