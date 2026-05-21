// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/semaphore.h"

#include "iree/base/threading/processor.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
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

  // Queue affinity provided at creation. When DEVICE_LOCAL is set this is the
  // complete set of queues that may legally signal or wait on the semaphore.
  iree_hal_queue_affinity_t queue_affinity;

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
    semaphore->queue_affinity = queue_affinity;
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

iree_hal_semaphore_flags_t iree_hal_amdgpu_semaphore_flags(
    iree_hal_semaphore_t* semaphore) {
  return ((const iree_hal_amdgpu_semaphore_t*)semaphore)->flags;
}

iree_hal_queue_affinity_t iree_hal_amdgpu_semaphore_queue_affinity(
    iree_hal_semaphore_t* semaphore) {
  return ((const iree_hal_amdgpu_semaphore_t*)semaphore)->queue_affinity;
}

bool iree_hal_amdgpu_semaphore_has_private_stream_semantics(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_amdgpu_logical_device_t* device) {
  if (!iree_hal_amdgpu_semaphore_is_local(semaphore, device)) return false;

  const iree_hal_semaphore_flags_t flags =
      iree_hal_amdgpu_semaphore_flags(semaphore);
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

static bool iree_hal_amdgpu_semaphore_epoch_is_reached(
    hsa_signal_value_t signal_value, hsa_signal_value_t compare_value) {
  return signal_value < compare_value;
}

static iree_status_t iree_hal_amdgpu_host_queue_epoch_wait_check_error(
    const iree_hal_amdgpu_host_queue_epoch_wait_t* wait_state) {
  iree_status_t error = (iree_status_t)iree_atomic_load(
      wait_state->error_status, iree_memory_order_acquire);
  return iree_status_is_ok(error) ? iree_ok_status() : iree_status_clone(error);
}

static uint64_t iree_hal_amdgpu_host_queue_epoch_wait_hint(
    const iree_hal_amdgpu_host_queue_epoch_wait_t* wait_state,
    iree_time_t deadline_ns) {
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    return wait_state->wait_timeout_hint;
  }

  const iree_time_t now_ns = iree_time_now();
  if (now_ns >= deadline_ns) return 0;

  const uint64_t remaining_ns = (uint64_t)(deadline_ns - now_ns);
  const uint64_t timestamp_frequency = wait_state->timestamp_frequency;
  uint64_t remaining_ticks = wait_state->wait_timeout_hint;
  if (timestamp_frequency != 0) {
    if (remaining_ns > UINT64_MAX / timestamp_frequency) {
      remaining_ticks = UINT64_MAX;
    } else {
      remaining_ticks =
          (remaining_ns * timestamp_frequency + 999999999ull) / 1000000000ull;
    }
  }
  if (remaining_ticks == 0) remaining_ticks = 1;
  return iree_min(remaining_ticks, wait_state->wait_timeout_hint);
}

static iree_status_t iree_hal_amdgpu_semaphore_wait_for_epoch(
    const iree_hal_amdgpu_host_queue_epoch_wait_t* wait_state,
    uint64_t producer_epoch, iree_timeout_t timeout,
    iree_async_wait_flags_t flags) {
  const hsa_signal_value_t compare_value =
      (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                           producer_epoch + 1);

  hsa_signal_value_t signal_value = iree_hsa_signal_load_scacquire(
      IREE_LIBHSA(wait_state->libhsa), wait_state->epoch_signal);
  if (iree_hal_amdgpu_semaphore_epoch_is_reached(signal_value, compare_value)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_epoch_wait_check_error(wait_state));
  if (iree_timeout_is_immediate(timeout)) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  const iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  const bool is_active_wait =
      iree_any_bit_set(flags, IREE_ASYNC_WAIT_FLAG_ACTIVE);
  const bool is_yield_wait =
      !is_active_wait && iree_any_bit_set(flags, IREE_ASYNC_WAIT_FLAG_YIELD);

  if (is_active_wait || is_yield_wait) {
    const iree_time_t spin_deadline_ns =
        is_active_wait ? IREE_TIME_INFINITE_FUTURE : iree_time_now() + 500;
    for (;;) {
      signal_value = iree_hsa_signal_load_scacquire(
          IREE_LIBHSA(wait_state->libhsa), wait_state->epoch_signal);
      if (iree_hal_amdgpu_semaphore_epoch_is_reached(signal_value,
                                                     compare_value)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_host_queue_epoch_wait_check_error(wait_state));
      const iree_time_t now_ns =
          deadline_ns == IREE_TIME_INFINITE_FUTURE ? 0 : iree_time_now();
      if (deadline_ns != IREE_TIME_INFINITE_FUTURE && now_ns >= deadline_ns) {
        return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
      if (is_yield_wait && iree_time_now() >= spin_deadline_ns) break;
      iree_processor_yield();
    }
  }

  for (;;) {
    const uint64_t wait_hint =
        iree_hal_amdgpu_host_queue_epoch_wait_hint(wait_state, deadline_ns);
    if (wait_hint == 0) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    signal_value = iree_hsa_signal_wait_scacquire(
        IREE_LIBHSA(wait_state->libhsa), wait_state->epoch_signal,
        HSA_SIGNAL_CONDITION_LT, compare_value, wait_hint,
        HSA_WAIT_STATE_BLOCKED);
    if (iree_hal_amdgpu_semaphore_epoch_is_reached(signal_value,
                                                   compare_value)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_epoch_wait_check_error(wait_state));
  }
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

  iree_hal_amdgpu_last_signal_flags_t last_signal_flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
  iree_async_axis_t producer_axis = 0;
  uint64_t producer_epoch = 0;
  uint64_t producer_value = 0;
  if (iree_hal_amdgpu_last_signal_load(&semaphore->last_signal,
                                       &last_signal_flags, &producer_axis,
                                       &producer_epoch, &producer_value) &&
      iree_all_bits_set(
          last_signal_flags,
          IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT) &&
      producer_value >= value) {
    iree_hal_amdgpu_host_queue_epoch_wait_t wait_state;
    if (iree_hal_amdgpu_logical_device_lookup_host_queue_epoch_wait(
            semaphore->device, producer_axis, &wait_state)) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_semaphore_wait_for_epoch(
          &wait_state, producer_epoch, timeout, flags));
      iree_hal_amdgpu_host_queue_drain_completions_for_waiter(
          wait_state.host_queue);
      return iree_async_semaphore_publish_untainted(&semaphore->async,
                                                    producer_value, NULL);
    }
  }

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
