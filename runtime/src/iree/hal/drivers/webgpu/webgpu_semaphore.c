// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_semaphore.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_semaphore_t {
  iree_async_semaphore_t async;
  iree_allocator_t host_allocator;
  // Submitted signal provenance for FIFO wait elision. Records the axis and
  // value of the most recent signal that has been submitted to the GPU queue
  // but may not have completed yet. Same-queue fast paths check these fields
  // to determine if GPU FIFO ordering guarantees a wait will be satisfied
  // without needing an async proactor wait.
  iree_async_axis_t submitted_signal_axis;
  iree_atomic_int64_t submitted_signal_value;
} iree_hal_webgpu_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_webgpu_semaphore_vtable;

static iree_hal_webgpu_semaphore_t* iree_hal_webgpu_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_semaphore_vtable);
  return (iree_hal_webgpu_semaphore_t*)base_value;
}

iree_status_t iree_hal_webgpu_semaphore_create(
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  iree_hal_webgpu_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0, total_size = 0;
  // Frontier capacity 1: single queue axis. Signals carry a frontier recording
  // the queue's axis/epoch, enabling cross-device causal ordering.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_semaphore_layout(sizeof(*semaphore),
                                      /*frontier_capacity=*/1, &frontier_offset,
                                      &total_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore));
  iree_async_semaphore_initialize(
      (const iree_async_semaphore_vtable_t*)&iree_hal_webgpu_semaphore_vtable,
      proactor, initial_value, frontier_offset, /*frontier_capacity=*/1,
      &semaphore->async);
  semaphore->host_allocator = host_allocator;
  semaphore->submitted_signal_axis = 0;
  iree_atomic_store(&semaphore->submitted_signal_value, 0,
                    iree_memory_order_relaxed);

  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_webgpu_semaphore_has_submitted_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t axis,
    uint64_t minimum_value) {
  iree_hal_webgpu_semaphore_t* webgpu_semaphore =
      iree_hal_webgpu_semaphore_cast(semaphore);
  if (webgpu_semaphore->submitted_signal_axis != axis) return false;
  uint64_t submitted_value = (uint64_t)iree_atomic_load(
      &webgpu_semaphore->submitted_signal_value, iree_memory_order_acquire);
  return submitted_value >= minimum_value;
}

void iree_hal_webgpu_semaphore_mark_submitted_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t axis, uint64_t value) {
  iree_hal_webgpu_semaphore_t* webgpu_semaphore =
      iree_hal_webgpu_semaphore_cast(semaphore);
  webgpu_semaphore->submitted_signal_axis = axis;
  iree_atomic_store(&webgpu_semaphore->submitted_signal_value, (int64_t)value,
                    iree_memory_order_release);
}

static void iree_hal_webgpu_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_webgpu_semaphore_t* semaphore =
      iree_hal_webgpu_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static uint64_t iree_hal_webgpu_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  // Check for failure status and encode it as the HAL failure sentinel value.
  // The HAL dispatch layer (semaphore.c) decodes this back to the original
  // status code via iree_hal_semaphore_failure_as_status().
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &base_semaphore->failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&base_semaphore->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_webgpu_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  // WebGPU has no hardware signal primitive. All signaling is CPU-side, driven
  // by onSubmittedWorkDone() completions routed through the proactor. Use the
  // standard CAS timeline advance + frontier merge + timepoint dispatch.
  iree_status_t status =
      iree_async_semaphore_advance_timeline(base_semaphore, value, frontier);
  if (iree_status_code(status) == IREE_STATUS_INVALID_ARGUMENT) {
    // Timeline was not advanced (duplicate signal or non-monotonic value).
    return status;
  }
  // Timeline was advanced. Dispatch satisfied timepoints even if a frontier
  // merge error occurred — waiters must be woken since the value has been
  // published via the CAS in advance_timeline.
  iree_async_semaphore_dispatch_timepoints(base_semaphore, value);
  return status;
}

static iree_status_t iree_hal_webgpu_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  iree_async_semaphore_t* async_semaphore =
      (iree_async_semaphore_t*)base_semaphore;

  // Check for failure before checking the timeline value. Failure is sticky:
  // once set, all waits return the failure code immediately.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_semaphore->failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_from_code(iree_status_code(failure));
  }

  // Fast path: check if the semaphore has already reached the target value.
  // Avoids timepoint registration and notification overhead for the common case
  // where work has already completed by the time the host checks.
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &async_semaphore->timeline_value, iree_memory_order_acquire);
  if (current_value >= value) {
    return iree_ok_status();
  }

  // Immediate timeout: return DEADLINE_EXCEEDED without blocking.
  // Uses the raw status code (not a full status object) because immediate
  // timeouts are used for polling and the overhead of backtrace capture and
  // allocation would be wasteful.
  if (iree_timeout_is_immediate(timeout)) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  // Blocking wait via the async semaphore multi-wait infrastructure. This
  // registers a timepoint, blocks on a notification, and handles cancellation
  // and cleanup. Using multi_wait with count=1 reuses the well-tested
  // timeout, failure detection, and spin/futex logic.
  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, &async_semaphore, &value, /*count=*/1, timeout,
      flags, iree_allocator_system());
}

static iree_status_t iree_hal_webgpu_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  // WebGPU has no native timepoint types (no CUDA events, no HIP events).
  // ASYNC_PRIMITIVE types are handled in the base HAL layer via the
  // semaphore's proactor, so this vtable method is only reached for
  // driver-specific types that WebGPU does not support.
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "WebGPU does not support driver-specific timepoint import");
}

static iree_status_t iree_hal_webgpu_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "WebGPU does not support driver-specific timepoint export");
}

static const iree_hal_semaphore_vtable_t iree_hal_webgpu_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_webgpu_semaphore_destroy,
            .query = iree_hal_webgpu_semaphore_query,
            .signal = iree_hal_webgpu_semaphore_signal,
            // on_fail: NULL — no device-side failure signaling needed. WebGPU
            // has no GPU-side waiter to notify on failure.
        },
    .wait = iree_hal_webgpu_semaphore_wait,
    .import_timepoint = iree_hal_webgpu_semaphore_import_timepoint,
    .export_timepoint = iree_hal_webgpu_semaphore_export_timepoint,
};
