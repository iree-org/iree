// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/shared_event.h"

#import <Metal/Metal.h>

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

typedef struct iree_hal_metal_shared_event_t {
  // Async semaphore with timeline value, failure status, and frontier.
  // Must be at offset 0 for toll-free bridging.
  iree_async_semaphore_t async;

  id<MTLSharedEvent> shared_event;
  // A listener object used for dispatching notifications; owned by the device.
  MTLSharedEventListener* event_listener;

  iree_allocator_t host_allocator;
} iree_hal_metal_shared_event_t;

static const iree_hal_semaphore_vtable_t iree_hal_metal_shared_event_vtable;

static iree_hal_metal_shared_event_t* iree_hal_metal_shared_event_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_shared_event_vtable);
  return (iree_hal_metal_shared_event_t*)base_value;
}

static const iree_hal_metal_shared_event_t* iree_hal_metal_shared_event_const_cast(
    const iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_shared_event_vtable);
  return (const iree_hal_metal_shared_event_t*)base_value;
}

bool iree_hal_metal_shared_event_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is(semaphore, &iree_hal_metal_shared_event_vtable);
}

id<MTLSharedEvent> iree_hal_metal_shared_event_handle(const iree_hal_semaphore_t* base_semaphore) {
  const iree_hal_metal_shared_event_t* semaphore =
      iree_hal_metal_shared_event_const_cast(base_semaphore);
  return semaphore->shared_event;
}

iree_status_t iree_hal_metal_shared_event_create(id<MTLDevice> device, uint64_t initial_value,
                                                 MTLSharedEventListener* listener,
                                                 iree_allocator_t host_allocator,
                                                 iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  iree_hal_metal_shared_event_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0, total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_semaphore_layout(sizeof(*semaphore), 0, &frontier_offset, &total_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore));
  iree_async_semaphore_initialize(
      (const iree_async_semaphore_vtable_t*)&iree_hal_metal_shared_event_vtable, initial_value,
      frontier_offset, 0, &semaphore->async);
  semaphore->shared_event = [device newSharedEvent];  // +1
  semaphore->shared_event.signaledValue = initial_value;
  semaphore->event_listener = listener;
  semaphore->host_allocator = host_allocator;
  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_metal_shared_event_destroy(iree_async_semaphore_t* base_semaphore) {
  iree_hal_metal_shared_event_t* semaphore =
      iree_hal_metal_shared_event_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_semaphore_deinitialize(&semaphore->async);
  [semaphore->shared_event release];  // -1
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

// Queries the Metal shared event and syncs the timeline.
// MTLSharedEvent is the source of truth; the timeline is a cache
// for async dispatch and causal tracking.
static uint64_t iree_hal_metal_shared_event_query(iree_async_semaphore_t* base_semaphore) {
  iree_hal_metal_shared_event_t* semaphore =
      iree_hal_metal_shared_event_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_async_semaphore_t* async_sem = &semaphore->async;

  // Check failure first (lock-free).
  iree_status_t failure =
      (iree_status_t)iree_atomic_load(&async_sem->failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_hal_status_as_semaphore_failure(failure);
  }

  // Read the hardware value and sync the timeline.
  uint64_t value = semaphore->shared_event.signaledValue;
  if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    return iree_hal_status_as_semaphore_failure(iree_status_from_code(IREE_STATUS_ABORTED));
  }

  // Update timeline atomically. Don't use advance_timeline because queries may
  // return the same value on consecutive calls without violating monotonicity.
  iree_atomic_store(&async_sem->timeline_value, (int64_t)value, iree_memory_order_release);
  iree_async_semaphore_dispatch_timepoints(base_semaphore, value);
  return value;
}

static iree_status_t iree_hal_metal_shared_event_signal(iree_async_semaphore_t* base_semaphore,
                                                        uint64_t new_value,
                                                        const iree_async_frontier_t* frontier) {
  iree_hal_metal_shared_event_t* semaphore =
      iree_hal_metal_shared_event_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_async_semaphore_t* async_sem = &semaphore->async;

  // Check failure first (lock-free).
  iree_status_t failure =
      (iree_status_t)iree_atomic_load(&async_sem->failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    return iree_status_clone(failure);
  }

  // Signal the Metal shared event.
  semaphore->shared_event.signaledValue = new_value;

  // Advance the software timeline (CAS) and merge frontier. Each timeline
  // value must be signaled exactly once — CAS failure here indicates a
  // structural error (duplicate signal or non-monotonic scheduling).
  iree_status_t advance_status =
      iree_async_semaphore_advance_timeline(base_semaphore, new_value, frontier);
  if (IREE_UNLIKELY(!iree_status_is_ok(advance_status))) {
    iree_async_semaphore_fail(base_semaphore, advance_status);
    // The Metal shared event was already signaled — return OK for the Metal
    // side but the async semaphore is now failed so waiters get the diagnostic.
    return iree_ok_status();
  }
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);
  return iree_ok_status();
}

static void iree_hal_metal_shared_event_fail(iree_async_semaphore_t* base_semaphore,
                                             iree_status_t status) {
  iree_hal_metal_shared_event_t* semaphore =
      iree_hal_metal_shared_event_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_async_semaphore_t* async_sem = &semaphore->async;
  IREE_TRACE_ZONE_BEGIN(z0);

  // First failure wins via CAS. Clone for storage, pass original to dispatch.
  iree_status_t stored = iree_status_clone(status);
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(&async_sem->failure_status, &expected, (intptr_t)stored,
                                           iree_memory_order_release, iree_memory_order_acquire)) {
    iree_status_free(stored);
    iree_status_free(status);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Signal Metal to the failure sentinel value.
  semaphore->shared_event.signaledValue = IREE_HAL_SEMAPHORE_FAILURE_VALUE;

  // Dispatch all pending timepoints with the failure status.
  // Takes ownership of |status| (clones per-timepoint, frees original).
  iree_async_semaphore_dispatch_timepoints_failed(base_semaphore, status);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_metal_shared_event_wait(iree_hal_semaphore_t* base_semaphore,
                                                      uint64_t value, iree_timeout_t timeout,
                                                      iree_async_wait_flags_t flags) {
  iree_hal_metal_shared_event_t* semaphore = iree_hal_metal_shared_event_cast(base_semaphore);

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  uint64_t timeout_ns;
  dispatch_time_t apple_timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    timeout_ns = UINT64_MAX;
    apple_timeout_ns = DISPATCH_TIME_FOREVER;
  } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    timeout_ns = 0;
    apple_timeout_ns = DISPATCH_TIME_NOW;
  } else {
    iree_time_t now_ns = iree_time_now();
    if (deadline_ns < now_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    timeout_ns = (uint64_t)(deadline_ns - now_ns);
    apple_timeout_ns = dispatch_time(DISPATCH_TIME_NOW, timeout_ns);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Check failure status first (lock-free). Host-side failures set the atomic
  // but may not trigger Metal's notifyListener blocks (CPU-side signaledValue
  // assignment does not reliably fire pending listeners).
  iree_status_t failure =
      (iree_status_t)iree_atomic_load(&semaphore->async.failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }

  // Quick path for impatient waiting.
  if (timeout_ns == 0) {
    uint64_t current_value = semaphore->shared_event.signaledValue;
    if (current_value >= value) {
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  __block dispatch_semaphore_t work_done = dispatch_semaphore_create(0);
  __block bool did_fail = false;

  // Use a listener to the MTLSharedEvent to notify us when the work is done on
  // GPU by signaling a semaphore. The signaling will happen in a new dispatch
  // queue; the current thread will wait on the semaphore.
  [semaphore->shared_event notifyListener:semaphore->event_listener
                                  atValue:value
                                    block:^(id<MTLSharedEvent> se, uint64_t v) {
                                      if (v >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) did_fail = true;
                                      dispatch_semaphore_signal(work_done);
                                    }];

  // If the work is not done immediately, dispatch_semaphore_wait decreases the
  // semaphore value to less than zero first and then puts the current thread
  // into wait state.
  intptr_t timed_out = dispatch_semaphore_wait(work_done, apple_timeout_ns);
  dispatch_release(work_done);

  // Re-check failure status after waiting. A host-side failure may not have
  // triggered the Metal listener, causing a timeout that is really a failure.
  if (!did_fail) {
    failure = (iree_status_t)iree_atomic_load(&semaphore->async.failure_status,
                                              iree_memory_order_acquire);
    if (IREE_UNLIKELY(!iree_status_is_ok(failure))) did_fail = true;
  }

  IREE_TRACE_ZONE_END(z0);
  if (IREE_UNLIKELY(did_fail)) return iree_status_from_code(IREE_STATUS_ABORTED);
  if (timed_out) return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_shared_event_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_metal_shared_event_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "timepoint export is not yet implemented");
}

static const iree_hal_semaphore_vtable_t iree_hal_metal_shared_event_vtable = {
    .async =
        {
            .destroy = iree_hal_metal_shared_event_destroy,
            .query = iree_hal_metal_shared_event_query,
            .signal = iree_hal_metal_shared_event_signal,
            .fail = iree_hal_metal_shared_event_fail,
        },
    .wait = iree_hal_metal_shared_event_wait,
    .import_timepoint = iree_hal_metal_shared_event_import_timepoint,
    .export_timepoint = iree_hal_metal_shared_event_export_timepoint,
};
