// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/event_semaphore.h"

#include "iree/base/internal/wait_handle.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/cuda_status_util.h"
#include "iree/hal/drivers/cuda/timepoint_pool.h"
#include "iree/hal/utils/deferred_work_queue.h"

typedef struct iree_hal_cuda_semaphore_t {
  // Async semaphore with timeline value, failure status, and frontier.
  // Must be at offset 0 for toll-free bridging.
  iree_async_semaphore_t async;

  // The allocator used to create this semaphore.
  iree_allocator_t host_allocator;
  // The symbols used to issue CUDA API calls.
  const iree_hal_cuda_dynamic_symbols_t* symbols;

  // The timepoint pool to acquire timepoint objects.
  iree_hal_cuda_timepoint_pool_t* timepoint_pool;

  // The list of pending queue actions that this semaphore need to advance on
  // new signaled values.
  iree_hal_deferred_work_queue_t* work_queue;
} iree_hal_cuda_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_cuda_semaphore_vtable;

static iree_hal_cuda_semaphore_t* iree_hal_cuda_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_semaphore_vtable);
  return (iree_hal_cuda_semaphore_t*)base_value;
}

iree_status_t iree_hal_cuda_event_semaphore_create(
    uint64_t initial_value, const iree_hal_cuda_dynamic_symbols_t* symbols,
    iree_hal_cuda_timepoint_pool_t* timepoint_pool,
    iree_hal_deferred_work_queue_t* work_queue, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(timepoint_pool);
  IREE_ASSERT_ARGUMENT(work_queue);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_semaphore_t* semaphore = NULL;
  iree_host_size_t frontier_offset = 0, total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_semaphore_layout(sizeof(*semaphore), 0, &frontier_offset,
                                      &total_size));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore));
  iree_async_semaphore_initialize(
      (const iree_async_semaphore_vtable_t*)&iree_hal_cuda_semaphore_vtable,
      initial_value, frontier_offset, 0, &semaphore->async);
  semaphore->host_allocator = host_allocator;
  semaphore->symbols = symbols;
  semaphore->timepoint_pool = timepoint_pool;
  semaphore->work_queue = work_queue;

  *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_semaphore_deinitialize(&semaphore->async);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static uint64_t iree_hal_cuda_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_async_semaphore_t* async_sem = (iree_async_semaphore_t*)base_semaphore;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Both fields are atomic — fully lock-free query.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  uint64_t value;
  if (!iree_status_is_ok(failure)) {
    value = iree_hal_status_as_semaphore_failure(failure);
  } else {
    value = (uint64_t)iree_atomic_load(&async_sem->timeline_value,
                                       iree_memory_order_acquire);
  }

  IREE_TRACE_ZONE_END(z0);
  return value;
}

static iree_status_t iree_hal_cuda_semaphore_signal(
    iree_async_semaphore_t* base_semaphore, uint64_t new_value,
    const iree_async_frontier_t* frontier) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  IREE_TRACE_ZONE_BEGIN(z0);

  // Advance the timeline (CAS) and merge frontier.
  iree_status_t status = iree_async_semaphore_advance_timeline(
      base_semaphore, new_value, frontier);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Dispatch satisfied timepoints.
  iree_async_semaphore_dispatch_timepoints(base_semaphore, new_value);

  // Advance the deferred work queue if possible.
  status = iree_hal_deferred_work_queue_issue(semaphore->work_queue);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_semaphore_fail(iree_async_semaphore_t* base_semaphore,
                                         iree_status_t status) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_async_semaphore_t* async_sem = (iree_async_semaphore_t*)base_semaphore;
  IREE_TRACE_ZONE_BEGIN(z0);

  // First failure wins via CAS. Clone for storage, pass original to dispatch.
  iree_status_t stored = iree_status_clone(status);
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &async_sem->failure_status, &expected, (intptr_t)stored,
          iree_memory_order_release, iree_memory_order_acquire)) {
    iree_status_free(stored);
    iree_status_free(status);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Dispatch all pending timepoints with the failure status.
  // Takes ownership of |status| (clones per-timepoint, frees original).
  iree_async_semaphore_dispatch_timepoints_failed(base_semaphore, status);

  // Advance the deferred work queue if possible.
  iree_status_ignore(iree_hal_deferred_work_queue_issue(semaphore->work_queue));

  IREE_TRACE_ZONE_END(z0);
}

// Handles host wait timepoints when the semaphore timeline advances past
// the target value (or the semaphore fails/is cancelled).
// Fires under the semaphore's internal lock (dispatch-under-lock).
static void iree_hal_cuda_semaphore_timepoint_host_wait_callback(
    void* user_data, iree_async_semaphore_timepoint_t* async_timepoint,
    iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda_timepoint_t* timepoint =
      (iree_hal_cuda_timepoint_t*)async_timepoint;
  iree_event_set(&timepoint->timepoint.host_wait);
  iree_status_ignore(status);
  IREE_TRACE_ZONE_END(z0);
}

// Acquires a timepoint to wait the timeline to reach at least the given
// |min_value| from the host.
static iree_status_t iree_hal_cuda_semaphore_acquire_timepoint_host_wait(
    iree_hal_cuda_semaphore_t* semaphore, uint64_t min_value,
    iree_timeout_t timeout, iree_hal_cuda_timepoint_t** out_timepoint) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_timepoint_pool_acquire_host_wait(
              semaphore->timepoint_pool, 1, out_timepoint));
  // Register the timepoint with the async semaphore's timepoint list.
  // The callback fires under the semaphore's lock when the value is reached.
  (*out_timepoint)->base.callback =
      iree_hal_cuda_semaphore_timepoint_host_wait_callback;
  (*out_timepoint)->base.user_data = NULL;
  iree_async_semaphore_insert_timepoint(&semaphore->async, min_value,
                                        &(*out_timepoint)->base);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_cuda_semaphore_acquire_event_host_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t min_value,
    iree_hal_cuda_event_t** out_event) {
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);

  // Scan through the timepoint list and try to find a device event signal to
  // wait on. We need to lock with the async semaphore mutex here.
  iree_slim_mutex_lock(&semaphore->async.mutex);
  for (iree_async_semaphore_timepoint_t* tp =
           semaphore->base.async.timepoints_head;
       tp != NULL; tp = tp->next) {
    iree_hal_cuda_timepoint_t* signal_timepoint =
        (iree_hal_cuda_timepoint_t*)tp;
    if (signal_timepoint->kind == IREE_HAL_CUDA_TIMEPOINT_KIND_DEVICE_SIGNAL &&
        signal_timepoint->base.minimum_value >= min_value) {
      *out_event = signal_timepoint->timepoint.device_signal;
      iree_hal_cuda_event_retain(*out_event);
      break;
    }
  }
  iree_slim_mutex_unlock(&semaphore->async.mutex);

  IREE_TRACE_ZONE_END(z0);
  return *out_event != NULL;
}

// Checks if the semaphore has to wait to reach `value`.
// If it has to wait, then acquires a wait timepoint and returns it.
// If we don't need to wait, then *out_timepoint is set to NULL.
static iree_status_t iree_hal_cuda_semaphore_try_wait_or_acquire_wait_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_hal_cuda_timepoint_t** out_timepoint) {
  *out_timepoint = NULL;
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  iree_async_semaphore_t* async_sem = &semaphore->async;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Fast paths using atomic reads (no lock needed).
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &async_sem->timeline_value, iree_memory_order_acquire);
  if (current_value >= value) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (iree_timeout_is_immediate(timeout)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  // Slow path: acquire a timepoint (handles its own synchronization via the
  // async semaphore's internal lock and re-check-after-insert pattern).
  iree_status_t status = iree_hal_cuda_semaphore_acquire_timepoint_host_wait(
      semaphore, value, timeout, out_timepoint);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_hal_wait_flags_t flags) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  iree_async_semaphore_t* async_sem = &semaphore->async;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_timepoint_t* timepoint;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_semaphore_try_wait_or_acquire_wait_timepoint(
              base_semaphore, value, timeout, &timepoint));
  if (!timepoint) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Re-check for failure before the blocking wait (atomic, no lock needed).
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    iree_async_semaphore_remove_timepoint(&semaphore->async, &timepoint->base);
    iree_hal_cuda_timepoint_pool_release(semaphore->timepoint_pool, 1,
                                         &timepoint);
    IREE_TRACE_ZONE_END(z0);
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }

  // Wait until the timepoint resolves.
  // If satisfied the timepoint is automatically cleaned up and we are done. If
  // the deadline is reached before satisfied then we have to clean it up.
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_status_t status =
      iree_wait_one(&timepoint->timepoint.host_wait, deadline_ns);
  if (!iree_status_is_ok(status)) {
    iree_async_semaphore_remove_timepoint(&semaphore->async, &timepoint->base);
  }
  iree_hal_cuda_timepoint_pool_release(semaphore->timepoint_pool, 1,
                                       &timepoint);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Recheck after waking (atomic, no lock needed).
  failure = (iree_status_t)iree_atomic_load(&async_sem->failure_status,
                                            iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Handles device signal timepoints when the semaphore timeline advances past
// the target value. Releases the timepoint (and its CUDA event) back to the
// pool. Fires under the semaphore's internal lock (dispatch-under-lock).
static void iree_hal_cuda_semaphore_timepoint_device_signal_callback(
    void* user_data, iree_async_semaphore_timepoint_t* async_timepoint,
    iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda_timepoint_t* timepoint =
      (iree_hal_cuda_timepoint_t*)async_timepoint;
  iree_hal_cuda_timepoint_pool_release(timepoint->pool, 1, &timepoint);
  iree_status_ignore(status);
  IREE_TRACE_ZONE_END(z0);
}

// Acquires a timepoint to signal the timeline to the given |to_value| from the
// device.
iree_status_t iree_hal_cuda_event_semaphore_acquire_timepoint_device_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t to_value,
    CUevent* out_event) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  iree_hal_cuda_timepoint_t* signal_timepoint = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_timepoint_pool_acquire_device_signal(
              semaphore->timepoint_pool, 1, &signal_timepoint));

  // Check for failure or already-signaled before registering the timepoint.
  // iree_async_semaphore_acquire_timepoint has a synchronous fast path that
  // fires the callback immediately (releasing the timepoint and its CUDA event
  // back to the pool) when the timeline is already at/past to_value or the
  // semaphore has failed. We must bail early to avoid accessing a freed
  // timepoint. There is a narrow TOCTOU between this check and registration,
  // but it requires concurrent failure (another thread failing the semaphore)
  // during a single-digit instruction window — and the program is already on
  // an error path at that point.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    iree_hal_cuda_timepoint_pool_release(semaphore->timepoint_pool, 1,
                                         &signal_timepoint);
    IREE_TRACE_ZONE_END(z0);
    return iree_status_clone(failure);
  }
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &semaphore->async.timeline_value, iree_memory_order_acquire);
  if (IREE_UNLIKELY(current_value >= to_value)) {
    iree_hal_cuda_timepoint_pool_release(semaphore->timepoint_pool, 1,
                                         &signal_timepoint);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "timeline already at %" PRIu64
                            ", cannot signal %" PRIu64,
                            current_value, to_value);
  }

  // Cache event pointer before registration. After acquire_timepoint, the
  // timepoint is owned by the semaphore's timepoint list and will be freed
  // when the callback fires (which happens asynchronously in the normal case).
  iree_hal_cuda_event_t* event = signal_timepoint->timepoint.device_signal;

  // Register the timepoint with the async semaphore's timepoint list.
  signal_timepoint->base.callback =
      iree_hal_cuda_semaphore_timepoint_device_signal_callback;
  signal_timepoint->base.user_data = NULL;
  iree_async_semaphore_insert_timepoint(&semaphore->async, to_value,
                                        &signal_timepoint->base);
  iree_hal_cuda_event_t* event = signal_timepoint->timepoint.device_signal;

  // Scan through the timepoint list and update device wait timepoints to wait
  // for this device signal when possible. We need to lock with the async
  // semaphore mutex here.
  iree_slim_mutex_lock(&semaphore->async.mutex);
  for (iree_async_semaphore_timepoint_t* tp =
           semaphore->base.async.timepoints_head;
       tp != NULL; tp = tp->next) {
    iree_hal_cuda_timepoint_t* wait_timepoint = (iree_hal_cuda_timepoint_t*)tp;
    if (wait_timepoint->kind == IREE_HAL_CUDA_TIMEPOINT_KIND_DEVICE_WAIT &&
        wait_timepoint->timepoint.device_wait == NULL &&
        wait_timepoint->base.minimum_value <= to_value) {
      iree_hal_cuda_event_retain(event);
      wait_timepoint->timepoint.device_wait = event;
    }
  }
  iree_slim_mutex_unlock(&semaphore->async.mutex);

  *out_event = iree_hal_cuda_event_handle(event);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Handles device wait timepoints when the semaphore timeline advances past
// the target value. Releases the timepoint (and its CUDA event) back to the
// pool. Fires under the semaphore's internal lock (dispatch-under-lock).
static void iree_hal_cuda_semaphore_timepoint_device_wait_callback(
    void* user_data, iree_async_semaphore_timepoint_t* async_timepoint,
    iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda_timepoint_t* timepoint =
      (iree_hal_cuda_timepoint_t*)async_timepoint;
  iree_hal_cuda_timepoint_pool_release(timepoint->pool, 1, &timepoint);
  iree_status_ignore(status);
  IREE_TRACE_ZONE_END(z0);
}

// Acquires a timepoint to wait the timeline to reach at least the given
// |min_value| on the device.
iree_status_t iree_hal_cuda_event_semaphore_acquire_timepoint_device_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t min_value,
    CUevent* out_event) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  iree_hal_cuda_timepoint_t* wait_timepoint = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_timepoint_pool_acquire_device_wait(
              semaphore->timepoint_pool, 1, &wait_timepoint));

  // Same pre-check as the signal path: bail early if the semaphore has failed
  // or the timeline is already past min_value to avoid the synchronous fast
  // path in acquire_timepoint releasing the timepoint under us.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &semaphore->async.failure_status, iree_memory_order_acquire);
  if (IREE_UNLIKELY(!iree_status_is_ok(failure))) {
    iree_hal_cuda_timepoint_pool_release(semaphore->timepoint_pool, 1,
                                         &wait_timepoint);
    IREE_TRACE_ZONE_END(z0);
    return iree_status_clone(failure);
  }
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &semaphore->async.timeline_value, iree_memory_order_acquire);
  if (current_value >= min_value) {
    // Already satisfied — no device wait needed. Release the timepoint and
    // return the event as-is (the caller may choose to skip the wait).
    *out_event =
        iree_hal_cuda_event_handle(wait_timepoint->timepoint.device_wait);
    iree_hal_cuda_timepoint_pool_release(semaphore->timepoint_pool, 1,
                                         &wait_timepoint);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Register the timepoint with the async semaphore's timepoint list.
  // The pre-check above prevents the synchronous fast path (same TOCTOU
  // rationale as the signal path). In the normal case, the timepoint is
  // enqueued and remains valid until its callback fires asynchronously.
  wait_timepoint->base.callback =
      iree_hal_cuda_semaphore_timepoint_device_wait_callback;
  wait_timepoint->base.user_data = NULL;
  iree_async_semaphore_insert_timepoint(&semaphore->async, min_value,
                                        &wait_timepoint->base);

  iree_hal_cuda_event_t* wait_event = NULL;
  if (iree_hal_cuda_semaphore_acquire_event_host_wait(
          iree_hal_semaphore_cast(&semaphore->async), min_value, &wait_event)) {
    // We've found an existing signal timepoint to wait on; we don't need a
    // standalone wait timepoint anymore. Decrease its refcount before
    // overwriting it to return it back to the pool and retain the existing one.
    iree_hal_cuda_event_release(wait_timepoint->timepoint.device_wait);
    wait_timepoint->timepoint.device_wait = wait_event;
  }

  *out_event =
      iree_hal_cuda_event_handle(wait_timepoint->timepoint.device_wait);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_cuda_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export is not yet implemented");
}

static uint8_t iree_hal_cuda_semaphore_query_frontier(
    iree_async_semaphore_t* semaphore, iree_async_frontier_t* out_frontier,
    uint8_t capacity) {
  (void)semaphore;
  (void)out_frontier;
  (void)capacity;
  return 0;
}

static iree_status_t iree_hal_cuda_semaphore_export_primitive(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_async_primitive_t* out_primitive) {
  (void)semaphore;
  (void)minimum_value;
  (void)out_primitive;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "primitive export not supported");
}

static const iree_hal_semaphore_vtable_t iree_hal_cuda_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_cuda_semaphore_destroy,
            .query = iree_hal_cuda_semaphore_query,
            .signal = iree_hal_cuda_semaphore_signal,
            .query_frontier = iree_hal_cuda_semaphore_query_frontier,
            .fail = iree_hal_cuda_semaphore_fail,
            .export_primitive = iree_hal_cuda_semaphore_export_primitive,
        },
    .wait = iree_hal_cuda_semaphore_wait,
    .import_timepoint = iree_hal_cuda_semaphore_import_timepoint,
    .export_timepoint = iree_hal_cuda_semaphore_export_timepoint,
};
