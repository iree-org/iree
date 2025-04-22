// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/event_semaphore.h"

#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/cuda_status_util.h"
#include "iree/hal/drivers/cuda/timepoint_pool.h"
#include "iree/hal/utils/deferred_work_queue.h"
#include "iree/hal/utils/semaphore_base.h"

typedef struct iree_hal_cuda_semaphore_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_semaphore_t base;

  // The allocator used to create this semaphore.
  iree_allocator_t host_allocator;
  // The symbols used to issue CUDA API calls.
  const iree_hal_cuda_dynamic_symbols_t* symbols;

  // The timepoint pool to acquire timepoint objects.
  iree_hal_cuda_timepoint_pool_t* timepoint_pool;

  // The list of pending queue actions that this semaphore need to advance on
  // new signaled values.
  iree_hal_deferred_work_queue_t* work_queue;

  // Guards value and status. We expect low contention on semaphores and since
  // iree_slim_mutex_t is (effectively) just a CAS this keeps things simpler
  // than trying to make the entire structure lock-free.
  // If we need to hold mutex and base.timepoint_mutex locked together, the
  // locking order must be (mutex, base.timepoint_mutex).
  iree_slim_mutex_t mutex;

  // Current signaled value. May be IREE_HAL_SEMAPHORE_FAILURE_VALUE to
  // indicate that the semaphore has been signaled for failure and
  // |failure_status| contains the error.
  uint64_t current_value IREE_GUARDED_BY(mutex);

  // OK or the status passed to iree_hal_semaphore_fail. Owned by the semaphore.
  iree_status_t failure_status IREE_GUARDED_BY(mutex);
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
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*semaphore),
                                (void**)&semaphore));

  iree_hal_semaphore_initialize(&iree_hal_cuda_semaphore_vtable,
                                &semaphore->base);
  semaphore->host_allocator = host_allocator;
  semaphore->symbols = symbols;
  semaphore->timepoint_pool = timepoint_pool;
  semaphore->work_queue = work_queue;
  iree_slim_mutex_initialize(&semaphore->mutex);
  semaphore->current_value = initial_value;
  semaphore->failure_status = iree_ok_status();

  *out_semaphore = &semaphore->base;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_ignore(semaphore->failure_status);
  iree_slim_mutex_deinitialize(&semaphore->mutex);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&semaphore->mutex);

  *out_value = semaphore->current_value;

  iree_status_t status = iree_ok_status();
  if (*out_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    status = iree_status_clone(semaphore->failure_status);
  }

  iree_slim_mutex_unlock(&semaphore->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&semaphore->mutex);

  if (new_value <= semaphore->current_value) {
    uint64_t current_value IREE_ATTRIBUTE_UNUSED = semaphore->current_value;
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "semaphore values must be monotonically "
                            "increasing; current_value=%" PRIu64
                            ", new_value=%" PRIu64,
                            current_value, new_value);
  }

  semaphore->current_value = new_value;

  iree_slim_mutex_unlock(&semaphore->mutex);

  // Notify timepoints - note that this must happen outside the lock.
  iree_hal_semaphore_notify(&semaphore->base, new_value, IREE_STATUS_OK);

  // Advance the deferred work queue if possible. This also must happen
  // outside the lock to avoid nesting.
  iree_status_t status =
      iree_hal_deferred_work_queue_issue(semaphore->work_queue);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                         iree_status_t status) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_status_code_t status_code = iree_status_code(status);

  iree_slim_mutex_lock(&semaphore->mutex);

  // Try to set our local status - we only preserve the first failure so only
  // do this if we are going from a valid semaphore to a failed one.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Signal to our failure sentinel value.
  semaphore->current_value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  semaphore->failure_status = status;

  iree_slim_mutex_unlock(&semaphore->mutex);

  // Notify timepoints - note that this must happen outside the lock.
  iree_hal_semaphore_notify(&semaphore->base, IREE_HAL_SEMAPHORE_FAILURE_VALUE,
                            status_code);

  // Advance the deferred work queue if possible. This also must happen
  // outside the lock to avoid nesting.
  status = iree_hal_deferred_work_queue_issue(semaphore->work_queue);
  iree_status_ignore(status);

  IREE_TRACE_ZONE_END(z0);
}

// Handles host wait timepoints on the host when the |semaphore| timeline
// advances past the given |value|.
//
// Note that this callback is invoked by the a host thread.
static iree_status_t iree_hal_cuda_semaphore_timepoint_host_wait_callback(
    void* user_data, iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_status_code_t status_code) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda_timepoint_t* timepoint = (iree_hal_cuda_timepoint_t*)user_data;
  iree_event_set(&timepoint->timepoint.host_wait);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
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
  // Initialize the timepoint with the value and callback, and connect it to
  // this semaphore.
  iree_hal_semaphore_acquire_timepoint(
      &semaphore->base, min_value, timeout,
      (iree_hal_semaphore_callback_t){
          .fn = iree_hal_cuda_semaphore_timepoint_host_wait_callback,
          .user_data = *out_timepoint,
      },
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
  // wait on. We need to lock with the timepoint list mutex here.
  iree_slim_mutex_lock(&semaphore->base.timepoint_mutex);
  for (iree_hal_semaphore_timepoint_t* tp = semaphore->base.timepoint_list.head;
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
  iree_slim_mutex_unlock(&semaphore->base.timepoint_mutex);

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
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&semaphore->mutex);
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Fastest path: failed; return an error to tell callers to query for it.
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }
  if (semaphore->current_value >= value) {
    // Fast path: already satisfied.
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (iree_timeout_is_immediate(timeout)) {
    // Not satisfied but a poll, so can avoid the expensive wait handle work.
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  // Slow path: acquire a timepoint. This should happen inside of the lock too.
  // If not locked the semaphore may be signal before acquiring a timepoint.
  // Then we would miss the signal.
  iree_status_t status = iree_hal_cuda_semaphore_acquire_timepoint_host_wait(
      semaphore, value, timeout, out_timepoint);

  iree_slim_mutex_unlock(&semaphore->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_timepoint_t* timepoint;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_semaphore_try_wait_or_acquire_wait_timepoint(
              base_semaphore, value, timeout, &timepoint));
  if (!timepoint) {
    // We don't need to wait on a timepoint.
    // The wait condition is satisfied.
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_slim_mutex_lock(&semaphore->mutex);
  if (semaphore->current_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    iree_slim_mutex_unlock(&semaphore->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_ABORTED);
  }
  iree_slim_mutex_unlock(&semaphore->mutex);

  // Wait until the timepoint resolves.
  // If satisfied the timepoint is automatically cleaned up and we are done. If
  // the deadline is reached before satisfied then we have to clean it up.
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_status_t status =
      iree_wait_one(&timepoint->timepoint.host_wait, deadline_ns);
  status = iree_wait_one(&timepoint->timepoint.host_wait, deadline_ns);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_cancel_timepoint(&semaphore->base, &timepoint->base);
  }
  iree_hal_cuda_timepoint_pool_release(semaphore->timepoint_pool, 1,
                                       &timepoint);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_slim_mutex_lock(&semaphore->mutex);
  if (semaphore->current_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    status = iree_make_status(IREE_STATUS_ABORTED);
  }
  iree_slim_mutex_unlock(&semaphore->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_cuda_semaphore_multi_wait(
    const iree_hal_semaphore_list_t semaphore_list,
    iree_hal_wait_mode_t wait_mode, iree_timeout_t timeout,
    iree_arena_block_pool_t* block_pool) {
  if (semaphore_list.count == 0) return iree_ok_status();

  if (semaphore_list.count == 1) {
    // Fast-path for a single semaphore.
    return iree_hal_semaphore_wait(semaphore_list.semaphores[0],
                                   semaphore_list.payload_values[0], timeout);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  // Avoid heap allocations by using the device block pool for the wait set.
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  iree_wait_set_t* wait_set = NULL;
  iree_status_t status = iree_wait_set_allocate(
      semaphore_list.count, iree_arena_allocator(&arena), &wait_set);

  // Acquire a host wait handle for each semaphore timepoint we are to wait on.
  iree_host_size_t timepoint_count = 0;
  iree_hal_cuda_timepoint_t** timepoints = NULL;
  iree_host_size_t total_timepoint_size =
      semaphore_list.count * sizeof(timepoints[0]);
  bool needs_wait = true;
  status =
      iree_arena_allocate(&arena, total_timepoint_size, (void**)&timepoints);
  if (iree_status_is_ok(status)) {
    memset(timepoints, 0, total_timepoint_size);
    for (iree_host_size_t i = 0; i < semaphore_list.count && needs_wait; ++i) {
      iree_hal_cuda_timepoint_t* timepoint;
      status = iree_hal_cuda_semaphore_try_wait_or_acquire_wait_timepoint(
          semaphore_list.semaphores[i], semaphore_list.payload_values[i],
          timeout, &timepoint);
      if (!iree_status_is_ok(status)) break;
      if (!timepoint) {
        // We don't need to wait on a timepoint.
        // The wait condition is satisfied.
        if (wait_mode == IREE_HAL_WAIT_MODE_ANY) {
          needs_wait = false;
          break;
        }
        continue;
      }

      timepoints[timepoint_count++] = timepoint;
      status = iree_wait_set_insert(wait_set, timepoint->timepoint.host_wait);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Perform the wait.
  if (iree_status_is_ok(status) && needs_wait) {
    if (wait_mode == IREE_HAL_WAIT_MODE_ANY) {
      status = iree_wait_any(wait_set, deadline_ns, /*out_wake_handle=*/NULL);
    } else {
      status = iree_wait_all(wait_set, deadline_ns);
    }
  }

  for (iree_host_size_t i = 0; i < timepoint_count; ++i) {
    iree_hal_cuda_timepoint_t* timepoint = timepoints[i];
    iree_hal_semaphore_t* semaphore = timepoint->base.semaphore;
    // Cancel if this is still an unresolved host wait.
    if (semaphore) {
      iree_hal_semaphore_cancel_timepoint(semaphore, &timepoint->base);
    }
    iree_hal_cuda_timepoint_pool_release(timepoint->pool, 1, &timepoint);
  }
  iree_wait_set_free(wait_set);
  iree_arena_deinitialize(&arena);

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_cuda_semaphore_t* semaphore =
        iree_hal_cuda_semaphore_cast(semaphore_list.semaphores[i]);
    iree_slim_mutex_lock(&semaphore->mutex);
    if (semaphore->current_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
      iree_slim_mutex_unlock(&semaphore->mutex);
      status = iree_make_status(IREE_STATUS_ABORTED);
      break;
    }
    iree_slim_mutex_unlock(&semaphore->mutex);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Handles device signal timepoints on the host when the |semaphore| timeline
// advances past the given |value|.
//
// Note that this callback is invoked by the a host thread after the CUDA host
// function callback function is triggered in the CUDA driver.
static iree_status_t iree_hal_cuda_semaphore_timepoint_device_signal_callback(
    void* user_data, iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_status_code_t status_code) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda_timepoint_t* timepoint = (iree_hal_cuda_timepoint_t*)user_data;
  // Just release the timepoint back to the pool. This will decrease the
  // reference count of the underlying CUDA event internally.
  iree_hal_cuda_timepoint_pool_release(timepoint->pool, 1, &timepoint);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
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

  // Initialize the timepoint with the value and callback, and connect it to
  // this semaphore.
  iree_hal_semaphore_acquire_timepoint(
      &semaphore->base, to_value, iree_infinite_timeout(),
      (iree_hal_semaphore_callback_t){
          .fn = iree_hal_cuda_semaphore_timepoint_device_signal_callback,
          .user_data = signal_timepoint,
      },
      &signal_timepoint->base);
  iree_hal_cuda_event_t* event = signal_timepoint->timepoint.device_signal;

  // Scan through the timepoint list and update device wait timepoints to wait
  // for this device signal when possible. We need to lock with the timepoint
  // list mutex here.
  iree_slim_mutex_lock(&semaphore->base.timepoint_mutex);
  for (iree_hal_semaphore_timepoint_t* tp = semaphore->base.timepoint_list.head;
       tp != NULL; tp = tp->next) {
    iree_hal_cuda_timepoint_t* wait_timepoint = (iree_hal_cuda_timepoint_t*)tp;
    if (wait_timepoint->kind == IREE_HAL_CUDA_TIMEPOINT_KIND_DEVICE_WAIT &&
        wait_timepoint->timepoint.device_wait == NULL &&
        wait_timepoint->base.minimum_value <= to_value) {
      iree_hal_cuda_event_retain(event);
      wait_timepoint->timepoint.device_wait = event;
    }
  }
  iree_slim_mutex_unlock(&semaphore->base.timepoint_mutex);

  *out_event = iree_hal_cuda_event_handle(event);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Handles device wait timepoints on the host when the |semaphore| timeline
// advances past the given |value|.
//
// Note that this callback is invoked by the a host thread.
static iree_status_t iree_hal_cuda_semaphore_timepoint_device_wait_callback(
    void* user_data, iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_status_code_t status_code) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_cuda_timepoint_t* timepoint = (iree_hal_cuda_timepoint_t*)user_data;
  // Just release the timepoint back to the pool. This will decrease the
  // reference count of the underlying CUDA event internally.
  iree_hal_cuda_timepoint_pool_release(timepoint->pool, 1, &timepoint);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
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

  // Initialize the timepoint with the value and callback, and connect it to
  // this semaphore.
  iree_hal_semaphore_acquire_timepoint(
      &semaphore->base, min_value, iree_infinite_timeout(),
      (iree_hal_semaphore_callback_t){
          .fn = iree_hal_cuda_semaphore_timepoint_device_wait_callback,
          .user_data = wait_timepoint,
      },
      &wait_timepoint->base);

  iree_hal_cuda_event_t* wait_event = NULL;
  if (iree_hal_cuda_semaphore_acquire_event_host_wait(&semaphore->base,
                                                      min_value, &wait_event)) {
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

static const iree_hal_semaphore_vtable_t iree_hal_cuda_semaphore_vtable = {
    .destroy = iree_hal_cuda_semaphore_destroy,
    .query = iree_hal_cuda_semaphore_query,
    .signal = iree_hal_cuda_semaphore_signal,
    .fail = iree_hal_cuda_semaphore_fail,
    .wait = iree_hal_cuda_semaphore_wait,
    .import_timepoint = iree_hal_cuda_semaphore_import_timepoint,
    .export_timepoint = iree_hal_cuda_semaphore_export_timepoint,
};
