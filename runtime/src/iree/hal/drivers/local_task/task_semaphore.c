// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_semaphore.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/hal/utils/semaphore_base.h"

//===----------------------------------------------------------------------===//
// iree_hal_task_timepoint_t
//===----------------------------------------------------------------------===//

// Represents a point in the timeline that someone is waiting to be reached.
// When the semaphore is signaled to at least the specified value then the
// given event will be signaled and the timepoint discarded.
//
// Instances are owned and retained by the caller that requested them - usually
// in the arena associated with the submission, but could be on the stack of a
// synchronously waiting thread.
typedef struct iree_hal_task_timepoint_t {
  iree_hal_semaphore_timepoint_t base;
  iree_hal_semaphore_t* semaphore;
  iree_event_t event;
} iree_hal_task_timepoint_t;

// Handles timepoint callbacks when either the timepoint is reached or it fails.
// We set the event in either case and let the waiters deal with the fallout.
static iree_status_t iree_hal_task_semaphore_timepoint_callback(
    void* user_data, iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_status_code_t status_code) {
  iree_hal_task_timepoint_t* timepoint = (iree_hal_task_timepoint_t*)user_data;
  iree_event_set(&timepoint->event);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_task_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_task_semaphore_t {
  iree_hal_semaphore_t base;
  iree_allocator_t host_allocator;
  iree_event_pool_t* event_pool;

  // Guards all mutable fields. We expect low contention on semaphores and since
  // iree_slim_mutex_t is (effectively) just a CAS this keeps things simpler
  // than trying to make the entire structure lock-free.
  iree_slim_mutex_t mutex;

  // Current signaled value. May be IREE_HAL_SEMAPHORE_FAILURE_VALUE to
  // indicate that the semaphore has been signaled for failure and
  // |failure_status| contains the error.
  uint64_t current_value;

  // OK or the status passed to iree_hal_semaphore_fail. Owned by the semaphore.
  iree_status_t failure_status;
} iree_hal_task_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_task_semaphore_vtable;

static iree_hal_task_semaphore_t* iree_hal_task_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_task_semaphore_vtable);
  return (iree_hal_task_semaphore_t*)base_value;
}

iree_status_t iree_hal_task_semaphore_create(
    iree_event_pool_t* event_pool, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(event_pool);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_initialize(&iree_hal_task_semaphore_vtable,
                                  &semaphore->base);
    semaphore->host_allocator = host_allocator;
    semaphore->event_pool = event_pool;

    iree_slim_mutex_initialize(&semaphore->mutex);
    semaphore->current_value = initial_value;
    semaphore->failure_status = iree_ok_status();

    *out_semaphore = &semaphore->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_task_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_deinitialize(&semaphore->mutex);
  iree_status_ignore(semaphore->failure_status);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_task_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is(&semaphore->resource,
                              &iree_hal_task_semaphore_vtable);
}

static iree_status_t iree_hal_task_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  *out_value = semaphore->current_value;

  iree_status_t status = iree_ok_status();
  if (*out_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    status = iree_status_clone(semaphore->failure_status);
  }

  iree_slim_mutex_unlock(&semaphore->mutex);

  return status;
}

static iree_status_t iree_hal_task_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  if (new_value <= semaphore->current_value) {
    uint64_t current_value IREE_ATTRIBUTE_UNUSED = semaphore->current_value;
    iree_slim_mutex_unlock(&semaphore->mutex);
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

  return iree_ok_status();
}

static void iree_hal_task_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                         iree_status_t status) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  const iree_status_code_t status_code = iree_status_code(status);

  iree_slim_mutex_lock(&semaphore->mutex);

  // Try to set our local status - we only preserve the first failure so only
  // do this if we are going from a valid semaphore to a failed one.
  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    return;
  }

  // Signal to our failure sentinel value.
  semaphore->current_value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
  semaphore->failure_status = status;

  iree_slim_mutex_unlock(&semaphore->mutex);

  // Notify timepoints - note that this must happen outside the lock.
  iree_hal_semaphore_notify(&semaphore->base, IREE_HAL_SEMAPHORE_FAILURE_VALUE,
                            status_code);
}

// Acquires a timepoint waiting for the given value.
// |out_timepoint| is owned by the caller and must be kept live until the
// timepoint has been reached (or it is cancelled by the caller).
static iree_status_t iree_hal_task_semaphore_acquire_timepoint(
    iree_hal_task_semaphore_t* semaphore, uint64_t minimum_value,
    iree_timeout_t timeout, iree_hal_task_timepoint_t* out_timepoint) {
  IREE_RETURN_IF_ERROR(
      iree_event_pool_acquire(semaphore->event_pool, 1, &out_timepoint->event));
  out_timepoint->semaphore = &semaphore->base;
  iree_hal_semaphore_acquire_timepoint(
      &semaphore->base, minimum_value, timeout,
      (iree_hal_semaphore_callback_t){
          .fn = iree_hal_task_semaphore_timepoint_callback,
          .user_data = out_timepoint,
      },
      &out_timepoint->base);
  return iree_ok_status();
}

typedef struct iree_hal_task_semaphore_wait_cmd_t {
  iree_task_wait_t task;
  iree_hal_task_semaphore_t* semaphore;
  iree_hal_task_timepoint_t timepoint;
} iree_hal_task_semaphore_wait_cmd_t;

// Cleans up a wait task by returning the event used to the pool and - if the
// task failed - ensuring we scrub it from the timepoint list.
static void iree_hal_task_semaphore_wait_cmd_cleanup(
    iree_task_t* task, iree_status_code_t status_code) {
  iree_hal_task_semaphore_wait_cmd_t* cmd =
      (iree_hal_task_semaphore_wait_cmd_t*)task;
  if (IREE_UNLIKELY(status_code != IREE_STATUS_OK)) {
    // Abort the timepoint. Note that this is not designed to be fast as
    // semaphore failure is an exceptional case.
    iree_hal_semaphore_cancel_timepoint(&cmd->semaphore->base,
                                        &cmd->timepoint.base);
  }
  iree_event_pool_release(cmd->semaphore->event_pool, 1, &cmd->timepoint.event);
  iree_hal_semaphore_release((iree_hal_semaphore_t*)cmd->semaphore);
}

iree_status_t iree_hal_task_semaphore_enqueue_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t minimum_value,
    iree_task_t* issue_task, iree_arena_allocator_t* arena,
    iree_task_submission_t* submission) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  iree_status_t status = iree_ok_status();
  if (semaphore->current_value >= minimum_value) {
    // Fast path: already satisfied.
  } else if (!iree_status_is_ok(semaphore->failure_status)) {
    // Semaphore failed; can't enqueue timepoints (they'll reject immediately).
    status = iree_status_clone(semaphore->failure_status);
  } else {
    // Slow path: acquire a system wait handle and perform a full wait.
    iree_hal_task_semaphore_wait_cmd_t* cmd = NULL;
    status = iree_arena_allocate(arena, sizeof(*cmd), (void**)&cmd);
    if (iree_status_is_ok(status)) {
      status = iree_hal_task_semaphore_acquire_timepoint(
          semaphore, minimum_value, iree_infinite_timeout(), &cmd->timepoint);
    }
    if (iree_status_is_ok(status)) {
      iree_task_wait_initialize(issue_task->scope,
                                iree_event_await(&cmd->timepoint.event),
                                IREE_TIME_INFINITE_FUTURE, &cmd->task);
      iree_task_set_cleanup_fn(&cmd->task.header,
                               iree_hal_task_semaphore_wait_cmd_cleanup);
      iree_task_set_completion_task(&cmd->task.header, issue_task);
      cmd->semaphore = semaphore;
      iree_hal_semaphore_retain(base_semaphore);
      iree_task_submission_enqueue(submission, &cmd->task.header);
    }
  }

  iree_slim_mutex_unlock(&semaphore->mutex);
  return status;
}

static iree_status_t iree_hal_task_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  if (!iree_status_is_ok(semaphore->failure_status)) {
    // Fastest path: failed; return an error to tell callers to query for it.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_status_from_code(IREE_STATUS_ABORTED);
  } else if (semaphore->current_value >= value) {
    // Fast path: already satisfied.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_ok_status();
  } else if (iree_timeout_is_immediate(timeout)) {
    // Not satisfied but a poll, so can avoid the expensive wait handle work.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  // Slow path: acquire a timepoint while we hold the lock.
  iree_hal_task_timepoint_t timepoint;
  iree_status_t status = iree_hal_task_semaphore_acquire_timepoint(
      semaphore, value, timeout, &timepoint);

  iree_slim_mutex_unlock(&semaphore->mutex);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) return status;

  // Wait until the timepoint resolves.
  // If satisfied the timepoint is automatically cleaned up and we are done. If
  // the deadline is reached before satisfied then we have to clean it up.
  status = iree_wait_one(&timepoint.event, deadline_ns);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_cancel_timepoint(&semaphore->base, &timepoint.base);
  }
  iree_event_pool_release(semaphore->event_pool, 1, &timepoint.event);

  return status;
}

iree_status_t iree_hal_task_semaphore_multi_wait(
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_event_pool_t* event_pool, iree_arena_block_pool_t* block_pool) {
  if (semaphore_list.count == 0) {
    return iree_ok_status();
  } else if (semaphore_list.count == 1) {
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

  // Acquire a wait handle for each semaphore timepoint we are to wait on.
  // TODO(benvanik): flip this API around so we can batch request events from
  // the event pool. We should be acquiring all required time points in one
  // call.
  iree_host_size_t timepoint_count = 0;
  iree_hal_task_timepoint_t* timepoints = NULL;
  iree_host_size_t total_timepoint_size =
      semaphore_list.count * sizeof(timepoints[0]);
  bool needs_wait = true;
  status =
      iree_arena_allocate(&arena, total_timepoint_size, (void**)&timepoints);
  if (iree_status_is_ok(status)) {
    memset(timepoints, 0, total_timepoint_size);
    for (iree_host_size_t i = 0; i < semaphore_list.count && needs_wait; ++i) {
      iree_hal_task_semaphore_t* semaphore =
          iree_hal_task_semaphore_cast(semaphore_list.semaphores[i]);
      iree_slim_mutex_lock(&semaphore->mutex);
      if (semaphore->current_value >= semaphore_list.payload_values[i]) {
        // Fast path: already satisfied.
        // If in ANY wait mode, this is sufficient and we don't actually need
        // to wait. This also skips acquiring timepoints for any remaining
        // semaphores. We still exit normally otherwise so as to cleanup
        // any timepoints already acquired.
        if (wait_mode == IREE_HAL_WAIT_MODE_ANY) {
          needs_wait = false;
        }
      } else {
        // Slow path: get a native wait handle for the timepoint.
        iree_hal_task_timepoint_t* timepoint = &timepoints[timepoint_count++];
        status = iree_hal_task_semaphore_acquire_timepoint(
            semaphore, semaphore_list.payload_values[i], timeout, timepoint);
        if (iree_status_is_ok(status)) {
          status = iree_wait_set_insert(wait_set, timepoint->event);
        }
      }
      iree_slim_mutex_unlock(&semaphore->mutex);
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

  // TODO(benvanik): if we flip the API to multi-acquire events from the pool
  // above then we can multi-release here too.
  for (iree_host_size_t i = 0; i < timepoint_count; ++i) {
    iree_hal_semaphore_t* semaphore = timepoints[i].semaphore;
    if (semaphore) {
      iree_hal_semaphore_cancel_timepoint(semaphore, &timepoints[i].base);
      iree_event_pool_release(event_pool, 1, &timepoints[i].event);
    }
  }
  iree_wait_set_free(wait_set);
  iree_arena_deinitialize(&arena);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_semaphore_vtable_t iree_hal_task_semaphore_vtable = {
    .destroy = iree_hal_task_semaphore_destroy,
    .query = iree_hal_task_semaphore_query,
    .signal = iree_hal_task_semaphore_signal,
    .fail = iree_hal_task_semaphore_fail,
    .wait = iree_hal_task_semaphore_wait,
};
