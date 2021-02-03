// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/local/task_semaphore.h"

#include <inttypes.h>

#include "iree/base/internal/wait_handle.h"
#include "iree/base/synchronization.h"
#include "iree/base/tracing.h"

// Sentinel used the semaphore has failed and an error status is set.
#define IREE_HAL_TASK_SEMAPHORE_FAILURE_VALUE UINT64_MAX

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
typedef struct iree_hal_task_timepoint_s {
  struct iree_hal_task_timepoint_s* next;
  struct iree_hal_task_timepoint_s* prev;
  uint64_t payload_value;
  iree_event_t event;
} iree_hal_task_timepoint_t;

// A doubly-linked FIFO list of timepoints.
// The order of the timepoints does *not* match increasing payload values but
// instead the order they were added to the list.
//
// Note that the timepoints are not owned by the list - this just nicely
// stitches together timepoints for the semaphore.
typedef struct {
  iree_hal_task_timepoint_t* head;
  iree_hal_task_timepoint_t* tail;
} iree_hal_task_timepoint_list_t;

static void iree_hal_task_timepoint_list_initialize(
    iree_hal_task_timepoint_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
}

// Moves |source_list| into |out_target_list|.
// |source_list| will be reset and the prior contents of |out_target_list| will
// be discarded.
static void iree_hal_task_timepoint_list_move(
    iree_hal_task_timepoint_list_t* source_list,
    iree_hal_task_timepoint_list_t* out_target_list) {
  memcpy(out_target_list, source_list, sizeof(*out_target_list));
  memset(source_list, 0, sizeof(*source_list));
}

// Appends a timepoint to the end of the timepoint list.
static void iree_hal_task_timepoint_list_append(
    iree_hal_task_timepoint_list_t* list,
    iree_hal_task_timepoint_t* timepoint) {
  timepoint->next = NULL;
  timepoint->prev = list->tail;
  if (list->tail != NULL) {
    list->tail->next = timepoint;
    list->tail = timepoint;
  } else {
    list->head = timepoint;
    list->tail = timepoint;
  }
}

// Erases a timepoint from the list.
static void iree_hal_task_timepoint_list_erase(
    iree_hal_task_timepoint_list_t* list,
    iree_hal_task_timepoint_t* timepoint) {
  if (timepoint->prev != NULL) timepoint->prev->next = timepoint->next;
  if (timepoint == list->head) list->head = timepoint->next;
  if (timepoint == list->tail) list->tail = timepoint->prev;
  timepoint->prev = NULL;
  timepoint->next = NULL;
}

// Scans the |pending_list| for all timepoints that are satisfied by the
// timeline having reached |payload_value|. Each satisfied timepoint will be
// moved to |out_ready_list|.
static void iree_hal_task_timepoint_list_take_ready(
    iree_hal_task_timepoint_list_t* pending_list, uint64_t payload_value,
    iree_hal_task_timepoint_list_t* out_ready_list) {
  iree_hal_task_timepoint_list_initialize(out_ready_list);
  iree_hal_task_timepoint_t* next = pending_list->head;
  while (next != NULL) {
    iree_hal_task_timepoint_t* timepoint = next;
    next = timepoint->next;
    bool is_satisfied = timepoint->payload_value <= payload_value;
    if (!is_satisfied) continue;

    // Remove from pending list.
    iree_hal_task_timepoint_list_erase(pending_list, timepoint);

    // Add to ready list.
    iree_hal_task_timepoint_list_append(out_ready_list, timepoint);
  }
}

// Notifies all of the timepoints in the |ready_list| that their condition has
// been satisfied. |ready_list| will be reset as ownership of the events is
// held by the originator.
static void iree_hal_task_timepoint_list_notify_ready(
    iree_hal_task_timepoint_list_t* ready_list) {
  iree_hal_task_timepoint_t* next = ready_list->head;
  while (next != NULL) {
    iree_hal_task_timepoint_t* timepoint = next;
    next = timepoint->next;
    timepoint->next = NULL;
    timepoint->prev = NULL;
    iree_event_set(&timepoint->event);
  }
  iree_hal_task_timepoint_list_initialize(ready_list);
}

//===----------------------------------------------------------------------===//
// iree_hal_task_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_local_event_pool_t* event_pool;

  // Guards all mutable fields. We expect low contention on semaphores and since
  // iree_slim_mutex_t is (effectively) just a CAS this keeps things simpler
  // than trying to make the entire structure lock-free.
  iree_slim_mutex_t mutex;

  // Current signaled value. May be IREE_HAL_TASK_SEMAPHORE_FAILURE_VALUE to
  // indicate that the semaphore has been signaled for failure and
  // |failure_status| contains the error.
  uint64_t current_value;

  // OK or the status passed to iree_hal_semaphore_fail. Owned by the semaphore.
  iree_status_t failure_status;

  // In-process notification signaled when the semaphore value changes. This is
  // used exclusively for wait-ones to avoid going to the kernel for a full wait
  // handle operation.
  iree_notification_t notification;

  // A list of all reserved timepoints waiting for the semaphore to reach a
  // certain payload value.
  iree_hal_task_timepoint_list_t timepoint_list;
} iree_hal_task_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_task_semaphore_vtable;

static iree_hal_task_semaphore_t* iree_hal_task_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_task_semaphore_vtable);
  return (iree_hal_task_semaphore_t*)base_value;
}

iree_status_t iree_hal_task_semaphore_create(
    iree_hal_local_event_pool_t* event_pool, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(event_pool);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_task_semaphore_vtable,
                                 &semaphore->resource);
    semaphore->host_allocator = host_allocator;
    semaphore->event_pool = event_pool;

    iree_slim_mutex_initialize(&semaphore->mutex);
    semaphore->current_value = initial_value;
    semaphore->failure_status = iree_ok_status();
    iree_notification_initialize(&semaphore->notification);
    iree_hal_task_timepoint_list_initialize(&semaphore->timepoint_list);

    *out_semaphore = (iree_hal_semaphore_t*)semaphore;
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

  iree_status_free(semaphore->failure_status);
  iree_notification_deinitialize(&semaphore->notification);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_task_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  *out_value = semaphore->current_value;

  iree_status_t status = iree_ok_status();
  if (*out_value >= IREE_HAL_TASK_SEMAPHORE_FAILURE_VALUE) {
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
    uint64_t current_value = semaphore->current_value;
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "semaphore values must be monotonically "
                            "increasing; current_value=%" PRIu64
                            ", new_value=%" PRIu64,
                            current_value, new_value);
  }

  semaphore->current_value = new_value;

  // Scan for all timepoints that are now satisfied and move them to our local
  // ready list. This way we can notify them without needing to continue holding
  // the semaphore lock.
  iree_hal_task_timepoint_list_t ready_list;
  iree_hal_task_timepoint_list_take_ready(&semaphore->timepoint_list, new_value,
                                          &ready_list);

  iree_notification_post(&semaphore->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&semaphore->mutex);

  // Notify all waiters - note that this must happen outside the lock.
  iree_hal_task_timepoint_list_notify_ready(&ready_list);

  return iree_ok_status();
}

static void iree_hal_task_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                         iree_status_t status) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);

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
  semaphore->current_value = IREE_HAL_TASK_SEMAPHORE_FAILURE_VALUE;
  semaphore->failure_status = status;

  // Take the whole timepoint list as we'll be signaling all of them. Since
  // we hold the lock no other timepoints can be created while we are cleaning
  // up.
  iree_hal_task_timepoint_list_t ready_list;
  iree_hal_task_timepoint_list_move(&semaphore->timepoint_list, &ready_list);

  iree_notification_post(&semaphore->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&semaphore->mutex);

  // Notify all waiters - note that this must happen outside the lock.
  iree_hal_task_timepoint_list_notify_ready(&ready_list);
}

// Acquires a timepoint waiting for the given value.
// |out_timepoint| is owned by the caller and must be kept live until the
// timepoint has been reached (or it is cancelled by the caller).
static iree_status_t iree_hal_task_semaphore_acquire_timepoint(
    iree_hal_task_semaphore_t* semaphore, uint64_t minimum_value,
    iree_hal_task_timepoint_t* out_timepoint) {
  memset(out_timepoint, 0, sizeof(*out_timepoint));
  out_timepoint->payload_value = minimum_value;
  IREE_RETURN_IF_ERROR(iree_hal_local_event_pool_acquire(
      semaphore->event_pool, 1, &out_timepoint->event));
  iree_hal_task_timepoint_list_append(&semaphore->timepoint_list,
                                      out_timepoint);
  return iree_ok_status();
}

typedef struct {
  iree_task_wait_t task;
  iree_hal_task_semaphore_t* semaphore;
  iree_hal_task_timepoint_t timepoint;
} iree_hal_task_semaphore_wait_cmd_t;

// Cleans up a wait task by returning the event used to the pool and - if the
// task failed - ensuring we scrub it from the timepoint list.
static void iree_hal_task_semaphore_wait_cmd_cleanup(iree_task_t* task,
                                                     iree_status_t status) {
  iree_hal_task_semaphore_wait_cmd_t* cmd =
      (iree_hal_task_semaphore_wait_cmd_t*)task;
  iree_hal_local_event_pool_release(cmd->semaphore->event_pool, 1,
                                    &cmd->timepoint.event);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // Abort the timepoint. Note that this is not designed to be fast as
    // semaphore failure is an exceptional case.
    iree_slim_mutex_lock(&cmd->semaphore->mutex);
    iree_hal_task_timepoint_list_erase(&cmd->semaphore->timepoint_list,
                                       &cmd->timepoint);
    iree_slim_mutex_unlock(&cmd->semaphore->mutex);
  }
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
  } else {
    // Slow path: acquire a system wait handle and perform a full wait.
    iree_hal_task_semaphore_wait_cmd_t* cmd = NULL;
    status = iree_arena_allocate(arena, sizeof(*cmd), (void**)&cmd);
    if (iree_status_is_ok(status)) {
      status = iree_hal_task_semaphore_acquire_timepoint(
          semaphore, minimum_value, &cmd->timepoint);
    }
    if (iree_status_is_ok(status)) {
      iree_task_wait_initialize(issue_task->scope, cmd->timepoint.event,
                                &cmd->task);
      iree_task_set_cleanup_fn(&cmd->task.header,
                               iree_hal_task_semaphore_wait_cmd_cleanup);
      iree_task_set_completion_task(&cmd->task.header, issue_task);
      cmd->semaphore = semaphore;
      iree_task_submission_enqueue(submission, &cmd->task.header);
    }
  }

  iree_slim_mutex_unlock(&semaphore->mutex);
  return status;
}

static iree_status_t iree_hal_task_semaphore_wait_with_deadline(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_time_t deadline_ns) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);

  iree_slim_mutex_lock(&semaphore->mutex);

  if (semaphore->current_value >= value) {
    // Fast path: already satisfied.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_ok_status();
  } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    // Not satisfied but a poll, so can avoid the expensive wait handle work.
    iree_slim_mutex_unlock(&semaphore->mutex);
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  // Slow path: acquire a timepoint while we hold the lock.
  iree_hal_task_timepoint_t timepoint;
  iree_status_t status =
      iree_hal_task_semaphore_acquire_timepoint(semaphore, value, &timepoint);

  iree_slim_mutex_unlock(&semaphore->mutex);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) return status;

  // Wait until the timepoint resolves.
  // If satisfied the timepoint is automatically cleaned up and we are done. If
  // the deadline is reached before satisfied then we have to clean it up.
  status = iree_wait_one(&timepoint.event, deadline_ns);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&semaphore->mutex);
    iree_hal_task_timepoint_list_erase(&semaphore->timepoint_list, &timepoint);
    iree_slim_mutex_unlock(&semaphore->mutex);
  }
  iree_hal_local_event_pool_release(semaphore->event_pool, 1, &timepoint.event);
  return status;
}

static iree_status_t iree_hal_task_semaphore_wait_with_timeout(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_duration_t timeout_ns) {
  return iree_hal_task_semaphore_wait_with_deadline(
      base_semaphore, value, iree_relative_timeout_to_deadline_ns(timeout_ns));
}

iree_status_t iree_hal_task_semaphore_multi_wait(
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns,
    iree_hal_local_event_pool_t* event_pool,
    iree_arena_block_pool_t* block_pool) {
  IREE_ASSERT_ARGUMENT(semaphore_list);
  if (semaphore_list->count == 0) {
    return iree_ok_status();
  } else if (semaphore_list->count == 1) {
    // Fast-path for a single semaphore.
    return iree_hal_semaphore_wait_with_deadline(
        semaphore_list->semaphores[0], semaphore_list->payload_values[0],
        deadline_ns);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Avoid heap allocations by using the device block pool for the wait set.
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  iree_wait_set_t* wait_set = NULL;
  iree_status_t status = iree_wait_set_allocate(
      semaphore_list->count, iree_arena_allocator(&arena), &wait_set);

  // Acquire a wait handle for each semaphore timepoint we are to wait on.
  // TODO(benvanik): flip this API around so we can batch request events from
  // the event pool. We should be acquiring all required time points in one
  // call.
  iree_host_size_t timepoint_count = 0;
  iree_hal_task_timepoint_t* timepoints = NULL;
  iree_host_size_t total_timepoint_size =
      semaphore_list->count * sizeof(timepoints[0]);
  status =
      iree_arena_allocate(&arena, total_timepoint_size, (void**)&timepoints);
  if (iree_status_is_ok(status)) {
    memset(timepoints, 0, total_timepoint_size);
    for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
      iree_hal_task_semaphore_t* semaphore =
          iree_hal_task_semaphore_cast(semaphore_list->semaphores[i]);
      iree_slim_mutex_lock(&semaphore->mutex);
      if (semaphore->current_value >= semaphore_list->payload_values[i]) {
        // Fast path: already satisfied.
      } else {
        // Slow path: get a native wait handle for the timepoint.
        iree_hal_task_timepoint_t* timepoint = &timepoints[timepoint_count++];
        status = iree_hal_task_semaphore_acquire_timepoint(
            semaphore, semaphore_list->payload_values[i], timepoint);
        if (iree_status_is_ok(status)) {
          status = iree_wait_set_insert(wait_set, timepoint->event);
        }
      }
      iree_slim_mutex_unlock(&semaphore->mutex);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Perform the wait.
  if (iree_status_is_ok(status)) {
    if (wait_mode == IREE_HAL_WAIT_MODE_ANY) {
      status = iree_wait_any(wait_set, deadline_ns, /*out_wake_handle=*/NULL);
    } else {
      status = iree_wait_all(wait_set, deadline_ns);
    }
  }

  if (timepoints != NULL) {
    // TODO(benvanik): if we flip the API to multi-acquire events from the pool
    // above then we can multi-release here too.
    for (iree_host_size_t i = 0; i < timepoint_count; ++i) {
      iree_hal_local_event_pool_release(event_pool, 1, &timepoints[i].event);
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
    .wait_with_deadline = iree_hal_task_semaphore_wait_with_deadline,
    .wait_with_timeout = iree_hal_task_semaphore_wait_with_timeout,
};
