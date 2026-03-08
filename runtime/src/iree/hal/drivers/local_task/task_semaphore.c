// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_semaphore.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/internal/wait_handle.h"
#include "iree/hal/utils/semaphore_base.h"
#include "iree/task/executor.h"
#include "iree/task/scope.h"

//===----------------------------------------------------------------------===//
// iree_hal_task_timepoint_t (legacy — used only for synchronous blocking waits)
//===----------------------------------------------------------------------===//

typedef struct iree_hal_task_timepoint_t {
  iree_hal_semaphore_timepoint_t base;
  iree_hal_semaphore_t* semaphore;
  iree_event_t event;
} iree_hal_task_timepoint_t;

static iree_status_t iree_hal_task_semaphore_timepoint_callback(
    void* user_data, iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_status_code_t status_code) {
  iree_hal_task_timepoint_t* timepoint = (iree_hal_task_timepoint_t*)user_data;
  iree_event_set(&timepoint->event);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Direct semaphore wait (async task submission path)
//===----------------------------------------------------------------------===//

// Arena-allocated wrapper for a direct semaphore timepoint that feeds completed
// waits back into the task executor without intermediate events or wait tasks.
//
// When the semaphore reaches the target value (or fails), the callback
// decrements the issue task's pending_dependency_count. If this was the last
// outstanding dependency, the issue task is submitted directly to the executor.
typedef struct iree_hal_task_semaphore_direct_wait_t {
  // Timepoint registered with the async semaphore (intrusive list storage).
  iree_async_semaphore_timepoint_t timepoint;
  // Executor to submit ready tasks to.
  iree_task_executor_t* executor;
  // The task waiting on this semaphore. Has had its pending_dependency_count
  // incremented to account for this wait.
  iree_task_t* issue_task;
  // Retained reference to the semaphore (released in callback).
  iree_hal_semaphore_t* semaphore;
} iree_hal_task_semaphore_direct_wait_t;

// Callback fired by the async semaphore when the target value is reached or the
// semaphore fails. Fires with the semaphore's internal lock held — must be fast
// and must not call signal/fail on the same semaphore.
//
// Replicates the retirement logic from iree_task_retire: on success the
// dependency count is decremented and the task submitted when ready; on failure
// the scope is marked failed and the task is either flagged ABORTED (if other
// deps are still pending) or discarded immediately (if this was the last dep).
static void iree_hal_task_semaphore_direct_wait_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_task_semaphore_direct_wait_t* wait =
      (iree_hal_task_semaphore_direct_wait_t*)user_data;
  iree_task_t* issue_task = wait->issue_task;
  iree_task_executor_t* executor = wait->executor;

  if (!iree_status_is_ok(status)) {
    // Semaphore failed — notify the scope. scope_fail takes ownership.
    iree_task_scope_t* scope = issue_task->scope;
    iree_task_scope_fail(scope, status);

    // Decrement the dependency count.
    int32_t previous_count = iree_atomic_fetch_sub(
        &issue_task->pending_dependency_count, 1, iree_memory_order_acq_rel);
    if (previous_count == 1) {
      // Last dependency — discard the issue task and its subtree.
      iree_task_scope_begin(scope);
      iree_task_list_t discard_worklist;
      iree_task_list_initialize(&discard_worklist);
      iree_task_discard(issue_task, &discard_worklist);
      iree_task_list_discard(&discard_worklist);
      iree_task_scope_end(scope);
    } else {
      // Other dependencies still pending — flag for abort so the last one
      // to complete will discard rather than execute.
      issue_task->flags |= IREE_TASK_FLAG_ABORTED;
    }
  } else {
    // Semaphore reached the target value.
    int32_t previous_count = iree_atomic_fetch_sub(
        &issue_task->pending_dependency_count, 1, iree_memory_order_acq_rel);
    if (previous_count == 1) {
      // Last dependency satisfied — submit the issue task for execution.
      iree_task_submission_t submission;
      iree_task_submission_initialize(&submission);
      iree_task_submission_enqueue(&submission, issue_task);
      iree_task_executor_submit(executor, &submission);
      iree_task_executor_flush(executor);
    }
  }

  iree_hal_semaphore_release(wait->semaphore);
}

//===----------------------------------------------------------------------===//
// iree_hal_task_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_task_semaphore_t {
  iree_hal_semaphore_t base;
  iree_allocator_t host_allocator;
  iree_event_pool_t* event_pool;
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
  iree_host_size_t frontier_offset = 0, total_size = 0;
  iree_status_t status = iree_async_semaphore_layout(
      sizeof(*semaphore), 0, &frontier_offset, &total_size);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_initialize(&iree_hal_task_semaphore_vtable,
                                  initial_value, frontier_offset, 0,
                                  &semaphore->base);
    semaphore->host_allocator = host_allocator;
    semaphore->event_pool = event_pool;

    *out_semaphore = &semaphore->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_task_semaphore_destroy(
    iree_async_semaphore_t* base_semaphore) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(iree_hal_semaphore_cast(base_semaphore));
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_task_semaphore_isa(iree_hal_semaphore_t* semaphore) {
  return iree_hal_resource_is((const iree_hal_resource_t*)semaphore,
                              &iree_hal_task_semaphore_vtable);
}

static uint64_t iree_hal_task_semaphore_query(
    iree_async_semaphore_t* base_semaphore) {
  iree_async_semaphore_t* async_sem = (iree_async_semaphore_t*)base_semaphore;

  // Both fields are atomic — fully lock-free query.
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_hal_status_as_semaphore_failure(failure);
  }
  return (uint64_t)iree_atomic_load(&async_sem->timeline_value,
                                    iree_memory_order_acquire);
}

static iree_status_t iree_hal_task_semaphore_signal(
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

static void iree_hal_task_semaphore_fail(iree_async_semaphore_t* base_semaphore,
                                         iree_status_t status) {
  // First failure wins via CAS. Clone for storage, pass original to dispatch.
  iree_status_t stored = iree_status_clone(status);
  intptr_t expected = 0;
  iree_async_semaphore_t* async_sem = (iree_async_semaphore_t*)base_semaphore;
  if (!iree_atomic_compare_exchange_strong(
          &async_sem->failure_status, &expected, (intptr_t)stored,
          iree_memory_order_release, iree_memory_order_acquire)) {
    iree_status_free(stored);
    iree_status_free(status);
    return;
  }

  // Dispatch all pending timepoints with the failure status.
  // Takes ownership of |status| (clones per-timepoint, frees original).
  iree_async_semaphore_dispatch_timepoints_failed(base_semaphore, status);
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

iree_status_t iree_hal_task_semaphore_enqueue_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t minimum_value,
    iree_task_t* issue_task, iree_task_executor_t* executor,
    iree_arena_allocator_t* arena) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  iree_async_semaphore_t* async_sem = &semaphore->base.async;

  // Fast path: check failure and satisfaction atomically (no lock needed).
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &async_sem->timeline_value, iree_memory_order_acquire);
  if (current_value >= minimum_value) {
    return iree_ok_status();
  }

  // Slow path: register a direct timepoint on the async semaphore. When the
  // value is reached (or the semaphore fails), the callback decrements
  // issue_task's pending_dependency_count and submits it to the executor
  // when all dependencies are satisfied.
  iree_hal_task_semaphore_direct_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*wait), (void**)&wait));

  wait->executor = executor;
  wait->issue_task = issue_task;
  wait->semaphore = base_semaphore;
  iree_hal_semaphore_retain(base_semaphore);

  // Register ourselves as a pending dependency of the issue task.
  // The callback will decrement this when the semaphore is satisfied.
  iree_atomic_fetch_add(&issue_task->pending_dependency_count, 1,
                        iree_memory_order_acq_rel);

  // Set up the timepoint and insert it into the semaphore's list.
  // The callback may fire synchronously if the value was reached between
  // our fast-path check and the insertion (the semaphore rechecks under lock).
  wait->timepoint.callback = iree_hal_task_semaphore_direct_wait_resolved;
  wait->timepoint.user_data = wait;
  return iree_async_semaphore_insert_timepoint(async_sem, minimum_value,
                                               &wait->timepoint);
}

static iree_status_t iree_hal_task_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_hal_wait_flags_t flags) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  iree_async_semaphore_t* async_sem = &semaphore->base.async;

  // Fast paths using atomic reads (no lock needed).
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &async_sem->timeline_value, iree_memory_order_acquire);
  if (current_value >= value) return iree_ok_status();

  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }

  if (iree_timeout_is_immediate(timeout)) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  // Slow path: acquire a timepoint (handles its own synchronization).
  iree_hal_task_timepoint_t timepoint;
  iree_status_t status = iree_hal_task_semaphore_acquire_timepoint(
      semaphore, value, timeout, &timepoint);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) return status;

  // Wait until the timepoint resolves.
  // If satisfied the timepoint is automatically cleaned up and we are done. If
  // the deadline is reached before satisfied then we have to clean it up.
  status = iree_wait_one(&timepoint.event, deadline_ns);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_cancel_timepoint(&semaphore->base, &timepoint.base);
  }
  iree_event_pool_release(semaphore->event_pool, 1, &timepoint.event);

  // Recheck after waking. Both fields are atomic.
  if (iree_status_is_ok(status)) {
    current_value = (uint64_t)iree_atomic_load(&async_sem->timeline_value,
                                               iree_memory_order_acquire);
    if (current_value >= value) return iree_ok_status();
    failure = (iree_status_t)iree_atomic_load(&async_sem->failure_status,
                                              iree_memory_order_acquire);
    return iree_status_from_code(!iree_status_is_ok(failure)
                                     ? IREE_STATUS_ABORTED
                                     : IREE_STATUS_DEADLINE_EXCEEDED);
  }

  return status;
}

iree_status_t iree_hal_task_semaphore_multi_wait(
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags, iree_event_pool_t* event_pool,
    iree_arena_block_pool_t* block_pool) {
  if (semaphore_list.count == 0) {
    return iree_ok_status();
  } else if (semaphore_list.count == 1) {
    // Fast-path for a single semaphore.
    return iree_hal_semaphore_wait(semaphore_list.semaphores[0],
                                   semaphore_list.payload_values[0], timeout,
                                   flags);
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
  iree_host_size_t total_timepoint_size = 0;
  bool needs_wait = true;
  if (!iree_host_size_checked_mul(semaphore_list.count, sizeof(timepoints[0]),
                                  &total_timepoint_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "semaphore list count overflow");
  } else {
    status =
        iree_arena_allocate(&arena, total_timepoint_size, (void**)&timepoints);
  }
  if (iree_status_is_ok(status)) {
    memset(timepoints, 0, total_timepoint_size);
    for (iree_host_size_t i = 0; i < semaphore_list.count && needs_wait; ++i) {
      iree_hal_task_semaphore_t* semaphore =
          iree_hal_task_semaphore_cast(semaphore_list.semaphores[i]);
      iree_async_semaphore_t* async_sem = &semaphore->base.async;

      // Atomic checks — no lock needed.
      iree_status_t failure = (iree_status_t)iree_atomic_load(
          &async_sem->failure_status, iree_memory_order_acquire);
      if (!iree_status_is_ok(failure)) {
        status = iree_status_from_code(IREE_STATUS_ABORTED);
        break;
      }
      uint64_t current_value = (uint64_t)iree_atomic_load(
          &async_sem->timeline_value, iree_memory_order_acquire);
      if (current_value >= semaphore_list.payload_values[i]) {
        // Already satisfied. In ANY mode we can skip remaining semaphores.
        // We still exit normally otherwise so as to cleanup any timepoints
        // already acquired.
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

  // Post-wait: recheck all semaphores for failures that may have caused the
  // wake. The wait handles fire for both signal and failure, so a successful
  // wait return does not guarantee the semaphores are in a healthy state.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
      iree_status_t failure = (iree_status_t)iree_atomic_load(
          &semaphore_list.semaphores[i]->async.failure_status,
          iree_memory_order_acquire);
      if (!iree_status_is_ok(failure)) {
        status = iree_status_from_code(IREE_STATUS_ABORTED);
        break;
      }
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

static iree_status_t iree_hal_task_semaphore_import_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_task_semaphore_export_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_external_timepoint_type_t requested_type,
    iree_hal_external_timepoint_flags_t requested_flags,
    iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "timepoint export is not yet implemented");
}

static const iree_hal_semaphore_vtable_t iree_hal_task_semaphore_vtable = {
    .async =
        {
            .destroy = iree_hal_task_semaphore_destroy,
            .query = iree_hal_task_semaphore_query,
            .signal = iree_hal_task_semaphore_signal,
            .query_frontier = iree_hal_semaphore_default_query_frontier,
            .fail = iree_hal_task_semaphore_fail,
            .acquire_timepoint = iree_hal_semaphore_default_acquire_timepoint,
            .cancel_timepoint = iree_hal_semaphore_default_cancel_timepoint,
            .export_primitive = iree_hal_semaphore_default_export_primitive,
        },
    .wait = iree_hal_task_semaphore_wait,
    .import_timepoint = iree_hal_task_semaphore_import_timepoint,
    .export_timepoint = iree_hal_task_semaphore_export_timepoint,
};
