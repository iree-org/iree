// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_semaphore.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/async/semaphore.h"
#include "iree/task/executor.h"
#include "iree/task/scope.h"

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
// semaphore fails. Fires without the semaphore lock held.
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
  iree_hal_semaphore_t* semaphore = wait->semaphore;

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

  // NOTE: wait may be freed by executor flush above (arena freed inline).
  // Use the stack-local semaphore captured before any executor interaction.
  iree_hal_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// iree_hal_task_semaphore_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_task_semaphore_t {
  iree_async_semaphore_t async;
  iree_allocator_t host_allocator;
} iree_hal_task_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_task_semaphore_vtable;

static iree_hal_task_semaphore_t* iree_hal_task_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_task_semaphore_vtable);
  return (iree_hal_task_semaphore_t*)base_value;
}

iree_status_t iree_hal_task_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(proactor);
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
    iree_async_semaphore_initialize(
        (const iree_async_semaphore_vtable_t*)&iree_hal_task_semaphore_vtable,
        proactor, initial_value, frontier_offset, 0, &semaphore->async);
    semaphore->host_allocator = host_allocator;

    *out_semaphore = iree_hal_semaphore_cast(&semaphore->async);
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

  iree_async_semaphore_deinitialize(&semaphore->async);
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

iree_status_t iree_hal_task_semaphore_enqueue_timepoint(
    iree_hal_semaphore_t* base_semaphore, uint64_t minimum_value,
    iree_task_t* issue_task, iree_task_executor_t* executor,
    iree_arena_allocator_t* arena) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  iree_async_semaphore_t* async_sem = &semaphore->async;

  // Fast path: check failure and satisfaction atomically (no lock needed).
  iree_status_t failure = (iree_status_t)iree_atomic_load(
      &async_sem->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_status_from_code(iree_status_code(failure));
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
  return iree_async_semaphore_acquire_timepoint(async_sem, minimum_value,
                                                &wait->timepoint);
}

static iree_status_t iree_hal_task_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  // Delegate to the centralized async semaphore wait which uses a stack-local
  // futex-based notification — no event pool allocation or eventfd overhead.
  return iree_async_semaphore_multi_wait(
      IREE_ASYNC_WAIT_MODE_ALL, (iree_async_semaphore_t**)&base_semaphore,
      &value, 1, timeout, flags, iree_allocator_system());
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
        },
    .wait = iree_hal_task_semaphore_wait,
    .import_timepoint = iree_hal_task_semaphore_import_timepoint,
    .export_timepoint = iree_hal_task_semaphore_export_timepoint,
};
