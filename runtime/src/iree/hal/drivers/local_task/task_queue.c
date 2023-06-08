// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_queue.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/tracing.h"
#include "iree/hal/drivers/local_task/task_command_buffer.h"
#include "iree/hal/drivers/local_task/task_semaphore.h"
#include "iree/task/submission.h"

// Each submission is turned into a DAG for execution:
//
//  +--------------------+    To preserve the sequential issue order an edge is
//  |  (previous issue)  |    added between the previous outstanding issue (if
//  +--------------------+    it exists) such that all issues run in the order
//    |                       they were submitted to the queue. Note that this
//    v                       is *only* the issue; the commands issued by two
//  +--------------------+    submissions may still overlap and are only
//  |  sequence barrier  |    guaranteed to begin execution in order.
//  +--------------------+
//    |
//    |   +--------------+
//    +-> | +--------------+  Unsatisfied waits are scheduled as wait tasks and
//    .   +-|  sema waits  |  block the issuing of commands until all have
//    .     +--------------+  been satisfied. If the wait is immediately
//    .        | | | | |      following a signal from the same queue then it
//    +--------+-+-+-+-+      elided - only cross-queue or external waits
//    |                       actually go down to system wait handles.
//    v
//  +--------------------+    Command buffers in the batch are issued in-order
//  |   command issue    |    as if all commands had been recorded into the same
//  +--------------------+    command buffer (excluding recording state like
//    |                       push constants). The dependencies between commands
//    |   +--------------+    are determined by the events and barriers recorded
//    +-> | +--------------+  in each command buffer.
//    .   +-|   commands   |
//    .     +--------------+
//    .        | | | | |
//    +--------+-+-+-+-+
//    |
//    v
//  +--------------------+    After all commands within the batch complete the
//  | semaphore signals  |    submission is retired and all semaphores are
//  +--------------------+    signaled. Note that this may happen *before* other
//    |                       earlier submissions complete if there were no
//   ...                      dependencies between the commands in each batch.
//
// Could this be simplified? Probably. Improvements to the task system to allow
// for efficient multiwaits and better stitching of independent DAGs would help.

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Clones a list of semaphores into an |arena| and initializes |out_target_list|
// to reference the newly-cloned data.
static iree_status_t iree_hal_semaphore_list_clone(
    const iree_hal_semaphore_list_t* source_list, iree_arena_allocator_t* arena,
    iree_hal_semaphore_list_t* out_target_list) {
  iree_host_size_t semaphores_size =
      source_list->count * sizeof(out_target_list->semaphores[0]);
  iree_host_size_t payload_values_size =
      source_list->count * sizeof(out_target_list->payload_values[0]);
  iree_host_size_t total_size = semaphores_size + payload_values_size;
  uint8_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, total_size, (void**)&buffer));

  out_target_list->count = source_list->count;
  out_target_list->semaphores = (iree_hal_semaphore_t**)buffer;
  out_target_list->payload_values = (uint64_t*)(buffer + semaphores_size);

  for (iree_host_size_t i = 0; i < source_list->count; ++i) {
    out_target_list->semaphores[i] = source_list->semaphores[i];
    iree_hal_semaphore_retain(out_target_list->semaphores[i]);
    out_target_list->payload_values[i] = source_list->payload_values[i];
  }

  return iree_ok_status();
}

static void iree_hal_semaphore_list_release(iree_hal_semaphore_list_t* list) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_hal_semaphore_release(list->semaphores[i]);
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_wait_cmd_t
//===----------------------------------------------------------------------===//

// Task to fork out and wait on one or more semaphores.
// This optimizes for same-queue semaphore chaining by ensuring that semaphores
// used to stitch together subsequent submissions never have to go to the system
// to wait as the implicit queue ordering ensures that the signals would have
// happened prior to the sequence command being executed. Cross-queue semaphores
// will still cause waits if they have not yet been signaled.
typedef struct iree_hal_task_queue_wait_cmd_t {
  // Call to iree_hal_task_queue_wait_cmd.
  iree_task_call_t task;

  // Arena used for the submission - additional tasks can be allocated from
  // this.
  iree_arena_allocator_t* arena;

  // A list of semaphores to wait on prior to issuing the rest of the
  // submission.
  iree_hal_semaphore_list_t wait_semaphores;
} iree_hal_task_queue_wait_cmd_t;

// Forks out multiple wait tasks prior to issuing the commands.
static iree_status_t iree_hal_task_queue_wait_cmd(
    void* user_context, iree_task_t* task,
    iree_task_submission_t* pending_submission) {
  iree_hal_task_queue_wait_cmd_t* cmd = (iree_hal_task_queue_wait_cmd_t*)task;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < cmd->wait_semaphores.count; ++i) {
    status = iree_hal_task_semaphore_enqueue_timepoint(
        cmd->wait_semaphores.semaphores[i],
        cmd->wait_semaphores.payload_values[i],
        cmd->task.header.completion_task, cmd->arena, pending_submission);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Cleanup for iree_hal_task_queue_wait_cmd_t that releases the retained
// semaphores.
static void iree_hal_task_queue_wait_cmd_cleanup(
    iree_task_t* task, iree_status_code_t status_code) {
  iree_hal_task_queue_wait_cmd_t* cmd = (iree_hal_task_queue_wait_cmd_t*)task;
  iree_hal_semaphore_list_release(&cmd->wait_semaphores);
}

// Allocates and initializes a iree_hal_task_queue_wait_cmd_t task.
static iree_status_t iree_hal_task_queue_wait_cmd_allocate(
    iree_task_scope_t* scope, const iree_hal_semaphore_list_t* wait_semaphores,
    iree_arena_allocator_t* arena, iree_hal_task_queue_wait_cmd_t** out_cmd) {
  iree_hal_task_queue_wait_cmd_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*cmd), (void**)&cmd));
  iree_task_call_initialize(
      scope, iree_task_make_call_closure(iree_hal_task_queue_wait_cmd, 0),
      &cmd->task);
  iree_task_set_cleanup_fn(&cmd->task.header,
                           iree_hal_task_queue_wait_cmd_cleanup);
  cmd->arena = arena;

  // Clone the wait semaphores from the batch - we retain them and their
  // payloads.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_clone(wait_semaphores, arena,
                                                     &cmd->wait_semaphores));

  *out_cmd = cmd;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_issue_cmd_t
//===----------------------------------------------------------------------===//

// Task to issue all the command buffers in the batch.
// After this task completes the commands have been issued but have not yet
// completed and the issued commands may complete in any order.
typedef struct iree_hal_task_queue_issue_cmd_t {
  // Call to iree_hal_task_queue_issue_cmd.
  iree_task_call_t task;

  // Arena used for the submission - additional tasks can be allocated from
  // this.
  iree_arena_allocator_t* arena;

  // Nasty back reference to the queue so that we can clear the tail_issue_task
  // if we are the last issue pending.
  iree_hal_task_queue_t* queue;

  // Command buffers to be issued in the order the appeared in the submission.
  iree_host_size_t command_buffer_count;
  iree_hal_command_buffer_t* command_buffers[];
} iree_hal_task_queue_issue_cmd_t;

// Issues a set of command buffers without waiting for them to complete.
static iree_status_t iree_hal_task_queue_issue_cmd(
    void* user_context, iree_task_t* task,
    iree_task_submission_t* pending_submission) {
  iree_hal_task_queue_issue_cmd_t* cmd = (iree_hal_task_queue_issue_cmd_t*)task;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // NOTE: it's ok for there to be no command buffers - in that case the
  // submission was purely for synchronization.
  if (cmd->command_buffer_count > 0) {
    for (iree_host_size_t i = 0; i < cmd->command_buffer_count; ++i) {
      if (iree_hal_task_command_buffer_isa(cmd->command_buffers[i])) {
        status = iree_hal_task_command_buffer_issue(
            cmd->command_buffers[i], &cmd->queue->state,
            cmd->task.header.completion_task, cmd->arena, pending_submission);
        iree_hal_command_buffer_release(cmd->command_buffers[i]);
        cmd->command_buffers[i] = NULL;
      } else {
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "unsupported command buffer type for task queue submission");
      }
      if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Cleanup for iree_hal_task_queue_issue_cmd_t to drop all resources.
// Any that need to remain live during execution are retained by the tasks
// performing that execution.
static void iree_hal_task_queue_issue_cmd_cleanup(
    iree_task_t* task, iree_status_code_t status_code) {
  iree_hal_task_queue_issue_cmd_t* cmd = (iree_hal_task_queue_issue_cmd_t*)task;

  // Release command buffers; some may have been released after issuing but this
  // handles leftovers that may appear due to failures.
  for (iree_host_size_t i = 0; i < cmd->command_buffer_count; ++i) {
    iree_hal_command_buffer_release(cmd->command_buffers[i]);
    cmd->command_buffers[i] = NULL;
  }
}

// Allocates and initializes a iree_hal_task_queue_issue_cmd_t task.
static iree_status_t iree_hal_task_queue_issue_cmd_allocate(
    iree_task_scope_t* scope, iree_hal_task_queue_t* queue,
    iree_task_t* retire_task, iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_arena_allocator_t* arena, iree_hal_task_queue_issue_cmd_t** out_cmd) {
  iree_hal_task_queue_issue_cmd_t* cmd = NULL;
  iree_host_size_t total_cmd_size =
      sizeof(*cmd) + command_buffer_count * sizeof(*cmd->command_buffers);
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, total_cmd_size, (void**)&cmd));
  iree_task_call_initialize(
      scope, iree_task_make_call_closure(iree_hal_task_queue_issue_cmd, 0),
      &cmd->task);
  iree_task_set_completion_task(&cmd->task.header, retire_task);
  iree_task_set_cleanup_fn(&cmd->task.header,
                           iree_hal_task_queue_issue_cmd_cleanup);
  cmd->arena = arena;
  cmd->queue = queue;

  cmd->command_buffer_count = command_buffer_count;
  for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
    cmd->command_buffers[i] = command_buffers[i];
    iree_hal_command_buffer_retain(cmd->command_buffers[i]);
  }

  *out_cmd = cmd;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_retire_cmd_t
//===----------------------------------------------------------------------===//

// Task to retire the submission and free the transient memory allocated for
// it. The task is issued only once all commands from all command buffers in
// the submission complete. Semaphores will be signaled and dependent
// submissions may be issued.
typedef struct iree_hal_task_queue_retire_cmd_t {
  // Call to iree_hal_task_queue_retire_cmd.
  iree_task_call_t task;

  // Original arena used for all transient allocations required for the
  // submission. All queue-related commands are allocated from this, **including
  // this retire command**.
  iree_arena_allocator_t arena;

  // A list of semaphores to signal upon retiring.
  iree_hal_semaphore_list_t signal_semaphores;

  // Resources retained until all have retired.
  // We could release them earlier but that would require tracking individual
  // resource-level completion.
  iree_host_size_t resource_count;
  iree_hal_resource_t* resources[];
} iree_hal_task_queue_retire_cmd_t;

// Retires a submission by signaling semaphores to their desired value and
// disposing of the temporary arena memory used for the submission.
static iree_status_t iree_hal_task_queue_retire_cmd(
    void* user_context, iree_task_t* task,
    iree_task_submission_t* pending_submission) {
  iree_hal_task_queue_retire_cmd_t* cmd =
      (iree_hal_task_queue_retire_cmd_t*)task;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release command buffers now that all are known to have retired.
  // We do this before signaling so that waiting threads can immediately reuse
  // resources that are released.
  for (iree_host_size_t i = 0; i < cmd->resource_count; ++i) {
    iree_hal_resource_release(cmd->resources[i]);
    cmd->resources[i] = NULL;
  }

  // Signal all semaphores to their new values.
  // Note that if any signal fails then the whole command will fail and all
  // semaphores will be signaled to the failure state.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < cmd->signal_semaphores.count; ++i) {
    status =
        iree_hal_semaphore_signal(cmd->signal_semaphores.semaphores[i],
                                  cmd->signal_semaphores.payload_values[i]);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Cleanup for iree_hal_task_queue_retire_cmd_t that ensures that the arena
// holding the submission is properly disposed and that semaphores are signaled
// (or signaled to failure if the command failed).
static void iree_hal_task_queue_retire_cmd_cleanup(
    iree_task_t* task, iree_status_code_t status_code) {
  iree_hal_task_queue_retire_cmd_t* cmd =
      (iree_hal_task_queue_retire_cmd_t*)task;

  // Release resources now that all are known to have retired.
  // In success cases we try to do this eagerly to allow for more potential
  // reuse but during full/partial failures they may still be live here.
  for (iree_host_size_t i = 0; i < cmd->resource_count; ++i) {
    iree_hal_resource_release(cmd->resources[i]);
    cmd->resources[i] = NULL;
  }

  // If the command failed then fail all semaphores to ensure future
  // submissions fail as well (including those on other queues).
  if (IREE_UNLIKELY(status_code != IREE_STATUS_OK)) {
    for (iree_host_size_t i = 0; i < cmd->signal_semaphores.count; ++i) {
      iree_hal_semaphore_fail(cmd->signal_semaphores.semaphores[i],
                              iree_status_from_code(status_code));
    }
  }

  // Release all semaphores.
  iree_hal_semaphore_list_release(&cmd->signal_semaphores);

  // Drop all memory used by the submission (**including cmd**).
  iree_arena_allocator_t arena = cmd->arena;
  cmd = NULL;
  iree_arena_deinitialize(&arena);
}

// Allocates and initializes a iree_hal_task_queue_retire_cmd_t task.
// The command will own an arena that can be used for other submission-related
// allocations.
static iree_status_t iree_hal_task_queue_retire_cmd_allocate(
    iree_task_scope_t* scope, iree_host_size_t resource_count,
    iree_hal_resource_t* const* resources,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_arena_block_pool_t* block_pool,
    iree_hal_task_queue_retire_cmd_t** out_cmd) {
  // Make an arena we'll use for allocating the command itself.
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  // Allocate the command from the arena.
  iree_hal_task_queue_retire_cmd_t* cmd = NULL;
  iree_host_size_t total_cmd_size =
      sizeof(*cmd) + resource_count * sizeof(*cmd->resources);
  iree_status_t status =
      iree_arena_allocate(&arena, total_cmd_size, (void**)&cmd);
  if (iree_status_is_ok(status)) {
    iree_task_call_initialize(
        scope, iree_task_make_call_closure(iree_hal_task_queue_retire_cmd, 0),
        &cmd->task);
    iree_task_set_cleanup_fn(&cmd->task.header,
                             iree_hal_task_queue_retire_cmd_cleanup);
  }

  // Clone the signal semaphores from the batch - we retain them and their
  // payloads.
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(signal_semaphores, &arena,
                                           &cmd->signal_semaphores);
  }

  if (iree_status_is_ok(status)) {
    // Transfer ownership of the arena to command.
    memcpy(&cmd->arena, &arena, sizeof(cmd->arena));

    // Retain command buffers.
    cmd->resource_count = resource_count;
    for (iree_host_size_t i = 0; i < resource_count; ++i) {
      cmd->resources[i] = resources[i];
      iree_hal_resource_retain(cmd->resources[i]);
    }

    *out_cmd = cmd;
  } else {
    iree_arena_deinitialize(&arena);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_t
//===----------------------------------------------------------------------===//

void iree_hal_task_queue_initialize(iree_string_view_t identifier,
                                    iree_task_executor_t* executor,
                                    iree_arena_block_pool_t* block_pool,
                                    iree_hal_task_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, identifier.data, identifier.size);

  memset(out_queue, 0, sizeof(*out_queue));

  out_queue->executor = executor;
  iree_task_executor_retain(out_queue->executor);
  out_queue->block_pool = block_pool;

  iree_task_scope_initialize(identifier, &out_queue->scope);

  iree_hal_task_queue_state_initialize(&out_queue->state);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_ignore(
      iree_task_scope_wait_idle(&queue->scope, IREE_TIME_INFINITE_FUTURE));

  iree_hal_task_queue_state_deinitialize(&queue->state);
  iree_task_scope_deinitialize(&queue->scope);
  iree_task_executor_release(queue->executor);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_task_queue_trim(iree_hal_task_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_task_executor_trim(queue->executor);
}

static iree_status_t iree_hal_task_queue_submit_batch(
    iree_hal_task_queue_t* queue, const iree_hal_submission_batch_t* batch) {
  // Task to retire the submission and free the transient memory allocated for
  // it (including the command itself). We allocate this first so it can get an
  // arena which we will use to allocate all other commands.
  iree_hal_task_queue_retire_cmd_t* retire_cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_retire_cmd_allocate(
      &queue->scope, batch->command_buffer_count,
      (iree_hal_resource_t* const*)batch->command_buffers,
      &batch->signal_semaphores, queue->block_pool, &retire_cmd));

  // NOTE: if we fail from here on we must drop the retire_cmd arena.
  iree_status_t status = iree_ok_status();

  // A fence we'll use to detect when the entire submission has completed.
  // TODO(benvanik): fold into the retire command.
  iree_task_fence_t* fence = NULL;
  status =
      iree_task_executor_acquire_fence(queue->executor, &queue->scope, &fence);
  iree_task_set_completion_task(&retire_cmd->task.header, &fence->header);

  // Task to fork and wait for unsatisfied semaphore dependencies.
  // This is optional and only required if we have previous submissions still
  // in-flight - if the queue is empty then we can directly schedule the waits.
  iree_hal_task_queue_wait_cmd_t* wait_cmd = NULL;
  if (iree_status_is_ok(status) && batch->wait_semaphores.count > 0) {
    status = iree_hal_task_queue_wait_cmd_allocate(
        &queue->scope, &batch->wait_semaphores, &retire_cmd->arena, &wait_cmd);
  }

  // Task to issue all the command buffers in the batch.
  // After this task completes the commands have been issued but have not yet
  // completed and the issued commands may complete in any order.
  iree_hal_task_queue_issue_cmd_t* issue_cmd = NULL;
  if (iree_status_is_ok(status) && batch->command_buffer_count > 0) {
    status = iree_hal_task_queue_issue_cmd_allocate(
        &queue->scope, queue, &retire_cmd->task.header,
        batch->command_buffer_count, batch->command_buffers, &retire_cmd->arena,
        &issue_cmd);
  }

  // Last chance for failure - from here on we are submitting.
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_arena_deinitialize(&retire_cmd->arena);
    return status;
  }

  iree_task_submission_t submission;
  iree_task_submission_initialize(&submission);

  // Sequencing: wait on semaphores or go directly into the executor queue.
  iree_task_t* head_task =
      issue_cmd ? &issue_cmd->task.header : &retire_cmd->task.header;
  if (wait_cmd != NULL) {
    // Ensure that we only issue command buffers after all waits have completed.
    iree_task_set_completion_task(&wait_cmd->task.header, head_task);
    iree_task_submission_enqueue(&submission, &wait_cmd->task.header);
  } else {
    // No waits needed; directly enqueue.
    iree_task_submission_enqueue(&submission, head_task);
  }

  // Submit the tasks immediately. The executor may queue them up until we
  // force the flush after all batches have been processed.
  iree_task_executor_submit(queue->executor, &submission);
  return iree_ok_status();
}

static iree_status_t iree_hal_task_queue_submit_batches(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  // For now we process each batch independently. To elide additional semaphore
  // work and prevent unneeded coordinator scheduling logic we could instead
  // build the whole DAG prior to submitting.
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    const iree_hal_submission_batch_t* batch = &batches[i];
    IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_batch(queue, batch));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_task_queue_submit(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_hal_task_queue_submit_batches(queue, batch_count, batches);
  if (iree_status_is_ok(status)) {
    iree_task_executor_flush(queue->executor);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_task_queue_wait_idle(iree_hal_task_queue_t* queue,
                                            iree_timeout_t timeout) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_status_t status = iree_task_scope_wait_idle(&queue->scope, deadline_ns);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
