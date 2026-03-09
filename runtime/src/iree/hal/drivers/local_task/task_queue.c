// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_queue.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/drivers/local_task/task_command_buffer.h"
#include "iree/hal/drivers/local_task/task_semaphore.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/resource_set.h"
#include "iree/task/submission.h"

// Each submission is turned into a DAG for execution:
//
//  +--------------------+
//  |  (previous issue)  |
//  +--------------------+
//    |
//    |   +--------------+
//    +-> | +--------------+  Unsatisfied waits register async semaphore
//    .   +-|  sema waits  |  timepoints that feed the issue task back to the
//    .     +--------------+  executor when satisfied. Same-queue semaphores are
//    .        | | | | |      typically already satisfied (fast path).
//    +--------+-+-+-+-+
//    |
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
//  |   retire command   |    submission is retired: semaphores are signaled,
//  +--------------------+    resources are released, and the scope is notified
//    |                       of completion (scope_end). This may happen before
//   ...                      earlier submissions if there were no dependencies.

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_wait_cmd_t
//===----------------------------------------------------------------------===//

// Task to register direct semaphore timepoints for one or more semaphores.
// Semaphores used to stitch together subsequent submissions on the same queue
// are typically already satisfied by the time this runs (due to implicit queue
// ordering), so the fast path in enqueue_timepoint returns without registering
// anything. Cross-queue semaphores register async timepoint callbacks that
// directly feed the issue task back to the executor when satisfied.
typedef struct iree_hal_task_queue_wait_cmd_t {
  // Call to iree_hal_task_queue_wait_cmd.
  iree_task_call_t task;

  // Executor to submit ready tasks to (from the queue).
  iree_task_executor_t* executor;

  // Arena used for the submission - additional tasks can be allocated from
  // this.
  iree_arena_allocator_t* arena;

  // A list of semaphores to wait on prior to issuing the rest of the
  // submission.
  iree_hal_semaphore_list_t wait_semaphores;
} iree_hal_task_queue_wait_cmd_t;

// Registers direct semaphore timepoints for each unsatisfied wait semaphore.
// Satisfied semaphores are skipped (fast path). Unsatisfied semaphores register
// async timepoint callbacks that decrement the issue task's dependency count
// and submit it to the executor when all dependencies are met.
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
        cmd->task.header.completion_task, cmd->executor, cmd->arena);
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
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_semaphore_list_release(cmd->wait_semaphores);

  IREE_TRACE_ZONE_END(z0);
}

// Allocates and initializes a iree_hal_task_queue_wait_cmd_t task.
static iree_status_t iree_hal_task_queue_wait_cmd_allocate(
    iree_task_scope_t* scope, iree_task_executor_t* executor,
    const iree_hal_semaphore_list_t* wait_semaphores,
    iree_arena_allocator_t* arena, iree_task_t** out_issue_task) {
  iree_hal_task_queue_wait_cmd_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*cmd), (void**)&cmd));
  iree_task_call_initialize(
      scope, iree_task_make_call_closure(iree_hal_task_queue_wait_cmd, 0),
      &cmd->task);
  iree_task_set_cleanup_fn(&cmd->task.header,
                           iree_hal_task_queue_wait_cmd_cleanup);
  cmd->executor = executor;
  cmd->arena = arena;

  // Clone the wait semaphores from the batch - we retain them and their
  // payloads.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_clone(
      wait_semaphores, iree_arena_allocator(arena), &cmd->wait_semaphores));

  *out_issue_task = &cmd->task.header;
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

  // A resource set containing all binding table buffers.
  // Owned by the retire command and any resources added will be retained until
  // the submission has completed (or failed).
  iree_hal_resource_set_t* resource_set;

  // Semaphores that were waited on prior to this issue command running.
  // Retained here so we can validate that none have failed between the time
  // the wait events fired and when we begin issuing commands. Without this
  // check, a wait-semaphore failure would silently allow the issue and retire
  // to proceed, signaling downstream semaphores as if nothing went wrong.
  iree_hal_semaphore_list_t wait_semaphores;

  // Semaphores to signal upon completion. Retained here so that when a
  // wait-semaphore failure is detected we can fail the signal semaphores
  // eagerly with the original failure status — before the retire_cmd cleanup
  // runs — ensuring downstream waiters see the failure as soon as possible.
  // iree_hal_semaphore_fail preserves the first failure, so the retire_cmd
  // cleanup's subsequent fail call is a safe no-op.
  iree_hal_semaphore_list_t signal_semaphores;

  // Command buffer to be issued.
  iree_hal_command_buffer_t* command_buffer;
  // Optional binding table for the command buffer.
  iree_hal_buffer_binding_table_t binding_table;
} iree_hal_task_queue_issue_cmd_t;

// Cleanup for iree_hal_task_queue_issue_cmd_t that releases the retained
// wait and signal semaphores used for failure validation and propagation.
static void iree_hal_task_queue_issue_cmd_cleanup(
    iree_task_t* task, iree_status_code_t status_code) {
  iree_hal_task_queue_issue_cmd_t* cmd = (iree_hal_task_queue_issue_cmd_t*)task;
  iree_hal_semaphore_list_release(cmd->wait_semaphores);
  iree_hal_semaphore_list_release(cmd->signal_semaphores);
}

static iree_status_t iree_hal_task_queue_issue_cmd_deferred(
    iree_hal_task_queue_issue_cmd_t* cmd,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_task_submission_t* pending_submission) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create a transient command buffer that we'll apply the deferred commands
  // into. It will live beyond this function as we'll issue the commands but
  // they may not run immediately.
  iree_hal_command_buffer_t* task_command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_task_command_buffer_create(
          cmd->queue->device_allocator, &cmd->queue->scope,
          iree_hal_command_buffer_mode(command_buffer) |
              IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
              IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED |
              // NOTE: we need to validate if a binding table is provided as the
              // bindings were not known when it was originally recorded.
              (iree_hal_buffer_binding_table_is_empty(binding_table)
                   ? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
                   : 0),
          iree_hal_command_buffer_allowed_categories(command_buffer),
          cmd->queue->affinity, /*binding_capacity=*/0,
          cmd->queue->large_block_pool,
          iree_hal_allocator_host_allocator(cmd->queue->device_allocator),
          &task_command_buffer));

  // Keep the command buffer live until the queue operation completes.
  iree_status_t status =
      iree_hal_resource_set_insert(cmd->resource_set, 1, &task_command_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(task_command_buffer);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Replay the commands from the deferred command buffer into the new task one.
  // This creates the task graph and captures the binding references but does
  // not yet issue the commands.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_deferred_command_buffer_apply(
              command_buffer, task_command_buffer, binding_table));

  // Issue the task command buffer as if it had been recorded directly to begin
  // with.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_task_command_buffer_issue(task_command_buffer,
                                             &cmd->queue->state,
                                             cmd->task.header.completion_task,
                                             cmd->arena, pending_submission));

  // Still retained in the resource set until retirement.
  iree_hal_command_buffer_release(task_command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Issues a set of command buffers without waiting for them to complete.
static iree_status_t iree_hal_task_queue_issue_cmd(
    void* user_context, iree_task_t* task,
    iree_task_submission_t* pending_submission) {
  iree_hal_task_queue_issue_cmd_t* cmd = (iree_hal_task_queue_issue_cmd_t*)task;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate that none of the wait semaphores have failed since the wait tasks
  // completed. Wait events fire for both successful signals and failures, so
  // arriving here does not guarantee the waits were satisfied — a wait
  // semaphore may have been failed concurrently. If any wait semaphore has
  // failed we fail the signal semaphores directly with the original failure
  // status (preserving the error message for downstream waiters) and abort.
  for (iree_host_size_t i = 0; i < cmd->wait_semaphores.count; ++i) {
    uint64_t value = 0;
    iree_status_t query_status =
        iree_hal_semaphore_query(cmd->wait_semaphores.semaphores[i], &value);
    if (!iree_status_is_ok(query_status)) {
      // Directly fail signal semaphores with the original wait failure status.
      // This preserves the error message chain so that downstream waiters can
      // see why their semaphore failed. The retire_cmd cleanup will also try
      // to fail them but iree_hal_semaphore_fail preserves the first failure.
      // Return a clone of the failure so iree_task_scope_fail gets the real
      // error (not a bare ABORTED code) for propagation to scope consumers.
      iree_status_t return_status = iree_status_clone(query_status);
      iree_hal_semaphore_list_fail(cmd->signal_semaphores, query_status);
      IREE_TRACE_ZONE_END(z0);
      return return_status;
    }
  }

  // NOTE: it's ok for there to be no command buffers - in that case the
  // submission was purely for synchronization.
  iree_status_t status = iree_ok_status();
  if (cmd->command_buffer != NULL) {
    if (iree_hal_task_command_buffer_isa(cmd->command_buffer)) {
      if (cmd->binding_table.count > 0) {
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "task command buffers do not support binding tables yet");
      } else {
        status = iree_hal_task_command_buffer_issue(
            cmd->command_buffer, &cmd->queue->state,
            cmd->task.header.completion_task, cmd->arena, pending_submission);
      }
    } else if (iree_hal_deferred_command_buffer_isa(cmd->command_buffer)) {
      status = iree_hal_task_queue_issue_cmd_deferred(
          cmd, cmd->command_buffer, cmd->binding_table, pending_submission);
    } else {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "unsupported command buffer type for task queue submission");
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Allocates and initializes a iree_hal_task_queue_issue_cmd_t task.
static iree_status_t iree_hal_task_queue_issue_cmd_allocate(
    void* user_data, iree_task_scope_t* scope, iree_hal_task_queue_t* queue,
    iree_task_t* retire_task, iree_arena_allocator_t* arena,
    iree_hal_resource_set_t* resource_set, iree_task_t** out_issue_task) {
  iree_hal_task_submission_batch_t* batch =
      (iree_hal_task_submission_batch_t*)user_data;

  iree_hal_task_queue_issue_cmd_t* cmd = NULL;
  iree_host_size_t total_cmd_size = 0;
  if (!iree_host_size_checked_mul_add(sizeof(*cmd), batch->binding_table.count,
                                      sizeof(*batch->binding_table.bindings),
                                      &total_cmd_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "binding table allocation size overflow");
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, total_cmd_size, (void**)&cmd));
  iree_task_call_initialize(
      scope, iree_task_make_call_closure(iree_hal_task_queue_issue_cmd, 0),
      &cmd->task);
  iree_task_set_cleanup_fn(&cmd->task.header,
                           iree_hal_task_queue_issue_cmd_cleanup);
  iree_task_set_completion_task(&cmd->task.header, retire_task);
  cmd->arena = arena;
  cmd->queue = queue;
  cmd->resource_set = resource_set;
  cmd->wait_semaphores = iree_hal_semaphore_list_empty();
  cmd->signal_semaphores = iree_hal_semaphore_list_empty();

  // Clone wait and signal semaphores for failure validation and propagation.
  // The wait tasks fire their events for both signals and failures, so by the
  // time issue_cmd executes we need to recheck the wait semaphore state.
  // Signal semaphores are cloned so we can fail them directly with the
  // original wait failure status (preserving error context).
  iree_status_t status = iree_hal_semaphore_list_clone(
      &batch->wait_semaphores, iree_arena_allocator(arena),
      &cmd->wait_semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(&batch->signal_semaphores,
                                           iree_arena_allocator(arena),
                                           &cmd->signal_semaphores);
  }

  cmd->command_buffer = batch->command_buffer;
  cmd->binding_table = iree_hal_buffer_binding_table_empty();

  // Binding tables are optional and we only need this extra work if there were
  // any non-empty binding tables provided during submission.
  if (iree_status_is_ok(status) && batch->binding_table.count > 0) {
    // Copy over binding tables and all of their contents.
    iree_hal_buffer_binding_t* binding_element_ptr =
        (iree_hal_buffer_binding_t*)((uint8_t*)cmd + sizeof(*cmd));
    const iree_host_size_t element_count = batch->binding_table.count;
    cmd->binding_table.count = element_count;
    cmd->binding_table.bindings = binding_element_ptr;
    memcpy((void*)cmd->binding_table.bindings, batch->binding_table.bindings,
           element_count * sizeof(*binding_element_ptr));
    binding_element_ptr += element_count;

    // Bulk insert all bindings into the resource set. This will keep the
    // referenced buffers live until the issue has completed.
    status = iree_hal_resource_set_insert_strided(
        cmd->resource_set, element_count, cmd->binding_table.bindings,
        offsetof(iree_hal_buffer_binding_t, buffer),
        sizeof(iree_hal_buffer_binding_t));
  }

  if (iree_status_is_ok(status)) {
    *out_issue_task = &cmd->task.header;
  } else {
    // Release retained refs from any successful clones. Lists were initialized
    // to empty so releasing a list where the clone never ran is a no-op.
    iree_hal_semaphore_list_release(cmd->wait_semaphores);
    iree_hal_semaphore_list_release(cmd->signal_semaphores);
  }
  return status;
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
  //
  // This resource set is allocated from the small block pool and is expected to
  // only have a small number of resources (command buffers, etc).
  iree_hal_resource_set_t* resource_set;
} iree_hal_task_queue_retire_cmd_t;

// Retires a submission by signaling semaphores to their desired value and
// disposing of the temporary arena memory used for the submission.
static iree_status_t iree_hal_task_queue_retire_cmd(
    void* user_context, iree_task_t* task,
    iree_task_submission_t* pending_submission) {
  iree_hal_task_queue_retire_cmd_t* cmd =
      (iree_hal_task_queue_retire_cmd_t*)task;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release retained resources (command buffers, etc).
  // We do this before signaling so that waiting threads can immediately reuse
  // resources that are released.
  iree_hal_resource_set_free(cmd->resource_set);
  cmd->resource_set = NULL;

  // Signal all semaphores to their new values.
  // Note that if any signal fails then the whole command will fail and all
  // semaphores will be signaled to the failure state.
  iree_status_t status = iree_hal_semaphore_list_signal(cmd->signal_semaphores);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Releases all resources owned by a retire_cmd and deallocates it by
// deinitializing its arena. After this call |cmd| is invalid.
// Does NOT fail semaphores — callers should call iree_hal_semaphore_list_fail
// first if the submission failed after being enqueued.
static void iree_hal_task_queue_retire_cmd_destroy(
    iree_hal_task_queue_retire_cmd_t* cmd) {
  if (cmd->resource_set) {
    iree_hal_resource_set_free(cmd->resource_set);
    cmd->resource_set = NULL;
  }
  iree_hal_semaphore_list_release(cmd->signal_semaphores);
  // Drop all memory used by the submission (**including cmd**).
  iree_arena_allocator_t arena = cmd->arena;
  cmd = NULL;
  iree_arena_deinitialize(&arena);
}

// Cleanup for iree_hal_task_queue_retire_cmd_t that ensures that the arena
// holding the submission is properly disposed and that semaphores are signaled
// (or signaled to failure if the command failed).
static void iree_hal_task_queue_retire_cmd_cleanup(
    iree_task_t* task, iree_status_code_t status_code) {
  iree_hal_task_queue_retire_cmd_t* cmd =
      (iree_hal_task_queue_retire_cmd_t*)task;
  IREE_TRACE_ZONE_BEGIN(z0);

  // If the command failed then fail all semaphores to ensure future
  // submissions fail as well (including those on other queues).
  // Consume the full failure status from the scope so that the original error
  // message propagates to semaphore waiters (e.g., "dispatch requires Xb of
  // local memory but only Yb is available" rather than a bare ABORTED code).
  if (IREE_UNLIKELY(status_code != IREE_STATUS_OK)) {
    iree_status_t failure_status = iree_task_scope_consume_status(task->scope);
    if (iree_status_is_ok(failure_status)) {
      // Scope status already consumed or was never set — fall back to code.
      failure_status = iree_status_from_code(status_code);
    }
    iree_hal_semaphore_list_fail(cmd->signal_semaphores, failure_status);
  }

  // Capture the scope before destroying the command — destroy deinitializes
  // the arena which frees this task's memory.
  iree_task_scope_t* scope = task->scope;
  iree_hal_task_queue_retire_cmd_destroy(cmd);

  // Notify the scope that this submission has completed. This must happen after
  // all submission memory is freed so that idle waiters can safely deallocate.
  // This task is arena-allocated (pool is NULL) so iree_task_cleanup has no
  // post-cleanup pool return step that could race with a scope-idle wakeup.
  if (scope) {
    iree_task_scope_end(scope);
  }

  IREE_TRACE_ZONE_END(z0);
}

// Allocates and initializes a iree_hal_task_queue_retire_cmd_t task.
// The command will own an arena that can be used for other submission-related
// allocations.
static iree_status_t iree_hal_task_queue_retire_cmd_allocate(
    iree_task_scope_t* scope,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_arena_block_pool_t* block_pool,
    iree_hal_task_queue_retire_cmd_t** out_cmd) {
  // Make an arena we'll use for allocating the command itself.
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  // Allocate the command from the arena.
  iree_hal_task_queue_retire_cmd_t* cmd = NULL;
  iree_status_t status =
      iree_arena_allocate(&arena, sizeof(*cmd), (void**)&cmd);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }

  iree_task_call_initialize(
      scope, iree_task_make_call_closure(iree_hal_task_queue_retire_cmd, 0),
      &cmd->task);
  iree_task_set_cleanup_fn(&cmd->task.header,
                           iree_hal_task_queue_retire_cmd_cleanup);
  cmd->signal_semaphores = iree_hal_semaphore_list_empty();
  cmd->resource_set = NULL;

  // Clone the signal semaphores from the batch - we retain them and their
  // payloads.
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(signal_semaphores,
                                           iree_arena_allocator(&arena),
                                           &cmd->signal_semaphores);
  }

  // Create a lightweight resource set to retain any resources used by the
  // command. Note that this is coming from the small block pool and is intended
  // only for a small number of resources.
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_allocate(block_pool, &cmd->resource_set);
  }

  if (iree_status_is_ok(status)) {
    // Transfer ownership of the arena to command.
    memcpy(&cmd->arena, &arena, sizeof(cmd->arena));
    *out_cmd = cmd;
  } else {
    if (cmd) {
      iree_hal_resource_set_free(cmd->resource_set);
      iree_hal_semaphore_list_release(cmd->signal_semaphores);
    }
    iree_arena_deinitialize(&arena);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_host_call_cmd_t
//===----------------------------------------------------------------------===//

// Task to call a user-defined function with host call semantics.
// This is an optimized version of the internal callback path that avoids the
// retire command latency and the resource set allocation as host calls need
// neither. Host calls also have a NON_BLOCKING mode that requires that we
// signal _before_ calling the user function and that doesn't fit with the
// normal wait-issue-execute-retire model.
typedef struct iree_hal_task_queue_host_call_cmd_t {
  // Call to iree_hal_task_queue_host_call_cmd.
  iree_task_call_t task;

  // Original arena used for all transient allocations required for the
  // submission.
  iree_arena_allocator_t arena;

  // Device the call was scheduled on. Unowned.
  iree_hal_device_t* device;
  // Queue affinity as originally requested.
  // We don't know where we'd actually run so we pass through without
  // modification.
  iree_hal_queue_affinity_t queue_affinity;
  // Target function to call.
  iree_hal_host_call_t call;
  // User arguments.
  uint64_t args[4];
  // Flags controlling call behavior.
  iree_hal_host_call_flags_t flags;

  // A list of semaphores to signal upon retiring.
  iree_hal_semaphore_list_t signal_semaphores;
} iree_hal_task_queue_host_call_cmd_t;

// Issues a host call submission and either calls the user function which
// transitively signals semaphores (blocking, call is responsible) or eagerly
// signals (NON_BLOCKING) and then calls the user function.
static iree_status_t iree_hal_task_queue_host_call_cmd(
    void* user_context, iree_task_t* task,
    iree_task_submission_t* pending_submission) {
  iree_hal_task_queue_host_call_cmd_t* cmd =
      (iree_hal_task_queue_host_call_cmd_t*)task;
  IREE_TRACE_ZONE_BEGIN(z0);

  // When non-blocking we want to eagerly signal all waiters prior to issuing
  // the call.
  // Note that if any signal fails then the whole command will fail and all
  // semaphores will be signaled to the failure state.
  const bool is_nonblocking =
      iree_any_bit_set(cmd->flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  iree_status_t status = iree_ok_status();
  if (is_nonblocking) {
    status = iree_hal_semaphore_list_signal(cmd->signal_semaphores);
  }

  // Issue the call.
  if (iree_status_is_ok(status)) {
    iree_hal_host_call_context_t context = {
        .device = cmd->device,
        .queue_affinity = cmd->queue_affinity,
        .signal_semaphore_list = is_nonblocking
                                     ? iree_hal_semaphore_list_empty()
                                     : cmd->signal_semaphores,
    };
    iree_status_t call_status =
        cmd->call.fn(cmd->call.user_data, cmd->args, &context);
    if (is_nonblocking || iree_status_is_deferred(call_status)) {
      // User callback will signal in the future (or they are fire-and-forget).
    } else if (iree_status_is_ok(call_status)) {
      // Signal callback completed synchronously.
      status = iree_hal_semaphore_list_signal(cmd->signal_semaphores);
    } else {
      // Callback failed; propagate the failure to all signal semaphores so
      // dependent submissions also fail.
      iree_hal_semaphore_list_fail(cmd->signal_semaphores, call_status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Releases all resources owned by a host_call_cmd and deallocates it by
// deinitializing its arena. After this call |cmd| is invalid.
static void iree_hal_task_queue_host_call_cmd_destroy(
    iree_hal_task_queue_host_call_cmd_t* cmd) {
  iree_hal_semaphore_list_release(cmd->signal_semaphores);
  // Drop all memory used by the submission (**including cmd**).
  iree_arena_allocator_t arena = cmd->arena;
  cmd = NULL;
  iree_arena_deinitialize(&arena);
}

// Cleanup for iree_hal_task_queue_host_call_cmd_t that ensures that the arena
// holding the submission is properly disposed and that semaphores are signaled
// (or signaled to failure if the command failed).
static void iree_hal_task_queue_host_call_cmd_cleanup(
    iree_task_t* task, iree_status_code_t status_code) {
  iree_hal_task_queue_host_call_cmd_t* cmd =
      (iree_hal_task_queue_host_call_cmd_t*)task;
  IREE_TRACE_ZONE_BEGIN(z0);

  // If the command failed then fail all semaphores to ensure future
  // submissions fail as well (including those on other queues).
  // Consume the full failure status from the scope — same rationale as
  // retire_cmd_cleanup.
  if (IREE_UNLIKELY(status_code != IREE_STATUS_OK)) {
    iree_status_t failure_status = iree_task_scope_consume_status(task->scope);
    if (iree_status_is_ok(failure_status)) {
      failure_status = iree_status_from_code(status_code);
    }
    iree_hal_semaphore_list_fail(cmd->signal_semaphores, failure_status);
  }

  // Capture the scope before destroying the command — destroy deinitializes
  // the arena which frees this task's memory.
  iree_task_scope_t* scope = task->scope;
  iree_hal_task_queue_host_call_cmd_destroy(cmd);

  // Notify the scope that this submission has completed. This must happen after
  // all submission memory is freed so that idle waiters can safely deallocate.
  // This task is arena-allocated (pool is NULL) so iree_task_cleanup has no
  // post-cleanup pool return step that could race with a scope-idle wakeup.
  if (scope) {
    iree_task_scope_end(scope);
  }

  IREE_TRACE_ZONE_END(z0);
}

// Allocates and initializes a iree_hal_task_queue_host_call_cmd_t task.
// The command will own an arena that can be used for other submission-related
// allocations.
static iree_status_t iree_hal_task_queue_host_call_cmd_allocate(
    iree_task_scope_t* scope,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_arena_block_pool_t* block_pool,
    iree_hal_task_queue_host_call_cmd_t** out_cmd) {
  // Make an arena we'll use for allocating the command itself.
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  // Allocate the command from the arena.
  iree_hal_task_queue_host_call_cmd_t* cmd = NULL;
  iree_status_t status =
      iree_arena_allocate(&arena, sizeof(*cmd), (void**)&cmd);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }

  iree_task_call_initialize(
      scope, iree_task_make_call_closure(iree_hal_task_queue_host_call_cmd, 0),
      &cmd->task);
  iree_task_set_cleanup_fn(&cmd->task.header,
                           iree_hal_task_queue_host_call_cmd_cleanup);
  cmd->signal_semaphores = iree_hal_semaphore_list_empty();

  // Clone the signal semaphores from the batch - we retain them and their
  // payloads.
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(signal_semaphores,
                                           iree_arena_allocator(&arena),
                                           &cmd->signal_semaphores);
  }

  if (iree_status_is_ok(status)) {
    // Transfer ownership of the arena to command.
    memcpy(&cmd->arena, &arena, sizeof(cmd->arena));
    *out_cmd = cmd;
  } else {
    if (cmd) {
      iree_hal_semaphore_list_release(cmd->signal_semaphores);
    }
    iree_arena_deinitialize(&arena);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_t
//===----------------------------------------------------------------------===//

void iree_hal_task_queue_initialize(
    iree_string_view_t identifier, iree_hal_queue_affinity_t affinity,
    iree_task_scope_flags_t scope_flags, iree_task_executor_t* executor,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_arena_block_pool_t* small_block_pool,
    iree_arena_block_pool_t* large_block_pool,
    iree_hal_allocator_t* device_allocator, iree_hal_task_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, identifier.data, identifier.size);

  memset(out_queue, 0, sizeof(*out_queue));

  out_queue->affinity = affinity;
  out_queue->executor = executor;
  iree_task_executor_retain(out_queue->executor);
  out_queue->proactor = proactor;
  out_queue->frontier_tracker = frontier_tracker;
  out_queue->small_block_pool = small_block_pool;
  out_queue->large_block_pool = large_block_pool;
  out_queue->device_allocator = device_allocator;
  iree_hal_allocator_retain(out_queue->device_allocator);

  iree_task_scope_initialize(identifier, scope_flags, &out_queue->scope);

  iree_hal_task_queue_state_initialize(&out_queue->state);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_ignore(
      iree_task_scope_wait_idle(&queue->scope, IREE_TIME_INFINITE_FUTURE));

  iree_hal_task_queue_state_deinitialize(&queue->state);
  iree_task_scope_deinitialize(&queue->scope);
  iree_hal_allocator_release(queue->device_allocator);
  iree_task_executor_release(queue->executor);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_task_queue_trim(iree_hal_task_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_task_executor_trim(queue->executor);
}

typedef iree_status_t(IREE_API_PTR* iree_hal_task_queue_issue_t)(
    void* user_data, iree_task_scope_t* scope, iree_hal_task_queue_t* queue,
    iree_task_t* retire_task, iree_arena_allocator_t* arena,
    iree_hal_resource_set_t* resource_set, iree_task_t** out_issue_task);

static iree_status_t iree_hal_task_queue_submit(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_host_size_t resource_count, iree_hal_resource_t* const* resources,
    iree_hal_task_queue_issue_t issue, void* user_data) {
  // Task to retire the submission and free the transient memory allocated for
  // it (including the command itself). We allocate this first so it can get an
  // arena which we will use to allocate all other commands.
  iree_hal_task_queue_retire_cmd_t* retire_cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_retire_cmd_allocate(
      &queue->scope, &signal_semaphores, queue->small_block_pool, &retire_cmd));

  // Mark the scope as having a pending submission. The matching scope_end is
  // called from iree_hal_task_queue_retire_cmd_cleanup when the retire command
  // completes (or is discarded on failure). We call this unconditionally here
  // so that the failure path below can always call scope_end.
  iree_task_scope_begin(&queue->scope);

  // If the caller provided any resources they wanted to retain we add them to
  // the resource set for them. This is just a helper to avoid needing to pass
  // too much state back to issue callbacks.
  //
  // NOTE: if we fail from here on we must drop the retire_cmd arena and end
  // the scope.
  iree_status_t status = iree_ok_status();
  if (resource_count > 0) {
    status = iree_hal_resource_set_insert(retire_cmd->resource_set,
                                          resource_count, resources);
  }

  // Task to fork and wait for unsatisfied semaphore dependencies.
  // This is optional and only required if we have previous submissions still
  // in-flight - if the queue is empty then we can directly schedule the waits.
  iree_task_t* wait_task = NULL;
  if (iree_status_is_ok(status) && wait_semaphores.count > 0) {
    status = iree_hal_task_queue_wait_cmd_allocate(
        &queue->scope, queue->executor, &wait_semaphores, &retire_cmd->arena,
        &wait_task);
  }

  // Task to issue all the command buffers in the batch.
  // After this task completes the commands have been issued but have not yet
  // completed and the issued commands may complete in any order.
  iree_task_t* issue_cmd = NULL;
  if (iree_status_is_ok(status) && issue != NULL) {
    status = issue(user_data, &queue->scope, queue, &retire_cmd->task.header,
                   &retire_cmd->arena, retire_cmd->resource_set, &issue_cmd);
  }

  // Last chance for failure - from here on we are submitting.
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_task_queue_retire_cmd_destroy(retire_cmd);
    iree_task_scope_end(&queue->scope);  // matches scope_begin above
    return status;
  }

  iree_task_submission_t submission;
  iree_task_submission_initialize(&submission);

  // Sequencing: wait on semaphores or go directly into the executor queue.
  iree_task_t* head_task = issue_cmd ? issue_cmd : &retire_cmd->task.header;
  if (wait_task != NULL) {
    // Ensure that we only issue command buffers after all waits have completed.
    iree_task_set_completion_task(wait_task, head_task);
    iree_task_submission_enqueue(&submission, wait_task);
  } else {
    // No waits needed; directly enqueue.
    iree_task_submission_enqueue(&submission, head_task);
  }

  // Submit the tasks immediately. The executor may queue them up until we
  // force the flush after all batches have been processed.
  iree_task_executor_submit(queue->executor, &submission);
  return iree_ok_status();
}

iree_status_t iree_hal_task_queue_submit_barrier(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_task_queue_submit(
      queue, wait_semaphores, signal_semaphores, 0, NULL, NULL, NULL);
  if (iree_status_is_ok(status)) {
    iree_task_executor_flush(queue->executor);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_queue_submit_batches(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_task_submission_batch_t* batches) {
  // For now we process each batch independently. To elide additional semaphore
  // work and prevent unneeded coordinator scheduling logic we could instead
  // build the whole DAG prior to submitting.
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    const iree_hal_task_submission_batch_t* batch = &batches[i];
    IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit(
        queue, batch->wait_semaphores, batch->signal_semaphores, 1,
        (iree_hal_resource_t* const*)&batch->command_buffer,
        iree_hal_task_queue_issue_cmd_allocate, (void*)batch));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_task_queue_submit_commands(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_task_submission_batch_t* batches) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_hal_task_queue_submit_batches(queue, batch_count, batches);
  if (iree_status_is_ok(status)) {
    iree_task_executor_flush(queue->executor);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_queue_callback_cmd_allocate(
    void* user_data, iree_task_scope_t* scope, iree_hal_task_queue_t* queue,
    iree_task_t* retire_task, iree_arena_allocator_t* arena,
    iree_hal_resource_set_t* resource_set, iree_task_t** out_issue_task) {
  iree_task_call_closure_t callback = *(iree_task_call_closure_t*)user_data;

  iree_task_call_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*cmd), (void**)&cmd));
  iree_task_call_initialize(scope, callback, cmd);
  iree_task_set_completion_task(&cmd->header, retire_task);

  *out_issue_task = &cmd->header;
  return iree_ok_status();
}

iree_status_t iree_hal_task_queue_submit_callback(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_host_size_t resource_count, iree_hal_resource_t* const* resources,
    iree_task_call_closure_t callback) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_task_queue_submit(
      queue, wait_semaphores, signal_semaphores, resource_count, resources,
      iree_hal_task_queue_callback_cmd_allocate, &callback);
  if (iree_status_is_ok(status)) {
    iree_task_executor_flush(queue->executor);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_task_queue_submit_host_call(
    iree_hal_task_queue_t* queue, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the task that tracks the host call state and dependencies.
  // NOTE: unlike most other submissions host calls do not use a retire command.
  iree_hal_task_queue_host_call_cmd_t* call_cmd = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_task_queue_host_call_cmd_allocate(
              &queue->scope, &signal_semaphores, queue->small_block_pool,
              &call_cmd));
  call_cmd->device = device;  // unowned
  call_cmd->queue_affinity = queue_affinity;
  call_cmd->call = call;
  memcpy(call_cmd->args, args, sizeof(call_cmd->args));
  call_cmd->flags = flags;

  // Mark the scope as having a pending submission. The matching scope_end is
  // called from iree_hal_task_queue_host_call_cmd_cleanup when the command
  // completes (or is discarded on failure).
  iree_task_scope_begin(&queue->scope);

  // Task to fork and wait for unsatisfied semaphore dependencies.
  // This is optional and only required if we have previous submissions still
  // in-flight - if the queue is empty then we can directly schedule the waits.
  iree_task_t* wait_task = NULL;
  iree_status_t status = iree_ok_status();
  if (wait_semaphores.count > 0) {
    status = iree_hal_task_queue_wait_cmd_allocate(
        &queue->scope, queue->executor, &wait_semaphores, &call_cmd->arena,
        &wait_task);
  }

  // Last chance for failure - from here on we are submitting.
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_task_queue_host_call_cmd_destroy(call_cmd);
    iree_task_scope_end(&queue->scope);  // matches scope_begin above
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_submission_t submission;
  iree_task_submission_initialize(&submission);

  // Sequencing: wait on semaphores or go directly into the executor queue.
  iree_task_t* head_task = &call_cmd->task.header;
  if (wait_task != NULL) {
    // Ensure that we only issue command buffers after all waits have completed.
    iree_task_set_completion_task(wait_task, head_task);
    iree_task_submission_enqueue(&submission, wait_task);
  } else {
    // No waits needed; directly enqueue.
    iree_task_submission_enqueue(&submission, head_task);
  }

  // Submit the tasks immediately. The executor may queue them up until we
  // force the flush after all batches have been processed.
  iree_task_executor_submit(queue->executor, &submission);
  iree_task_executor_flush(queue->executor);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
