// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_queue.h"

#include <stddef.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/semaphore.h"
#include "iree/hal/drivers/local_task/block_command_buffer.h"
#include "iree/hal/utils/resource_set.h"

// Each submission creates an arena-allocated operation that flows through:
//
//  submit()
//    |
//    |   +--------------+
//    +-> | +--------------+  Unsatisfied waits register async semaphore
//    .   +-|  sema waits  |  timepoints. Same-queue semaphores are typically
//    .     +--------------+  already satisfied (fast path skips entirely).
//    .        | | | | |
//    +--------+-+-+-+-+
//    |
//    v
//  +--------------------+
//  |   ready list       |  Operations with all waits satisfied. MPSC slist.
//  +--------------------+
//    |
//    v
//  +--------------------+    Queue process (budget-1) pops operations and
//  |   queue drain()    |    handles them by type: command buffers are issued
//  +--------------------+    as separate compute processes; barriers and host
//    |                       calls are handled inline.
//    v
//  +--------------------+    CB process completion (or inline for barriers/
//  |   completion       |    host calls): signal semaphores, advance frontier,
//  +--------------------+    end scope, free arena.

//===----------------------------------------------------------------------===//
// Operation lifecycle
//===----------------------------------------------------------------------===//

// Destroys an operation, failing signal semaphores if |failure_status| is
// non-OK. Releases all retained resources, ends the scope, and deinitializes
// the arena (which frees the operation itself and all transient allocations).
static void iree_hal_task_queue_op_destroy(iree_hal_task_queue_op_t* operation,
                                           iree_status_t failure_status) {
  // Fail signal semaphores on error (stores the error status in each), then
  // always release the semaphore references regardless of success/failure.
  if (!iree_status_is_ok(failure_status)) {
    iree_hal_semaphore_list_fail(operation->signal_semaphores, failure_status);
  }
  iree_hal_semaphore_list_release(operation->signal_semaphores);

  // Release retained resources (command buffers, binding table buffers).
  if (operation->resource_set) {
    iree_hal_resource_set_free(operation->resource_set);
    operation->resource_set = NULL;
  }

  // Capture scope before arena deinitialization frees the operation.
  iree_task_scope_t* scope = operation->scope;

  // Free all transient memory (including this operation).
  iree_arena_allocator_t arena = operation->arena;
  operation = NULL;
  iree_arena_deinitialize(&arena);

  // Notify the scope that this operation has completed. Must happen after
  // all operation memory is freed so that idle waiters can safely deallocate.
  if (scope) {
    iree_task_scope_end(scope);
  }
}

// Completes an operation successfully: signals semaphores, advances the
// frontier, then destroys the operation (freeing the arena).
static void iree_hal_task_queue_op_complete(
    iree_hal_task_queue_op_t* operation) {
  // Signal all semaphores to their new values.
  iree_status_t status = iree_hal_semaphore_list_signal(
      operation->signal_semaphores, /*frontier=*/NULL);

  // Advance the frontier tracker after signaling.
  if (operation->frontier_tracker && iree_status_is_ok(status)) {
    uint64_t epoch =
        (uint64_t)iree_atomic_fetch_add(operation->epoch_counter, 1,
                                        iree_memory_order_acq_rel) +
        1;
    iree_async_frontier_tracker_advance(operation->frontier_tracker,
                                        operation->axis, epoch);
  }

  // If signaling failed, fail the frontier and propagate the error.
  if (!iree_status_is_ok(status)) {
    if (operation->frontier_tracker) {
      iree_async_frontier_tracker_fail_axis(
          operation->frontier_tracker, operation->axis,
          iree_status_from_code(iree_status_code(status)));
    }
  }

  iree_hal_task_queue_op_destroy(operation, status);
}

// Allocates and initializes a queue operation from a fresh arena.
// The operation is allocated from the arena (so the arena owns the memory).
// Signal semaphores are cloned into the arena, and a resource set is created.
static iree_status_t iree_hal_task_queue_op_allocate(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_type_t type,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_hal_task_queue_op_t** out_operation) {
  *out_operation = NULL;

  // Create the arena that owns all transient memory for this operation.
  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->small_block_pool, &arena);

  // Allocate the operation from the arena.
  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status =
      iree_arena_allocate(&arena, sizeof(*operation), (void**)&operation);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }

  memset(operation, 0, sizeof(*operation));
  operation->type = type;
  memcpy(&operation->arena, &arena, sizeof(arena));
  operation->scope = &queue->scope;
  operation->frontier_tracker = queue->frontier_tracker;
  operation->axis = queue->axis;
  operation->epoch_counter = &queue->epoch;
  operation->queue = queue;

  // Clone signal semaphores into the arena.
  status = iree_hal_semaphore_list_clone(
      signal_semaphores, iree_arena_allocator(&operation->arena),
      &operation->signal_semaphores);

  // Create a resource set for retaining command buffers and binding table
  // buffers through completion.
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_allocate(queue->small_block_pool,
                                            &operation->resource_set);
  }

  if (iree_status_is_ok(status)) {
    *out_operation = operation;
  } else {
    iree_hal_semaphore_list_release(operation->signal_semaphores);
    if (operation->resource_set) {
      iree_hal_resource_set_free(operation->resource_set);
    }
    iree_arena_deinitialize(&operation->arena);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Semaphore wait integration
//===----------------------------------------------------------------------===//

// Arena-allocated wrapper for a semaphore timepoint that feeds completed waits
// into the queue's ready list. When all waits on an operation are satisfied,
// the operation is pushed to the ready list and the queue process is woken.
typedef struct iree_hal_task_queue_wait_entry_t {
  // Timepoint registered with the async semaphore.
  iree_async_semaphore_timepoint_t timepoint;
  // The operation waiting on this semaphore.
  iree_hal_task_queue_op_t* operation;
  // Retained reference to the HAL semaphore.
  iree_hal_semaphore_t* semaphore;
} iree_hal_task_queue_wait_entry_t;

// Callback fired when a semaphore timepoint is resolved (value reached or
// semaphore failed). Decrements the operation's wait_count and, if this was
// the last outstanding wait, either pushes the operation to the ready list
// (success) or destroys it (failure).
static void iree_hal_task_queue_wait_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_task_queue_wait_entry_t* entry =
      (iree_hal_task_queue_wait_entry_t*)user_data;
  iree_hal_task_queue_op_t* operation = entry->operation;
  iree_hal_semaphore_t* semaphore = entry->semaphore;

  // Record the first failure via CAS.
  if (!iree_status_is_ok(status)) {
    intptr_t expected = 0;
    if (!iree_atomic_compare_exchange_strong(
            &operation->error_status, &expected, (intptr_t)status,
            iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
      // Another wait already recorded an error; drop ours.
      iree_status_ignore(status);
    }
  }

  // Decrement wait count regardless of success/failure.
  int32_t previous_count = iree_atomic_fetch_sub(&operation->wait_count, 1,
                                                 iree_memory_order_acq_rel);
  if (previous_count == 1) {
    // Last wait resolved. Check for errors.
    iree_status_t error = (iree_status_t)iree_atomic_exchange(
        &operation->error_status, 0, iree_memory_order_acquire);
    if (!iree_status_is_ok(error)) {
      // At least one wait failed. Fail signals and destroy.
      iree_hal_task_queue_op_destroy(operation, error);
    } else {
      // All waits satisfied. Push to ready list and wake queue.
      iree_hal_task_queue_op_slist_push(&operation->queue->ready_list,
                                        operation);
      iree_task_executor_schedule_process(operation->queue->executor,
                                          &operation->queue->process);
    }
  }

  // Release retained semaphore reference. Must happen after all operation
  // access — the semaphore does not depend on the operation, but we should
  // not access the entry (arena memory) after the operation might be freed.
  iree_hal_semaphore_release(semaphore);
}

// Queries wait semaphores front-to-back, removing already-satisfied entries.
// If all are satisfied, sets count to 0 so the caller skips wait registration.
// If entry i is the first unsatisfied, slices the list to [i, count) so only
// the unsatisfied remainder gets timepoint registration.
//
// Returns a failure status if any queried semaphore has been failed.
static iree_status_t iree_hal_task_queue_try_satisfy_waits(
    iree_hal_semaphore_list_t* wait_semaphores) {
  for (iree_host_size_t i = 0; i < wait_semaphores->count; ++i) {
    uint64_t current_value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(
        wait_semaphores->semaphores[i], &current_value));
    if (current_value < wait_semaphores->payload_values[i]) {
      // Slice past the satisfied entries we already verified.
      wait_semaphores->semaphores += i;
      wait_semaphores->payload_values += i;
      wait_semaphores->count -= i;
      return iree_ok_status();
    }
  }
  wait_semaphores->count = 0;
  return iree_ok_status();
}

// Registers semaphore timepoints for each unsatisfied wait. The operation's
// wait_count is set to the number of unsatisfied waits. Each timepoint callback
// decrements wait_count; the last one to fire pushes the operation to the
// ready list.
//
// If all waits are already satisfied (count == 0), the operation is pushed
// directly to the ready list.
static iree_status_t iree_hal_task_queue_enqueue_waits(
    iree_hal_task_queue_op_t* operation,
    iree_hal_semaphore_list_t wait_semaphores) {
  if (wait_semaphores.count == 0) {
    // All waits already satisfied — push directly.
    iree_hal_task_queue_op_slist_push(&operation->queue->ready_list, operation);
    iree_task_executor_schedule_process(operation->queue->executor,
                                        &operation->queue->process);
    return iree_ok_status();
  }

  // Set the wait count before registering any timepoints. A timepoint callback
  // may fire synchronously during acquire_timepoint if the semaphore value
  // was reached between try_satisfy_waits and here.
  iree_atomic_store(&operation->wait_count, (int32_t)wait_semaphores.count,
                    iree_memory_order_release);

  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_hal_semaphore_t* semaphore = wait_semaphores.semaphores[i];
    uint64_t minimum_value = wait_semaphores.payload_values[i];

    // Allocate the wait entry from the operation's arena.
    iree_hal_task_queue_wait_entry_t* entry = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&operation->arena, sizeof(*entry), (void**)&entry));
    entry->operation = operation;
    entry->semaphore = semaphore;
    iree_hal_semaphore_retain(semaphore);

    // Register the timepoint. The callback may fire synchronously.
    entry->timepoint.callback = iree_hal_task_queue_wait_resolved;
    entry->timepoint.user_data = entry;
    iree_status_t status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)semaphore, minimum_value, &entry->timepoint);
    if (!iree_status_is_ok(status)) {
      // Registration failed. Release the semaphore ref we just took and
      // propagate the error. The caller will destroy the operation.
      iree_hal_semaphore_release(semaphore);
      return status;
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// CB execution (command buffer process completion)
//===----------------------------------------------------------------------===//

// Wraps the block issue context with a back-pointer to the queue operation.
// Allocated with cache-line alignment (required by the embedded process in
// the issue context). The issue_context is at offset 0 so the completion
// callback can cast from issue_context* to cmd_context*.
typedef struct iree_hal_task_queue_cmd_context_t {
  iree_hal_block_issue_context_t issue_context;
  iree_hal_task_queue_op_t* operation;
  // Allocator used to free this cmd_context (and the resource_set). Snapshotted
  // at allocation time so the deferred release callback doesn't chase through
  // queue->device_allocator (which may be destroyed before release fires).
  iree_allocator_t host_allocator;
  // Retained resources (command buffer, buffer bindings) that workers access
  // during drain. Moved here from the operation's resource_set in the eager
  // completion callback so they survive arena deinitialization and are freed
  // in the deferred release callback (after all workers exit drain).
  iree_hal_resource_set_t* resource_set;
  // Scope for this operation. Snapshotted from the operation in the eager
  // completion callback (before the arena is freed). scope_end is deferred
  // to the release callback so that scope_wait_idle (used during device
  // teardown) blocks until all workers have exited drain and all release
  // callbacks have completed — ensuring device-owned resources (block pools,
  // allocators) are still alive for the entire release path.
  iree_task_scope_t* scope;
} iree_hal_task_queue_cmd_context_t;

// CB process eager completion callback. Fires immediately when the first
// worker observes the block processor has completed — signals semaphores,
// advances the frontier, and frees the operation's arena. Other workers may
// still be inside drain() reading the issue context and command buffer
// recording, so the cmd_context and resource_set are NOT freed here
// (deferred to release).
static void iree_hal_task_queue_cmd_completion(iree_task_process_t* process,
                                               iree_status_t status) {
  iree_hal_block_issue_context_t* issue_context =
      (iree_hal_block_issue_context_t*)process->user_data;
  iree_hal_task_queue_cmd_context_t* cmd_context =
      (iree_hal_task_queue_cmd_context_t*)issue_context;
  iree_hal_task_queue_op_t* operation = cmd_context->operation;

  // Move the resource_set to cmd_context. Workers read command buffer
  // recordings and buffer bindings during drain — the resource_set must
  // stay alive until all workers exit. op_destroy would free it, so we
  // clear the operation's pointer and defer the release.
  cmd_context->resource_set = operation->resource_set;
  operation->resource_set = NULL;

  // Defer scope_end to the release callback. This ensures scope_wait_idle
  // (called during device teardown) blocks until all workers have exited
  // drain and all release callbacks have completed — device-owned resources
  // (block pools, allocators) remain alive for the entire release path.
  cmd_context->scope = operation->scope;
  operation->scope = NULL;

  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    // Propagate failure to frontier.
    if (operation->frontier_tracker) {
      iree_async_frontier_tracker_fail_axis(
          operation->frontier_tracker, operation->axis,
          iree_status_from_code(iree_status_code(status)));
    }
    iree_hal_task_queue_op_destroy(operation, status);
  }
}

// CB process deferred release callback. Fires when the last active drainer
// exits — all workers have finished calling drain() and it is safe to free
// resources they accessed: the resource_set (command buffer recordings, buffer
// bindings) and the cmd_context (issue context, embedded process, per-worker
// state).
static void iree_hal_task_queue_cmd_release(iree_task_process_t* process) {
  iree_hal_block_issue_context_t* issue_context =
      (iree_hal_block_issue_context_t*)process->user_data;
  iree_hal_task_queue_cmd_context_t* cmd_context =
      (iree_hal_task_queue_cmd_context_t*)issue_context;

  // Release retained resources (command buffer, buffer bindings). Workers
  // were reading from these during drain — now safe to release.
  if (cmd_context->resource_set) {
    iree_hal_resource_set_free(cmd_context->resource_set);
    cmd_context->resource_set = NULL;
  }

  // End the scope now that all workers have exited drain and all device-owned
  // resources have been released. This unblocks scope_wait_idle in the queue
  // deinitialize path, which in turn allows device teardown to proceed safely.
  iree_task_scope_t* scope = cmd_context->scope;

  // Use the snapshotted allocator — the queue/device may already be destroyed.
  iree_allocator_t host_allocator = cmd_context->host_allocator;
  iree_allocator_free_aligned(host_allocator, cmd_context);

  // scope_end after freeing cmd_context: idle waiters may deallocate the
  // scope's owner (the queue/device), so no cmd_context access after this.
  if (scope) {
    iree_task_scope_end(scope);
  }
}

//===----------------------------------------------------------------------===//
// Queue process drain function
//===----------------------------------------------------------------------===//

// Handles a COMMANDS operation: issues the block command buffer as a compute
// process and schedules it on the executor.
static iree_status_t iree_hal_task_queue_drain_commands(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(queue->device_allocator);

  // Verify the command buffer is a block command buffer.
  if (!iree_hal_block_command_buffer_isa(operation->commands.command_buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue only accepts block command buffers; got unrecognized type");
  }

  // Allocate the cmd_context with cache-line alignment (required by the
  // embedded process in the issue context).
  iree_hal_task_queue_cmd_context_t* cmd_context = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_aligned(host_allocator, sizeof(*cmd_context),
                                    iree_hardware_destructive_interference_size,
                                    /*offset=*/0, (void**)&cmd_context));
  memset(cmd_context, 0, sizeof(*cmd_context));
  cmd_context->operation = operation;
  cmd_context->host_allocator = host_allocator;

  // Issue the command buffer. This allocates the processor context internally
  // and initializes the embedded process with the drain adapter.
  const iree_hal_buffer_binding_table_t* binding_table =
      operation->commands.binding_table.count > 0
          ? &operation->commands.binding_table
          : NULL;
  iree_status_t status = iree_hal_block_command_buffer_issue(
      operation->commands.command_buffer, binding_table,
      (uint32_t)iree_task_executor_worker_count(queue->executor),
      host_allocator, &cmd_context->issue_context);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free_aligned(host_allocator, cmd_context);
    return status;
  }

  // Set the completion and release callbacks. The internal completion fn
  // consumes the processor result and chains to our completion callback
  // (eager: signals, frontier, arena). The internal release fn frees the
  // processor context and chains to our release callback (deferred: frees
  // cmd_context after all workers exit drain).
  cmd_context->issue_context.user_completion_fn =
      iree_hal_task_queue_cmd_completion;
  cmd_context->issue_context.user_release_fn = iree_hal_task_queue_cmd_release;

  // Schedule the CB process on the executor.
  iree_task_executor_schedule_process(queue->executor,
                                      &cmd_context->issue_context.process);

  return iree_ok_status();
}

// Handles a HOST_CALL operation: executes the user function inline and
// completes the operation.
static iree_status_t iree_hal_task_queue_drain_host_call(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  const bool is_nonblocking = iree_any_bit_set(
      operation->host_call.flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);

  // Non-blocking calls signal before executing.
  iree_status_t status = iree_ok_status();
  if (is_nonblocking) {
    status = iree_hal_semaphore_list_signal(operation->signal_semaphores,
                                            /*frontier=*/NULL);
  }

  // Execute the user function.
  if (iree_status_is_ok(status)) {
    iree_hal_host_call_context_t context = {
        .device = operation->host_call.device,
        .queue_affinity = operation->host_call.queue_affinity,
        .signal_semaphore_list = is_nonblocking
                                     ? iree_hal_semaphore_list_empty()
                                     : operation->signal_semaphores,
    };
    iree_status_t call_status =
        operation->host_call.call.fn(operation->host_call.call.user_data,
                                     operation->host_call.args, &context);
    if (is_nonblocking || iree_status_is_deferred(call_status)) {
      // User callback will signal in the future (or fire-and-forget).
    } else if (iree_status_is_ok(call_status)) {
      // Signal callback completed synchronously.
      if (!is_nonblocking) {
        status = iree_hal_semaphore_list_signal(operation->signal_semaphores,
                                                /*frontier=*/NULL);
      }
    } else {
      // Callback failed; propagate failure to signal semaphores, release the
      // references, and clear the list to prevent op_complete from
      // double-signaling or double-releasing.
      iree_hal_semaphore_list_fail(operation->signal_semaphores, call_status);
      iree_hal_semaphore_list_release(operation->signal_semaphores);
      operation->signal_semaphores = iree_hal_semaphore_list_empty();
    }
  }

  // Advance frontier and clean up.
  if (iree_status_is_ok(status)) {
    if (operation->frontier_tracker) {
      uint64_t epoch =
          (uint64_t)iree_atomic_fetch_add(operation->epoch_counter, 1,
                                          iree_memory_order_acq_rel) +
          1;
      iree_async_frontier_tracker_advance(operation->frontier_tracker,
                                          operation->axis, epoch);
    }
  } else {
    if (operation->frontier_tracker) {
      iree_async_frontier_tracker_fail_axis(
          operation->frontier_tracker, operation->axis,
          iree_status_from_code(iree_status_code(status)));
    }
  }

  iree_hal_task_queue_op_destroy(operation, status);
  return iree_ok_status();
}

// Queue process completion callback. Fires when the queue process reaches a
// terminal state (shutting_down set). Calls scope_end to unblock
// scope_wait_idle in the deinitialize path.
static void iree_hal_task_queue_process_completion(iree_task_process_t* process,
                                                   iree_status_t status) {
  iree_hal_task_queue_t* queue = (iree_hal_task_queue_t*)process->user_data;
  iree_status_ignore(status);
  iree_task_scope_end(&queue->scope);
}

// Queue process drain function. Pops one operation from the ready list and
// handles it by type. Returns did_work=false when the ready list is empty
// (the executor's sleeping protocol will park the process). Returns
// completed=true when shutting_down is set (during deinitialize).
static iree_status_t iree_hal_task_queue_process_drain(
    iree_task_process_t* process, uint32_t worker_index,
    iree_task_process_drain_result_t* out_result) {
  iree_hal_task_queue_t* queue = (iree_hal_task_queue_t*)process->user_data;

  // Check for shutdown. When set, complete the process so the completion
  // callback fires scope_end, unblocking scope_wait_idle in deinitialize.
  if (iree_atomic_load(&queue->shutting_down, iree_memory_order_acquire)) {
    out_result->did_work = false;
    out_result->completed = true;
    return iree_ok_status();
  }

  iree_hal_task_queue_op_t* operation =
      iree_hal_task_queue_op_slist_pop(&queue->ready_list);
  if (!operation) {
    out_result->did_work = false;
    out_result->completed = false;
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  switch (operation->type) {
    case IREE_HAL_TASK_QUEUE_OP_COMMANDS:
      status = iree_hal_task_queue_drain_commands(queue, operation);
      if (!iree_status_is_ok(status)) {
        // Issue failed — destroy the operation with the error.
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();  // Don't fail the queue process itself.
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_BARRIER:
      // Barriers just signal and clean up — no CB to execute.
      iree_hal_task_queue_op_complete(operation);
      break;
    case IREE_HAL_TASK_QUEUE_OP_HOST_CALL:
      status = iree_hal_task_queue_drain_host_call(queue, operation);
      break;
  }

  out_result->did_work = true;
  out_result->completed = false;
  return status;
}

//===----------------------------------------------------------------------===//
// Submit paths
//===----------------------------------------------------------------------===//

// Common submit path: allocates an operation, registers semaphore waits, and
// pushes to the ready list when all waits are satisfied.
static iree_status_t iree_hal_task_queue_submit_op(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_type_t type,
    iree_hal_semaphore_list_t wait_semaphores,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_host_size_t resource_count, iree_hal_resource_t* const* resources) {
  // Allocate the operation (arena + signal semaphore clone + resource set).
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_op_allocate(
      queue, type, signal_semaphores, &operation));

  // Mark the scope as having a pending operation. The matching scope_end is
  // called when the operation completes (in op_destroy/op_complete).
  iree_task_scope_begin(&queue->scope);

  // Retain any resources provided by the caller.
  iree_status_t status = iree_ok_status();
  if (resource_count > 0) {
    status = iree_hal_resource_set_insert(operation->resource_set,
                                          resource_count, resources);
  }

  // Fast path: check if all wait semaphores are already satisfied.
  if (iree_status_is_ok(status) && wait_semaphores.count > 0) {
    status = iree_hal_task_queue_try_satisfy_waits(&wait_semaphores);
  }

  // Register timepoints for unsatisfied waits (or push directly if all
  // waits are satisfied).
  if (iree_status_is_ok(status)) {
    status = iree_hal_task_queue_enqueue_waits(operation, wait_semaphores);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
    return status;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_t
//===----------------------------------------------------------------------===//

void iree_hal_task_queue_initialize(
    iree_string_view_t identifier, iree_hal_queue_affinity_t affinity,
    iree_task_scope_flags_t scope_flags, iree_task_executor_t* executor,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
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
  out_queue->axis = axis;
  iree_atomic_store(&out_queue->epoch, 0, iree_memory_order_relaxed);
  out_queue->small_block_pool = small_block_pool;
  out_queue->large_block_pool = large_block_pool;
  out_queue->device_allocator = device_allocator;
  iree_hal_allocator_retain(out_queue->device_allocator);

  iree_task_scope_initialize(identifier, scope_flags, &out_queue->scope);

  // Initialize the ready list.
  iree_hal_task_queue_op_slist_initialize(&out_queue->ready_list);

  // Initialize the queue process. Budget-1 (sequential), starts suspended
  // with suspend_count=0 (immediately runnable but not scheduled until the
  // first submission arrives).
  iree_task_process_initialize(iree_hal_task_queue_process_drain,
                               /*suspend_count=*/0, /*worker_budget=*/1,
                               &out_queue->process);
  out_queue->process.user_data = out_queue;
  out_queue->process.completion_fn = iree_hal_task_queue_process_completion;

  // The queue process participates in the scope so that scope_wait_idle
  // blocks until the process has fully completed (no worker touching
  // queue/device memory). The matching scope_end fires in the completion
  // callback when the process terminates during deinitialize.
  iree_task_scope_begin(&out_queue->scope);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Signal the queue process to complete. The next drain call sees this flag
  // and returns completed=true, triggering normal process completion.
  iree_atomic_store(&queue->shutting_down, 1, iree_memory_order_release);

  // Schedule the queue process so a worker picks it up and sees the shutdown
  // flag. If the process is already being drained, schedule_process sets
  // needs_drain and the worker will see shutting_down on the next iteration.
  // If the process is IDLE, schedule_process pushes it to the immediate list.
  iree_task_executor_schedule_process(queue->executor, &queue->process);

  // Wait for all outstanding operations AND the queue process to complete.
  // The queue process's scope_begin (at init) pairs with scope_end (in its
  // completion callback). Each submitted operation also has a scope_begin/end
  // pair. scope_wait_idle returns only when all of these have resolved —
  // meaning every worker has finished touching queue/device resources.
  iree_status_ignore(
      iree_task_scope_wait_idle(&queue->scope, IREE_TIME_INFINITE_FUTURE));

  // Drain and destroy any remaining operations in the ready list.
  // These are operations that were queued but never drained (e.g., submitted
  // after the queue process went idle but before shutdown was signaled).
  iree_hal_task_queue_op_t* remaining = NULL;
  while ((remaining = iree_hal_task_queue_op_slist_pop(&queue->ready_list)) !=
         NULL) {
    iree_hal_task_queue_op_destroy(
        remaining,
        iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down"));
  }
  iree_hal_task_queue_op_slist_deinitialize(&queue->ready_list);

  iree_task_scope_deinitialize(&queue->scope);
  iree_hal_allocator_release(queue->device_allocator);
  iree_task_executor_release(queue->executor);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_task_queue_trim(iree_hal_task_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_task_executor_trim(queue->executor);
}

iree_status_t iree_hal_task_queue_submit_barrier(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_task_queue_submit_op(
      queue, IREE_HAL_TASK_QUEUE_OP_BARRIER, wait_semaphores,
      &signal_semaphores, 0, NULL);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_task_queue_submit_commands(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_task_submission_batch_t* batches) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    const iree_hal_task_submission_batch_t* batch = &batches[i];

    // Handle NULL command buffer as a barrier.
    if (batch->command_buffer == NULL) {
      status = iree_hal_task_queue_submit_op(
          queue, IREE_HAL_TASK_QUEUE_OP_BARRIER, batch->wait_semaphores,
          &batch->signal_semaphores, 0, NULL);
      if (!iree_status_is_ok(status)) break;
      continue;
    }

    // Allocate the operation.
    iree_hal_task_queue_op_t* operation = NULL;
    status =
        iree_hal_task_queue_op_allocate(queue, IREE_HAL_TASK_QUEUE_OP_COMMANDS,
                                        &batch->signal_semaphores, &operation);
    if (!iree_status_is_ok(status)) break;

    iree_task_scope_begin(&queue->scope);

    // Retain the command buffer.
    status = iree_hal_resource_set_insert(
        operation->resource_set, 1,
        (iree_hal_resource_t* const*)&batch->command_buffer);

    // Store the command buffer and binding table in the operation.
    operation->commands.command_buffer = batch->command_buffer;
    operation->commands.binding_table = iree_hal_buffer_binding_table_empty();

    // Copy binding table entries into the arena if present.
    if (iree_status_is_ok(status) && batch->binding_table.count > 0) {
      iree_hal_buffer_binding_t* bindings = NULL;
      status = iree_arena_allocate(
          &operation->arena, batch->binding_table.count * sizeof(*bindings),
          (void**)&bindings);
      if (iree_status_is_ok(status)) {
        memcpy(bindings, batch->binding_table.bindings,
               batch->binding_table.count * sizeof(*bindings));
        operation->commands.binding_table.count = batch->binding_table.count;
        operation->commands.binding_table.bindings = bindings;

        // Retain all binding table buffer references.
        status = iree_hal_resource_set_insert_strided(
            operation->resource_set, batch->binding_table.count, bindings,
            offsetof(iree_hal_buffer_binding_t, buffer),
            sizeof(iree_hal_buffer_binding_t));
      }
    }

    // Fast path: check if all wait semaphores are already satisfied.
    iree_hal_semaphore_list_t wait_semaphores = batch->wait_semaphores;
    if (iree_status_is_ok(status) && wait_semaphores.count > 0) {
      status = iree_hal_task_queue_try_satisfy_waits(&wait_semaphores);
    }

    // Register timepoints for unsatisfied waits.
    if (iree_status_is_ok(status)) {
      status = iree_hal_task_queue_enqueue_waits(operation, wait_semaphores);
    }

    if (!iree_status_is_ok(status)) {
      iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
      break;
    }
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

  // Allocate the operation.
  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_HOST_CALL, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  // Store host call parameters.
  operation->host_call.device = device;
  operation->host_call.queue_affinity = queue_affinity;
  operation->host_call.call = call;
  memcpy(operation->host_call.args, args, sizeof(operation->host_call.args));
  operation->host_call.flags = flags;

  // Fast path: check if all wait semaphores are already satisfied.
  if (wait_semaphores.count > 0) {
    status = iree_hal_task_queue_try_satisfy_waits(&wait_semaphores);
  }

  // Register timepoints for unsatisfied waits.
  if (iree_status_is_ok(status)) {
    status = iree_hal_task_queue_enqueue_waits(operation, wait_semaphores);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
