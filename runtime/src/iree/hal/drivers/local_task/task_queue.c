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
#include "iree/hal/drivers/local_task/transient_buffer.h"
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
// Per-operation drain functions
//===----------------------------------------------------------------------===//

// Handles a COMMANDS operation: fills a recording item from the free pool
// and pushes it to the compute process's pending list.
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

  // Acquire a recording item from the free pool.
  iree_hal_task_queue_compute_item_t* item =
      iree_hal_task_queue_compute_item_slist_pop(&queue->compute_free_pool);
  if (!item) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "compute recording pool exhausted (pool size %d); too many "
        "concurrent command buffers in flight",
        IREE_HAL_TASK_QUEUE_COMPUTE_POOL_SIZE);
  }

  // Get the recording from the command buffer.
  const iree_hal_cmd_block_recording_t* recording =
      iree_hal_block_command_buffer_recording(
          operation->commands.command_buffer);

  // Allocate the block processor execution context. One-shot command buffers
  // use direct fixups (binding_table=NULL). Indirect command buffers are not
  // supported by the block ISA.
  uint32_t worker_count =
      (uint32_t)iree_task_executor_worker_count(queue->executor);
  if (worker_count == 0) worker_count = 1;
  iree_hal_cmd_block_processor_context_t* processor_context = NULL;
  iree_status_t status = iree_hal_cmd_block_processor_context_allocate(
      recording, /*binding_table=*/NULL, /*binding_table_length=*/0,
      worker_count, host_allocator, &processor_context);
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_compute_item_slist_push(&queue->compute_free_pool,
                                                item);
    return status;
  }

  // Fill the recording item.
  item->processor_context = processor_context;
  item->worker_count = worker_count;
  item->operation = operation;
  item->resource_set = NULL;
  item->scope = NULL;
  item->host_allocator = host_allocator;
  memset(item->worker_states, 0, sizeof(item->worker_states[0]) * worker_count);
  iree_atomic_store(&item->active_drainers, 0, iree_memory_order_relaxed);
  iree_atomic_store(&item->release_pending, 0, iree_memory_order_relaxed);

  // Push to the compute pending list.
  iree_hal_task_queue_compute_item_slist_push(&queue->compute_pending, item);

  // Schedule the compute process. For the first recording, this places the
  // process in a compute slot (CAS IDLE→DRAINING). For subsequent recordings,
  // the CAS fails (already DRAINING) but wake_workers still runs, ensuring
  // workers pick up the new recording from the pending list.
  iree_task_executor_schedule_process(queue->executor, &queue->compute_process);

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

// Handles an ALLOCA operation: allocates real backing memory and commits it
// into the transient buffer.
static iree_status_t iree_hal_task_queue_drain_alloca(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  iree_hal_buffer_t* backing = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      operation->alloca.device_allocator, operation->alloca.params,
      operation->alloca.allocation_size, &backing);
  if (iree_status_is_ok(status)) {
    iree_hal_task_transient_buffer_commit(operation->alloca.transient_buffer,
                                          backing);
    // The transient buffer now retains the backing. Release our reference.
    iree_hal_buffer_release(backing);
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_destroy(operation, status);
  }
  return iree_ok_status();
}

// Handles a DEALLOCA operation: decommits the transient buffer, releasing the
// backing memory.
static void iree_hal_task_queue_drain_dealloca(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  iree_hal_task_transient_buffer_decommit(operation->dealloca.transient_buffer);
  iree_hal_task_queue_op_complete(operation);
}

//===----------------------------------------------------------------------===//
// Compute process (data plane)
//===----------------------------------------------------------------------===//

// Sentinel value for compute_current when no recording is active.
// Uses an index outside the valid pool range (0..POOL_SIZE-1).
#define IREE_HAL_TASK_QUEUE_COMPUTE_NULL_TAG ((int64_t)(uint32_t)UINT32_MAX)

// Constructs a tagged compute_current value from a generation and pool index.
static inline int64_t iree_hal_task_queue_compute_item_tag(
    int32_t generation, uint32_t pool_index) {
  return ((int64_t)(uint32_t)generation << 32) | (int64_t)pool_index;
}

// Extracts the generation from a tagged compute_current value.
static inline int32_t iree_hal_task_queue_compute_item_tag_generation(
    int64_t tag) {
  return (int32_t)(uint32_t)(tag >> 32);
}

// Extracts the pool index from a tagged compute_current value.
static inline uint32_t iree_hal_task_queue_compute_item_tag_index(int64_t tag) {
  return (uint32_t)tag;
}

// Returns true if the tagged value represents no active recording.
static inline bool iree_hal_task_queue_compute_item_tag_is_null(int64_t tag) {
  return (uint32_t)tag >= IREE_HAL_TASK_QUEUE_COMPUTE_POOL_SIZE;
}

// Fires eager completion for a recording item. Called by the first worker to
// observe completed=true (won the CAS on release_pending). Consumes the
// processor's error, signals semaphores, advances the frontier, moves
// resources to the item for deferred release, and destroys the operation
// (freeing the arena).
static void iree_hal_task_queue_compute_item_complete(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_compute_item_t* item) {
  iree_hal_task_queue_op_t* operation = item->operation;

  // Consume the processor's accumulated error.
  iree_status_t status = iree_hal_cmd_block_processor_context_consume_result(
      item->processor_context);

  // Move resources and scope to the item for deferred release. Workers still
  // read command buffer recordings and buffer bindings during drain — the
  // resource_set must survive until all workers exit. Similarly, scope_end is
  // deferred so that scope_wait_idle blocks until all workers have fully
  // exited.
  item->resource_set = operation->resource_set;
  operation->resource_set = NULL;
  item->scope = operation->scope;
  operation->scope = NULL;

  // Complete or destroy the operation (signals semaphores, advances frontier,
  // frees arena).
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    if (operation->frontier_tracker) {
      iree_async_frontier_tracker_fail_axis(
          operation->frontier_tracker, operation->axis,
          iree_status_from_code(iree_status_code(status)));
    }
    iree_hal_task_queue_op_destroy(operation, status);
  }

  // Clear the back-pointer — the operation's arena is freed.
  item->operation = NULL;
}

// Fires deferred release for a recording item. Called by the last worker to
// decrement active_drainers to 0 (when release_pending is set). Frees the
// processor context, releases retained resources, calls scope_end, and
// returns the item to the free pool.
static void iree_hal_task_queue_compute_item_release(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_compute_item_t* item) {
  // Free the processor context. Safe — no workers are inside drain.
  if (item->processor_context) {
    iree_hal_cmd_block_processor_context_free(item->processor_context,
                                              item->host_allocator);
    item->processor_context = NULL;
  }

  // Release retained resources (command buffer, buffer bindings).
  if (item->resource_set) {
    iree_hal_resource_set_free(item->resource_set);
    item->resource_set = NULL;
  }

  // Capture scope before returning the item to the pool.
  iree_task_scope_t* scope = item->scope;
  item->scope = NULL;

  // Increment generation (ABA prevention), reset counters, return to pool.
  iree_atomic_fetch_add(&item->generation, 1, iree_memory_order_release);
  iree_atomic_store(&item->release_pending, 0, iree_memory_order_relaxed);
  iree_hal_task_queue_compute_item_slist_push(&queue->compute_free_pool, item);

  // scope_end after returning the item: idle waiters may deallocate the
  // scope's owner (the queue/device), so no item access after this.
  if (scope) {
    iree_task_scope_end(scope);
  }
}

// Compute process drain function. Called by workers from the compute slot.
// Loads the current recording item, drains it cooperatively via the block
// processor, and handles per-recording two-phase completion.
//
// Memory ordering protocol for compute_current:
//   The tagged pointer is loaded with acquire (pairs with release stores by
//   the completer or the null-branch installer). This ensures the item's
//   fields (processor_context, worker_count, worker_states) are visible.
//
// Per-recording active_drainers protocol:
//   Workers increment active_drainers before entering processor_drain and
//   decrement after. The increment uses acq_rel; the re-check of
//   compute_current after incrementing closes the TOCTOU window where the
//   item could be recycled between the initial load and the increment.
static iree_status_t iree_hal_task_queue_compute_process_drain(
    iree_task_process_t* process, uint32_t worker_index,
    iree_task_process_drain_result_t* out_result) {
  iree_hal_task_queue_t* queue = (iree_hal_task_queue_t*)process->user_data;

  // Check for shutdown.
  if (iree_atomic_load(&queue->shutting_down, iree_memory_order_acquire)) {
    out_result->did_work = false;
    out_result->completed = true;
    return iree_ok_status();
  }

  // Load the current recording item.
  int64_t tagged =
      iree_atomic_load(&queue->compute_current, iree_memory_order_acquire);

  if (!iree_hal_task_queue_compute_item_tag_is_null(tagged)) {
    uint32_t index = iree_hal_task_queue_compute_item_tag_index(tagged);
    int32_t expected_generation =
        iree_hal_task_queue_compute_item_tag_generation(tagged);
    iree_hal_task_queue_compute_item_t* item = &queue->compute_items[index];

    // Verify the item's generation matches. A mismatch means the item was
    // recycled (ABA) — retry on the next drain call.
    if (iree_atomic_load(&item->generation, iree_memory_order_acquire) !=
        expected_generation) {
      out_result->did_work = true;
      out_result->completed = false;
      return iree_ok_status();
    }

    // Commit to draining: increment active_drainers.
    iree_atomic_fetch_add(&item->active_drainers, 1, iree_memory_order_acq_rel);

    // Re-check compute_current. If it changed between our load and the
    // fetch_add, the item may have been completed and recycled. Back out.
    if (iree_atomic_load(&queue->compute_current, iree_memory_order_acquire) !=
        tagged) {
      iree_atomic_fetch_sub(&item->active_drainers, 1,
                            iree_memory_order_release);
      out_result->did_work = true;
      out_result->completed = false;
      return iree_ok_status();
    }

    // Drain the block processor.
    iree_hal_cmd_block_processor_worker_state_t* worker_state =
        &item->worker_states[worker_index % item->worker_count];
    iree_hal_cmd_block_processor_drain_result_t processor_result;
    memset(&processor_result, 0, sizeof(processor_result));
    iree_hal_cmd_block_processor_drain(item->processor_context, worker_index,
                                       worker_state, &processor_result);

    // Handle per-recording completion.
    if (processor_result.completed) {
      // Try to claim completion. First worker to CAS wins.
      int32_t expected = 0;
      if (iree_atomic_compare_exchange_strong(&item->release_pending, &expected,
                                              1, iree_memory_order_acq_rel,
                                              iree_memory_order_relaxed)) {
        // Fire eager completion: signal semaphores, advance frontier, free
        // the operation arena.
        iree_hal_task_queue_compute_item_complete(queue, item);

        // Install the next pending recording (or null if none).
        iree_hal_task_queue_compute_item_t* next =
            iree_hal_task_queue_compute_item_slist_pop(&queue->compute_pending);
        if (next) {
          int32_t next_generation =
              iree_atomic_load(&next->generation, iree_memory_order_relaxed);
          iree_atomic_store(&queue->compute_current,
                            iree_hal_task_queue_compute_item_tag(
                                next_generation, next->pool_index),
                            iree_memory_order_release);
        } else {
          iree_atomic_store(&queue->compute_current,
                            IREE_HAL_TASK_QUEUE_COMPUTE_NULL_TAG,
                            iree_memory_order_release);
        }
      }
    }

    // Release our drainer claim. If we're the last drainer and the recording
    // is completed (release_pending set), fire deferred release.
    int32_t old_drainers = iree_atomic_fetch_sub(&item->active_drainers, 1,
                                                 iree_memory_order_acq_rel);
    if (old_drainers == 1 &&
        iree_atomic_load(&item->release_pending, iree_memory_order_acquire)) {
      iree_hal_task_queue_compute_item_release(queue, item);
    }

    out_result->did_work =
        processor_result.tiles_executed > 0 || processor_result.completed;
    out_result->completed = false;
    return iree_ok_status();
  }

  // No current item. Try to pop from the pending list and install it.
  iree_hal_task_queue_compute_item_t* item =
      iree_hal_task_queue_compute_item_slist_pop(&queue->compute_pending);
  if (item) {
    int32_t generation =
        iree_atomic_load(&item->generation, iree_memory_order_relaxed);
    int64_t new_tag =
        iree_hal_task_queue_compute_item_tag(generation, item->pool_index);

    // CAS(null → new_tag) prevents racing with a completer that installed
    // a new item between our null-check and this point.
    int64_t expected = tagged;
    if (iree_atomic_compare_exchange_strong(&queue->compute_current, &expected,
                                            new_tag, iree_memory_order_release,
                                            iree_memory_order_relaxed)) {
      out_result->did_work = true;
      out_result->completed = false;
      return iree_ok_status();
    }

    // CAS failed — someone else installed an item. Push ours back.
    iree_hal_task_queue_compute_item_slist_push(&queue->compute_pending, item);
    out_result->did_work = true;
    out_result->completed = false;
    return iree_ok_status();
  }

  // Nothing available. Workers will park via the sleeping protocol.
  out_result->did_work = false;
  out_result->completed = false;
  return iree_ok_status();
}

// Compute process completion callback. Fires eagerly when the first worker
// observes the process has completed (shutting_down set). Other workers may
// still be inside the drain function at this point, so we must NOT call
// scope_end here — scope_wait_idle returning would allow the main thread to
// free the queue while workers are still accessing it.
//
// The process-level scope_end is deferred to the release callback, which
// fires only after the last drainer exits (active_drainers reaches 0).
static void iree_hal_task_queue_compute_process_completion(
    iree_task_process_t* process, iree_status_t status) {
  iree_status_ignore(status);
}

// Cleans up a compute item during shutdown. Handles both cases:
//   - Eager completion never fired: operation is still on the item. Destroy
//     it (which signals semaphores with failure, ends scope, frees arena).
//   - Eager completion fired but deferred release didn't: operation was
//     destroyed but resources and scope were moved to the item. Free
//     resources and end scope.
static void iree_hal_task_queue_compute_item_cleanup(
    iree_hal_task_queue_compute_item_t* item) {
  // Consume the processor's error.
  iree_status_t processor_status =
      iree_hal_cmd_block_processor_context_consume_result(
          item->processor_context);
  iree_hal_cmd_block_processor_context_free(item->processor_context,
                                            item->host_allocator);
  item->processor_context = NULL;

  // Release retained resources (either on the item from eager completion,
  // or still on the operation if completion never fired).
  if (item->resource_set) {
    iree_hal_resource_set_free(item->resource_set);
    item->resource_set = NULL;
  }

  if (item->operation) {
    // Eager completion never fired — the operation owns its resources and
    // scope. op_destroy handles everything (fail semaphores, scope_end,
    // free arena).
    iree_hal_task_queue_op_destroy(
        item->operation,
        iree_status_is_ok(processor_status)
            ? iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down")
            : processor_status);
    item->operation = NULL;
  } else {
    iree_status_ignore(processor_status);
    // Eager completion already handled the operation. The scope was moved
    // to the item for deferred release — end it now.
    if (item->scope) {
      iree_task_scope_end(item->scope);
      item->scope = NULL;
    }
  }
}

// Compute process release callback. Fires when the last worker exits the
// compute slot (active_drainers reaches 0 after process completion). At this
// point, no workers are touching compute process state — this is the safe
// place for both item cleanup and scope_end.
//
// The process-level scope_end (paired with scope_begin at initialization)
// fires at the very end of this function, AFTER all queue state accesses.
// This ensures scope_wait_idle does not unblock until every worker has fully
// exited drain and all item cleanup is complete. Placing scope_end in the
// completion callback (which fires while other workers may still be draining)
// would allow the main thread to free the queue prematurely.
static void iree_hal_task_queue_compute_process_release(
    iree_task_process_t* process) {
  iree_hal_task_queue_t* queue = (iree_hal_task_queue_t*)process->user_data;

  // Scan ALL pool items for any that need cleanup. Items can be in three
  // states at shutdown:
  //
  //   (a) In the free pool (processor_context=NULL, scope=NULL) — clean.
  //   (b) Currently installed as compute_current or in compute_pending —
  //       either mid-drain (shutdown interrupted processing) or waiting to
  //       be installed. processor_context is non-NULL.
  //   (c) Eagerly completed but deferred release hasn't fired — the
  //       completer signaled semaphores and installed null/next for
  //       compute_current, but a worker hasn't decremented the item's
  //       active_drainers to 0 yet. processor_context is non-NULL,
  //       operation is NULL, scope is non-NULL (moved from operation).
  //
  // Case (c) is critical: these items are NOT reachable from compute_current
  // or compute_pending. They're in limbo between eager completion and
  // deferred release. Without scanning the full pool, their scope_end would
  // never fire and scope_wait_idle would hang.
  //
  // This is safe because the slot release only fires after all slot drainers
  // have exited (active_drainers sentinel CAS). Since per-recording
  // active_drainers are decremented inside the drain function (before the
  // worker exits the slot), no worker can be accessing any item at this point.
  for (uint32_t i = 0; i < IREE_HAL_TASK_QUEUE_COMPUTE_POOL_SIZE; ++i) {
    iree_hal_task_queue_compute_item_t* item = &queue->compute_items[i];
    if (item->processor_context || item->scope) {
      iree_hal_task_queue_compute_item_cleanup(item);
    }
  }

  // Reset compute_current (may have been pointing to a cleaned-up item).
  iree_atomic_store(&queue->compute_current,
                    IREE_HAL_TASK_QUEUE_COMPUTE_NULL_TAG,
                    iree_memory_order_relaxed);

  // Discard the pending list. Items here were filled by the budget-1 process
  // but never installed as compute_current. They were already cleaned up
  // in the pool scan above; just clear the slist head.
  iree_hal_task_queue_compute_item_slist_discard(&queue->compute_pending);

  // End the scope for the compute process (paired with scope_begin at init).
  // This MUST be the last access to queue state. After this call,
  // scope_wait_idle may unblock and the main thread may immediately free
  // the queue and all embedded fields.
  iree_task_scope_end(&queue->scope);
}

//===----------------------------------------------------------------------===//
// Control process (budget-1 queue drain)
//===----------------------------------------------------------------------===//

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
    case IREE_HAL_TASK_QUEUE_OP_ALLOCA:
      status = iree_hal_task_queue_drain_alloca(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_DEALLOCA:
      iree_hal_task_queue_drain_dealloca(queue, operation);
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

  // Initialize the compute process. Budget-N where N is the worker count.
  // Not scheduled until the first recording is pushed to compute_pending
  // by drain_commands. The first schedule_process call places it in a
  // compute slot; subsequent calls just wake workers.
  //
  // The matching scope_end fires in the RELEASE callback (not the completion
  // callback). For budget>1 processes, completion fires eagerly while other
  // workers may still be draining — scope_end there would allow the main
  // thread to free the queue prematurely. The release callback fires only
  // after the last drainer exits, making it safe to unblock scope_wait_idle.
  iree_task_process_initialize(
      iree_hal_task_queue_compute_process_drain,
      /*suspend_count=*/0,
      /*worker_budget=*/(int32_t)iree_task_executor_worker_count(executor),
      &out_queue->compute_process);
  out_queue->compute_process.user_data = out_queue;
  out_queue->compute_process.completion_fn =
      iree_hal_task_queue_compute_process_completion;
  out_queue->compute_process.release_fn =
      iree_hal_task_queue_compute_process_release;
  iree_task_scope_begin(&out_queue->scope);

  // Initialize the compute pending and free pool lists.
  iree_hal_task_queue_compute_item_slist_initialize(
      &out_queue->compute_pending);
  iree_hal_task_queue_compute_item_slist_initialize(
      &out_queue->compute_free_pool);

  // Set compute_current to the null sentinel (memset set it to 0 which
  // would alias {generation=0, index=0} — a valid item).
  iree_atomic_store(&out_queue->compute_current,
                    IREE_HAL_TASK_QUEUE_COMPUTE_NULL_TAG,
                    iree_memory_order_relaxed);

  // Initialize pool items and push them all to the free pool.
  for (uint32_t i = 0; i < IREE_HAL_TASK_QUEUE_COMPUTE_POOL_SIZE; ++i) {
    iree_hal_task_queue_compute_item_t* item = &out_queue->compute_items[i];
    memset(item, 0, sizeof(*item));
    item->pool_index = i;
    iree_hal_task_queue_compute_item_slist_push(&out_queue->compute_free_pool,
                                                item);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Signal both processes to complete. The next drain call on each sees this
  // flag and returns completed=true, triggering normal process completion.
  iree_atomic_store(&queue->shutting_down, 1, iree_memory_order_release);

  // Schedule both processes so workers pick them up and see the shutdown flag.
  // If a process is already being drained, schedule_process sets needs_drain
  // and the worker will see shutting_down on the next iteration. If IDLE,
  // schedule_process pushes it to the appropriate run list.
  iree_task_executor_schedule_process(queue->executor, &queue->process);
  iree_task_executor_schedule_process(queue->executor, &queue->compute_process);

  // Wait for all outstanding operations and both processes to complete.
  // The budget-1 process's scope_end fires in its completion callback.
  // The compute process's scope_end fires in its release callback (after
  // the last drainer exits), which also cleans up any in-flight items.
  // Each submitted operation has its own scope_begin/end pair. scope_wait_idle
  // returns only when all of these have resolved — meaning every worker has
  // finished touching queue/device resources.
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

  // Deinitialize the compute lists. Items are stack-allocated in the queue
  // struct so only the slist state needs cleanup.
  iree_hal_task_queue_compute_item_slist_deinitialize(&queue->compute_pending);
  iree_hal_task_queue_compute_item_slist_deinitialize(
      &queue->compute_free_pool);

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

iree_status_t iree_hal_task_queue_submit_alloca(
    iree_hal_task_queue_t* queue, iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_t* transient_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_ALLOCA, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  // Store alloca parameters for the drain handler.
  operation->alloca.device_allocator = device_allocator;
  operation->alloca.params = params;
  operation->alloca.allocation_size = allocation_size;
  operation->alloca.transient_buffer = transient_buffer;

  // Retain the transient buffer so it survives until the operation completes.
  status = iree_hal_resource_set_insert(
      operation->resource_set, 1,
      (iree_hal_resource_t* const*)&transient_buffer);

  // Fast path: check if all wait semaphores are already satisfied.
  if (iree_status_is_ok(status) && wait_semaphores.count > 0) {
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

iree_status_t iree_hal_task_queue_submit_dealloca(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* transient_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_DEALLOCA, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  // Store the transient buffer for decommit in the drain handler.
  operation->dealloca.transient_buffer = transient_buffer;

  // Retain the transient buffer so it survives until the operation completes.
  status = iree_hal_resource_set_insert(
      operation->resource_set, 1,
      (iree_hal_resource_t* const*)&transient_buffer);

  // Fast path: check if all wait semaphores are already satisfied.
  if (iree_status_is_ok(status) && wait_semaphores.count > 0) {
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
