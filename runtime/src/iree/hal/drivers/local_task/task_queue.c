// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_queue.h"

#include <stddef.h>
#include <string.h>

#include "iree/async/file.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/operations/file.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/async/span.h"
#include "iree/hal/drivers/local_task/block_builder.h"
#include "iree/hal/drivers/local_task/block_command_buffer.h"
#include "iree/hal/drivers/local_task/block_command_ops.h"
#include "iree/hal/drivers/local_task/block_processor.h"
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
  // Unmap SCOPED binding table mappings before signaling semaphores: the
  // unmap may flush non-coherent memory, so waiters must not observe the
  // signal until writes are visible. Must also precede resource_set_free
  // (which drops the buffer references the mappings point into).
  //
  // The mappings array is 1:1 with binding_table entries; NULL-buffer slots
  // have zeroed mappings that unmap_range handles as no-ops.
  //
  // On the success path, an unmap failure becomes the operation's failure
  // status (surfaced to semaphores below). On the failure path, we already
  // have an error to propagate so unmap failures are secondary.
  if (operation->type == IREE_HAL_TASK_QUEUE_OP_COMMANDS &&
      operation->commands.binding_mappings) {
    for (iree_host_size_t i = 0; i < operation->commands.binding_table.count;
         ++i) {
      iree_status_t unmap_status =
          iree_hal_buffer_unmap_range(&operation->commands.binding_mappings[i]);
      if (!iree_status_is_ok(unmap_status)) {
        if (iree_status_is_ok(failure_status)) {
          failure_status = unmap_status;
        } else {
          iree_status_ignore(unmap_status);
        }
      }
    }
  }

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

// Fails an operation: propagates the error to the frontier tracker (if present)
// and destroys the operation with the given failure status.
static void iree_hal_task_queue_op_fail(iree_hal_task_queue_op_t* operation,
                                        iree_status_t status) {
  if (operation->frontier_tracker) {
    iree_async_frontier_tracker_fail_axis(
        operation->frontier_tracker, operation->axis,
        iree_status_from_code(iree_status_code(status)));
  }
  iree_hal_task_queue_op_destroy(operation, status);
}

// Completes an operation successfully: signals semaphores, advances the
// frontier, then destroys the operation (freeing the arena).
static void iree_hal_task_queue_op_complete(
    iree_hal_task_queue_op_t* operation) {
  // Signal all semaphores to their new values.
  iree_status_t status =
      iree_hal_semaphore_list_signal(operation->signal_semaphores);

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
    iree_hal_task_queue_op_fail(operation, status);
    return;
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
    iree_status_t status =
        iree_arena_allocate(&operation->arena, sizeof(*entry), (void**)&entry);

    // Register the timepoint. The callback may fire synchronously.
    if (iree_status_is_ok(status)) {
      entry->operation = operation;
      entry->semaphore = semaphore;
      iree_hal_semaphore_retain(semaphore);
      entry->timepoint.callback = iree_hal_task_queue_wait_resolved;
      entry->timepoint.user_data = entry;
      status = iree_async_semaphore_acquire_timepoint(
          (iree_async_semaphore_t*)semaphore, minimum_value, &entry->timepoint);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_release(semaphore);
      }
    }

    if (!iree_status_is_ok(status)) {
      // Registration failed at index i. Timepoints 0..i-1 are already
      // registered and their callbacks will fire asynchronously. We cannot
      // destroy the operation here — the callbacks would access freed arena
      // memory. Instead, record the error and subtract the unregistered
      // count from wait_count so the existing callbacks can drain and
      // destroy the operation when the last one completes.
      intptr_t expected = 0;
      if (!iree_atomic_compare_exchange_strong(
              &operation->error_status, &expected, (intptr_t)status,
              iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
        iree_status_ignore(status);
      }
      // Subtract count for this entry and all remaining unregistered ones.
      int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      int32_t previous_count = iree_atomic_fetch_sub(
          &operation->wait_count, unregistered, iree_memory_order_acq_rel);
      if (previous_count == unregistered) {
        // All registered timepoints already fired. We are the last
        // decrement — destroy the operation with the recorded error.
        iree_status_t error = (iree_status_t)iree_atomic_exchange(
            &operation->error_status, 0, iree_memory_order_acquire);
        iree_hal_task_queue_op_destroy(operation, error);
      }
      // Return OK: the operation is owned by the timepoint callbacks (or
      // already destroyed above). The caller must not touch it.
      return iree_ok_status();
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Per-operation drain functions
//===----------------------------------------------------------------------===//

// Routes a recording through the compute process for multi-worker execution.
// Acquires a compute item, allocates the processor context, and schedules the
// compute process. The recording is referenced (not copied) — the caller
// ensures it stays alive until the compute item's deferred release.
//
// If |owned_recording| is non-NULL, the compute item takes ownership and
// releases the blocks in its deferred release path. If NULL, the caller
// retains ownership (e.g., the recording lives inside a command buffer that
// the resource_set keeps alive).
static iree_status_t iree_hal_task_queue_drain_recording(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation,
    const iree_hal_cmd_block_recording_t* recording,
    iree_hal_cmd_block_recording_t* owned_recording,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length) {
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(queue->device_allocator);

  // Acquire a recording item from the free pool.
  iree_hal_task_queue_compute_item_t* item =
      iree_hal_task_queue_compute_item_slist_pop(&queue->compute_free_pool);
  if (!item) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "compute recording pool exhausted (pool size %d); too many "
        "concurrent recordings in flight",
        IREE_HAL_TASK_QUEUE_COMPUTE_POOL_SIZE);
  }

  // Allocate the block processor execution context.
  uint32_t worker_count =
      (uint32_t)iree_task_executor_worker_count(queue->executor);
  if (worker_count == 0) worker_count = 1;
  iree_hal_cmd_block_processor_context_t* processor_context = NULL;
  iree_status_t status = iree_hal_cmd_block_processor_context_allocate(
      recording, binding_table, binding_table_length, worker_count,
      host_allocator, &processor_context);
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
  if (owned_recording) {
    item->recording = *owned_recording;
    memset(owned_recording, 0, sizeof(*owned_recording));
  } else {
    memset(&item->recording, 0, sizeof(item->recording));
  }
  memset(item->worker_states, 0, sizeof(item->worker_states[0]) * worker_count);
  // drainers retains generation from previous lifecycle. Count and CLOSED
  // were cleared during the previous cleanup. First use: memset zeroed it.

  // Push to the compute pending list.
  iree_hal_task_queue_compute_item_slist_push(&queue->compute_pending, item);

  // Schedule the compute process. For the first recording, this places the
  // process in a compute slot (CAS IDLE→DRAINING). For subsequent recordings,
  // the CAS fails (already DRAINING) but wake_workers still runs, ensuring
  // workers pick up the new recording from the pending list.
  iree_task_executor_schedule_process(queue->executor, &queue->compute_process);

  return iree_ok_status();
}

// Handles a COMMANDS operation: extracts the recording from the command buffer
// and routes it through the compute process.
static iree_status_t iree_hal_task_queue_drain_commands(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  // Verify the command buffer is a block command buffer.
  if (!iree_hal_block_command_buffer_isa(operation->commands.command_buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue only accepts block command buffers; got unrecognized type");
  }

  // Get the recording from the command buffer. The CB is retained in the
  // operation's resource_set, so the recording stays alive until the compute
  // item's deferred release frees the resource_set.
  const iree_hal_cmd_block_recording_t* recording =
      iree_hal_block_command_buffer_recording(
          operation->commands.command_buffer);

  // Resolve the HAL binding table to block ISA binding entries. Each entry
  // maps a HAL buffer to a host pointer for indirect fixup resolution.
  // The entries and SCOPED mappings are allocated from the operation's arena.
  // Mappings are tracked on the operation for unmap in op_destroy.
  const iree_hal_cmd_binding_entry_t* binding_table = NULL;
  iree_host_size_t binding_table_length = 0;
  const iree_hal_buffer_binding_table_t hal_table =
      operation->commands.binding_table;
  if (hal_table.count > 0) {
    iree_hal_cmd_binding_entry_t* entries = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(&operation->arena,
                                             hal_table.count * sizeof(*entries),
                                             (void**)&entries));
    iree_hal_buffer_mapping_t* mappings = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &operation->arena, hal_table.count * sizeof(*mappings),
        (void**)&mappings));
    memset(mappings, 0, hal_table.count * sizeof(*mappings));
    for (iree_host_size_t i = 0; i < hal_table.count; ++i) {
      const iree_hal_buffer_binding_t* binding = &hal_table.bindings[i];
      if (binding->buffer) {
        IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
            binding->buffer, IREE_HAL_MAPPING_MODE_SCOPED,
            IREE_HAL_MEMORY_ACCESS_ANY, binding->offset, binding->length,
            &mappings[i]));
        entries[i].base = mappings[i].contents.data;
        entries[i].length = mappings[i].contents.data_length;
      } else {
        entries[i].base = NULL;
        entries[i].length = 0;
      }
    }
    operation->commands.binding_mappings = mappings;
    binding_table = entries;
    binding_table_length = hal_table.count;
  }

  return iree_hal_task_queue_drain_recording(
      queue, operation, recording,
      /*owned_recording=*/NULL, binding_table, binding_table_length);
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
    status = iree_hal_semaphore_list_signal(operation->signal_semaphores);
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
        status = iree_hal_semaphore_list_signal(operation->signal_semaphores);
      }
    } else {
      // Callback failed; propagate failure to signal semaphores, release the
      // references, and clear the list to prevent op_destroy from
      // double-signaling or double-releasing. Clone the status before
      // passing to semaphore_list_fail (which consumes it for the last
      // semaphore) so we retain a copy for frontier failure below.
      status = iree_status_clone(call_status);
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
// Inline recording execution (fill, copy, update, inline dispatch)
//===----------------------------------------------------------------------===//

// Executes a block recording synchronously with a single worker.
// Used by the drain handlers for fill/copy/update and inline dispatch.
// On success, completes the operation. On failure, destroys it with the error.
//
// The processor context is stack-allocated. The .data state is allocated from
// the operation's arena (which is backed by the small block pool — typically
// fits in the same 4KB block that already holds the operation).
static void iree_hal_task_queue_execute_recording_inline(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation,
    iree_hal_cmd_block_recording_t* recording) {
  iree_status_t status = iree_ok_status();

  // Allocate .data from the operation's arena.
  const iree_host_size_t state_size = iree_hal_cmd_block_state_size(
      recording->max_region_dispatch_count, recording->max_total_binding_count);
  void* state_storage = NULL;
  if (recording->first_block && state_size > 0) {
    status = iree_arena_allocate_aligned(
        &operation->arena, state_size, iree_alignof(iree_hal_cmd_block_state_t),
        &state_storage);
  }

  if (iree_status_is_ok(status) && recording->first_block) {
    // Context on the stack — no heap allocation.
    iree_hal_cmd_block_processor_context_t processor_context;
    iree_hal_cmd_block_processor_context_initialize(
        &processor_context, recording, /*binding_table=*/NULL,
        /*binding_table_length=*/0, (iree_hal_cmd_block_state_t*)state_storage,
        state_size);

    iree_hal_cmd_block_processor_worker_state_t worker_state = {0};
    iree_hal_cmd_block_processor_drain_result_t result;
    iree_hal_cmd_block_processor_drain(&processor_context, 0, &worker_state,
                                       &result);
    status =
        iree_hal_cmd_block_processor_context_consume_result(&processor_context);
  }

  iree_hal_cmd_block_recording_release(recording);

  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_destroy(operation, status);
  }
}

// Handles a FILL operation: builds a single-command recording and executes
// it inline via the block processor.
static iree_status_t iree_hal_task_queue_drain_fill(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  iree_hal_buffer_mapping_t mapping = {{0}};

  iree_status_t status = iree_hal_buffer_map_range(
      operation->fill.target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, operation->fill.target_offset,
      operation->fill.length, &mapping);

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(queue->large_block_pool, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_begin(&builder);
  }

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_build_fill(
        &builder, operation->fill.length, operation->fill.pattern,
        operation->fill.pattern_length, &fixups, &token);
  }
  if (iree_status_is_ok(status)) {
    fixups[0].host_ptr = mapping.contents.data;
    fixups[0].offset = 0;
    fixups[0].length = mapping.contents.data_length;
    fixups[0].slot = 0;
    fixups[0].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
  }

  iree_hal_cmd_block_recording_t recording;
  memset(&recording, 0, sizeof(recording));
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_end(&builder, &recording);
  }

  iree_hal_cmd_block_builder_deinitialize(&builder);
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_execute_recording_inline(queue, operation, &recording);
  }

  // Unmap the buffer after inline execution completes. Safe on zero-initialized
  // mappings (no-op if map_range was never called or failed).
  iree_hal_buffer_unmap_range(&mapping);
  return status;
}

// Handles a COPY operation: builds a single-command recording and executes
// it inline via the block processor.
static iree_status_t iree_hal_task_queue_drain_copy(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  iree_hal_buffer_mapping_t source_mapping = {{0}};
  iree_hal_buffer_mapping_t target_mapping = {{0}};

  iree_status_t status = iree_hal_buffer_map_range(
      operation->copy.source_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, operation->copy.source_offset,
      operation->copy.length, &source_mapping);
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        operation->copy.target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, operation->copy.target_offset,
        operation->copy.length, &target_mapping);
  }

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(queue->large_block_pool, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_begin(&builder);
  }

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_build_copy(&builder, operation->copy.length, &fixups,
                                     &token);
  }
  if (iree_status_is_ok(status)) {
    fixups[0].host_ptr = source_mapping.contents.data;
    fixups[0].offset = 0;
    fixups[0].length = source_mapping.contents.data_length;
    fixups[0].slot = 0;
    fixups[0].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
    fixups[1].host_ptr = target_mapping.contents.data;
    fixups[1].offset = 0;
    fixups[1].length = target_mapping.contents.data_length;
    fixups[1].slot = 0;
    fixups[1].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
  }

  iree_hal_cmd_block_recording_t recording;
  memset(&recording, 0, sizeof(recording));
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_end(&builder, &recording);
  }

  iree_hal_cmd_block_builder_deinitialize(&builder);
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_execute_recording_inline(queue, operation, &recording);
  }

  iree_hal_buffer_unmap_range(&source_mapping);
  iree_hal_buffer_unmap_range(&target_mapping);
  return status;
}

// Handles an UPDATE operation: builds a single-command recording and executes
// it inline via the block processor.
static iree_status_t iree_hal_task_queue_drain_update(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  iree_hal_buffer_mapping_t mapping = {{0}};
  iree_status_t status = iree_hal_buffer_map_range(
      operation->update.target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, operation->update.target_offset,
      operation->update.length, &mapping);

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(queue->large_block_pool, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_begin(&builder);
  }

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_cmd_build_update(&builder, operation->update.source_data, 0,
                                  operation->update.length, &fixups, &token);
  }
  if (iree_status_is_ok(status)) {
    fixups[0].host_ptr = mapping.contents.data;
    fixups[0].offset = 0;
    fixups[0].length = mapping.contents.data_length;
    fixups[0].slot = 0;
    fixups[0].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
  }

  iree_hal_cmd_block_recording_t recording;
  memset(&recording, 0, sizeof(recording));
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_end(&builder, &recording);
  }

  iree_hal_cmd_block_builder_deinitialize(&builder);
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_execute_recording_inline(queue, operation, &recording);
  }

  iree_hal_buffer_unmap_range(&mapping);
  return status;
}

// Handles a DISPATCH operation: builds a single-dispatch recording and either
// executes it inline (ALLOW_INLINE_EXECUTION) or routes it through the compute
// process for multi-worker tile distribution.
static iree_status_t iree_hal_task_queue_drain_dispatch(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation);

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

  // Complete or fail the operation (signals semaphores, advances frontier,
  // frees arena).
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_fail(operation, status);
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

  // Release queue-built recording blocks (if any). For command buffer
  // recordings, first_block is NULL (the CB retains its own recording).
  if (item->recording.first_block) {
    iree_hal_cmd_block_recording_release(&item->recording);
    memset(&item->recording, 0, sizeof(item->recording));
  }

  // Release retained resources (command buffer, buffer bindings).
  if (item->resource_set) {
    iree_hal_resource_set_free(item->resource_set);
    item->resource_set = NULL;
  }

  // Capture scope before returning the item to the pool.
  iree_task_scope_t* scope = item->scope;
  item->scope = NULL;

  // Reset drainers: increment generation, clear count and CLOSED flag.
  int64_t old_drainers =
      iree_atomic_load(&item->drainers, iree_memory_order_relaxed);
  int64_t next_gen = (old_drainers & ~(int64_t)UINT32_MAX) +
                     IREE_HAL_TASK_QUEUE_ITEM_GEN_INCREMENT;
  iree_atomic_store(&item->drainers, next_gen, iree_memory_order_release);
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

    // Register as a drainer via fetch_add on the 64-bit drainers field.
    // The item is immortal (embedded in the queue), so this is always safe.
    // If the CLOSED flag is set (bit 31 of the low 32 bits), the recording
    // is being cleaned up — bail immediately.
    int64_t prev_drainers =
        iree_atomic_fetch_add(&item->drainers, 1, iree_memory_order_acq_rel);
    if (IREE_UNLIKELY((int32_t)prev_drainers < 0)) {
      iree_atomic_fetch_sub(&item->drainers, 1, iree_memory_order_release);
      out_result->did_work = true;
      out_result->completed = false;
      return iree_ok_status();
    }

    // ABA check: verify generation in drainers matches the tag from
    // compute_current. If the item was recycled between our tag load and
    // the fetch_add, bail. Stale fetch_sub on a recycled item is harmless
    // because the new generation's CLOSED flag is clear.
    if ((prev_drainers >> 32) != (int64_t)(uint32_t)expected_generation) {
      iree_atomic_fetch_sub(&item->drainers, 1, iree_memory_order_release);
      out_result->did_work = true;
      out_result->completed = false;
      return iree_ok_status();
    }

    // Re-check compute_current for identity safety.
    if (iree_atomic_load(&queue->compute_current, iree_memory_order_acquire) !=
        tagged) {
      iree_atomic_fetch_sub(&item->drainers, 1, iree_memory_order_release);
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

    // Handle per-recording completion via CLOSED flag. fetch_or atomically
    // sets the flag AND returns the previous value — no TOCTOU possible.
    if (processor_result.completed) {
      int64_t close_prev = iree_atomic_fetch_or(
          &item->drainers, IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT,
          iree_memory_order_acq_rel);
      if (!((int32_t)close_prev &
            (int32_t)IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT)) {
        // We set the CLOSED flag first — fire eager completion.
        iree_hal_task_queue_compute_item_complete(queue, item);

        // Install the next pending recording (or null if none).
        iree_hal_task_queue_compute_item_t* next =
            iree_hal_task_queue_compute_item_slist_pop(&queue->compute_pending);
        if (next) {
          int64_t next_drainers =
              iree_atomic_load(&next->drainers, iree_memory_order_relaxed);
          int32_t next_generation = (int32_t)(next_drainers >> 32);
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

    // Release our drainer claim. If we were the last drainer after close,
    // fire deferred cleanup. We check the low 32 bits only: if they equal
    // (CLOSED | 1), our fetch_sub transitions to (CLOSED | 0) and we own
    // cleanup. The generation in the high bits doesn't affect this check.
    {
      int64_t exit_prev =
          iree_atomic_fetch_sub(&item->drainers, 1, iree_memory_order_acq_rel);
      if ((int32_t)exit_prev ==
          ((int32_t)IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT | 1)) {
        iree_hal_task_queue_compute_item_release(queue, item);
      }
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
    int64_t item_drainers =
        iree_atomic_load(&item->drainers, iree_memory_order_relaxed);
    int32_t generation = (int32_t)(item_drainers >> 32);
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

  // Release queue-built recording blocks (if any).
  if (item->recording.first_block) {
    iree_hal_cmd_block_recording_release(&item->recording);
    memset(&item->recording, 0, sizeof(item->recording));
  }

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
  // have exited (active_drainers sentinel CAS), and the process's
  // schedule_state transitions to IDLE only in release_compute_process
  // (not in eager_complete), preventing overlapping slot lifetimes.
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
// File I/O (proactor-based async read/write)
//===----------------------------------------------------------------------===//

// Context for an in-flight file read or write operation. Arena-allocated from
// the queue operation's arena and valid until the proactor completion callback
// fires. The callback unmaps the buffer and completes/destroys the operation,
// which frees the arena (and this context with it).
typedef struct iree_hal_task_queue_io_context_t {
  iree_hal_task_queue_op_t* operation;
  iree_hal_buffer_mapping_t mapping;
  union {
    iree_async_file_read_operation_t read_op;
    iree_async_file_write_operation_t write_op;
  };
} iree_hal_task_queue_io_context_t;

// Proactor completion callback for file read operations.
static void iree_hal_task_queue_io_read_completion(
    void* user_data, iree_async_operation_t* base_op, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_task_queue_io_context_t* io_context =
      (iree_hal_task_queue_io_context_t*)user_data;
  iree_hal_task_queue_op_t* operation = io_context->operation;

  // Validate the full requested amount was read.
  if (iree_status_is_ok(status)) {
    if (io_context->read_op.bytes_read < io_context->read_op.buffer.length) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "short read: requested %" PRIhsz " bytes, got %" PRIhsz,
          io_context->read_op.buffer.length, io_context->read_op.bytes_read);
    }
  }

  // Flush non-coherent memory: proactor wrote data into the mapped buffer
  // and the device needs to see it.
  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(
          iree_hal_buffer_memory_type(io_context->mapping.buffer),
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_status_join(
        status,
        iree_hal_buffer_mapping_flush_range(
            &io_context->mapping, 0, io_context->mapping.contents.data_length));
  }

  // Unmap the buffer.
  status = iree_status_join(status,
                            iree_hal_buffer_unmap_range(&io_context->mapping));

  // Complete or fail the operation. This frees the arena (and io_context).
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_fail(operation, status);
  }
}

// Proactor completion callback for file write operations.
static void iree_hal_task_queue_io_write_completion(
    void* user_data, iree_async_operation_t* base_op, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_task_queue_io_context_t* io_context =
      (iree_hal_task_queue_io_context_t*)user_data;
  iree_hal_task_queue_op_t* operation = io_context->operation;

  // Validate the full requested amount was written.
  if (iree_status_is_ok(status)) {
    if (io_context->write_op.bytes_written <
        io_context->write_op.buffer.length) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "short write: requested %" PRIhsz
                                " bytes, wrote %" PRIhsz,
                                io_context->write_op.buffer.length,
                                io_context->write_op.bytes_written);
    }
  }

  // Unmap the buffer.
  status = iree_status_join(status,
                            iree_hal_buffer_unmap_range(&io_context->mapping));

  // Complete or fail the operation. This frees the arena (and io_context).
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_fail(operation, status);
  }
}

// Handles a READ operation: maps the target buffer, submits an async proactor
// read, and returns immediately. The proactor callback handles unmapping and
// operation completion.
static iree_status_t iree_hal_task_queue_drain_read(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  // Map the target buffer for writing.
  iree_hal_task_queue_io_context_t* io_context = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &operation->arena, sizeof(*io_context), (void**)&io_context));
  memset(io_context, 0, sizeof(*io_context));
  io_context->operation = operation;

  iree_status_t status = iree_hal_buffer_map_range(
      operation->read.buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, operation->read.buffer_offset,
      operation->read.length, &io_context->mapping);
  if (!iree_status_is_ok(status)) return status;

  // Initialize the proactor read operation.
  iree_async_operation_zero(&io_context->read_op.base,
                            sizeof(io_context->read_op));
  iree_async_operation_initialize(
      &io_context->read_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_task_queue_io_read_completion,
      io_context);
  io_context->read_op.file = operation->read.async_file;
  io_context->read_op.offset = operation->read.file_offset;
  io_context->read_op.buffer =
      iree_async_span_from_ptr(io_context->mapping.contents.data,
                               (iree_host_size_t)operation->read.length);

  // Submit to the proactor and return immediately.
  iree_status_t submit_status = iree_async_proactor_submit_one(
      queue->proactor, &io_context->read_op.base);
  if (!iree_status_is_ok(submit_status)) {
    iree_hal_buffer_unmap_range(&io_context->mapping);
  }
  return submit_status;
}

// Handles a WRITE operation: maps the source buffer, submits an async proactor
// write, and returns immediately. The proactor callback handles unmapping and
// operation completion.
static iree_status_t iree_hal_task_queue_drain_write(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  // Map the source buffer for reading.
  iree_hal_task_queue_io_context_t* io_context = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &operation->arena, sizeof(*io_context), (void**)&io_context));
  memset(io_context, 0, sizeof(*io_context));
  io_context->operation = operation;

  iree_status_t status = iree_hal_buffer_map_range(
      operation->write.buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, operation->write.buffer_offset,
      operation->write.length, &io_context->mapping);
  if (!iree_status_is_ok(status)) return status;

  // Invalidate non-coherent memory: the device may have written data into
  // the buffer and we need to see it before reading for the file write.
  if (!iree_all_bits_set(
          iree_hal_buffer_memory_type(io_context->mapping.buffer),
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_invalidate_range(
        &io_context->mapping, 0, io_context->mapping.contents.data_length);
    if (!iree_status_is_ok(status)) {
      iree_hal_buffer_unmap_range(&io_context->mapping);
      return status;
    }
  }

  // Initialize the proactor write operation.
  iree_async_operation_zero(&io_context->write_op.base,
                            sizeof(io_context->write_op));
  iree_async_operation_initialize(
      &io_context->write_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_task_queue_io_write_completion,
      io_context);
  io_context->write_op.file = operation->write.async_file;
  io_context->write_op.offset = operation->write.file_offset;
  io_context->write_op.buffer =
      iree_async_span_from_ptr(io_context->mapping.contents.data,
                               (iree_host_size_t)operation->write.length);

  // Submit to the proactor and return immediately.
  iree_status_t submit_status = iree_async_proactor_submit_one(
      queue->proactor, &io_context->write_op.base);
  if (!iree_status_is_ok(submit_status)) {
    iree_hal_buffer_unmap_range(&io_context->mapping);
  }
  return submit_status;
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
    case IREE_HAL_TASK_QUEUE_OP_READ:
      status = iree_hal_task_queue_drain_read(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_WRITE:
      status = iree_hal_task_queue_drain_write(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_FILL:
      status = iree_hal_task_queue_drain_fill(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_COPY:
      status = iree_hal_task_queue_drain_copy(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_UPDATE:
      status = iree_hal_task_queue_drain_update(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_DISPATCH:
      status = iree_hal_task_queue_drain_dispatch(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
  }

  out_result->did_work = true;
  out_result->completed = false;
  return status;
}

//===----------------------------------------------------------------------===//
// Submit paths
//===----------------------------------------------------------------------===//

// Phase 1 of submit: allocates the operation and begins scope tracking.
// The caller fills type-specific union fields and retains resources on the
// returned operation, then calls submit_op_finish to register waits and
// enqueue. On failure between begin and finish, the caller must call
// op_destroy on the operation.
static iree_status_t iree_hal_task_queue_submit_op_begin(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_type_t type,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_hal_task_queue_op_t** out_operation) {
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_op_allocate(
      queue, type, signal_semaphores, out_operation));
  iree_task_scope_begin(&queue->scope);
  return iree_ok_status();
}

// Phase 2 of submit: retains resources, registers semaphore waits, and
// enqueues the operation. On failure, destroys the operation.
static iree_status_t iree_hal_task_queue_submit_op_finish(
    iree_hal_task_queue_op_t* operation,
    iree_hal_semaphore_list_t wait_semaphores, iree_host_size_t resource_count,
    iree_hal_resource_t* const* resources) {
  iree_status_t status = iree_ok_status();
  if (resource_count > 0) {
    status = iree_hal_resource_set_insert(operation->resource_set,
                                          resource_count, resources);
  }
  if (iree_status_is_ok(status) && wait_semaphores.count > 0) {
    status = iree_hal_task_queue_try_satisfy_waits(&wait_semaphores);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_task_queue_enqueue_waits(operation, wait_semaphores);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
  }
  return status;
}

// Convenience wrapper: allocates, retains resources, and enqueues in one call.
// For operations with no type-specific union fields (barriers, host calls
// with simple parameters).
static iree_status_t iree_hal_task_queue_submit_op(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_type_t type,
    iree_hal_semaphore_list_t wait_semaphores,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_host_size_t resource_count, iree_hal_resource_t* const* resources) {
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, type, signal_semaphores, &operation));
  return iree_hal_task_queue_submit_op_finish(operation, wait_semaphores,
                                              resource_count, resources);
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
  // These are operations that were queued but never drained (e.g.,
  // submitted after the queue process went idle but before shutdown was
  // signaled).
  iree_hal_task_queue_op_t* remaining = NULL;
  while ((remaining = iree_hal_task_queue_op_slist_pop(&queue->ready_list)) !=
         NULL) {
    iree_hal_task_queue_op_destroy(
        remaining,
        iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down"));
  }
  iree_hal_task_queue_op_slist_deinitialize(&queue->ready_list);

  // Deinitialize the compute lists. Items are stack-allocated in the
  // queue struct so only the slist state needs cleanup.
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
    operation->commands.binding_mappings = NULL;

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

iree_status_t iree_hal_task_queue_submit_read(
    iree_hal_task_queue_t* queue, iree_hal_file_t* source_file,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_READ, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  // Store the async file handle (borrowed — the HAL file owns it).
  operation->read.async_file = iree_hal_file_async_handle(source_file);
  operation->read.file_offset = source_offset;
  operation->read.buffer = target_buffer;
  operation->read.buffer_offset = target_offset;
  operation->read.length = length;

  // Retain the HAL file and buffer through the operation lifetime.
  // The HAL file keeps the async_file alive as a borrowed pointer.
  iree_hal_resource_t* resources[2] = {
      (iree_hal_resource_t*)source_file,
      (iree_hal_resource_t*)target_buffer,
  };
  status = iree_hal_resource_set_insert(operation->resource_set, 2, resources);

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

iree_status_t iree_hal_task_queue_submit_write(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_file_t* target_file,
    uint64_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_WRITE, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  // Store the async file handle (borrowed — the HAL file owns it).
  operation->write.async_file = iree_hal_file_async_handle(target_file);
  operation->write.file_offset = target_offset;
  operation->write.buffer = source_buffer;
  operation->write.buffer_offset = source_offset;
  operation->write.length = length;

  // Retain the HAL file and buffer through the operation lifetime.
  iree_hal_resource_t* resources[2] = {
      (iree_hal_resource_t*)target_file,
      (iree_hal_resource_t*)source_buffer,
  };
  status = iree_hal_resource_set_insert(operation->resource_set, 2, resources);

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

//===----------------------------------------------------------------------===//
// Native queue fill/copy/update/dispatch submit
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_task_queue_submit_fill(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, IREE_HAL_TASK_QUEUE_OP_FILL, &signal_semaphores, &operation));
  operation->fill.target_buffer = target_buffer;
  operation->fill.target_offset = target_offset;
  operation->fill.length = length;
  operation->fill.pattern_length = (uint8_t)pattern_length;
  memcpy(operation->fill.pattern, pattern, pattern_length);
  return iree_hal_task_queue_submit_op_finish(
      operation, wait_semaphores, 1,
      (iree_hal_resource_t* const*)&target_buffer);
}

iree_status_t iree_hal_task_queue_submit_copy(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, IREE_HAL_TASK_QUEUE_OP_COPY, &signal_semaphores, &operation));
  operation->copy.source_buffer = source_buffer;
  operation->copy.source_offset = source_offset;
  operation->copy.target_buffer = target_buffer;
  operation->copy.target_offset = target_offset;
  operation->copy.length = length;
  iree_hal_resource_t* copy_resources[2] = {
      (iree_hal_resource_t*)source_buffer,
      (iree_hal_resource_t*)target_buffer,
  };
  return iree_hal_task_queue_submit_op_finish(operation, wait_semaphores, 2,
                                              copy_resources);
}

iree_status_t iree_hal_task_queue_submit_update(
    iree_hal_task_queue_t* queue, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, IREE_HAL_TASK_QUEUE_OP_UPDATE, &signal_semaphores, &operation));

  // Arena-allocate source data copy (caller's buffer may not outlive submit).
  void* source_data_copy = NULL;
  iree_status_t status = iree_arena_allocate(
      &operation->arena, (iree_host_size_t)length, &source_data_copy);
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
    return status;
  }
  memcpy(source_data_copy, (const uint8_t*)source_buffer + source_offset,
         (size_t)length);
  operation->update.target_buffer = target_buffer;
  operation->update.target_offset = target_offset;
  operation->update.length = length;
  operation->update.source_data = source_data_copy;

  return iree_hal_task_queue_submit_op_finish(
      operation, wait_semaphores, 1,
      (iree_hal_resource_t* const*)&target_buffer);
}

iree_status_t iree_hal_task_queue_submit_dispatch(
    iree_hal_task_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_t* bindings, iree_host_size_t binding_count,
    iree_hal_dispatch_flags_t flags, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, IREE_HAL_TASK_QUEUE_OP_DISPATCH, &signal_semaphores, &operation));

  operation->dispatch.executable = executable;
  operation->dispatch.export_ordinal = export_ordinal;
  operation->dispatch.config = config;
  operation->dispatch.binding_count = binding_count;
  operation->dispatch.flags = flags;

  // Arena-allocate copies of constants and bindings.
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status) && constants.data_length > 0) {
    uint32_t* constants_copy = NULL;
    status = iree_arena_allocate(&operation->arena, constants.data_length,
                                 (void**)&constants_copy);
    if (iree_status_is_ok(status)) {
      memcpy(constants_copy, constants.data, constants.data_length);
      operation->dispatch.constants = constants_copy;
      operation->dispatch.constant_count =
          (uint16_t)(constants.data_length / sizeof(uint32_t));
    }
  }
  if (iree_status_is_ok(status) && binding_count > 0) {
    iree_hal_buffer_ref_t* bindings_copy = NULL;
    status = iree_arena_allocate(&operation->arena,
                                 binding_count * sizeof(iree_hal_buffer_ref_t),
                                 (void**)&bindings_copy);
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, bindings,
             binding_count * sizeof(iree_hal_buffer_ref_t));
      operation->dispatch.bindings = bindings_copy;
    }
  }

  // Retain executable and all bound buffers.
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(
        operation->resource_set, 1, (iree_hal_resource_t* const*)&executable);
  }
  if (iree_status_is_ok(status) && binding_count > 0) {
    status = iree_hal_resource_set_insert_strided(
        operation->resource_set, binding_count, bindings,
        offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t));
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
    return status;
  }

  return iree_hal_task_queue_submit_op_finish(operation, wait_semaphores, 0,
                                              NULL);
}

//===----------------------------------------------------------------------===//
// Dispatch drain handler
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_queue_drain_dispatch(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  const bool allow_inline = iree_any_bit_set(
      operation->dispatch.flags, IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION);
  const iree_host_size_t binding_count = operation->dispatch.binding_count;

  // For inline execution, track mappings on the stack so we can unmap after.
  // For non-inline, persistent mappings are used (pointer must survive across
  // threads until the compute process finishes).
  iree_hal_buffer_mapping_t* mappings = NULL;
  iree_status_t status = iree_ok_status();
  if (allow_inline && binding_count > 0) {
    mappings = (iree_hal_buffer_mapping_t*)iree_alloca(binding_count *
                                                       sizeof(*mappings));
    memset(mappings, 0, binding_count * sizeof(*mappings));
  }

  // Map all binding buffers. Host pointers and lengths are stored directly
  // in the fixups below (no span indirection needed).
  void** host_ptrs = NULL;
  size_t* host_lengths = NULL;
  if (binding_count > 0) {
    if (allow_inline) {
      host_ptrs = (void**)iree_alloca(binding_count * sizeof(*host_ptrs));
      host_lengths =
          (size_t*)iree_alloca(binding_count * sizeof(*host_lengths));
    } else {
      status = iree_arena_allocate(&operation->arena,
                                   binding_count * sizeof(*host_ptrs),
                                   (void**)&host_ptrs);
      if (iree_status_is_ok(status)) {
        status = iree_arena_allocate(&operation->arena,
                                     binding_count * sizeof(*host_lengths),
                                     (void**)&host_lengths);
      }
    }
  }
  iree_hal_mapping_mode_t mapping_mode = allow_inline
                                             ? IREE_HAL_MAPPING_MODE_SCOPED
                                             : IREE_HAL_MAPPING_MODE_PERSISTENT;
  for (iree_host_size_t i = 0; i < binding_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_buffer_ref_t* binding = &operation->dispatch.bindings[i];
    iree_hal_buffer_mapping_t mapping = {{0}};
    status = iree_hal_buffer_map_range(
        binding->buffer, mapping_mode, IREE_HAL_MEMORY_ACCESS_ANY,
        binding->offset, binding->length, &mapping);
    if (iree_status_is_ok(status)) {
      host_ptrs[i] = mapping.contents.data;
      host_lengths[i] = mapping.contents.data_length;
      if (mappings) mappings[i] = mapping;
    }
  }

  // Build a single-dispatch recording.
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(queue->large_block_pool, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_begin(&builder);
  }

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  if (iree_status_is_ok(status)) {
    iree_const_byte_span_t dispatch_constants = {
        .data = (const uint8_t*)operation->dispatch.constants,
        .data_length = operation->dispatch.constant_count * sizeof(uint32_t),
    };
    status = iree_hal_cmd_build_dispatch(
        &builder, operation->dispatch.executable,
        operation->dispatch.export_ordinal, operation->dispatch.config,
        dispatch_constants, binding_count, operation->dispatch.flags, &fixups,
        &token);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      fixups[i].host_ptr = host_ptrs[i];
      fixups[i].offset = 0;
      fixups[i].length = host_lengths[i];
      fixups[i].slot = 0;
      fixups[i].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
    }
  }

  iree_hal_cmd_block_recording_t recording;
  memset(&recording, 0, sizeof(recording));
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_end(&builder, &recording);
  }
  iree_hal_cmd_block_builder_deinitialize(&builder);

  if (iree_status_is_ok(status)) {
    if (allow_inline) {
      // execute_recording_inline executes the processor then signals
      // semaphores via op_complete. We unmap AFTER because the processor
      // reads through the host pointers during execution. For local_task
      // (cache-coherent CPU), unmap is a no-op — no flush needed.
      iree_hal_task_queue_execute_recording_inline(queue, operation,
                                                   &recording);
      for (iree_host_size_t i = 0; i < binding_count; ++i) {
        iree_hal_buffer_unmap_range(&mappings[i]);
      }
      return status;
    }
    // drain_recording takes ownership on success (via owned_recording).
    // Non-inline uses persistent mappings — no unmap needed (the buffer
    // retain in the resource_set keeps the pointer valid).
    status = iree_hal_task_queue_drain_recording(
        queue, operation, &recording, &recording,
        /*binding_table=*/NULL, /*binding_table_length=*/0);
  }

  // On any failure path, release the recording to prevent leaking block pool
  // blocks. Safe on zero-initialized recordings (first_block == NULL → no-op).
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_block_recording_release(&recording);
    // Unmap any scoped bindings that were mapped before the error.
    if (mappings) {
      for (iree_host_size_t i = 0; i < binding_count; ++i) {
        iree_hal_buffer_unmap_range(&mappings[i]);
      }
    }
  }

  return status;
}
