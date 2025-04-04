// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"
#include "queue_state.h"
#include "iree/task/executor.h"
#include "iree/task/scope.h"
#include "iree/task/task.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A single batch of command buffers submitted to a device queue.
// All of the wait semaphores must reach or exceed the given payload values
// prior to the batch beginning execution. Each command buffer begins execution
// in the order it is present in the list, though note that the command buffers
// execute concurrently and require internal synchronization via events if there
// are any dependencies between them. Only after all command buffers have
// completed will the signal semaphores be updated to the provided payload
// values.
typedef struct iree_hal_task_submission_batch_t {
  // Semaphores to wait on prior to executing any command buffer.
  iree_hal_semaphore_list_t wait_semaphores;

  // Command buffers to execute, in order, and optional binding tables 1:1.
  iree_host_size_t command_buffer_count;
  iree_hal_command_buffer_t* const* command_buffers;
  iree_hal_buffer_binding_table_t const* binding_tables;

  // Semaphores to signal once all command buffers have completed execution.
  iree_hal_semaphore_list_t signal_semaphores;
} iree_hal_task_submission_batch_t;

typedef struct iree_hal_task_queue_t {
  // Affinity mask this queue processes.
  iree_hal_queue_affinity_t affinity;

  // Shared executor that the queue submits tasks to.
  iree_task_executor_t* executor;

  // Shared block pool for allocating submission transients (tasks/events/etc).
  iree_arena_block_pool_t* small_block_pool;
  // Shared block pool for large allocations (command buffers/etc).
  iree_arena_block_pool_t* large_block_pool;

  // Device allocator used for transient allocations/tracking.
  iree_hal_allocator_t* device_allocator;

  // Scope used for all tasks in the queue.
  // This allows for easy waits on all outstanding queue tasks as well as
  // differentiation of tasks within the executor.
  iree_task_scope_t scope;

  // State tracking used during command buffer issue.
  // The intra-queue synchronization (barriers/events) carries across command
  // buffers and this is used to rendezvous the tasks in each set.
  iree_hal_task_queue_state_t state;
} iree_hal_task_queue_t;

void iree_hal_task_queue_initialize(iree_string_view_t identifier,
                                    iree_hal_queue_affinity_t affinity,
                                    iree_task_scope_flags_t scope_flags,
                                    iree_task_executor_t* executor,
                                    iree_arena_block_pool_t* small_block_pool,
                                    iree_arena_block_pool_t* large_block_pool,
                                    iree_hal_allocator_t* device_allocator,
                                    iree_hal_task_queue_t* out_queue);

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue);

void iree_hal_task_queue_trim(iree_hal_task_queue_t* queue);

iree_status_t iree_hal_task_queue_submit_barrier(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_commands(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_task_submission_batch_t* batches);

iree_status_t iree_hal_task_queue_submit_callback(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_host_size_t resource_count, iree_hal_resource_t* const* resources,
    iree_task_call_closure_t callback);

iree_status_t iree_hal_task_queue_wait_idle(iree_hal_task_queue_t* queue,
                                            iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_H_
