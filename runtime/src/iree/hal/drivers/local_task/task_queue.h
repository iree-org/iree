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
#include "iree/hal/drivers/local_task/task_queue_state.h"
#include "iree/task/executor.h"
#include "iree/task/scope.h"
#include "iree/task/task.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_task_queue_t {
  // Shared executor that the queue submits tasks to.
  iree_task_executor_t* executor;

  // Shared block pool for allocating submission transients (tasks/events/etc).
  iree_arena_block_pool_t* block_pool;

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
                                    iree_task_scope_flags_t scope_flags,
                                    iree_task_executor_t* executor,
                                    iree_arena_block_pool_t* block_pool,
                                    iree_hal_task_queue_t* out_queue);

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue);

void iree_hal_task_queue_trim(iree_hal_task_queue_t* queue);

iree_status_t iree_hal_task_queue_submit(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches);

iree_status_t iree_hal_task_queue_wait_idle(iree_hal_task_queue_t* queue,
                                            iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_TASK_QUEUE_H_
