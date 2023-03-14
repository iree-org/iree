// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_TASK_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_TASK_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/event_pool.h"
#include "iree/hal/api.h"
#include "iree/task/submission.h"
#include "iree/task/task.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a semaphore that integrates with the task system to allow for
// pipelined wait and signal operations.
iree_status_t iree_hal_task_semaphore_create(
    iree_event_pool_t* event_pool, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is a task system semaphore.
bool iree_hal_task_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Reserves a new timepoint in the timeline for the given minimum payload value.
// |issue_task| will wait until the timeline semaphore is signaled to at least
// |minimum_value| before proceeding, with a possible wait task generated and
// appended to the |submission|. Allocations for any intermediates will be made
// from |arena| whose lifetime must be tied to the submission.
iree_status_t iree_hal_task_semaphore_enqueue_timepoint(
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value,
    iree_task_t* issue_task, iree_arena_allocator_t* arena,
    iree_task_submission_t* submission);

// Performs a multi-wait on one or more semaphores.
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the wait does not complete before
// |deadline_ns| elapses.
iree_status_t iree_hal_task_semaphore_multi_wait(
    iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_event_pool_t* event_pool, iree_arena_block_pool_t* block_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_TASK_SEMAPHORE_H_
