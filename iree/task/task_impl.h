// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_TASK_IMPL_H_
#define IREE_TASK_TASK_IMPL_H_

#include "iree/task/list.h"
#include "iree/task/pool.h"
#include "iree/task/post_batch.h"
#include "iree/task/submission.h"
#include "iree/task/task.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// IREE_TASK_TYPE_NOP
//==============================================================================

// Retires a no-op task.
// No-op tasks don't *do* anything but must still be handled like any other
// task in the system so dependent tasks are properly scheduled.
void iree_task_nop_retire(iree_task_nop_t* task,
                          iree_task_submission_t* pending_submission);

//==============================================================================
// IREE_TASK_TYPE_CALL
//==============================================================================

// Executes and retires a user call.
// May block the caller for an indeterminate amount of time and should only be
// called from threads owned by or donated to the executor.
// Returns the status of the user call.
iree_status_t iree_task_call_execute(
    iree_task_call_t* task, iree_task_submission_t* pending_submission);

//==============================================================================
// IREE_TASK_TYPE_BARRIER
//==============================================================================

// Retires a barrier task by notifying all dependent tasks.
// May add zero or more tasks to the |pending_submission| if they are ready.
//
// Only called during coordination and expects the coordinator lock to be held.
void iree_task_barrier_retire(iree_task_barrier_t* task,
                              iree_task_submission_t* pending_submission);

//==============================================================================
// IREE_TASK_TYPE_FENCE
//==============================================================================

// Retires a fence task by updating the scope state.
//
// Only called during coordination and expects the coordinator lock to be held.
void iree_task_fence_retire(iree_task_fence_t* task,
                            iree_task_submission_t* pending_submission);

//==============================================================================
// IREE_TASK_TYPE_WAIT
//==============================================================================

// Returns true if the user-specified condition on the task is true.
//
// Only called during coordination and expects the coordinator lock to be held.
bool iree_task_wait_check_condition(iree_task_wait_t* task);

// Retires a wait when it has completed waiting (successfully or not).
//
// Only called during coordination and expects the coordinator lock to be held.
void iree_task_wait_retire(iree_task_wait_t* task,
                           iree_task_submission_t* pending_submission);

//==============================================================================
// IREE_TASK_TYPE_DISPATCH
//==============================================================================

// Schedules a dispatch by forking out to zero or more slices that will be
// executed on workers. The slices are allocated from an executor-owned pool
// and are generally not user-visible - they'll just see their dispatch begin
// execution prior to the slices and end execution after the last slice
// finishes.
//
// Only called during coordination and expects the coordinator lock to be held.
void iree_task_dispatch_issue_sliced(iree_task_dispatch_t* dispatch_task,
                                     iree_task_pool_t* slice_task_pool,
                                     iree_task_submission_t* pending_submission,
                                     iree_task_post_batch_t* post_batch);

// Schedules a dispatch by forking out to zero or more shards that will be
// executed on workers. The shards are allocated from an executor-owned pool
// and are generally not user-visible - they'll just see their dispatch begin
// execution prior to the slices and end execution after the last shard
// finishes.
//
// Only called during coordination and expects the coordinator lock to be held.
void iree_task_dispatch_issue_sharded(
    iree_task_dispatch_t* dispatch_task, iree_task_pool_t* shard_task_pool,
    iree_task_submission_t* pending_submission,
    iree_task_post_batch_t* post_batch);

// Retires a dispatch when all issued slices have completed executing.
//
// Only called during coordination and expects the coordinator lock to be held.
void iree_task_dispatch_retire(iree_task_dispatch_t* dispatch_task,
                               iree_task_submission_t* pending_submission);

//==============================================================================
// IREE_TASK_TYPE_DISPATCH_SLICE
//==============================================================================

// Allocates a dispatch slice task from the shared executor task pool.
// The slice will be released back to the pool when it has completed execution.
iree_task_dispatch_slice_t* iree_task_dispatch_slice_allocate(
    iree_task_dispatch_t* dispatch_task, const uint32_t workgroup_base[3],
    const uint32_t workgroup_range[3], const uint32_t workgroup_count[3],
    iree_task_pool_t* slice_task_pool);

// Executes and retires a dispatch slice task.
// May block the caller for an indeterminate amount of time and should only be
// called from threads owned by or donated to the executor.
//
// |local_memory| is a block of memory exclusively available to the slice
// during execution. Contents are undefined both before and after execution.
//
// Returns ok if all tiles were successfully executed and otherwise returns
// an unspecified status (probably the first non-ok status hit).
iree_status_t iree_task_dispatch_slice_execute(
    iree_task_dispatch_slice_t* task, iree_byte_span_t local_memory,
    iree_task_submission_t* pending_submission);

//==============================================================================
// IREE_TASK_TYPE_DISPATCH_SHARD
//==============================================================================

// Allocates a dispatch shard task from the shared executor task pool.
// The shard will be released back to the pool when it has completed execution.
iree_task_dispatch_shard_t* iree_task_dispatch_shard_allocate(
    iree_task_dispatch_t* dispatch_task,
    iree_task_dispatch_shard_state_t* shared_state,
    iree_task_pool_t* shard_task_pool);

// Executes and retires a dispatch shard task.
// May block the caller for an indeterminate amount of time and should only be
// called from threads owned by or donated to the executor.
//
// |local_memory| is a block of memory exclusively available to the shard
// during execution. Contents are undefined both before and after execution.
//
// Returns ok if all tiles processed in the shard successfully executed and
// otherwise returns an unspecified status (probably the first non-ok status
// hit).
iree_status_t iree_task_dispatch_shard_execute(
    iree_task_dispatch_shard_t* task, iree_byte_span_t local_memory,
    iree_task_submission_t* pending_submission);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_TASK_IMPL_H_
