// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Block-ISA command buffer: compiles HAL commands into a block recording
// during recording, then executes via the cooperative block processor at
// issue time.
//
// Replaces iree_hal_task_command_buffer_t. The key difference is that
// recording produces an immutable .text stream (block_isa.h types) rather
// than a mutable task DAG. Execution is a single dispatch to the task
// executor that fans out to N workers, each calling execute_worker on the
// block processor. No per-command task allocation at issue time.

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/block_isa.h"
#include "iree/task/executor.h"
#include "iree/task/scope.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_block_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_task_scope_t* scope,
    iree_task_executor_t* executor, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a block-ISA command buffer.
bool iree_hal_block_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns the recording from a completed (end() called) block command buffer.
// The returned pointer is valid until the command buffer is destroyed.
const iree_hal_cmd_block_recording_t* iree_hal_block_command_buffer_recording(
    iree_hal_command_buffer_t* command_buffer);

// Issues the recorded command buffer for execution. Allocates a processor
// context from |arena|, submits a dispatch task to the executor that fans
// out to worker_count workers, and chains completion to |retire_task|.
//
// |binding_table| provides resolved buffer bindings for indirect fixups.
// May be NULL if all fixups are direct (span-based).
//
// The arena must outlive the execution (freed by the retire path).
// |pending_submission| receives the tasks to submit to the executor.
iree_status_t iree_hal_block_command_buffer_issue(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_buffer_binding_table_t* binding_table,
    iree_task_t* retire_task, iree_arena_allocator_t* arena,
    iree_task_submission_t* pending_submission);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_BUFFER_H_
