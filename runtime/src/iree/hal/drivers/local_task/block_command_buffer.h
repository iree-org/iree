// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Block-ISA command buffer: compiles HAL commands into a block recording
// during recording, then executes via the cooperative block processor at
// issue time.
//
// Recording produces an immutable .text stream (block_isa.h types). Execution
// is a single process submitted to the task executor that fans out to N
// workers cooperatively draining the block processor. No per-command task
// allocation at issue time.

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/block_isa.h"
#include "iree/hal/drivers/local_task/block_processor.h"
#include "iree/task/executor.h"
#include "iree/task/process.h"
#include "iree/task/scope.h"
#include "iree/task/tuning.h"

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

//===----------------------------------------------------------------------===//
// Issue context
//===----------------------------------------------------------------------===//

// Glue between the executor's process scheduling and the block processor.
// Allocated by the caller and initialized by
// iree_hal_block_command_buffer_issue. Contains an embedded process (for
// executor scheduling), the processor context, and per-worker drain state.
//
// The caller is responsible for the issue context's memory lifetime: the memory
// must remain valid from issue through process completion. Typically the issue
// context is arena-allocated (freed by the retire path's arena
// deinitialization) or heap-allocated (freed by the user completion callback).
typedef struct iree_hal_block_issue_context_t {
  // Executor process. Schedule via iree_task_executor_schedule_process.
  // The process drain function bridges to the block processor's drain.
  iree_task_process_t process;

  // Block processor execution context (separately allocated with cache-line
  // alignment). Freed by the internal release callback (deferred until all
  // workers have exited drain).
  iree_hal_cmd_block_processor_context_t* processor_context;

  // Allocator used for processor_context. Stored so the internal completion
  // callback can free it.
  iree_allocator_t context_allocator;

  // Number of workers expected to participate in draining.
  uint32_t worker_count;

  // User-provided completion callback, called eagerly when the process
  // completes (first worker to observe terminal state). Receives the merged
  // error status (processor + process). The issue function sets an internal
  // completion_fn on the process that consumes the processor result and then
  // chains to this callback. Set this before scheduling.
  //
  // For budget>1 processes, other workers may still be inside drain() when
  // this fires. The callback must NOT free resources accessed during drain
  // (processor_context, worker_states, the issue context itself). Use
  // user_release_fn for deferred resource cleanup.
  iree_task_process_completion_fn_t user_completion_fn;

  // User-provided release callback, called when it is safe to free resources
  // accessed during drain(). For budget>1 processes with cooperative
  // multi-worker draining, this fires after user_completion_fn — when the
  // last active drainer exits. For budget-1 processes, fires immediately
  // after user_completion_fn.
  //
  // Typical use: freeing the issue context wrapper and any other allocations
  // that workers touch during drain().
  iree_task_process_release_fn_t user_release_fn;

  // Per-worker state for the block processor drain calls. Each worker
  // maintains a block_sequence counter to detect block transitions. Zero-
  // initialized by the issue function. Indexed by worker_index modulo
  // worker_count.
  iree_hal_cmd_block_processor_worker_state_t
      worker_states[IREE_TASK_EXECUTOR_MAX_WORKER_COUNT];
} iree_hal_block_issue_context_t;

// Issues the recorded command buffer for execution by initializing
// |out_context| with a process that cooperatively drains the block processor.
//
// |binding_table| provides resolved buffer bindings for indirect fixups.
// May be NULL if all fixups are direct (span-based), which is the case for
// one-shot command buffers.
//
// |worker_count| specifies how many workers will participate in draining.
// Use iree_task_executor_worker_count() for the executor's full capacity.
//
// |host_allocator| is used for the processor context (requires aligned
// allocation). The allocator must outlive process execution.
//
// After this returns successfully:
//   - Set out_context->user_completion_fn for post-execution cleanup.
//   - Set out_context->process.dependents for downstream activation.
//   - Call iree_task_executor_schedule_process() to begin execution.
//
// The command buffer must outlive process execution (the processor
// references the recording's .text blocks and fixup tables).
iree_status_t iree_hal_block_command_buffer_issue(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_buffer_binding_table_t* binding_table, uint32_t worker_count,
    iree_allocator_t host_allocator,
    iree_hal_block_issue_context_t* out_context);

// Deinitializes the issue context, freeing the processor context. Call this
// only for abnormal cleanup (process was never scheduled or was cancelled
// before completion). For normally-completed processes, the internal
// completion callback handles cleanup automatically.
void iree_hal_block_issue_context_deinitialize(
    iree_hal_block_issue_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_COMMAND_BUFFER_H_
