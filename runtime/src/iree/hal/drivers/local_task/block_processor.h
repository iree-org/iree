// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Block processor: cooperative multi-worker execution engine for the block ISA.
//
// Multiple workers call drain() concurrently, sharing .data (scheduling
// atomics, resolved bindings) and cooperatively processing work via atomic
// tile claiming. Each call to drain() attempts to claim and execute tiles
// from the current active region, then returns control to the caller.
//
// The drain/return model enables user-mode cooperative scheduling: callers
// maintain an active list of block processor contexts and scan across them,
// calling drain() on each to claim work. When drain() returns with no work
// (tiles_executed == 0 and not completed), the caller yields or moves to
// other contexts. No hooks, no callbacks, no implicit blocking — all yield
// and scheduling policy lives in the caller.
//
// The inner loop is tight: one amortized atomic CAS per tile reservation,
// zero task system interaction between dispatches within a region. Region
// transitions are handled by the completer (elected via remaining_tiles
// countdown). The completer reinitializes .data (setting epoch-tagged
// tile_indices) and advances active_region_index. No arrival barrier is
// needed: the epoch tag on each tile_index CAS ensures stale workers from
// a previous region fail harmlessly.
//
// Execution paths:
//   - Single-worker (worker_count=1): drain() executes the entire recording
//     synchronously with no atomics, returning completed=true on the first
//     call.
//   - Multi-worker (worker_count>1): callers invoke drain() repeatedly from
//     N concurrent contexts. Each call processes one region pass and returns.

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_PROCESSOR_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_PROCESSOR_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/local_task/block_isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Execution context
//===----------------------------------------------------------------------===//

// Execution context shared across all workers processing a recording.
// Contains the .data pointer (mutable execution state) and synchronization
// state for multi-worker coordination.
//
// For single-worker execution, declare on the stack and use
// context_initialize. For multi-worker, use context_allocate (which
// allocates the context + .data together with proper alignment).
typedef struct iree_hal_cmd_block_processor_context_t {
  // The recording being executed (immutable .text).
  const iree_hal_cmd_block_recording_t* recording;

  // Binding table for indirect fixup resolution.
  const iree_hal_cmd_binding_entry_t* binding_table;
  iree_host_size_t binding_table_length;

  // Per-block mutable execution state (.data). Sized to the recording's
  // highwater marks, reused across blocks. Separately allocated (arena or
  // trailing allocation from context_allocate).
  iree_hal_cmd_block_state_t* state;
  iree_host_size_t state_size;

  // Total workers cooperating on this recording.
  uint32_t worker_count;

  // Set to 1 when RETURN is reached or an error occurs. Workers check this
  // on each drain() entry and exit immediately when set.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int32_t completed;

  // Block sequence counter. Incremented after initializing each new block's
  // .data. Workers compare their cached value against this to detect block
  // transitions and reset per-block local state.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int32_t block_sequence;

  // First error encountered by any worker. Set via CAS — only the first
  // error wins, subsequent errors are dropped. Stored as an intptr_t
  // because iree_status_t is a pointer-tagged value.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_intptr_t error_status;

  // The current block being processed. Updated by the completer at block
  // transitions before incrementing block_sequence. Stored as intptr_t
  // for TSAN visibility (the release/acquire on block_sequence provides
  // ordering).
  iree_atomic_intptr_t current_block;

  // Sizing parameters from the recording (cached for .data access).
  uint16_t max_region_dispatch_count;
  uint16_t max_total_binding_count;

  // Global epoch counter for tile_index CAS tags. Monotonically increasing
  // across region and block transitions. Only the completer writes this
  // (single-writer, not atomic). Workers read the epoch from
  // state->region_epoch instead.
  int32_t next_epoch;

  // Budget tracking for dynamic worker_budget updates at region transitions.
  // The completer stores min(next_region_tiles, worker_count) into
  // worker_budget_ptr and adds ramp-up delta to desired_wake_ptr. Workers
  // claim wake credits via relay_wake on their next pump iteration.
  iree_atomic_int32_t* worker_budget_ptr;
  iree_atomic_int32_t* desired_wake_ptr;
} iree_hal_cmd_block_processor_context_t;

// Per-worker state maintained by the caller across drain() calls. Each worker
// has its own instance — no sharing, no atomic access. Initialize with
// memset(&state, 0, sizeof(state)) before the first drain() call.
typedef struct iree_hal_cmd_block_processor_worker_state_t {
  // Block sequence counter cached by this worker. Compared against the
  // context's block_sequence to detect block transitions.
  int32_t block_sequence;
} iree_hal_cmd_block_processor_worker_state_t;

// Result of a single drain() call.
typedef struct iree_hal_cmd_block_processor_drain_result_t {
  // Number of tiles executed by this worker during this drain() call.
  // Zero means no work was available (either the region was exhausted by
  // other workers or the processor is between region transitions).
  uint32_t tiles_executed;
  // True when the entire recording has finished (all blocks processed and
  // RETURN reached) or an error has been reported. Once true, all
  // subsequent drain() calls also return completed=true.
  bool completed;
} iree_hal_cmd_block_processor_drain_result_t;

// Initializes a caller-owned context for single-worker synchronous execution.
//
// |state| is the pre-allocated .data buffer, sized via
// iree_hal_cmd_block_state_size(recording->max_region_dispatch_count,
// recording->max_total_binding_count). The caller is responsible for
// allocating this (e.g., from an arena).
//
// For multi-worker execution, use context_allocate instead.
void iree_hal_cmd_block_processor_context_initialize(
    iree_hal_cmd_block_processor_context_t* out_context,
    const iree_hal_cmd_block_recording_t* recording,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length, iree_hal_cmd_block_state_t* state,
    iree_host_size_t state_size);

// Allocates an execution context for processing a block recording.
//
// Allocates .data (block state) sized to the recording's highwater marks.
// For worker_count > 1, initializes the first block's .data and sets
// block_sequence so workers can begin immediately upon calling drain().
//
// If the recording is empty (no blocks), returns iree_ok_status() and
// sets *out_context to NULL. drain() and context_free accept NULL.
//
// The binding_table is used for indirect fixups (span == NULL). It may be
// NULL if all fixups are direct.
iree_status_t iree_hal_cmd_block_processor_context_allocate(
    const iree_hal_cmd_block_recording_t* recording,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length, uint32_t worker_count,
    iree_allocator_t allocator,
    iree_hal_cmd_block_processor_context_t** out_context);

// Attempts to claim and execute tiles from the current active region.
//
// For worker_count == 1: executes the entire recording synchronously on the
// first call and returns completed=true. No atomics, no coordination.
//
// For worker_count > 1: claims tiles from the current region via atomic
// CAS (epoch-tagged for ABA safety), executes them, and participates in
// completer election via remaining_tiles countdown. If elected as completer,
// advances the region or handles block transitions before returning.
// Returns control to the caller after one pass.
//
// The caller is responsible for yield/scheduling policy:
//   do {
//     iree_hal_cmd_block_processor_drain(context, worker_index,
//                                        &worker_state, &result);
//     if (!result.completed && result.tiles_executed == 0) {
//       iree_thread_yield();  // or: scan other contexts, park, etc.
//     }
//   } while (!result.completed);
//
// Errors are stored in the context (first error wins via CAS). The caller
// retrieves the result from context_consume_result after all workers have
// stopped calling drain().
//
// Accepts NULL context (returns completed=true immediately).
void iree_hal_cmd_block_processor_drain(
    iree_hal_cmd_block_processor_context_t* context, uint32_t worker_index,
    iree_hal_cmd_block_processor_worker_state_t* worker_state,
    iree_hal_cmd_block_processor_drain_result_t* out_result);

// Consumes and returns the first error encountered by any worker during
// execution. Returns iree_ok_status() if all workers completed
// successfully. The caller takes ownership of the returned status.
//
// Must be called exactly once after all workers have stopped calling
// drain(). Accepts NULL (returns iree_ok_status()).
iree_status_t iree_hal_cmd_block_processor_context_consume_result(
    iree_hal_cmd_block_processor_context_t* context);

// Frees the execution context. Must be called after consume_result.
// With arena allocators this is a no-op (the arena handles bulk cleanup).
// Accepts NULL.
void iree_hal_cmd_block_processor_context_free(
    iree_hal_cmd_block_processor_context_t* context,
    iree_allocator_t allocator);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_PROCESSOR_H_
