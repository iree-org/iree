// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_queue.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/operation.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/js/proactor.h"
#include "iree/hal/drivers/webgpu/webgpu.h"
#include "iree/hal/drivers/webgpu/webgpu_buffer.h"
#include "iree/hal/drivers/webgpu/webgpu_command_buffer.h"
#include "iree/hal/drivers/webgpu/webgpu_executable.h"
#include "iree/hal/drivers/webgpu/webgpu_fd_file.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"
#include "iree/hal/drivers/webgpu/webgpu_semaphore.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_queue_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_queue_initialize(
    iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_handle_t queue_handle,
    const iree_hal_webgpu_builtins_t* builtins, iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
    iree_allocator_t host_allocator, iree_hal_webgpu_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(out_queue);

  out_queue->device_handle = device_handle;
  out_queue->queue_handle = queue_handle;
  out_queue->builtins = builtins;
  out_queue->proactor = proactor;
  out_queue->frontier_tracker = frontier_tracker;
  out_queue->axis = axis;
  iree_atomic_store(&out_queue->epoch, 0, iree_memory_order_relaxed);
  out_queue->host_allocator = host_allocator;

  // Initialize the shared block pool for instruction stream builders.
  iree_arena_block_pool_initialize(/*total_block_size=*/65536, host_allocator,
                                   &out_queue->block_pool);

  // Initialize the scratch builder for queue operations (dynamic_count = 0,
  // all slots are static).
  iree_status_t status = iree_hal_webgpu_builder_initialize(
      &out_queue->block_pool, /*dynamic_count=*/0, host_allocator,
      &out_queue->scratch_builder);
  if (!iree_status_is_ok(status)) {
    iree_arena_block_pool_deinitialize(&out_queue->block_pool);
  }
  return status;
}

void iree_hal_webgpu_queue_deinitialize(iree_hal_webgpu_queue_t* queue) {
  iree_hal_webgpu_builder_deinitialize(&queue->scratch_builder);
  iree_arena_block_pool_deinitialize(&queue->block_pool);
}

//===----------------------------------------------------------------------===//
// Epoch tracking
//===----------------------------------------------------------------------===//

// Atomically increments the epoch counter and returns the new value. Called at
// submit time to establish causal ordering. The frontier tracker is NOT
// advanced here — that happens at completion time (or immediately after for
// fast-path ops where submit IS completion).
static uint64_t iree_hal_webgpu_queue_reserve_epoch(
    iree_hal_webgpu_queue_t* queue) {
  return (uint64_t)iree_atomic_fetch_add(&queue->epoch, 1,
                                         iree_memory_order_acq_rel) +
         1;
}

// Advances the frontier tracker to the given epoch. Called at completion time
// (after GPU work finishes via onSubmittedWorkDone) or immediately after
// reserve_epoch for fast-path ops where submit IS completion.
static void iree_hal_webgpu_queue_advance_tracker(
    iree_hal_webgpu_queue_t* queue, uint64_t epoch) {
  if (!queue->frontier_tracker) return;
  iree_async_frontier_tracker_advance(queue->frontier_tracker, queue->axis,
                                      epoch);
}

//===----------------------------------------------------------------------===//
// Frontier construction
//===----------------------------------------------------------------------===//

// Builds a single-entry frontier on caller-provided stack storage for the
// queue's axis at the given epoch. Returns NULL if the queue has no frontier
// tracker (frontiers disabled). The returned pointer is valid for the lifetime
// of |out_frontier|.
static const iree_async_frontier_t* iree_hal_webgpu_queue_build_frontier(
    iree_hal_webgpu_queue_t* queue, uint64_t epoch,
    iree_async_single_frontier_t* out_frontier) {
  if (!queue->frontier_tracker) return NULL;
  iree_async_single_frontier_initialize(out_frontier, queue->axis, epoch);
  return iree_async_single_frontier_as_const_frontier(out_frontier);
}

//===----------------------------------------------------------------------===//
// Scratch builder execution
//===----------------------------------------------------------------------===//

// Executes the scratch builder's instruction stream via the one-shot path
// (execute_instructions). Builds the binding table from the builder's slot map
// and passes it with the builtins descriptor to the JS processor.
static iree_status_t iree_hal_webgpu_queue_execute_scratch_builder(
    iree_hal_webgpu_queue_t* queue) {
  iree_hal_webgpu_builder_t* builder = &queue->scratch_builder;

  uint32_t total_slots = iree_hal_webgpu_builder_total_slot_count(builder);
  uint32_t static_count = iree_hal_webgpu_builder_static_slot_count(builder);

  // Build the flat binding table in wire format.
  // Use stack allocation for small tables (covers the vast majority of queue
  // ops which touch 1-3 buffers).
  iree_hal_webgpu_isa_binding_table_entry_t inline_entries[8];
  iree_hal_webgpu_isa_binding_table_entry_t* entries = inline_entries;
  if (total_slots > IREE_ARRAYSIZE(inline_entries)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        queue->host_allocator, total_slots,
        sizeof(iree_hal_webgpu_isa_binding_table_entry_t), (void**)&entries));
  }

  const iree_hal_webgpu_builder_slot_entry_t* slot_entries =
      iree_hal_webgpu_builder_static_slot_entries(builder);
  for (uint32_t i = 0; i < static_count; ++i) {
    entries[slot_entries[i].slot].gpu_buffer_handle =
        slot_entries[i].gpu_buffer_handle;
    entries[slot_entries[i].slot].base_offset = 0;
  }

  iree_hal_webgpu_isa_builtins_descriptor_t builtins_descriptor;
  iree_hal_webgpu_builtins_get_descriptor(queue->builtins,
                                          &builtins_descriptor);

  uint32_t result = iree_hal_webgpu_import_execute_instructions(
      queue->device_handle, queue->queue_handle,
      (uint32_t)(uintptr_t)iree_hal_webgpu_builder_block_table(builder),
      iree_hal_webgpu_builder_block_count(builder),
      iree_hal_webgpu_builder_block_word_capacity(builder),
      iree_hal_webgpu_builder_last_block_word_count(builder),
      (uint32_t)(uintptr_t)entries, total_slots,
      (uint32_t)(uintptr_t)&builtins_descriptor);

  if (entries != inline_entries) {
    iree_allocator_free(queue->host_allocator, entries);
  }

  if (result != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "JS execute_instructions failed with code %u",
                            result);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Signal completion (proactor-driven onSubmittedWorkDone)
//===----------------------------------------------------------------------===//

// Holds an async operation and a snapshot of the signal semaphore list for
// deferred signaling after onSubmittedWorkDone completes. Allocated as a single
// block: [struct | semaphore_ptrs[] | payload_values[]].
typedef struct iree_hal_webgpu_signal_completion_t {
  iree_async_operation_t operation;
  iree_hal_semaphore_list_t signal_semaphore_list;
  iree_hal_webgpu_queue_t* queue;  // Borrowed, outlives completion.
  uint64_t epoch;
  iree_allocator_t allocator;
} iree_hal_webgpu_signal_completion_t;

// Completion callback invoked by the proactor when onSubmittedWorkDone fires.
// Builds a frontier for the queue's axis/epoch, signals (or fails) all
// semaphores with it, advances the frontier tracker, releases references,
// and frees the completion struct.
static void iree_hal_webgpu_signal_completion_fn(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_signal_completion_t* completion =
      (iree_hal_webgpu_signal_completion_t*)base_operation;

  if (iree_status_is_ok(status)) {
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(
            completion->queue, completion->epoch, &frontier_storage);
    iree_status_ignore(iree_hal_semaphore_list_signal(
        completion->signal_semaphore_list, frontier));
  } else {
    iree_hal_semaphore_list_fail(completion->signal_semaphore_list, status);
  }

  iree_hal_webgpu_queue_advance_tracker(completion->queue, completion->epoch);
  iree_hal_semaphore_list_release(completion->signal_semaphore_list);
  iree_allocator_free(completion->allocator, completion);
}

// Registers an onSubmittedWorkDone callback that signals all semaphores in
// |signal_semaphore_list| when the currently submitted GPU work completes.
// The |epoch| is the pre-reserved epoch from reserve_epoch; the completion
// callback builds a frontier from it and advances the tracker.
// If the semaphore list is empty, advances the tracker immediately and returns.
static iree_status_t iree_hal_webgpu_queue_register_signal_completion(
    iree_hal_webgpu_queue_t* queue, uint64_t epoch,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  if (signal_semaphore_list.count == 0) {
    iree_hal_webgpu_queue_advance_tracker(queue, epoch);
    return iree_ok_status();
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t semaphores_offset = 0;
  iree_host_size_t payload_values_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_webgpu_signal_completion_t), &total_size,
      IREE_STRUCT_FIELD(signal_semaphore_list.count, iree_hal_semaphore_t*,
                        &semaphores_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, uint64_t,
                        &payload_values_offset)));
  iree_hal_webgpu_signal_completion_t* completion = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator, total_size,
                                             (void**)&completion));

  completion->signal_semaphore_list = iree_hal_semaphore_list_at_offsets(
      completion, signal_semaphore_list.count, semaphores_offset,
      payload_values_offset);
  completion->queue = queue;
  completion->epoch = epoch;
  completion->allocator = queue->host_allocator;
  iree_hal_semaphore_list_clone_into(signal_semaphore_list,
                                     completion->signal_semaphore_list);

  iree_async_operation_initialize(
      &completion->operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_webgpu_signal_completion_fn,
      /*user_data=*/NULL);

  // Submit to the JS proactor's token table to get a completion token.
  uint32_t token = UINT32_MAX;
  iree_status_t status = iree_async_proactor_js_submit_external(
      queue->proactor, &completion->operation, &token);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_release(completion->signal_semaphore_list);
    iree_allocator_free(queue->host_allocator, completion);
    return status;
  }

  // Register onSubmittedWorkDone with the JS bridge. When the GPU finishes
  // the submitted work, JS writes {token, status_code} to the completion
  // ring. The proactor's drain path dispatches our callback.
  iree_hal_webgpu_import_queue_on_submitted_work_done(queue->queue_handle,
                                                      token);
  return iree_ok_status();
}

// Forward declaration — defined in the FIFO wait elision section.
static void iree_hal_webgpu_queue_mark_signals_submitted(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list);

// Finalizes and executes the scratch builder, then registers signal completion.
// Used by fast-path queue operations (fill, update, copy, dispatch) that build
// a single instruction into the scratch builder and submit it synchronously.
//
// Reserves an epoch and registers an onSubmittedWorkDone completion that
// carries the epoch's frontier. The completion callback signals semaphores
// with the frontier and advances the frontier tracker. On success, also marks
// the signal semaphores with submitted provenance for FIFO wait elision.
//
// |status| is the accumulated status from the caller's builder commands.
// If already failed, skips finalize/execute and fails the signal semaphores.
static iree_status_t iree_hal_webgpu_queue_submit_scratch_and_signal(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_status_t status) {
  uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_builder_finalize(&queue->scratch_builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_queue_execute_scratch_builder(queue);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_queue_register_signal_completion(
        queue, epoch, signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_mark_signals_submitted(queue, signal_semaphore_list);
  } else {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    iree_hal_webgpu_queue_advance_tracker(queue, epoch);
  }
  return status;
}

// Forward declarations for queue_execute helpers used in the wait callback.
static iree_status_t iree_hal_webgpu_queue_execute_recording(
    iree_hal_webgpu_queue_t* queue, iree_hal_webgpu_handle_t recording_handle,
    iree_hal_buffer_binding_table_t binding_table);
static iree_status_t iree_hal_webgpu_queue_execute_one_shot(
    iree_hal_webgpu_queue_t* queue, iree_hal_command_buffer_t* command_buffer);

//===----------------------------------------------------------------------===//
// Unified async queue operation state
//===----------------------------------------------------------------------===//
//
// All async queue operations (except host_call, which has unique DEFERRED
// semantics) share a common state structure. The wait completion callback
// dispatches on op_type to perform the operation's work, then either signals
// inline (CPU-only ops) or transfers slab ownership to an embedded signal
// completion (GPU-submit ops) for GPU completion tracking.
//
// CPU-only (barrier, alloca, dealloca):
//   wait → CPU work → signal inline → advance epoch → free slab.
//
// GPU-submit (fill, update, copy, dispatch, execute):
//   wait → GPU work → release per-op resources → init embedded signal
//   → register onSubmittedWorkDone → [slab lives on]
//   → onSubmittedWorkDone fires → signal → advance epoch → free slab.

typedef enum iree_hal_webgpu_queue_op_type_e {
  // CPU-only: wait → CPU work → signal inline.
  IREE_HAL_WEBGPU_QUEUE_OP_BARRIER,
  IREE_HAL_WEBGPU_QUEUE_OP_ALLOCA,
  IREE_HAL_WEBGPU_QUEUE_OP_DEALLOCA,
  // GPU-submit: wait → GPU work → signal on GPU completion.
  IREE_HAL_WEBGPU_QUEUE_OP_FILL,
  IREE_HAL_WEBGPU_QUEUE_OP_UPDATE,
  IREE_HAL_WEBGPU_QUEUE_OP_COPY,
  IREE_HAL_WEBGPU_QUEUE_OP_READ,
  IREE_HAL_WEBGPU_QUEUE_OP_DISPATCH,
  IREE_HAL_WEBGPU_QUEUE_OP_EXECUTE,
} iree_hal_webgpu_queue_op_type_t;

typedef struct iree_hal_webgpu_queue_state_t {
  // Must be first — the proactor casts between base operation and this.
  iree_async_semaphore_wait_operation_t wait_operation;

  // Embedded signal operation for GPU-submit ops. After the wait callback
  // submits GPU work, this is initialized as a NOP external operation with a
  // proactor token. The slab stays alive until onSubmittedWorkDone fires and
  // the signal callback runs.
  iree_async_operation_t signal_operation;

  iree_hal_webgpu_queue_t* queue;

  // Pre-incremented epoch. The atomic counter is incremented at submit time
  // for causal ordering; the frontier tracker is advanced at completion time.
  uint64_t epoch;

  iree_hal_semaphore_list_t wait_semaphore_list;
  iree_hal_semaphore_list_t signal_semaphore_list;

  iree_hal_webgpu_queue_op_type_t op_type;

  union {
    struct {
      iree_hal_buffer_t* buffer;  // Retained stub.
    } alloca_op;

    struct {
      iree_hal_buffer_t* buffer;  // Retained.
    } dealloca;

    struct {
      iree_hal_buffer_t* target_buffer;  // Retained.
      iree_device_size_t target_offset;
      iree_device_size_t length;
      uint32_t pattern;
      iree_host_size_t pattern_length;
    } fill;

    struct {
      iree_hal_buffer_t* target_buffer;  // Retained.
      iree_device_size_t target_offset;
      iree_device_size_t length;
      void* captured_data;                 // Points into acquired block.
      iree_arena_block_t* captured_block;  // For releasing back to pool.
    } update;

    struct {
      iree_hal_buffer_t* source_buffer;  // Retained.
      iree_device_size_t source_offset;
      iree_hal_buffer_t* target_buffer;  // Retained.
      iree_device_size_t target_offset;
      iree_device_size_t length;
    } copy;

    struct {
      iree_hal_buffer_t* storage;    // Retained if non-NULL (HOST_LOCAL).
      iree_hal_file_t* source_file;  // Retained if storage is NULL (FD).
      uint64_t source_offset;
      iree_hal_buffer_t* target_buffer;  // Retained.
      iree_device_size_t target_offset;
      iree_device_size_t length;
    } read;

    struct {
      iree_hal_webgpu_handle_t pipeline_handle;
      iree_hal_webgpu_handle_t bind_group_layout_handle;
      uint32_t workgroup_count[3];
      iree_hal_executable_t* executable;  // Retained.
      iree_hal_buffer_ref_t* bindings;    // Points into trailing slab.
      uint32_t binding_count;
    } dispatch;

    struct {
      iree_hal_command_buffer_t* command_buffer;  // Retained.
      iree_hal_buffer_binding_t* binding_table;   // Points into trailing slab.
      iree_host_size_t binding_count;
    } execute;
  };

  iree_allocator_t allocator;

  // Trailing arrays (via IREE_STRUCT_LAYOUT):
  //   iree_async_semaphore_t* wait_semaphores[wait_count]
  //   uint64_t wait_values[wait_count]
  //   iree_hal_semaphore_t* signal_semaphores[signal_count]
  //   uint64_t signal_values[signal_count]
  //   Per-op trailing data:
  //     DISPATCH: iree_hal_buffer_ref_t bindings[binding_count]
  //     EXECUTE: iree_hal_buffer_binding_t binding_table[binding_count]
} iree_hal_webgpu_queue_state_t;

// Releases per-op retained resources from the state's union. Called on both
// success (after work is done) and failure (wait error, submit error) paths.
// Each op type retains resources in its submit function and releases them here.
static void iree_hal_webgpu_queue_op_release_resources(
    iree_hal_webgpu_queue_state_t* state) {
  switch (state->op_type) {
    case IREE_HAL_WEBGPU_QUEUE_OP_BARRIER:
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_ALLOCA:
      iree_hal_buffer_release(state->alloca_op.buffer);
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_DEALLOCA:
      iree_hal_buffer_release(state->dealloca.buffer);
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_FILL:
      iree_hal_buffer_release(state->fill.target_buffer);
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_UPDATE:
      iree_hal_buffer_release(state->update.target_buffer);
      iree_arena_block_pool_release(&state->queue->block_pool,
                                    state->update.captured_block,
                                    state->update.captured_block);
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_COPY:
      iree_hal_buffer_release(state->copy.source_buffer);
      iree_hal_buffer_release(state->copy.target_buffer);
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_READ:
      if (state->read.storage) {
        iree_hal_buffer_release(state->read.storage);
      } else {
        iree_hal_file_release(state->read.source_file);
      }
      iree_hal_buffer_release(state->read.target_buffer);
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_DISPATCH:
      iree_hal_executable_release(state->dispatch.executable);
      for (uint32_t i = 0; i < state->dispatch.binding_count; ++i) {
        iree_hal_buffer_release(state->dispatch.bindings[i].buffer);
      }
      break;
    case IREE_HAL_WEBGPU_QUEUE_OP_EXECUTE:
      iree_hal_command_buffer_release(state->execute.command_buffer);
      for (iree_host_size_t i = 0; i < state->execute.binding_count; ++i) {
        iree_hal_buffer_release(state->execute.binding_table[i].buffer);
      }
      break;
    default:
      break;
  }
}

// Cleans up a queue state slab after proactor submission fails. Fails signal
// semaphores (so downstream waiters see the error rather than hanging),
// releases per-op resources, both semaphore lists, and frees the slab.
static void iree_hal_webgpu_queue_state_submit_failed(
    iree_hal_webgpu_queue_state_t* state, iree_status_t submit_status) {
  iree_hal_semaphore_list_fail(state->signal_semaphore_list,
                               iree_status_clone(submit_status));
  iree_hal_webgpu_queue_op_release_resources(state);
  iree_hal_semaphore_list_release(state->wait_semaphore_list);
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_allocator_free(state->allocator, state);
}

// Signal completion callback for GPU-submit ops. Fires when
// onSubmittedWorkDone delivers after GPU work completes. Recovers the
// queue_state from the embedded signal_operation via offsetof.
static void iree_hal_webgpu_queue_op_signal_completion(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_queue_state_t* state =
      (iree_hal_webgpu_queue_state_t*)((uint8_t*)base_operation -
                                       offsetof(iree_hal_webgpu_queue_state_t,
                                                signal_operation));

  if (iree_status_is_ok(status)) {
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(state->queue, state->epoch,
                                             &frontier_storage);
    iree_status_ignore(
        iree_hal_semaphore_list_signal(state->signal_semaphore_list, frontier));
  } else {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
  }

  iree_hal_webgpu_queue_advance_tracker(state->queue, state->epoch);
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_allocator_free(state->allocator, state);
}

// Registers the embedded signal_operation for onSubmittedWorkDone completion
// tracking. Called from GPU-submit cases in the wait callback after GPU work
// has been submitted. On success: releases per-op resources (GPU holds its
// own references) and returns true — the caller MUST return immediately
// (slab ownership transfers to the signal callback). On failure: fails the
// signal semaphores and returns false — the caller should break to shared
// cleanup.
static bool iree_hal_webgpu_queue_op_register_embedded_signal(
    iree_hal_webgpu_queue_state_t* state) {
  iree_async_operation_initialize(
      &state->signal_operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_webgpu_queue_op_signal_completion, /*user_data=*/NULL);
  uint32_t token = UINT32_MAX;
  iree_status_t status = iree_async_proactor_js_submit_external(
      state->queue->proactor, &state->signal_operation, &token);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
    return false;
  }
  iree_hal_webgpu_import_queue_on_submitted_work_done(
      state->queue->queue_handle, token);
  // Per-op resources released now — GPU has its own references to buffers.
  iree_hal_webgpu_queue_op_release_resources(state);
  return true;
}

// Finalizes the scratch builder, executes it, and registers the embedded
// signal for GPU completion tracking. Called from GPU-submit cases in the wait
// callback after per-op builder commands have been recorded. Returns true on
// success (caller must return immediately — slab ownership transfers to the
// signal callback); false on failure (caller should break to shared cleanup).
//
// |work_status| is the accumulated status from the per-op builder calls.
// If already failed, skips finalize/execute and fails the signal semaphores.
static bool iree_hal_webgpu_queue_op_finalize_and_submit(
    iree_hal_webgpu_queue_state_t* state, iree_status_t work_status) {
  if (iree_status_is_ok(work_status)) {
    work_status =
        iree_hal_webgpu_builder_finalize(&state->queue->scratch_builder);
  }
  if (iree_status_is_ok(work_status)) {
    work_status = iree_hal_webgpu_queue_execute_scratch_builder(state->queue);
  }
  if (!iree_status_is_ok(work_status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, work_status);
    return false;
  }
  return iree_hal_webgpu_queue_op_register_embedded_signal(state);
}

// Wait completion callback for all unified async queue operations. Dispatches
// on op_type to perform the operation's work after input semaphores are
// satisfied. CPU-only ops signal inline and fall through to shared cleanup.
// GPU-submit ops submit GPU work and return, transferring slab ownership to
// the signal completion callback.
static void iree_hal_webgpu_queue_op_wait_completion(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t wait_status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_queue_state_t* state =
      (iree_hal_webgpu_queue_state_t*)base_operation;

  // Wait semaphores consumed regardless of outcome.
  iree_hal_semaphore_list_release(state->wait_semaphore_list);

  if (!iree_status_is_ok(wait_status)) {
    // Propagate wait failure to signal semaphores and clean up everything.
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, wait_status);
    iree_hal_webgpu_queue_op_release_resources(state);
    iree_hal_semaphore_list_release(state->signal_semaphore_list);
    iree_allocator_free(state->allocator, state);
    return;
  }

  // Build frontier once for all cases that signal inline. GPU-submit success
  // cases don't use it here (they build their own in the signal callback).
  iree_async_single_frontier_t frontier_storage;
  const iree_async_frontier_t* frontier = iree_hal_webgpu_queue_build_frontier(
      state->queue, state->epoch, &frontier_storage);

  switch (state->op_type) {
    case IREE_HAL_WEBGPU_QUEUE_OP_BARRIER:
      // CPU-only: no work — signal passes straight through.
      iree_status_ignore(iree_hal_semaphore_list_signal(
          state->signal_semaphore_list, frontier));
      break;

    case IREE_HAL_WEBGPU_QUEUE_OP_ALLOCA: {
      // CPU-only: create the GPU buffer on the stub, then signal.
      iree_status_t bind_status = iree_hal_webgpu_buffer_bind(
          state->alloca_op.buffer, state->queue->device_handle);
      if (iree_status_is_ok(bind_status)) {
        iree_status_ignore(iree_hal_semaphore_list_signal(
            state->signal_semaphore_list, frontier));
      } else {
        iree_hal_semaphore_list_fail(state->signal_semaphore_list, bind_status);
      }
      break;
    }

    case IREE_HAL_WEBGPU_QUEUE_OP_DEALLOCA:
      // CPU-only: detach GPU buffer from the wrapper, then signal.
      iree_hal_webgpu_buffer_unbind(state->dealloca.buffer);
      iree_status_ignore(iree_hal_semaphore_list_signal(
          state->signal_semaphore_list, frontier));
      break;

    case IREE_HAL_WEBGPU_QUEUE_OP_FILL: {
      // GPU-submit: scratch build fill → GPU execute → signal on completion.
      iree_hal_webgpu_queue_t* queue = state->queue;
      iree_status_t work_status =
          iree_hal_webgpu_builder_reset(&queue->scratch_builder);
      if (iree_status_is_ok(work_status)) {
        iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
            state->fill.target_buffer, state->fill.target_offset,
            state->fill.length);
        work_status = iree_hal_webgpu_builder_fill_buffer(
            &queue->scratch_builder, target_ref, &state->fill.pattern,
            state->fill.pattern_length);
      }
      if (iree_hal_webgpu_queue_op_finalize_and_submit(state, work_status))
        return;
      break;
    }

    case IREE_HAL_WEBGPU_QUEUE_OP_UPDATE: {
      // GPU-submit: scratch build update → GPU execute → signal on completion.
      // Source data was captured into a block pool block at submit time.
      iree_hal_webgpu_queue_t* queue = state->queue;
      iree_status_t work_status =
          iree_hal_webgpu_builder_reset(&queue->scratch_builder);
      if (iree_status_is_ok(work_status)) {
        iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
            state->update.target_buffer, state->update.target_offset,
            state->update.length);
        work_status = iree_hal_webgpu_builder_update_buffer(
            &queue->scratch_builder, state->update.captured_data,
            /*source_offset=*/0, target_ref);
      }
      if (iree_hal_webgpu_queue_op_finalize_and_submit(state, work_status))
        return;
      break;
    }

    case IREE_HAL_WEBGPU_QUEUE_OP_COPY: {
      // GPU-submit: scratch build copy → GPU execute → signal on completion.
      iree_hal_webgpu_queue_t* queue = state->queue;
      iree_status_t work_status =
          iree_hal_webgpu_builder_reset(&queue->scratch_builder);
      if (iree_status_is_ok(work_status)) {
        iree_hal_buffer_ref_t source_ref = iree_hal_make_buffer_ref(
            state->copy.source_buffer, state->copy.source_offset,
            state->copy.length);
        iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
            state->copy.target_buffer, state->copy.target_offset,
            state->copy.length);
        work_status = iree_hal_webgpu_builder_copy_buffer(
            &queue->scratch_builder, source_ref, target_ref);
      }
      if (iree_hal_webgpu_queue_op_finalize_and_submit(state, work_status))
        return;
      break;
    }

    case IREE_HAL_WEBGPU_QUEUE_OP_READ: {
      // GPU-submit: file → GPU via bridge import → signal on completion.
      // No scratch builder needed — calls queue.writeBuffer() directly.
      iree_hal_webgpu_queue_t* queue = state->queue;
      iree_hal_webgpu_handle_t gpu_handle = iree_hal_webgpu_buffer_handle(
          iree_hal_buffer_allocated_buffer(state->read.target_buffer));
      uint64_t gpu_offset =
          iree_hal_buffer_byte_offset(state->read.target_buffer) +
          state->read.target_offset;
      iree_status_t work_status = iree_ok_status();
      if (state->read.storage) {
        // HOST_LOCAL: map the storage buffer and upload from host pointer.
        iree_hal_buffer_mapping_t mapping = {{0}};
        work_status = iree_hal_buffer_map_range(
            state->read.storage, IREE_HAL_MAPPING_MODE_SCOPED,
            IREE_HAL_MEMORY_ACCESS_READ, state->read.source_offset,
            state->read.length, &mapping);
        if (iree_status_is_ok(work_status)) {
          iree_hal_webgpu_import_queue_write_buffer(
              queue->queue_handle, gpu_handle, gpu_offset,
              (uint32_t)(uintptr_t)mapping.contents.data, state->read.length);
          iree_hal_buffer_unmap_range(&mapping);
        }
      } else {
        // FD: use zero-copy bridge import.
        int fd = iree_hal_webgpu_fd_file_fd(state->read.source_file);
        iree_hal_webgpu_import_queue_write_buffer_from_file(
            queue->queue_handle, gpu_handle, gpu_offset, (uint32_t)fd,
            state->read.source_offset, state->read.length);
      }
      if (!iree_status_is_ok(work_status)) {
        iree_hal_semaphore_list_fail(state->signal_semaphore_list, work_status);
        break;
      }
      if (!iree_hal_webgpu_queue_op_register_embedded_signal(state)) break;
      return;
    }

    case IREE_HAL_WEBGPU_QUEUE_OP_DISPATCH: {
      // GPU-submit: scratch build dispatch → GPU execute → signal on
      // completion.
      iree_hal_webgpu_queue_t* queue = state->queue;
      iree_hal_buffer_ref_list_t binding_list = {
          .values = state->dispatch.bindings,
          .count = state->dispatch.binding_count,
      };
      iree_status_t work_status =
          iree_hal_webgpu_builder_reset(&queue->scratch_builder);
      if (iree_status_is_ok(work_status)) {
        work_status = iree_hal_webgpu_builder_dispatch(
            &queue->scratch_builder, state->dispatch.pipeline_handle,
            state->dispatch.bind_group_layout_handle,
            state->dispatch.workgroup_count, binding_list);
      }
      if (iree_hal_webgpu_queue_op_finalize_and_submit(state, work_status))
        return;
      break;
    }

    case IREE_HAL_WEBGPU_QUEUE_OP_EXECUTE: {
      // GPU-submit: command buffer submit → signal on GPU completion.
      iree_hal_webgpu_queue_t* queue = state->queue;
      iree_hal_webgpu_handle_t recording_handle =
          iree_hal_webgpu_command_buffer_recording_handle(
              state->execute.command_buffer);
      iree_status_t work_status;
      if (recording_handle) {
        iree_hal_buffer_binding_table_t binding_table = {
            .count = state->execute.binding_count,
            .bindings = state->execute.binding_table,
        };
        work_status = iree_hal_webgpu_queue_execute_recording(
            queue, recording_handle, binding_table);
      } else {
        work_status = iree_hal_webgpu_queue_execute_one_shot(
            queue, state->execute.command_buffer);
      }
      if (!iree_status_is_ok(work_status)) {
        iree_hal_semaphore_list_fail(state->signal_semaphore_list, work_status);
        break;
      }
      if (!iree_hal_webgpu_queue_op_register_embedded_signal(state)) break;
      return;
    }

      // GPU-submit cases that succeed return from their case above,
      // transferring slab ownership to the signal completion callback. Error
      // cases break to the shared cleanup below.

    default:
      // Ops that haven't been converted to the unified async pattern never
      // create queue_state_t slabs. Reaching here is a programming error.
      iree_hal_semaphore_list_fail(
          state->signal_semaphore_list,
          iree_make_status(IREE_STATUS_INTERNAL,
                           "unexpected op type %d in queue wait callback",
                           (int)state->op_type));
      break;
  }

  // Shared cleanup for CPU-only completion and GPU-submit error recovery.
  // GPU-submit success cases return from their switch case before reaching
  // here.
  iree_hal_webgpu_queue_advance_tracker(state->queue, state->epoch);
  iree_hal_webgpu_queue_op_release_resources(state);
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_allocator_free(state->allocator, state);
}

// Initializes the common fields of a pre-allocated queue state slab.
// Sets up the wait operation with semaphore arrays pointing into the trailing
// slab (aliased via cast — valid because iree_async_semaphore_t is at offset 0
// in iree_hal_webgpu_semaphore_t), clones both semaphore lists, and initializes
// the operation base with the unified wait completion callback.
static void iree_hal_webgpu_queue_state_initialize(
    iree_hal_webgpu_queue_state_t* state,
    iree_hal_webgpu_queue_op_type_t op_type, iree_hal_webgpu_queue_t* queue,
    uint64_t epoch, const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t wait_semaphores_offset,
    iree_host_size_t wait_values_offset,
    iree_host_size_t signal_semaphores_offset,
    iree_host_size_t signal_values_offset, iree_allocator_t allocator) {
  state->op_type = op_type;
  state->queue = queue;
  state->epoch = epoch;
  state->allocator = allocator;

  // Clone the wait semaphore list into the trailing slab. The wait operation's
  // semaphore pointer array aliases the same memory.
  state->wait_semaphore_list = iree_hal_semaphore_list_at_offsets(
      state, wait_semaphore_list.count, wait_semaphores_offset,
      wait_values_offset);
  iree_hal_semaphore_list_clone_into(wait_semaphore_list,
                                     state->wait_semaphore_list);
  state->wait_operation.semaphores =
      (iree_async_semaphore_t**)state->wait_semaphore_list.semaphores;
  state->wait_operation.values = state->wait_semaphore_list.payload_values;
  state->wait_operation.count = wait_semaphore_list.count;
  state->wait_operation.mode = IREE_ASYNC_WAIT_MODE_ALL;
  state->wait_operation.satisfied_index = 0;

  // Clone the signal semaphore list into the trailing slab.
  state->signal_semaphore_list = iree_hal_semaphore_list_at_offsets(
      state, signal_semaphore_list.count, signal_semaphores_offset,
      signal_values_offset);
  iree_hal_semaphore_list_clone_into(signal_semaphore_list,
                                     state->signal_semaphore_list);

  // Initialize the wait operation base with the unified completion callback.
  iree_async_operation_initialize(
      &state->wait_operation.base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_webgpu_queue_op_wait_completion,
      /*user_data=*/NULL);
}

// Allocates and initializes a queue state slab for an async queue operation.
// Pre-increments the queue epoch, computes the slab layout (base struct +
// 4 trailing semaphore arrays + optional per-op trailing data), allocates the
// slab, and initializes common fields. On failure, fails the signal semaphores
// and returns the error.
//
// |trailing_count| and |trailing_element_size| specify an optional trailing
// array for per-op data (dispatch bindings, execute binding table). Pass 0
// for both when no extra trailing data is needed. If non-zero,
// |out_trailing_offset| receives the byte offset from the slab base.
static iree_status_t iree_hal_webgpu_queue_state_allocate(
    iree_hal_webgpu_queue_t* queue, iree_hal_webgpu_queue_op_type_t op_type,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t trailing_count, iree_host_size_t trailing_element_size,
    iree_host_size_t* out_trailing_offset,
    iree_hal_webgpu_queue_state_t** out_state) {
  *out_state = NULL;

  uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);

  iree_host_size_t total_size = 0;
  iree_host_size_t wait_semaphores_offset = 0;
  iree_host_size_t wait_values_offset = 0;
  iree_host_size_t signal_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_webgpu_queue_state_t), &total_size,
      IREE_STRUCT_FIELD(wait_semaphore_list.count, iree_async_semaphore_t*,
                        &wait_semaphores_offset),
      IREE_STRUCT_FIELD(wait_semaphore_list.count, uint64_t,
                        &wait_values_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, iree_hal_semaphore_t*,
                        &signal_semaphores_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, uint64_t,
                        &signal_values_offset));

  // Append per-op trailing data after the semaphore arrays. STRUCT_LAYOUT
  // produces a max_align_t-aligned total_size, which satisfies alignment for
  // all per-op element types (they contain pointers and device_size_t fields,
  // both <= max_align_t).
  if (iree_status_is_ok(status) && trailing_count > 0) {
    if (out_trailing_offset) *out_trailing_offset = total_size;
    iree_host_size_t trailing_bytes = 0;
    if (!iree_host_size_checked_mul(trailing_count, trailing_element_size,
                                    &trailing_bytes) ||
        !iree_host_size_checked_add(total_size, trailing_bytes, &total_size)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "trailing allocation size overflow");
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }

  iree_hal_webgpu_queue_state_t* state = NULL;
  status =
      iree_allocator_malloc(queue->host_allocator, total_size, (void**)&state);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }

  iree_hal_webgpu_queue_state_initialize(
      state, op_type, queue, epoch, wait_semaphore_list, signal_semaphore_list,
      wait_semaphores_offset, wait_values_offset, signal_semaphores_offset,
      signal_values_offset, queue->host_allocator);

  *out_state = state;
  return iree_ok_status();
}

// Submits an initialized queue state to the proactor. On failure, cleans up
// the state (fails signals, releases resources and semaphore lists, frees
// slab).
static iree_status_t iree_hal_webgpu_queue_state_submit(
    iree_hal_webgpu_queue_state_t* state) {
  iree_status_t status = iree_async_proactor_submit_one(
      state->queue->proactor, &state->wait_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_state_submit_failed(state, status);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// FIFO wait elision
//===----------------------------------------------------------------------===//

// Returns true if all wait semaphores can be elided because this queue has
// already submitted (but possibly not completed) signals that will satisfy
// every wait. Uses the submitted signal provenance fields on each semaphore
// to check if GPU FIFO ordering guarantees the wait will be satisfied.
static bool iree_hal_webgpu_queue_can_elide_waits(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list) {
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    if (!iree_hal_webgpu_semaphore_has_submitted_signal(
            wait_semaphore_list.semaphores[i], queue->axis,
            wait_semaphore_list.payload_values[i])) {
      return false;
    }
  }
  return true;
}

// Marks all semaphores in the signal list as having a pending submitted signal
// from this queue. Called after GPU work is submitted in fast-path queue
// operations so that subsequent same-queue ops can use FIFO wait elision.
static void iree_hal_webgpu_queue_mark_signals_submitted(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_hal_webgpu_semaphore_mark_submitted_signal(
        signal_semaphore_list.semaphores[i], queue->axis,
        signal_semaphore_list.payload_values[i]);
  }
}

//===----------------------------------------------------------------------===//
// queue_alloca / queue_dealloca
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_queue_alloca(
    iree_hal_webgpu_queue_t* queue, iree_hal_allocator_t* device_allocator,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  *out_buffer = NULL;

  // Validate and coerce parameters through the allocator. This ensures the
  // stub buffer stores the correct memory type and usage flags for when
  // buffer_bind creates the actual GPU buffer.
  iree_hal_buffer_params_t compat_params;
  iree_device_size_t compat_allocation_size = allocation_size;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator, params, allocation_size, &compat_params,
          &compat_allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot serve the requested buffer parameters");
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }

  // Create a stub buffer (handle = 0) with the coerced parameters. The GPU
  // buffer is created later by buffer_bind, either inline (fast path) or in
  // the async wait callback.
  iree_hal_buffer_placement_t placement = {
      .device = NULL,
      .queue_affinity = compat_params.queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  iree_hal_buffer_t* buffer = NULL;
  {
    iree_status_t status = iree_hal_webgpu_buffer_create_stub(
        placement, compat_params.type, compat_params.access,
        compat_params.usage, compat_allocation_size, queue->host_allocator,
        &buffer);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
      return status;
    }
  }

  // Fast path: waits already satisfied (or FIFO-elided) — bind, signal, return.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
    iree_status_t status =
        iree_hal_webgpu_buffer_bind(buffer, queue->device_handle);
    if (iree_status_is_ok(status)) {
      iree_async_single_frontier_t frontier_storage;
      const iree_async_frontier_t* frontier =
          iree_hal_webgpu_queue_build_frontier(queue, epoch, &frontier_storage);
      status = iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_webgpu_queue_advance_tracker(queue, epoch);
      *out_buffer = buffer;
    } else {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
      iree_hal_webgpu_queue_advance_tracker(queue, epoch);
      iree_hal_buffer_release(buffer);
    }
    return status;
  }

  // Async path: give the stub to the caller immediately, then submit a wait
  // that binds the GPU buffer in its completion callback.
  iree_hal_webgpu_queue_state_t* state = NULL;
  iree_status_t status = iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_ALLOCA, wait_semaphore_list,
      signal_semaphore_list, /*trailing_count=*/0,
      /*trailing_element_size=*/0, /*out_trailing_offset=*/NULL, &state);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    return status;
  }

  // Retain the buffer for the slab — the caller also holds a reference via
  // *out_buffer. The slab's retain is released in release_resources (callback
  // success) or in the submit failure cleanup below.
  iree_hal_buffer_retain(buffer);
  state->alloca_op.buffer = buffer;
  *out_buffer = buffer;

  status = iree_hal_webgpu_queue_state_submit(state);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);  // Caller's ref — buffer unusable.
    *out_buffer = NULL;
  }
  return status;
}

iree_status_t iree_hal_webgpu_queue_dealloca(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  // Fast path: waits already satisfied (or FIFO-elided) — unbind, signal.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
    iree_hal_webgpu_buffer_unbind(buffer);
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(queue, epoch, &frontier_storage);
    iree_status_t status =
        iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
    iree_hal_webgpu_queue_advance_tracker(queue, epoch);
    return status;
  }

  // Async path: retain the buffer, submit a wait, and unbind in the callback.
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_DEALLOCA, wait_semaphore_list,
      signal_semaphore_list, /*trailing_count=*/0,
      /*trailing_element_size=*/0, /*out_trailing_offset=*/NULL, &state));

  iree_hal_buffer_retain(buffer);
  state->dealloca.buffer = buffer;

  return iree_hal_webgpu_queue_state_submit(state);
}

//===----------------------------------------------------------------------===//
// Queue operations (scratch builder)
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_queue_fill(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  // Fast path: waits satisfied or FIFO-elided — execute synchronously.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    iree_status_t status =
        iree_hal_webgpu_builder_reset(&queue->scratch_builder);
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_ref_t target_ref =
          iree_hal_make_buffer_ref(target_buffer, target_offset, length);
      status = iree_hal_webgpu_builder_fill_buffer(
          &queue->scratch_builder, target_ref, pattern, pattern_length);
    }
    return iree_hal_webgpu_queue_submit_scratch_and_signal(
        queue, signal_semaphore_list, status);
  }

  // Async path: capture params and submit wait. The callback does the scratch
  // build + GPU execute and registers an embedded signal for completion.
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_FILL, wait_semaphore_list,
      signal_semaphore_list, /*trailing_count=*/0,
      /*trailing_element_size=*/0, /*out_trailing_offset=*/NULL, &state));

  iree_hal_buffer_retain(target_buffer);
  state->fill.target_buffer = target_buffer;
  state->fill.target_offset = target_offset;
  state->fill.length = length;
  IREE_ASSERT(pattern_length <= sizeof(state->fill.pattern));
  memcpy(&state->fill.pattern, pattern, pattern_length);
  state->fill.pattern_length = pattern_length;

  return iree_hal_webgpu_queue_state_submit(state);
}

iree_status_t iree_hal_webgpu_queue_update(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  // Fast path: waits satisfied or FIFO-elided — execute synchronously. The
  // builder copies source data into the instruction stream blocks inline, so
  // the caller's source_buffer is consumed before this function returns.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    iree_status_t status =
        iree_hal_webgpu_builder_reset(&queue->scratch_builder);
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_ref_t target_ref =
          iree_hal_make_buffer_ref(target_buffer, target_offset, length);
      status = iree_hal_webgpu_builder_update_buffer(
          &queue->scratch_builder, source_buffer, source_offset, target_ref);
    }
    return iree_hal_webgpu_queue_submit_scratch_and_signal(
        queue, signal_semaphore_list, status);
  }

  // Async path: source data may be stack memory that becomes invalid after
  // this function returns. Validate length fits in a block, then acquire a
  // block from the pool and copy the data.
  if (length > queue->block_pool.total_block_size) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue_update length %" PRIdsz " exceeds block capacity %" PRIhsz,
        length, queue->block_pool.total_block_size);
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }
  iree_arena_block_t* captured_block = NULL;
  void* captured_data = NULL;
  {
    iree_status_t status = iree_arena_block_pool_acquire(
        &queue->block_pool, &captured_block, &captured_data);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
      return status;
    }
  }
  memcpy(captured_data, (const uint8_t*)source_buffer + source_offset,
         (size_t)length);

  iree_hal_webgpu_queue_state_t* state = NULL;
  iree_status_t status = iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_UPDATE, wait_semaphore_list,
      signal_semaphore_list, /*trailing_count=*/0,
      /*trailing_element_size=*/0, /*out_trailing_offset=*/NULL, &state);
  if (!iree_status_is_ok(status)) {
    iree_arena_block_pool_release(&queue->block_pool, captured_block,
                                  captured_block);
    return status;
  }

  iree_hal_buffer_retain(target_buffer);
  state->update.target_buffer = target_buffer;
  state->update.target_offset = target_offset;
  state->update.length = length;
  state->update.captured_data = captured_data;
  state->update.captured_block = captured_block;

  return iree_hal_webgpu_queue_state_submit(state);
}

iree_status_t iree_hal_webgpu_queue_copy(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  // Fast path: waits satisfied or FIFO-elided — execute synchronously.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    iree_status_t status =
        iree_hal_webgpu_builder_reset(&queue->scratch_builder);
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_ref_t source_ref =
          iree_hal_make_buffer_ref(source_buffer, source_offset, length);
      iree_hal_buffer_ref_t target_ref =
          iree_hal_make_buffer_ref(target_buffer, target_offset, length);
      status = iree_hal_webgpu_builder_copy_buffer(&queue->scratch_builder,
                                                   source_ref, target_ref);
    }
    return iree_hal_webgpu_queue_submit_scratch_and_signal(
        queue, signal_semaphore_list, status);
  }

  // Async path: capture params (retain both buffers) and submit wait.
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_COPY, wait_semaphore_list,
      signal_semaphore_list, /*trailing_count=*/0,
      /*trailing_element_size=*/0, /*out_trailing_offset=*/NULL, &state));

  iree_hal_buffer_retain(source_buffer);
  state->copy.source_buffer = source_buffer;
  state->copy.source_offset = source_offset;
  iree_hal_buffer_retain(target_buffer);
  state->copy.target_buffer = target_buffer;
  state->copy.target_offset = target_offset;
  state->copy.length = length;

  return iree_hal_webgpu_queue_state_submit(state);
}

// Performs a file-to-GPU transfer inline (no wait required). Calls the
// appropriate bridge import based on the file type, then registers signal
// completion for the GPU queue submission.
static iree_status_t iree_hal_webgpu_queue_read_inline(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* storage, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_webgpu_handle_t gpu_handle = iree_hal_webgpu_buffer_handle(
      iree_hal_buffer_allocated_buffer(target_buffer));
  uint64_t gpu_offset =
      iree_hal_buffer_byte_offset(target_buffer) + target_offset;

  iree_status_t status = iree_ok_status();
  if (storage) {
    // HOST_LOCAL: map the storage buffer and upload from host pointer.
    iree_hal_buffer_mapping_t mapping = {{0}};
    status = iree_hal_buffer_map_range(storage, IREE_HAL_MAPPING_MODE_SCOPED,
                                       IREE_HAL_MEMORY_ACCESS_READ,
                                       source_offset, length, &mapping);
    if (iree_status_is_ok(status)) {
      iree_hal_webgpu_import_queue_write_buffer(
          queue->queue_handle, gpu_handle, gpu_offset,
          (uint32_t)(uintptr_t)mapping.contents.data, length);
      iree_hal_buffer_unmap_range(&mapping);
    }
  } else {
    // FD: use zero-copy bridge import. The import_file dispatch guarantees
    // that files with NULL storage are webgpu_fd_files.
    int fd = iree_hal_webgpu_fd_file_fd(source_file);
    iree_hal_webgpu_import_queue_write_buffer_from_file(
        queue->queue_handle, gpu_handle, gpu_offset, (uint32_t)fd,
        source_offset, length);
  }

  // Register onSubmittedWorkDone for signal completion.
  uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_queue_register_signal_completion(
        queue, epoch, signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_mark_signals_submitted(queue, signal_semaphore_list);
  } else {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    iree_hal_webgpu_queue_advance_tracker(queue, epoch);
  }
  return status;
}

iree_status_t iree_hal_webgpu_queue_read(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  if (source_offset + length > iree_hal_file_length(source_file)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "read range [%" PRIu64 ", %" PRIu64
                            ") exceeds file length %" PRIu64,
                            source_offset, source_offset + length,
                            iree_hal_file_length(source_file));
  }

  // Determine the data source: HOST_LOCAL storage buffer or FD.
  iree_hal_buffer_t* storage = iree_hal_file_storage_buffer(source_file);

  // Fast path: waits satisfied or FIFO-elided — transfer inline.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    return iree_hal_webgpu_queue_read_inline(
        queue, signal_semaphore_list, source_file, source_offset, storage,
        target_buffer, target_offset, length);
  }

  // Async path: capture params and submit wait.
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_READ, wait_semaphore_list,
      signal_semaphore_list, /*trailing_count=*/0,
      /*trailing_element_size=*/0, /*out_trailing_offset=*/NULL, &state));

  if (storage) {
    iree_hal_buffer_retain(storage);
    state->read.storage = storage;
    state->read.source_file = NULL;
  } else {
    state->read.storage = NULL;
    iree_hal_file_retain(source_file);
    state->read.source_file = source_file;
  }
  state->read.source_offset = source_offset;
  iree_hal_buffer_retain(target_buffer);
  state->read.target_buffer = target_buffer;
  state->read.target_offset = target_offset;
  state->read.length = length;

  return iree_hal_webgpu_queue_state_submit(state);
}

//===----------------------------------------------------------------------===//
// queue_write (GPU → file) — three-phase async
//===----------------------------------------------------------------------===//
//
// GPU readback is always async in WebGPU (no synchronous map for non-mappable
// buffers). The three phases are:
//
//   Phase 1 (waits satisfied): Create MAP_READ|COPY_DST staging buffer, encode
//     copyBufferToBuffer(source → staging), submit, register
//     onSubmittedWorkDone.
//
//   Phase 2 (copy complete): mapAsync(staging, MAP_READ), register map
//     completion via proactor.
//
//   Phase 3 (map complete): Read data from staging into file (HOST_LOCAL:
//     buffer_get_mapped_range → host ptr; FD: file_write_from_mapped). Unmap
//     and destroy staging, signal semaphores, advance tracker, free state.

typedef struct iree_hal_webgpu_queue_write_state_t {
  // Phase 0: wait for input semaphores (if async path).
  iree_async_semaphore_wait_operation_t wait_operation;
  // Phase 1→2: onSubmittedWorkDone after staging copy.
  iree_async_operation_t copy_completion;
  // Phase 2→3: mapAsync on staging buffer.
  iree_async_operation_t map_completion;

  iree_hal_webgpu_queue_t* queue;
  uint64_t epoch;

  iree_hal_semaphore_list_t wait_semaphore_list;
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Source GPU buffer and range.
  iree_hal_buffer_t* source_buffer;  // Retained until Phase 1 submits.
  iree_device_size_t source_offset;
  iree_device_size_t length;

  // Target: HOST_LOCAL storage buffer (from file's storage_buffer()) or FD.
  iree_hal_buffer_t* target_storage;  // Retained if non-NULL.
  iree_hal_file_t* target_file;       // Retained if target_storage NULL.
  uint64_t target_offset;

  // MAP_READ|COPY_DST staging buffer created in Phase 1.
  iree_hal_webgpu_handle_t staging_handle;

  iree_allocator_t allocator;
  // Trailing: semaphore arrays for wait and signal lists.
} iree_hal_webgpu_queue_write_state_t;

// Forward declarations for the three phase callbacks.
static void iree_hal_webgpu_queue_write_phase1(
    iree_hal_webgpu_queue_write_state_t* state);
static void iree_hal_webgpu_queue_write_phase2(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags);
static void iree_hal_webgpu_queue_write_phase3(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags);

// Cleans up the write state on failure. Fails signal semaphores, releases
// all retained resources, and frees the slab.
static void iree_hal_webgpu_queue_write_state_fail(
    iree_hal_webgpu_queue_write_state_t* state, iree_status_t error) {
  iree_hal_semaphore_list_fail(state->signal_semaphore_list, error);
  iree_hal_webgpu_queue_advance_tracker(state->queue, state->epoch);
  if (state->source_buffer) iree_hal_buffer_release(state->source_buffer);
  if (state->target_storage) iree_hal_buffer_release(state->target_storage);
  if (state->target_file) iree_hal_file_release(state->target_file);
  if (state->staging_handle) {
    iree_hal_webgpu_import_buffer_destroy(state->staging_handle);
  }
  iree_hal_semaphore_list_release(state->wait_semaphore_list);
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_allocator_free(state->allocator, state);
}

// Phase 1: Create staging buffer, copy source → staging, submit.
// Called either directly (fast path, waits already satisfied) or from the
// wait completion callback (async path).
static void iree_hal_webgpu_queue_write_phase1(
    iree_hal_webgpu_queue_write_state_t* state) {
  iree_hal_webgpu_queue_t* queue = state->queue;

  // Create a MAP_READ | COPY_DST staging buffer.
  state->staging_handle = iree_hal_webgpu_import_device_create_buffer(
      queue->device_handle,
      IREE_HAL_WEBGPU_BUFFER_USAGE_MAP_READ |
          IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_DST,
      state->length, /*mapped_at_creation=*/0);
  if (state->staging_handle == 0) {
    iree_hal_webgpu_queue_write_state_fail(
        state, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to create staging buffer for "
                                "queue_write (%" PRIdsz " bytes)",
                                state->length));
    return;
  }

  // Encode copyBufferToBuffer(source → staging).
  iree_hal_webgpu_handle_t source_handle = iree_hal_webgpu_buffer_handle(
      iree_hal_buffer_allocated_buffer(state->source_buffer));
  uint64_t source_gpu_offset =
      iree_hal_buffer_byte_offset(state->source_buffer) + state->source_offset;

  uint32_t encoder_handle =
      iree_hal_webgpu_import_device_create_command_encoder(
          queue->device_handle);
  iree_hal_webgpu_import_encoder_copy_buffer_to_buffer(
      encoder_handle, source_handle, source_gpu_offset, state->staging_handle,
      /*dst_offset=*/0, state->length);
  uint32_t command_buffer_handle =
      iree_hal_webgpu_import_encoder_finish(encoder_handle);

  // Submit and release the source buffer (GPU has its own references).
  iree_hal_webgpu_import_queue_submit(queue->queue_handle,
                                      command_buffer_handle);
  iree_hal_buffer_release(state->source_buffer);
  state->source_buffer = NULL;

  // Register onSubmittedWorkDone → Phase 2.
  iree_async_operation_initialize(
      &state->copy_completion, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_webgpu_queue_write_phase2,
      /*user_data=*/NULL);
  uint32_t token = UINT32_MAX;
  iree_status_t status = iree_async_proactor_js_submit_external(
      queue->proactor, &state->copy_completion, &token);
  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_write_state_fail(state, status);
    return;
  }
  iree_hal_webgpu_import_queue_on_submitted_work_done(queue->queue_handle,
                                                      token);
}

// Phase 2: Copy complete → mapAsync on staging buffer.
static void iree_hal_webgpu_queue_write_phase2(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_queue_write_state_t* state =
      (iree_hal_webgpu_queue_write_state_t*)((uint8_t*)base_operation -
                                             offsetof(
                                                 iree_hal_webgpu_queue_write_state_t,
                                                 copy_completion));

  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_write_state_fail(state, status);
    return;
  }

  // Register mapAsync completion → Phase 3.
  iree_async_operation_initialize(
      &state->map_completion, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_webgpu_queue_write_phase3,
      /*user_data=*/NULL);
  uint32_t token = UINT32_MAX;
  status = iree_async_proactor_js_submit_external(
      state->queue->proactor, &state->map_completion, &token);
  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_write_state_fail(state, status);
    return;
  }

  // Initiate mapAsync(staging, MAP_READ).
  iree_hal_webgpu_import_buffer_map_async(state->staging_handle,
                                          /*mode=*/1 /*MAP_READ*/, /*offset=*/0,
                                          state->length, token);
}

// Phase 3: Map complete → read data into file → cleanup → signal.
static void iree_hal_webgpu_queue_write_phase3(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_queue_write_state_t* state =
      (iree_hal_webgpu_queue_write_state_t*)((uint8_t*)base_operation -
                                             offsetof(
                                                 iree_hal_webgpu_queue_write_state_t,
                                                 map_completion));

  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_write_state_fail(state, status);
    return;
  }

  // Read data from the mapped staging buffer into the target file.
  if (state->target_storage) {
    // HOST_LOCAL: map the target storage buffer and copy from staging.
    iree_hal_buffer_mapping_t mapping = {{0}};
    status = iree_hal_buffer_map_range(
        state->target_storage, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_WRITE, state->target_offset, state->length,
        &mapping);
    if (iree_status_is_ok(status)) {
      iree_hal_webgpu_import_buffer_get_mapped_range(
          state->staging_handle, /*offset=*/0, state->length,
          (uint32_t)(uintptr_t)mapping.contents.data);
      iree_hal_buffer_unmap_range(&mapping);
    }
  } else {
    // FD: write from mapped staging buffer directly to the file object.
    int fd = iree_hal_webgpu_fd_file_fd(state->target_file);
    iree_hal_webgpu_import_file_write_from_mapped(
        state->staging_handle, /*buffer_offset=*/0, state->length, (uint32_t)fd,
        state->target_offset);
  }

  // Cleanup staging buffer.
  iree_hal_webgpu_import_buffer_unmap(state->staging_handle);
  iree_hal_webgpu_import_buffer_destroy(state->staging_handle);
  state->staging_handle = 0;

  // Release target resources.
  if (state->target_storage) {
    iree_hal_buffer_release(state->target_storage);
    state->target_storage = NULL;
  }
  if (state->target_file) {
    iree_hal_file_release(state->target_file);
    state->target_file = NULL;
  }

  // Signal or fail.
  if (iree_status_is_ok(status)) {
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(state->queue, state->epoch,
                                             &frontier_storage);
    iree_status_ignore(
        iree_hal_semaphore_list_signal(state->signal_semaphore_list, frontier));
  } else {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
  }

  iree_hal_webgpu_queue_advance_tracker(state->queue, state->epoch);
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_allocator_free(state->allocator, state);
}

// Wait completion callback for the async path. Releases wait semaphores,
// then enters Phase 1.
static void iree_hal_webgpu_queue_write_wait_completion(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t wait_status, iree_async_completion_flags_t flags) {
  iree_hal_webgpu_queue_write_state_t* state =
      (iree_hal_webgpu_queue_write_state_t*)base_operation;
  iree_hal_semaphore_list_release(state->wait_semaphore_list);
  state->wait_semaphore_list.count = 0;  // Prevent double-release in fail.

  if (!iree_status_is_ok(wait_status)) {
    iree_hal_webgpu_queue_write_state_fail(state, wait_status);
    return;
  }

  iree_hal_webgpu_queue_write_phase1(state);
}

iree_status_t iree_hal_webgpu_queue_write(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  if (target_offset + length > iree_hal_file_length(target_file)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "write range [%" PRIu64 ", %" PRIu64
                            ") exceeds file length %" PRIu64,
                            target_offset, target_offset + length,
                            iree_hal_file_length(target_file));
  }

  // Determine the target type: HOST_LOCAL storage buffer or FD.
  iree_hal_buffer_t* target_storage = iree_hal_file_storage_buffer(target_file);

  // Allocate the write state slab with trailing semaphore arrays.
  uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);

  iree_host_size_t total_size = 0;
  iree_host_size_t wait_semaphores_offset = 0;
  iree_host_size_t wait_values_offset = 0;
  iree_host_size_t signal_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_webgpu_queue_write_state_t), &total_size,
      IREE_STRUCT_FIELD(wait_semaphore_list.count, iree_async_semaphore_t*,
                        &wait_semaphores_offset),
      IREE_STRUCT_FIELD(wait_semaphore_list.count, uint64_t,
                        &wait_values_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, iree_hal_semaphore_t*,
                        &signal_semaphores_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, uint64_t,
                        &signal_values_offset));
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }

  iree_hal_webgpu_queue_write_state_t* state = NULL;
  status =
      iree_allocator_malloc(queue->host_allocator, total_size, (void**)&state);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }

  memset(state, 0, sizeof(*state));
  state->queue = queue;
  state->epoch = epoch;
  state->allocator = queue->host_allocator;

  // Clone semaphore lists into trailing slab.
  state->wait_semaphore_list = iree_hal_semaphore_list_at_offsets(
      state, wait_semaphore_list.count, wait_semaphores_offset,
      wait_values_offset);
  iree_hal_semaphore_list_clone_into(wait_semaphore_list,
                                     state->wait_semaphore_list);
  state->signal_semaphore_list = iree_hal_semaphore_list_at_offsets(
      state, signal_semaphore_list.count, signal_semaphores_offset,
      signal_values_offset);
  iree_hal_semaphore_list_clone_into(signal_semaphore_list,
                                     state->signal_semaphore_list);

  // Capture operation parameters.
  iree_hal_buffer_retain(source_buffer);
  state->source_buffer = source_buffer;
  state->source_offset = source_offset;
  state->length = length;
  state->target_offset = target_offset;
  if (target_storage) {
    iree_hal_buffer_retain(target_storage);
    state->target_storage = target_storage;
  } else {
    iree_hal_file_retain(target_file);
    state->target_file = target_file;
  }

  // Fast path: waits already satisfied — go directly to Phase 1.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    iree_hal_semaphore_list_release(state->wait_semaphore_list);
    state->wait_semaphore_list.count = 0;  // Prevent double-release in fail.
    iree_hal_webgpu_queue_write_phase1(state);
    return iree_ok_status();
  }

  // Async path: set up the wait operation and submit to proactor.
  state->wait_operation.semaphores =
      (iree_async_semaphore_t**)state->wait_semaphore_list.semaphores;
  state->wait_operation.values = state->wait_semaphore_list.payload_values;
  state->wait_operation.count = wait_semaphore_list.count;
  state->wait_operation.mode = IREE_ASYNC_WAIT_MODE_ALL;
  state->wait_operation.satisfied_index = 0;
  iree_async_operation_initialize(
      &state->wait_operation.base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_webgpu_queue_write_wait_completion, /*user_data=*/NULL);

  status = iree_async_proactor_submit_one(queue->proactor,
                                          &state->wait_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_webgpu_queue_write_state_fail(state, iree_status_clone(status));
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Host call (inline + proactor-driven async)
//===----------------------------------------------------------------------===//

// Executes a host call inline and handles the result status.
// On OK: signals all signal semaphores with the given frontier.
// On DEFERRED: the callback has cloned the signal list and will signal later.
//   The DEFERRED path does not use the frontier — the callback manages its own
//   signaling and can build a frontier itself if needed.
// On error: fails all signal semaphores with the error status.
// If NON_BLOCKING: signals semaphores before calling, ignores the result.
static iree_status_t iree_hal_webgpu_queue_execute_host_call(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* frontier, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags) {
  if (flags & IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING) {
    // Signal semaphores immediately, then fire the call. The callback cannot
    // observe the signal list and its result is ignored.
    iree_status_t signal_status =
        iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
    iree_hal_host_call_context_t context = {
        .device = device,
        .queue_affinity = queue_affinity,
        .signal_semaphore_list = iree_hal_semaphore_list_empty(),
    };
    iree_status_ignore(call.fn(call.user_data, args, &context));
    return signal_status;
  }

  iree_hal_host_call_context_t context = {
      .device = device,
      .queue_affinity = queue_affinity,
      .signal_semaphore_list = signal_semaphore_list,
  };
  iree_status_t call_status = call.fn(call.user_data, args, &context);
  if (iree_status_is_ok(call_status)) {
    return iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
  } else if (iree_status_code(call_status) == IREE_STATUS_DEFERRED) {
    // The callback has cloned the signal list and will signal later.
    iree_status_ignore(call_status);
    return iree_ok_status();
  } else {
    // Fail all signal semaphores with the error.
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(call_status));
    return call_status;
  }
}

// State for a deferred host call that waits on input semaphores via the
// proactor before executing. Allocated as a single slab:
//   [struct | wait iree_async_semaphore_t*[] | wait uint64_t[] |
//    signal iree_hal_semaphore_t*[] | signal uint64_t[]]
typedef struct iree_hal_webgpu_host_call_state_t {
  iree_async_semaphore_wait_operation_t wait_operation;
  iree_hal_host_call_t call;
  uint64_t args[4];
  iree_hal_host_call_flags_t flags;
  iree_hal_device_t* device;
  iree_hal_webgpu_queue_t* queue;
  iree_hal_queue_affinity_t queue_affinity;
  uint64_t epoch;  // Pre-incremented at submit for causal ordering.
  iree_hal_semaphore_list_t signal_semaphore_list;
  iree_hal_semaphore_list_t wait_semaphore_list;
  iree_allocator_t allocator;
} iree_hal_webgpu_host_call_state_t;

// Completion callback invoked by the proactor when all wait semaphores are
// satisfied (or one has failed).
static void iree_hal_webgpu_host_call_completion_fn(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t wait_status, iree_async_completion_flags_t completion_flags) {
  iree_hal_webgpu_host_call_state_t* state =
      (iree_hal_webgpu_host_call_state_t*)base_operation;

  if (iree_status_is_ok(wait_status)) {
    // Wait succeeded — build frontier and execute the host call. This handles
    // signaling/failing the signal semaphores based on the call result.
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(state->queue, state->epoch,
                                             &frontier_storage);
    iree_status_ignore(iree_hal_webgpu_queue_execute_host_call(
        state->device, state->queue_affinity, state->signal_semaphore_list,
        frontier, state->call, state->args, state->flags));
  } else {
    // Wait itself failed (semaphore failure propagation). Fail all signal
    // semaphores with the wait failure status.
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, wait_status);
  }

  iree_hal_webgpu_queue_advance_tracker(state->queue, state->epoch);

  // Release both semaphore lists and free the slab.
  iree_hal_semaphore_list_release(state->signal_semaphore_list);
  iree_hal_semaphore_list_release(state->wait_semaphore_list);
  iree_allocator_free(state->allocator, state);
}

iree_status_t iree_hal_webgpu_queue_host_call(
    iree_hal_webgpu_queue_t* queue, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  // Fast path: no wait semaphores or all already satisfied — execute inline.
  // NO FIFO elision — host calls execute on the CPU, not the GPU FIFO.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list)) {
    uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
    iree_async_single_frontier_t frontier_storage;
    const iree_async_frontier_t* frontier =
        iree_hal_webgpu_queue_build_frontier(queue, epoch, &frontier_storage);
    iree_status_t status = iree_hal_webgpu_queue_execute_host_call(
        device, queue_affinity, signal_semaphore_list, frontier, call, args,
        flags);
    iree_hal_webgpu_queue_advance_tracker(queue, epoch);
    return status;
  }

  uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);

  // Async path: allocate a slab with the wait operation, call state, and
  // trailing arrays for both wait and signal semaphore lists.
  iree_host_size_t total_size = 0;
  iree_host_size_t wait_semaphores_offset = 0;
  iree_host_size_t wait_values_offset = 0;
  iree_host_size_t signal_semaphores_offset = 0;
  iree_host_size_t signal_values_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_webgpu_host_call_state_t), &total_size,
      IREE_STRUCT_FIELD(wait_semaphore_list.count, iree_async_semaphore_t*,
                        &wait_semaphores_offset),
      IREE_STRUCT_FIELD(wait_semaphore_list.count, uint64_t,
                        &wait_values_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, iree_hal_semaphore_t*,
                        &signal_semaphores_offset),
      IREE_STRUCT_FIELD(signal_semaphore_list.count, uint64_t,
                        &signal_values_offset));
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }

  iree_hal_webgpu_host_call_state_t* state = NULL;
  status =
      iree_allocator_malloc(queue->host_allocator, total_size, (void**)&state);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    return status;
  }

  // Copy call parameters.
  state->call = call;
  memcpy(state->args, args, sizeof(state->args));
  state->flags = flags;
  state->device = device;
  state->queue = queue;
  state->queue_affinity = queue_affinity;
  state->epoch = epoch;
  state->allocator = queue->host_allocator;

  // Set up the wait semaphore list and clone the caller's data into it.
  // The wait operation's semaphore pointers alias the same array: WebGPU
  // semaphores have iree_async_semaphore_t at offset 0, so the pointer bits
  // are identical regardless of which type they're cast to.
  state->wait_semaphore_list = iree_hal_semaphore_list_at_offsets(
      state, wait_semaphore_list.count, wait_semaphores_offset,
      wait_values_offset);
  iree_hal_semaphore_list_clone_into(wait_semaphore_list,
                                     state->wait_semaphore_list);
  state->wait_operation.semaphores =
      (iree_async_semaphore_t**)state->wait_semaphore_list.semaphores;
  state->wait_operation.values = state->wait_semaphore_list.payload_values;
  state->wait_operation.count = wait_semaphore_list.count;
  state->wait_operation.mode = IREE_ASYNC_WAIT_MODE_ALL;
  state->wait_operation.satisfied_index = 0;

  // Set up the signal semaphore list and clone the caller's data into it.
  state->signal_semaphore_list = iree_hal_semaphore_list_at_offsets(
      state, signal_semaphore_list.count, signal_semaphores_offset,
      signal_values_offset);
  iree_hal_semaphore_list_clone_into(signal_semaphore_list,
                                     state->signal_semaphore_list);

  // Initialize the wait operation base and submit to the proactor.
  iree_async_operation_initialize(
      &state->wait_operation.base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_webgpu_host_call_completion_fn,
      /*user_data=*/NULL);

  status = iree_async_proactor_submit_one(queue->proactor,
                                          &state->wait_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list,
                                 iree_status_clone(status));
    iree_hal_semaphore_list_release(state->signal_semaphore_list);
    iree_hal_semaphore_list_release(state->wait_semaphore_list);
    iree_allocator_free(queue->host_allocator, state);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// queue_dispatch
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_queue_dispatch(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  // Extract pipeline/bgl handles at submit time. These are bridge table
  // indices (uint32 values) that remain valid as long as the executable lives.
  iree_hal_webgpu_handle_t pipeline_handle =
      iree_hal_webgpu_executable_pipeline_handle(executable, export_ordinal);
  iree_hal_webgpu_handle_t bind_group_layout_handle =
      iree_hal_webgpu_executable_bind_group_layout_handle(executable,
                                                          export_ordinal);

  // Fast path: waits already satisfied (or FIFO-elided) — execute
  // synchronously.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    iree_status_t status =
        iree_hal_webgpu_builder_reset(&queue->scratch_builder);
    if (iree_status_is_ok(status)) {
      status = iree_hal_webgpu_builder_dispatch(
          &queue->scratch_builder, pipeline_handle, bind_group_layout_handle,
          config.workgroup_count, bindings);
    }
    return iree_hal_webgpu_queue_submit_scratch_and_signal(
        queue, signal_semaphore_list, status);
  }

  // Async path: capture params, retain executable + per-binding buffers,
  // snapshot bindings into trailing slab.
  iree_host_size_t bindings_offset = 0;
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_DISPATCH, wait_semaphore_list,
      signal_semaphore_list, bindings.count, sizeof(iree_hal_buffer_ref_t),
      &bindings_offset, &state));

  state->dispatch.pipeline_handle = pipeline_handle;
  state->dispatch.bind_group_layout_handle = bind_group_layout_handle;
  memcpy(state->dispatch.workgroup_count, config.workgroup_count,
         sizeof(state->dispatch.workgroup_count));

  iree_hal_executable_retain(executable);
  state->dispatch.executable = executable;

  // Snapshot bindings into the trailing slab and retain each buffer.
  state->dispatch.bindings =
      (iree_hal_buffer_ref_t*)((uint8_t*)state + bindings_offset);
  state->dispatch.binding_count = (uint32_t)bindings.count;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    state->dispatch.bindings[i] = bindings.values[i];
    if (bindings.values[i].buffer) {
      iree_hal_buffer_retain(bindings.values[i].buffer);
    }
  }

  return iree_hal_webgpu_queue_state_submit(state);
}

// Submits an async barrier (wait → signal inline) to the proactor using the
// unified queue state. Called when iree_hal_semaphore_list_poll indicates the
// wait semaphores are not yet satisfied. On success, ownership transfers to
// the proactor — the wait completion callback signals and frees the state.
static iree_status_t iree_hal_webgpu_queue_submit_barrier(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_BARRIER, wait_semaphore_list,
      signal_semaphore_list, /*trailing_count=*/0,
      /*trailing_element_size=*/0, /*out_trailing_offset=*/NULL, &state));
  return iree_hal_webgpu_queue_state_submit(state);
}

//===----------------------------------------------------------------------===//
// queue_execute
//===----------------------------------------------------------------------===//

// Executes a reusable command buffer's cached recording. Dynamic bindings from
// the binding table are resolved to GPU buffer handles for the JS processor.
static iree_status_t iree_hal_webgpu_queue_execute_recording(
    iree_hal_webgpu_queue_t* queue, iree_hal_webgpu_handle_t recording_handle,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_webgpu_isa_binding_table_entry_t inline_entries[8];
  iree_hal_webgpu_isa_binding_table_entry_t* dynamic_entries = inline_entries;
  if (binding_table.count > IREE_ARRAYSIZE(inline_entries)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        queue->host_allocator, binding_table.count,
        sizeof(iree_hal_webgpu_isa_binding_table_entry_t),
        (void**)&dynamic_entries));
  }

  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    const iree_hal_buffer_binding_t* binding = &binding_table.bindings[i];
    if (binding->buffer) {
      iree_hal_buffer_t* allocated =
          iree_hal_buffer_allocated_buffer(binding->buffer);
      dynamic_entries[i].gpu_buffer_handle =
          iree_hal_webgpu_buffer_handle(allocated);
      dynamic_entries[i].base_offset =
          (uint32_t)iree_hal_buffer_byte_offset(binding->buffer);
    } else {
      dynamic_entries[i].gpu_buffer_handle = 0;
      dynamic_entries[i].base_offset = 0;
    }
  }

  uint32_t result = iree_hal_webgpu_import_execute_recording(
      recording_handle, queue->queue_handle,
      (uint32_t)(uintptr_t)dynamic_entries);

  if (dynamic_entries != inline_entries) {
    iree_allocator_free(queue->host_allocator, dynamic_entries);
  }
  if (result != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "JS execute_recording failed with code %u", result);
  }
  return iree_ok_status();
}

// Executes a ONE_SHOT command buffer's instruction stream directly. Builds the
// binding table from the builder's static slot map and passes it with the
// builtins descriptor to the JS processor.
static iree_status_t iree_hal_webgpu_queue_execute_one_shot(
    iree_hal_webgpu_queue_t* queue, iree_hal_command_buffer_t* command_buffer) {
  iree_hal_webgpu_builder_t* builder =
      iree_hal_webgpu_command_buffer_builder(command_buffer);

  uint32_t total_slots = iree_hal_webgpu_builder_total_slot_count(builder);
  uint32_t static_count = iree_hal_webgpu_builder_static_slot_count(builder);

  iree_hal_webgpu_isa_binding_table_entry_t inline_entries[8];
  iree_hal_webgpu_isa_binding_table_entry_t* entries = inline_entries;
  if (total_slots > IREE_ARRAYSIZE(inline_entries)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        queue->host_allocator, total_slots,
        sizeof(iree_hal_webgpu_isa_binding_table_entry_t), (void**)&entries));
  }

  const iree_hal_webgpu_builder_slot_entry_t* slot_entries =
      iree_hal_webgpu_builder_static_slot_entries(builder);
  for (uint32_t i = 0; i < static_count; ++i) {
    entries[slot_entries[i].slot].gpu_buffer_handle =
        slot_entries[i].gpu_buffer_handle;
    entries[slot_entries[i].slot].base_offset = 0;
  }

  iree_hal_webgpu_isa_builtins_descriptor_t builtins_descriptor;
  iree_hal_webgpu_builtins_get_descriptor(queue->builtins,
                                          &builtins_descriptor);

  uint32_t result = iree_hal_webgpu_import_execute_instructions(
      queue->device_handle, queue->queue_handle,
      (uint32_t)(uintptr_t)iree_hal_webgpu_builder_block_table(builder),
      iree_hal_webgpu_builder_block_count(builder),
      iree_hal_webgpu_builder_block_word_capacity(builder),
      iree_hal_webgpu_builder_last_block_word_count(builder),
      (uint32_t)(uintptr_t)entries, total_slots,
      (uint32_t)(uintptr_t)&builtins_descriptor);

  if (entries != inline_entries) {
    iree_allocator_free(queue->host_allocator, entries);
  }
  if (result != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "JS execute_instructions failed with code %u",
                            result);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_queue_execute(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  // Barrier-only submission: no command buffer, just wait->signal.
  // NO FIFO elision — barrier signals are CPU-side with no GPU FIFO backing,
  // so FIFO ordering does not guarantee the signal is visible to consumers.
  if (!command_buffer) {
    if (wait_semaphore_list.count == 0 ||
        iree_hal_semaphore_list_poll(wait_semaphore_list)) {
      uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
      iree_async_single_frontier_t frontier_storage;
      const iree_async_frontier_t* frontier =
          iree_hal_webgpu_queue_build_frontier(queue, epoch, &frontier_storage);
      iree_status_t status =
          iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
      iree_hal_webgpu_queue_advance_tracker(queue, epoch);
      return status;
    }
    return iree_hal_webgpu_queue_submit_barrier(queue, wait_semaphore_list,
                                                signal_semaphore_list);
  }

  // Fast path: waits already satisfied (or FIFO-elided) — execute
  // synchronously.
  if (wait_semaphore_list.count == 0 ||
      iree_hal_semaphore_list_poll(wait_semaphore_list) ||
      iree_hal_webgpu_queue_can_elide_waits(queue, wait_semaphore_list)) {
    uint64_t epoch = iree_hal_webgpu_queue_reserve_epoch(queue);
    iree_hal_webgpu_handle_t recording_handle =
        iree_hal_webgpu_command_buffer_recording_handle(command_buffer);
    iree_status_t status;
    if (recording_handle) {
      status = iree_hal_webgpu_queue_execute_recording(queue, recording_handle,
                                                       binding_table);
    } else {
      status = iree_hal_webgpu_queue_execute_one_shot(queue, command_buffer);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_webgpu_queue_register_signal_completion(
          queue, epoch, signal_semaphore_list);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_webgpu_queue_mark_signals_submitted(queue,
                                                   signal_semaphore_list);
    } else {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
                                   iree_status_clone(status));
    }
    return status;
  }

  // Async path: retain command buffer, snapshot binding table into trailing
  // slab (retaining each buffer), submit wait.
  iree_host_size_t binding_table_offset = 0;
  iree_hal_webgpu_queue_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_queue_state_allocate(
      queue, IREE_HAL_WEBGPU_QUEUE_OP_EXECUTE, wait_semaphore_list,
      signal_semaphore_list, binding_table.count,
      sizeof(iree_hal_buffer_binding_t), &binding_table_offset, &state));

  iree_hal_command_buffer_retain(command_buffer);
  state->execute.command_buffer = command_buffer;

  // Snapshot the binding table into the trailing slab and retain each buffer.
  state->execute.binding_table =
      (iree_hal_buffer_binding_t*)((uint8_t*)state + binding_table_offset);
  state->execute.binding_count = binding_table.count;
  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    state->execute.binding_table[i] = binding_table.bindings[i];
    if (binding_table.bindings[i].buffer) {
      iree_hal_buffer_retain(binding_table.bindings[i].buffer);
    }
  }

  return iree_hal_webgpu_queue_state_submit(state);
}

//===----------------------------------------------------------------------===//
// queue_flush
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_queue_flush(iree_hal_webgpu_queue_t* queue) {
  // WebGPU's queue.submit() is not buffered — commands are submitted
  // immediately. No flush is needed.
  return iree_ok_status();
}
