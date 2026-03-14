// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_processor.h"

#include <stddef.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Shared context
//===----------------------------------------------------------------------===//

// Shared state across all workers processing a recording. Allocated by
// context_allocate, freed by context_free.
// iree_hal_cmd_block_processor_context_t is defined in block_processor.h.

//===----------------------------------------------------------------------===//
// Error reporting
//===----------------------------------------------------------------------===//

// Records the first error encountered by any worker. Thread-safe via CAS.
// If another worker already recorded an error, this error is dropped.
// Also sets the completed flag so that all workers exit on their next
// drain() call.
static void iree_hal_cmd_block_processor_report_error(
    iree_hal_cmd_block_processor_context_t* context, iree_status_t status) {
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &context->error_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_ignore(status);
  }
  iree_atomic_store(&context->completed, 1, iree_memory_order_release);
}

// Returns true if any worker has reported an error (early exit check).
static bool iree_hal_cmd_block_processor_has_error(
    const iree_hal_cmd_block_processor_context_t* context) {
  return iree_atomic_load(&context->error_status, iree_memory_order_relaxed) !=
         0;
}

//===----------------------------------------------------------------------===//
// Pattern fill
//===----------------------------------------------------------------------===//

// Fills |length| bytes at |target| with a repeating pattern of
// |pattern_length| bytes from |pattern|. |length| must be a multiple of
// |pattern_length|. |pattern_length| must be 1, 2, or 4.
static void iree_hal_cmd_fill_pattern(uint8_t* target,
                                      iree_device_size_t length,
                                      uint64_t pattern,
                                      uint8_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      memset(target, (uint8_t)pattern, (size_t)length);
      break;
    }
    case 2: {
      uint16_t value = (uint16_t)pattern;
      for (iree_device_size_t i = 0; i < length; i += 2) {
        memcpy(target + i, &value, 2);
      }
      break;
    }
    case 4: {
      uint32_t value = (uint32_t)pattern;
      for (iree_device_size_t i = 0; i < length; i += 4) {
        memcpy(target + i, &value, 4);
      }
      break;
    }
    default:
      IREE_ASSERT(false, "unsupported fill pattern length");
      break;
  }
}

//===----------------------------------------------------------------------===//
// Binding resolution
//===----------------------------------------------------------------------===//

// Resolves all binding pointers for a block via its fixup table.
// Populates state->binding_ptrs[fixup.data_index] for each fixup entry.
static void iree_hal_cmd_block_processor_resolve_bindings(
    const iree_hal_cmd_block_header_t* block, iree_hal_cmd_block_state_t* state,
    uint16_t max_region_dispatch_count,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length) {
  void** binding_ptrs =
      iree_hal_cmd_block_state_binding_ptrs(state, max_region_dispatch_count);

  const iree_hal_cmd_fixup_t* fixups = iree_hal_cmd_block_fixups(block);
  for (uint16_t i = 0; i < block->fixup_count; ++i) {
    const iree_hal_cmd_fixup_t* fixup = &fixups[i];
    if (fixup->span) {
      // Direct fixup: dereference the buffer's span pointer.
      binding_ptrs[fixup->data_index] =
          iree_async_span_ptr(*fixup->span) + fixup->offset;
    } else {
      // Indirect fixup: look up in the binding table.
      IREE_ASSERT(fixup->slot < binding_table_length);
      binding_ptrs[fixup->data_index] =
          (uint8_t*)binding_table[fixup->slot].base + fixup->offset;
    }
  }
}

//===----------------------------------------------------------------------===//
// Per-command tile execution
//===----------------------------------------------------------------------===//

// Executes tiles for a DISPATCH command. Claims tiles via epoch-tagged CAS
// on the command's 64-bit tile_index entry, executes each tile by calling the
// kernel function, and returns the total number of tiles completed.
//
// The CAS validates the epoch (region_index in upper 32 bits) atomically
// with the tile claim. If the epoch doesn't match (stale worker from a
// previous region), the CAS fails and the worker gets 0 tiles.
//
// For single-worker mode (worker_count==1): the tile_index atomic is not
// touched (unnecessary overhead). All tiles are executed sequentially.
static iree_status_t iree_hal_cmd_execute_dispatch_tiles(
    const iree_hal_cmd_dispatch_t* dispatch, void** binding_ptrs,
    iree_atomic_int64_t* tile_index, int32_t region_epoch,
    uint32_t worker_count, uint32_t* out_tiles_completed) {
  *out_tiles_completed = 0;

  // Resolve workgroup count: direct (inline) or indirect (from buffer).
  uint32_t workgroup_count[3];
  if (dispatch->header.flags & IREE_HAL_CMD_FLAG_INDIRECT) {
    const void* params_buffer =
        binding_ptrs[dispatch->params.indirect.params_binding];
    const iree_hal_dispatch_params_t* params =
        (const iree_hal_dispatch_params_t*)((const uint8_t*)params_buffer +
                                            dispatch->params.indirect
                                                .params_offset);
    workgroup_count[0] = params->workgroup_count[0];
    workgroup_count[1] = params->workgroup_count[1];
    workgroup_count[2] = params->workgroup_count[2];
  } else {
    workgroup_count[0] = dispatch->params.direct.workgroup_count[0];
    workgroup_count[1] = dispatch->params.direct.workgroup_count[1];
    workgroup_count[2] = dispatch->params.direct.workgroup_count[2];
  }

  // Predicate check: skip if all workgroup counts are zero.
  if ((dispatch->header.flags & IREE_HAL_CMD_FLAG_PREDICATED) &&
      workgroup_count[0] == 0 && workgroup_count[1] == 0 &&
      workgroup_count[2] == 0) {
    return iree_ok_status();
  }

  const uint32_t tile_count =
      workgroup_count[0] * workgroup_count[1] * workgroup_count[2];
  if (tile_count == 0) return iree_ok_status();

  // Build the dispatch state on the stack. Every pointer field is a direct
  // assignment into .text or .data — no layout computation, no memcpy.
  iree_hal_executable_dispatch_state_v0_t dispatch_state;
  memset(&dispatch_state, 0, sizeof(dispatch_state));
  dispatch_state.workgroup_size_x = dispatch->workgroup_size[0];
  dispatch_state.workgroup_size_y = dispatch->workgroup_size[1];
  dispatch_state.workgroup_size_z = (uint16_t)dispatch->workgroup_size[2];
  dispatch_state.workgroup_count_x = workgroup_count[0];
  dispatch_state.workgroup_count_y = workgroup_count[1];
  dispatch_state.workgroup_count_z = (uint16_t)workgroup_count[2];
  dispatch_state.constant_count = (uint16_t)dispatch->constant_count;
  dispatch_state.constants = dispatch->constants;
  dispatch_state.binding_count = dispatch->binding_count;
  dispatch_state.binding_ptrs =
      (void* const*)&binding_ptrs[dispatch->binding_data_base];
  dispatch_state.binding_lengths = NULL;

  const uint32_t xy_count = workgroup_count[0] * workgroup_count[1];
  const uint32_t tiles_per_reservation =
      dispatch->tiles_per_reservation > 0 ? dispatch->tiles_per_reservation : 1;

  // Allocate local memory for this dispatch (per-worker, stack-allocated).
  uint8_t* local_memory = NULL;
  if (dispatch->local_memory_size > 0) {
    local_memory = (uint8_t*)iree_alloca(dispatch->local_memory_size);
    memset(local_memory, 0, dispatch->local_memory_size);
  }

  if (worker_count == 1) {
    // Single-worker fast path: no atomics needed. Execute all tiles.
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
      iree_hal_executable_workgroup_state_v0_t workgroup_state;
      memset(&workgroup_state, 0, sizeof(workgroup_state));
      workgroup_state.workgroup_id_x = tile % workgroup_count[0];
      workgroup_state.workgroup_id_y =
          (tile / workgroup_count[0]) % workgroup_count[1];
      workgroup_state.workgroup_id_z = (uint16_t)(tile / xy_count);
      workgroup_state.processor_id = 0;
      workgroup_state.local_memory = local_memory;
      workgroup_state.local_memory_size = dispatch->local_memory_size;

      int ret = dispatch->function(dispatch->environment, &dispatch_state,
                                   &workgroup_state);
      if (IREE_UNLIKELY(ret != 0)) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "dispatch kernel returned non-zero (%d)", ret);
      }
    }
    *out_tiles_completed = tile_count;
  } else {
    // Multi-worker: claim tiles via epoch-tagged CAS on 64-bit atomic.
    // Upper 32 bits = global epoch, lower 32 bits = tile counter.
    // CAS atomically validates the epoch and claims tiles. Stale workers
    // (wrong epoch) fail harmlessly with zero side effects.
    uint32_t completed = 0;
    const int64_t epoch_shifted = (int64_t)region_epoch << 32;
    int64_t current = iree_atomic_load(tile_index, iree_memory_order_relaxed);
    while (true) {
      // Validate epoch and check for exhaustion.
      if ((current >> 32) != region_epoch) break;
      uint32_t counter = (uint32_t)(current & 0xFFFFFFFF);
      if (counter >= tile_count) break;

      // Compute the desired value (clamp to tile_count).
      uint32_t new_counter = counter + tiles_per_reservation;
      if (new_counter > tile_count) new_counter = tile_count;
      int64_t desired = epoch_shifted | (int64_t)new_counter;

      if (iree_atomic_compare_exchange_weak(tile_index, &current, desired,
                                            iree_memory_order_relaxed,
                                            iree_memory_order_relaxed)) {
        // CAS succeeded — execute claimed tiles [counter, new_counter).
        for (uint32_t tile = counter; tile < new_counter; ++tile) {
          iree_hal_executable_workgroup_state_v0_t workgroup_state;
          memset(&workgroup_state, 0, sizeof(workgroup_state));
          workgroup_state.workgroup_id_x = tile % workgroup_count[0];
          workgroup_state.workgroup_id_y =
              (tile / workgroup_count[0]) % workgroup_count[1];
          workgroup_state.workgroup_id_z = (uint16_t)(tile / xy_count);
          workgroup_state.processor_id = 0;
          workgroup_state.local_memory = local_memory;
          workgroup_state.local_memory_size = dispatch->local_memory_size;

          int ret = dispatch->function(dispatch->environment, &dispatch_state,
                                       &workgroup_state);
          if (IREE_UNLIKELY(ret != 0)) {
            return iree_make_status(IREE_STATUS_INTERNAL,
                                    "dispatch kernel returned non-zero (%d)",
                                    ret);
          }
          ++completed;
        }
        // Reload for next claim attempt.
        current = iree_atomic_load(tile_index, iree_memory_order_relaxed);
      }
      // On CAS failure, |current| is updated by compare_exchange_weak.
    }
    *out_tiles_completed = completed;
  }

  return iree_ok_status();
}

// Executes a FILL command. For multi-worker: claims the single tile via
// epoch-tagged CAS; exactly one worker performs the fill. Returns 1 if this
// worker performed the fill, 0 otherwise.
static uint32_t iree_hal_cmd_execute_fill(const iree_hal_cmd_fill_t* fill,
                                          void** binding_ptrs,
                                          iree_atomic_int64_t* tile_index,
                                          int32_t region_epoch,
                                          uint32_t worker_count) {
  if (worker_count > 1) {
    int64_t expected = (int64_t)region_epoch << 32;
    int64_t desired = expected | 1;
    if (!iree_atomic_compare_exchange_strong(tile_index, &expected, desired,
                                             iree_memory_order_relaxed,
                                             iree_memory_order_relaxed)) {
      return 0;
    }
  }

  uint8_t* target = (uint8_t*)binding_ptrs[fill->target_binding];

  if (fill->header.flags & IREE_HAL_CMD_FLAG_INDIRECT) {
    const void* params_buffer =
        binding_ptrs[fill->params.indirect.params_binding];
    const iree_hal_fill_params_t* params =
        (const iree_hal_fill_params_t*)((const uint8_t*)params_buffer +
                                        fill->params.indirect.params_offset);
    iree_hal_cmd_fill_pattern(target + params->target_offset, params->length,
                              params->pattern, fill->pattern_length);
  } else {
    iree_hal_cmd_fill_pattern(
        target + fill->params.direct.target_offset, fill->params.direct.length,
        fill->params.direct.pattern, fill->pattern_length);
  }

  return 1;
}

// Executes a COPY command. For multi-worker: claims the single tile via
// epoch-tagged CAS; exactly one worker performs the copy. Returns 1 if this
// worker performed the copy, 0 otherwise.
static uint32_t iree_hal_cmd_execute_copy(const iree_hal_cmd_copy_t* copy,
                                          void** binding_ptrs,
                                          iree_atomic_int64_t* tile_index,
                                          int32_t region_epoch,
                                          uint32_t worker_count) {
  if (worker_count > 1) {
    int64_t expected = (int64_t)region_epoch << 32;
    int64_t desired = expected | 1;
    if (!iree_atomic_compare_exchange_strong(tile_index, &expected, desired,
                                             iree_memory_order_relaxed,
                                             iree_memory_order_relaxed)) {
      return 0;
    }
  }

  const uint8_t* source = (const uint8_t*)binding_ptrs[copy->source_binding];
  uint8_t* target = (uint8_t*)binding_ptrs[copy->target_binding];

  if (copy->header.flags & IREE_HAL_CMD_FLAG_INDIRECT) {
    const void* params_buffer =
        binding_ptrs[copy->params.indirect.params_binding];
    const iree_hal_copy_params_t* params =
        (const iree_hal_copy_params_t*)((const uint8_t*)params_buffer +
                                        copy->params.indirect.params_offset);
    memcpy(target + params->target_offset, source + params->source_offset,
           (size_t)params->length);
  } else {
    memcpy(target + copy->params.direct.target_offset,
           source + copy->params.direct.source_offset,
           (size_t)copy->params.direct.length);
  }

  return 1;
}

// Executes an UPDATE command. Copies inline host data from .text to a device
// buffer. For multi-worker: claims the single tile via epoch-tagged CAS;
// exactly one worker performs the memcpy. Returns 1 if this worker performed
// the update, 0 otherwise.
static uint32_t iree_hal_cmd_execute_update(const iree_hal_cmd_update_t* update,
                                            void** binding_ptrs,
                                            iree_atomic_int64_t* tile_index,
                                            int32_t region_epoch,
                                            uint32_t worker_count) {
  if (worker_count > 1) {
    int64_t expected = (int64_t)region_epoch << 32;
    int64_t desired = expected | 1;
    if (!iree_atomic_compare_exchange_strong(tile_index, &expected, desired,
                                             iree_memory_order_relaxed,
                                             iree_memory_order_relaxed)) {
      return 0;
    }
  }

  uint8_t* target = (uint8_t*)binding_ptrs[update->target_binding];
  memcpy(target + update->target_offset, update->source_data,
         (size_t)update->length);

  return 1;
}

//===----------------------------------------------------------------------===//
// Region processing
//===----------------------------------------------------------------------===//

// Processes one barrier-delimited region cooperatively. Each worker scans
// the dispatch_count commands following the barrier and claims tiles from
// each via epoch-tagged CAS. Returns the total tiles completed by THIS worker.
//
// |region_epoch| is the global monotonically-increasing epoch used as the
// upper 32 bits of each tile_index CAS. This is NOT the region index — it
// spans block boundaries to prevent cross-block CAS collisions.
//
// On kernel error, reports the error to the context and returns immediately.
// Other workers will see the error flag and exit their loops.
static uint32_t iree_hal_cmd_block_processor_process_region(
    const iree_hal_cmd_barrier_t* barrier, int32_t region_epoch,
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_header_t** out_next_cmd) {
  uint32_t tiles_completed = 0;
  iree_hal_cmd_block_state_t* state = context->state;
  void** binding_ptrs = iree_hal_cmd_block_state_binding_ptrs(
      state, context->max_region_dispatch_count);

  // Advance past the barrier to the first work command.
  const iree_hal_cmd_header_t* cmd = iree_hal_cmd_next(&barrier->header);

  for (uint8_t d = 0; d < barrier->dispatch_count; ++d) {
    if (IREE_UNLIKELY(iree_hal_cmd_block_processor_has_error(context))) break;

    iree_atomic_int64_t* tile_idx =
        iree_hal_cmd_block_state_tile_index(state, cmd->dispatch_index);

    switch (cmd->opcode) {
      case IREE_HAL_CMD_DISPATCH: {
        uint32_t dispatch_tiles = 0;
        iree_status_t status = iree_hal_cmd_execute_dispatch_tiles(
            (const iree_hal_cmd_dispatch_t*)cmd, binding_ptrs, tile_idx,
            region_epoch, context->worker_count, &dispatch_tiles);
        if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
          iree_hal_cmd_block_processor_report_error(context, status);
          for (uint8_t remaining = d + 1; remaining < barrier->dispatch_count;
               ++remaining) {
            cmd = iree_hal_cmd_next(cmd);
          }
          *out_next_cmd = iree_hal_cmd_next(cmd);
          return tiles_completed;
        }
        tiles_completed += dispatch_tiles;
        break;
      }
      case IREE_HAL_CMD_FILL: {
        tiles_completed += iree_hal_cmd_execute_fill(
            (const iree_hal_cmd_fill_t*)cmd, binding_ptrs, tile_idx,
            region_epoch, context->worker_count);
        break;
      }
      case IREE_HAL_CMD_COPY: {
        tiles_completed += iree_hal_cmd_execute_copy(
            (const iree_hal_cmd_copy_t*)cmd, binding_ptrs, tile_idx,
            region_epoch, context->worker_count);
        break;
      }
      case IREE_HAL_CMD_UPDATE: {
        tiles_completed += iree_hal_cmd_execute_update(
            (const iree_hal_cmd_update_t*)cmd, binding_ptrs, tile_idx,
            region_epoch, context->worker_count);
        break;
      }
      default: {
        iree_hal_cmd_block_processor_report_error(
            context,
            iree_make_status(IREE_STATUS_INTERNAL,
                             "unexpected opcode %d in region", cmd->opcode));
        break;
      }
    }

    cmd = iree_hal_cmd_next(cmd);
  }

  *out_next_cmd = cmd;
  return tiles_completed;
}

//===----------------------------------------------------------------------===//
// Command stream navigation
//===----------------------------------------------------------------------===//

// Walks the command stream to find the barrier at the given region index.
// Returns NULL if not found. O(dispatches) per barrier — negligible for
// typical blocks with 1-5 regions and 1-20 dispatches per region.
static const iree_hal_cmd_barrier_t* iree_hal_cmd_find_barrier(
    const iree_hal_cmd_block_header_t* block, uint16_t region_index) {
  const iree_hal_cmd_header_t* cmd = iree_hal_cmd_block_commands(block);
  const uint8_t* stream_end = (const uint8_t*)cmd + block->used_bytes;
  uint16_t current_region = 0;
  while ((const uint8_t*)cmd < stream_end) {
    if (cmd->opcode == IREE_HAL_CMD_BARRIER) {
      if (current_region == region_index) {
        return (const iree_hal_cmd_barrier_t*)cmd;
      }
      // Skip past the barrier and its dispatch commands to find the next.
      const iree_hal_cmd_barrier_t* barrier =
          (const iree_hal_cmd_barrier_t*)cmd;
      cmd = iree_hal_cmd_next(&barrier->header);
      for (uint8_t d = 0; d < barrier->dispatch_count; ++d) {
        cmd = iree_hal_cmd_next(cmd);
      }
      current_region++;
    } else if (cmd->opcode == IREE_HAL_CMD_BRANCH ||
               cmd->opcode == IREE_HAL_CMD_RETURN) {
      break;
    } else {
      break;
    }
  }
  return NULL;
}

// Walks past all regions to find the block terminator (BRANCH or RETURN).
// Returns NULL if not found (malformed stream).
static const iree_hal_cmd_header_t* iree_hal_cmd_find_terminator(
    const iree_hal_cmd_block_header_t* block) {
  const iree_hal_cmd_header_t* cmd = iree_hal_cmd_block_commands(block);
  const uint8_t* stream_end = (const uint8_t*)cmd + block->used_bytes;
  while ((const uint8_t*)cmd < stream_end) {
    if (cmd->opcode == IREE_HAL_CMD_BARRIER) {
      const iree_hal_cmd_barrier_t* barrier =
          (const iree_hal_cmd_barrier_t*)cmd;
      cmd = iree_hal_cmd_next(&barrier->header);
      for (uint8_t d = 0; d < barrier->dispatch_count; ++d) {
        cmd = iree_hal_cmd_next(cmd);
      }
    } else if (cmd->opcode == IREE_HAL_CMD_BRANCH ||
               cmd->opcode == IREE_HAL_CMD_RETURN) {
      return cmd;
    } else {
      break;
    }
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Block and region initialization
//===----------------------------------------------------------------------===//

// Initializes .data for a new block. All counters are zero-initialized via
// memset; only binding pointers, active_region_index, tile_index epochs,
// and remaining_tiles require non-zero initialization.
//
// active_region_index is set to the first non-empty region (the first
// region with initial_remaining_tiles > 0). Workers skip empty regions;
// active_region_index only gates non-empty regions. If all regions are
// empty, active_region_index is set to region_count and the completer
// must handle the block terminator.
//
// remaining_tiles is set to the first active region's total tiles so
// that workers can begin completer election immediately.
static void iree_hal_cmd_block_processor_init_block(
    const iree_hal_cmd_block_header_t* block,
    iree_hal_cmd_block_processor_context_t* context) {
  // Find the first non-empty region before modifying state (we need
  // the epoch before setting tile_indices).
  const uint32_t* initial_tiles =
      iree_hal_cmd_block_initial_remaining_tiles(block);
  uint16_t first_active = 0;
  while (first_active < block->region_count &&
         initial_tiles[first_active] == 0) {
    first_active++;
  }

  // Assign a unique global epoch for this block's first active region.
  // The epoch monotonically increases across block and region transitions,
  // ensuring that stale workers from a previous block CAS-fail immediately
  // (their old epoch doesn't match the new tile_index epochs). This is
  // critical for correctness: without it, a stale worker in process_region
  // during a block transition could CAS-succeed on tile_indices that were
  // reset to the same per-block epoch (0), executing a kernel with partially
  // initialized binding_ptrs.
  int32_t block_epoch = context->next_epoch;
  context->next_epoch = block_epoch + 1;

  // Set tile_index epochs using the global epoch. ALL slots are set (not
  // just dispatch_count for the first region) because max_region_dispatch_count
  // covers the largest region in the block.
  int64_t epoch_value = (int64_t)block_epoch << 32;
  for (uint16_t i = 0; i < context->max_region_dispatch_count; ++i) {
    iree_atomic_store(iree_hal_cmd_block_state_tile_index(context->state, i),
                      epoch_value, iree_memory_order_relaxed);
  }

  // Set scheduling state for the new block.
  iree_atomic_store(&context->state->active_region_index, (int32_t)first_active,
                    iree_memory_order_relaxed);
  iree_atomic_store(&context->state->region_epoch, block_epoch,
                    iree_memory_order_relaxed);
  iree_atomic_store(&context->state->remaining_tiles, 0,
                    iree_memory_order_relaxed);
  if (first_active < block->region_count) {
    iree_atomic_store(&context->state->remaining_tiles,
                      (int32_t)initial_tiles[first_active],
                      iree_memory_order_relaxed);
  }

  // Resolve bindings via fixup. Workers only dereference binding_ptrs
  // after a successful CAS (which requires the new epoch), so stale
  // workers from the previous block never access these values.
  iree_hal_cmd_block_processor_resolve_bindings(
      block, context->state, context->max_region_dispatch_count,
      context->binding_table, context->binding_table_length);
}

// Prepares .data for the next region at a region transition. Called by the
// completer after remaining_tiles reaches zero, guaranteeing all tiles in
// the previous region have been executed.
//
// |region_epoch| is the global epoch for the new region (from
// context->next_epoch, pre-incremented by the caller). All tile_index
// entries are set to the new epoch with counter=0. All
// max_region_dispatch_count entries are set, not just the previous region's
// count, because the next region may have more dispatches.
//
// The caller must write active_region_index (relaxed) BEFORE calling this
// function. region_epoch is stored last with release semantics, publishing
// all prior writes (tile_indices, remaining_tiles, active_region_index).
// Workers acquire region_epoch to synchronize with this release.
static void iree_hal_cmd_block_processor_init_region(
    iree_hal_cmd_block_state_t* state, uint16_t max_region_dispatch_count,
    int32_t region_epoch, uint32_t next_remaining_tiles) {
  // Set tile_index epochs for the new region using the global epoch.
  int64_t epoch_value = (int64_t)region_epoch << 32;
  for (uint16_t i = 0; i < max_region_dispatch_count; ++i) {
    iree_atomic_store(iree_hal_cmd_block_state_tile_index(state, i),
                      epoch_value, iree_memory_order_relaxed);
  }
  // Set remaining tiles for the new region.
  iree_atomic_store(&state->remaining_tiles, (int32_t)next_remaining_tiles,
                    iree_memory_order_relaxed);
  // region_epoch is the synchronization point for intra-block region
  // transitions. Release semantics ensure all prior relaxed writes
  // (tile_indices, remaining_tiles, active_region_index) are visible to
  // workers that acquire region_epoch.
  iree_atomic_store(&state->region_epoch, region_epoch,
                    iree_memory_order_release);
}

//===----------------------------------------------------------------------===//
// Single-worker fast path
//===----------------------------------------------------------------------===//

// Executes the entire recording on a single thread. No atomics, no spinning,
// no synchronization — just a straight-line interpreter. This is the path
// for inline execution and small command buffers.
static iree_status_t iree_hal_cmd_block_processor_execute_single_worker(
    iree_hal_cmd_block_processor_context_t* context) {
  const iree_hal_cmd_block_header_t* block = context->recording->first_block;

  while (block) {
    iree_hal_cmd_block_processor_init_block(block, context);

    iree_hal_cmd_block_state_t* state = context->state;
    void** binding_ptrs = iree_hal_cmd_block_state_binding_ptrs(
        state, context->max_region_dispatch_count);

    const iree_hal_cmd_header_t* cmd = iree_hal_cmd_block_commands(block);
    const uint8_t* stream_end = (const uint8_t*)cmd + block->used_bytes;

    while ((const uint8_t*)cmd < stream_end) {
      switch (cmd->opcode) {
        case IREE_HAL_CMD_DISPATCH: {
          uint32_t tiles = 0;
          IREE_RETURN_IF_ERROR(iree_hal_cmd_execute_dispatch_tiles(
              (const iree_hal_cmd_dispatch_t*)cmd, binding_ptrs, NULL, 0, 1,
              &tiles));
          break;
        }
        case IREE_HAL_CMD_FILL: {
          iree_hal_cmd_execute_fill((const iree_hal_cmd_fill_t*)cmd,
                                    binding_ptrs, NULL, 0, 1);
          break;
        }
        case IREE_HAL_CMD_COPY: {
          iree_hal_cmd_execute_copy((const iree_hal_cmd_copy_t*)cmd,
                                    binding_ptrs, NULL, 0, 1);
          break;
        }
        case IREE_HAL_CMD_UPDATE: {
          iree_hal_cmd_execute_update((const iree_hal_cmd_update_t*)cmd,
                                      binding_ptrs, NULL, 0, 1);
          break;
        }
        case IREE_HAL_CMD_BARRIER: {
          // Single-worker: barriers are no-ops (execution is sequential).
          break;
        }
        case IREE_HAL_CMD_BRANCH: {
          block = ((const iree_hal_cmd_branch_t*)cmd)->target;
          goto next_block;
        }
        case IREE_HAL_CMD_RETURN: {
          return iree_ok_status();
        }
        default: {
          return iree_make_status(
              IREE_STATUS_INTERNAL, "unknown opcode %d at stream offset %zu",
              cmd->opcode,
              (size_t)((const uint8_t*)cmd - (const uint8_t*)block));
        }
      }
      cmd = iree_hal_cmd_next(cmd);
    }

    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "block command stream ended without BRANCH or RETURN");

  next_block:
    continue;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Multi-worker drain/return
//===----------------------------------------------------------------------===//
//
// N workers call drain() concurrently, each processing one region pass per
// call. The algorithm uses two levels of synchronization:
//   - Per-dispatch:  tile_indices[] (epoch-tagged CAS for work-stealing)
//   - Per-region:    remaining_tiles (completer election) +
//                    active_region_index (region gating)
//   - Per-block:     block_sequence (block transition detection)
//
// COMPLETER ELECTION
//
// After processing a region, each worker decrements remaining_tiles
// by the number of tiles it completed (fetch_sub with acq_rel). The worker
// whose decrement drives the count to zero becomes the completer. Workers
// with 0 tiles skip the election to avoid false positives.
//
// EPOCH-TAGGED TILE CLAIMING
//
// Each tile_index is a 64-bit atomic: global epoch in the upper 32 bits,
// tile counter in the lower 32 bits. Workers claim tiles via CAS,
// atomically validating the epoch and incrementing the counter. The epoch
// is a monotonically increasing counter across all region and block
// transitions (stored in context->next_epoch, published via
// state->region_epoch). This eliminates the need for an arrival barrier:
// if a stale worker (preempted during a region or block transition) tries
// to CAS with the old epoch, it fails with zero side effects. No tiles
// are stolen from the wrong region, even across block boundaries.
//
// REGION TRANSITIONS
//
// The completer initializes .data for the next region (set tile_index
// epochs, set remaining_tiles) and stores active_region_index with release
// semantics to unlock workers. Workers see the new region on their next
// drain() call via an acquire load of active_region_index.
//
// BLOCK TRANSITIONS (SEQLOCK)
//
// When all regions in a block are done, the completer handles the block
// terminator:
//   - BRANCH: uses a seqlock on block_sequence to protect init_block.
//     The completer increments block_sequence to even (in progress),
//     initializes the next block's .data via init_block, then increments
//     to odd (ready). Workers check parity: even = bail immediately,
//     odd = proceed. A re-check after reading state detects transitions
//     that started between the initial read and state access.
//   - RETURN: sets completed. All subsequent drain() calls return
//     completed=true.
//
// The seqlock is necessary because .data is reused across blocks. Without
// it, a worker re-entering drain during init_block could see partially-
// initialized state: active_region_index reset to 0 (new block) while
// the block pointer is still from the old block. The global epoch on
// tile_indices protects against stale CAS (a worker from the old block
// can't accidentally claim tiles in the new block), but the seqlock
// additionally protects non-epoch-guarded reads (block pointer, region
// count, binding_ptrs mid-resolution).
//
// Empty block chains (all regions empty) are followed iteratively by the
// completer while block_sequence remains even (in progress). Workers stay
// out until a block with work is found and the ready increment fires.
//
// DEADLOCK FREEDOM
//
// There are no internal spin loops. The completer advances immediately
// after being elected (remaining_tiles reaches 0). Non-completer workers
// simply return and the caller decides whether to yield, scan other
// contexts, or retry. Stale workers' CAS failures are safe and immediate.
// Workers that observe an even block_sequence (transition in progress)
// bail and retry — no spinning, just return to the caller.
//
// INVARIANTS
//
//   I1: active_region_index is monotonically non-decreasing within a block.
//   I2: tile_index epochs, remaining_tiles, and active_region_index are set
//       BEFORE region_epoch (the release point). Workers acquire
//       region_epoch, guaranteeing all prior writes are visible.
//   I3: Workers use the acquired region_epoch for CAS validation. If the
//       epoch is stale, CAS fails harmlessly (epoch mismatch on tile_indices
//       that were reset to the new epoch). If fresh, all state is consistent.
//   I4: The epoch tag on each CAS validates that the worker is operating on
//       the correct region. Stale workers fail harmlessly (epoch mismatch).
//   I5: Empty regions are skipped by init_block and the completer.
//   I6: block_sequence uses seqlock parity: odd = ready, even = transition
//       in progress. Workers check parity on entry and re-check after
//       reading state. All init_block writes are bracketed by the even/odd
//       increments (release semantics), ensuring visibility.

// Handles a completed region: advances to the next non-empty region or
// processes the block terminator (BRANCH/RETURN).
static void iree_hal_cmd_block_processor_handle_region_completion(
    iree_hal_cmd_block_processor_context_t* context,
    const iree_hal_cmd_block_header_t* block,
    const iree_hal_cmd_barrier_t* completed_barrier,
    int32_t completed_region_index,
    iree_hal_cmd_block_processor_drain_result_t* out_result) {
  iree_hal_cmd_block_state_t* state = context->state;
  const uint32_t* initial_tiles =
      iree_hal_cmd_block_initial_remaining_tiles(block);

  // Find next non-empty region (skip empty regions with 0 tiles).
  int32_t next_region = completed_region_index + 1;
  while (next_region < (int32_t)block->region_count &&
         initial_tiles[next_region] == 0) {
    next_region++;
  }

  if (next_region < (int32_t)block->region_count) {
    // Another region in this block. Assign a new global epoch and
    // initialize .data for the next region.
    int32_t region_epoch = context->next_epoch;
    context->next_epoch = region_epoch + 1;
    // Write active_region_index BEFORE init_region. init_region's release
    // on region_epoch publishes this write along with tile_indices and
    // remaining_tiles.
    iree_atomic_store(&state->active_region_index, next_region,
                      iree_memory_order_relaxed);
    iree_hal_cmd_block_processor_init_region(
        state, context->max_region_dispatch_count, region_epoch,
        initial_tiles[next_region]);
    return;
  }

  // All regions in this block are done. Handle the terminator.
  const iree_hal_cmd_header_t* terminator = iree_hal_cmd_find_terminator(block);
  if (!terminator) {
    iree_hal_cmd_block_processor_report_error(
        context, iree_make_status(IREE_STATUS_INTERNAL,
                                  "block ended without BRANCH or RETURN"));
    out_result->completed = true;
    return;
  }

  if (terminator->opcode == IREE_HAL_CMD_RETURN) {
    iree_atomic_store(&context->completed, 1, iree_memory_order_release);
    out_result->completed = true;
    return;
  }

  if (terminator->opcode == IREE_HAL_CMD_BRANCH) {
    // Signal block transition in progress (odd → even). Workers seeing the
    // even value bail out of drain immediately, preventing them from reading
    // state while init_block modifies it. This is the writer-side of a
    // seqlock: even = transition in progress, odd = ready.
    iree_atomic_fetch_add(&context->block_sequence, 1,
                          iree_memory_order_release);

    // Follow BRANCH to next block. Handle chains of empty blocks iteratively
    // to avoid stack overflow (pathological case: many blocks all with empty
    // predicated regions).
    const iree_hal_cmd_block_header_t* next_block =
        ((const iree_hal_cmd_branch_t*)terminator)->target;
    while (next_block) {
      iree_atomic_store(&context->current_block, (intptr_t)next_block,
                        iree_memory_order_relaxed);
      iree_hal_cmd_block_processor_init_block(next_block, context);

      // Check if the block has any work (init_block sets active_region_index
      // to the first non-empty region, or region_count if all are empty).
      int32_t first_active = iree_atomic_load(&state->active_region_index,
                                              iree_memory_order_relaxed);
      if (first_active < (int32_t)next_block->region_count) {
        // Block has work. Signal ready (even → odd).
        iree_atomic_fetch_add(&context->block_sequence, 1,
                              iree_memory_order_release);
        return;
      }

      // Entirely empty block. Follow its terminator.
      const iree_hal_cmd_header_t* next_terminator =
          iree_hal_cmd_find_terminator(next_block);
      if (!next_terminator || next_terminator->opcode == IREE_HAL_CMD_RETURN) {
        iree_atomic_store(&context->completed, 1, iree_memory_order_release);
        out_result->completed = true;
        return;
      }
      if (next_terminator->opcode == IREE_HAL_CMD_BRANCH) {
        next_block = ((const iree_hal_cmd_branch_t*)next_terminator)->target;
      } else {
        iree_hal_cmd_block_processor_report_error(
            context, iree_make_status(IREE_STATUS_INTERNAL,
                                      "unknown terminator opcode %d",
                                      next_terminator->opcode));
        out_result->completed = true;
        return;
      }
    }
    // Fell off the block chain (next_block == NULL). This should not happen
    // in a well-formed recording — block chains always end with RETURN.
    iree_hal_cmd_block_processor_report_error(
        context,
        iree_make_status(IREE_STATUS_INTERNAL, "block chain ends with NULL"));
    out_result->completed = true;
    return;
  }

  // Unknown terminator.
  iree_hal_cmd_block_processor_report_error(
      context,
      iree_make_status(IREE_STATUS_INTERNAL, "unknown terminator opcode %d",
                       terminator->opcode));
  out_result->completed = true;
}

// Multi-worker drain: one pass through the current active region.
static void iree_hal_cmd_block_processor_drain_multi_worker(
    iree_hal_cmd_block_processor_context_t* context, uint32_t worker_index,
    iree_hal_cmd_block_processor_worker_state_t* worker_state,
    iree_hal_cmd_block_processor_drain_result_t* out_result) {
  // Check for completion (error or RETURN reached).
  if (iree_atomic_load(&context->completed, iree_memory_order_acquire)) {
    out_result->completed = true;
    return;
  }

  // Seqlock reader: block_sequence uses odd/even parity to gate block
  // transitions. Odd = ready (workers may proceed), even = transition in
  // progress (workers must bail). The acquire pairs with the completer's
  // release, ensuring all .data writes from init_block are visible when
  // the worker sees an odd (ready) value.
  int32_t block_sequence =
      iree_atomic_load(&context->block_sequence, iree_memory_order_acquire);
  if ((block_sequence & 1) == 0) {
    // Block transition in progress. Bail and retry on next drain().
    return;
  }
  if (block_sequence != worker_state->block_sequence) {
    worker_state->block_sequence = block_sequence;
  }

  const iree_hal_cmd_block_header_t* block =
      (const iree_hal_cmd_block_header_t*)iree_atomic_load(
          &context->current_block, iree_memory_order_relaxed);
  iree_hal_cmd_block_state_t* state = context->state;

  // Read global epoch and active region index. region_epoch is the
  // synchronization point for intra-block region transitions: the completer
  // stores it with release after writing tile_indices, remaining_tiles,
  // and active_region_index. Acquiring region_epoch guarantees all those
  // writes are visible.
  //
  // For block transitions, block_sequence (acquired above) provides
  // ordering — region_epoch is redundantly visible but harmless.
  //
  // Critical ordering: region_epoch MUST be acquired BEFORE
  // active_region_index is loaded. If the order were reversed, a worker
  // could acquire a stale active_region_index but load a fresh
  // region_epoch (unordered relaxed load), then CAS-succeed on tile_indices
  // that were reset for the next region — re-executing old dispatches with
  // the new epoch and corrupting the next region's tile claims.
  int32_t region_epoch =
      iree_atomic_load(&state->region_epoch, iree_memory_order_acquire);
  int32_t active_region =
      iree_atomic_load(&state->active_region_index, iree_memory_order_relaxed);

  // All regions in this block are done. The completer is handling the
  // block terminator (or has already set completed).
  if (active_region >= (int32_t)block->region_count) {
    if (iree_atomic_load(&context->completed, iree_memory_order_acquire)) {
      out_result->completed = true;
    }
    return;
  }

  // Seqlock validation: re-read block_sequence to detect block transitions
  // that started between our initial read and here. If the value changed,
  // our reads of current_block, active_region_index, etc. may reflect
  // partially-initialized state from init_block. Bail and retry.
  int32_t block_sequence_recheck =
      iree_atomic_load(&context->block_sequence, iree_memory_order_acquire);
  if (block_sequence_recheck != block_sequence) {
    return;
  }

  // Check for errors before starting work.
  if (iree_hal_cmd_block_processor_has_error(context)) {
    out_result->completed = true;
    return;
  }

  // Find the barrier for the active region by walking the command stream.
  // This is O(commands before the barrier) which is negligible: typical
  // blocks have 1-5 regions with 1-20 dispatches each.
  const iree_hal_cmd_barrier_t* barrier =
      iree_hal_cmd_find_barrier(block, (uint16_t)active_region);
  if (!barrier) {
    iree_hal_cmd_block_processor_report_error(
        context, iree_make_status(IREE_STATUS_INTERNAL,
                                  "failed to find barrier for region %d",
                                  active_region));
    out_result->completed = true;
    return;
  }

  // Process the region's work commands cooperatively.
  const iree_hal_cmd_header_t* next_cmd = NULL;
  uint32_t my_tiles = iree_hal_cmd_block_processor_process_region(
      barrier, region_epoch, context, &next_cmd);
  out_result->tiles_executed = my_tiles;

  // Completer election via remaining_tiles countdown.
  // fetch_sub with acq_rel: release publishes this worker's tile writes,
  // acquire lets the completer see all prior workers' writes.
  // Workers with 0 tiles skip the election to avoid false positives
  // (fetch_sub(0) could see old == 0 when total_tiles > 0 hasn't been
  // decremented to 0 yet).
  bool is_completer = false;
  if (my_tiles > 0) {
    int32_t old = iree_atomic_fetch_sub(
        &state->remaining_tiles, (int32_t)my_tiles, iree_memory_order_acq_rel);
    is_completer = ((old - (int32_t)my_tiles) == 0);
  }

  if (!is_completer) return;

  // === COMPLETER ===
  // No arrival wait needed: the epoch tag on each tile_index CAS ensures
  // stale workers from the previous region fail harmlessly. The completer
  // can safely reset tile_indices immediately because:
  //   - All tiles have been executed (remaining_tiles reached 0).
  //   - All workers have either returned or are between CAS attempts.
  //   - Any CAS with the old epoch will fail after the reset.

  // Handle the completed region: advance to the next region or process
  // the block terminator.
  iree_hal_cmd_block_processor_handle_region_completion(
      context, block, barrier, active_region, out_result);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Sets up the context for multi-worker execution by finding the first block
// with actual work. Follows empty block chains (blocks where all regions
// have 0 tiles) iteratively until a block with work is found or RETURN is
// reached. If no work exists, sets completed=true so drain() returns
// immediately.
static void iree_hal_cmd_block_processor_setup_first_block(
    iree_hal_cmd_block_processor_context_t* context) {
  const iree_hal_cmd_block_header_t* block = context->recording->first_block;
  while (block) {
    iree_atomic_store(&context->current_block, (intptr_t)block,
                      iree_memory_order_relaxed);
    iree_hal_cmd_block_processor_init_block(block, context);

    // Check if the block has any work (init_block sets active_region_index
    // to the first non-empty region, or leaves it at 0 which may equal
    // region_count for blocks with 0 regions).
    int32_t first_active = iree_atomic_load(
        &context->state->active_region_index, iree_memory_order_relaxed);
    if (first_active < (int32_t)block->region_count) {
      // Block has work. Set sequence so workers can begin.
      iree_atomic_store(&context->block_sequence, 1, iree_memory_order_release);
      return;
    }

    // Entirely empty block. Follow its terminator.
    const iree_hal_cmd_header_t* terminator =
        iree_hal_cmd_find_terminator(block);
    if (!terminator || terminator->opcode == IREE_HAL_CMD_RETURN) {
      iree_atomic_store(&context->completed, 1, iree_memory_order_release);
      return;
    }
    if (terminator->opcode == IREE_HAL_CMD_BRANCH) {
      block = ((const iree_hal_cmd_branch_t*)terminator)->target;
    } else {
      iree_hal_cmd_block_processor_report_error(
          context,
          iree_make_status(IREE_STATUS_INTERNAL, "unknown terminator opcode %d",
                           terminator->opcode));
      return;
    }
  }
  // Fell off the block chain (NULL target). Should not happen in a
  // well-formed recording.
  iree_atomic_store(&context->completed, 1, iree_memory_order_release);
}

void iree_hal_cmd_block_processor_context_initialize(
    iree_hal_cmd_block_processor_context_t* out_context,
    const iree_hal_cmd_block_recording_t* recording,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length, iree_hal_cmd_block_state_t* state,
    iree_host_size_t state_size) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->recording = recording;
  out_context->binding_table = binding_table;
  out_context->binding_table_length = binding_table_length;
  out_context->state = state;
  out_context->state_size = state_size;
  out_context->worker_count = 1;
  out_context->max_region_dispatch_count = recording->max_region_dispatch_count;
  out_context->max_total_binding_count = recording->max_total_binding_count;
  if (recording->first_block) {
    iree_atomic_store(&out_context->current_block,
                      (intptr_t)recording->first_block,
                      iree_memory_order_relaxed);
  }
}

iree_status_t iree_hal_cmd_block_processor_context_allocate(
    const iree_hal_cmd_block_recording_t* recording,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length, uint32_t worker_count,
    iree_allocator_t allocator,
    iree_hal_cmd_block_processor_context_t** out_context) {
  *out_context = NULL;
  if (!recording->first_block) return iree_ok_status();
  if (worker_count == 0) worker_count = 1;

  // Allocate .data sized to the highwater mark across all blocks.
  const iree_host_size_t state_size = iree_hal_cmd_block_state_size(
      recording->max_region_dispatch_count, recording->max_total_binding_count);

  // Allocate context + state in one allocation with cache line alignment
  // so the iree_alignas(64) atomic fields land at proper boundaries.
  const iree_host_size_t context_size =
      iree_host_align(sizeof(iree_hal_cmd_block_processor_context_t),
                      iree_hardware_destructive_interference_size);
  const iree_host_size_t total_size = context_size + state_size;

  void* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      allocator, total_size, iree_hardware_destructive_interference_size,
      /*offset=*/0, &allocation));

  iree_hal_cmd_block_processor_context_t* context =
      (iree_hal_cmd_block_processor_context_t*)allocation;
  iree_hal_cmd_block_state_t* state =
      (iree_hal_cmd_block_state_t*)((uint8_t*)allocation + context_size);
  iree_hal_cmd_block_processor_context_initialize(
      context, recording, binding_table, binding_table_length, state,
      state_size);
  context->worker_count = worker_count;

  if (worker_count > 1) {
    // Multi-worker: find the first block with work (following empty block
    // chains) and initialize .data. If no work exists, sets completed=true
    // so drain() returns immediately.
    iree_hal_cmd_block_processor_setup_first_block(context);
  }

  *out_context = context;
  return iree_ok_status();
}

void iree_hal_cmd_block_processor_drain(
    iree_hal_cmd_block_processor_context_t* context, uint32_t worker_index,
    iree_hal_cmd_block_processor_worker_state_t* worker_state,
    iree_hal_cmd_block_processor_drain_result_t* out_result) {
  out_result->tiles_executed = 0;
  out_result->completed = false;

  if (!context) {
    out_result->completed = true;
    return;
  }

  if (context->worker_count == 1) {
    // Single-worker fast path: no atomics, no synchronization.
    iree_status_t status =
        iree_hal_cmd_block_processor_execute_single_worker(context);
    if (!iree_status_is_ok(status)) {
      iree_hal_cmd_block_processor_report_error(context, status);
    }
    out_result->completed = true;
    return;
  }

  iree_hal_cmd_block_processor_drain_multi_worker(context, worker_index,
                                                  worker_state, out_result);
}

iree_status_t iree_hal_cmd_block_processor_context_consume_result(
    iree_hal_cmd_block_processor_context_t* context) {
  if (!context) return iree_ok_status();
  intptr_t error = iree_atomic_exchange(&context->error_status, 0,
                                        iree_memory_order_acquire);
  return (iree_status_t)error;
}

void iree_hal_cmd_block_processor_context_free(
    iree_hal_cmd_block_processor_context_t* context,
    iree_allocator_t allocator) {
  if (!context) return;
  iree_allocator_free_aligned(allocator, context);
}
