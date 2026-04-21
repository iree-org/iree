// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_builder.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Block pool helpers
//===----------------------------------------------------------------------===//

// Returns the iree_arena_block_t footer for a block whose data starts at
// |block_data|. The footer is at the end of the allocation, immediately
// after the usable area.
static iree_arena_block_t* iree_hal_cmd_block_data_to_arena_block(
    iree_arena_block_pool_t* block_pool, void* block_data) {
  return (iree_arena_block_t*)((uint8_t*)block_data +
                               block_pool->usable_block_size);
}

//===----------------------------------------------------------------------===//
// Recording output
//===----------------------------------------------------------------------===//

void iree_hal_cmd_block_recording_release(
    iree_hal_cmd_block_recording_t* recording) {
  if (!recording->first_block) return;

  // Walk our block chain and reconstruct the arena_block_t chain for release.
  iree_arena_block_t* arena_head = NULL;
  iree_arena_block_t* arena_tail = NULL;
  iree_hal_cmd_block_header_t* block = recording->first_block;
  while (block) {
    iree_arena_block_t* arena_block =
        iree_hal_cmd_block_data_to_arena_block(recording->block_pool, block);
    if (!arena_head) {
      arena_head = arena_block;
    } else {
      arena_tail->next = arena_block;
    }
    arena_tail = arena_block;
    block = block->next_block;
  }
  arena_tail->next = NULL;

  iree_arena_block_pool_release(recording->block_pool, arena_head, arena_tail);
  recording->first_block = NULL;
  recording->block_count = 0;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

void iree_hal_cmd_block_builder_initialize(
    iree_arena_block_pool_t* block_pool,
    iree_hal_cmd_block_builder_t* out_builder) {
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->block_pool = block_pool;
}

void iree_hal_cmd_block_builder_deinitialize(
    iree_hal_cmd_block_builder_t* builder) {
  if (!builder->block_pool) return;

  // Release the current in-progress block (if any).
  if (builder->current_header) {
    iree_arena_block_t* arena_block = iree_hal_cmd_block_data_to_arena_block(
        builder->block_pool, builder->current_header);
    arena_block->next = NULL;
    iree_arena_block_pool_release(builder->block_pool, arena_block,
                                  arena_block);
    builder->current_header = NULL;
  }

  // Release all finalized blocks.
  if (builder->first_block) {
    iree_hal_cmd_block_recording_t recording = {
        .block_pool = builder->block_pool,
        .first_block = builder->first_block,
    };
    iree_hal_cmd_block_recording_release(&recording);
    builder->first_block = NULL;
    builder->last_block = NULL;
  }

  builder->block_pool = NULL;
}

//===----------------------------------------------------------------------===//
// Internal block management
//===----------------------------------------------------------------------===//

// Acquires a new block from the pool and sets it up as the current block.
// Writes the initial region header at the start of the command stream.
static iree_status_t iree_hal_cmd_block_builder_begin_block(
    iree_hal_cmd_block_builder_t* builder) {
  IREE_ASSERT(!builder->current_header,
              "must finalize current block before beginning a new one");

  // Acquire a raw block from the pool.
  iree_arena_block_t* arena_block = NULL;
  void* block_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_block_pool_acquire(
      builder->block_pool, &arena_block, &block_data));

  // The block header occupies the first bytes.
  iree_hal_cmd_block_header_t* header =
      (iree_hal_cmd_block_header_t*)block_data;
  memset(header, 0, sizeof(*header));
  header->block_size = (uint16_t)builder->block_pool->usable_block_size;

  // Set up cursors.
  builder->current_header = header;
  builder->cmd_cursor = (uint8_t*)header + sizeof(iree_hal_cmd_block_header_t);
  builder->block_end =
      (uint8_t*)header + builder->block_pool->usable_block_size;
  builder->fixup_cursor = builder->block_end;

  // Reset per-block accounting.
  builder->region_count = 0;
  builder->max_region_dispatch_count = 0;
  builder->total_binding_count = 0;
  builder->total_dispatch_count = 0;
  builder->fixup_count = 0;

  // Reset region-local tracking.
  builder->region_dispatch_count = 0;
  builder->current_region_tiles = 0;
  builder->current_barrier = NULL;

  ++builder->block_count;

  // Write the entry barrier for the first region.
  iree_hal_cmd_barrier_t* entry_barrier =
      (iree_hal_cmd_barrier_t*)builder->cmd_cursor;
  memset(entry_barrier, 0, sizeof(*entry_barrier));
  entry_barrier->header.opcode = IREE_HAL_CMD_BARRIER;
  entry_barrier->header.flags = IREE_HAL_CMD_FLAG_NONE;
  entry_barrier->header.size_qwords = sizeof(iree_hal_cmd_barrier_t) / 8;
  entry_barrier->header.dispatch_index = 0;
  builder->current_barrier = entry_barrier;
  builder->cmd_cursor += sizeof(iree_hal_cmd_barrier_t);
  builder->region_count = 1;

  return iree_ok_status();
}

// Finalizes the current region: patches the barrier's dispatch_count and
// saves the tile count to the scratch buffer.
static void iree_hal_cmd_block_builder_finalize_region(
    iree_hal_cmd_block_builder_t* builder) {
  IREE_ASSERT(builder->current_barrier);
  IREE_ASSERT(builder->region_count > 0);
  IREE_ASSERT(builder->region_count <= IREE_HAL_CMD_MAX_REGIONS_PER_BLOCK);
  IREE_ASSERT(builder->region_dispatch_count <= UINT8_MAX);

  // Patch the barrier with the actual dispatch count for this region.
  builder->current_barrier->dispatch_count =
      (uint8_t)builder->region_dispatch_count;
  // Wake budget: 0 means the processor determines it dynamically based
  // on tile counts and available workers.
  builder->current_barrier->wake_budget = 0;

  // Save tile count for this region. The scratch array is indexed by
  // region_count - 1 (the current region is the last one).
  builder->region_tiles_scratch[builder->region_count - 1] =
      builder->current_region_tiles;

  // Update per-block highwater mark.
  if (builder->region_dispatch_count > builder->max_region_dispatch_count) {
    builder->max_region_dispatch_count = builder->region_dispatch_count;
  }

  builder->current_barrier = NULL;
}

// Finalizes the current block: patches the block header, writes the
// initial_remaining_tiles at the end, and adds the block to the chain.
//
// The fixup entries currently occupy [fixup_cursor, block_end). They need
// to shift down by region_count * sizeof(uint32_t) to make room for the
// initial_remaining_tiles at the very end of the block.
static void iree_hal_cmd_block_builder_finalize_block(
    iree_hal_cmd_block_builder_t* builder) {
  IREE_ASSERT(builder->current_header);

  // The region should already be finalized by the caller (barrier/end/split
  // all call finalize_region before finalize_block).
  IREE_ASSERT(!builder->current_barrier);

  iree_hal_cmd_block_header_t* header = builder->current_header;

  // Write the block header.
  header->used_bytes = (uint16_t)(builder->cmd_cursor - (uint8_t*)(header + 1));
  header->region_count = builder->region_count;
  header->max_region_dispatch_count = builder->max_region_dispatch_count;
  header->total_binding_count = builder->total_binding_count;
  header->fixup_count = builder->fixup_count;
  header->total_dispatch_count = builder->total_dispatch_count;
  // block_size was set at block acquisition.

  // Shift fixups down to make room for initial_remaining_tiles at the end.
  // The tile reservation is rounded up to fixup alignment so the fixup table
  // always starts at a properly aligned address.
  const iree_host_size_t tile_reservation =
      iree_hal_cmd_block_tile_reservation_size(builder->region_count);
  const iree_host_size_t fixup_bytes =
      (iree_host_size_t)builder->fixup_count * sizeof(iree_hal_cmd_fixup_t);
  if (fixup_bytes > 0 && tile_reservation > 0) {
    memmove(builder->fixup_cursor - tile_reservation, builder->fixup_cursor,
            fixup_bytes);
  }

  // Write initial_remaining_tiles packed at the very end of the block.
  const iree_host_size_t tile_bytes =
      (iree_host_size_t)builder->region_count * sizeof(uint32_t);
  uint32_t* tiles = (uint32_t*)(builder->block_end - tile_bytes);
  memcpy(tiles, builder->region_tiles_scratch, tile_bytes);

  // Update global highwater marks.
  if (builder->max_region_dispatch_count >
      builder->global_max_region_dispatch_count) {
    builder->global_max_region_dispatch_count =
        builder->max_region_dispatch_count;
  }
  if (builder->total_binding_count > builder->global_max_total_binding_count) {
    builder->global_max_total_binding_count = builder->total_binding_count;
  }

  // Append to block chain.
  header->next_block = NULL;
  if (builder->last_block) {
    builder->last_block->next_block = header;
  } else {
    builder->first_block = header;
  }
  builder->last_block = header;

  builder->current_header = NULL;
  builder->cmd_cursor = NULL;
  builder->fixup_cursor = NULL;
  builder->block_end = NULL;
}

// Returns the number of usable bytes remaining between the forward and
// backward cursors, accounting for reserved space at the end for the
// initial_remaining_tiles that will be written at finalization.
static iree_host_size_t iree_hal_cmd_block_builder_remaining(
    const iree_hal_cmd_block_builder_t* builder) {
  // Reserve space for tiles at the end of the block, +1 for a potential new
  // region from a barrier/split. Uses the padded reservation to ensure
  // fixup alignment.
  const iree_host_size_t tile_reservation =
      iree_hal_cmd_block_tile_reservation_size(builder->region_count + 1);
  iree_host_size_t backward =
      (iree_host_size_t)(builder->fixup_cursor - builder->cmd_cursor);
  if (backward < tile_reservation) return 0;
  return backward - tile_reservation;
}

// Splits the current block by inserting a BRANCH to a new block.
// The current region is finalized before the split and continued in the
// new block with a fresh region header.
static iree_status_t iree_hal_cmd_block_builder_split_block(
    iree_hal_cmd_block_builder_t* builder) {
  IREE_ASSERT(builder->current_header);

  // Finalize the current region (whatever tiles have accumulated so far
  // belong to this block's portion of the region).
  iree_hal_cmd_block_builder_finalize_region(builder);

  // Write BRANCH command. The target will be set after we acquire the new
  // block.
  iree_hal_cmd_branch_t* branch = (iree_hal_cmd_branch_t*)builder->cmd_cursor;
  memset(branch, 0, sizeof(*branch));
  branch->header.opcode = IREE_HAL_CMD_BRANCH;
  branch->header.flags = IREE_HAL_CMD_FLAG_NONE;
  branch->header.size_qwords = sizeof(iree_hal_cmd_branch_t) / 8;
  branch->header.dispatch_index = 0;
  builder->cmd_cursor += sizeof(iree_hal_cmd_branch_t);

  // Finalize and chain the current block.
  iree_hal_cmd_block_header_t* old_header = builder->current_header;
  iree_hal_cmd_block_builder_finalize_block(builder);

  // Acquire the new block.
  IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_begin_block(builder));

  // Patch the BRANCH target to point at the new block.
  branch->target = builder->current_header;
  (void)old_header;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Recording session
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cmd_block_builder_begin(
    iree_hal_cmd_block_builder_t* builder) {
  if (IREE_UNLIKELY(builder->current_header)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "recording already in progress; call end() before begin()");
  }
  return iree_hal_cmd_block_builder_begin_block(builder);
}

iree_status_t iree_hal_cmd_block_builder_end(
    iree_hal_cmd_block_builder_t* builder,
    iree_hal_cmd_block_recording_t* out_recording) {
  memset(out_recording, 0, sizeof(*out_recording));
  if (IREE_UNLIKELY(!builder->current_header)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no recording in progress; call begin() first");
  }

  // Finalize the last region.
  iree_hal_cmd_block_builder_finalize_region(builder);

  // Write RETURN command.
  iree_hal_cmd_header_t* ret = (iree_hal_cmd_header_t*)builder->cmd_cursor;
  ret->opcode = IREE_HAL_CMD_RETURN;
  ret->flags = IREE_HAL_CMD_FLAG_NONE;
  ret->size_qwords = 1;  // 8 bytes (padded from 4-byte header).
  ret->dispatch_index = 0;
  builder->cmd_cursor += 8;

  // Finalize and chain the last block.
  iree_hal_cmd_block_builder_finalize_block(builder);

  // Transfer ownership to the recording.
  out_recording->block_pool = builder->block_pool;
  out_recording->first_block = builder->first_block;
  out_recording->block_count = builder->block_count;
  out_recording->max_region_dispatch_count =
      builder->global_max_region_dispatch_count;
  out_recording->max_total_binding_count =
      builder->global_max_total_binding_count;

  // Clear builder state (it no longer owns the blocks).
  builder->first_block = NULL;
  builder->last_block = NULL;
  builder->block_count = 0;
  builder->global_max_region_dispatch_count = 0;
  builder->global_max_total_binding_count = 0;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Command recording
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cmd_block_builder_append_cmd(
    iree_hal_cmd_block_builder_t* builder, iree_hal_cmd_opcode_t opcode,
    iree_hal_cmd_flags_t flags, iree_host_size_t cmd_bytes,
    uint16_t fixup_count, uint16_t binding_count, uint32_t tile_count,
    void** out_cmd, iree_hal_cmd_fixup_t** out_fixups) {
  IREE_ASSERT(out_cmd);
  *out_cmd = NULL;
  if (out_fixups) *out_fixups = NULL;

  if (IREE_UNLIKELY(!builder->current_header)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no recording in progress; call begin() first");
  }
  if (IREE_UNLIKELY(cmd_bytes % 8 != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command size (%" PRIhsz ") must be a multiple of 8", cmd_bytes);
  }
  if (IREE_UNLIKELY(cmd_bytes > 255 * 8)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command size (%" PRIhsz ") exceeds maximum (2040 bytes)", cmd_bytes);
  }
  if (IREE_UNLIKELY(opcode == IREE_HAL_CMD_BARRIER ||
                    opcode == IREE_HAL_CMD_BRANCH ||
                    opcode == IREE_HAL_CMD_RETURN)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "BARRIER/BRANCH/RETURN are not valid for append_cmd; "
        "use barrier() and end()");
  }

  // Total space needed for this command.
  const iree_host_size_t fixup_bytes =
      (iree_host_size_t)fixup_count * sizeof(iree_hal_cmd_fixup_t);
  const iree_host_size_t total_needed = cmd_bytes + fixup_bytes;

  // Region-local work command indices and barrier dispatch counts are 8-bit.
  // Split before appending the 256th command so the encoded count cannot wrap
  // while the tile count continues accumulating.
  if (builder->region_dispatch_count >= UINT8_MAX) {
    IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_split_block(builder));
  }

  // Check if we need to split. Reserve extra space for the BRANCH (16 bytes)
  // we'd need to insert if splitting, so we're never stuck without room for
  // the terminator. Also reserve for the entry barrier that follows a split
  // (8 bytes).
  const iree_host_size_t split_overhead =
      sizeof(iree_hal_cmd_branch_t) + sizeof(iree_hal_cmd_barrier_t);
  iree_host_size_t available = iree_hal_cmd_block_builder_remaining(builder);

  if (total_needed + split_overhead > available) {
    // Also check if the region limit would be exceeded.
    IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_split_block(builder));
    available = iree_hal_cmd_block_builder_remaining(builder);
    // After split, the command must fit. If it still doesn't, the command
    // itself is larger than the block capacity (a genuine error).
    if (IREE_UNLIKELY(total_needed + split_overhead > available)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "command (%" PRIhsz " bytes + %" PRIhsz
                              " fixup bytes) exceeds block capacity (%" PRIhsz
                              " bytes available)",
                              cmd_bytes, fixup_bytes, available);
    }
  }

  // Write the command at the forward cursor.
  iree_hal_cmd_header_t* header = (iree_hal_cmd_header_t*)builder->cmd_cursor;
  memset(header, 0, cmd_bytes);
  header->opcode = (uint8_t)opcode;
  header->flags = flags;
  header->size_qwords = (uint8_t)(cmd_bytes / 8);

  // All work commands (DISPATCH, FILL, COPY, UPDATE) get a region-local
  // dispatch index for their tile_index entry in .data. The multi-worker
  // processor uses this to distribute tiles across workers for all command
  // types.
  header->dispatch_index = (uint8_t)builder->region_dispatch_count;
  builder->region_dispatch_count++;
  builder->total_dispatch_count++;

  builder->cmd_cursor += cmd_bytes;

  // Reserve fixup space at the backward cursor. The caller fills the entries
  // directly in-place — no copy. Fixups within a command are stored in
  // forward order (out_fixups[0] at the lowest address).
  if (fixup_count > 0) {
    builder->fixup_cursor -= fixup_bytes;
    if (out_fixups) {
      *out_fixups = (iree_hal_cmd_fixup_t*)builder->fixup_cursor;
    }
  }
  builder->fixup_count += fixup_count;

  // Update per-block accounting.
  builder->total_binding_count += binding_count;
  builder->current_region_tiles += tile_count;

  *out_cmd = header;
  return iree_ok_status();
}

void iree_hal_cmd_block_builder_pop_cmd(iree_hal_cmd_block_builder_t* builder,
                                        iree_host_size_t cmd_bytes,
                                        uint16_t fixup_count,
                                        uint16_t binding_count,
                                        uint32_t tile_count) {
  IREE_ASSERT(builder->current_header,
              "pop_cmd requires an active recording session");
  IREE_ASSERT(builder->cmd_cursor >= (uint8_t*)builder->current_header +
                                         sizeof(iree_hal_cmd_block_header_t) +
                                         cmd_bytes,
              "pop_cmd would underflow the command cursor");

  // Restore forward cursor.
  builder->cmd_cursor -= cmd_bytes;

  // Restore backward cursor.
  builder->fixup_cursor +=
      (iree_host_size_t)fixup_count * sizeof(iree_hal_cmd_fixup_t);

  // Restore counters.
  builder->fixup_count -= fixup_count;
  builder->total_binding_count -= binding_count;
  builder->region_dispatch_count--;
  builder->total_dispatch_count--;
  builder->current_region_tiles -= tile_count;
}

iree_status_t iree_hal_cmd_block_builder_barrier(
    iree_hal_cmd_block_builder_t* builder) {
  if (IREE_UNLIKELY(!builder->current_header)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no recording in progress; call begin() first");
  }

  // Space needed: one BARRIER command (8 bytes).
  const iree_host_size_t barrier_bytes = sizeof(iree_hal_cmd_barrier_t);
  const iree_host_size_t split_overhead =
      sizeof(iree_hal_cmd_branch_t) + sizeof(iree_hal_cmd_barrier_t);

  // Check if we'd exceed the region scratch buffer limit.
  if (builder->region_count >= IREE_HAL_CMD_MAX_REGIONS_PER_BLOCK) {
    IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_split_block(builder));
  }

  iree_host_size_t available = iree_hal_cmd_block_builder_remaining(builder);
  if (barrier_bytes + split_overhead > available) {
    IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_split_block(builder));
    // After split, we already have a fresh region from the new block's
    // entry barrier. The user's barrier creates a second region boundary:
    // an empty region (0 tiles), which the processor skips.
  }

  // Finalize the current region (patches the active barrier's dispatch_count).
  iree_hal_cmd_block_builder_finalize_region(builder);

  // Write the new BARRIER command (which also serves as the next region's
  // scheduling header).
  iree_hal_cmd_barrier_t* barrier =
      (iree_hal_cmd_barrier_t*)builder->cmd_cursor;
  memset(barrier, 0, sizeof(*barrier));
  barrier->header.opcode = IREE_HAL_CMD_BARRIER;
  barrier->header.flags = IREE_HAL_CMD_FLAG_NONE;
  barrier->header.size_qwords = sizeof(iree_hal_cmd_barrier_t) / 8;
  barrier->header.dispatch_index = 0;
  builder->current_barrier = barrier;
  builder->cmd_cursor += sizeof(iree_hal_cmd_barrier_t);
  builder->region_count++;

  // Reset region-local counters.
  builder->region_dispatch_count = 0;
  builder->current_region_tiles = 0;

  return iree_ok_status();
}
