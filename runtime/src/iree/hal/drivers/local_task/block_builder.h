// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Write-forward compiler for the block ISA command format.
//
// Translates HAL command buffer API calls into the compact binary format
// defined in block_isa.h. Recording is compilation: HAL API calls become a
// sequence of .text blocks (immutable command stream + fixup tables) linked
// by BRANCH commands, with sizing metadata for .data allocation at issue time.
//
// The builder uses a dual-cursor design within each block pool block:
// commands grow forward from the header, fixup entries and region metadata
// grow backward from the end. Block splitting is automatic when the cursors
// would collide — the builder inserts a BRANCH, finalizes the current block,
// acquires a new one, and continues transparently.
//
// All memory comes from the block pool. The builder itself is stack-allocable
// (~550 bytes including scratch buffers). No system allocator usage during
// recording.

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_BUILDER_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_BUILDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/local_task/block_isa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Recording output
//===----------------------------------------------------------------------===//

// Maximum barrier-delimited regions within a single block pool block.
// The builder splits the block if this limit is reached. Sizes the per-block
// scratch buffer for initial_remaining_tiles accumulation (64 × 4 = 256 bytes).
//
// In practice, blocks have 1-10 regions. The pathological case (every command
// followed by a barrier) reaches 64 in roughly 1KB of command stream.
#define IREE_HAL_CMD_MAX_REGIONS_PER_BLOCK 64

// iree_hal_cmd_block_recording_t is defined in block_isa.h (shared between
// the builder that produces recordings and the processor that executes them).

// Releases all blocks in a recording back to the pool.
// After this call, the recording is empty (first_block = NULL).
// Safe to call on a zero-initialized or already-released recording.
void iree_hal_cmd_block_recording_release(
    iree_hal_cmd_block_recording_t* recording);

//===----------------------------------------------------------------------===//
// Block builder
//===----------------------------------------------------------------------===//

// Write-forward compiler state for recording commands into the block ISA
// format. Stack-allocable; no heap usage beyond block pool acquisition.
//
// Usage:
//   iree_hal_cmd_block_builder_t builder;
//   iree_hal_cmd_block_builder_initialize(pool, &builder);
//   IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_begin(&builder));
//
//   // Record a dispatch.
//   iree_hal_cmd_dispatch_t* dispatch = NULL;
//   iree_hal_cmd_fixup_t* fixups = NULL;
//   iree_host_size_t cmd_size =
//       iree_host_align(offsetof(iree_hal_cmd_dispatch_t, constants) +
//                       constant_count * sizeof(uint32_t), 8);
//   IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
//       &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE,
//       cmd_size, binding_count, binding_count,
//       tile_count, (void**)&dispatch, &fixups));
//   // Fill fixups[0..binding_count-1] with binding resolution entries.
//   dispatch->function = function;
//   dispatch->environment = environment;
//   dispatch->workgroup_size[0] = 64;
//   // ... fill remaining fields ...
//   memcpy(dispatch->constants, constants, constant_count * sizeof(uint32_t));
//
//   IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_barrier(&builder));
//
//   // Record more commands ...
//
//   iree_hal_cmd_block_recording_t recording;
//   IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_end(
//       &builder, &recording));
//   // recording now owns the block chain.
//   iree_hal_cmd_block_builder_deinitialize(&builder);
typedef struct iree_hal_cmd_block_builder_t {
  // Block pool for acquiring new blocks.
  iree_arena_block_pool_t* block_pool;

  //=== Block chain ===========================================================

  // Linked list of finalized blocks. head is the first block to execute;
  // tail is the last (for O(1) append during recording).
  iree_hal_cmd_block_header_t* first_block;
  iree_hal_cmd_block_header_t* last_block;
  // Total blocks (finalized + current). Incremented on acquire.
  uint16_t block_count;

  //=== Current block =========================================================

  // Block header at byte 0 of the current block pool block.
  // NULL when not actively recording (before begin or after end).
  iree_hal_cmd_block_header_t* current_header;
  // Forward cursor: next byte to write a command.
  // Starts at (uint8_t*)current_header + sizeof(iree_hal_cmd_block_header_t).
  uint8_t* cmd_cursor;
  // Backward cursor: next byte to write a fixup entry (grows toward lower
  // addresses). Starts at block_end and decrements by sizeof(fixup) per entry.
  // Fixups are written in recording order (first recorded = highest address).
  uint8_t* fixup_cursor;
  // End of the usable block area. Cached from pool->usable_block_size.
  uint8_t* block_end;

  //=== Current region tracking ================================================

  // Pointer to the BARRIER command for the active region. Patched at region
  // finalization with the actual dispatch_count. Set at begin() (entry barrier)
  // and after each barrier()/split.
  iree_hal_cmd_barrier_t* current_barrier;
  // Number of work commands (DISPATCH, FILL, COPY, UPDATE) in the current
  // region. This is split before reaching 256 because command indices and
  // barrier dispatch counts are encoded as uint8_t.
  uint16_t region_dispatch_count;
  // Total tiles accumulated in the current region. Written to
  // region_tiles_scratch at region finalization.
  uint32_t current_region_tiles;

  //=== Per-block accounting ===================================================

  // Number of regions in the current block.
  uint16_t region_count;
  // Highwater work-command count across regions in the current block.
  uint16_t max_region_dispatch_count;
  // Sum of binding_count across all work commands in the current block.
  uint16_t total_binding_count;
  // Total work-command count in the current block.
  uint16_t total_dispatch_count;
  // Number of fixup entries written backward from the end of the current block.
  uint16_t fixup_count;

  // Per-region initial_remaining_tiles. Accumulated during recording and
  // memcpy'd to the end of the block at finalization. This is the only
  // data movement at finalization — commands and fixups are in place.
  //
  // At finalization, fixups occupying [fixup_cursor, block_end) are shifted
  // down by region_count * sizeof(uint32_t) to make room for the tiles at
  // the very end of the block. The shift is small (fixup data is typically
  // <1KB, shift distance is <256 bytes) and happens once per block.
  uint32_t region_tiles_scratch[IREE_HAL_CMD_MAX_REGIONS_PER_BLOCK];

  //=== Global highwater marks =================================================

  // Maximum across all blocks in this recording. Used to size .data at
  // issue time. Updated at each block finalization.
  uint16_t global_max_region_dispatch_count;
  uint16_t global_max_total_binding_count;
} iree_hal_cmd_block_builder_t;

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

// Initializes a block builder. No blocks are acquired until begin().
void iree_hal_cmd_block_builder_initialize(
    iree_arena_block_pool_t* block_pool,
    iree_hal_cmd_block_builder_t* out_builder);

// Deinitializes the builder and releases any blocks that were not
// transferred to a recording via end(). If end() was called successfully,
// the builder owns no blocks and deinitialize is a no-op.
// Always safe to call, even if begin() was never called.
void iree_hal_cmd_block_builder_deinitialize(
    iree_hal_cmd_block_builder_t* builder);

//===----------------------------------------------------------------------===//
// Recording session
//===----------------------------------------------------------------------===//

// Begins a recording session. Acquires the first block from the pool and
// writes the entry barrier. The builder must not already be in a recording
// session (current_header must be NULL).
iree_status_t iree_hal_cmd_block_builder_begin(
    iree_hal_cmd_block_builder_t* builder);

// Finalizes recording with a RETURN command and packages the result.
// On success, ownership of the block chain transfers to |out_recording|.
// The builder is left in a clean state (no blocks, ready for another
// begin/end cycle or deinitialize).
//
// On failure, the builder retains ownership of all blocks.
// Deinitialize will release them.
iree_status_t iree_hal_cmd_block_builder_end(
    iree_hal_cmd_block_builder_t* builder,
    iree_hal_cmd_block_recording_t* out_recording);

//===----------------------------------------------------------------------===//
// Command recording
//===----------------------------------------------------------------------===//

// Appends a work command (DISPATCH, FILL, or COPY) to the command stream.
//
// If the command plus its fixup entries plus reserved space does not fit
// in the current block, the builder automatically splits: inserts BRANCH,
// finalizes the current block, acquires a new one, and writes an entry
// barrier for the continuation.
//
// Parameters:
//   opcode: IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FILL, or IREE_HAL_CMD_COPY.
//   flags: command flags (INDIRECT, PREDICATED, SEQUENTIAL).
//   cmd_bytes: total command size including header and trailing data
//              (e.g., sizeof(iree_hal_cmd_dispatch_t) + constant_count * 4).
//              Must be a multiple of 8. Maximum 255 * 8 = 2040 bytes.
//   fixup_count: number of fixup entries to reserve at the end of the block.
//   binding_count: .data binding_ptrs[] slots consumed by this command.
//                  Tracked in block header for .data sizing. Usually equals
//                  fixup_count but may differ for shared/aliased bindings.
//   tile_count: scheduling tiles contributed by this command.
//               DISPATCH: product of workgroup_count dimensions.
//               FILL/COPY: ceil(length / transfer tile length).
//               UPDATE: 0 or 1 due to inline command payload size.
//
// Returns a pointer to the command in the block via |out_cmd|. Header fields
// (opcode, flags, size_qwords) are initialized. For DISPATCH commands,
// dispatch_index is set to the region-local index. The caller fills
// command-specific fields.
//
// If |fixup_count| > 0, returns a pointer to the reserved fixup storage via
// |out_fixups|. The caller fills the fixup entries directly in-place (no
// copy). If a subsequent operation fails after append_cmd succeeds, the
// caller must call pop_cmd() to roll back the command.
//
// Lifetime: |out_cmd| and |out_fixups| point into the current block's
// memory and are invalidated by any subsequent builder call that may
// finalize the block: append_cmd (if it triggers a split), barrier(), or
// end(). Callers must complete all writes through these pointers before
// the next builder call. Do not cache them across builder operations.
//
// BARRIER, BRANCH, and RETURN are not valid here — use barrier() and end().
iree_status_t iree_hal_cmd_block_builder_append_cmd(
    iree_hal_cmd_block_builder_t* builder, iree_hal_cmd_opcode_t opcode,
    iree_hal_cmd_flags_t flags, iree_host_size_t cmd_bytes,
    uint16_t fixup_count, uint16_t binding_count, uint32_t tile_count,
    void** out_cmd, iree_hal_cmd_fixup_t** out_fixups);

// Rolls back the most recently appended command. Use this when a post-append
// operation (e.g., binding resolution) fails and the command must be removed.
//
// The caller passes the same parameters that were used for append_cmd so
// the builder can restore its internal counters. The parameters must match
// exactly — mismatches corrupt builder state.
//
// If append_cmd triggered a block split before writing the command, the split
// is permanent (the previous block was already finalized). Only the command
// in the current block is rolled back.
void iree_hal_cmd_block_builder_pop_cmd(iree_hal_cmd_block_builder_t* builder,
                                        iree_host_size_t cmd_bytes,
                                        uint16_t fixup_count,
                                        uint16_t binding_count,
                                        uint32_t tile_count);

// Records a barrier (region boundary). All work in the current region must
// complete before the next region begins.
//
// The builder:
//   - Finalizes the current region (patches barrier's dispatch_count, saves
//     tile count, updates per-block highwater marks).
//   - Writes a new BARRIER command that serves as the next region's header.
//   - Resets region-local counters (dispatch_index, tile count).
//
// If the BARRIER does not fit, the block is split first (which implicitly
// creates a region boundary at the block transition).
iree_status_t iree_hal_cmd_block_builder_barrier(
    iree_hal_cmd_block_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_BUILDER_H_
