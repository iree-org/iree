// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Block processor ISA: type definitions for an in-memory command format
// using a compile-and-execute model.
//
// Recording a command buffer compiles HAL API calls into a compact binary
// stream (.text) with per-block mutable execution state (.data). Issuing
// the command buffer initializes .data (fixup bindings, zero scheduling
// state) and submits to the task executor. The processor walks the command
// stream in a tight loop, claiming tiles via atomic work-stealing.
//
// This is task-system-specific: it depends on the executable dispatch ABI
// for kernel invocation. It is not designed for extraction to a generic layer.

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_ISA_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_ISA_H_

#include <stddef.h>
#include <stdint.h>

#include "iree/async/span.h"
#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/hal/local/executable_library.h"

typedef struct iree_hal_local_executable_t iree_hal_local_executable_t;
typedef struct iree_hal_buffer_t iree_hal_buffer_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Command opcodes and flags
//===----------------------------------------------------------------------===//

typedef enum iree_hal_cmd_opcode_e {
  IREE_HAL_CMD_DISPATCH = 0,  // Execute a tiled compute grid.
  IREE_HAL_CMD_FILL = 1,      // Fill a buffer region with a pattern.
  IREE_HAL_CMD_COPY = 2,      // Copy between buffer regions.
  IREE_HAL_CMD_UPDATE = 3,    // Copy inline host data to a buffer region.
  IREE_HAL_CMD_BARRIER = 4,   // Region boundary: all prior work completes.
  IREE_HAL_CMD_BRANCH = 5,    // Continue at target block.
  IREE_HAL_CMD_RETURN = 6,    // Command buffer complete.
} iree_hal_cmd_opcode_t;

typedef uint8_t iree_hal_cmd_flags_t;
enum iree_hal_cmd_flag_bits_e {
  IREE_HAL_CMD_FLAG_NONE = 0,
  // Parameters (workgroup count, offsets, lengths) are read from a buffer
  // at execution time rather than baked into the command.
  IREE_HAL_CMD_FLAG_INDIRECT = 1u << 0,
  // Skip execution if indirect parameters are all zero.
  IREE_HAL_CMD_FLAG_PREDICATED = 1u << 1,
  // Adjacent tiles share data (sequential dependency within the dispatch).
  IREE_HAL_CMD_FLAG_SEQUENTIAL = 1u << 2,
};

// Target byte length for each transfer tile. This is large enough to amortize
// scheduling and CAS overhead while still giving large fills/copies enough
// independent work to use the available CPU workers.
#define IREE_HAL_CMD_TRANSFER_TILE_LENGTH_BYTES ((iree_device_size_t)64 * 1024)

// Returns the number of transfer tiles required to cover |length| bytes.
static inline uint64_t iree_hal_cmd_transfer_tile_count_u64(
    iree_device_size_t length) {
  if (length == 0) return 0;
  return 1 + (uint64_t)((length - 1) / IREE_HAL_CMD_TRANSFER_TILE_LENGTH_BYTES);
}

// Returns true if |length| can be represented in the 32-bit tile counters used
// by the block processor.
static inline bool iree_hal_cmd_transfer_tile_count_is_representable(
    iree_device_size_t length) {
  return iree_hal_cmd_transfer_tile_count_u64(length) <= UINT32_MAX;
}

// Returns the number of transfer tiles required to cover |length| bytes.
// Callers must have checked that the result is representable.
static inline uint32_t iree_hal_cmd_transfer_tile_count(
    iree_device_size_t length) {
  return (uint32_t)iree_hal_cmd_transfer_tile_count_u64(length);
}

// Returns the byte offset of |tile| within a transfer range.
static inline iree_device_size_t iree_hal_cmd_transfer_tile_offset(
    uint32_t tile) {
  return (iree_device_size_t)tile * IREE_HAL_CMD_TRANSFER_TILE_LENGTH_BYTES;
}

// Returns the byte length of |tile| within a transfer of |total_length| bytes.
static inline iree_device_size_t iree_hal_cmd_transfer_tile_length(
    iree_device_size_t total_length, uint32_t tile) {
  const iree_device_size_t offset = iree_hal_cmd_transfer_tile_offset(tile);
  const iree_device_size_t remaining_length = total_length - offset;
  return remaining_length < IREE_HAL_CMD_TRANSFER_TILE_LENGTH_BYTES
             ? remaining_length
             : IREE_HAL_CMD_TRANSFER_TILE_LENGTH_BYTES;
}

//===----------------------------------------------------------------------===//
// Command header
//===----------------------------------------------------------------------===//

// Prefix of every command. Commands are 8-byte aligned; the next command is at
// (uint8_t*)header + header->size_qwords * 8.
typedef struct iree_hal_cmd_header_t {
  // Command type (iree_hal_cmd_opcode_t).
  uint8_t opcode;
  // Bitfield of iree_hal_cmd_flag_bits_e.
  uint8_t flags;
  // Total command size in 8-byte units (including this header). The next
  // command follows at (uint8_t*)this + size_qwords * 8. Maximum encodable
  // size is 255 * 8 = 2040 bytes. Push constants are inline; bindings are
  // NOT (they live in .data via fixup).
  uint8_t size_qwords;
  // Work commands (DISPATCH, FILL, COPY, UPDATE): region-local index into .data
  // tile_indices[]. The builder resets this to 0 at each barrier, so
  // commands in different regions reuse the same .data slots. The multi-
  // worker processor uses this to distribute tiles across workers for ALL
  // work command types. Must be 0 for non-work commands (BARRIER, BRANCH,
  // RETURN).
  uint8_t dispatch_index;
} iree_hal_cmd_header_t;

static_assert(sizeof(iree_hal_cmd_header_t) == 4,
              "command header must be exactly 4 bytes");

//===----------------------------------------------------------------------===//
// Indirect parameter structs
//===----------------------------------------------------------------------===//

// Bundled per-opcode. One buffer read per indirect command. The command opcode
// implies the parameter struct type — no kind discriminator needed.

// Indirect dispatch: workgroup count read from a buffer.
typedef struct iree_hal_dispatch_params_t {
  uint32_t workgroup_count[3];
} iree_hal_dispatch_params_t;

// Indirect copy: source/target offsets and length read from a buffer.
typedef struct iree_hal_copy_params_t {
  iree_device_size_t source_offset;
  iree_device_size_t target_offset;
  iree_device_size_t length;
} iree_hal_copy_params_t;

// Indirect fill: target offset, length, and pattern read from a buffer.
typedef struct iree_hal_fill_params_t {
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint64_t pattern;  // Zero-extended from actual pattern width.
} iree_hal_fill_params_t;

//===----------------------------------------------------------------------===//
// Binding table and fixup entries
//===----------------------------------------------------------------------===//

// A resolved binding entry provided at issue time. The binding table is a
// command-buffer-global array — typically 8-20 entries for thousands of
// dispatches. Each dispatch references an arbitrary subset via fixup entries.
typedef struct iree_hal_cmd_binding_entry_t {
  void* base;
  size_t length;
} iree_hal_cmd_binding_entry_t;

typedef uint32_t iree_hal_cmd_fixup_flags_t;
enum iree_hal_cmd_fixup_flag_bits_e {
  IREE_HAL_CMD_FIXUP_FLAG_NONE = 0,
  // The host_ptr field is an iree_async_span_t* requiring span dereference.
  // Used for async-backed buffers where the pointer may not be available
  // at recording time (registered I/O regions, etc.). Rare.
  IREE_HAL_CMD_FIXUP_FLAG_SPAN = 1u << 0,
};

// Unified fixup entry: resolves one .data binding_ptrs[] slot at block entry.
//
// Discrimination on host_ptr/buffer and flags:
//
//   host_ptr == NULL (indirect): binding resolved via binding table lookup.
//     Used for indirect command buffers where buffers are provided at submit
//     time. THE FAST PATH — all fixups in an indirect CB take this branch.
//
//   flags == 0, host_ptr != NULL (direct inline): binding resolved directly
//     from the host pointer stored in the fixup. Used for one-shot command
//     buffers where buffer pointers are known at recording time. The buffer
//     is persistently mapped and retained by the CB's resource_set.
//
//   flags & SPAN (span): binding resolved via span dereference. Used for
//     async-backed buffers (registered I/O regions). Rare.
typedef struct iree_hal_cmd_fixup_t {
  union {
    // Direct inline: resolved host pointer (flags == 0).
    void* host_ptr;
    // Span: pointer to iree_async_span_t (flags & SPAN).
    const iree_async_span_t* span;
  };
  // Byte offset within the buffer (64-bit for >4GB buffers).
  // For indirect fixups: per-reference offset within the binding.
  // For direct inline fixups: 0 (map_range already applied the offset).
  iree_device_size_t offset;
  // Length of the accessible buffer region in bytes. Populates .data
  // binding_lengths[] at block entry alongside binding_ptrs[].
  iree_device_size_t length;
  // .data binding_ptrs[] slot to write the resolved pointer into.
  uint16_t data_index;
  // For indirect fixups: binding table index. For direct: unused (0).
  uint16_t slot;
  iree_hal_cmd_fixup_flags_t flags;
} iree_hal_cmd_fixup_t;

static_assert(sizeof(iree_hal_cmd_fixup_t) == 32, "fixup entries are 4 qwords");

// Fixup resolution at block entry (single loop, branch-predicted):
//
//   for (i = 0; i < fixup_count; ++i) {
//     const iree_hal_cmd_fixup_t* f = &fixups[i];
//     if (!f->host_ptr) {
//       binding_ptrs[f->data_index] =
//           (uint8_t*)table[f->slot].base + f->offset;
//       binding_lengths[f->data_index] = f->length;
//     } else if (!iree_any_bit_set(f->flags, IREE_HAL_CMD_FIXUP_FLAG_SPAN)) {
//       binding_ptrs[f->data_index] = (uint8_t*)f->host_ptr + f->offset;
//       binding_lengths[f->data_index] = f->length;
//     } else {
//       binding_ptrs[f->data_index] =
//           iree_async_span_ptr(*f->span) + f->offset;
//       binding_lengths[f->data_index] = f->length;
//     }
//   }

//===----------------------------------------------------------------------===//
// Block header
//===----------------------------------------------------------------------===//

// Precedes the command stream in each block pool block. The block layout
// is designed for dual-cursor recording: commands grow forward from the
// header, fixups grow backward from the end. No memmove is ever needed.
//
// Block layout within a block pool block (block_size bytes total):
//
//   [block header (24 bytes)]
//   [command stream: 8-byte-aligned commands, used_bytes total]
//   ... (unused gap, if any) ...
//   [fixup_count × iree_hal_cmd_fixup_t, packed toward the end]
//   [initial_remaining_tiles: region_count × uint32_t, at the very end]
//
// During recording, the builder maintains two cursors:
//   cmd_cursor:   starts at sizeof(header), grows forward
//   fixup_cursor: starts at block_end - tail_metadata, grows backward
// When cmd_cursor + safety_margin >= fixup_cursor, the block is split.
//
// At block finalization, the header fields are written (all counts now
// known) and initial_remaining_tiles is written at the end of the block
// (the only data copied at finalization — everything else is in place).
//
// The builder owns the full block pool block allocation. The
// iree_arena_block_pool_t acquire/release API gives us raw memory; the
// arena_block_t trailer is only meaningful to the arena allocator, which
// we bypass. We manage our own chain via next_block.
typedef struct iree_hal_cmd_block_header_t {
  // Linked list pointer for the builder's block chain.
  struct iree_hal_cmd_block_header_t* next_block;
  // Bytes of command stream data (starting at sizeof(header)).
  uint16_t used_bytes;
  // Number of barrier-delimited regions in this block.
  uint16_t region_count;
  // Highwater mark of dispatch commands per region across all regions in
  // this block. Sizes .data tile_indices[].
  uint16_t max_region_dispatch_count;
  // Sum of binding_count across ALL dispatches in this block (not per
  // region). Sizes .data binding_ptrs[].
  uint16_t total_binding_count;
  // Number of fixup entries packed toward the end of the block.
  uint16_t fixup_count;
  // Total dispatch commands across all regions (for validation).
  uint16_t total_dispatch_count;
  // Total size of this block pool block in bytes. Required to locate
  // fixups and initial_remaining_tiles at the end of the block.
  uint16_t block_size;
  uint16_t reserved;
} iree_hal_cmd_block_header_t;

static_assert(sizeof(iree_hal_cmd_block_header_t) == 24,
              "block header is 3 qwords (no flexible array member)");

//===----------------------------------------------------------------------===//
// DISPATCH command
//===----------------------------------------------------------------------===//

// Variable-size dispatch command. The kernel function and environment are
// resolved at recording time from the executable + export ordinal — no
// indirection at execution. Bindings are NOT inline; they live in .data and
// are populated by the fixup tables at block entry.
//
// At execution, the processor builds iree_hal_executable_dispatch_state_v0_t
// on the stack per-dispatch with direct pointer assignments:
//   dispatch_state.constants = cmd->constants;
//   dispatch_state.binding_ptrs =
//       (void* const*)&state->binding_ptrs[cmd->binding_data_base];
//   dispatch_state.binding_lengths =
//       state->binding_lengths
//           ? (const size_t*)&state->binding_lengths[cmd->binding_data_base]
//           : NULL;
// No IREE_STRUCT_LAYOUT computation, no translation, no memcpy.
typedef struct iree_hal_cmd_dispatch_t {
  iree_hal_cmd_header_t header;  // opcode=DISPATCH

  // Packing: fills the 4-byte gap between the 4-byte header and the 8-byte-
  // aligned executable pointer.
  uint8_t constant_count;
  uint8_t binding_count;
  uint16_t binding_data_base;

  // Executable and export ordinal. Always set at recording time. Used by
  // the VM fallback path (function == NULL) for vtable dispatch, and by
  // profiling/diagnostics for per-export statistics tracking.
  iree_hal_local_executable_t* executable;

  // Kernel entry point resolved at recording time. NULL for VM-based
  // backends (VMVX) that dispatch through executable->vtable->issue_call.
  // When non-NULL, the processor calls:
  //   function(&executable->environment, &dispatch_state, &workgroup_state)
  iree_hal_executable_dispatch_v0_t function;

  // Export ordinal within the executable.
  uint16_t export_ordinal;

  // Reserved for future dispatch command fields.
  uint16_t reserved;

  // Profiling sideband data.
  struct {
    // Command-buffer-global operation index used for profiling joins.
    uint32_t command_index;
  } profile;

  // Workgroup grid dimensions.
  uint32_t workgroup_size[3];

  // Discriminated by the INDIRECT flag:
  //   Direct: inline workgroup count.
  //   Indirect: location of iree_hal_dispatch_params_t to read at execution.
  union {
    // Direct dispatch (INDIRECT flag NOT set): inline workgroup count.
    iree_hal_dispatch_params_t direct;
    // Indirect dispatch (INDIRECT flag set): where to read params from.
    struct {
      // .data binding_ptrs index for the buffer containing params.
      uint16_t params_binding;
      uint16_t reserved;
      // Byte offset to iree_hal_dispatch_params_t within the buffer.
      uint32_t params_offset;
      // Scheduling hint: expected total tile count for wake budget
      // computation. Overwritten at execution time with the actual value.
      uint32_t tile_count_hint;
    } indirect;
  } params;

  // Tile decomposition. For direct dispatches, precomputed from
  // workgroup_count at recording time. For indirect dispatches, these are
  // scheduling hints overwritten at execution after reading actual params.
  uint32_t tile_count;
  uint32_t tiles_per_reservation;

  // Transient local memory size in bytes. Each worker maintains a local memory
  // buffer pinned to its affinity; if a dispatch requires more than the worker
  // currently has, the worker grows the allocation. Computed at recording time
  // from dispatch_attrs.local_memory_pages * PAGE_SIZE + dynamic_local_memory.
  uint32_t local_memory_size;

  // Push constants, inline in .text. Accessed as cmd->constants[i].
  uint32_t constants[];
} iree_hal_cmd_dispatch_t;

static_assert(offsetof(iree_hal_cmd_dispatch_t, constants) == 68,
              "dispatch fixed part is 68 bytes; constants[] follows");

//===----------------------------------------------------------------------===//
// FILL command
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_fill_t {
  iree_hal_cmd_header_t header;  // opcode=FILL

  // .data binding_ptrs index for the target buffer.
  uint16_t target_binding;
  // Fill pattern width: 1, 2, or 4 bytes.
  uint8_t pattern_length;
  uint8_t reserved;

  union {
    // Direct fill (INDIRECT flag NOT set): parameters are inline.
    iree_hal_fill_params_t direct;
    // Indirect fill (INDIRECT flag set): parameters read from buffer.
    struct {
      uint16_t params_binding;  // .data binding_ptrs index
      uint16_t reserved[3];
      uint32_t params_offset;  // byte offset to iree_hal_fill_params_t
    } indirect;
  } params;
} iree_hal_cmd_fill_t;

static_assert(sizeof(iree_hal_cmd_fill_t) == 32, "fill command is 4 qwords");

//===----------------------------------------------------------------------===//
// COPY command
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_copy_t {
  iree_hal_cmd_header_t header;  // opcode=COPY

  // .data binding_ptrs indices for source and target buffers.
  uint16_t source_binding;
  uint16_t target_binding;

  union {
    // Direct copy (INDIRECT flag NOT set): parameters are inline.
    iree_hal_copy_params_t direct;
    // Indirect copy (INDIRECT flag set): parameters read from buffer.
    struct {
      uint16_t params_binding;  // .data binding_ptrs index
      uint16_t reserved[3];
      uint64_t params_offset;  // byte offset to iree_hal_copy_params_t
    } indirect;
  } params;
} iree_hal_cmd_copy_t;

static_assert(sizeof(iree_hal_cmd_copy_t) == 32, "copy command is 4 qwords");

//===----------------------------------------------------------------------===//
// UPDATE command
//===----------------------------------------------------------------------===//

// Copies inline host data (captured at recording time) to a device buffer.
// The source data follows the command header inline in .text. At execution
// the processor does a single memcpy. UPDATE is effectively single-tile because
// the inline command payload is capped at 2040 bytes.
typedef struct iree_hal_cmd_update_t {
  iree_hal_cmd_header_t header;  // opcode=UPDATE

  // .data binding_ptrs index for the target buffer.
  uint16_t target_binding;
  uint16_t reserved;

  // Offset within the target buffer to write to.
  iree_device_size_t target_offset;
  // Number of bytes to copy from the inline source data.
  iree_device_size_t length;

  // Inline source data captured at recording time. Accessed as
  // cmd->source_data[i]. Total command size is 8-byte aligned.
  uint8_t source_data[];
} iree_hal_cmd_update_t;

static_assert(sizeof(iree_hal_cmd_update_t) == 24,
              "update command is 3 qwords (plus trailing inline data)");

//===----------------------------------------------------------------------===//
// BARRIER, BRANCH, RETURN commands
//===----------------------------------------------------------------------===//

// BARRIER: region boundary and scheduling metadata. All prior work in the
// region must complete before the next region begins.
//
// Every block starts with an entry barrier, and every user-inserted barrier
// produces one of these. The dispatch_count and wake_budget describe the
// region FOLLOWING this barrier (up to the next barrier/branch/return).
// The processor uses dispatch_count to size tile_index zeroing and
// wake_budget to seed or ramp the executor wake tree.
typedef struct iree_hal_cmd_barrier_t {
  iree_hal_cmd_header_t header;  // opcode=BARRIER, size_qwords=1
  // Number of work commands in the following region. Used by the processor to
  // zero exactly this many tile_index entries at region entry.
  uint8_t dispatch_count;
  uint8_t reserved;
  // Wake demand hint for the following region. 0 means the processor decides
  // dynamically based on tile counts and available workers.
  uint16_t wake_budget;
} iree_hal_cmd_barrier_t;

static_assert(sizeof(iree_hal_cmd_barrier_t) == 8,
              "barrier command is 1 qword");

// BRANCH: continue execution at the target block. Inserted by the builder
// when a block pool block fills (same pattern as AMDGPU command_buffer.c
// split_block). The target block has its own header, fixup tables, and
// command stream.
typedef struct iree_hal_cmd_branch_t {
  iree_hal_cmd_header_t header;  // opcode=BRANCH
  uint32_t reserved;
  struct iree_hal_cmd_block_header_t* target;
} iree_hal_cmd_branch_t;

static_assert(sizeof(iree_hal_cmd_branch_t) == 16,
              "branch command is 2 qwords");

// RETURN: command buffer execution is complete. The processor signals
// completion and does not advance further.
typedef iree_hal_cmd_header_t iree_hal_cmd_return_t;

//===----------------------------------------------------------------------===//
// .data block state
//===----------------------------------------------------------------------===//

// Per-block mutable execution state (.data). Allocated at issue time, sized
// to the highwater mark across all blocks (known after recording). Reused
// across blocks: each block entry reinitializes the state.
//
// Block tasks are NOT in .data — they live in a separate contiguous slab
// allocated at issue time. The slab is pre-linked: each block task's
// completion_task chains to the next block's task. The final
// task in the slab is the retire task for resource cleanup and queue
// chaining. This allows the task system to sequence blocks automatically
// and gives a single slab for timing/statistics collection at CB retire.
//
// .data is purely scratch: scheduling atomics + resolved binding pointers.
//
// "Tiles" are the universal scheduling primitive: a dispatch of N tiles is
// N work items, a copy/fill is ceil(length / transfer tile length), and an
// UPDATE is 0 or 1 tile due to the inline source-data command-size limit. All
// commands decompose into tiles.
//
// Cache-line isolation is critical throughout. The existing task system
// loses ~57% of runtime on some models to 48-64 threads hammering the same
// cache lines in spin loops — workers actively harm each other even when
// idle. Every atomically-contested field in .data gets its own cache line.
//
// Layout in memory (all hot atomics are cache-line padded):
//
//   iree_hal_cmd_block_state_t  [fixed part: 2 cache lines]
//     - active_region_index     [cache line 0: mostly-read by all workers]
//     - remaining_tiles         [cache line 1: written on tile completion]
//   tile_index[0..max-1]        [per-dispatch work-stealing, 64-bit
//   epoch-tagged] void*    binding_ptrs[total_binding_count]   [cold: written
//   once at fixup] size_t   binding_lengths[total_binding_count] [cold: written
//   once at fixup]
//
// Each tile_index is a 64-bit atomic: upper 32 bits = epoch (region_index),
// lower 32 bits = tile counter. Workers claim tiles via CAS, atomically
// validating the epoch and incrementing the counter. If the epoch doesn't
// match the active region, the CAS fails with zero side effects — no tiles
// stolen, no counters corrupted. This eliminates the ABA-like vulnerability
// where a preempted worker's stale fetch_add could corrupt a later region's
// tile claiming (the TOCTOU window between reading active_region_index and
// modifying the tile_index).
//
// binding_lengths are always populated for correctness (the dispatch ABI
// may use them for bounds checking). If profiling shows the extra .data
// space matters, a compile-time toggle can skip them and pass NULL to the
// dispatch state.
//
// Initialization at block entry:
//   memset(state, 0, state_size) + fixup loop + compute remaining_tiles.
//   All counters (active_region_index, tile_indices) are zero-initialized.
//   If the first active region is not region 0, tile_index epochs are set
//   to the first active region's index.
//   remaining_tiles is set to the first active region's total tiles.
//   The only other non-zero data is binding_ptrs from fixup.
//
// At region boundary (completer only, immediate — no arrival wait):
//   - Set tile_index epochs for the next region (cache-line-strided stores).
//   - Set remaining_tiles from the next region's current tile counts.
//   - Store active_region_index (release) to unlock workers.

// Cache-line stride for tile_index entries. Each tile_index is padded to
// a full cache line so that workers claiming tiles from different dispatches
// within the same region never contend on the same cache line.
#define IREE_HAL_CMD_TILE_INDEX_STRIDE \
  ((iree_host_size_t)iree_hardware_destructive_interference_size)

typedef struct iree_hal_cmd_block_state_t {
  // Which region is currently executing. Workers read this to find the
  // barrier in the command stream (barrier index within the current block).
  // Written only at region transitions by the completer.
  // Own cache line: read traffic from workers must not compete with writes
  // on the remaining_tiles cache line.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int32_t active_region_index;

  // Global region epoch: monotonically increasing counter used as the CAS
  // epoch tag in tile_indices. Unlike active_region_index (which resets to
  // 0 per block), this never resets. Workers read this at drain entry and
  // pass it to process_region for CAS validation. When init_block sets
  // tile_indices to the new epoch, stale workers from the previous block
  // CAS-fail immediately (their old epoch doesn't match).
  //
  // Same cache line as active_region_index: both are read together.
  iree_atomic_int32_t region_epoch;

  // Cached pointer to the current region's barrier command. Set by
  // init_block (first region) and by the completer at region transitions.
  // Workers load this instead of walking the command stream from the block
  // header on each drain() call.
  //
  // Same cache line as active_region_index: both are read together.
  iree_atomic_intptr_t cached_barrier;

  // Epoch-tagged countdown of remaining tiles in the current region.
  // Upper 32 bits = region epoch, lower 32 bits = remaining count.
  // Initialized to (epoch << 32 | total_tiles) at block entry and region
  // transitions. Workers decrement via CAS after completing tiles: the CAS
  // validates the epoch, preventing stale workers from a completed region
  // from corrupting the count after the completer has advanced and reset
  // remaining_tiles for the next region. The worker whose CAS drives the
  // count to zero becomes the completer.
  //
  // Workers with 0 tiles skip the election to avoid false positives.
  //
  // TODO: CAS contention on this cache line may matter at high worker
  // counts. A per-worker local accumulator with a single fetch_sub at
  // drain exit would reduce contention at the cost of delayed completer
  // election. Profile before optimizing.
  //
  // Own cache line: workers write this on every reservation completion.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int64_t remaining_tiles;

  // Variable-length arrays follow at computed offsets.
  // Use the iree_hal_cmd_block_state_* accessors below.
} iree_hal_cmd_block_state_t;

// Returns a pointer to a specific tile_index entry. Each entry is a 64-bit
// atomic (epoch-tagged: upper 32 bits = region_index epoch, lower 32 bits =
// tile counter), padded to a full cache line so workers claiming tiles from
// different dispatches never touch the same cache line.
//
// Workers claim tiles via CAS: atomically validate the epoch matches the
// active region and increment the counter. Stale workers (preempted during
// a region transition) see an epoch mismatch and get 0 tiles with zero
// side effects.
static inline iree_atomic_int64_t* iree_hal_cmd_block_state_tile_index(
    iree_hal_cmd_block_state_t* state, uint16_t dispatch_index) {
  uint8_t* base = (uint8_t*)state + sizeof(iree_hal_cmd_block_state_t);
  return (iree_atomic_int64_t*)(base + (iree_host_size_t)dispatch_index *
                                           IREE_HAL_CMD_TILE_INDEX_STRIDE);
}

// Returns the binding_ptrs array (total_binding_count entries).
// Populated once at block entry by fixup, then read-only during execution.
// NOT region-shared — each dispatch has unique bindings identified by
// binding_data_base. Cold during execution; no cache-line padding needed.
static inline void** iree_hal_cmd_block_state_binding_ptrs(
    iree_hal_cmd_block_state_t* state, uint16_t max_region_dispatch_count) {
  uint8_t* base = (uint8_t*)state + sizeof(iree_hal_cmd_block_state_t);
  uintptr_t after_tiles =
      (uintptr_t)(base + (iree_host_size_t)max_region_dispatch_count *
                             IREE_HAL_CMD_TILE_INDEX_STRIDE);
  return (void**)iree_host_align(after_tiles, iree_alignof(void*));
}

// Returns the binding_lengths array (total_binding_count entries).
// Follows binding_ptrs contiguously. Cold during execution.
static inline size_t* iree_hal_cmd_block_state_binding_lengths(
    iree_hal_cmd_block_state_t* state, uint16_t max_region_dispatch_count,
    uint16_t total_binding_count) {
  void** binding_ptrs =
      iree_hal_cmd_block_state_binding_ptrs(state, max_region_dispatch_count);
  return (size_t*)(binding_ptrs + total_binding_count);
}

// Computes the total .data allocation size for a block with the given
// parameters.
static inline iree_host_size_t iree_hal_cmd_block_state_size(
    uint16_t max_region_dispatch_count, uint16_t total_binding_count) {
  iree_host_size_t size = sizeof(iree_hal_cmd_block_state_t);
  // Cache-line-padded tile indices.
  size += (iree_host_size_t)max_region_dispatch_count *
          IREE_HAL_CMD_TILE_INDEX_STRIDE;
  // Pointer-aligned cold binding arrays.
  size = iree_host_align(size, iree_alignof(void*));
  size += total_binding_count * sizeof(void*);
  size += total_binding_count * sizeof(iree_device_size_t);
  return size;
}

//===----------------------------------------------------------------------===//
// Block navigation
//===----------------------------------------------------------------------===//

// Returns a pointer to the start of the command stream within a block.
// Commands begin immediately after the fixed-size header.
static inline const iree_hal_cmd_header_t* iree_hal_cmd_block_commands(
    const iree_hal_cmd_block_header_t* header) {
  return (const iree_hal_cmd_header_t*)(header + 1);
}

// Returns the reservation size for the initial_remaining_tiles array at the
// end of a block, rounded up to fixup alignment so that the fixup table
// (packed immediately before) always starts at a properly aligned address.
// The actual tile data occupies the last region_count * 4 bytes of the block;
// any padding between fixups and tiles is unused.
static inline iree_host_size_t iree_hal_cmd_block_tile_reservation_size(
    uint16_t region_count) {
  return iree_host_align((iree_host_size_t)region_count * sizeof(uint32_t),
                         iree_alignof(iree_hal_cmd_fixup_t));
}

// Returns a pointer to the initial_remaining_tiles array at the very end
// of the block (region_count entries, packed at block end).
// Used as the completion threshold for .data region_sync.tiles_completed
// at each region boundary.
static inline const uint32_t* iree_hal_cmd_block_initial_remaining_tiles(
    const iree_hal_cmd_block_header_t* header) {
  const uint8_t* block_end =
      (const uint8_t*)header + (iree_host_size_t)header->block_size;
  return (const uint32_t*)(block_end - (iree_host_size_t)header->region_count *
                                           sizeof(uint32_t));
}

// Returns a pointer to the fixup table entries, which are packed between
// the command stream and initial_remaining_tiles. The tile reservation is
// rounded up to fixup alignment so fixups always start at a properly aligned
// address. Fixups grow backward from before the reservation during recording,
// so the first fixup entry (lowest address) is the last one recorded.
static inline const iree_hal_cmd_fixup_t* iree_hal_cmd_block_fixups(
    const iree_hal_cmd_block_header_t* header) {
  const uint8_t* block_end =
      (const uint8_t*)header + (iree_host_size_t)header->block_size;
  const uint8_t* fixup_end =
      block_end -
      iree_hal_cmd_block_tile_reservation_size(header->region_count);
  return (const iree_hal_cmd_fixup_t*)(fixup_end -
                                       (iree_host_size_t)header->fixup_count *
                                           sizeof(iree_hal_cmd_fixup_t));
}

// Advances to the next command in the stream.
static inline const iree_hal_cmd_header_t* iree_hal_cmd_next(
    const iree_hal_cmd_header_t* cmd) {
  return (const iree_hal_cmd_header_t*)((const uint8_t*)cmd +
                                        (iree_host_size_t)cmd->size_qwords * 8);
}

//===----------------------------------------------------------------------===//
// Block recording
//===----------------------------------------------------------------------===//

// Forward declaration — full type in iree/base/internal/arena.h.
// Only needed by code that calls iree_hal_cmd_block_recording_release()
// (which requires the block pool for deallocation). The processor and
// other consumers only use the block chain and sizing metadata.
typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;

// Result of a successful recording session. Contains the compiled block chain
// and the sizing parameters needed for .data allocation at issue time.
//
// Ownership: the recording owns the block chain. The caller must either:
//   - Issue the command buffer (transferring blocks to the executor), or
//   - Release the blocks back to the pool via
//     iree_hal_cmd_block_recording_release() (declared in block_builder.h).
//
// The block chain is a singly-linked list via next_block pointers. Each block
// is an independent compilation unit: its own command stream, fixup tables,
// and region metadata. The processor executes blocks sequentially via BRANCH
// commands, reinitializing .data at each block entry.
typedef struct iree_hal_cmd_block_recording_t {
  // Block pool the blocks were acquired from. Needed for release.
  iree_arena_block_pool_t* block_pool;
  // Head of the block chain (first block to execute).
  iree_hal_cmd_block_header_t* first_block;
  // Total blocks in the chain. Used at issue time to size the task slab.
  uint16_t block_count;
  // .data sizing: maximum max_region_dispatch_count across all blocks.
  // Determines the number of cache-line-padded tile_index entries in .data.
  // .data is sized to the highwater mark so it can be reused across blocks.
  uint16_t max_region_dispatch_count;
  // .data sizing: maximum total_binding_count across all blocks.
  // Determines the binding_ptrs[] and binding_lengths[] array sizes in .data.
  uint16_t max_total_binding_count;
} iree_hal_cmd_block_recording_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_BLOCK_ISA_H_
