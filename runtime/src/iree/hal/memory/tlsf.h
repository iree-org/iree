// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_TLSF_H_
#define IREE_HAL_MEMORY_TLSF_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// TLSF (Two-Level Segregated Fit) Offset Allocator
//===----------------------------------------------------------------------===//
//
// A TLSF offset allocator manages a contiguous range of offsets [0, N) using
// the Two-Level Segregated Fit algorithm. It is pure bookkeeping: it tracks
// which sub-ranges are allocated and which are free, but has no knowledge of
// physical memory, device pointers, or GPU drivers. All metadata lives in host
// memory — the allocator never reads from or writes to the managed range.
//
// ## Why TLSF
//
// TLSF provides O(1) allocation and deallocation with bounded external
// fragmentation. It was designed for real-time systems where worst-case latency
// matters as much as average throughput. Both VMA (Vulkan Memory Allocator) and
// D3D12MA evaluated multiple allocation algorithms and independently converged
// on TLSF, rejecting buddy allocators for the same reasons: buddy allocators
// have O(log N) split/merge chains, cannot produce allocations that aren't
// power-of-two multiples of the minimum block size, and suffer from internal
// fragmentation that wastes 25-50% of memory for non-power-of-two requests.
//
// TLSF's O(1) guarantee comes from encoding the free block search as two
// bitmap scans (one per level), each implemented with a single hardware
// count-trailing-zeros instruction. No loops, no trees, no hash tables.
//
// ## How it works
//
// Free blocks are organized in a two-dimensional matrix of segregated free
// lists, indexed by block size:
//
//   ┌────────────────────────────────────────────────────────────────────┐
//   │  First Level (FL): 64 bins, one per bit position of block size     │
//   │                                                                    │
//   │  FL index = MSB position of the block size. For a 1000-byte block, │
//   │  the MSB is bit 9 (since 512 <= 1000 < 1024), so FL = 9.           │
//   │                                                                    │
//   │  This groups blocks into power-of-two size classes:                │
//   │    FL 8:  [256, 512)                                               │
//   │    FL 9:  [512, 1024)                                              │
//   │    FL 10: [1024, 2048)                                             │
//   │    ...                                                             │
//   │    FL 63: [2^63, 2^64)                                             │
//   ├────────────────────────────────────────────────────────────────────┤
//   │  Second Level (SL): 32 sub-bins per FL level                       │
//   │                                                                    │
//   │  Each power-of-two range is subdivided into 32 equal-width bins.   │
//   │  For FL 10 (range [1024, 2048), width 1024):                       │
//   │    SL 0:  [1024, 1056)                                             │
//   │    SL 1:  [1056, 1088)                                             │
//   │    ...                                                             │
//   │    SL 31: [1992, 2048)                                             │
//   │                                                                    │
//   │  This gives ~3.1% granularity within each FL level.                │
//   └────────────────────────────────────────────────────────────────────┘
//
// Each (FL, SL) cell is a doubly-linked list of free blocks in that size
// class. Two bitmaps track which cells have blocks:
//
//   fl_bitmap:     64-bit, bit i set means some SL bin in FL level i is
//                  non-empty.
//   sl_bitmaps[i]: 32-bit, bit j set means the free list at (i, j) is
//                  non-empty.
//
// ### Allocation (O(1))
//
// 1. Round the requested size up to alignment and clamp to the minimum block
//    size.
// 2. Map the size to (FL, SL) indices.
// 3. Search for a block >= the requested size:
//    a. Scan sl_bitmaps[FL] starting at SL for the first set bit (ctz).
//    b. If no bit found, scan fl_bitmap starting at FL+1 for the next
//       populated FL level (ctz), then take any SL bit in that level.
// 4. Pop the head block from the found free list.
// 5. If the block is larger than needed by >= min_block_length, split it:
//    allocate the first part, insert the remainder as a new free block in
//    the appropriate (FL, SL) bin.
// 6. Return the allocated offset, length, and block handle.
//
// ### Deallocation (O(1))
//
// 1. Mark the block as free and attach its death frontier.
// 2. Check the physically adjacent left neighbor: if free, remove it from
//    its free list and merge (extend the left block's range to cover this
//    block; JOIN frontiers).
// 3. Check the physically adjacent right neighbor: if free, remove it from
//    its free list and merge (extend this block's range to cover the right
//    block; JOIN frontiers).
// 4. Insert the (possibly merged) block into the appropriate (FL, SL) bin.
//
// ### Death frontiers
//
// Each free block carries an inline death frontier: an `iree_async_frontier_t`
// with a configurable maximum entry capacity (set at allocator creation time,
// shared by all blocks in the instance). The frontier captures the causal
// position at which the block was freed — "everything that used this memory
// has been submitted to these queues at these epochs."
//
// When the async allocator considers reusing a free block, it checks whether
// the requester's frontier dominates the block's death frontier. If yes, the
// reuse is safe without any device synchronization. If not, the block is
// skipped or a device wait is required.
//
// When two adjacent free blocks are coalesced, their frontiers are merged
// using `iree_async_frontier_merge()` (component-wise maximum across all
// axes). The merged block is safe for reuse only when BOTH predecessors'
// causal histories are satisfied.
//
// If a frontier merge exceeds the inline capacity (too many distinct axes),
// the block is marked **tainted**. A tainted frontier conservatively means
// "not safe for zero-sync reuse" — the async allocator must wait for device
// confirmation before reusing the block. Taint is self-healing: when a
// tainted block is eventually allocated and later freed again, it gets a
// fresh frontier from its new usage context.
//
// ### Memory overhead
//
// The TLSF data structure itself uses approximately:
//   - 8.3 KB for the free list matrix (64 FL x 32 SL x 4 bytes per head
//     index, plus bitmaps)
//   - Per-block node: 40 bytes fixed fields + 8 bytes frontier header +
//     16 bytes per frontier entry. With the default 8-entry capacity:
//     40 + 8 + 128 = 176 bytes per block node.
//   - The block node pool grows dynamically as allocations fragment the
//     range into more blocks. A 1 GB range with 256-byte minimum blocks
//     has at most ~4M blocks (700 MB metadata). In practice, the number
//     of live blocks is much smaller — a well-utilized pool has O(100s)
//     of blocks, not millions.
//
// ### Thread safety
//
// The TLSF allocator is NOT thread-safe. All calls must be serialized by the
// caller. The async allocator provides concurrency at a higher level: freed
// blocks are staged in a lock-free queue and drained into TLSF under a
// per-pool mutex, so only one thread touches the TLSF at a time.

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Number of first-level bins. One per bit position in a uint64_t, covering
// block sizes from 2^0 to 2^63.
#define IREE_HAL_MEMORY_TLSF_FL_COUNT 64

// Log2 of the number of second-level sub-bins per first-level bin.
// 2^5 = 32 sub-bins provides ~3.1% size granularity within each FL range.
#define IREE_HAL_MEMORY_TLSF_SL_LOG2 5

// Number of second-level sub-bins per first-level bin.
#define IREE_HAL_MEMORY_TLSF_SL_COUNT (1 << IREE_HAL_MEMORY_TLSF_SL_LOG2)

// Default death frontier capacity per free block. 8 entries (136 bytes
// including the frontier header) covers single-queue, multi-queue, and
// collective workloads without taint. Multi-GPU training may need higher.
#define IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY 8

// Minimum alignment for all allocations. Ensures that the FL/SL decomposition
// has meaningful sub-bins at the lowest levels (FL_MIN >= 4 gives 32 sub-bins
// within the [16, 32) range).
#define IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT 16

// Opaque handle to a block within the TLSF allocator. This is an index into
// the internal block node pool. The caller receives a handle from allocate()
// and passes it back to free(). Handles are stable across allocator operations
// (they are indices, not pointers).
typedef uint32_t iree_hal_memory_tlsf_block_index_t;

// Sentinel value indicating no block (analogous to NULL for pointers).
#define IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE UINT32_MAX

// Bitfield flags for a block's current state. Callers typically check these
// on the allocation result to determine frontier validity.
typedef uint32_t iree_hal_memory_tlsf_block_flags_t;
enum iree_hal_memory_tlsf_block_flag_bits_e {
  IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_NONE = 0u,

  // Block is free (in a FL/SL free list, not allocated to a caller).
  IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE = 1u << 0,

  // Block is the physically last block in the managed range. Its right
  // neighbor does not exist; coalescing to the right is not possible.
  IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST = 1u << 1,

  // Block's death frontier is tainted: a prior coalescing operation tried to
  // JOIN two frontiers that together exceeded the inline capacity. The
  // frontier data is zeroed and cannot be used for dominance checking. The
  // async allocator must treat this block as conservatively "not yet safe
  // for zero-sync reuse" and fall back to device-side confirmation.
  //
  // Taint is self-healing: when a tainted block is allocated and later freed
  // with a new death frontier, the fresh frontier replaces the taint. The
  // flag is cleared on allocation (the block leaves the free list and its
  // frontier is no longer meaningful until the next free).
  IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED = 1u << 2,
};

// Fixed-size metadata for a single block in the TLSF allocator. Each block
// represents a contiguous sub-range of the managed offset space. Blocks are
// stored in a flat array with stride arithmetic — the actual storage size per
// block includes trailing inline frontier data (not represented in this
// struct).
//
// This struct is exposed in the header so that sizeof() is available for
// IREE_STRUCT_LAYOUT block stride computation, but callers must not access
// block fields directly. All access goes through the TLSF API.
typedef struct iree_hal_memory_tlsf_block_t {
  // Start offset of this block within [0, range_length).
  iree_device_size_t offset;

  // Length of this block in bytes.
  iree_device_size_t length;

  // Index of the physically adjacent left neighbor (lower address), or
  // IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE if this is the first block.
  iree_hal_memory_tlsf_block_index_t prev_physical;

  // Index of the physically adjacent right neighbor (higher address), or
  // IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE if this is the last block.
  iree_hal_memory_tlsf_block_index_t next_physical;

  // Previous block in the same FL/SL free list (if free), or in the unused
  // node list (if recycled). IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE if head.
  iree_hal_memory_tlsf_block_index_t prev_free;

  // Next block in the same FL/SL free list (if free), or in the unused node
  // list (if recycled). IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE if tail.
  iree_hal_memory_tlsf_block_index_t next_free;

  // Current block state (see iree_hal_memory_tlsf_block_flag_bits_e).
  iree_hal_memory_tlsf_block_flags_t flags;
} iree_hal_memory_tlsf_block_t;

// Options for creating a TLSF allocator instance.
typedef struct iree_hal_memory_tlsf_options_t {
  // Total range to manage: offsets [0, range_length). Must be > 0.
  // The range represents an abstract offset space — it could map to GPU VRAM,
  // host memory, a file, or any contiguous address space. The TLSF allocator
  // does not access the underlying storage.
  iree_device_size_t range_length;

  // Alignment for all returned offsets and allocated lengths. Every allocation
  // offset will be a multiple of this value, and every allocated length will
  // be rounded up to a multiple of this value. Must be a power of two and
  // >= IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT (16).
  //
  // This also determines the minimum block size: blocks smaller than the
  // alignment cannot be created. The alignment value is the floor of the
  // TLSF size-class hierarchy.
  iree_device_size_t alignment;

  // Initial capacity of the block node pool (number of block metadata
  // entries to pre-allocate). The pool grows dynamically when exhausted,
  // doubling in capacity. Set to 0 to use a default derived from
  // range_length / alignment (clamped to [64, 4096]).
  //
  // Each block node costs approximately (40 + 8 + 16 * frontier_capacity)
  // bytes of host memory.
  iree_host_size_t initial_block_capacity;

  // Maximum number of frontier entries per free block. Each free block stores
  // an inline `iree_async_frontier_t` with up to this many (axis, epoch)
  // pairs. When coalescing merges frontiers that together require more than
  // this many entries, the merged block is marked tainted.
  //
  // Guideline for sizing:
  //   1 — single-queue inference (minimal, any coalescing with cross-queue
  //       frees will taint)
  //   4 — single-device, multi-queue or small collective workloads
  //   8 — (default) multi-queue with moderate coalescing headroom
  //  12 — 8-GPU tensor-parallel training
  //  24 — 64-GPU DP+TP training
  //
  // Set to 0 to use IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY (8).
  uint8_t frontier_capacity;
} iree_hal_memory_tlsf_options_t;

// Describes the result of a successful allocation from the TLSF allocator.
typedef struct iree_hal_memory_tlsf_allocation_t {
  // Start offset of the allocated range within [0, range_length). Guaranteed
  // to be a multiple of the allocator's alignment.
  iree_device_size_t offset;

  // Length of the allocated range in bytes. May exceed the requested length
  // due to alignment rounding or because the block could not be split (the
  // remainder would have been smaller than the minimum block size). Guaranteed
  // to be a multiple of the allocator's alignment.
  iree_device_size_t length;

  // Opaque handle for this allocation. Pass to iree_hal_memory_tlsf_free()
  // to release the range back to the allocator.
  iree_hal_memory_tlsf_block_index_t block_index;

  // The death frontier that was attached to this block when it was in the
  // free list. This is the causal snapshot from the block's previous
  // deallocation. Points into the block's inline frontier storage (valid and
  // stable while the block is allocated). NULL if the block had no frontier
  // entries (e.g., the initial free block or a block freed with a NULL
  // frontier).
  //
  // The async allocator checks whether the requester's current frontier
  // dominates this death frontier. If yes, the memory is safe for zero-sync
  // reuse. If not, a device wait or alternative block selection is needed.
  const iree_async_frontier_t* death_frontier;

  // Flags from the block at allocation time. Check
  // IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED to determine whether the death
  // frontier is trustworthy for dominance checking.
  iree_hal_memory_tlsf_block_flags_t block_flags;
} iree_hal_memory_tlsf_allocation_t;

// Running statistics for a TLSF allocator. All counters are maintained
// incrementally — querying stats is O(1) with no internal walks.
typedef struct iree_hal_memory_tlsf_stats_t {
  // Total bytes currently occupied by live allocations.
  iree_device_size_t bytes_allocated;

  // Total bytes currently in free blocks (= range_length - bytes_allocated).
  iree_device_size_t bytes_free;

  // Number of currently live allocations (blocks returned by allocate() that
  // have not yet been freed).
  uint32_t allocation_count;

  // Number of free blocks in the FL/SL free lists. A lower count relative to
  // bytes_free indicates less fragmentation (large contiguous free regions).
  uint32_t free_block_count;

  // Cumulative number of coalescing operations where the frontier merge
  // overflowed inline capacity, resulting in a tainted block. A nonzero value
  // indicates the frontier_capacity may be too small for the workload's axis
  // diversity.
  uint64_t tainted_coalesce_count;
} iree_hal_memory_tlsf_stats_t;

// The TLSF allocator instance. Initialized with
// iree_hal_memory_tlsf_initialize() and cleaned up with
// iree_hal_memory_tlsf_deinitialize(). All fields are implementation details
// and must not be accessed directly by callers.
typedef struct iree_hal_memory_tlsf_t {
  // Total managed range: [0, range_length).
  iree_device_size_t range_length;

  // Minimum block size and allocation alignment.
  iree_device_size_t alignment;

  // FL index corresponding to alignment (= ctz(alignment)). FL bins below
  // this index are unused because no block can be smaller than alignment.
  uint8_t fl_min;

  // Maximum frontier entries per block (set at creation time).
  uint8_t frontier_capacity;

  // First-level bitmap: bit i set means sl_bitmaps[i] has at least one set
  // bit (some SL bin in FL level i is non-empty).
  uint64_t fl_bitmap;

  // Second-level bitmaps: one per FL level. Bit j set means
  // free_lists[i][j] is non-empty.
  uint32_t sl_bitmaps[IREE_HAL_MEMORY_TLSF_FL_COUNT];

  // Free list heads: each cell is the index of the first free block in that
  // size class, or IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE if empty.
  iree_hal_memory_tlsf_block_index_t free_lists[IREE_HAL_MEMORY_TLSF_FL_COUNT]
                                               [IREE_HAL_MEMORY_TLSF_SL_COUNT];

  // Block node pool: a flat array of fixed-stride block nodes. Each node
  // contains the block's fixed fields followed by inline frontier storage.
  // Accessed via stride arithmetic from internal helpers.
  uint8_t* block_storage;

  // Bytes per block node in the pool (fixed fields + frontier header +
  // frontier_capacity * entry_size, aligned up).
  iree_host_size_t block_stride;

  // Byte offset from block start to the iree_async_frontier_t header within
  // each block node.
  iree_host_size_t frontier_offset;

  // Number of block node slots that have been initialized (high water mark).
  iree_host_size_t block_count;

  // Total allocated slots in block_storage.
  iree_host_size_t block_capacity;

  // Head of the unused node free list. When blocks are coalesced, the
  // absorbed block's node is returned here for reuse. When a new node is
  // needed (for splitting), it is popped from here (or the pool is grown
  // if empty). Uses the next_free field for linking.
  iree_hal_memory_tlsf_block_index_t unused_node_head;

  // Running statistics (updated incrementally on every alloc/free).
  iree_device_size_t bytes_allocated;
  iree_device_size_t bytes_free;
  uint32_t allocation_count;
  uint32_t free_block_count;
  uint64_t tainted_coalesce_count;

  // Host allocator for block_storage growth.
  iree_allocator_t host_allocator;
} iree_hal_memory_tlsf_t;

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

// Initializes a TLSF allocator managing the offset range [0, range_length).
// The entire range starts as a single free block with an empty death frontier.
//
// |options| configures the range, alignment, block pool capacity, and frontier
// capacity (see iree_hal_memory_tlsf_options_t documentation for defaults).
//
// |host_allocator| is used for the internal block node pool and will be stored
// for future pool growth. It is not used on the allocation/free hot path after
// the pool reaches steady-state size.
//
// Returns IREE_STATUS_INVALID_ARGUMENT if options are invalid (zero range,
// non-power-of-two alignment, alignment below minimum, range not a multiple
// of alignment).
iree_status_t iree_hal_memory_tlsf_initialize(
    iree_hal_memory_tlsf_options_t options, iree_allocator_t host_allocator,
    iree_hal_memory_tlsf_t* out_tlsf);

// Deinitializes a TLSF allocator, freeing all internal metadata.
// Does NOT free any backing memory (the TLSF manages offsets, not memory).
// The caller is responsible for ensuring no allocated blocks are leaked
// (in debug builds, a warning is emitted if allocation_count > 0).
void iree_hal_memory_tlsf_deinitialize(iree_hal_memory_tlsf_t* tlsf);

// Allocates a contiguous range of at least |length| bytes.
//
// The returned offset is aligned to the allocator's alignment, and the
// returned length is rounded up to a multiple of alignment. The length may
// exceed |length| if the block could not be split (the remainder would have
// been smaller than the minimum block size).
//
// The allocation result includes the death frontier and flags from the free
// block that was selected. The async allocator uses these for dominance
// checking (see iree_hal_memory_tlsf_allocation_t documentation).
//
// Returns IREE_STATUS_INVALID_ARGUMENT if length is 0.
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if no free block is large enough.
iree_status_t iree_hal_memory_tlsf_allocate(
    iree_hal_memory_tlsf_t* tlsf, iree_device_size_t length,
    iree_hal_memory_tlsf_allocation_t* out_allocation);

// Frees a previously allocated block, returning its offset range to the
// allocator for future reuse.
//
// |block_index| is the handle returned by iree_hal_memory_tlsf_allocate().
// |death_frontier| is the causal snapshot to attach to the freed block. May
// be NULL for an empty frontier (the block is immediately available for
// zero-sync reuse by any requester). The frontier entries are copied into the
// block's inline storage; the caller retains ownership of |death_frontier|.
//
// The block is coalesced with physically adjacent free neighbors:
//   - Left coalesce: if the left neighbor is free, merge frontiers and extend
//     the left block's range.
//   - Right coalesce: if the right neighbor is free, merge frontiers and
//     extend this block's range.
//   - Frontier merge: component-wise maximum via iree_async_frontier_merge().
//     If the merge overflows inline capacity, the merged block is marked
//     tainted (IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED) and the
//     tainted_coalesce_count diagnostic counter is incremented.
//
// This function cannot fail. Coalescing is always possible, bitmap updates are
// pure computation, and frontier merge overflow is handled by setting the
// taint flag (not by returning an error).
void iree_hal_memory_tlsf_free(iree_hal_memory_tlsf_t* tlsf,
                               iree_hal_memory_tlsf_block_index_t block_index,
                               const iree_async_frontier_t* death_frontier);

// Copies the allocator's running statistics into |out_stats|. O(1) — all
// counters are maintained incrementally on every alloc/free operation.
void iree_hal_memory_tlsf_query_stats(const iree_hal_memory_tlsf_t* tlsf,
                                      iree_hal_memory_tlsf_stats_t* out_stats);

// Returns the length of the largest free block. O(1) via bitmap scan: finds
// the highest populated FL/SL bin and reads the head block's actual length.
// Returns 0 if no free blocks exist.
iree_device_size_t iree_hal_memory_tlsf_largest_free_block(
    const iree_hal_memory_tlsf_t* tlsf);

// Returns the death frontier of a block (free or allocated). The returned
// pointer is into the block's inline frontier storage and remains valid until
// the block is freed (if currently allocated) or allocated (if currently
// free). Returns NULL if the frontier has zero entries.
const iree_async_frontier_t* iree_hal_memory_tlsf_block_death_frontier(
    const iree_hal_memory_tlsf_t* tlsf,
    iree_hal_memory_tlsf_block_index_t block_index);

// Returns the flags of a block (free or allocated). Useful for inspecting
// TAINTED status or checking whether a block is free.
iree_hal_memory_tlsf_block_flags_t iree_hal_memory_tlsf_block_flags(
    const iree_hal_memory_tlsf_t* tlsf,
    iree_hal_memory_tlsf_block_index_t block_index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_TLSF_H_
