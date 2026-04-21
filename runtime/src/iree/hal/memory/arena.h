// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_ARENA_H_
#define IREE_HAL_MEMORY_ARENA_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Arena Offset Allocator
//===----------------------------------------------------------------------===//
//
// A bump-pointer offset allocator over a fixed range [0, capacity). Allocations
// advance a monotonic pointer; individual allocations cannot be freed. Space is
// reclaimed only when ALL outstanding acquisitions have been released, at which
// point the pointer resets to 0 and the arena is available for a new batch.
//
// ## Why an arena
//
// For allocation patterns where all items share a lifetime - allocated
// incrementally during one batch and freed all at once when the batch
// completes - an arena is simpler and faster than TLSF or a block pool:
//
//   - O(1) acquire: align and advance a pointer
//   - O(1) release: decrement a counter, JOIN a frontier
//   - Implicit bulk reset when the last release brings the count to zero
//   - Zero fragmentation: no individual holes, no free list management
//   - Zero per-allocation metadata: no block nodes, no handles
//
// ## How it works
//
// The arena manages offsets within [0, capacity). Acquisitions bump the
// pointer forward by the requested length (aligned to the per-call alignment).
// Each release decrements the outstanding count and JOINs its death frontier
// into an accumulator. When the count reaches zero, the arena resets:
//
//   - Bump pointer returns to 0
//   - Accumulated frontier becomes the "previous batch" frontier
//   - Accumulator clears for the next batch
//
// Future acquisitions return the previous batch's frontier, allowing the async
// allocator to check dominance for zero-sync reuse of the scratch memory.
//
// ## Death frontiers
//
// Unlike the block pool (per-block frontiers) and TLSF (per-free-block
// frontiers with coalescing), the arena has a single frontier for the entire
// range. Each release JOINs (component-wise maximum) its frontier into the
// accumulator. When the arena resets, the accumulated frontier represents
// "all work that used any part of this arena has been submitted at these
// causal positions."
//
// If a frontier JOIN overflows the inline capacity (too many distinct axes),
// the arena is marked tainted. Taint is self-healing: if the next batch's
// releases produce a frontier that fits, the taint clears on reset.
//
// ## Thread safety
//
// The arena is NOT thread-safe. All calls must be serialized by the caller.
// The async allocator provides concurrency at a higher level: it serializes
// access to each pool via a per-pool mutex.

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Default death frontier capacity. Since the arena JOINs across all releases
// in a batch, the capacity needs to cover the total number of distinct axes
// across all releases - not just one allocation's axes. 16 provides headroom
// for multi-queue, multi-device workloads without taint.
#define IREE_HAL_MEMORY_ARENA_DEFAULT_FRONTIER_CAPACITY 16

// Bitfield flags reported with each acquisition.
typedef uint32_t iree_hal_memory_arena_flags_t;
enum iree_hal_memory_arena_flag_bits_e {
  IREE_HAL_MEMORY_ARENA_FLAG_NONE = 0u,

  // The previous batch's death frontier is tainted: a frontier JOIN during
  // accumulation overflowed the inline capacity. The frontier data is zeroed
  // and cannot be used for dominance checking. The async allocator must treat
  // this arena's memory as conservatively "not yet safe for zero-sync reuse."
  //
  // Taint is self-healing: if the next batch's releases produce a frontier
  // that fits within capacity, the taint clears on reset.
  IREE_HAL_MEMORY_ARENA_FLAG_TAINTED = 1u << 0,
};

// Options for creating an arena instance.
typedef struct iree_hal_memory_arena_options_t {
  // Total managed range: [0, capacity). Must be > 0.
  iree_device_size_t capacity;

  // Maximum number of frontier entries for the death frontier accumulator.
  // Set to 0 to use IREE_HAL_MEMORY_ARENA_DEFAULT_FRONTIER_CAPACITY.
  uint16_t frontier_capacity;
} iree_hal_memory_arena_options_t;

// Describes the result of a successful acquisition from the arena.
typedef struct iree_hal_memory_arena_allocation_t {
  // Start offset of the allocated region within [0, capacity).
  iree_device_size_t offset;

  // The death frontier from the previous batch. This is the JOIN of all
  // frontiers from the previous batch's releases. Points into the arena's
  // inline storage (valid and stable while the arena exists). NULL if no
  // previous batch exists or the previous batch had no frontier entries.
  //
  // The async allocator checks whether the requester's current frontier
  // dominates this death frontier. If yes, the memory is safe for zero-sync
  // reuse. If not, a device wait is needed.
  const iree_async_frontier_t* death_frontier;

  // Flags from the arena at acquisition time. Check
  // IREE_HAL_MEMORY_ARENA_FLAG_TAINTED to determine whether the death
  // frontier is trustworthy for dominance checking.
  iree_hal_memory_arena_flags_t flags;
} iree_hal_memory_arena_allocation_t;

// Running statistics for an arena. O(1) to query - all values are maintained
// incrementally.
typedef struct iree_hal_memory_arena_stats_t {
  // Total arena capacity in bytes.
  iree_device_size_t capacity;

  // Current bump pointer position (bytes consumed by acquisitions +
  // alignment padding).
  iree_device_size_t bytes_used;

  // Number of currently outstanding acquisitions (not yet released).
  uint32_t allocation_count;
} iree_hal_memory_arena_stats_t;

// The arena instance. Allocated with iree_hal_memory_arena_allocate() and
// freed with iree_hal_memory_arena_free(). All fields are implementation
// details and must not be accessed directly by callers.
typedef struct iree_hal_memory_arena_t {
  // --- Read-only after creation ---

  // Total managed range: [0, capacity).
  iree_device_size_t capacity;

  // Maximum frontier entries per slot.
  uint16_t frontier_capacity;

  // Host allocator used for the struct allocation itself.
  iree_allocator_t host_allocator;

  // Pointers into trailing storage (set once during allocate, stable for
  // lifetime). Each points to an iree_async_frontier_t with inline entry
  // storage for frontier_capacity entries.
  iree_async_frontier_t* previous_frontier;
  iree_async_frontier_t* accumulator;

  // --- Mutable state ---

  // Current bump pointer position: the next acquire starts from here
  // (after alignment).
  iree_device_size_t used;

  // Number of outstanding acquisitions.
  uint32_t allocation_count;

  // True if the current batch's frontier accumulation has overflowed.
  // Becomes the tainted flag on reset.
  bool accumulator_tainted;

  // True if the previous batch's frontier is tainted (overflow during
  // accumulation). Reported to acquire() callers.
  bool tainted;
} iree_hal_memory_arena_t;

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

// Allocates an arena managing the offset range [0, capacity).
// The arena starts empty with no previous batch frontier.
//
// |host_allocator| is used for the arena struct allocation (a single
// cache-line-aligned allocation including trailing frontier storage).
//
// Returns IREE_STATUS_INVALID_ARGUMENT if capacity is 0.
//
// Thread-safety: NOT thread-safe. Must be called before any concurrent access.
iree_status_t iree_hal_memory_arena_allocate(
    iree_hal_memory_arena_options_t options, iree_allocator_t host_allocator,
    iree_hal_memory_arena_t** out_arena);

// Frees an arena, releasing all memory.
// In debug builds, asserts if any acquisitions are still outstanding.
//
// Thread-safety: NOT thread-safe. Must be called after all access has ceased.
void iree_hal_memory_arena_free(iree_hal_memory_arena_t* arena);

// Acquires a region of |length| bytes with the given |alignment| from the
// arena.
//
// |alignment| must be a power of two and > 0. The returned offset is aligned
// to this value. The arena's bump pointer advances by the aligned offset plus
// the requested length (alignment padding is consumed from the arena's
// capacity).
//
// The allocation result includes the previous batch's death frontier and flags
// for dominance checking by the async allocator.
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the arena does not have enough
// remaining capacity (after alignment padding) to satisfy the request.
// Returns IREE_STATUS_INVALID_ARGUMENT if length is 0 or alignment is not a
// power of two.
//
// Thread-safety: NOT thread-safe.
iree_status_t iree_hal_memory_arena_acquire(
    iree_hal_memory_arena_t* arena, iree_device_size_t length,
    iree_device_size_t alignment,
    iree_hal_memory_arena_allocation_t* out_allocation);

// Releases one outstanding acquisition.
//
// |death_frontier| is the causal snapshot to JOIN into the batch accumulator.
// May be NULL for an empty frontier. The frontier entries are merged into the
// arena's accumulator; the caller retains ownership of |death_frontier|.
//
// When the last outstanding acquisition is released (allocation_count reaches
// 0), the arena resets: the bump pointer returns to 0, the accumulated
// frontier becomes the previous batch frontier, and the accumulator clears.
//
// If a frontier JOIN overflows the inline capacity, the arena is marked
// tainted (IREE_HAL_MEMORY_ARENA_FLAG_TAINTED on the next batch).
//
// Thread-safety: NOT thread-safe.
void iree_hal_memory_arena_release(iree_hal_memory_arena_t* arena,
                                   const iree_async_frontier_t* death_frontier);

// Copies the arena's running statistics into |out_stats|. O(1).
//
// Thread-safety: NOT thread-safe.
void iree_hal_memory_arena_query_stats(
    const iree_hal_memory_arena_t* arena,
    iree_hal_memory_arena_stats_t* out_stats);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_ARENA_H_
