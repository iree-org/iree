// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf.h"

#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns a pointer to the block node at |index| via stride arithmetic.
static inline iree_hal_memory_tlsf_block_t* iree_hal_memory_tlsf_block_at(
    const iree_hal_memory_tlsf_t* tlsf,
    iree_hal_memory_tlsf_block_index_t index) {
  return (iree_hal_memory_tlsf_block_t*)(tlsf->block_storage +
                                         (iree_host_size_t)index *
                                             tlsf->block_stride);
}

// Returns a pointer to the inline frontier of a block node.
static inline iree_async_frontier_t* iree_hal_memory_tlsf_block_frontier(
    const iree_hal_memory_tlsf_t* tlsf, iree_hal_memory_tlsf_block_t* block) {
  return (iree_async_frontier_t*)((uint8_t*)block + tlsf->frontier_offset);
}

// Grows the block node pool using 2x doubling. New nodes are linked into the
// unused_node_head free list. Uses iree_allocator_grow_array for
// overflow-checked capacity doubling and realloc.
static iree_status_t iree_hal_memory_tlsf_grow_pool(
    iree_hal_memory_tlsf_t* tlsf) {
  if (tlsf->block_capacity >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "block pool at maximum capacity (%" PRIu32 ")",
                            (uint32_t)UINT32_MAX);
  }
  iree_host_size_t old_capacity = tlsf->block_capacity;
  iree_host_size_t new_capacity = old_capacity;
  IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
      tlsf->host_allocator, /*minimum_capacity=*/old_capacity + 1,
      tlsf->block_stride, &new_capacity, (void**)&tlsf->block_storage));
  // Cap at UINT32_MAX since block indices are uint32_t.
  if (new_capacity > UINT32_MAX) new_capacity = UINT32_MAX;
  // Initialize new nodes and link them into the unused list.
  for (iree_host_size_t i = old_capacity; i < new_capacity; ++i) {
    iree_hal_memory_tlsf_block_t* block =
        iree_hal_memory_tlsf_block_at(tlsf, (uint32_t)i);
    memset(block, 0, tlsf->block_stride);
    block->next_free = tlsf->unused_node_head;
    tlsf->unused_node_head = (uint32_t)i;
  }
  tlsf->block_capacity = new_capacity;
  return iree_ok_status();
}

// Allocates a block node from the pool. Grows the pool if no unused nodes
// are available.
static iree_status_t iree_hal_memory_tlsf_alloc_node(
    iree_hal_memory_tlsf_t* tlsf,
    iree_hal_memory_tlsf_block_index_t* out_index) {
  if (tlsf->unused_node_head == IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
    IREE_RETURN_IF_ERROR(iree_hal_memory_tlsf_grow_pool(tlsf));
  }
  iree_hal_memory_tlsf_block_index_t index = tlsf->unused_node_head;
  iree_hal_memory_tlsf_block_t* block =
      iree_hal_memory_tlsf_block_at(tlsf, index);
  tlsf->unused_node_head = block->next_free;
  memset(block, 0, tlsf->block_stride);
  *out_index = index;
  return iree_ok_status();
}

// Returns a block node to the unused pool.
static void iree_hal_memory_tlsf_free_node(
    iree_hal_memory_tlsf_t* tlsf, iree_hal_memory_tlsf_block_index_t index) {
  iree_hal_memory_tlsf_block_t* block =
      iree_hal_memory_tlsf_block_at(tlsf, index);
  memset(block, 0, tlsf->block_stride);
  block->next_free = tlsf->unused_node_head;
  tlsf->unused_node_head = index;
}

// Maps a block length to (FL, SL) indices for insertion into the free list
// matrix. The block goes into the bin whose range contains |length|.
static inline void iree_hal_memory_tlsf_mapping_insert(
    iree_device_size_t length, uint8_t* out_fl, uint8_t* out_sl) {
  // FL = position of the most significant set bit.
  int fl = 63 - iree_math_count_leading_zeros_u64(length);
  // SL = next SL_LOG2 bits below the MSB.
  int sl_shift = (fl > IREE_HAL_MEMORY_TLSF_SL_LOG2)
                     ? (fl - IREE_HAL_MEMORY_TLSF_SL_LOG2)
                     : 0;
  *out_fl = (uint8_t)fl;
  *out_sl =
      (uint8_t)((length >> sl_shift) & (IREE_HAL_MEMORY_TLSF_SL_COUNT - 1));
}

// Maps a requested allocation length to (FL, SL) indices for searching the
// free list matrix. We want the smallest bin that could contain a block >=
// |length|, so we round up within the current FL level's SL range.
static inline void iree_hal_memory_tlsf_mapping_search(
    iree_device_size_t length, uint8_t* out_fl, uint8_t* out_sl) {
  // For sizes that span SL bins within an FL level, round up to ensure we
  // find a block that is at least |length|. We add (1 << sl_shift) - 1
  // before extracting the SL index, which effectively rounds up.
  int fl = 63 - iree_math_count_leading_zeros_u64(length);
  if (fl > IREE_HAL_MEMORY_TLSF_SL_LOG2) {
    int sl_shift = fl - IREE_HAL_MEMORY_TLSF_SL_LOG2;
    iree_device_size_t rounded =
        length + (((iree_device_size_t)1 << sl_shift) - 1);
    // If rounding caused overflow into the next FL level, use that instead.
    int new_fl = 63 - iree_math_count_leading_zeros_u64(rounded);
    if (new_fl != fl) {
      *out_fl = (uint8_t)new_fl;
      *out_sl = 0;
      return;
    }
    *out_fl = (uint8_t)fl;
    *out_sl =
        (uint8_t)((rounded >> sl_shift) & (IREE_HAL_MEMORY_TLSF_SL_COUNT - 1));
  } else {
    *out_fl = (uint8_t)fl;
    *out_sl = (uint8_t)(length & (IREE_HAL_MEMORY_TLSF_SL_COUNT - 1));
  }
}

// Inserts a free block into the appropriate (FL, SL) free list and updates
// the bitmaps.
static void iree_hal_memory_tlsf_insert_free_block(
    iree_hal_memory_tlsf_t* tlsf, iree_hal_memory_tlsf_block_index_t index) {
  iree_hal_memory_tlsf_block_t* block =
      iree_hal_memory_tlsf_block_at(tlsf, index);
  IREE_ASSERT(block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE);

  uint8_t fl = 0, sl = 0;
  iree_hal_memory_tlsf_mapping_insert(block->length, &fl, &sl);

  // Insert at the head of the free list for this (FL, SL) bin.
  iree_hal_memory_tlsf_block_index_t old_head = tlsf->free_lists[fl][sl];
  block->prev_free = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
  block->next_free = old_head;
  if (old_head != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
    iree_hal_memory_tlsf_block_at(tlsf, old_head)->prev_free = index;
  }
  tlsf->free_lists[fl][sl] = index;

  // Set bitmap bits.
  tlsf->fl_bitmap |= (1ull << fl);
  tlsf->sl_bitmaps[fl] |= (1u << sl);

  tlsf->free_block_count++;
}

// Removes a free block from its (FL, SL) free list and updates bitmaps if
// the list becomes empty.
static void iree_hal_memory_tlsf_remove_free_block(
    iree_hal_memory_tlsf_t* tlsf, iree_hal_memory_tlsf_block_index_t index) {
  iree_hal_memory_tlsf_block_t* block =
      iree_hal_memory_tlsf_block_at(tlsf, index);
  IREE_ASSERT(block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE);

  uint8_t fl = 0, sl = 0;
  iree_hal_memory_tlsf_mapping_insert(block->length, &fl, &sl);

  // Unlink from the doubly-linked free list.
  if (block->prev_free != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
    iree_hal_memory_tlsf_block_at(tlsf, block->prev_free)->next_free =
        block->next_free;
  } else {
    // Was the head of the list.
    tlsf->free_lists[fl][sl] = block->next_free;
  }
  if (block->next_free != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
    iree_hal_memory_tlsf_block_at(tlsf, block->next_free)->prev_free =
        block->prev_free;
  }

  block->prev_free = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
  block->next_free = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;

  // Clear bitmap bits if the list is now empty.
  if (tlsf->free_lists[fl][sl] == IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
    tlsf->sl_bitmaps[fl] &= ~(1u << sl);
    if (tlsf->sl_bitmaps[fl] == 0) {
      tlsf->fl_bitmap &= ~(1ull << fl);
    }
  }

  tlsf->free_block_count--;
}

// Searches for a free block of at least |length| bytes using bitmap scans.
// Returns IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE if no suitable block exists.
static iree_hal_memory_tlsf_block_index_t
iree_hal_memory_tlsf_find_suitable_block(iree_hal_memory_tlsf_t* tlsf,
                                         iree_device_size_t length) {
  uint8_t fl = 0, sl = 0;
  iree_hal_memory_tlsf_mapping_search(length, &fl, &sl);

  // Try to find a block in the current FL level at or above the target SL.
  uint32_t sl_map = tlsf->sl_bitmaps[fl] & (~0u << sl);
  if (sl_map != 0) {
    int found_sl = iree_math_count_trailing_zeros_u32(sl_map);
    return tlsf->free_lists[fl][found_sl];
  }

  // No block in the current FL level — search higher FL levels.
  // The mask ~0ull << (fl + 1) is UB when fl==63 (shift by 64). Rewrite using
  // iree_shr: iree_shr(~0ull, 63 - fl) produces a mask with the bottom fl+1
  // bits set, and its complement gives us only the FL levels above fl.
  uint64_t fl_map = tlsf->fl_bitmap & ~iree_shr(~0ull, 63 - fl);
  if (fl_map == 0) {
    return IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
  }
  int found_fl = iree_math_count_trailing_zeros_u64(fl_map);
  // Take the first SL bin in the found FL level.
  IREE_ASSERT(tlsf->sl_bitmaps[found_fl] != 0);
  int found_sl = iree_math_count_trailing_zeros_u32(tlsf->sl_bitmaps[found_fl]);
  return tlsf->free_lists[found_fl][found_sl];
}

// Merges |source_frontier| into |target_block|'s inline frontier. If the
// merge overflows capacity, marks the target block as tainted and increments
// the diagnostic counter.
static void iree_hal_memory_tlsf_merge_frontiers(
    iree_hal_memory_tlsf_t* tlsf, iree_hal_memory_tlsf_block_t* target_block,
    const iree_async_frontier_t* source_frontier) {
  if (!source_frontier || source_frontier->entry_count == 0) return;

  iree_async_frontier_t* target_frontier =
      iree_hal_memory_tlsf_block_frontier(tlsf, target_block);

  // If target is already tainted, nothing to do — the frontier is meaningless.
  if (target_block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED) return;

  if (!iree_async_frontier_merge(target_frontier, tlsf->frontier_capacity,
                                 source_frontier)) {
    // Overflow: mark tainted and zero the frontier.
    target_block->flags |= IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED;
    target_frontier->entry_count = 0;
    tlsf->tainted_coalesce_count++;
  }
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_memory_tlsf_initialize(
    iree_hal_memory_tlsf_options_t options, iree_allocator_t host_allocator,
    iree_hal_memory_tlsf_t* out_tlsf) {
  IREE_ASSERT_ARGUMENT(out_tlsf);
  memset(out_tlsf, 0, sizeof(*out_tlsf));

  // Validate options.
  if (options.range_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range_length must be > 0");
  }
  if (options.alignment == 0) {
    options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  }
  if (options.alignment < IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "alignment (%" PRIdsz ") must be >= %" PRIdsz, options.alignment,
        (iree_device_size_t)IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT);
  }
  if (!iree_device_size_is_power_of_two(options.alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "alignment (%" PRIdsz ") must be a power of two",
                            options.alignment);
  }
  if (options.range_length < options.alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range_length (%" PRIdsz
                            ") must be >= alignment (%" PRIdsz ")",
                            options.range_length, options.alignment);
  }
  // Round range_length down to a multiple of alignment.
  options.range_length = options.range_length & ~(options.alignment - 1);
  if (options.frontier_capacity == 0) {
    options.frontier_capacity = IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY;
  }

  // Compute block node layout using overflow-checked struct math.
  // Each block node is: [fixed fields] [padding] [frontier header] [entries]
  // The frontier must be aligned for its entry type (8-byte aligned).
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t block_stride = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_memory_tlsf_block_t), &block_stride,
      IREE_STRUCT_FIELD_ALIGNED(1, iree_async_frontier_t,
                                iree_alignof(iree_async_frontier_entry_t),
                                &frontier_offset),
      IREE_STRUCT_FIELD(options.frontier_capacity, iree_async_frontier_entry_t,
                        NULL)));

  // Determine initial block pool capacity.
  iree_host_size_t initial_capacity = options.initial_block_capacity;
  if (initial_capacity == 0) {
    // Heuristic: start with enough nodes for a moderately fragmented pool.
    iree_device_size_t max_blocks = options.range_length / options.alignment;
    if (max_blocks > 4096) max_blocks = 4096;
    if (max_blocks < 64) max_blocks = 64;
    initial_capacity = (iree_host_size_t)max_blocks;
  }

  // Allocate block storage (overflow-checked array allocation).
  uint8_t* block_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, initial_capacity, block_stride, (void**)&block_storage));

  // Initialize the TLSF instance.
  out_tlsf->range_length = options.range_length;
  out_tlsf->alignment = options.alignment;
  out_tlsf->fl_min =
      (uint8_t)iree_math_count_trailing_zeros_u64(options.alignment);
  out_tlsf->frontier_capacity = options.frontier_capacity;
  out_tlsf->fl_bitmap = 0;
  memset(out_tlsf->sl_bitmaps, 0, sizeof(out_tlsf->sl_bitmaps));
  for (int fl = 0; fl < IREE_HAL_MEMORY_TLSF_FL_COUNT; ++fl) {
    for (int sl = 0; sl < IREE_HAL_MEMORY_TLSF_SL_COUNT; ++sl) {
      out_tlsf->free_lists[fl][sl] = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
    }
  }
  out_tlsf->block_storage = block_storage;
  out_tlsf->block_stride = block_stride;
  out_tlsf->frontier_offset = frontier_offset;
  out_tlsf->block_count = 0;
  out_tlsf->block_capacity = initial_capacity;
  out_tlsf->unused_node_head = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
  out_tlsf->bytes_allocated = 0;
  out_tlsf->bytes_free = options.range_length;
  out_tlsf->allocation_count = 0;
  out_tlsf->free_block_count = 0;
  out_tlsf->tainted_coalesce_count = 0;
  out_tlsf->host_allocator = host_allocator;

  // Link all pre-allocated nodes into the unused list (except node 0 which
  // we will use for the initial free block).
  for (iree_host_size_t i = 1; i < initial_capacity; ++i) {
    iree_hal_memory_tlsf_block_t* block =
        iree_hal_memory_tlsf_block_at(out_tlsf, (uint32_t)i);
    block->next_free = out_tlsf->unused_node_head;
    out_tlsf->unused_node_head = (uint32_t)i;
  }
  out_tlsf->block_count = initial_capacity;

  // Create the initial free block spanning the entire range.
  iree_hal_memory_tlsf_block_t* initial_block =
      iree_hal_memory_tlsf_block_at(out_tlsf, 0);
  initial_block->offset = 0;
  initial_block->length = options.range_length;
  initial_block->prev_physical = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
  initial_block->next_physical = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
  initial_block->prev_free = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
  initial_block->next_free = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
  initial_block->flags = IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE |
                         IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST;
  // Frontier starts empty.
  iree_async_frontier_t* initial_frontier =
      iree_hal_memory_tlsf_block_frontier(out_tlsf, initial_block);
  iree_async_frontier_initialize(initial_frontier, 0);

  // Insert the initial free block into the appropriate FL/SL bin.
  iree_hal_memory_tlsf_insert_free_block(out_tlsf, 0);

  return iree_ok_status();
}

void iree_hal_memory_tlsf_deinitialize(iree_hal_memory_tlsf_t* tlsf) {
  IREE_ASSERT_ARGUMENT(tlsf);
  if (tlsf->allocation_count > 0) {
    IREE_ASSERT(false, "TLSF deinitialize with %" PRIu32 " leaked allocations",
                tlsf->allocation_count);
  }
  iree_allocator_free(tlsf->host_allocator, tlsf->block_storage);
  memset(tlsf, 0, sizeof(*tlsf));
}

iree_status_t iree_hal_memory_tlsf_allocate(
    iree_hal_memory_tlsf_t* tlsf, iree_device_size_t length,
    iree_hal_memory_tlsf_allocation_t* out_allocation) {
  IREE_ASSERT_ARGUMENT(tlsf);
  IREE_ASSERT_ARGUMENT(out_allocation);
  memset(out_allocation, 0, sizeof(*out_allocation));

  if (length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation length must be > 0");
  }

  // Guard against overflow: if length is so large that rounding up to
  // alignment would wrap around, reject immediately. This prevents
  // near-SIZE_MAX requests from silently succeeding as tiny allocations.
  if (length > IREE_DEVICE_SIZE_MAX - (tlsf->alignment - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation length %" PRIdsz
                            " overflows when aligned to %" PRIdsz,
                            length, tlsf->alignment);
  }

  // Round up to alignment (and ensure at least minimum block size).
  iree_device_size_t aligned_length =
      iree_device_align(length, tlsf->alignment);
  if (aligned_length < tlsf->alignment) {
    aligned_length = tlsf->alignment;
  }

  // Find a suitable free block.
  iree_hal_memory_tlsf_block_index_t block_index =
      iree_hal_memory_tlsf_find_suitable_block(tlsf, aligned_length);
  if (block_index == IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "no free block of %" PRIdsz " bytes (aligned from %" PRIdsz
        " requested); largest free block is %" PRIdsz " bytes",
        aligned_length, length, iree_hal_memory_tlsf_largest_free_block(tlsf));
  }

  // Remove the block from the free list.
  iree_hal_memory_tlsf_remove_free_block(tlsf, block_index);
  iree_hal_memory_tlsf_block_t* block =
      iree_hal_memory_tlsf_block_at(tlsf, block_index);

  // Split if the remainder is large enough.
  iree_device_size_t remainder = block->length - aligned_length;
  if (remainder >= tlsf->alignment) {
    // Allocate a new node for the remainder block.
    iree_hal_memory_tlsf_block_index_t remainder_index;
    iree_status_t status =
        iree_hal_memory_tlsf_alloc_node(tlsf, &remainder_index);
    if (iree_status_is_ok(status)) {
      iree_hal_memory_tlsf_block_t* remainder_block =
          iree_hal_memory_tlsf_block_at(tlsf, remainder_index);
      remainder_block->offset = block->offset + aligned_length;
      remainder_block->length = remainder;
      remainder_block->prev_physical = block_index;
      remainder_block->next_physical = block->next_physical;
      remainder_block->flags = IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE;

      // Transfer LAST flag if the original block was last.
      if (block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST) {
        remainder_block->flags |= IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST;
        block->flags &= ~IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST;
      }

      // Update the old right neighbor's prev_physical to point to the
      // remainder.
      if (remainder_block->next_physical !=
          IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
        iree_hal_memory_tlsf_block_at(tlsf, remainder_block->next_physical)
            ->prev_physical = remainder_index;
      }

      // Link the remainder into the physical list.
      block->next_physical = remainder_index;
      block->length = aligned_length;

      // The remainder block gets an empty frontier (it was just split from
      // a block being allocated — it has no independent usage history).
      iree_async_frontier_t* remainder_frontier =
          iree_hal_memory_tlsf_block_frontier(tlsf, remainder_block);
      iree_async_frontier_initialize(remainder_frontier, 0);

      // Insert the remainder into the free list.
      iree_hal_memory_tlsf_insert_free_block(tlsf, remainder_index);

      // Update free bytes (the remainder stays free).
      tlsf->bytes_free -= aligned_length;
    } else {
      // Could not allocate a node for the remainder. Give the entire block
      // to the caller (slightly over-sized). This is not an error — it just
      // means we can't split right now.
      iree_status_ignore(status);
      tlsf->bytes_free -= block->length;
    }
  } else {
    // Cannot split — give the whole block (may be slightly over-sized).
    tlsf->bytes_free -= block->length;
  }

  // Mark the block as allocated: clear the FREE flag before populating the
  // result so the caller sees only the allocated-state flags (LAST, TAINTED).
  block->flags &= ~IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE;

  iree_async_frontier_t* frontier =
      iree_hal_memory_tlsf_block_frontier(tlsf, block);
  out_allocation->offset = block->offset;
  out_allocation->length = block->length;
  out_allocation->block_index = block_index;
  out_allocation->death_frontier =
      (frontier->entry_count > 0) ? frontier : NULL;
  out_allocation->block_flags = block->flags;

  tlsf->bytes_allocated += block->length;
  tlsf->allocation_count++;

  return iree_ok_status();
}

void iree_hal_memory_tlsf_free(iree_hal_memory_tlsf_t* tlsf,
                               iree_hal_memory_tlsf_block_index_t block_index,
                               const iree_async_frontier_t* death_frontier) {
  IREE_ASSERT_ARGUMENT(tlsf);
  IREE_ASSERT(block_index != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE);

  iree_hal_memory_tlsf_block_t* block =
      iree_hal_memory_tlsf_block_at(tlsf, block_index);
  IREE_ASSERT(!(block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE));

  // Update stats for the deallocation.
  tlsf->bytes_allocated -= block->length;
  tlsf->bytes_free += block->length;
  tlsf->allocation_count--;

  // Mark as free and clear taint (fresh frontier from this free).
  block->flags |= IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE;
  block->flags &= ~IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED;

  // Copy the death frontier into the block's inline storage.
  iree_async_frontier_t* block_frontier =
      iree_hal_memory_tlsf_block_frontier(tlsf, block);
  if (death_frontier && death_frontier->entry_count > 0) {
    // Copy entries (clamped to capacity, though the caller should have sized
    // the frontier to fit). If the death frontier itself exceeds capacity,
    // we taint immediately.
    if (death_frontier->entry_count <= tlsf->frontier_capacity) {
      memcpy(block_frontier, death_frontier,
             sizeof(iree_async_frontier_t) +
                 (iree_host_size_t)death_frontier->entry_count *
                     sizeof(iree_async_frontier_entry_t));
    } else {
      // Death frontier too large for inline storage — taint.
      iree_async_frontier_initialize(block_frontier, 0);
      block->flags |= IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED;
      tlsf->tainted_coalesce_count++;
    }
  } else {
    iree_async_frontier_initialize(block_frontier, 0);
  }

  // Coalesce with the right neighbor first (so that the left coalesce can
  // absorb the combined block).
  if (!(block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST)) {
    iree_hal_memory_tlsf_block_index_t right_index = block->next_physical;
    if (right_index != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
      iree_hal_memory_tlsf_block_t* right_block =
          iree_hal_memory_tlsf_block_at(tlsf, right_index);
      if (right_block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE) {
        // Remove the right block from its free list.
        iree_hal_memory_tlsf_remove_free_block(tlsf, right_index);

        // Merge frontiers: right into this block.
        iree_hal_memory_tlsf_merge_frontiers(
            tlsf, block,
            iree_hal_memory_tlsf_block_frontier(tlsf, right_block));

        // Absorb geometry: extend this block to cover the right block.
        block->length += right_block->length;

        // Transfer LAST flag.
        if (right_block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST) {
          block->flags |= IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST;
        }

        // Transfer taint flag. A tainted right neighbor means the merged
        // block's frontier is untrustworthy — the merge_frontiers call above
        // will have been a no-op (tainted source has entry_count==0), but the
        // taint itself must propagate. Zero the frontier to prevent stale data
        // from being misinterpreted as valid entries.
        if (right_block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED) {
          block->flags |= IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED;
          iree_hal_memory_tlsf_block_frontier(tlsf, block)->entry_count = 0;
        }

        // Update physical links.
        block->next_physical = right_block->next_physical;
        if (right_block->next_physical !=
            IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
          iree_hal_memory_tlsf_block_at(tlsf, right_block->next_physical)
              ->prev_physical = block_index;
        }

        // Release the right block node.
        iree_hal_memory_tlsf_free_node(tlsf, right_index);
      }
    }
  }

  // Coalesce with the left neighbor.
  if (block->prev_physical != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
    iree_hal_memory_tlsf_block_index_t left_index = block->prev_physical;
    iree_hal_memory_tlsf_block_t* left_block =
        iree_hal_memory_tlsf_block_at(tlsf, left_index);
    if (left_block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_FREE) {
      // Remove the left block from its free list.
      iree_hal_memory_tlsf_remove_free_block(tlsf, left_index);

      // Merge frontiers: this block into left.
      iree_hal_memory_tlsf_merge_frontiers(
          tlsf, left_block, iree_hal_memory_tlsf_block_frontier(tlsf, block));

      // Absorb geometry: extend left to cover this block.
      left_block->length += block->length;

      // Transfer LAST flag.
      if (block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST) {
        left_block->flags |= IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_LAST;
      }

      // Transfer taint flag. As with right coalesce, a tainted source means
      // the merged block's frontier cannot be trusted — zero it.
      if (block->flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED) {
        left_block->flags |= IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED;
        iree_hal_memory_tlsf_block_frontier(tlsf, left_block)->entry_count = 0;
      }

      // Update physical links.
      left_block->next_physical = block->next_physical;
      if (block->next_physical != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
        iree_hal_memory_tlsf_block_at(tlsf, block->next_physical)
            ->prev_physical = left_index;
      }

      // Release this block node and switch to inserting the left block.
      iree_hal_memory_tlsf_free_node(tlsf, block_index);
      block_index = left_index;
      block = left_block;
    }
  }

  // Insert the (possibly merged) block into the appropriate free list.
  iree_hal_memory_tlsf_insert_free_block(tlsf, block_index);
}

void iree_hal_memory_tlsf_query_stats(const iree_hal_memory_tlsf_t* tlsf,
                                      iree_hal_memory_tlsf_stats_t* out_stats) {
  IREE_ASSERT_ARGUMENT(tlsf);
  IREE_ASSERT_ARGUMENT(out_stats);
  out_stats->bytes_allocated = tlsf->bytes_allocated;
  out_stats->bytes_free = tlsf->bytes_free;
  out_stats->allocation_count = tlsf->allocation_count;
  out_stats->free_block_count = tlsf->free_block_count;
  out_stats->tainted_coalesce_count = tlsf->tainted_coalesce_count;
}

iree_device_size_t iree_hal_memory_tlsf_largest_free_block(
    const iree_hal_memory_tlsf_t* tlsf) {
  IREE_ASSERT_ARGUMENT(tlsf);
  if (tlsf->fl_bitmap == 0) return 0;

  // Find the highest populated FL level.
  int fl = 63 - iree_math_count_leading_zeros_u64(tlsf->fl_bitmap);
  IREE_ASSERT(tlsf->sl_bitmaps[fl] != 0);
  // Find the highest populated SL within that FL level.
  int sl = 31 - iree_math_count_leading_zeros_u32(tlsf->sl_bitmaps[fl]);
  // The head block of that list is the largest (or close to it).
  iree_hal_memory_tlsf_block_index_t head = tlsf->free_lists[fl][sl];
  IREE_ASSERT(head != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE);
  return iree_hal_memory_tlsf_block_at(tlsf, head)->length;
}

const iree_async_frontier_t* iree_hal_memory_tlsf_block_death_frontier(
    const iree_hal_memory_tlsf_t* tlsf,
    iree_hal_memory_tlsf_block_index_t block_index) {
  IREE_ASSERT_ARGUMENT(tlsf);
  IREE_ASSERT(block_index != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE);
  iree_hal_memory_tlsf_block_t* block =
      iree_hal_memory_tlsf_block_at(tlsf, block_index);
  iree_async_frontier_t* frontier =
      iree_hal_memory_tlsf_block_frontier(tlsf, block);
  return (frontier->entry_count > 0) ? frontier : NULL;
}

iree_hal_memory_tlsf_block_flags_t iree_hal_memory_tlsf_block_flags(
    const iree_hal_memory_tlsf_t* tlsf,
    iree_hal_memory_tlsf_block_index_t block_index) {
  IREE_ASSERT_ARGUMENT(tlsf);
  IREE_ASSERT(block_index != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE);
  return iree_hal_memory_tlsf_block_at(tlsf, block_index)->flags;
}
