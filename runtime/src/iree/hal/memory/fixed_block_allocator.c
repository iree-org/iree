// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/fixed_block_allocator.h"

#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns a pointer to the block metadata at |block_index| via stride
// arithmetic on the trailing block_storage FAM.
static inline iree_hal_memory_fixed_block_allocator_block_t*
iree_hal_memory_fixed_block_allocator_block_at(
    const iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index) {
  return (
      iree_hal_memory_fixed_block_allocator_block_t*)(pool->block_storage +
                                                      (iree_host_size_t)
                                                              block_index *
                                                          pool->block_stride);
}

// Returns a pointer to the inline frontier of a block.
static inline iree_async_frontier_t*
iree_hal_memory_fixed_block_allocator_block_frontier_at(
    const iree_hal_memory_fixed_block_allocator_t* pool,
    iree_hal_memory_fixed_block_allocator_block_t* block) {
  return (iree_async_frontier_t*)((uint8_t*)block + pool->frontier_offset);
}

static inline void iree_hal_memory_fixed_block_allocator_release_bit(
    iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index) {
  uint16_t word_index = (uint16_t)(block_index / 64);
  int bit = block_index % 64;
  uint64_t bit_mask = UINT64_C(1) << bit;
  uint64_t old IREE_ATTRIBUTE_UNUSED = iree_atomic_fetch_and(
      &pool->bitmap[word_index], ~bit_mask, iree_memory_order_release);
  IREE_ASSERT(old & bit_mask, "double-release: block %u was not acquired",
              (unsigned)block_index);
  iree_atomic_fetch_sub(&pool->allocation_count, 1, iree_memory_order_relaxed);
}

iree_status_t iree_hal_memory_fixed_block_allocator_allocate(
    iree_hal_memory_fixed_block_allocator_options_t options,
    iree_allocator_t host_allocator,
    iree_hal_memory_fixed_block_allocator_t** out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;

  // Validate options.
  if (options.block_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "block_size must be > 0");
  }
  if (options.block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "block_count must be > 0");
  }
  if (options.block_count > IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "block_count (%u) exceeds "
        "IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS (%u)",
        (unsigned)options.block_count,
        (unsigned)IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS);
  }
  // Verify that the total offset range [0, block_count * block_size) doesn't
  // overflow iree_device_size_t. If it did, multiple blocks would map to the
  // same offset.
  iree_device_size_t total_range = 0;
  if (!iree_device_size_checked_mul(options.block_count, options.block_size,
                                    &total_range)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "block_count * block_size overflows");
  }

  if (options.frontier_capacity == 0) {
    options.frontier_capacity =
        IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_DEFAULT_FRONTIER_CAPACITY;
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Compute per-block metadata layout using overflow-checked struct math.
  // Each block node is: [flags] [padding] [frontier header] [entries]
  iree_host_size_t frontier_offset = 0;
  iree_host_size_t block_stride = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          sizeof(iree_hal_memory_fixed_block_allocator_block_t), &block_stride,
          IREE_STRUCT_FIELD_ALIGNED(1, iree_async_frontier_t,
                                    iree_alignof(iree_async_frontier_entry_t),
                                    &frontier_offset),
          IREE_STRUCT_FIELD(options.frontier_capacity,
                            iree_async_frontier_entry_t, NULL)));

  // Compute total allocation size: fixed struct (including bitmap) plus
  // trailing per-block metadata FAM. All arithmetic is overflow-checked.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_hal_memory_fixed_block_allocator_t), &total_size,
              IREE_STRUCT_ARRAY_FIELD(options.block_count, block_stride,
                                      uint8_t, NULL)));

  // Single cache-line-aligned allocation for the entire allocator.
  iree_hal_memory_fixed_block_allocator_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc_aligned(host_allocator, total_size,
                                    iree_hardware_destructive_interference_size,
                                    /*offset=*/0, (void**)&pool));

  // Initialize pool fields.
  pool->block_size = options.block_size;
  pool->block_count = options.block_count;
  pool->frontier_capacity = options.frontier_capacity;
  pool->word_count =
      (uint16_t)((options.block_count + 63) / 64);  // ceil(block_count / 64)
  pool->block_stride = block_stride;
  pool->frontier_offset = frontier_offset;
  pool->host_allocator = host_allocator;
  iree_atomic_store(&pool->alloc_hint_word, 0, iree_memory_order_relaxed);
  iree_atomic_store(&pool->allocation_count, 0, iree_memory_order_relaxed);

  // Initialize all bitmap words to 0 (all blocks free).
  for (uint16_t w = 0; w < IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BITMAP_WORDS;
       ++w) {
    iree_atomic_store(&pool->bitmap[w], 0, iree_memory_order_relaxed);
  }

  // Set trailing invalid bits in the last word to 1 (permanently acquired)
  // so they are never returned by acquire. When block_count is not a multiple
  // of 64, the last word has (64 - block_count % 64) invalid bits at the
  // high end.
  uint32_t blocks_in_last_word = options.block_count % 64;
  if (blocks_in_last_word != 0) {
    // Set bits [blocks_in_last_word, 63] to 1.
    uint64_t invalid_mask = ~((UINT64_C(1) << blocks_in_last_word) - 1);
    iree_atomic_store(&pool->bitmap[pool->word_count - 1], invalid_mask,
                      iree_memory_order_relaxed);
  }

  // Initialize per-block metadata: flags = NONE, frontier = empty.
  for (uint32_t i = 0; i < options.block_count; ++i) {
    iree_hal_memory_fixed_block_allocator_block_t* block =
        iree_hal_memory_fixed_block_allocator_block_at(pool, i);
    block->flags = IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_NONE;
    iree_async_frontier_t* frontier =
        iree_hal_memory_fixed_block_allocator_block_frontier_at(pool, block);
    iree_async_frontier_initialize(frontier, 0);
  }

  *out_pool = pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_memory_fixed_block_allocator_free(
    iree_hal_memory_fixed_block_allocator_t* pool) {
  if (!pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  uint32_t count =
      iree_atomic_load(&pool->allocation_count, iree_memory_order_relaxed);
  if (count > 0) {
    IREE_ASSERT(false,
                "fixed-block allocator free with %" PRIu32
                " leaked acquisitions",
                count);
  }
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_allocator_free_aligned(host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_memory_fixed_block_allocator_try_acquire(
    iree_hal_memory_fixed_block_allocator_t* pool,
    iree_hal_memory_fixed_block_allocator_allocation_t* out_allocation,
    iree_hal_memory_fixed_block_allocator_acquire_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_allocation);
  IREE_ASSERT_ARGUMENT(out_result);
  memset(out_allocation, 0, sizeof(*out_allocation));
  *out_result = IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED;

  const uint16_t word_count = pool->word_count;

  // Start scanning from the roving hint for better locality.
  uint16_t start_word = (uint16_t)iree_atomic_load(&pool->alloc_hint_word,
                                                   iree_memory_order_relaxed);
  if (start_word >= word_count) start_word = 0;

  // Scan all bitmap words starting from the hint, wrapping around.
  uint16_t word_index = start_word;
  for (uint16_t pass = 0; pass < word_count; ++pass) {
    // Load the word. If all bits are set, skip (no free blocks here).
    uint64_t word =
        iree_atomic_load(&pool->bitmap[word_index], iree_memory_order_acquire);

    while (word != ~UINT64_C(0)) {
      // Find the first clear bit (first free block).
      int bit = iree_math_count_trailing_zeros_u64(~word);

      // Attempt to claim it with atomic OR.
      uint64_t bit_mask = UINT64_C(1) << bit;
      uint64_t old = iree_atomic_fetch_or(&pool->bitmap[word_index], bit_mask,
                                          iree_memory_order_acq_rel);

      if (!(old & bit_mask)) {
        // We won the race — the bit was clear and we set it.
        uint32_t block_index = (uint32_t)word_index * 64 + bit;

        // Update the roving hint (best-effort, relaxed ordering).
        iree_atomic_store(&pool->alloc_hint_word, word_index,
                          iree_memory_order_relaxed);

        // Read block metadata. The acquire semantics on the bitmap OR ensure
        // that the previous owner's frontier writes (done before the bitmap
        // AND with release) are visible to us.
        iree_hal_memory_fixed_block_allocator_block_t* block =
            iree_hal_memory_fixed_block_allocator_block_at(pool, block_index);
        iree_async_frontier_t* frontier =
            iree_hal_memory_fixed_block_allocator_block_frontier_at(pool,
                                                                    block);

        out_allocation->offset =
            (iree_device_size_t)block_index * pool->block_size;
        out_allocation->block_index = block_index;
        out_allocation->death_frontier =
            (frontier->entry_count > 0) ? frontier : NULL;
        out_allocation->block_flags = block->flags;

        iree_atomic_fetch_add(&pool->allocation_count, 1,
                              iree_memory_order_relaxed);
        *out_result = IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_OK;
        return iree_ok_status();
      }

      // Lost the race — another thread claimed this bit. Reload the word
      // and try the next free bit in the same word.
      word = old | bit_mask;
    }

    // Advance to next word, wrapping around without division.
    if (++word_index >= word_count) word_index = 0;
  }

  return iree_ok_status();
}

iree_status_t iree_hal_memory_fixed_block_allocator_acquire(
    iree_hal_memory_fixed_block_allocator_t* pool,
    iree_hal_memory_fixed_block_allocator_allocation_t* out_allocation) {
  iree_hal_memory_fixed_block_allocator_acquire_result_t result =
      IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED;
  IREE_RETURN_IF_ERROR(iree_hal_memory_fixed_block_allocator_try_acquire(
      pool, out_allocation, &result));
  if (result == IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "fixed-block allocator exhausted: all %u blocks are acquired",
        (unsigned)pool->block_count);
  }
  return iree_ok_status();
}

void iree_hal_memory_fixed_block_allocator_release(
    iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index,
    const iree_async_frontier_t* death_frontier) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT(block_index < pool->block_count);

  // Write the death frontier into the block's inline storage BEFORE releasing
  // the bitmap bit. Release semantics on the bitmap AND ensure these writes
  // are visible to the next acquirer (which uses acquire on bitmap OR).
  iree_hal_memory_fixed_block_allocator_block_t* block =
      iree_hal_memory_fixed_block_allocator_block_at(pool, block_index);
  iree_async_frontier_t* block_frontier =
      iree_hal_memory_fixed_block_allocator_block_frontier_at(pool, block);

  // Clear taint — fresh frontier from this release.
  block->flags = IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_NONE;

  if (death_frontier && death_frontier->entry_count > 0) {
    if (death_frontier->entry_count <= pool->frontier_capacity) {
      // Copy the frontier (header + entries). Skip if the caller passed back
      // the block's own inline frontier pointer (acquire() exposes it via
      // out_allocation->death_frontier).
      if (block_frontier != death_frontier) {
        memcpy(block_frontier, death_frontier,
               sizeof(iree_async_frontier_t) +
                   (iree_host_size_t)death_frontier->entry_count *
                       sizeof(iree_async_frontier_entry_t));
      }
    } else {
      // Death frontier exceeds inline capacity — mark tainted.
      iree_async_frontier_initialize(block_frontier, 0);
      block->flags |= IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED;
    }
  } else {
    iree_async_frontier_initialize(block_frontier, 0);
  }

  // Release the block by clearing its bitmap bit. Release semantics ensure
  // the frontier writes above are visible before the block becomes available.
  iree_hal_memory_fixed_block_allocator_release_bit(pool, block_index);
}

void iree_hal_memory_fixed_block_allocator_restore(
    iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT(block_index < pool->block_count);
  iree_hal_memory_fixed_block_allocator_release_bit(pool, block_index);
}

void iree_hal_memory_fixed_block_allocator_query_stats(
    const iree_hal_memory_fixed_block_allocator_t* pool,
    iree_hal_memory_fixed_block_allocator_stats_t* out_stats) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_stats);
  out_stats->block_count = pool->block_count;
  out_stats->allocation_count = iree_atomic_load(
      &((iree_hal_memory_fixed_block_allocator_t*)pool)->allocation_count,
      iree_memory_order_relaxed);
}

const iree_async_frontier_t*
iree_hal_memory_fixed_block_allocator_block_death_frontier(
    const iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT(block_index < pool->block_count);
  iree_hal_memory_fixed_block_allocator_block_t* block =
      iree_hal_memory_fixed_block_allocator_block_at(pool, block_index);
  iree_async_frontier_t* frontier =
      iree_hal_memory_fixed_block_allocator_block_frontier_at(pool, block);
  return (frontier->entry_count > 0) ? frontier : NULL;
}

iree_hal_memory_fixed_block_allocator_block_flags_t
iree_hal_memory_fixed_block_allocator_block_flags(
    const iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT(block_index < pool->block_count);
  return iree_hal_memory_fixed_block_allocator_block_at(pool, block_index)
      ->flags;
}
