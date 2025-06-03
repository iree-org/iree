// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/block_pool.h"

#include "iree/hal/drivers/amdgpu/util/bitmap.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_block_pool_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_block_pool_grow(
    iree_hal_amdgpu_block_pool_t* block_pool);

iree_status_t iree_hal_amdgpu_block_pool_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_block_pool_options_t options, hsa_agent_t agent,
    hsa_amd_memory_pool_t memory_pool, iree_allocator_t host_allocator,
    iree_hal_amdgpu_block_pool_t* out_block_pool) {
  IREE_ASSERT_ARGUMENT(out_block_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, options.block_size);

  memset(out_block_pool, 0, sizeof(*out_block_pool));

  if (!options.block_size ||
      !iree_device_size_is_power_of_two(options.block_size)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "block size must be a power-of-two; got %" PRIdsz,
                             options.block_size));
  }

  out_block_pool->libhsa = libhsa;
  out_block_pool->host_allocator = host_allocator;
  out_block_pool->agent = agent;
  out_block_pool->memory_pool = memory_pool;
  out_block_pool->block_size = options.block_size;

  // Query the memory pool for its allocation granularity.
  // This is not the minimum allocation size
  // (HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE) but the recommended size
  // to prevent internal fragmentation. We will always make allocations of this
  // size and then suballocate the block size.
  size_t alloc_rec_granule = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_amd_memory_pool_get_info(
          IREE_LIBHSA(libhsa), memory_pool,
          HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
          &alloc_rec_granule),
      "querying HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE to "
      "determine blocks/allocation");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, alloc_rec_granule);

  // If no min block count was provided we pick one as either the number that
  // will fit into the recommended allocation granule.
  const iree_host_size_t min_blocks_per_allocation =
      options.min_blocks_per_allocation
          ? options.min_blocks_per_allocation
          : iree_host_size_ceil_div(alloc_rec_granule, options.block_size);

  // Always allocate aligned to the recommended granularity.
  // This may lead to more blocks than the user requested but the extra memory
  // would likely be unused anyway (or used poorly).
  const iree_device_size_t allocation_size = iree_device_align(
      options.block_size * min_blocks_per_allocation, alloc_rec_granule);
  out_block_pool->blocks_per_allocation =
      (iree_host_size_t)(allocation_size / options.block_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, out_block_pool->blocks_per_allocation);

  iree_slim_mutex_initialize(&out_block_pool->mutex);
  iree_slim_mutex_lock(&out_block_pool->mutex);

  // Preallocate as many allocations as required to hold the requested initial
  // block count.
  iree_status_t status = iree_ok_status();
  iree_host_size_t initial_allocation_count = iree_host_size_ceil_div(
      options.initial_capacity, out_block_pool->blocks_per_allocation);
  for (iree_host_size_t i = 0; i < initial_allocation_count; ++i) {
    status = iree_hal_amdgpu_block_pool_grow(out_block_pool);
    if (!iree_status_is_ok(status)) break;
  }

  iree_slim_mutex_unlock(&out_block_pool->mutex);

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_block_pool_deinitialize(out_block_pool);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_block_pool_deinitialize(
    iree_hal_amdgpu_block_pool_t* block_pool) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Should have freed everything so we can just trim the pool to drop all
  // blocks.
  iree_hal_amdgpu_block_pool_trim(block_pool);
  IREE_ASSERT(!block_pool->allocations_head,
              "must have freed all blocks prior to deallocating the pool");

  iree_slim_mutex_deinitialize(&block_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

// Grows the block pool by one block allocation and links all of the blocks
// contained into the block pool free list.
//
// Must be called with the pool lock held.
static iree_status_t iree_hal_amdgpu_block_pool_grow(
    iree_hal_amdgpu_block_pool_t* block_pool) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate device memory. This may fail if resources are exhausted.
  IREE_AMDGPU_DEVICE_PTR uint8_t* base_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_amd_memory_pool_allocate(
          IREE_LIBHSA(block_pool->libhsa), block_pool->memory_pool,
          block_pool->blocks_per_allocation * block_pool->block_size,
          HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&base_ptr),
      "growing block pool with one block of %" PRIdsz " bytes",
      block_pool->block_size * block_pool->blocks_per_allocation);

  // Allocate host memory container for the allocation.
  iree_hal_amdgpu_block_allocation_t* block_allocation = NULL;
  iree_status_t status = iree_allocator_malloc(
      block_pool->host_allocator,
      sizeof(*block_allocation) + block_pool->blocks_per_allocation *
                                      sizeof(block_allocation->blocks[0]),
      (void**)&block_allocation);
  if (iree_status_is_ok(status)) {
    block_allocation->next = block_pool->allocations_head;
    block_allocation->base_ptr = base_ptr;
    block_allocation->used_count = 0;
    block_pool->allocations_head = block_allocation;

    // Setup all blocks to point at their relevant memory.
    // We append to the block pool free list as we go.
    for (iree_host_size_t i = 0; i < block_pool->blocks_per_allocation; ++i) {
      iree_hal_amdgpu_block_t* block = &block_allocation->blocks[i];
      block->ptr = base_ptr + i * block_pool->block_size;
      block->allocation = block_allocation;
      block->next = block_pool->free_blocks_head;
      block_pool->free_blocks_head = block;
    }
  } else {
    IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(
        IREE_LIBHSA(block_pool->libhsa), base_ptr));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_block_pool_trim(iree_hal_amdgpu_block_pool_t* block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: we could steal the whole list and free it outside of the lock but we
  // actually want to prevent anyone else from growing the pool until we've
  // completed - otherwise a sequence of trim+alloc could cause higher peak
  // usage.
  iree_slim_mutex_lock(&block_pool->mutex);

  // Preprocess the block free list to remove all blocks whose allocation has
  // no used blocks. This is so that the walk over the allocation list can free
  // the host memory that contains the blocks below.
  //
  // This isn't great but compared to the cost of calling into HSA to deallocate
  // the device memory this is nothing. Trims only happen when latency is not
  // important.
  iree_hal_amdgpu_block_t* free_block = block_pool->free_blocks_head;
  iree_hal_amdgpu_block_t* prev_free_block = NULL;
  while (free_block) {
    iree_hal_amdgpu_block_t* next_free_block = free_block->next;
    if (free_block->allocation->used_count == 0) {
      // Allocation will be freed below - unlink.
      if (free_block == block_pool->free_blocks_head) {
        block_pool->free_blocks_head = next_free_block;
        if (prev_free_block) prev_free_block->next = next_free_block;
      } else {
        prev_free_block->next = next_free_block;
      }
    } else {
      // Allocation still has uses - keep the block in the list.
      prev_free_block = free_block;
    }
    free_block = next_free_block;
  }

  // Walk each allocation and free it if it has no outstanding blocks allocated.
  // Note that we already cleaned up the free block list above.
  iree_hal_amdgpu_block_allocation_t* allocation = block_pool->allocations_head;
  iree_hal_amdgpu_block_allocation_t* prev_allocation = NULL;
  while (allocation) {
    iree_hal_amdgpu_block_allocation_t* next_allocation = allocation->next;
    if (allocation->used_count == 0) {
      // No blocks outstanding - can free and remove from the allocation list.
      IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(
          IREE_LIBHSA(block_pool->libhsa), allocation->base_ptr));
      if (allocation == block_pool->allocations_head) {
        block_pool->allocations_head = next_allocation;
        if (prev_allocation) prev_allocation->next = next_allocation;
      } else {
        prev_allocation->next = next_allocation;
      }
      iree_allocator_free(block_pool->host_allocator, allocation);
    } else {
      // Skip as blocks still outstanding.
      prev_allocation = allocation;
    }
    allocation = next_allocation;
  }

  iree_slim_mutex_unlock(&block_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_block_pool_acquire(
    iree_hal_amdgpu_block_pool_t* block_pool,
    iree_hal_amdgpu_block_t** out_block) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_block);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_block = NULL;

  iree_slim_mutex_lock(&block_pool->mutex);

  // If there are no free blocks available grow the pool by one block allocation
  // (which may allocate multiple blocks worth of memory).
  if (!block_pool->free_blocks_head) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdgpu_block_pool_grow(block_pool));
  }

  // Slice off the next free block.
  iree_hal_amdgpu_block_t* block = block_pool->free_blocks_head;
  block_pool->free_blocks_head = block->next;
  block->next = NULL;  // user may use this
  block->prev = NULL;
  memset(block->user_data, 0, sizeof(block->user_data));
  ++block->allocation->used_count;

  iree_slim_mutex_unlock(&block_pool->mutex);

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_block_pool_release(
    iree_hal_amdgpu_block_pool_t* block_pool, iree_hal_amdgpu_block_t* block) {
  IREE_ASSERT_ARGUMENT(block_pool);
  if (!block) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&block_pool->mutex);

  // Return the block to the pool free list and update the allocation tracking.
  block->next = block_pool->free_blocks_head;
  block_pool->free_blocks_head = block;
  --block->allocation->used_count;

  iree_slim_mutex_unlock(&block_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amdgpu_block_pool_release_list(
    iree_hal_amdgpu_block_pool_t* block_pool,
    iree_hal_amdgpu_block_t* block_head) {
  IREE_ASSERT_ARGUMENT(block_pool);
  if (!block_head) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&block_pool->mutex);

  // Return the blocks to the pool free list and update the allocation tracking.
  // Note that each block has allocation tracking that needs to be adjusted.
  iree_hal_amdgpu_block_t* block = block_head;
  iree_hal_amdgpu_block_t* block_tail = block;
  do {
    block_tail = block;
    --block->allocation->used_count;
    block = block->next;
  } while (block);

  // Prepend the list to the block pool free list.
  // The provided list is already linked so we just need to swap it in.
  // If we didn't have the per-block work we could do this without scanning the
  // list by taking a tail block as an argument (the caller may already have
  // it).
  block_tail->next = block_pool->free_blocks_head;
  block_pool->free_blocks_head = block_tail;

  iree_slim_mutex_unlock(&block_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}
//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_block_arena_t
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_block_arena_initialize(
    iree_hal_amdgpu_block_pool_t* block_pool,
    iree_hal_amdgpu_block_arena_t* out_arena) {
  memset(out_arena, 0, sizeof(*out_arena));
  out_arena->block_pool = block_pool;
}

void iree_hal_amdgpu_block_arena_deinitialize(
    iree_hal_amdgpu_block_arena_t* arena) {
  iree_hal_amdgpu_block_arena_reset(arena);
}

void iree_hal_amdgpu_block_arena_reset(iree_hal_amdgpu_block_arena_t* arena) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (arena->block_head != NULL) {
    iree_hal_amdgpu_block_pool_release_list(arena->block_pool,
                                            arena->block_head);
    arena->block_head = NULL;
    arena->block_tail = NULL;
  }

  arena->total_allocation_size = 0;
  arena->used_allocation_size = 0;
  arena->block_bytes_remaining = 0;

  IREE_TRACE_ZONE_END(z0);
}

iree_hal_amdgpu_block_t* iree_hal_amdgpu_block_arena_release_blocks(
    iree_hal_amdgpu_block_arena_t* arena) {
  iree_hal_amdgpu_block_t* block_head = arena->block_head;
  arena->block_head = NULL;
  arena->block_tail = NULL;
  arena->total_allocation_size = 0;
  arena->used_allocation_size = 0;
  arena->block_bytes_remaining = 0;
  return block_head;
}

iree_status_t iree_hal_amdgpu_block_arena_allocate(
    iree_hal_amdgpu_block_arena_t* arena, iree_device_size_t byte_length,
    IREE_AMDGPU_DEVICE_PTR void** out_ptr) {
  *out_ptr = NULL;

  iree_hal_amdgpu_block_pool_t* block_pool = arena->block_pool;

  // Pad length allocated so that each pointer bump is always ending at an
  // aligned address and the next allocation will start aligned.
  iree_device_size_t aligned_length =
      iree_device_align(byte_length, iree_hal_amdgpu_max_align_t);

  // Check to see if the current block (if any) has space - if not, get another.
  if (arena->block_head == NULL ||
      arena->block_bytes_remaining < aligned_length) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amdgpu_allocate_grow");
    iree_hal_amdgpu_block_t* block = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdgpu_block_pool_acquire(arena->block_pool, &block));
    block->next = NULL;
    if (arena->block_tail) {
      arena->block_tail->next = block;
    } else {
      arena->block_head = block;
    }
    arena->block_tail = block;
    arena->total_allocation_size += block_pool->block_size;
    arena->block_bytes_remaining = block_pool->block_size;
    IREE_TRACE_ZONE_END(z0);
  }

  // Slice out the allocation from the current block.
  IREE_AMDGPU_DEVICE_PTR void* ptr =
      (uint8_t*)arena->block_tail->ptr - arena->block_bytes_remaining;
  arena->block_bytes_remaining -= aligned_length;
  arena->used_allocation_size += aligned_length;
  *out_ptr = ptr;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_block_allocator_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_block_allocator_initialize(
    iree_hal_amdgpu_block_pool_t* block_pool, iree_host_size_t min_page_size,
    iree_hal_amdgpu_block_allocator_t* out_allocator) {
  // Verify preconditions.
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(min_page_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "min_page_size of %" PRIhsz
                            " bytes must be a power-of-two",
                            min_page_size);
  } else if (IREE_UNLIKELY(block_pool->block_size < min_page_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "min_page_size of %" PRIhsz
        " bytes must fit within the pooled block size of %" PRIhsz " bytes",
        min_page_size, block_pool->block_size);
  }

  // We limit our page count to how many bits we have in the usage bitmap.
  // We may use fewer bits if the granularity is large and the blocks are small.
  const iree_host_size_t max_page_count =
      IREE_HAL_AMDGPU_BLOCK_USER_DATA_SIZE * 8;
  const iree_host_size_t page_size =
      iree_max(min_page_size, block_pool->block_size / max_page_count);
  const iree_host_size_t page_count = block_pool->block_size / page_size;

  out_allocator->page_size = page_size;
  out_allocator->page_count = page_count;
  out_allocator->block_pool = block_pool;

  iree_slim_mutex_initialize(&out_allocator->mutex);
  out_allocator->block_head = NULL;
  out_allocator->block_tail = NULL;

  return iree_ok_status();
}

void iree_hal_amdgpu_block_allocator_deinitialize(
    iree_hal_amdgpu_block_allocator_t* allocator) {
  if (!allocator) return;

  IREE_ASSERT_EQ(allocator->block_head, NULL);
  IREE_ASSERT_EQ(allocator->block_tail, NULL);

  iree_slim_mutex_deinitialize(&allocator->mutex);

  memset(allocator, 0, sizeof(*allocator));
}

static iree_status_t iree_hal_amdgpu_block_allocator_allocate_with_lock(
    iree_hal_amdgpu_block_allocator_t* allocator, iree_host_size_t page_count,
    IREE_AMDGPU_DEVICE_PTR void** out_ptr,
    iree_hal_amdgpu_block_token_t* out_token) {
  // Scan the block list for sufficient contiguous free pages. The blocks are
  // roughly sorted with blocks that have free pages first and we rely on the
  // total block count being small to make this linear scan ok. We will need to
  // bucket by longest span or some other "real" allocator things if this ends
  // up not being enough.
  iree_hal_amdgpu_block_t* block = allocator->block_head;
  while (block) {
    const iree_hal_amdgpu_bitmap_t bitmap = {
        .bit_count = allocator->page_count,
        .words = &block->user_data[0],
    };
    const iree_host_size_t page_index =
        iree_hal_amdgpu_bitmap_find_first_unset_span(bitmap, 0, page_count);
    if (page_index == bitmap.bit_count) {
      // No span of sufficient size found - try the next block with free pages.
      block = block->next;
      continue;
    }
    // Span of pages found. Reserve and return the allocation.
    iree_hal_amdgpu_bitmap_set_span(bitmap, page_index, page_count);
    *out_ptr = (uint8_t*)block->ptr + page_index * allocator->page_size;
    out_token->page_count = (uint64_t)page_count;
    out_token->block = (uint64_t)block;
    return iree_ok_status();
  }

  // Acquire a new block from the block pool.
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_block_pool_acquire(allocator->block_pool, &block));

  // Reset the bitmap as the contents are undefined.
  const iree_hal_amdgpu_bitmap_t bitmap = {
      .bit_count = allocator->page_count,
      .words = &block->user_data[0],
  };
  iree_hal_amdgpu_bitmap_reset_all(bitmap);

  // Link the block into the list.
  // If it is full to start (page_count == pages per block) we move it to the
  // end of the list so it's not scanned.
  if (page_count == allocator->page_count) {
    block->next = NULL;
    block->prev = allocator->block_tail;
    if (allocator->block_tail) {
      allocator->block_tail->next = block;
    } else {
      allocator->block_head = block;
    }
    allocator->block_tail = block;
  } else {
    block->next = allocator->block_head;
    block->prev = NULL;
    if (allocator->block_head) {
      allocator->block_head->prev = block;
    } else {
      allocator->block_tail = block;
    }
    allocator->block_head = block;
  }

  // Reserve the the entire page range starting at index 0.
  const iree_host_size_t page_index = 0;
  iree_hal_amdgpu_bitmap_set_span(bitmap, page_index, page_count);
  *out_ptr = (uint8_t*)block->ptr + page_index * allocator->page_size;
  out_token->page_count = (uint64_t)page_count;
  out_token->block = (uint64_t)block;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_block_allocator_allocate(
    iree_hal_amdgpu_block_allocator_t* allocator, iree_host_size_t size,
    IREE_AMDGPU_DEVICE_PTR void** out_ptr,
    iree_hal_amdgpu_block_token_t* out_token) {
  // Round up the allocation size to the next page size.
  const iree_host_size_t page_count =
      iree_host_size_ceil_div(size, allocator->page_size);

  // If the allocation exceeds the page count of a block we cannot allocate it.
  // We could send these off to an oversized dedicated allocation pool but the
  // usage of this today shouldn't hit that. Everything should be on the fast
  // path and dedicated allocations for high-frequency transient allocations are
  // the slowest of paths.
  if (page_count > allocator->page_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "page count %" PRIhsz " for an allocation of %" PRIhsz
        " bytes exceeds block page capacity of %u x %u byte blocks",
        page_count, size, allocator->page_count, allocator->page_size);
  }

  iree_slim_mutex_lock(&allocator->mutex);
  iree_status_t status = iree_hal_amdgpu_block_allocator_allocate_with_lock(
      allocator, page_count, out_ptr, out_token);
  iree_slim_mutex_unlock(&allocator->mutex);
  return status;
}

static void iree_hal_amdgpu_block_allocator_free_with_lock(
    iree_hal_amdgpu_block_allocator_t* allocator,
    IREE_AMDGPU_DEVICE_PTR void* ptr, iree_hal_amdgpu_block_token_t token) {
  iree_hal_amdgpu_block_t* block =
      (iree_hal_amdgpu_block_t*)(((int64_t)token.block << 8) >> 8);

  // Calculate and clear the page bits corresponding to the allocated range.
  const uint64_t byte_offset = (uint64_t)ptr - (uint64_t)block->ptr;
  const iree_host_size_t page_index = byte_offset / allocator->page_size;
  const iree_hal_amdgpu_bitmap_t bitmap = {
      .bit_count = allocator->page_count,
      .words = &block->user_data[0],
  };
  iree_hal_amdgpu_bitmap_reset_span(bitmap, page_index, token.page_count);

  // We do two things: moving the block to the head of list so it's found in
  // scans and returning the block to the block pool if it has no more
  // allocations outstanding. In both cases we unlink it from the block list.
  if (block->next) {
    block->next->prev = block->prev;
  } else {
    allocator->block_tail = block->prev;
  }
  if (block->prev) {
    block->prev->next = block->next;
  } else {
    allocator->block_head = block->next;
  }
  block->prev = block->next = NULL;

  // If the block has no more remaining allocations outstanding it can be
  // returned to the block pool after we unlink it.
  if (iree_hal_amdgpu_bitmap_empty(bitmap)) {
    iree_hal_amdgpu_block_pool_release(allocator->block_pool, block);
    return;
  }

  // Move the block to the head of the list as we now know it has free pages.
  // When allocations are made in predictable patterns (most are) this ensures
  // the next set of allocations will find pages they are looking for early in
  // their scan. Or not - there's pathological cases where it'll just create
  // a ton of fragmentation.
  if (allocator->block_head) {
    allocator->block_head->prev = block;
  } else {
    allocator->block_tail = block;
  }
  block->next = allocator->block_head;
  allocator->block_head = block;
}

void iree_hal_amdgpu_block_allocator_free(
    iree_hal_amdgpu_block_allocator_t* allocator,
    IREE_AMDGPU_DEVICE_PTR void* ptr, iree_hal_amdgpu_block_token_t token) {
  if (!ptr) return;
  iree_slim_mutex_lock(&allocator->mutex);
  iree_hal_amdgpu_block_allocator_free_with_lock(allocator, ptr, token);
  iree_slim_mutex_unlock(&allocator->mutex);
}
