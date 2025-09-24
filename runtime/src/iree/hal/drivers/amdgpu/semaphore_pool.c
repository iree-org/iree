// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/semaphore_pool.h"

#include "iree/hal/drivers/amdgpu/device/semaphore.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

static void iree_hal_amdgpu_semaphore_pool_link_free_block(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool,
    iree_hal_amdgpu_semaphore_pool_block_t* block);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_semaphore_pool_block_t
//===----------------------------------------------------------------------===//

// A block of allocated semaphores. Manages both host heap memory and
// device-visible memory for the device-side library resources.
//
// Thread-safe; each block has its own lock for free list management.
typedef struct iree_hal_amdgpu_semaphore_pool_block_t {
  // Pool that owns this block.
  iree_hal_amdgpu_semaphore_pool_t* semaphore_pool;
  // Previous block in the pool block linked list.
  struct iree_hal_amdgpu_semaphore_pool_block_t* prev_block;
  // Next block in the pool block linked list.
  struct iree_hal_amdgpu_semaphore_pool_block_t* next_block;
  // Next block in the pool block linked list with free entries.
  struct iree_hal_amdgpu_semaphore_pool_block_t* next_free;
  // Capacity of the block in semaphores.
  iree_host_size_t capacity;
  // Device memory base pointer used for `iree_hal_amdgpu_device_semaphore_t`.
  IREE_AMDGPU_DEVICE_PTR uint8_t* device_allocation_ptr;
  // Mutex guarding the mutable block fields.
  iree_slim_mutex_t mutex;
  // Count of free semaphores in the block stored in the free_list.
  iree_host_size_t free_count IREE_GUARDED_BY(mutex);
  // Free semaphores that are available for use.
  iree_hal_amdgpu_internal_semaphore_t* free_list[/*capacity*/] IREE_GUARDED_BY(
      mutex);
  // Trailing list of iree_hal_amdgpu_internal_semaphore_t[capacity].
} iree_hal_amdgpu_semaphore_pool_block_t;

static void iree_hal_amdgpu_semaphore_pool_block_free(
    iree_hal_amdgpu_semaphore_pool_block_t* block);
static void iree_hal_amdgpu_semaphore_pool_block_recycle(
    void* user_data, iree_hal_amdgpu_internal_semaphore_t* semaphore);

// Allocates a block of |capacity| semaphores on host and device.
static iree_status_t iree_hal_amdgpu_semaphore_pool_block_allocate(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool, iree_host_size_t capacity,
    iree_hal_semaphore_flags_t flags,
    iree_hal_amdgpu_semaphore_pool_block_t** out_block) {
  IREE_ASSERT_ARGUMENT(out_block);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_block = NULL;

  const iree_hal_amdgpu_libhsa_t* libhsa = semaphore_pool->libhsa;

  // Allocate and initialize host memory.
  iree_hal_amdgpu_semaphore_pool_block_t* block = NULL;
  const iree_host_size_t free_list_size =
      capacity * sizeof(block->free_list[0]);
  const iree_host_size_t total_block_size =
      sizeof(*block) + free_list_size +
      capacity * sizeof(iree_hal_amdgpu_internal_semaphore_t);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(semaphore_pool->host_allocator,
                                total_block_size, (void**)&block));
  block->semaphore_pool = semaphore_pool;
  block->prev_block = NULL;
  block->next_block = NULL;
  block->next_free = NULL;
  block->capacity = capacity;
  block->device_allocation_ptr = NULL;
  iree_slim_mutex_initialize(&block->mutex);

  // Allocate device memory from the HSA memory pool.
  const iree_host_size_t total_device_size =
      capacity * sizeof(iree_hal_amdgpu_device_semaphore_t);
  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(libhsa), semaphore_pool->memory_pool, total_device_size,
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&block->device_allocation_ptr);

  // Make the allocation visible to all devices.
  if (iree_status_is_ok(status)) {
    const iree_hal_amdgpu_topology_t* topology = semaphore_pool->topology;
    status = iree_hsa_amd_agents_allow_access(
        IREE_LIBHSA(libhsa), topology->all_agent_count, topology->all_agents,
        /*flags=*/NULL, block->device_allocation_ptr);
  }

  // Initialize each host semaphore and build the free list.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_internal_semaphore_t* base_host_ptr =
        (iree_hal_amdgpu_internal_semaphore_t*)((uint8_t*)block +
                                                sizeof(*block) +
                                                free_list_size);
    iree_hal_amdgpu_device_semaphore_t* base_device_ptr =
        (iree_hal_amdgpu_device_semaphore_t*)block->device_allocation_ptr;
    block->free_count = capacity;
    iree_hal_amdgpu_internal_semaphore_release_callback_t release_callback = {
        .fn = iree_hal_amdgpu_semaphore_pool_block_recycle,
        .user_data = block,
    };
    for (iree_host_size_t i = 0; i < capacity; ++i) {
      iree_hal_amdgpu_internal_semaphore_t* semaphore = &base_host_ptr[i];
      iree_hal_amdgpu_device_semaphore_t* device_semaphore =
          &base_device_ptr[i];
      status = iree_hal_amdgpu_internal_semaphore_initialize(
          semaphore_pool->libhsa, semaphore_pool->options, flags,
          device_semaphore, release_callback, semaphore);
      if (!iree_status_is_ok(status)) break;
      block->free_list[i] = semaphore;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_block = block;
  } else {
    iree_hal_amdgpu_semaphore_pool_block_free(block);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Frees a |block| of semaphores and its device memory.
static void iree_hal_amdgpu_semaphore_pool_block_free(
    iree_hal_amdgpu_semaphore_pool_block_t* block) {
  IREE_ASSERT_ARGUMENT(block);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&block->mutex);
  IREE_ASSERT_EQ(block->free_count, block->capacity);
  iree_slim_mutex_unlock(&block->mutex);

  // Deinitialize all host semaphores. They are allocated as part of the block
  // and only need to be cleaned up.
  const iree_host_size_t free_list_size =
      block->capacity * sizeof(block->free_list[0]);
  iree_hal_amdgpu_internal_semaphore_t* base_host_ptr =
      (iree_hal_amdgpu_internal_semaphore_t*)((uint8_t*)block + sizeof(*block) +
                                              free_list_size);
  for (iree_host_size_t i = 0; i < block->capacity; ++i) {
    iree_hal_amdgpu_internal_semaphore_t* semaphore = &base_host_ptr[i];
    iree_hal_amdgpu_internal_semaphore_deinitialize(semaphore);
  }

  // Deallocate device memory.
  if (block->device_allocation_ptr) {
    IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(
        IREE_LIBHSA(block->semaphore_pool->libhsa),
        block->device_allocation_ptr));
    block->device_allocation_ptr = NULL;
  }

  // Frees the block and its embedded storage.
  iree_slim_mutex_deinitialize(&block->mutex);
  iree_allocator_free(block->semaphore_pool->host_allocator, block);

  IREE_TRACE_ZONE_END(z0);
}

// Recycles a semaphore after it has no remaining uses.
static void iree_hal_amdgpu_semaphore_pool_block_recycle(
    void* user_data, iree_hal_amdgpu_internal_semaphore_t* semaphore) {
  iree_hal_amdgpu_semaphore_pool_block_t* block =
      (iree_hal_amdgpu_semaphore_pool_block_t*)user_data;

  // Semaphore should have zero references before being recycled.
  IREE_ASSERT_REF_COUNT_ZERO(&semaphore->resource.ref_count);

  // Add to the block free list.
  iree_slim_mutex_lock(&block->mutex);

  const bool full_to_free = block->free_count == 0;
  block->free_list[block->free_count++] = semaphore;

  iree_slim_mutex_unlock(&block->mutex);

  // If the block has gone from 0 to >0 free entries then link it back into the
  // pool free list for use. Note that we can only do this on the transition
  // from full to free as otherwise the block is already in the free list.
  //
  // NOTE: this happens outside of the per-block lock as the pool will hold its
  // lock over the free list while acquiring a new entry. This may lead to
  // (safe) races where an acquire checks the free list while we are updating
  // the block above but before we update the free list but that's rare and
  // bounded (there may be one extra block in the pool).
  if (full_to_free) {
    iree_hal_amdgpu_semaphore_pool_link_free_block(block->semaphore_pool,
                                                   block);
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_semaphore_pool_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_semaphore_pool_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_host_size_t block_capacity,
    iree_hal_amdgpu_semaphore_options_t options,
    iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
    hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_semaphore_pool_t* out_semaphore_pool) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_semaphore_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  out_semaphore_pool->libhsa = libhsa;
  out_semaphore_pool->topology = topology;
  out_semaphore_pool->options = options;
  out_semaphore_pool->host_allocator = host_allocator;
  out_semaphore_pool->memory_pool = memory_pool;
  out_semaphore_pool->flags = flags;

  iree_slim_mutex_initialize(&out_semaphore_pool->mutex);
  out_semaphore_pool->list_head = NULL;
  out_semaphore_pool->free_head = NULL;

  // Query the memory pool for its allocation granularity.
  // This is not the minimum allocation size
  // (HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE) but the recommended size
  // to prevent internal fragmentation. We will always make allocations of this
  // size and adjust the block capacity to match.
  size_t alloc_rec_granule = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_amd_memory_pool_get_info(
          IREE_LIBHSA(libhsa), memory_pool,
          HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
          &alloc_rec_granule),
      "querying HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE to "
      "determine block capacity");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, alloc_rec_granule);

  // Allocate aligned to the recommended allocation granularity.
  // We'll always be some multiple of the recommended size so we waste no device
  // space.
  const iree_host_size_t min_capacity_per_allocation = iree_host_size_ceil_div(
      alloc_rec_granule, sizeof(iree_hal_amdgpu_device_semaphore_t));
  const iree_host_size_t capacity_per_allocation =
      iree_host_size_ceil_div(block_capacity, min_capacity_per_allocation) *
      min_capacity_per_allocation;
  out_semaphore_pool->block_capacity = capacity_per_allocation;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_semaphore_pool_deinitialize(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool) {
  IREE_ASSERT_ARGUMENT(semaphore_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&semaphore_pool->mutex);
  iree_hal_amdgpu_semaphore_pool_block_t* block = semaphore_pool->list_head;
  while (block != NULL) {
    iree_hal_amdgpu_semaphore_pool_block_t* next_block = block->next_block;
    IREE_ASSERT_EQ(block->free_count, block->capacity);
    iree_hal_amdgpu_semaphore_pool_block_free(block);
    block = next_block;
  }
  semaphore_pool->list_head = NULL;
  semaphore_pool->free_head = NULL;
  iree_slim_mutex_unlock(&semaphore_pool->mutex);

  iree_slim_mutex_deinitialize(&semaphore_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

// Grows the |semaphore_pool| by one block.
// Requires the pool lock be held.
static iree_status_t iree_hal_amdgpu_semaphore_pool_grow(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool) {
  IREE_ASSERT_ARGUMENT(semaphore_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the new block and its resources.
  iree_hal_amdgpu_semaphore_pool_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_semaphore_pool_block_allocate(
              semaphore_pool, semaphore_pool->block_capacity,
              semaphore_pool->flags, &block));

  // Link the block into the allocated list and the free list.
  block->prev_block = NULL;
  block->next_block = semaphore_pool->list_head;
  if (block->next_block) {
    block->next_block->prev_block = block;
  }
  semaphore_pool->list_head = block;
  block->next_free = semaphore_pool->free_head;
  semaphore_pool->free_head = block;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_semaphore_pool_preallocate(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool, iree_host_size_t count) {
  IREE_ASSERT_ARGUMENT(semaphore_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  const iree_host_size_t block_count =
      iree_host_size_ceil_div(count, semaphore_pool->block_capacity);
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < block_count;
       ++i) {
    status = iree_hal_amdgpu_semaphore_pool_grow(semaphore_pool);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_semaphore_pool_acquire(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(semaphore_pool);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  iree_slim_mutex_lock(&semaphore_pool->mutex);

  // If there are no blocks with free semaphores allocate a new one.
  iree_status_t status = iree_ok_status();
  if (semaphore_pool->free_head == NULL) {
    // TODO(benvanik): do this outside of the lock? This allocates device
    // resources. We could have an exclusive growth lock that does not block
    // recycling.
    status = iree_hal_amdgpu_semaphore_pool_grow(semaphore_pool);
  }

  // Get the next free semaphore and possibly maintain the free list.
  iree_hal_amdgpu_internal_semaphore_t* semaphore = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_semaphore_pool_block_t* block = semaphore_pool->free_head;
    iree_slim_mutex_lock(&block->mutex);

    // Pop the last free semaphore from the block.
    semaphore = block->free_list[block->free_count - 1];
    IREE_ASSERT_NE(semaphore, NULL);
    block->free_list[block->free_count - 1] = NULL;
    --block->free_count;

    // If there are no more free semaphores in the block remove it from the
    // free list.
    if (block->free_count == 0) {
      semaphore_pool->free_head = block->next_free;
      block->next_free = NULL;
    }

    iree_slim_mutex_unlock(&block->mutex);
  }

  iree_slim_mutex_unlock(&semaphore_pool->mutex);

  if (iree_status_is_ok(status)) {
    // Return with a 1 ref count as if we had allocated it.
    iree_atomic_ref_count_inc(&semaphore->resource.ref_count);
    iree_hal_amdgpu_internal_semaphore_reset(semaphore, initial_value);
    *out_semaphore = (iree_hal_semaphore_t*)semaphore;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Links |block| into the |semaphore_pool| free list.
// Must not already be in the list.
// The block is inserted at the head to try to have new acquisitions reuse it
// before any others and keep the utilization high.
static void iree_hal_amdgpu_semaphore_pool_link_free_block(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool,
    iree_hal_amdgpu_semaphore_pool_block_t* block) {
  iree_slim_mutex_lock(&semaphore_pool->mutex);
  block->next_free = semaphore_pool->free_head;
  semaphore_pool->free_head = block;
  iree_slim_mutex_unlock(&semaphore_pool->mutex);
}

void iree_hal_amdgpu_semaphore_pool_trim(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool) {
  IREE_ASSERT_ARGUMENT(semaphore_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Walk each block in the free list. If all semaphores are free then drop it.
  iree_slim_mutex_lock(&semaphore_pool->mutex);
  iree_hal_amdgpu_semaphore_pool_block_t* prev_block = NULL;
  iree_hal_amdgpu_semaphore_pool_block_t* block = semaphore_pool->free_head;
  while (block != NULL) {
    iree_hal_amdgpu_semaphore_pool_block_t* next_block = block->next_free;
    if (block->free_count != block->capacity) {
      // One or more semaphores in use - cannot free the block.
      prev_block = block;
      block = next_block;
      continue;
    }

    // Unlink the block from the free list.
    if (prev_block != NULL) {
      prev_block->next_free = next_block;
    } else {
      semaphore_pool->free_head = next_block;
    }

    // Unlink the block from the main list.
    if (block->prev_block != NULL) {
      block->prev_block->next_block = block->next_block;
    } else {
      semaphore_pool->list_head = block->next_block;
    }
    if (block->next_block != NULL) {
      block->next_block->prev_block = block->prev_block;
    }

    // Free the block and its resources.
    iree_hal_amdgpu_semaphore_pool_block_free(block);

    block = next_block;
  }

  iree_slim_mutex_unlock(&semaphore_pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}
