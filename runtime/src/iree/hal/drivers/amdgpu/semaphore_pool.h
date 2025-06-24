// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_POOL_H_
#define IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_POOL_H_

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_internal_semaphore_t
    iree_hal_amdgpu_internal_semaphore_t;
typedef struct iree_hal_amdgpu_semaphore_pool_block_t
    iree_hal_amdgpu_semaphore_pool_block_t;
typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_semaphore_pool_t
//===----------------------------------------------------------------------===//

// Default semaphore count per block in the pool.
// Larger is better to reduce the number of device memory allocations but we
// don't want to have too high of a fixed overhead. Most programs only have a
// few dozen live semaphores at a time.
#define IREE_HAL_AMDGPU_SEMAPHORE_POOL_DEFAULT_BLOCK_CAPACITY 512

// A pool of allocated HAL semaphores and their corresponding device resources.
// Semaphores are allocated in blocks to reduce the number of device allocations
// we make (as some devices/drivers may have limits). Blocks are allocated
// on-demand and contain a fixed-size set of HAL semaphores allocated inline.
//
// Thread-safe; multiple host threads may share the same pool.
typedef struct iree_hal_amdgpu_semaphore_pool_t {
  // Unowned libhsa handle. Must be retained by the owner.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Topology with all CPU and GPU agents. Semaphores must be visible to all.
  const iree_hal_amdgpu_topology_t* topology;
  // Global semaphore options.
  iree_hal_amdgpu_semaphore_options_t options;

  // Allocator used for host allocations.
  iree_allocator_t host_allocator;
  // HSA memory pool for device allocations.
  hsa_amd_memory_pool_t memory_pool;

  // Common semaphore flags for all allocated in the pool.
  // Semaphores acquired may adjust some flags if they don't change how the
  // semaphore is allocated.
  iree_hal_semaphore_flags_t flags;

  // Capacity of each block in semaphores.
  // Most likely IREE_HAL_AMDGPU_SEMAPHORE_POOL_DEFAULT_BLOCK_CAPACITY.
  iree_host_size_t block_capacity;

  // Guards pool resources during acquisition.
  iree_slim_mutex_t mutex;
  // A doubly-linked list of all allocated blocks.
  iree_hal_amdgpu_semaphore_pool_block_t* list_head IREE_GUARDED_BY(mutex);
  // A singly-linked list of blocks that have one or more free semaphore.
  iree_hal_amdgpu_semaphore_pool_block_t* free_head IREE_GUARDED_BY(mutex);
} iree_hal_amdgpu_semaphore_pool_t;

// Initializes |out_semaphore_pool| for use. Performs no allocation.
// Semaphores will be usable on all CPU and GPU devices in |topology|.
// The device |memory_pool| will be used for device-visible allocations.
iree_status_t iree_hal_amdgpu_semaphore_pool_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_host_size_t block_capacity,
    iree_hal_amdgpu_semaphore_options_t options,
    iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
    hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_semaphore_pool_t* out_semaphore_pool);

// Deinitializes |semaphore_pool| and releases underlying memory.
// All semaphores created from the pool must have been released back to it.
void iree_hal_amdgpu_semaphore_pool_deinitialize(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool);

// Preallocates |count| semaphores and adds them to the pool free list.
iree_status_t iree_hal_amdgpu_semaphore_pool_preallocate(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool, iree_host_size_t count);

// Acquires a semaphore from the pool with the given |initial_value|.
// |flags| must be compatible with the flags used for pool initialization.
iree_status_t iree_hal_amdgpu_semaphore_pool_acquire(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore);

// Trims all blocks that have no allocated semaphores.
void iree_hal_amdgpu_semaphore_pool_trim(
    iree_hal_amdgpu_semaphore_pool_t* semaphore_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_POOL_H_
