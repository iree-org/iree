// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_BUFFER_POOL_H_
#define IREE_HAL_DRIVERS_AMDGPU_BUFFER_POOL_H_

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_device_allocation_handle_t
    iree_hal_amdgpu_device_allocation_handle_t;

typedef struct iree_hal_amdgpu_buffer_pool_block_t
    iree_hal_amdgpu_buffer_pool_block_t;
typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_buffer_pool_t
//===----------------------------------------------------------------------===//

// Default buffer count per block in the pool.
// Larger is better to reduce the number of device memory allocations but we
// don't want to have too high of a fixed overhead. Most programs only have a
// few dozen live buffers at a time but some with heavy async behavior may have
// many more for outstanding async allocations.
#define IREE_HAL_AMDGPU_BUFFER_POOL_DEFAULT_BLOCK_CAPACITY (2 * 1024)

// A pool of transient buffers and their corresponding device handles.
// Buffers are allocated in blocks to reduce the number of device allocations
// we make (as some devices/drivers may have limits). Blocks are allocated
// on-demand and contain a fixed-size set of HAL buffers allocated inline.
//
// Thread-safe; multiple host threads may share the same pool.
typedef struct iree_hal_amdgpu_buffer_pool_t {
  // Unowned libhsa handle. Must be retained by the owner.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Topology with all CPU and GPU agents. Buffers must be visible to all.
  const iree_hal_amdgpu_topology_t* topology;

  // Placement of all buffers allocated from this pool. This is the logical HAL
  // device and queues that map to the physical device the pool is for.
  iree_hal_buffer_placement_t placement;

  // Allocator used for host allocations.
  iree_allocator_t host_allocator;
  // Device memory pool for device allocations.
  hsa_amd_memory_pool_t memory_pool;

  // Unused/opaque pool ID.
  // TODO(benvanik): make this link back to this pool somehow.
  uint64_t pool;

  // Capacity of each block in buffers.
  // Most likely IREE_HAL_AMDGPU_BUFFER_POOL_DEFAULT_BLOCK_CAPACITY (rounded up
  // to the recommended allocation granularity).
  iree_host_size_t block_capacity;

  // Guards pool resources during acquisition.
  iree_slim_mutex_t mutex;
  // A doubly-linked list of all allocated blocks.
  iree_hal_amdgpu_buffer_pool_block_t* list_head IREE_GUARDED_BY(mutex);
  // A singly-linked list of blocks that have one or more free buffer.
  iree_hal_amdgpu_buffer_pool_block_t* free_head IREE_GUARDED_BY(mutex);
} iree_hal_amdgpu_buffer_pool_t;

// Initializes |out_buffer_pool| for use. Performs no allocation.
// Buffers will be usable on all GPU devices in |topology|. |placement| is used
// as the base placement for all buffers but exact queues will be assigned when
// buffers are acquired. Device-accessible allocation handle storage will be
// allocated from |memory_pool| as needed (not the actual buffers - just
// iree_hal_amdgpu_device_allocation_handle_t).
iree_status_t iree_hal_amdgpu_buffer_pool_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_buffer_placement_t placement, iree_host_size_t block_capacity,
    iree_allocator_t host_allocator, hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_buffer_pool_t* out_buffer_pool);

// Deinitializes |buffer_pool| and releases underlying memory.
// All buffers created from the pool must have been released back to it.
void iree_hal_amdgpu_buffer_pool_deinitialize(
    iree_hal_amdgpu_buffer_pool_t* buffer_pool);

// Preallocates |count| buffer handles and adds them to the pool free list.
iree_status_t iree_hal_amdgpu_buffer_pool_preallocate(
    iree_hal_amdgpu_buffer_pool_t* buffer_pool, iree_host_size_t count);

// Acquires an |allocation_size| buffer from the pool with the given |params|.
// The pool must remain live until the returned |out_buffer| has been fully
// recycled (ref count 0). The returned |out_handle| is the device-side handle
// owned by the buffer and is provided to callers as a convenience; it can be
// accessed from the buffer in the future using
// iree_hal_amdgpu_resolve_transient_buffer.
iree_status_t iree_hal_amdgpu_buffer_pool_acquire(
    iree_hal_amdgpu_buffer_pool_t* buffer_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer,
    iree_hal_amdgpu_device_allocation_handle_t** out_handle);

// Trims all blocks that have no allocated buffers.
void iree_hal_amdgpu_buffer_pool_trim(
    iree_hal_amdgpu_buffer_pool_t* buffer_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_BUFFER_POOL_H_
