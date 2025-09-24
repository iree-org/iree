// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amdgpu/host_service.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/block_pool.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/virtual_queue.h"

typedef struct iree_hal_amdgpu_buffer_pool_t iree_hal_amdgpu_buffer_pool_t;
typedef struct iree_hal_amdgpu_host_memory_pools_t
    iree_hal_amdgpu_host_memory_pools_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_options_t
//===----------------------------------------------------------------------===//

// Power-of-two size for the per-device small block pool in bytes.
// Used for command buffer headers and other small data structures.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_SIZE_DEFAULT \
  (32 * 1024)

// Minimum number of small blocks per device allocation.
// Reduces allocation overhead at the cost of under-utilizing memory.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT \
  (128)

// Initial capacity in blocks of the per-device small block pool. Block pools
// will grow as needed but accounting is cleaner if we pre-initialize them to a
// (hopefully) sufficient size.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT \
  IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT

// Power-of-two size for the per-device large block pool in bytes.
// Used for command buffer commands and data. Must be large enough to fit inline
// command buffer uploads.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_SIZE_DEFAULT \
  (256 * 1024)

// Minimum number of large blocks per device allocation.
// Reduces allocation overhead at the cost of under-utilizing memory.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT \
  (16)

// Initial capacity in blocks of the per-device large block pool. Block pools
// will grow as needed but accounting is cleaner if we pre-initialize them to a
// (hopefully) sufficient size.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT \
  IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT

// Power-of-two size for the per-device host block pool in bytes.
// Since primarily used for transient submission-specific allocations it need
// not be large.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_HOST_BLOCK_SIZE_DEFAULT (8 * 1024)

// Total number of HAL queues on the physical device.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_QUEUE_COUNT (1)

// Options controlling how a physical device is initialized.
typedef struct iree_hal_amdgpu_physical_device_options_t {
  // Size of a block in each device block pool.
  // Used for both coarse-grained and fine-grained memory types.
  struct {
    // Small device block pool.
    // Used for command buffer headers and other small data structures.
    iree_hal_amdgpu_block_pool_options_t small;
    // Large device block pool.
    // Used for command buffer commands and data. Must be large enough to fit
    // inline command buffer uploads.
    iree_hal_amdgpu_block_pool_options_t large;
  } device_block_pools;

  // Size of the per-device small host block pool.
  // This is primarily used for per-submission resource sets and other transient
  // bookkeeping that should never be _too_ large or live _too_ long.
  iree_host_size_t host_block_pool_size;
  // Initial block count preallocated for the host block pool.
  iree_host_size_t host_block_pool_initial_capacity;

  // Total number of HAL queues on the physical device.
  iree_host_size_t queue_count;
  // Options used to initialize each queue.
  // Currently we assume queues are homogeneous but we may want to expose
  // bucketed types (e.g. host-side or device-side queues) to allow for tuning
  // each independently.
  iree_hal_amdgpu_queue_options_t queue_options;
} iree_hal_amdgpu_physical_device_options_t;

// Initializes |out_options| to its default values.
void iree_hal_amdgpu_physical_device_options_initialize(
    iree_hal_amdgpu_physical_device_options_t* out_options);

// Verifies device options to ensure they meet the agent requirements.
iree_status_t iree_hal_amdgpu_physical_device_options_verify(
    const iree_hal_amdgpu_physical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    hsa_agent_t gpu_agent);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_t
//===----------------------------------------------------------------------===//

// A physical device representing an HSA GPU agent.
// May contain one or more HAL queues that map to HSA queues on the agent.
typedef struct iree_hal_amdgpu_physical_device_t {
  // GPU agent.
  hsa_agent_t device_agent;
  // Ordinal of the GPU agent within the topology.
  iree_host_size_t device_ordinal;

  // Fine-grained block pools for device memory blocks of various sizes.
  iree_hal_amdgpu_block_pools_t fine_block_pools;
  // Fine-grained block pool-based allocators for small transient allocations.
  iree_hal_amdgpu_block_allocators_t fine_block_allocators;
  // Coarse-grained block pools for device memory blocks of various sizes.
  iree_hal_amdgpu_block_pools_t coarse_block_pools;
  // Coarse-grained block pool-based allocators for small transient allocations.
  iree_hal_amdgpu_block_allocators_t coarse_block_allocators;

  // Host-side small allocation block pool.
  // Shared amongst all queues in the physical device. We don't share with other
  // devices as they may be attached to different NUMA nodes. Though still
  // possible for queue entries to be allocated on one node and freed on another
  // the common case will be that the blocks are touched by the same device.
  iree_arena_block_pool_t fine_host_block_pool;

  // Host-side service worker for supporting device library requests.
  // Today we have one per physical device but could share them or even have
  // one per queue.
  iree_hal_amdgpu_host_service_t host_service;

  // HAL queues with associated device-side schedulers.
  iree_host_size_t queue_count;
  iree_hal_amdgpu_virtual_queue_t* queues[/*queue_count*/];

  // + queue storage; queues may be of mixed types and have different sizes
} iree_hal_amdgpu_physical_device_t;

// Returns the aligned heap size in bytes required to store the physical device
// data structure. Requires that the options have been verified.
iree_host_size_t iree_hal_amdgpu_physical_device_calculate_size(
    const iree_hal_amdgpu_physical_device_options_t* options);

// Initializes a physical device with one or more HAL queues.
// Requires that the |options| have been verified.
//
// |initialization_signal| will be incremented as asynchronous initialization
// operations are enqueued and decremented as they complete. Callers must wait
// for the completion signal to reach 0 prior to deinitializing the device even
// if initialization fails.
//
// NOTE: if initialization fails callers must call
// iree_hal_amdgpu_physical_device_deinitialize after |initialization_signal| is
// reached.
//
// |out_physical_device| must reference at least
// iree_hal_amdgpu_physical_device_calculate_size of valid host memory.
iree_status_t iree_hal_amdgpu_physical_device_initialize(
    iree_hal_amdgpu_system_t* system,
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_host_size_t host_ordinal,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_host_size_t device_ordinal, iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    iree_hal_amdgpu_error_callback_t error_callback,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device);

// Deinitializes a physical device and deallocates all device-specific
// resources.
void iree_hal_amdgpu_physical_device_deinitialize(
    iree_hal_amdgpu_physical_device_t* physical_device);

// Releases any unused pooled resources.
void iree_hal_amdgpu_physical_device_trim(
    iree_hal_amdgpu_physical_device_t* physical_device);

#endif  // IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
