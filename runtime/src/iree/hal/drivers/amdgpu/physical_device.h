// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/block_pool.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/signal_pool.h"
#include "iree/hal/memory/slab_provider.h"
#include "iree/hal/pool.h"

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
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_QUEUE_COUNT \
  IREE_HAL_AMDGPU_DEFAULT_GPU_AGENT_QUEUE_COUNT

// Default per-queue hardware AQL ring capacity in packets.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_AQL_CAPACITY \
  IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_CAPACITY

// Default per-queue completion/reclaim ring capacity in epochs and hot entries.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_NOTIFICATION_CAPACITY \
  IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY

// Default per-queue kernarg ring capacity in 64-byte blocks.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_KERNARG_CAPACITY \
  ((uint32_t)(IREE_HAL_AMDGPU_DEFAULT_KERNARG_RINGBUFFER_CAPACITY /         \
              sizeof(iree_hal_amdgpu_kernarg_block_t)))

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

  // Number of host queues created for this physical device.
  iree_host_size_t host_queue_count;
  // Per-host-queue HSA AQL ring capacity in packets.
  uint32_t host_queue_aql_capacity;
  // Per-host-queue completion/reclaim ring capacity.
  uint32_t host_queue_notification_capacity;
  // Per-host-queue kernarg ring capacity in 64-byte blocks.
  uint32_t host_queue_kernarg_capacity;

  // Forces cross-queue wait barriers to use software deferral instead of the
  // optimal device-side strategy for the GPU ISA.
  uint32_t force_wait_barrier_defer : 1;
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

  // Pool of HSA signals for host-waited semaphores and proactor integration.
  iree_hal_amdgpu_host_signal_pool_t host_signal_pool;

  // Default queue-allocation pool for this physical device.
  // This initially uses a pass-through pool over the device fine-grained HSA
  // pool so queue_alloca can share the generic pool protocol before we add a
  // frontier-aware suballocating pool. User-supplied pools remain caller-owned;
  // this object only provides the physical-device default.
  iree_async_notification_t* default_pool_notification;
  iree_hal_slab_provider_t* default_slab_provider;
  iree_hal_pool_t* default_pool;

  // Builtin kernel table for this GPU agent and a host/device-neutral transfer
  // context that points to it. Shared by all queues on the physical device.
  iree_hal_amdgpu_device_kernels_t device_kernels;
  iree_hal_amdgpu_device_buffer_transfer_context_t buffer_transfer_context;

  // Total number of host queue slots allocated in |host_queues|.
  iree_host_size_t host_queue_capacity;
  // Per-host-queue HSA AQL ring capacity in packets.
  uint32_t host_queue_aql_capacity;
  // Per-host-queue completion/reclaim ring capacity.
  uint32_t host_queue_notification_capacity;
  // Per-host-queue kernarg ring capacity in 64-byte blocks.
  uint32_t host_queue_kernarg_capacity;
  // Hardware strategy selected for cross-queue epoch waits on this GPU agent.
  iree_hal_amdgpu_wait_barrier_strategy_t wait_barrier_strategy;

  // Number of live host queues initialized in |host_queues|.
  iree_host_size_t host_queue_count;
  // One or more host queues mapped to HSA queues on this physical device.
  iree_hal_amdgpu_host_queue_t host_queues[/*host_queue_count*/];
} iree_hal_amdgpu_physical_device_t;

// Returns the aligned heap size in bytes required to store the physical device
// data structure. Requires that the options have been verified.
iree_host_size_t iree_hal_amdgpu_physical_device_calculate_size(
    const iree_hal_amdgpu_physical_device_options_t* options);

// Initializes a physical device.
// Requires that the |options| have been verified.
//
// |out_physical_device| must reference at least
// iree_hal_amdgpu_physical_device_calculate_size of valid host memory.
iree_status_t iree_hal_amdgpu_physical_device_initialize(
    iree_hal_device_t* logical_device, iree_hal_amdgpu_system_t* system,
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_async_proactor_t* proactor, iree_host_size_t host_ordinal,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_host_size_t device_ordinal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device);

// Binds and initializes this physical device's host queues after the logical
// device has been assigned a topology/frontier.
iree_status_t iree_hal_amdgpu_physical_device_assign_frontier(
    iree_hal_device_t* logical_device, iree_hal_amdgpu_system_t* system,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_async_axis_t base_axis,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_signal_table,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* physical_device);

// Deinitializes any host queues initialized by assign_frontier.
void iree_hal_amdgpu_physical_device_deassign_frontier(
    iree_hal_amdgpu_physical_device_t* physical_device);

// Deinitializes a physical device and deallocates all device-specific
// resources.
void iree_hal_amdgpu_physical_device_deinitialize(
    iree_hal_amdgpu_physical_device_t* physical_device);

// Releases any unused pooled resources.
iree_status_t iree_hal_amdgpu_physical_device_trim(
    iree_hal_amdgpu_physical_device_t* physical_device);

#endif  // IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
