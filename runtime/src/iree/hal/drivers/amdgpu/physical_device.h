// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_staging.h"
#include "iree/hal/drivers/amdgpu/physical_device_capabilities.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/drivers/amdgpu/util/block_pool.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/signal_pool.h"
#include "iree/hal/drivers/amdgpu/util/target_id.h"
#include "iree/hal/memory/slab_provider.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/pool.h"
#include "iree/hal/pool_set.h"

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

// Logical byte length for the default per-device queue-allocation pool.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_RANGE_LENGTH_DEFAULT \
  (64 * 1024 * 1024)

// Logical byte length for host-visible default queue-allocation pool slabs.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_HOST_POOL_RANGE_LENGTH_DEFAULT \
  (64 * 1024)

// Minimum byte alignment for default-pool suballocations.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_ALIGNMENT_DEFAULT 256

// Maximum death-frontier entries stored per free default-pool block.
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_POOL_FRONTIER_CAPACITY_DEFAULT \
  IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY

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
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_UPLOAD_CAPACITY 0

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
  // Per-host-queue device-visible control upload ring capacity in bytes. Zero
  // disables the optional upload ring.
  uint32_t host_queue_upload_capacity;

  // Default queue-allocation pool policy.
  struct {
    // Logical byte length of the default TLSF pool range.
    iree_device_size_t range_length;

    // Minimum byte alignment for every default-pool reservation.
    iree_device_size_t alignment;

    // Maximum death-frontier entry count stored per free TLSF block.
    uint8_t frontier_capacity;
  } default_pool;

  // Fixed-size queue_read/queue_write staging policy.
  iree_hal_amdgpu_staging_pool_options_t file_staging;

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
  // HSA driver identifier used when querying per-device clock counters.
  uint32_t driver_uid;
  // PCI domain from HSA_AMD_AGENT_INFO_DOMAIN.
  uint32_t pci_domain;
  // PCI bus decoded from HSA_AMD_AGENT_INFO_BDFID.
  uint32_t pci_bus;
  // PCI device decoded from HSA_AMD_AGENT_INFO_BDFID.
  uint32_t pci_device;
  // PCI function decoded from HSA_AMD_AGENT_INFO_BDFID.
  uint32_t pci_function;
  // True when the PCI identity fields contain HSA-provided values.
  uint32_t has_pci_identity : 1;
  // HSA ISA identity selected for this GPU agent.
  struct {
    // Storage backing |target_id.processor|.
    char target_id_processor[64];
    // Parsed target identity, including XNACK/SRAMECC support and mode.
    iree_hal_amdgpu_target_id_t target_id;
  } isa;
  // Stable physical device UUID bytes reported by HSA when available.
  uint8_t physical_device_uuid[16];
  // True when |physical_device_uuid| contains a stable HSA device identifier.
  uint32_t has_physical_device_uuid : 1;
  // NUMA node of the CPU agent nearest to |device_agent|.
  uint32_t host_numa_node;
  // Host memory pools for the CPU agent nearest to |device_agent|.
  iree_hal_amdgpu_host_memory_pools_t host_memory_pools;
  // Cold memory-system facts used to derive conservative topology flags.
  iree_hal_amdgpu_memory_system_capabilities_t memory_system;
  // CPU-visible coarse-grained device-memory capability for this GPU.
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t
      cpu_visible_device_coarse_memory;
  // Prepublished command-buffer kernarg storage capability for this GPU.
  iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
      prepublished_kernarg_storage;

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

  // Per-device pool of user-visible queue_alloca transient buffer wrappers.
  iree_hal_amdgpu_transient_buffer_pool_t transient_buffer_pool;

  // Per-device pool of materialized slab-backed HAL buffer view wrappers.
  iree_hal_amdgpu_buffer_pool_t materialized_buffer_pool;

  // Pool of HSA signals for host-waited semaphores and proactor integration.
  iree_hal_amdgpu_host_signal_pool_t host_signal_pool;

  // Default queue-allocation pool notification for this physical device.
  iree_async_notification_t* default_pool_notification;
  // Slab provider backing default and caller-created pools for this domain.
  iree_hal_slab_provider_t* default_slab_provider;
  // Host-local slab provider for mappable queue allocation transients.
  iree_hal_slab_provider_t* default_host_slab_provider;
  // TLSF options derived from device options and HSA memory-pool properties.
  iree_hal_tlsf_pool_options_t default_pool_options;
  // Routes default queue allocations to the best compatible memory pool.
  iree_hal_pool_set_t default_pool_set;
  // Frontier-aware suballocating pool used up to the TLSF slab length.
  iree_hal_pool_t* default_pool;
  // Direct per-allocation pool used for requests larger than one TLSF slab.
  iree_hal_pool_t* default_oversized_pool;
  // Frontier-aware suballocating pool for host-visible queue allocations.
  iree_hal_pool_t* default_host_pool;
  // Direct host-visible pool used for requests larger than one host TLSF slab.
  iree_hal_pool_t* default_host_oversized_pool;

  // Fixed-size staging pool for non-mappable queue_read/queue_write transfers.
  iree_hal_amdgpu_staging_pool_t file_staging_pool;

  // Builtin kernel table for this GPU agent.
  iree_hal_amdgpu_device_kernels_t device_kernels;
  // Host/device-neutral transfer context that points into |device_kernels|.
  iree_hal_amdgpu_device_buffer_transfer_context_t buffer_transfer_context;

  // Total number of host queue slots allocated in |host_queues|.
  iree_host_size_t host_queue_capacity;
  // Per-host-queue HSA AQL ring capacity in packets.
  uint32_t host_queue_aql_capacity;
  // Per-host-queue completion/reclaim ring capacity.
  uint32_t host_queue_notification_capacity;
  // Per-host-queue kernarg ring capacity in 64-byte blocks.
  uint32_t host_queue_kernarg_capacity;
  // Per-host-queue device-visible control upload ring capacity in bytes. Zero
  // disables the optional upload ring.
  uint32_t host_queue_upload_capacity;
  // AMD vendor-packet capabilities selected from this GPU agent's ISA.
  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities;
  // Hardware strategy selected for cross-queue epoch waits on this GPU agent.
  iree_hal_amdgpu_wait_barrier_strategy_t wait_barrier_strategy;
  // Queue-local PM4 timestamp strategy selected from this GPU agent's ISA.
  iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy;

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

// Enables or disables HSA dispatch timestamp population on all live queues.
//
// On enable failure, queues successfully enabled by this call are disabled
// before the status is returned. On disable failure, the function attempts all
// queues and joins failures.
iree_status_t iree_hal_amdgpu_physical_device_set_hsa_profiling_enabled(
    iree_hal_amdgpu_physical_device_t* physical_device, bool enabled);

// Deinitializes a physical device and deallocates all device-specific
// resources.
void iree_hal_amdgpu_physical_device_deinitialize(
    iree_hal_amdgpu_physical_device_t* physical_device);

// Releases any unused pooled resources.
iree_status_t iree_hal_amdgpu_physical_device_trim(
    iree_hal_amdgpu_physical_device_t* physical_device);

#endif  // IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
