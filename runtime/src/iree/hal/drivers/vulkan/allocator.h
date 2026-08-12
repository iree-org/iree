// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_VULKAN_ALLOCATOR_H_

#include "iree/async/api.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/physical_device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_allocator_t iree_hal_vulkan_allocator_t;
typedef struct iree_hal_vulkan_queue_t iree_hal_vulkan_queue_t;

typedef enum iree_hal_vulkan_queue_alloca_strategy_e {
  // Invalid zero value used to catch uninitialized alloca plans.
  IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_NONE = 0,

  // Materialize queue_alloca backing from a HAL pool reservation.
  IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_POOL = 1,

  // Materialize queue_alloca backing as a sparse Vulkan buffer and bind it as
  // queue-ordered work.
  IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_SPARSE = 2,
} iree_hal_vulkan_queue_alloca_strategy_t;

typedef struct iree_hal_vulkan_queue_alloca_plan_t {
  // Strategy used by queue.c to stage backing storage.
  iree_hal_vulkan_queue_alloca_strategy_t strategy;

  // Allocator used by sparse strategy to create queue-owned backing buffers.
  iree_hal_allocator_t* allocator;

  // Pool used by pool strategy to reserve and materialize backing buffers.
  iree_hal_pool_t* pool;
} iree_hal_vulkan_queue_alloca_plan_t;

// Creates the Vulkan allocator object for a logical device.
//
// The allocator owns the default Vulkan slab/pool policy used for synchronous
// HAL allocations. Each slab provider delegates whole-slab materialization back
// through the direct allocation helpers below so the Vulkan object creation and
// sparse-binding rules remain centralized.
iree_status_t iree_hal_vulkan_allocator_create(
    iree_hal_device_t* parent_device, const iree_hal_vulkan_device_syms_t* syms,
    VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_device_extensions_t enabled_extensions,
    iree_hal_queue_affinity_t queue_affinity_mask,
    iree_hal_vulkan_queue_t* sparse_binding_queue,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

// Allocates one whole Vulkan buffer from a required memory type index.
//
// This is the primitive used by Vulkan slab providers. It bypasses the default
// pool set and creates a standalone dense or fully-bound sparse buffer. Normal
// users should call iree_hal_allocator_allocate_buffer() instead.
iree_status_t iree_hal_vulkan_allocator_allocate_direct_buffer_from_type(
    iree_hal_vulkan_allocator_t* allocator, uint32_t memory_type_index,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer);

// Allocates one whole Vulkan buffer directly from a compatible memory type.
//
// This bypasses the default pool set and gives the caller exclusive ownership
// of the underlying VkDeviceMemory mapping lifetime.
iree_status_t iree_hal_vulkan_allocator_allocate_direct_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer);

// Returns the default queue-pool backend resources borrowed from |allocator|.
iree_status_t iree_hal_vulkan_allocator_query_queue_pool_backend(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend);

// Selects a backing strategy and compatible buffer parameters for queue_alloca.
//
// |allocation_size| is rounded to the Vulkan buffer size granularity. When
// |requested_pool| is NULL the allocator's default pool policy is used.
// Otherwise the requested pool is validated against the normalized parameters.
// Default allocations that exceed the per-memory-type Vulkan allocation limit
// are returned as sparse plans so queue.c can submit vkQueueBindSparse as the
// alloca epoch.
iree_status_t iree_hal_vulkan_allocator_select_queue_alloca_plan(
    iree_hal_allocator_t* base_allocator, iree_hal_pool_t* requested_pool,
    iree_hal_buffer_params_t* params, iree_device_size_t* allocation_size,
    iree_hal_vulkan_queue_alloca_plan_t* out_plan);

// Selects the Vulkan memory type index that best satisfies |params|.
//
// Only memory types present in |allowed_memory_type_bits| (as populated by
// VkMemoryRequirements::memoryTypeBits) are considered. On success |params| is
// updated to parameters compatible with the selection: unsupported optional
// mapping usage is stripped and the memory type is rewritten to the full flag
// set the winning memory type provides. Returns false when no memory type
// satisfies the parameters.
//
// Score ties are broken with the same priority heuristic used to choose the
// default pool memory types so resolved placements always name a memory type
// that has a backing pool. Exposed for testing; production code allocates
// through the iree_hal_allocator_t interface instead.
bool iree_hal_vulkan_allocator_select_memory_type(
    const VkPhysicalDeviceMemoryProperties* memory_properties,
    uint32_t allowed_memory_type_bits, iree_hal_buffer_params_t* params,
    uint32_t* out_memory_type_index);

// Returns the memory type index the default device pool is created from or
// UINT32_MAX when the device reports no compatible memory type.
//
// Exposed for testing: device-optimal allocations resolved by
// iree_hal_vulkan_allocator_select_memory_type must agree with this selection
// or default-pool queue allocations cannot be satisfied.
uint32_t iree_hal_vulkan_allocator_select_default_device_memory_type(
    const VkPhysicalDeviceMemoryProperties* memory_properties);

// Allocates a sparse buffer and returns queue-owned binds for queue_alloca.
//
// The returned buffer owns its VkBuffer and VkDeviceMemory blocks but the
// memory is not bound until |out_binds| is submitted with vkQueueBindSparse.
// The caller owns both |out_buffer| and |out_binds| on success.
iree_status_t iree_hal_vulkan_allocator_allocate_queue_sparse_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_placement_t placement,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer, iree_host_size_t* out_bind_count,
    VkSparseMemoryBind** out_binds);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_ALLOCATOR_H_
