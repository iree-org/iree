// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_VULKAN_ALLOCATOR_H_

#include "iree/async/api.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/physical_device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_allocator_t iree_hal_vulkan_allocator_t;

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
    iree_hal_queue_affinity_t queue_affinity_mask, VkQueue sparse_binding_queue,
    iree_slim_mutex_t* sparse_binding_queue_mutex,
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

// Returns the default queue-pool backend resources borrowed from |allocator|.
iree_status_t iree_hal_vulkan_allocator_query_queue_pool_backend(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend);

// Selects a pool and compatible buffer parameters for queue_alloca.
//
// |allocation_size| is rounded to the Vulkan buffer size granularity. When
// |requested_pool| is NULL the allocator's default pool policy is used.
// Otherwise the requested pool is validated against the normalized parameters
// and returned borrowed in |out_pool|.
iree_status_t iree_hal_vulkan_allocator_select_queue_pool(
    iree_hal_allocator_t* base_allocator, iree_hal_pool_t* requested_pool,
    iree_hal_buffer_params_t* params, iree_device_size_t* allocation_size,
    iree_hal_pool_t** out_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_ALLOCATOR_H_
