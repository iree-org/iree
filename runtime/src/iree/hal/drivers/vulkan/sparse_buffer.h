// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_sparse_buffer_t
//===----------------------------------------------------------------------===//

// Creates a fully-bound sparse Vulkan buffer.
//
// Sparse binding lets one VkBuffer span multiple VkDeviceMemory allocations.
// This is the primitive the allocator uses when a requested logical allocation
// is larger than maxMemoryAllocationSize. The returned HAL buffer owns |handle|
// and all allocated physical blocks.
//
// Sparse buffers are not host-mappable in this primitive. Future virtual-memory
// APIs can expose reserve/commit/decommit behavior directly, but this helper is
// deliberately synchronous and fully resident so normal HAL buffers can use it
// without gaining extra valid states.
iree_status_t iree_hal_vulkan_sparse_buffer_create_bound_sync(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkQueue sparse_queue, iree_hal_buffer_placement_t placement,
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, VkBuffer handle,
    VkMemoryRequirements memory_requirements, uint32_t memory_type_index,
    VkDeviceSize max_allocation_size,
    VkMemoryAllocateFlags memory_allocate_flags,
    iree_slim_mutex_t* sparse_queue_mutex, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is a Vulkan sparse HAL buffer.
bool iree_hal_vulkan_sparse_buffer_isa(iree_hal_buffer_t* buffer);

// Returns the Vulkan buffer handle backing |buffer|.
//
// Sparse buffers own many VkDeviceMemory blocks and therefore cannot report a
// single backing memory handle. |out_memory| is always VK_NULL_HANDLE.
iree_status_t iree_hal_vulkan_sparse_buffer_handle(iree_hal_buffer_t* buffer,
                                                   VkDeviceMemory* out_memory,
                                                   VkBuffer* out_handle);

// Returns the Vulkan buffer device address backing |buffer|.
iree_status_t iree_hal_vulkan_sparse_buffer_device_address(
    iree_hal_buffer_t* buffer, VkDeviceAddress* out_device_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_
