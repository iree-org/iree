// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_sparse_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_queue_t iree_hal_vulkan_queue_t;

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
    iree_hal_vulkan_queue_t* sparse_binding_queue,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, VkBuffer handle,
    VkMemoryRequirements memory_requirements, uint32_t memory_type_index,
    VkDeviceSize max_allocation_size,
    VkMemoryAllocateFlags memory_allocate_flags,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Creates a fully-resident sparse Vulkan buffer without submitting binds.
//
// The returned HAL buffer owns |handle| and all VkDeviceMemory blocks. The
// buffer is not usable until |out_binds| are submitted with vkQueueBindSparse.
// Queue implementations use this to make sparse binding the queue_alloca epoch
// instead of synchronously waiting in the allocator.
iree_status_t iree_hal_vulkan_sparse_buffer_create_pending_bind(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, VkBuffer handle,
    VkMemoryRequirements memory_requirements, uint32_t memory_type_index,
    VkDeviceSize max_allocation_size,
    VkMemoryAllocateFlags memory_allocate_flags,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer,
    iree_host_size_t* out_bind_count, VkSparseMemoryBind** out_binds);

// Creates an unbound sparse Vulkan buffer representing a virtual address range.
//
// The returned HAL buffer owns |handle| but no physical VkDeviceMemory. Callers
// map and unmap memory by submitting sparse bind operations against the
// returned handle. The buffer must have been created with sparse
// residency/aliasing flags when callers intend to partially bind or alias
// physical memory.
iree_status_t iree_hal_vulkan_sparse_buffer_create_unbound(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, VkBuffer handle,
    VkMemoryRequirements memory_requirements, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is a Vulkan sparse HAL buffer.
bool iree_hal_vulkan_sparse_buffer_isa(iree_hal_buffer_t* buffer);

// Returns true if |buffer| is an unbound sparse virtual memory reservation.
bool iree_hal_vulkan_sparse_buffer_is_virtual_reservation(
    iree_hal_buffer_t* buffer);

// Returns the Vulkan buffer handle backing |buffer|.
//
// Sparse buffers own many VkDeviceMemory blocks and therefore cannot report a
// single backing memory handle. |out_memory| is always VK_NULL_HANDLE.
iree_status_t iree_hal_vulkan_sparse_buffer_handle(iree_hal_buffer_t* buffer,
                                                   VkDeviceMemory* out_memory,
                                                   VkBuffer* out_handle);

// Returns the Vulkan memory requirements for |buffer|.
iree_status_t iree_hal_vulkan_sparse_buffer_memory_requirements(
    iree_hal_buffer_t* buffer, VkMemoryRequirements* out_memory_requirements);

// Returns the Vulkan buffer device address backing |buffer|.
iree_status_t iree_hal_vulkan_sparse_buffer_device_address(
    iree_hal_buffer_t* buffer, VkDeviceAddress* out_device_address);

// Submits sparse buffer memory binds and waits for them to complete.
//
// This is a host-synchronous allocator/control-path helper layered on the
// nonblocking queue submission primitive. Queue operations that already have
// HAL wait/signal edges should submit sparse binds directly through queue.h.
iree_status_t iree_hal_vulkan_sparse_buffer_bind_sync(
    iree_hal_vulkan_queue_t* sparse_binding_queue,
    iree_hal_buffer_placement_t placement, VkBuffer handle,
    iree_host_size_t bind_count, const VkSparseMemoryBind binds[]);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_
