// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_buffer_t
//===----------------------------------------------------------------------===//

// Wraps a bound Vulkan buffer allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_vulkan_buffer_create(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, VkMemoryPropertyFlags memory_property_flags,
    VkDeviceSize non_coherent_atom_size, VkDeviceMemory device_memory,
    VkBuffer handle, VkDeviceAddress device_address,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Wraps a borrowed byte range within an existing bound Vulkan buffer.
//
// The returned HAL buffer does not destroy |handle| or free |device_memory|.
// |allocation_size| is the backing allocation extent used for HAL range
// validation, and |byte_offset| is the start of the returned HAL buffer's
// valid range within that allocation. |release_callback| is invoked when the
// wrapper is destroyed and typically returns a pool reservation to the source
// pool.
iree_status_t iree_hal_vulkan_buffer_create_borrowed(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    VkMemoryPropertyFlags memory_property_flags,
    VkDeviceSize non_coherent_atom_size, VkDeviceMemory device_memory,
    VkBuffer handle, VkDeviceAddress device_address,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is a Vulkan HAL buffer.
bool iree_hal_vulkan_buffer_isa(iree_hal_buffer_t* buffer);

// Resolves |buffer| to a Vulkan-backed buffer view suitable for queue packet
// emission. Transient queue-allocation wrappers return their staged backing
// view even before the wrapper is committed to host-visible accessors.
iree_status_t iree_hal_vulkan_buffer_resolve_backing(
    iree_hal_buffer_t* buffer, iree_hal_buffer_t** out_backing_buffer);

// Returns the byte offset into |backing_buffer| for |buffer| plus
// |local_byte_offset|. When |buffer| is a subspan of a transient wrapper this
// preserves both the staged backing view offset and the original wrapper view
// offset.
iree_status_t iree_hal_vulkan_buffer_resolve_backing_offset(
    iree_hal_buffer_t* buffer, iree_hal_buffer_t* backing_buffer,
    iree_device_size_t local_byte_offset,
    iree_device_size_t* out_backing_byte_offset);

// Returns the Vulkan memory and buffer handles backing |buffer|.
iree_status_t iree_hal_vulkan_buffer_handle(iree_hal_buffer_t* buffer,
                                            VkDeviceMemory* out_memory,
                                            VkBuffer* out_handle);

// Returns the Vulkan buffer device address backing |buffer|.
iree_status_t iree_hal_vulkan_buffer_device_address(
    iree_hal_buffer_t* buffer, VkDeviceAddress* out_device_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_BUFFER_H_
