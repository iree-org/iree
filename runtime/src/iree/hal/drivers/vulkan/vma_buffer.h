// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_VMA_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_VMA_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/internal_vk_mem_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Wraps a VMA allocation in an iree_hal_buffer_t.
// The allocation will be released back to VMA when the buffer is released.
iree_status_t iree_hal_vulkan_vma_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    VmaAllocator vma, VkBuffer handle, VmaAllocation allocation,
    VmaAllocationInfo allocation_info, iree_hal_buffer_t** out_buffer);

// Returns the Vulkan handle backing the given |buffer|.
// This is the entire allocated_buffer and must be offset by the buffer
// byte_offset and byte_length when used.
VkBuffer iree_hal_vulkan_vma_buffer_handle(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_VMA_BUFFER_H_
