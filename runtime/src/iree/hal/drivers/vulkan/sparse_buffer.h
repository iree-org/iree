// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/drivers/vulkan/vulkan_headers.h"  // IWYU pragma: export
// clang-format on

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// EXPERIMENTAL: allocate a buffer with fully bound memory with undefined
// contents. Allocation and binding will happen synchronously on the calling
// thread.
//
// This will eventually be replaced with HAL device APIs for controlling the
// reserve/commit/decommit/release behavior of the virtual/physical storage.
iree_status_t iree_hal_vulkan_sparse_buffer_create_bound_sync(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree::hal::vulkan::VkDeviceHandle* logical_device, VkQueue queue,
    VkBuffer handle, VkMemoryRequirements requirements,
    uint32_t memory_type_index, VkDeviceSize max_allocation_size,
    iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_
