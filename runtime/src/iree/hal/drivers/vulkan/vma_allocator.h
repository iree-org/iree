// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_VMA_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_VULKAN_VMA_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a VMA-based allocator that performs internal suballocation and a
// bunch of other fancy things.
//
// This uses the Vulkan Memory Allocator (VMA) to manage memory.
// VMA (//third_party/vulkan_memory_allocator) provides dlmalloc-like behavior
// with suballocations made with various policies (best fit, first fit, etc).
// This reduces the number of allocations we need from the Vulkan implementation
// (which can sometimes be limited to as little as 4096 total allowed) and
// manages higher level allocation semantics like slab allocation and
// defragmentation.
//
// VMA is internally synchronized and the functionality exposed on the HAL
// interface is thread-safe.
//
// More information:
//   https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
//   https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/
iree_status_t iree_hal_vulkan_vma_allocator_create(
    const iree_hal_vulkan_device_options_t* options, VkInstance instance,
    VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_hal_device_t* device, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_VMA_ALLOCATOR_H_
