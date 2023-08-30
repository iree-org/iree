// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_BASE_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_BASE_BUFFER_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/drivers/vulkan/vulkan_headers.h"
// clang-format on

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Memory types
//===----------------------------------------------------------------------===//

// Set of memory type indices into VkPhysicalDeviceMemoryProperties.
// The set is bucketed by usage semantics to allow us to quickly map an
// allocation request to an underlying memory type. Some buckets may point to
// the same index - for example, on a CPU or integrated GPU all of them may
// point at the same memory space.
//
// Major categories:
// - Dispatch
//   High-bandwidth device-local memory where we want to ensure that all
//   accesses don't require going over a slow bus (PCI/etc).
// - Bulk transfer
//   Low-bandwidth often host-local or host-visible memory for
//   uploading/downloading large buffers, usually backed by system memory.
//   Not expected to be usable by dispatches and just used for staging.
// - Staging transfer
//   High-bandwidth device-local memory that is also host-visible for
//   low-latency staging. These are generally small buffers used by dispatches
//   (like uniform buffers) as the amount of memory available can be very
//   limited (~256MB). Because of the device limits we only use these for
//   TRANSIENT allocations that are used by dispatches.
typedef union {
  struct {
    // Preferred memory type for device-local dispatch operations.
    // This memory _may_ be host visible, though we try to select the most
    // exclusive device local memory when available.
    int dispatch_idx;

    // Preferred memory type for bulk uploads (host->device).
    // These may be slow to access from dispatches or not possible at all, but
    // generally have the entire system memory available for storage.
    int bulk_upload_idx;
    // Preferred memory type for bulk downloads (device->host).
    int bulk_download_idx;

    // Preferred memory type for staging uploads (host->device).
    int staging_upload_idx;
    // Preferred memory type for staging downloads (device->host).
    int staging_download_idx;
  };
  int indices[5];
} iree_hal_vulkan_memory_types_t;

// Finds the memory type that satisfies the required and preferred buffer
// |params| and returns it in |out_memory_type_index|. Only memory types present
// in |allowed_type_indices| will be returned. Fails if no memory type satisfies
// the requirements.
iree_status_t iree_hal_vulkan_find_memory_type(
    const VkPhysicalDeviceProperties* device_props,
    const VkPhysicalDeviceMemoryProperties* memory_props,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    uint32_t allowed_type_indices, uint32_t* out_memory_type_index);

// Queries the underlying Vulkan implementation to decide which memory type
// should be used for particular operations.
iree_status_t iree_hal_vulkan_populate_memory_types(
    const VkPhysicalDeviceProperties* device_props,
    const VkPhysicalDeviceMemoryProperties* memory_props,
    iree_hal_vulkan_memory_types_t* out_memory_types);

// Common implementation of iree_hal_allocator_query_memory_heaps.
iree_status_t iree_hal_vulkan_query_memory_heaps(
    const VkPhysicalDeviceProperties* device_props,
    const VkPhysicalDeviceMemoryProperties* memory_props,
    const iree_hal_vulkan_memory_types_t* memory_types,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count);

//===----------------------------------------------------------------------===//
// Base buffer implementation
//===----------------------------------------------------------------------===//

// Base type all Vulkan HAL buffers must implement to allow the implementation
// to get access to the API VkBuffer handle.
typedef struct iree_hal_vulkan_base_buffer_t {
  iree_hal_buffer_t base;
  // NOTE: may be VK_NULL_HANDLE if sparse residency is used to back the buffer
  // with multiple device memory allocations.
  VkDeviceMemory device_memory;
  VkBuffer handle;
} iree_hal_vulkan_base_buffer_t;

// Returns the Vulkan handle backing the given |buffer|.
// This is the entire allocated_buffer and must be offset by the buffer
// byte_offset and byte_length when used.
static inline VkBuffer iree_hal_vulkan_buffer_handle(
    iree_hal_buffer_t* buffer) {
  return buffer ? ((iree_hal_vulkan_base_buffer_t*)
                       iree_hal_buffer_allocated_buffer(buffer))
                      ->handle
                : VK_NULL_HANDLE;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_BASE_BUFFER_H_
