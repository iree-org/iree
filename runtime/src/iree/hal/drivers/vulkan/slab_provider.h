// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SLAB_PROVIDER_H_
#define IREE_HAL_DRIVERS_VULKAN_SLAB_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/allocator.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_slab_provider_t
//===----------------------------------------------------------------------===//

// Creation options for a Vulkan slab provider bound to one VkMemoryType.
typedef struct iree_hal_vulkan_slab_provider_options_t {
  // Parent HAL device used for buffer placement metadata.
  iree_hal_device_t* parent_device;

  // Device-level Vulkan dispatch table copied from the logical device.
  const iree_hal_vulkan_device_syms_t* syms;

  // Vulkan logical device that owns slabs from this provider.
  VkDevice logical_device;

  // Vulkan memory type index used for whole-slab allocations.
  uint32_t memory_type_index;

  // Vulkan memory property flags for |memory_type_index|.
  VkMemoryPropertyFlags memory_property_flags;

  // HAL memory type exposed by slabs from this provider.
  iree_hal_memory_type_t memory_type;

  // HAL buffer usage bits supported by slabs from this provider.
  iree_hal_buffer_usage_t supported_usage;

  // Queue affinity mask valid for buffers materialized from this provider.
  iree_hal_queue_affinity_t queue_affinity_mask;

  // Minimum alignment used by the default suballocating pool over this
  // provider.
  iree_device_size_t min_alignment;

  // Physical-device nonCoherentAtomSize used for mapped-memory ranges.
  VkDeviceSize non_coherent_atom_size;
} iree_hal_vulkan_slab_provider_options_t;

// Creates a slab provider that acquires whole Vulkan buffers.
//
// |allocator| is borrowed and must outlive the provider and all pools backed by
// it. This mirrors the device-owned pool lifetime model: default pools are
// owned by the Vulkan allocator, and materialized buffers borrow their source
// pools instead of retaining them.
iree_status_t iree_hal_vulkan_slab_provider_create(
    iree_hal_vulkan_allocator_t* allocator,
    iree_hal_vulkan_slab_provider_options_t options,
    iree_string_view_t trace_name, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SLAB_PROVIDER_H_
