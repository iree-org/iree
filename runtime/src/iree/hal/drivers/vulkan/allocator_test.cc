// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/allocator.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

#include "iree/testing/gtest.h"

namespace iree::hal::vulkan {
namespace {

VkPhysicalDeviceMemoryProperties MakeMemoryProperties(
    std::initializer_list<VkMemoryPropertyFlags> type_flags) {
  VkPhysicalDeviceMemoryProperties memory_properties;
  std::memset(&memory_properties, 0, sizeof(memory_properties));
  memory_properties.memoryHeapCount = 1;
  memory_properties.memoryHeaps[0].size = 1ull << 30;
  memory_properties.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
  uint32_t type_count = 0;
  for (VkMemoryPropertyFlags flags : type_flags) {
    memory_properties.memoryTypes[type_count].propertyFlags = flags;
    memory_properties.memoryTypes[type_count].heapIndex = 0;
    ++type_count;
  }
  memory_properties.memoryTypeCount = type_count;
  return memory_properties;
}

// Device-optimal dispatch parameters as produced for transient dispatch
// buffers: no mapping usage requested.
iree_hal_buffer_params_t MakeDeviceLocalDispatchParams() {
  iree_hal_buffer_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  return params;
}

// Regression test for https://github.com/iree-org/iree/issues/24788.
//
// Mirrors the PowerVR memory layout from the issue: two device-local
// host-visible memory types (#3 and #4) that differ only in host
// cached/coherent flags. Without mapping usage both receive the same
// allocation-time score while default pool creation prefers #4, so a
// first-wins tie-break resolved placements to #3 and default-pool queue
// allocations failed with "no Vulkan queue allocation pool can satisfy
// allocation".
TEST(VulkanAllocatorMemoryTypeSelectionTest,
     TieBrokenConsistentlyWithDefaultDevicePool) {
  const VkPhysicalDeviceMemoryProperties memory_properties =
      MakeMemoryProperties({
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      });

  iree_hal_buffer_params_t params = MakeDeviceLocalDispatchParams();
  uint32_t memory_type_index = 0;
  ASSERT_TRUE(iree_hal_vulkan_allocator_select_memory_type(
      &memory_properties, /*allowed_memory_type_bits=*/UINT32_MAX, &params,
      &memory_type_index));
  EXPECT_EQ(memory_type_index, 4u);
  EXPECT_EQ(memory_type_index,
            iree_hal_vulkan_allocator_select_default_device_memory_type(
                &memory_properties));
  // The resolved params must carry the full flag set of the winning type so
  // pool capability matching sees consistent flags.
  EXPECT_TRUE(iree_all_bits_set(
      params.type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                       IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                       IREE_HAL_MEMORY_TYPE_HOST_CACHED |
                       IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

// Same tied memory types in the opposite enumeration order: the winner must
// follow the pool priority heuristic, not the enumeration order.
TEST(VulkanAllocatorMemoryTypeSelectionTest,
     TieBreakIndependentOfEnumerationOrder) {
  const VkPhysicalDeviceMemoryProperties memory_properties =
      MakeMemoryProperties({
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      });

  iree_hal_buffer_params_t params = MakeDeviceLocalDispatchParams();
  uint32_t memory_type_index = 0;
  ASSERT_TRUE(iree_hal_vulkan_allocator_select_memory_type(
      &memory_properties, /*allowed_memory_type_bits=*/UINT32_MAX, &params,
      &memory_type_index));
  EXPECT_EQ(memory_type_index, 0u);
  EXPECT_EQ(memory_type_index,
            iree_hal_vulkan_allocator_select_default_device_memory_type(
                &memory_properties));
}

// A strictly better score must still win regardless of pool priority: when
// mapping is requested the host-visible types outscore the device-only types
// and the coherent+cached type outscores the cached-only type.
TEST(VulkanAllocatorMemoryTypeSelectionTest, StrictScoreStillWins) {
  const VkPhysicalDeviceMemoryProperties memory_properties =
      MakeMemoryProperties({
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      });

  iree_hal_buffer_params_t params = MakeDeviceLocalDispatchParams();
  params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  uint32_t memory_type_index = 0;
  ASSERT_TRUE(iree_hal_vulkan_allocator_select_memory_type(
      &memory_properties, /*allowed_memory_type_bits=*/UINT32_MAX, &params,
      &memory_type_index));
  EXPECT_EQ(memory_type_index, 2u);
}

}  // namespace
}  // namespace iree::hal::vulkan
