// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_VIRTUAL_MEMORY_TEST_H_
#define IREE_HAL_CTS_VIRTUAL_MEMORY_TEST_H_

#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

class VirtualMemoryTest : public CTSTestBase<> {};

// Tests capability detection.
TEST_F(VirtualMemoryTest, SupportsVirtualMemory) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
  
  bool supports = iree_hal_allocator_supports_virtual_memory(allocator);
  // We don't require support, but we log it for debugging.
  if (supports) {
    std::cout << "Virtual memory is supported\n";
  } else {
    std::cout << "Virtual memory is NOT supported, tests will be skipped\n";
  }
}

// Tests granularity query.
TEST_F(VirtualMemoryTest, QueryGranularity) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
  
  if (!iree_hal_allocator_supports_virtual_memory(allocator)) {
    GTEST_SKIP() << "Virtual memory not supported on this device";
  }

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  iree_device_size_t min_page_size = 0;
  iree_device_size_t rec_page_size = 0;
  
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, &min_page_size, &rec_page_size));

  EXPECT_GT(min_page_size, 0u);
  EXPECT_GT(rec_page_size, 0u);
  EXPECT_GE(rec_page_size, min_page_size);

  std::cout << "Min page size: " << min_page_size 
            << ", Recommended page size: " << rec_page_size << "\n";
}

// Tests virtual address reservation and release.
TEST_F(VirtualMemoryTest, ReserveAndRelease) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
  
  if (!iree_hal_allocator_supports_virtual_memory(allocator)) {
    GTEST_SKIP() << "Virtual memory not supported on this device";
  }

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  iree_device_size_t min_page_size = 0;
  iree_device_size_t rec_page_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, &min_page_size, &rec_page_size));

  // Reserve virtual address space (use recommended page size)
  iree_device_size_t size = rec_page_size;
  iree_hal_buffer_t* virtual_buffer = NULL;
  
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, IREE_HAL_QUEUE_AFFINITY_ANY, size, &virtual_buffer));
  
  ASSERT_NE(virtual_buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_allocation_size(virtual_buffer), size);

  // Release
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_release(allocator, 
                                                           virtual_buffer));
  iree_hal_buffer_release(virtual_buffer);
}

// Tests physical memory allocation and free.
TEST_F(VirtualMemoryTest, PhysicalMemoryAllocateAndFree) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
  
  if (!iree_hal_allocator_supports_virtual_memory(allocator)) {
    GTEST_SKIP() << "Virtual memory not supported on this device";
  }

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  iree_device_size_t min_page_size = 0;
  iree_device_size_t rec_page_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, &min_page_size, &rec_page_size));

  // Allocate physical memory
  iree_device_size_t size = rec_page_size;
  iree_hal_physical_memory_t* physical_memory = NULL;
  
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, params, size, iree_allocator_system(), &physical_memory));
  
  ASSERT_NE(physical_memory, nullptr);

  // Free
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_free(allocator, 
                                                         physical_memory));
}

// Tests complete workflow: reserve, allocate, map, unmap, free, release.
TEST_F(VirtualMemoryTest, CompleteWorkflow) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
  
  if (!iree_hal_allocator_supports_virtual_memory(allocator)) {
    GTEST_SKIP() << "Virtual memory not supported on this device";
  }

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  iree_device_size_t min_page_size = 0;
  iree_device_size_t rec_page_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, &min_page_size, &rec_page_size));

  iree_device_size_t size = rec_page_size;

  // Step 1: Reserve virtual address space
  iree_hal_buffer_t* virtual_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, IREE_HAL_QUEUE_AFFINITY_ANY, size, &virtual_buffer));
  ASSERT_NE(virtual_buffer, nullptr);

  // Step 2: Allocate physical memory
  iree_hal_physical_memory_t* physical_memory = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, params, size, iree_allocator_system(), &physical_memory));
  ASSERT_NE(physical_memory, nullptr);

  // Step 3: Map physical memory to virtual address
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      allocator, virtual_buffer, /*virtual_offset=*/0, 
      physical_memory, /*physical_offset=*/0, size));

  // Step 4: Set protection to read-write
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      allocator, virtual_buffer, /*virtual_offset=*/0, size,
      IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ | IREE_HAL_MEMORY_PROTECTION_WRITE));

  // Step 5: Advise (prefetch)
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_advise(
      allocator, virtual_buffer, /*virtual_offset=*/0, size,
      IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ADVICE_WILL_NEED));

  // Step 6: Unmap
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      allocator, virtual_buffer, /*virtual_offset=*/0, size));

  // Step 7: Free physical memory
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_free(allocator, 
                                                         physical_memory));

  // Step 8: Release virtual address space
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_release(allocator, 
                                                           virtual_buffer));
  iree_hal_buffer_release(virtual_buffer);
}

// Tests memory aliasing (same physical mapped to multiple virtual ranges).
TEST_F(VirtualMemoryTest, MemoryAliasing) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
  
  if (!iree_hal_allocator_supports_virtual_memory(allocator)) {
    GTEST_SKIP() << "Virtual memory not supported on this device";
  }

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  iree_device_size_t min_page_size = 0;
  iree_device_size_t rec_page_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, &min_page_size, &rec_page_size));

  iree_device_size_t size = rec_page_size;

  // Reserve two virtual address ranges
  iree_hal_buffer_t* virtual_buffer1 = NULL;
  iree_hal_buffer_t* virtual_buffer2 = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, IREE_HAL_QUEUE_AFFINITY_ANY, size, &virtual_buffer1));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, IREE_HAL_QUEUE_AFFINITY_ANY, size, &virtual_buffer2));

  // Allocate one physical memory
  iree_hal_physical_memory_t* physical_memory = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, params, size, iree_allocator_system(), &physical_memory));

  // Map same physical memory to both virtual ranges
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      allocator, virtual_buffer1, /*virtual_offset=*/0, 
      physical_memory, /*physical_offset=*/0, size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      allocator, virtual_buffer2, /*virtual_offset=*/0, 
      physical_memory, /*physical_offset=*/0, size));

  // Set protection for both
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      allocator, virtual_buffer1, /*virtual_offset=*/0, size,
      IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ | IREE_HAL_MEMORY_PROTECTION_WRITE));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      allocator, virtual_buffer2, /*virtual_offset=*/0, size,
      IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ | IREE_HAL_MEMORY_PROTECTION_WRITE));

  // Cleanup
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      allocator, virtual_buffer1, /*virtual_offset=*/0, size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      allocator, virtual_buffer2, /*virtual_offset=*/0, size));
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_free(allocator, 
                                                         physical_memory));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_release(allocator, 
                                                           virtual_buffer1));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_release(allocator, 
                                                           virtual_buffer2));
  iree_hal_buffer_release(virtual_buffer1);
  iree_hal_buffer_release(virtual_buffer2);
}

// Tests protection changes.
TEST_F(VirtualMemoryTest, ProtectionChanges) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
  
  if (!iree_hal_allocator_supports_virtual_memory(allocator)) {
    GTEST_SKIP() << "Virtual memory not supported on this device";
  }

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
  params.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  iree_device_size_t min_page_size = 0;
  iree_device_size_t rec_page_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, &min_page_size, &rec_page_size));

  iree_device_size_t size = rec_page_size;

  // Setup
  iree_hal_buffer_t* virtual_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, IREE_HAL_QUEUE_AFFINITY_ANY, size, &virtual_buffer));
  
  iree_hal_physical_memory_t* physical_memory = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, params, size, iree_allocator_system(), &physical_memory));
  
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      allocator, virtual_buffer, /*virtual_offset=*/0, 
      physical_memory, /*physical_offset=*/0, size));

  // Test different protection modes
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      allocator, virtual_buffer, /*virtual_offset=*/0, size,
      IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_PROTECTION_READ));

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      allocator, virtual_buffer, /*virtual_offset=*/0, size,
      IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ | IREE_HAL_MEMORY_PROTECTION_WRITE));

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      allocator, virtual_buffer, /*virtual_offset=*/0, size,
      IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_PROTECTION_NONE));

  // Cleanup
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      allocator, virtual_buffer, /*virtual_offset=*/0, size));
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_free(allocator, 
                                                         physical_memory));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_release(allocator, 
                                                           virtual_buffer));
  iree_hal_buffer_release(virtual_buffer);
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_VIRTUAL_MEMORY_TEST_H_

