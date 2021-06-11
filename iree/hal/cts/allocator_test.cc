// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

namespace {

constexpr iree_device_size_t kAllocationSize = 1024;

}  // namespace

class AllocatorTest : public CtsTestBase {};

// All allocators must support some baseline capabilities.
//
// Certain capabilities or configurations are optional and may vary between
// driver implementations or target devices, such as:
//   IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL
//   IREE_HAL_BUFFER_USAGE_MAPPING
TEST_P(AllocatorTest, BaselineBufferCompatibility) {
  // Need at least one way to get data between the host and device.
  iree_hal_buffer_compatibility_t transfer_compatibility_host =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_,
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          /*allowed_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER,
          /*intended_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER, kAllocationSize);
  iree_hal_buffer_compatibility_t transfer_compatibility_device =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_,
          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          /*allowed_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER,
          /*intended_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER, kAllocationSize);
  iree_hal_buffer_compatibility_t required_transfer_compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  EXPECT_TRUE(iree_all_bits_set(transfer_compatibility_host,
                                required_transfer_compatibility) ||
              iree_all_bits_set(transfer_compatibility_device,
                                required_transfer_compatibility));

  // Need to be able to use some type of buffer as dispatch inputs or outputs.
  iree_hal_buffer_compatibility_t dispatch_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          /*allowed_usage=*/IREE_HAL_BUFFER_USAGE_DISPATCH,
          /*intended_usage=*/IREE_HAL_BUFFER_USAGE_DISPATCH, kAllocationSize);
  EXPECT_TRUE(
      iree_all_bits_set(dispatch_compatibility,
                        IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                            IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH));
}

TEST_P(AllocatorTest, BufferAllowedUsageDeterminesCompatibility) {
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          /*allowed_usage=*/IREE_HAL_BUFFER_USAGE_NONE,
          /*intended_usage=*/IREE_HAL_BUFFER_USAGE_ALL, kAllocationSize);
  EXPECT_TRUE(iree_all_bits_set(compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE));
  EXPECT_FALSE(iree_all_bits_set(compatibility,
                                 IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER));
  EXPECT_FALSE(iree_all_bits_set(compatibility,
                                 IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH));
}

TEST_P(AllocatorTest, AllocateBuffer) {
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  iree_hal_buffer_usage_t buffer_usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_t* buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, memory_type, buffer_usage, kAllocationSize, &buffer));

  EXPECT_EQ(device_allocator_, iree_hal_buffer_allocator(buffer));
  // At a mimimum, the requested memory type should be respected.
  // Additional bits may be optionally set depending on the allocator.
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_memory_type(buffer), memory_type));
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer), buffer_usage));
  EXPECT_GE(iree_hal_buffer_allocation_size(buffer),
            kAllocationSize);  // Larger is okay.

  iree_hal_buffer_release(buffer);
}

// While empty allocations aren't particularly useful, they can occur in
// practice so we should at least be able to create them without errors.
TEST_P(AllocatorTest, AllocateEmptyBuffer) {
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  iree_hal_buffer_usage_t buffer_usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_t* buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, memory_type, buffer_usage, /*allocation_size=*/0,
      &buffer));

  iree_hal_buffer_release(buffer);
}

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, AllocatorTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
