// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class AllocatorTest : public CtsTestBase {};

// Tests for baseline buffer compatibility that all HAL drivers must support.
TEST_P(AllocatorTest, QueryBufferCompatibility) {
  iree_host_size_t allocation_size = 1024;

  // Need at least one way to get data between the host and device.
  iree_hal_buffer_compatibility_t transfer_compatibility_host =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_,
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          /*allowed_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER,
          /*intended_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER, allocation_size);
  iree_hal_buffer_compatibility_t transfer_compatibility_device =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_,
          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          /*allowed_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER,
          /*intended_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER, allocation_size);
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
          /*intended_usage=*/IREE_HAL_BUFFER_USAGE_DISPATCH, allocation_size);
  EXPECT_TRUE(
      iree_all_bits_set(dispatch_compatibility,
                        IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                            IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH));
}

TEST_P(AllocatorTest, AllocateBuffer) {
  iree_hal_memory_type_t memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  iree_hal_buffer_usage_t buffer_usage = IREE_HAL_BUFFER_USAGE_ALL;
  iree_host_size_t allocation_size = 1024;

  iree_hal_buffer_t* buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, memory_type, buffer_usage, allocation_size, &buffer));

  EXPECT_EQ(device_allocator_, iree_hal_buffer_allocator(buffer));
  // At a mimimum, the requested memory type should be respected.
  // Additional bits may be optionally set depending on the allocator.
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_memory_type(buffer), memory_type));
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer), buffer_usage));
  EXPECT_GE(iree_hal_buffer_allocation_size(buffer),
            allocation_size);  // Larger is okay.

  iree_hal_buffer_release(buffer);
}

// TODO(scotttodd): iree_hal_allocator_wrap_buffer
//     * if implemented (skip test if status is "IREE_STATUS_UNAVAILABLE")

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, AllocatorTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
