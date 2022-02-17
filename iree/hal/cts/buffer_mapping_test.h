// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_BUFFER_MAPPING_TEST_H_
#define IREE_HAL_CTS_BUFFER_MAPPING_TEST_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

using ::testing::ContainerEq;

namespace {

constexpr iree_device_size_t kDefaultAllocationSize = 64;

}  // namespace

class buffer_mapping_test : public CtsTestBase {
 protected:
  void AllocateUninitializedBuffer(iree_device_size_t buffer_size,
                                   iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_),
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
        IREE_HAL_BUFFER_USAGE_MAPPING, buffer_size,
        iree_const_byte_span_empty(), &device_buffer));
    *out_buffer = device_buffer;
  }
};

TEST_P(buffer_mapping_test, AllocatorSupportsBufferMapping) {
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  iree_hal_buffer_usage_t buffer_usage = IREE_HAL_BUFFER_USAGE_MAPPING;

  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          device_allocator_, memory_type,
          /*allowed_usage=*/buffer_usage,
          /*intended_usage=*/buffer_usage, kDefaultAllocationSize);
  EXPECT_TRUE(iree_all_bits_set(compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE));

  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer);

  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_memory_type(buffer), memory_type));
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer), buffer_usage));
  EXPECT_GE(iree_hal_buffer_allocation_size(buffer), kDefaultAllocationSize);

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, Zero) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);
  std::vector<uint8_t> actual_data(buffer_size);

  // Zero the entire buffer.
  IREE_ASSERT_OK(
      iree_hal_buffer_zero(buffer, /*byte_offset=*/0, IREE_WHOLE_BUFFER));
  // Check that the contents match what we expect.
  IREE_ASSERT_OK(iree_hal_buffer_read_data(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_zero_buffer{0x00, 0x00, 0x00, 0x00,  //
                                             0x00, 0x00, 0x00, 0x00,  //
                                             0x00, 0x00, 0x00, 0x00,  //
                                             0x00, 0x00, 0x00, 0x00};
  EXPECT_THAT(actual_data, ContainerEq(reference_zero_buffer));

  // Fill the entire buffer then zero only a segment of it.
  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(iree_hal_buffer_fill(buffer, /*byte_offset=*/0,
                                      IREE_WHOLE_BUFFER, &fill_value,
                                      sizeof(fill_value)));
  IREE_ASSERT_OK(
      iree_hal_buffer_zero(buffer, /*byte_offset=*/4, /*byte_length=*/8));
  // Check that the contents match what we expect.
  IREE_ASSERT_OK(iree_hal_buffer_read_data(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_offset_buffer{0xFF, 0xFF, 0xFF, 0xFF,  //
                                               0x00, 0x00, 0x00, 0x00,  //
                                               0x00, 0x00, 0x00, 0x00,  //
                                               0xFF, 0xFF, 0xFF, 0xFF};
  EXPECT_THAT(actual_data, ContainerEq(reference_offset_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, FillEmpty) {
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer);

  // Zero the whole buffer then "fill" 0 bytes with a different pattern.
  IREE_ASSERT_OK(iree_hal_buffer_zero(buffer, 0, IREE_WHOLE_BUFFER));
  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(iree_hal_buffer_fill(buffer, /*byte_offset=*/0,
                                      /*byte_length=*/0,  // <---- empty!
                                      /*pattern=*/&fill_value,
                                      /*pattern_length=*/sizeof(fill_value)));

  // Check that the buffer is still all zeroes.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_buffer_read_data(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), 0, kDefaultAllocationSize);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, FillWhole) {
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer);

  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(iree_hal_buffer_fill(buffer, /*byte_offset=*/0,
                                      /*byte_length=*/IREE_WHOLE_BUFFER,
                                      /*pattern=*/&fill_value,
                                      /*pattern_length=*/sizeof(fill_value)));

  // Check that the buffer is filled with the pattern.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_buffer_read_data(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), fill_value, kDefaultAllocationSize);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, FillOffset) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Zero the entire buffer then fill only a segment of it.
  IREE_ASSERT_OK(
      iree_hal_buffer_zero(buffer, /*byte_offset=*/0, IREE_WHOLE_BUFFER));
  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(iree_hal_buffer_fill(buffer, /*byte_offset=*/4,
                                      /*byte_length=*/8,
                                      /*pattern=*/&fill_value,
                                      /*pattern_length=*/sizeof(fill_value)));

  // Check that only the segment of the buffer is filled with the pattern.
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_read_data(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_offset_buffer{0x00, 0x00, 0x00, 0x00,  //
                                               0xFF, 0xFF, 0xFF, 0xFF,  //
                                               0xFF, 0xFF, 0xFF, 0xFF,  //
                                               0x00, 0x00, 0x00, 0x00};
  EXPECT_THAT(actual_data, ContainerEq(reference_offset_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, WriteData) {
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer);

  uint8_t fill_value = 0x07;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), fill_value, kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_buffer_write_data(buffer, /*target_offset=*/0,
                                            reference_buffer.data(),
                                            reference_buffer.size()));

  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_buffer_read_data(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
}
TEST_P(buffer_mapping_test, CopyData) {
  iree_hal_buffer_t* buffer_a;
  iree_hal_buffer_t* buffer_b;
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer_a);
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer_b);

  uint8_t fill_value = 0x07;
  IREE_ASSERT_OK(iree_hal_buffer_fill(buffer_a, /*byte_offset=*/0,
                                      /*byte_length=*/kDefaultAllocationSize,
                                      /*pattern=*/&fill_value,
                                      /*pattern_length=*/sizeof(fill_value)));
  IREE_ASSERT_OK(iree_hal_buffer_copy_data(
      /*source_buffer=*/buffer_a,
      /*source_offset=*/0, /*target_buffer=*/buffer_b, /*target_offset=*/0,
      /*data_length=*/kDefaultAllocationSize));

  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), fill_value, kDefaultAllocationSize);

  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_buffer_read_data(
      buffer_b, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer_a);
  iree_hal_buffer_release(buffer_b);
}

// TODO(scotttodd): iree_hal_allocator_wrap_buffer
// TODO(scotttodd): iree_hal_heap_buffer_wrap
// TODO(scotttodd): iree_hal_buffer_map_range
// TODO(scotttodd): revive old tests:
//   https://github.com/google/iree/blob/440edee8a3190d73dbceb24986eed847cac8bd31/iree/hal/buffer_mapping_test.cc

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_BUFFER_MAPPING_TEST_H_
