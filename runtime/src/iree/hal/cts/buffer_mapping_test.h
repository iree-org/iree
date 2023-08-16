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
constexpr iree_device_size_t kDefaultAllocationSize = 1024;
}  // namespace

// Tests for buffer mapping (IREE_HAL_BUFFER_USAGE_MAPPING) support and
// for `iree_hal_buffer_*` functions which require buffer mapping.
//
// Note that most of these tests first write into a buffer using one or more
// functions then read the (possibly partial) contents of that buffer using
// `iree_hal_buffer_map_read`. As the buffer read implementation is
// nontrivial, particularly on implementations with complex host/device splits,
// test failures may indicate issues in either the code doing the writing or the
// code doing the reading.
//
// Where applicable, tests for each function are organized in increasing order
// of complexity, such as:
//   * write to full buffer
//   * write with an offset and length
//   * write into a subspan of a buffer

class buffer_mapping_test : public CtsTestBase {
 protected:
  void AllocateUninitializedBuffer(iree_device_size_t buffer_size,
                                   iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), params, buffer_size,
        &device_buffer));
    *out_buffer = device_buffer;
  }
};

TEST_P(buffer_mapping_test, AllocatorSupportsBufferMapping) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_device_size_t allocation_size = 0;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(device_allocator_, params,
                                                    kDefaultAllocationSize,
                                                    &params, &allocation_size);
  EXPECT_TRUE(iree_all_bits_set(compatibility,
                                IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE));

  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(allocation_size, &buffer);

  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_memory_type(buffer), params.type));
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer), params.usage));
  EXPECT_GE(iree_hal_buffer_allocation_size(buffer), allocation_size);

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, ZeroWholeBuffer) {
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer);

  // Zero the entire buffer.
  IREE_ASSERT_OK(
      iree_hal_buffer_map_zero(buffer, /*byte_offset=*/0, IREE_WHOLE_BUFFER));

  // Check that the contents match what we expect.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), 0, kDefaultAllocationSize);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, ZeroWithOffset) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Fill the entire buffer then zero only a segment of it.
  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, /*byte_offset=*/0,
                                          IREE_WHOLE_BUFFER, &fill_value,
                                          sizeof(fill_value)));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_zero(buffer, /*byte_offset=*/4, /*byte_length=*/8));

  // Check that the contents match what we expect.
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer{0xFF, 0xFF, 0xFF, 0xFF,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0xFF, 0xFF, 0xFF, 0xFF};
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, ZeroSubspan) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Fill the entire buffer.
  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, /*byte_offset=*/0,
                                          IREE_WHOLE_BUFFER, &fill_value,
                                          sizeof(fill_value)));

  // Create a subspan.
  iree_device_size_t subspan_length = 8;
  iree_hal_buffer_t* buffer_subspan = NULL;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, /*byte_offset=*/4,
                                         subspan_length, &buffer_subspan));

  // Zero part of the subspan.
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(buffer_subspan, /*byte_offset=*/4,
                                          /*byte_length=*/4));

  // Check that the contents match what we expect.
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer{0xFF, 0xFF, 0xFF, 0xFF,  //
                                        0xFF, 0xFF, 0xFF, 0xFF,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0xFF, 0xFF, 0xFF, 0xFF};
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
  // Also check the subspan.
  std::vector<uint8_t> actual_data_subspan(subspan_length);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer_subspan, /*source_offset=*/0,
                                          actual_data_subspan.data(),
                                          actual_data_subspan.size()));
  std::vector<uint8_t> reference_buffer_subspan{0xFF, 0xFF, 0xFF, 0xFF,  //
                                                0x00, 0x00, 0x00, 0x00};
  EXPECT_THAT(actual_data_subspan, ContainerEq(reference_buffer_subspan));

  iree_hal_buffer_release(buffer_subspan);
  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, FillEmpty) {
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer);

  // Zero the whole buffer then "fill" 0 bytes with a different pattern.
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(buffer, 0, IREE_WHOLE_BUFFER));
  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer, /*byte_offset=*/0,
                               /*byte_length=*/0,  // <---- empty!
                               /*pattern=*/&fill_value,
                               /*pattern_length=*/sizeof(fill_value)));

  // Check that the buffer is still all zeroes.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), 0, kDefaultAllocationSize);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, FillWholeBuffer) {
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer);

  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer, /*byte_offset=*/0,
                               /*byte_length=*/IREE_WHOLE_BUFFER,
                               /*pattern=*/&fill_value,
                               /*pattern_length=*/sizeof(fill_value)));

  // Check that the buffer is filled with the pattern.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), fill_value, kDefaultAllocationSize);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, FillWithOffset) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Zero the entire buffer then fill only a segment of it.
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(buffer, 0, IREE_WHOLE_BUFFER));
  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer, /*byte_offset=*/4,
                               /*byte_length=*/8,
                               /*pattern=*/&fill_value,
                               /*pattern_length=*/sizeof(fill_value)));

  // Check that only the segment of the buffer is filled with the pattern.
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_offset_buffer{0x00, 0x00, 0x00, 0x00,  //
                                               0xFF, 0xFF, 0xFF, 0xFF,  //
                                               0xFF, 0xFF, 0xFF, 0xFF,  //
                                               0x00, 0x00, 0x00, 0x00};
  EXPECT_THAT(actual_data, ContainerEq(reference_offset_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, FillSubspan) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Zero the entire buffer.
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(buffer, 0, IREE_WHOLE_BUFFER));

  // Create a subspan.
  iree_device_size_t subspan_length = 8;
  iree_hal_buffer_t* buffer_subspan = NULL;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, /*byte_offset=*/4,
                                         subspan_length, &buffer_subspan));

  // Fill part of the subspan.
  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer_subspan, /*byte_offset=*/4,
                               /*byte_length=*/4,
                               /*pattern=*/&fill_value,
                               /*pattern_length=*/sizeof(fill_value)));

  // Check that the contents match what we expect.
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0xFF, 0xFF, 0xFF, 0xFF,  //
                                        0x00, 0x00, 0x00, 0x00};
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
  // Also check the subspan.
  std::vector<uint8_t> actual_data_subspan(subspan_length);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer_subspan, /*source_offset=*/0,
                                          actual_data_subspan.data(),
                                          actual_data_subspan.size()));
  std::vector<uint8_t> reference_buffer_subspan{0x00, 0x00, 0x00, 0x00,  //
                                                0xFF, 0xFF, 0xFF, 0xFF};
  EXPECT_THAT(actual_data_subspan, ContainerEq(reference_buffer_subspan));

  iree_hal_buffer_release(buffer_subspan);
  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, ReadData) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Zero the first half, fill the second half.
  IREE_ASSERT_OK(
      iree_hal_buffer_map_zero(buffer, /*byte_offset=*/0, /*byte_length=*/8));
  uint8_t fill_value = 0xFF;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer, /*byte_offset=*/8,
                               /*byte_length=*/8,
                               /*pattern=*/&fill_value,
                               /*pattern_length=*/sizeof(fill_value)));

  // Read the entire buffer.
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0xFF, 0xFF, 0xFF, 0xFF,  //
                                        0xFF, 0xFF, 0xFF, 0xFF};
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Read only a segment of the buffer.
  std::vector<uint8_t> actual_data_offset(8);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer, /*source_offset=*/4,
                                          actual_data_offset.data(),
                                          actual_data_offset.size()));
  std::vector<uint8_t> reference_buffer_offset{0x00, 0x00, 0x00, 0x00,  //
                                               0xFF, 0xFF, 0xFF, 0xFF};
  EXPECT_THAT(actual_data_offset, ContainerEq(reference_buffer_offset));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, ReadDataSubspan) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Fill a few segments with distinct values.
  uint8_t value = 0xAA;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, 4, &value, sizeof(value)));
  value = 0xBB;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 4, 4, &value, sizeof(value)));
  value = 0xCC;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 8, 4, &value, sizeof(value)));
  value = 0xDD;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer, 12, 4, &value, sizeof(value)));

  // Create a subspan.
  iree_device_size_t subspan_length = 8;
  iree_hal_buffer_t* buffer_subspan = NULL;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, /*byte_offset=*/4,
                                         subspan_length, &buffer_subspan));

  // Read the entire buffer subspan.
  std::vector<uint8_t> actual_data(subspan_length);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer_subspan, /*source_offset=*/0,
                                          actual_data.data(),
                                          actual_data.size()));
  std::vector<uint8_t> reference_buffer{0xBB, 0xBB, 0xBB, 0xBB,  //
                                        0xCC, 0xCC, 0xCC, 0xCC};
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Read only a segment of the buffer.
  std::vector<uint8_t> actual_data_offset(4);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer_subspan, /*source_offset=*/4,
                                          actual_data_offset.data(),
                                          actual_data_offset.size()));
  std::vector<uint8_t> reference_buffer_offset{0xCC, 0xCC, 0xCC, 0xCC};
  EXPECT_THAT(actual_data_offset, ContainerEq(reference_buffer_offset));

  iree_hal_buffer_release(buffer_subspan);
  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, WriteDataWholeBuffer) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Write over the whole buffer.
  uint8_t fill_value = 0xFF;
  std::vector<uint8_t> reference_buffer(buffer_size);
  std::memset(reference_buffer.data(), fill_value, buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(buffer, /*target_offset=*/0,
                                           reference_buffer.data(),
                                           reference_buffer.size()));

  // Check that entire buffer was written to.
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, WriteDataWithOffset) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Zero the entire buffer.
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(buffer, 0, IREE_WHOLE_BUFFER));

  // Write over part of the buffer.
  std::vector<uint8_t> fill_buffer{0x11, 0x22, 0x33, 0x44,  //
                                   0x55, 0x66, 0x77, 0x88};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      buffer, /*target_offset=*/4, fill_buffer.data(), fill_buffer.size()));

  // Check that the contents match what we expect.
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x00, 0x00,  //
                                        0x11, 0x22, 0x33, 0x44,  //
                                        0x55, 0x66, 0x77, 0x88,  //
                                        0x00, 0x00, 0x00, 0x00};
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, WriteDataSubspan) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  // Zero the entire buffer.
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(buffer, 0, IREE_WHOLE_BUFFER));

  // Create a subspan.
  iree_device_size_t subspan_length = 8;
  iree_hal_buffer_t* buffer_subspan = NULL;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, /*byte_offset=*/4,
                                         subspan_length, &buffer_subspan));

  // Write over part of the subspan.
  std::vector<uint8_t> fill_buffer{0x11, 0x22, 0x33, 0x44};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(buffer_subspan, /*target_offset=*/4,
                                           fill_buffer.data(),
                                           fill_buffer.size()));

  // Check that the contents match what we expect.
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x11, 0x22, 0x33, 0x44,  //
                                        0x00, 0x00, 0x00, 0x00};
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
  // Also check the subspan.
  std::vector<uint8_t> actual_data_subspan(subspan_length);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer_subspan, /*source_offset=*/0,
                                          actual_data_subspan.data(),
                                          actual_data_subspan.size()));
  std::vector<uint8_t> reference_buffer_subspan{0x00, 0x00, 0x00, 0x00,  //
                                                0x11, 0x22, 0x33, 0x44};
  EXPECT_THAT(actual_data_subspan, ContainerEq(reference_buffer_subspan));

  iree_hal_buffer_release(buffer_subspan);
  iree_hal_buffer_release(buffer);
}

TEST_P(buffer_mapping_test, CopyData) {
  iree_hal_buffer_t* buffer_a = NULL;
  iree_hal_buffer_t* buffer_b = NULL;
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer_a);
  AllocateUninitializedBuffer(kDefaultAllocationSize, &buffer_b);

  uint8_t fill_value = 0x07;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer_a, /*byte_offset=*/0,
                               /*byte_length=*/kDefaultAllocationSize,
                               /*pattern=*/&fill_value,
                               /*pattern_length=*/sizeof(fill_value)));
  IREE_ASSERT_OK(iree_hal_buffer_map_copy(
      /*source_buffer=*/buffer_a,
      /*source_offset=*/0, /*target_buffer=*/buffer_b, /*target_offset=*/0,
      /*data_length=*/kDefaultAllocationSize));

  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), fill_value, kDefaultAllocationSize);

  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer_b, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer_a);
  iree_hal_buffer_release(buffer_b);
}

// Maps a buffer range for reading from device -> host.
// This is roughly what iree_hal_buffer_map_read does internally.
TEST_P(buffer_mapping_test, MapRangeRead) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  uint8_t fill_value = 0xEF;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, /*byte_offset=*/0,
                                          IREE_WHOLE_BUFFER, &fill_value,
                                          sizeof(fill_value)));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/buffer_size, &mapping));
  EXPECT_EQ(buffer, mapping.buffer);
  EXPECT_GE(mapping.contents.data_length, (iree_host_size_t)buffer_size);

  std::vector<uint8_t> reference_buffer(buffer_size);
  std::memset(reference_buffer.data(), fill_value, buffer_size);
  std::vector<uint8_t> mapping_data(
      mapping.contents.data,
      mapping.contents.data + mapping.contents.data_length);
  EXPECT_THAT(mapping_data, ContainerEq(reference_buffer));

  iree_hal_buffer_unmap_range(&mapping);
  iree_hal_buffer_release(buffer);
}

// Maps a buffer range for writing from host -> device.
// This is roughly what iree_hal_buffer_map_write does internally.
TEST_P(buffer_mapping_test, MapRangeWrite) {
  iree_device_size_t buffer_size = 16;
  iree_hal_buffer_t* buffer = NULL;
  AllocateUninitializedBuffer(buffer_size, &buffer);

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
      /*byte_offset=*/0, /*byte_length=*/buffer_size, &mapping));
  EXPECT_EQ(buffer, mapping.buffer);
  EXPECT_GE(mapping.contents.data_length, (iree_host_size_t)buffer_size);

  // Write into the mapped memory, flush for device access, then read back.
  uint8_t fill_value = 0x12;
  std::memset(mapping.contents.data, fill_value, buffer_size);
  IREE_ASSERT_OK(
      iree_hal_buffer_mapping_flush_range(&mapping, /*byte_offset=*/0,
                                          /*byte_length=*/buffer_size));
  std::vector<uint8_t> actual_data(buffer_size);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      buffer, /*source_offset=*/0, actual_data.data(), actual_data.size()));
  std::vector<uint8_t> reference_buffer(buffer_size);
  std::memset(reference_buffer.data(), fill_value, buffer_size);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_unmap_range(&mapping);
  iree_hal_buffer_release(buffer);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_BUFFER_MAPPING_TEST_H_
