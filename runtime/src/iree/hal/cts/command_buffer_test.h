// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_COMMAND_BUFFER_TEST_H_
#define IREE_HAL_CTS_COMMAND_BUFFER_TEST_H_

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

class command_buffer_test : public CtsTestBase {
 protected:
  void CreateZeroedDeviceBuffer(iree_device_size_t buffer_size,
                                iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), params, buffer_size,
        iree_const_byte_span_empty(), &device_buffer));
    IREE_ASSERT_OK(
        iree_hal_buffer_map_zero(device_buffer, 0, IREE_WHOLE_BUFFER));
    *out_buffer = device_buffer;
  }

  std::vector<uint8_t> RunFillBufferTest(iree_device_size_t buffer_size,
                                         iree_device_size_t target_offset,
                                         iree_device_size_t fill_length,
                                         const void* pattern,
                                         iree_host_size_t pattern_length) {
    iree_hal_buffer_t* device_buffer = NULL;
    CreateZeroedDeviceBuffer(buffer_size, &device_buffer);

    iree_hal_command_buffer_t* command_buffer = NULL;
    IREE_CHECK_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, &command_buffer));
    IREE_CHECK_OK(iree_hal_command_buffer_begin(command_buffer));

    // Fill the pattern.
    IREE_CHECK_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer, device_buffer, target_offset, fill_length, pattern,
        pattern_length));
    IREE_CHECK_OK(iree_hal_command_buffer_end(command_buffer));
    IREE_CHECK_OK(SubmitCommandBufferAndWait(command_buffer));

    // Read data for returning.
    std::vector<uint8_t> actual_data(buffer_size);
    IREE_CHECK_OK(iree_hal_device_transfer_d2h(
        device_, device_buffer, /*source_offset=*/0,
        /*target_buffer=*/actual_data.data(),
        /*data_length=*/buffer_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

    // Cleanup and return.
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(device_buffer);
    return actual_data;
  }
};

TEST_P(command_buffer_test, Create) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  EXPECT_TRUE((iree_hal_command_buffer_allowed_categories(command_buffer) &
               IREE_HAL_COMMAND_CATEGORY_DISPATCH) ==
              IREE_HAL_COMMAND_CATEGORY_DISPATCH);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(command_buffer_test, BeginEnd) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(command_buffer_test, SubmitEmpty) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(command_buffer_test, CopyWholeBuffer) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  uint8_t i8_val = 0x54;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data(), i8_val, kDefaultAllocationSize);

  // Create and fill a host buffer.
  iree_hal_buffer_params_t host_params = {0};
  host_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                      IREE_HAL_BUFFER_USAGE_TRANSFER |
                      IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* host_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, host_params, kDefaultAllocationSize,
      iree_make_const_byte_span(reference_buffer.data(),
                                reference_buffer.size()),
      &host_buffer));

  // Create a device buffer.
  iree_hal_buffer_params_t device_params = {0};
  device_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  device_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* device_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      iree_const_byte_span_empty(), &device_buffer));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, /*source_buffer=*/host_buffer, /*source_offset=*/0,
      /*target_buffer=*/device_buffer, /*target_offset=*/0,
      /*length=*/kDefaultAllocationSize));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0,
      /*target_buffer=*/actual_data.data(),
      /*data_length=*/kDefaultAllocationSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

TEST_P(command_buffer_test, CopySubBuffer) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  iree_hal_buffer_params_t device_params = {0};
  device_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  device_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* device_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, device_params, kDefaultAllocationSize,
      iree_const_byte_span_empty(), &device_buffer));

  uint8_t i8_val = 0x88;
  std::vector<uint8_t> reference_buffer(kDefaultAllocationSize);
  std::memset(reference_buffer.data() + 8, i8_val,
              kDefaultAllocationSize / 2 - 4);

  // Create another host buffer with a smaller size.
  iree_hal_buffer_params_t host_params = {0};
  host_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  host_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                      IREE_HAL_BUFFER_USAGE_TRANSFER |
                      IREE_HAL_BUFFER_USAGE_MAPPING;
  std::vector<uint8_t> host_buffer_data(kDefaultAllocationSize, i8_val);
  iree_hal_buffer_t* host_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, host_params, host_buffer_data.size() / 2,
      iree_make_const_byte_span(host_buffer_data.data(),
                                host_buffer_data.size() / 2),
      &host_buffer));

  // Copy the host buffer to the device buffer; zero fill the untouched bytes.
  uint8_t zero_val = 0x0;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, device_buffer, /*target_offset=*/0, /*length=*/8,
      &zero_val, /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, /*source_buffer=*/host_buffer, /*source_offset=*/4,
      /*target_buffer=*/device_buffer, /*target_offset=*/8,
      /*length=*/kDefaultAllocationSize / 2 - 4));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, device_buffer,
      /*target_offset=*/8 + kDefaultAllocationSize / 2 - 4,
      /*length=*/kDefaultAllocationSize - (8 + kDefaultAllocationSize / 2 - 4),
      &zero_val,
      /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kDefaultAllocationSize);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0,
      /*target_buffer=*/actual_data.data(),
      /*data_length=*/kDefaultAllocationSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size1_offset0_length1) {
  iree_device_size_t buffer_size = 1;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 1;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size5_offset0_length5) {
  iree_device_size_t buffer_size = 5;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 5;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x07,  //
                                        0x07};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size16_offset0_length1) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 1;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size16_offset0_length3) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 3;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size16_offset0_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 8;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x07,  //
                                        0x07, 0x07, 0x07, 0x07,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size16_offset2_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 2;
  iree_device_size_t fill_length = 8;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x07, 0x07,  //
                                        0x07, 0x07, 0x07, 0x07,  //
                                        0x07, 0x07, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern2_size2_offset0_length2) {
  iree_device_size_t buffer_size = 2;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 2;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern2_size16_offset0_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 8;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern2_size16_offset0_length10) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 10;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern2_size16_offset2_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 2;
  iree_device_size_t fill_length = 8;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern4_size4_offset0_length4) {
  iree_device_size_t buffer_size = 4;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 4;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x45, 0xCD, 0x23, 0xAB};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern4_size4_offset16_length4) {
  iree_device_size_t buffer_size = 20;
  iree_device_size_t target_offset = 16;
  iree_device_size_t fill_length = 4;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer(buffer_size, 0);
  *reinterpret_cast<uint32_t*>(&reference_buffer[target_offset]) = pattern;
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern4_size16_offset0_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 8;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x45, 0xCD, 0x23, 0xAB,  //
                                        0x45, 0xCD, 0x23, 0xAB,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern4_size16_offset8_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 8;
  iree_device_size_t fill_length = 8;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x45, 0xCD, 0x23, 0xAB,  //
                                        0x45, 0xCD, 0x23, 0xAB};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, UpdateBufferWholeBuffer) {
  iree_device_size_t target_buffer_size = 16;
  std::vector<uint8_t> source_buffer{0x01, 0x02, 0x03, 0x04,  //
                                     0x05, 0x06, 0x07, 0x08,  //
                                     0xA1, 0xA2, 0xA3, 0xA4,  //
                                     0xA5, 0xA6, 0xA7, 0xA8};

  iree_hal_buffer_t* device_buffer = NULL;
  CreateZeroedDeviceBuffer(target_buffer_size, &device_buffer);

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_CHECK_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_CHECK_OK(iree_hal_command_buffer_begin(command_buffer));

  // Issue the update_buffer command.
  IREE_CHECK_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, source_buffer.data(), /*source_offset=*/0, device_buffer,
      /*target_offset=*/0, /*length=*/target_buffer_size));
  IREE_CHECK_OK(iree_hal_command_buffer_end(command_buffer));
  IREE_CHECK_OK(SubmitCommandBufferAndWait(command_buffer));

  // Check that the contents match what we expect.
  std::vector<uint8_t> actual_data(target_buffer_size);
  IREE_CHECK_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0, actual_data.data(),
      actual_data.size(), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(source_buffer));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
}

TEST_P(command_buffer_test, UpdateBufferWithOffsets) {
  iree_device_size_t target_buffer_size = 16;
  std::vector<uint8_t> source_buffer{0x01, 0x02, 0x03, 0x04,  //
                                     0x05, 0x06, 0x07, 0x08,  //
                                     0xA1, 0xA2, 0xA3, 0xA4,  //
                                     0xA5, 0xA6, 0xA7, 0xA8};

  iree_hal_buffer_t* device_buffer = NULL;
  CreateZeroedDeviceBuffer(target_buffer_size, &device_buffer);

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_CHECK_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_CHECK_OK(iree_hal_command_buffer_begin(command_buffer));

  // Issue the update_buffer command.
  IREE_CHECK_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, source_buffer.data(), /*source_offset=*/4, device_buffer,
      /*target_offset=*/4, /*length=*/8));
  IREE_CHECK_OK(iree_hal_command_buffer_end(command_buffer));
  IREE_CHECK_OK(SubmitCommandBufferAndWait(command_buffer));

  // Check that the contents match what we expect.
  std::vector<uint8_t> actual_data(target_buffer_size);
  IREE_CHECK_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0, actual_data.data(),
      actual_data.size(), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x00, 0x00,  //
                                        0x05, 0x06, 0x07, 0x08,  //
                                        0xA1, 0xA2, 0xA3, 0xA4,  //
                                        0x00, 0x00, 0x00, 0x00};
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
}

TEST_P(command_buffer_test, UpdateBufferSubspan) {
  iree_device_size_t target_buffer_size = 16;
  std::vector<uint8_t> source_buffer{0x01, 0x02, 0x03, 0x04,  //
                                     0x05, 0x06, 0x07, 0x08,  //
                                     0xA1, 0xA2, 0xA3, 0xA4,  //
                                     0xA5, 0xA6, 0xA7, 0xA8};

  iree_hal_buffer_t* device_buffer = NULL;
  CreateZeroedDeviceBuffer(target_buffer_size, &device_buffer);

  // Create a subspan.
  iree_device_size_t subspan_length = 8;
  iree_hal_buffer_t* buffer_subspan;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(device_buffer, /*byte_offset=*/4,
                                         subspan_length, &buffer_subspan));

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_CHECK_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_CHECK_OK(iree_hal_command_buffer_begin(command_buffer));

  // Issue the update_buffer command.
  IREE_CHECK_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, source_buffer.data(), /*source_offset=*/4, buffer_subspan,
      /*target_offset=*/4, /*length=*/4));
  IREE_CHECK_OK(iree_hal_command_buffer_end(command_buffer));
  IREE_CHECK_OK(SubmitCommandBufferAndWait(command_buffer));

  // Check that the contents match what we expect.
  std::vector<uint8_t> actual_data(target_buffer_size);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, device_buffer, /*source_offset=*/0, actual_data.data(),
      actual_data.size(), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x05, 0x06, 0x07, 0x08,  //
                                        0x00, 0x00, 0x00, 0x00};
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
  // Also check the subspan.
  std::vector<uint8_t> actual_data_subspan(subspan_length);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, buffer_subspan, /*source_offset=*/0, actual_data_subspan.data(),
      actual_data_subspan.size(), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  std::vector<uint8_t> reference_buffer_subspan{0x00, 0x00, 0x00, 0x00,  //
                                                0x05, 0x06, 0x07, 0x08};
  EXPECT_THAT(actual_data_subspan, ContainerEq(reference_buffer_subspan));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer_subspan);
  iree_hal_buffer_release(device_buffer);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_COMMAND_BUFFER_TEST_H_
