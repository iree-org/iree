// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_COMMAND_BUFFER_UPDATE_BUFFER_TEST_H_
#define IREE_HAL_CTS_COMMAND_BUFFER_UPDATE_BUFFER_TEST_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class CommandBufferUpdateBufferTest : public CTSTestBase<> {};

TEST_F(CommandBufferUpdateBufferTest, UpdateBufferWholeBuffer) {
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
      command_buffer,
      /*source_buffer=*/source_buffer.data(), /*source_offset=*/0,
      iree_hal_make_buffer_ref(device_buffer, 0, target_buffer_size),
      IREE_HAL_UPDATE_FLAG_NONE));
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

TEST_F(CommandBufferUpdateBufferTest, UpdateBufferWithOffsets) {
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
      command_buffer,
      /*source_buffer=*/source_buffer.data(), /*source_offset=*/4,
      iree_hal_make_buffer_ref(device_buffer,
                               /*target_offset=*/4, /*length=*/8),
      IREE_HAL_UPDATE_FLAG_NONE));
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

TEST_F(CommandBufferUpdateBufferTest, UpdateBufferSubspan) {
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
      command_buffer,
      /*source_buffer=*/source_buffer.data(), /*source_offset=*/4,
      iree_hal_make_buffer_ref(buffer_subspan,
                               /*target_offset=*/4, /*length=*/4),
      IREE_HAL_UPDATE_FLAG_NONE));
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

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_COMMAND_BUFFER_UPDATE_BUFFER_TEST_H_
