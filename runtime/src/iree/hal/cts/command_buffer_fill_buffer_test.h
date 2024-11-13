// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_COMMAND_BUFFER_FILL_BUFFER_TEST_H_
#define IREE_HAL_CTS_COMMAND_BUFFER_FILL_BUFFER_TEST_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class CommandBufferFillBufferTest
    : public CTSTestBase<::testing::TestWithParam<RecordingType>> {
 protected:
  std::vector<uint8_t> RunFillBufferTest(RecordingType recording_type,
                                         iree_device_size_t buffer_size,
                                         iree_device_size_t target_offset,
                                         iree_device_size_t fill_length,
                                         const void* pattern,
                                         iree_host_size_t pattern_length) {
    switch (recording_type) {
      default:
      case RecordingType::kDirect:
        return RunFillBufferDirectTest(buffer_size, target_offset, fill_length,
                                       pattern, pattern_length);
      case RecordingType::kIndirect:
        return RunFillBufferIndirectTest(buffer_size, target_offset,
                                         fill_length, pattern, pattern_length);
    }
  }

  std::vector<uint8_t> RunFillBufferDirectTest(
      iree_device_size_t buffer_size, iree_device_size_t target_offset,
      iree_device_size_t fill_length, const void* pattern,
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
        command_buffer,
        iree_hal_make_buffer_ref(device_buffer, target_offset, fill_length),
        pattern, pattern_length, IREE_HAL_FILL_FLAG_NONE));
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

  std::vector<uint8_t> RunFillBufferIndirectTest(
      iree_device_size_t buffer_size, iree_device_size_t target_offset,
      iree_device_size_t fill_length, const void* pattern,
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
        command_buffer,
        iree_hal_make_buffer_ref(device_buffer, target_offset, fill_length),
        pattern, pattern_length, IREE_HAL_FILL_FLAG_NONE));
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

TEST_P(CommandBufferFillBufferTest, FillBuffer_pattern1_size1_offset0_length1) {
  iree_device_size_t buffer_size = 1;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 1;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, FillBuffer_pattern1_size5_offset0_length5) {
  iree_device_size_t buffer_size = 5;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 5;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x07,  //
                                        0x07};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern1_size16_offset0_length1) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 1;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern1_size16_offset0_length3) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 3;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern1_size16_offset0_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 8;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x07,  //
                                        0x07, 0x07, 0x07, 0x07,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern1_size16_offset2_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 2;
  iree_device_size_t fill_length = 8;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x07, 0x07,  //
                                        0x07, 0x07, 0x07, 0x07,  //
                                        0x07, 0x07, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, FillBuffer_pattern2_size2_offset0_length2) {
  iree_device_size_t buffer_size = 2;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 2;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern2_size16_offset0_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 8;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern2_size16_offset0_length10) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 10;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern2_size16_offset2_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 2;
  iree_device_size_t fill_length = 8;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, FillBuffer_pattern4_size4_offset0_length4) {
  iree_device_size_t buffer_size = 4;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 4;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x45, 0xCD, 0x23, 0xAB};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern4_size4_offset16_length4) {
  iree_device_size_t buffer_size = 20;
  iree_device_size_t target_offset = 16;
  iree_device_size_t fill_length = 4;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer(buffer_size, 0);
  *reinterpret_cast<uint32_t*>(&reference_buffer[target_offset]) = pattern;
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern4_size16_offset0_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 8;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x45, 0xCD, 0x23, 0xAB,  //
                                        0x45, 0xCD, 0x23, 0xAB,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest,
       FillBuffer_pattern4_size16_offset8_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 8;
  iree_device_size_t fill_length = 8;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x45, 0xCD, 0x23, 0xAB,  //
                                        0x45, 0xCD, 0x23, 0xAB};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(GetParam(), buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

INSTANTIATE_TEST_SUITE_P(CommandBufferFillBufferTest,
                         CommandBufferFillBufferTest,
                         ::testing::Values(RecordingType::kDirect,
                                           RecordingType::kIndirect),
                         GenerateTestName());

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_COMMAND_BUFFER_FILL_BUFFER_TEST_H_
