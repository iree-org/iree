// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// TODO(scotttodd): split into several tests, for example:
//     command_buffer_recording_test (recording/lifetime)
//     command_buffer_dispatch_test
//     command_buffer_fill_test (filling buffers)
//     command_buffer_e2e_test (barriers, dispatches)

namespace iree {
namespace hal {
namespace cts {

using ::testing::ContainerEq;

class CommandBufferTest : public CtsTestBase {
 public:
  CommandBufferTest() {
    // TODO(#4680): command buffer recording so that this can run on sync HAL.
    SkipUnavailableDriver("dylib-sync");
    SkipUnavailableDriver("vmvx-sync");
  }

 protected:
  std::vector<uint8_t> RunFillBufferTest(iree_device_size_t buffer_size,
                                         iree_device_size_t target_offset,
                                         iree_device_size_t fill_length,
                                         const void* pattern,
                                         iree_host_size_t pattern_length) {
    iree_hal_command_buffer_t* command_buffer;
    IREE_CHECK_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        &command_buffer));
    iree_hal_buffer_t* device_buffer;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_),
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL, buffer_size, &device_buffer));

    IREE_CHECK_OK(iree_hal_command_buffer_begin(command_buffer));
    // Start with a zero fill on the entire buffer...
    uint8_t zero_val = 0x0;
    IREE_CHECK_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer, device_buffer, /*target_offset=*/0,
        /*length=*/buffer_size, &zero_val,
        /*pattern_length=*/sizeof(zero_val)));
    // (buffer barrier between the fill operations)
    iree_hal_buffer_barrier_t buffer_barrier;
    buffer_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE;
    buffer_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                                  IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
    buffer_barrier.buffer = device_buffer;
    buffer_barrier.offset = 0;
    buffer_barrier.length = buffer_size;
    IREE_CHECK_OK(iree_hal_command_buffer_execution_barrier(
        command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_STAGE_TRANSFER | IREE_HAL_EXECUTION_STAGE_DISPATCH,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/0, NULL,
        /*buffer_barrier_count=*/1, &buffer_barrier));
    // ... then fill the pattern on top.
    IREE_CHECK_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer, device_buffer,
        /*target_offset=*/target_offset, /*length=*/fill_length,
        /*pattern=*/pattern,
        /*pattern_length=*/pattern_length));
    IREE_CHECK_OK(iree_hal_command_buffer_end(command_buffer));
    IREE_CHECK_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_ANY,
                                             command_buffer));

    std::vector<uint8_t> actual_data(buffer_size);
    IREE_CHECK_OK(
        iree_hal_buffer_read_data(device_buffer, /*source_offset=*/0,
                                  /*target_buffer=*/actual_data.data(),
                                  /*data_length=*/buffer_size));

    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(device_buffer);

    return actual_data;
  }

  static constexpr iree_device_size_t kBufferSize = 4096;
};

TEST_P(CommandBufferTest, Create) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  EXPECT_TRUE((iree_hal_command_buffer_allowed_categories(command_buffer) &
               IREE_HAL_COMMAND_CATEGORY_DISPATCH) ==
              IREE_HAL_COMMAND_CATEGORY_DISPATCH);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(CommandBufferTest, BeginEnd) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(CommandBufferTest, SubmitEmpty) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                            command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(CommandBufferTest, CopyWholeBuffer) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  // Create and fill a host buffer.
  iree_hal_buffer_t* host_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_,
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, kBufferSize, &host_buffer));
  uint8_t i8_val = 0x54;
  IREE_ASSERT_OK(iree_hal_buffer_fill(host_buffer, /*byte_offset=*/0,
                                      /*byte_length=*/kBufferSize, &i8_val,
                                      /*pattern_length=*/sizeof(i8_val)));
  std::vector<uint8_t> reference_buffer(kBufferSize);
  std::memset(reference_buffer.data(), i8_val, kBufferSize);

  // Create a device buffer.
  iree_hal_buffer_t* device_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_,
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, kBufferSize, &device_buffer));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, /*source_buffer=*/host_buffer, /*source_offset=*/0,
      /*target_buffer=*/device_buffer, /*target_offset=*/0,
      /*length=*/kBufferSize));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                            command_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kBufferSize);
  IREE_ASSERT_OK(iree_hal_buffer_read_data(device_buffer, /*source_offset=*/0,
                                           /*target_buffer=*/actual_data.data(),
                                           /*data_length=*/kBufferSize));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

TEST_P(CommandBufferTest, CopySubBuffer) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  iree_hal_buffer_t* device_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_,
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, kBufferSize, &device_buffer));

  // Create another host buffer with a smaller size.
  iree_hal_buffer_t* host_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_,
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, kBufferSize / 2, &host_buffer));

  // Fill the host buffer.
  uint8_t i8_val = 0x88;
  IREE_ASSERT_OK(iree_hal_buffer_fill(host_buffer, /*byte_offset=*/0,
                                      /*byte_length=*/kBufferSize / 2, &i8_val,
                                      /*pattern_length=*/sizeof(i8_val)));
  std::vector<uint8_t> reference_buffer(kBufferSize);
  std::memset(reference_buffer.data() + 8, i8_val, kBufferSize / 2 - 4);

  // Copy the host buffer to the device buffer; zero fill the untouched bytes.
  uint8_t zero_val = 0x0;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, device_buffer, /*target_offset=*/0, /*length=*/8,
      &zero_val, /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, /*source_buffer=*/host_buffer, /*source_offset=*/4,
      /*target_buffer=*/device_buffer, /*target_offset=*/8,
      /*length=*/kBufferSize / 2 - 4));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, device_buffer, /*target_offset=*/8 + kBufferSize / 2 - 4,
      /*length=*/kBufferSize - (8 + kBufferSize / 2 - 4), &zero_val,
      /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                            command_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kBufferSize);
  IREE_ASSERT_OK(iree_hal_buffer_read_data(device_buffer, /*source_offset=*/0,
                                           /*target_buffer=*/actual_data.data(),
                                           /*data_length=*/kBufferSize));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

TEST_P(CommandBufferTest, FillBuffer_pattern1_size1_offset0_length1) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern1_size5_offset0_length5) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern1_size16_offset0_length1) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern1_size16_offset0_length3) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern1_size16_offset0_length8) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern1_size16_offset2_length8) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern2_size2_offset0_length2) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern2_size16_offset0_length8) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern2_size16_offset0_length10) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern2_size16_offset2_length8) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern4_size4_offset0_length4) {
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

TEST_P(CommandBufferTest, FillBuffer_pattern4_size16_offset0_length8) {
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

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, CommandBufferTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
