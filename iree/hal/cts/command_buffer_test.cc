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

#include <cstring>
#include <vector>

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
 protected:
  static constexpr iree_device_size_t kBufferSize = 4096;
};

TEST_P(CommandBufferTest, Create) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, &command_buffer));

  EXPECT_TRUE((iree_hal_command_buffer_allowed_categories(command_buffer) &
               IREE_HAL_COMMAND_CATEGORY_DISPATCH) ==
              IREE_HAL_COMMAND_CATEGORY_DISPATCH);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(CommandBufferTest, BeginEnd) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(CommandBufferTest, SubmitEmpty) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                            command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(CommandBufferTest, FillBufferWithRepeatedBytes) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, &command_buffer));

  iree_hal_buffer_t* device_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_,
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, kBufferSize, &device_buffer));

  std::vector<uint8_t> reference_buffer(kBufferSize);

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  // Fill the device buffer with segments of different values so that we can
  // test both fill and offset/size.
  uint8_t val1 = 0x07;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, device_buffer,
      /*target_offset=*/0, /*length=*/kBufferSize / 4, /*pattern=*/&val1,
      /*pattern_length=*/sizeof(val1)));
  std::memset(reference_buffer.data(), val1, kBufferSize / 4);

  uint8_t val2 = 0xbe;
  IREE_ASSERT_OK(
      iree_hal_command_buffer_fill_buffer(command_buffer, device_buffer,
                                          /*target_offset=*/kBufferSize / 4,
                                          /*length=*/kBufferSize / 4,
                                          /*pattern=*/&val2,
                                          /*pattern_length=*/sizeof(val2)));
  std::memset(reference_buffer.data() + kBufferSize / 4, val2, kBufferSize / 4);

  uint8_t val3 = 0x54;
  IREE_ASSERT_OK(
      iree_hal_command_buffer_fill_buffer(command_buffer, device_buffer,
                                          /*target_offset=*/kBufferSize / 2,
                                          /*length=*/kBufferSize / 2,
                                          /*pattern=*/&val3,
                                          /*pattern_length=*/sizeof(val3)));
  std::memset(reference_buffer.data() + kBufferSize / 2, val3, kBufferSize / 2);

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
}

TEST_P(CommandBufferTest, CopyWholeBuffer) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, &command_buffer));

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
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, &command_buffer));

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

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, /*source_buffer=*/host_buffer, /*source_offset=*/4,
      /*target_buffer=*/device_buffer, /*target_offset=*/8,
      /*length=*/kBufferSize / 2 - 4));
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

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, CommandBufferTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
