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

#include "iree/base/status.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

using ::testing::ContainerEq;

class CommandBufferTest : public CtsTestBase {
 protected:
  static constexpr device_size_t kBufferNumBytes = 16;

  void SubmitAndWait(CommandQueue* command_queue,
                     CommandBuffer* command_buffer) {
    IREE_ASSERT_OK_AND_ASSIGN(auto signal_semaphore,
                              device_->CreateSemaphore(0ull));

    IREE_ASSERT_OK(command_queue->Submit(
        {{}, {command_buffer}, {{signal_semaphore.get(), 1ull}}}));
    IREE_ASSERT_OK(signal_semaphore->Wait(1ull, InfiniteFuture()));
  }
};

TEST_P(CommandBufferTest, Create) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      device_->CreateCommandBuffer(CommandBufferMode::kOneShot,
                                   CommandCategory::kDispatch));

  EXPECT_TRUE((command_buffer->mode() & CommandBufferMode::kOneShot) ==
              CommandBufferMode::kOneShot);
  EXPECT_TRUE((command_buffer->command_categories() &
               CommandCategory::kDispatch) == CommandCategory::kDispatch);
  EXPECT_FALSE(command_buffer->is_recording());
}

TEST_P(CommandBufferTest, BeginEnd) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      device_->CreateCommandBuffer(CommandBufferMode::kOneShot,
                                   CommandCategory::kDispatch));

  EXPECT_FALSE(command_buffer->is_recording());
  IREE_EXPECT_OK(command_buffer->Begin());
  EXPECT_TRUE(command_buffer->is_recording());
  IREE_EXPECT_OK(command_buffer->End());
  EXPECT_FALSE(command_buffer->is_recording());
}

TEST_P(CommandBufferTest, FillBufferWithRepeatedBytes) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      device_->CreateCommandBuffer(CommandBufferMode::kOneShot,
                                   CommandCategory::kTransfer));

  IREE_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      device_->allocator()->Allocate(
          MemoryType::kDeviceLocal | MemoryType::kHostVisible,
          BufferUsage::kAll, kBufferNumBytes));

  std::vector<uint8_t> reference_buffer(kBufferNumBytes);

  IREE_EXPECT_OK(command_buffer->Begin());

  // Fill the device buffer with segments of different values so that we can
  // test both fill and offset/size.

  uint8_t val1 = 0x07;
  IREE_EXPECT_OK(command_buffer->FillBuffer(device_buffer.get(),
                                            /*target_offset=*/0,
                                            /*length=*/kBufferNumBytes / 4,
                                            &val1,
                                            /*pattern_length=*/1));
  std::memset(reference_buffer.data(), val1, kBufferNumBytes / 4);

  uint8_t val2 = 0xbe;
  IREE_EXPECT_OK(
      command_buffer->FillBuffer(device_buffer.get(),
                                 /*target_offset=*/kBufferNumBytes / 4,
                                 /*length=*/kBufferNumBytes / 4, &val2,
                                 /*pattern_length=*/1));
  std::memset(reference_buffer.data() + kBufferNumBytes / 4, val2,
              kBufferNumBytes / 4);

  uint8_t val3 = 0x54;
  IREE_EXPECT_OK(
      command_buffer->FillBuffer(device_buffer.get(),
                                 /*target_offset=*/kBufferNumBytes / 2,
                                 /*length=*/kBufferNumBytes / 2, &val3,
                                 /*pattern_length=*/1));
  std::memset(reference_buffer.data() + kBufferNumBytes / 2, val3,
              kBufferNumBytes / 2);

  IREE_EXPECT_OK(command_buffer->End());

  SubmitAndWait(device_->transfer_queues()[0], command_buffer.get());

  // Read back the device buffer.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto mapped_memory,
      device_buffer->MapMemory<uint8_t>(MemoryAccess::kRead));
  IREE_EXPECT_OK(mapped_memory.Invalidate());

  std::vector<uint8_t> actual_data(mapped_memory.data(),
                                   mapped_memory.data() + kBufferNumBytes);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferTest, CopyWholeBuffer) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      device_->CreateCommandBuffer(CommandBufferMode::kOneShot,
                                   CommandCategory::kTransfer));

  // Create a host buffer.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto host_buffer, device_->allocator()->Allocate(
                            MemoryType::kHostVisible | MemoryType::kHostCached |
                                MemoryType::kDeviceVisible,
                            BufferUsage::kAll, kBufferNumBytes));

  // Fill the host buffer.
  uint8_t i8_val = 0x55;
  IREE_EXPECT_OK(host_buffer->Fill8(0, kWholeBuffer, i8_val));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto host_mapped_memory,
      // Cannot use kDiscard here given we filled in the above.
      host_buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));
  IREE_EXPECT_OK(host_mapped_memory.Flush());

  std::vector<uint8_t> reference_buffer(kBufferNumBytes);
  std::memset(reference_buffer.data(), i8_val, kBufferNumBytes);

  // Create a device buffer.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      device_->allocator()->Allocate(
          MemoryType::kDeviceLocal | MemoryType::kHostVisible,
          BufferUsage::kAll, kBufferNumBytes));

  // Copy the host buffer to the device buffer.
  IREE_EXPECT_OK(command_buffer->Begin());
  IREE_EXPECT_OK(
      command_buffer->CopyBuffer(host_buffer.get(), /*source_offset=*/0,
                                 device_buffer.get(), /*target_offset=*/0,
                                 /*length=*/kBufferNumBytes));
  IREE_EXPECT_OK(command_buffer->End());

  SubmitAndWait(device_->transfer_queues()[0], command_buffer.get());

  // Read back the device buffer.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto device_mapped_memory,
      device_buffer->MapMemory<uint8_t>(MemoryAccess::kRead));
  IREE_EXPECT_OK(device_mapped_memory.Invalidate());

  std::vector<uint8_t> actual_data(
      device_mapped_memory.data(),
      device_mapped_memory.data() + kBufferNumBytes);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferTest, CopySubBuffer) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      device_->CreateCommandBuffer(CommandBufferMode::kOneShot,
                                   CommandCategory::kTransfer));
  // Create a device buffer.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto device_buffer,
      device_->allocator()->Allocate(
          MemoryType::kDeviceLocal | MemoryType::kHostVisible,
          BufferUsage::kAll, kBufferNumBytes));

  // Create another host buffer with a smaller size.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto host_buffer, device_->allocator()->Allocate(
                            MemoryType::kHostVisible | MemoryType::kHostCached |
                                MemoryType::kDeviceVisible,
                            BufferUsage::kAll, kBufferNumBytes / 2));

  // Fill the host buffer.
  uint8_t i8_val = 0x88;
  IREE_EXPECT_OK(host_buffer->Fill8(0, kWholeBuffer, i8_val));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto host_mapped_memory,
      // Cannot use kDiscard here given we filled in the above.
      host_buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));
  IREE_EXPECT_OK(host_mapped_memory.Flush());

  std::vector<uint8_t> reference_buffer(kBufferNumBytes);
  std::memset(reference_buffer.data() + 8, i8_val, kBufferNumBytes / 2 - 4);

  // Copy the host buffer to the device buffer.
  IREE_EXPECT_OK(command_buffer->Begin());
  IREE_EXPECT_OK(
      command_buffer->CopyBuffer(host_buffer.get(), /*source_offset=*/4,
                                 device_buffer.get(), /*target_offset=*/8,
                                 /*length=*/kBufferNumBytes / 2 - 4));
  IREE_EXPECT_OK(command_buffer->End());

  SubmitAndWait(device_->transfer_queues()[0], command_buffer.get());

  // Read back the device buffer.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto device_mapped_memory,
      device_buffer->MapMemory<uint8_t>(MemoryAccess::kRead));
  IREE_EXPECT_OK(device_mapped_memory.Invalidate());

  std::vector<uint8_t> actual_data(
      device_mapped_memory.data(),
      device_mapped_memory.data() + kBufferNumBytes);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

// TODO(scotttodd): UpdateBuffer, Dispatch, Sync, etc.

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, CommandBufferTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
