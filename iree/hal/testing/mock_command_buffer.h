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

#ifndef IREE_HAL_TESTING_MOCK_COMMAND_BUFFER_H_
#define IREE_HAL_TESTING_MOCK_COMMAND_BUFFER_H_

#include "iree/hal/command_buffer.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace testing {

class MockCommandBuffer : public ::testing::StrictMock<CommandBuffer> {
 public:
  MockCommandBuffer(CommandBufferModeBitfield mode,
                    CommandCategoryBitfield command_categories)
      : ::testing::StrictMock<CommandBuffer>(mode, command_categories) {}

  bool is_recording() const override { return false; }

  MOCK_METHOD(Status, Begin, (), (override));
  MOCK_METHOD(Status, End, (), (override));

  MOCK_METHOD(Status, ExecutionBarrier,
              (ExecutionStageBitfield source_stage_mask,
               ExecutionStageBitfield target_stage_mask,
               absl::Span<const MemoryBarrier> memory_barriers,
               absl::Span<const BufferBarrier> buffer_barriers),
              (override));

  MOCK_METHOD(Status, SignalEvent,
              (Event * event, ExecutionStageBitfield source_stage_mask),
              (override));

  MOCK_METHOD(Status, ResetEvent,
              (Event * event, ExecutionStageBitfield source_stage_mask),
              (override));

  MOCK_METHOD(Status, WaitEvents,
              (absl::Span<Event*> events,
               ExecutionStageBitfield source_stage_mask,
               ExecutionStageBitfield target_stage_mask,
               absl::Span<const MemoryBarrier> memory_barriers,
               absl::Span<const BufferBarrier> buffer_barriers),
              (override));

  MOCK_METHOD(Status, FillBuffer,
              (Buffer * target_buffer, device_size_t target_offset,
               device_size_t length, const void* pattern,
               size_t pattern_length),
              (override));

  MOCK_METHOD(Status, DiscardBuffer, (Buffer * buffer), (override));

  MOCK_METHOD(Status, UpdateBuffer,
              (const void* source_buffer, device_size_t source_offset,
               Buffer* target_buffer, device_size_t target_offset,
               device_size_t length),
              (override));

  MOCK_METHOD(Status, CopyBuffer,
              (Buffer * source_buffer, device_size_t source_offset,
               Buffer* target_buffer, device_size_t target_offset,
               device_size_t length),
              (override));

  MOCK_METHOD(Status, PushConstants,
              (ExecutableLayout * executable_layout, size_t offset,
               absl::Span<const uint32_t> values),
              (override));

  MOCK_METHOD(Status, PushDescriptorSet,
              (ExecutableLayout * executable_layout, int32_t set,
               absl::Span<const DescriptorSet::Binding> bindings),
              (override));
  MOCK_METHOD(Status, BindDescriptorSet,
              (ExecutableLayout * executable_layout, int32_t set,
               DescriptorSet* descriptor_set,
               absl::Span<const device_size_t> dynamic_offsets),
              (override));

  MOCK_METHOD(Status, Dispatch,
              (Executable * executable, int32_t entry_point,
               (std::array<uint32_t, 3> workgroups)),
              (override));

  MOCK_METHOD(Status, DispatchIndirect,
              (Executable * executable, int32_t entry_point,
               Buffer* workgroups_buffer, device_size_t workgroups_offset),
              (override));
};

}  // namespace testing
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_TESTING_MOCK_COMMAND_BUFFER_H_
