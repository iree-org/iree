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

#include "gmock/gmock.h"
#include "iree/hal/command_buffer.h"

namespace iree {
namespace hal {
namespace testing {

class MockCommandBuffer : public ::testing::StrictMock<CommandBuffer> {
 public:
  MockCommandBuffer(Allocator* allocator, CommandBufferModeBitfield mode,
                    CommandCategoryBitfield command_categories)
      : ::testing::StrictMock<CommandBuffer>(allocator, mode,
                                             command_categories) {}

  bool is_recording() const override { return false; }

  MOCK_METHOD0(Begin, Status());
  MOCK_METHOD0(End, Status());

  MOCK_METHOD4(ExecutionBarrier,
               Status(ExecutionStageBitfield source_stage_mask,
                      ExecutionStageBitfield target_stage_mask,
                      absl::Span<const MemoryBarrier> memory_barriers,
                      absl::Span<const BufferBarrier> buffer_barriers));

  MOCK_METHOD2(SignalEvent,
               Status(Event* event, ExecutionStageBitfield source_stage_mask));

  MOCK_METHOD2(ResetEvent,
               Status(Event* event, ExecutionStageBitfield source_stage_mask));

  MOCK_METHOD5(WaitEvents,
               Status(absl::Span<Event*> events,
                      ExecutionStageBitfield source_stage_mask,
                      ExecutionStageBitfield target_stage_mask,
                      absl::Span<const MemoryBarrier> memory_barriers,
                      absl::Span<const BufferBarrier> buffer_barriers));

  MOCK_METHOD5(FillBuffer,
               Status(Buffer* target_buffer, device_size_t target_offset,
                      device_size_t length, const void* pattern,
                      size_t pattern_length));

  MOCK_METHOD1(DiscardBuffer, Status(Buffer* buffer));

  MOCK_METHOD5(UpdateBuffer,
               Status(const void* source_buffer, device_size_t source_offset,
                      Buffer* target_buffer, device_size_t target_offset,
                      device_size_t length));

  MOCK_METHOD5(CopyBuffer,
               Status(Buffer* source_buffer, device_size_t source_offset,
                      Buffer* target_buffer, device_size_t target_offset,
                      device_size_t length));

  MOCK_METHOD1(Dispatch, Status(const DispatchRequest& dispatch_request));
};

}  // namespace testing
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_TESTING_MOCK_COMMAND_BUFFER_H_
