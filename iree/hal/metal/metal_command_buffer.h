// Copyright 2020 Google LLC
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

#ifndef IREE_HAL_METAL_METAL_COMMAND_BUFFER_H_
#define IREE_HAL_METAL_METAL_COMMAND_BUFFER_H_

#import <Metal/Metal.h>

#include "iree/hal/command_buffer.h"

namespace iree {
namespace hal {
namespace metal {

// A command buffer implementation for Metal that directly wraps a
// MTLCommandBuffer.
//
// Objects of this class are not expected to be accessed by multiple threads.
class MetalCommandBuffer final : public CommandBuffer {
 public:
  static StatusOr<ref_ptr<CommandBuffer>> Create(
      CommandBufferModeBitfield mode,
      CommandCategoryBitfield command_categories,
      id<MTLCommandBuffer> command_buffer);
  ~MetalCommandBuffer() override;

  id<MTLCommandBuffer> handle() const { return metal_handle_; }

  bool is_recording() const override { return is_recording_; }

  Status Begin() override;
  Status End() override;

  Status ExecutionBarrier(
      ExecutionStageBitfield source_stage_mask,
      ExecutionStageBitfield target_stage_mask,
      absl::Span<const MemoryBarrier> memory_barriers,
      absl::Span<const BufferBarrier> buffer_barriers) override;

  Status SignalEvent(Event* event,
                     ExecutionStageBitfield source_stage_mask) override;
  Status ResetEvent(Event* event,
                    ExecutionStageBitfield source_stage_mask) override;
  Status WaitEvents(absl::Span<Event*> events,
                    ExecutionStageBitfield source_stage_mask,
                    ExecutionStageBitfield target_stage_mask,
                    absl::Span<const MemoryBarrier> memory_barriers,
                    absl::Span<const BufferBarrier> buffer_barriers) override;

  Status FillBuffer(Buffer* target_buffer, device_size_t target_offset,
                    device_size_t length, const void* pattern,
                    size_t pattern_length) override;
  Status DiscardBuffer(Buffer* buffer) override;
  Status UpdateBuffer(const void* source_buffer, device_size_t source_offset,
                      Buffer* target_buffer, device_size_t target_offset,
                      device_size_t length) override;
  Status CopyBuffer(Buffer* source_buffer, device_size_t source_offset,
                    Buffer* target_buffer, device_size_t target_offset,
                    device_size_t length) override;

  Status PushConstants(ExecutableLayout* executable_layout, size_t offset,
                       absl::Span<const uint32_t> values) override;

  Status PushDescriptorSet(
      ExecutableLayout* executable_layout, int32_t set,
      absl::Span<const DescriptorSet::Binding> bindings) override;
  Status BindDescriptorSet(
      ExecutableLayout* executable_layout, int32_t set,
      DescriptorSet* descriptor_set,
      absl::Span<const device_size_t> dynamic_offsets) override;

  Status Dispatch(Executable* executable, int32_t entry_point,
                  std::array<uint32_t, 3> workgroups) override;
  Status DispatchIndirect(Executable* executable, int32_t entry_point,
                          Buffer* workgroups_buffer,
                          device_size_t workgroups_offset) override;

 private:
  MetalCommandBuffer(CommandBufferModeBitfield mode,
                     CommandCategoryBitfield command_categories,
                     id<MTLCommandBuffer> command_buffer);

  bool is_recording_ = false;
  id<MTLCommandBuffer> metal_handle_;
};

}  // namespace metal
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_METAL_METAL_COMMAND_BUFFER_H_
