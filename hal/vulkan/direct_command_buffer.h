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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_HAL_VULKAN_DIRECT_COMMAND_BUFFER_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_HAL_VULKAN_DIRECT_COMMAND_BUFFER_H_

#include <vulkan/vulkan.h>

#include "third_party/mlir_edge/iree/hal/command_buffer.h"
#include "third_party/mlir_edge/iree/hal/vulkan/dynamic_symbols.h"
#include "third_party/mlir_edge/iree/hal/vulkan/handle_util.h"
#include "third_party/mlir_edge/iree/hal/vulkan/native_event.h"
#include "third_party/mlir_edge/iree/hal/vulkan/pipeline_executable.h"
#include "third_party/mlir_edge/iree/hal/vulkan/vma_buffer.h"

namespace iree {
namespace hal {
namespace vulkan {

// Command buffer implementation that directly maps to VkCommandBuffer.
// This records the commands on the calling thread without additional threading
// indirection.
class DirectCommandBuffer final : public CommandBuffer {
 public:
  DirectCommandBuffer(Allocator* allocator, CommandBufferModeBitfield mode,
                      CommandCategoryBitfield command_categories,
                      const ref_ptr<VkCommandPoolHandle>& command_pool,
                      VkCommandBuffer command_buffer);
  ~DirectCommandBuffer() override;

  VkCommandBuffer handle() const { return command_buffer_; }

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

  Status Dispatch(const DispatchRequest& dispatch_request) override;

 private:
  const ref_ptr<DynamicSymbols>& syms() const { return command_pool_->syms(); }

  StatusOr<NativeEvent*> CastEvent(Event* event) const;
  StatusOr<VmaBuffer*> CastBuffer(Buffer* buffer) const;

  Status UpdateAndBindDescriptorSet(PipelineExecutable* executable,
                                    absl::Span<const BufferBinding> bindings);

  bool is_recording_ = false;
  ref_ptr<VkCommandPoolHandle> command_pool_;
  VkCommandBuffer command_buffer_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_HAL_VULKAN_DIRECT_COMMAND_BUFFER_H_
