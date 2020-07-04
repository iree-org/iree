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

#ifndef IREE_HAL_HOST_SERIAL_SERIAL_COMMAND_PROCESSOR_H_
#define IREE_HAL_HOST_SERIAL_SERIAL_COMMAND_PROCESSOR_H_

#include "absl/container/inlined_vector.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/host/host_executable.h"

namespace iree {
namespace hal {
namespace host {

// Host-local command processor for dispatching transfer operations against
// buffers allocated from the HostLocalAllocator.
// This assumes that all buffers are host-visible (if not local) and that all
// buffers can be mapped for access.
//
// Uses HostExecutable to perform tiled dispatch processing.
//
// Thread-compatible (as with CommandBuffer itself).
class SerialCommandProcessor final : public CommandBuffer {
 public:
  explicit SerialCommandProcessor(CommandCategoryBitfield command_categories);
  ~SerialCommandProcessor() override;

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
  Status DispatchGrid(Executable* executable, int32_t entry_point,
                      std::array<uint32_t, 3> workgroup_count);

  bool is_recording_ = false;

  PushConstantBlock push_constants_;
  absl::InlinedVector<absl::InlinedVector<DescriptorSet::Binding, 8>, 2>
      descriptor_sets_;
};

}  // namespace host
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_SERIAL_SERIAL_COMMAND_PROCESSOR_H_
