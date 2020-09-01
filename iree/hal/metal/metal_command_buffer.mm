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

#include "iree/hal/metal/metal_command_buffer.h"

#include "iree/base/status.h"
#include "iree/base/tracing.h"

namespace iree {
namespace hal {
namespace metal {

// static
StatusOr<ref_ptr<CommandBuffer>> MetalCommandBuffer::Create(
    CommandBufferModeBitfield mode, CommandCategoryBitfield command_categories,
    id<MTLCommandBuffer> command_buffer) {
  return assign_ref(new MetalCommandBuffer(mode, command_categories, command_buffer));
}

MetalCommandBuffer::MetalCommandBuffer(CommandBufferModeBitfield mode,
                                       CommandCategoryBitfield command_categories,
                                       id<MTLCommandBuffer> command_buffer)
    : CommandBuffer(mode, command_categories), metal_handle_([command_buffer retain]) {}

MetalCommandBuffer::~MetalCommandBuffer() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::dtor");
  [metal_handle_ release];
}

Status MetalCommandBuffer::Begin() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::Begin");
  is_recording_ = true;
  return OkStatus();
}

Status MetalCommandBuffer::End() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::End");
  is_recording_ = false;
  return OkStatus();
}

Status MetalCommandBuffer::ExecutionBarrier(ExecutionStageBitfield source_stage_mask,
                                            ExecutionStageBitfield target_stage_mask,
                                            absl::Span<const MemoryBarrier> memory_barriers,
                                            absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::ExecutionBarrier");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::ExecutionBarrier";
}

Status MetalCommandBuffer::SignalEvent(Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::SignalEvent");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::SignalEvent";
}

Status MetalCommandBuffer::ResetEvent(Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::ResetEvent");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::ResetEvent";
}

Status MetalCommandBuffer::WaitEvents(absl::Span<Event*> events,
                                      ExecutionStageBitfield source_stage_mask,
                                      ExecutionStageBitfield target_stage_mask,
                                      absl::Span<const MemoryBarrier> memory_barriers,
                                      absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::WaitEvents");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::WaitEvents";
}

Status MetalCommandBuffer::FillBuffer(Buffer* target_buffer, device_size_t target_offset,
                                      device_size_t length, const void* pattern,
                                      size_t pattern_length) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::FillBuffer");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::FillBuffer";
}

Status MetalCommandBuffer::DiscardBuffer(Buffer* buffer) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::DiscardBuffer");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::DiscardBuffer";
}

Status MetalCommandBuffer::UpdateBuffer(const void* source_buffer, device_size_t source_offset,
                                        Buffer* target_buffer, device_size_t target_offset,
                                        device_size_t length) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::UpdateBuffer");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::UpdateBuffer";
}

Status MetalCommandBuffer::CopyBuffer(Buffer* source_buffer, device_size_t source_offset,
                                      Buffer* target_buffer, device_size_t target_offset,
                                      device_size_t length) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::CopyBuffer");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::CopyBuffer";
}

Status MetalCommandBuffer::PushConstants(ExecutableLayout* executable_layout, size_t offset,
                                         absl::Span<const uint32_t> values) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::PushConstants");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::PushConstants";
}

Status MetalCommandBuffer::PushDescriptorSet(ExecutableLayout* executable_layout, int32_t set,
                                             absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::PushDescriptorSet");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::PushDescriptorSet";
}

Status MetalCommandBuffer::BindDescriptorSet(ExecutableLayout* executable_layout, int32_t set,
                                             DescriptorSet* descriptor_set,
                                             absl::Span<const device_size_t> dynamic_offsets) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::BindDescriptorSet");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::BindDescriptorSet";
}

Status MetalCommandBuffer::Dispatch(Executable* executable, int32_t entry_point,
                                    std::array<uint32_t, 3> workgroups) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::Dispatch");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::Dispatch";
}

Status MetalCommandBuffer::DispatchIndirect(Executable* executable, int32_t entry_point,
                                            Buffer* workgroups_buffer,
                                            device_size_t workgroups_offset) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::DispatchIndirect");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::DispatchIndirect";
}

}  // namespace metal
}  // namespace hal
}  // namespace iree
