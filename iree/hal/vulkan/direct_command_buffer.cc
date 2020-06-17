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

#include "iree/hal/vulkan/direct_command_buffer.h"

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/math.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

VkPipelineStageFlags ConvertPipelineStageFlags(
    ExecutionStageBitfield stage_mask) {
  VkPipelineStageFlags flags = 0;
  flags |= AnyBitSet(stage_mask & ExecutionStage::kCommandIssue)
               ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
               : 0;
  flags |= AnyBitSet(stage_mask & ExecutionStage::kCommandProcess)
               ? VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT
               : 0;
  flags |= AnyBitSet(stage_mask & ExecutionStage::kDispatch)
               ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
               : 0;
  flags |= AnyBitSet(stage_mask & ExecutionStage::kTransfer)
               ? VK_PIPELINE_STAGE_TRANSFER_BIT
               : 0;
  flags |= AnyBitSet(stage_mask & ExecutionStage::kCommandRetire)
               ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
               : 0;
  flags |= AnyBitSet(stage_mask & ExecutionStage::kHost)
               ? VK_PIPELINE_STAGE_HOST_BIT
               : 0;
  return flags;
}

VkAccessFlags ConvertAccessMask(AccessScopeBitfield access_mask) {
  VkAccessFlags flags = 0;
  flags |= AnyBitSet(access_mask & AccessScope::kIndirectCommandRead)
               ? VK_ACCESS_INDIRECT_COMMAND_READ_BIT
               : 0;
  flags |= AnyBitSet(access_mask & AccessScope::kConstantRead)
               ? VK_ACCESS_UNIFORM_READ_BIT
               : 0;
  flags |= AnyBitSet(access_mask & AccessScope::kDispatchRead)
               ? VK_ACCESS_SHADER_READ_BIT
               : 0;
  flags |= AnyBitSet(access_mask & AccessScope::kDispatchWrite)
               ? VK_ACCESS_SHADER_WRITE_BIT
               : 0;
  flags |= AnyBitSet(access_mask & AccessScope::kTransferRead)
               ? VK_ACCESS_TRANSFER_READ_BIT
               : 0;
  flags |= AnyBitSet(access_mask & AccessScope::kTransferWrite)
               ? VK_ACCESS_TRANSFER_WRITE_BIT
               : 0;
  flags |= AnyBitSet(access_mask & AccessScope::kHostRead)
               ? VK_ACCESS_HOST_READ_BIT
               : 0;
  flags |= AnyBitSet(access_mask & AccessScope::kHostWrite)
               ? VK_ACCESS_HOST_WRITE_BIT
               : 0;
  flags |= AnyBitSet(access_mask & AccessScope::kMemoryRead)
               ? VK_ACCESS_MEMORY_READ_BIT
               : 0;
  flags |= AnyBitSet(access_mask & AccessScope::kMemoryWrite)
               ? VK_ACCESS_MEMORY_WRITE_BIT
               : 0;
  return flags;
}

// Splats a pattern value of 1, 2, or 4 bytes out to a 4 byte value.
uint32_t SplatPattern(const void* pattern, size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *static_cast<const uint8_t*>(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *static_cast<const uint16_t*>(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *static_cast<const uint32_t*>(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

}  // namespace

DirectCommandBuffer::DirectCommandBuffer(
    CommandBufferModeBitfield mode, CommandCategoryBitfield command_categories,
    ref_ptr<DescriptorPoolCache> descriptor_pool_cache,
    ref_ptr<VkCommandPoolHandle> command_pool, VkCommandBuffer command_buffer)
    : CommandBuffer(mode, command_categories),
      command_pool_(std::move(command_pool)),
      command_buffer_(command_buffer),
      descriptor_set_arena_(std::move(descriptor_pool_cache)) {}

DirectCommandBuffer::~DirectCommandBuffer() {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::dtor");
  descriptor_set_group_.Reset().IgnoreError();
  absl::MutexLock lock(command_pool_->mutex());
  syms()->vkFreeCommandBuffers(*command_pool_->logical_device(), *command_pool_,
                               1, &command_buffer_);
}

StatusOr<NativeEvent*> DirectCommandBuffer::CastEvent(Event* event) const {
  // TODO(benvanik): assert the event is valid.
  return static_cast<NativeEvent*>(event);
}

StatusOr<VmaBuffer*> DirectCommandBuffer::CastBuffer(Buffer* buffer) const {
  // TODO(benvanik): assert that the buffer is from the right allocator and
  // that it is compatible with our target queue family.
  return static_cast<VmaBuffer*>(buffer->allocated_buffer());
}

StatusOr<NativeDescriptorSet*> DirectCommandBuffer::CastDescriptorSet(
    DescriptorSet* descriptor_set) const {
  // TODO(benvanik): assert the descriptor_set is valid.
  return static_cast<NativeDescriptorSet*>(descriptor_set);
}

StatusOr<PipelineExecutableLayout*> DirectCommandBuffer::CastExecutableLayout(
    ExecutableLayout* executable_layout) const {
  // TODO(benvanik): assert the executable_layout is valid.
  return static_cast<PipelineExecutableLayout*>(executable_layout);
}

StatusOr<PipelineExecutable*> DirectCommandBuffer::CastExecutable(
    Executable* executable) const {
  // TODO(benvanik): assert the executable is valid.
  return static_cast<PipelineExecutable*>(executable);
}

Status DirectCommandBuffer::Begin() {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::Begin");

  is_recording_ = true;

  // NOTE: we require that command buffers not be recorded while they are
  // in-flight so this is safe.
  RETURN_IF_ERROR(descriptor_set_group_.Reset());

  VkCommandBufferBeginInfo begin_info;
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = nullptr;
  begin_info.flags = AllBitsSet(mode(), CommandBufferMode::kOneShot)
                         ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                         : 0;
  begin_info.pInheritanceInfo = nullptr;
  VK_RETURN_IF_ERROR(
      syms()->vkBeginCommandBuffer(command_buffer_, &begin_info));

  return OkStatus();
}

Status DirectCommandBuffer::End() {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::End");

  VK_RETURN_IF_ERROR(syms()->vkEndCommandBuffer(command_buffer_));

  // Flush all pending descriptor set writes (if any).
  ASSIGN_OR_RETURN(descriptor_set_group_, descriptor_set_arena_.Flush());

  is_recording_ = false;

  return OkStatus();
}

Status DirectCommandBuffer::ExecutionBarrier(
    ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::ExecutionBarrier");

  absl::InlinedVector<VkMemoryBarrier, 8> memory_barrier_infos(
      memory_barriers.size());
  for (int i = 0; i < memory_barriers.size(); ++i) {
    const auto& memory_barrier = memory_barriers[i];
    auto& info = memory_barrier_infos[i];
    info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    info.pNext = nullptr;
    info.srcAccessMask = ConvertAccessMask(memory_barrier.source_scope);
    info.dstAccessMask = ConvertAccessMask(memory_barrier.target_scope);
  }

  absl::InlinedVector<VkBufferMemoryBarrier, 8> buffer_barrier_infos(
      buffer_barriers.size());
  for (int i = 0; i < buffer_barriers.size(); ++i) {
    const auto& buffer_barrier = buffer_barriers[i];
    auto& info = buffer_barrier_infos[i];
    info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info.pNext = nullptr;
    info.srcAccessMask = ConvertAccessMask(buffer_barrier.source_scope);
    info.dstAccessMask = ConvertAccessMask(buffer_barrier.target_scope);
    info.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ASSIGN_OR_RETURN(auto* device_buffer, CastBuffer(buffer_barrier.buffer));
    info.buffer = device_buffer->handle();
    info.offset = buffer_barrier.offset;
    info.size = buffer_barrier.length;
  }

  syms()->vkCmdPipelineBarrier(
      command_buffer_, ConvertPipelineStageFlags(source_stage_mask),
      ConvertPipelineStageFlags(target_stage_mask), /*dependencyFlags=*/0,
      memory_barrier_infos.size(), memory_barrier_infos.data(),
      buffer_barrier_infos.size(), buffer_barrier_infos.data(), 0, nullptr);

  return OkStatus();
}

Status DirectCommandBuffer::SignalEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::SignalEvent");
  ASSIGN_OR_RETURN(auto* device_event, CastEvent(event));
  syms()->vkCmdSetEvent(command_buffer_, device_event->handle(),
                        ConvertPipelineStageFlags(source_stage_mask));
  return OkStatus();
}

Status DirectCommandBuffer::ResetEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::ResetEvent");
  ASSIGN_OR_RETURN(auto* device_event, CastEvent(event));
  syms()->vkCmdResetEvent(command_buffer_, device_event->handle(),
                          ConvertPipelineStageFlags(source_stage_mask));
  return OkStatus();
}

Status DirectCommandBuffer::WaitEvents(
    absl::Span<Event*> events, ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::WaitEvents");

  absl::InlinedVector<VkEvent, 4> event_handles(events.size());
  for (int i = 0; i < events.size(); ++i) {
    ASSIGN_OR_RETURN(auto* device_event, CastEvent(events[i]));
    event_handles[i] = device_event->handle();
  }

  absl::InlinedVector<VkMemoryBarrier, 8> memory_barrier_infos(
      memory_barriers.size());
  for (int i = 0; i < memory_barriers.size(); ++i) {
    const auto& memory_barrier = memory_barriers[i];
    auto& info = memory_barrier_infos[i];
    info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    info.pNext = nullptr;
    info.srcAccessMask = ConvertAccessMask(memory_barrier.source_scope);
    info.dstAccessMask = ConvertAccessMask(memory_barrier.target_scope);
  }

  absl::InlinedVector<VkBufferMemoryBarrier, 8> buffer_barrier_infos(
      buffer_barriers.size());
  for (int i = 0; i < buffer_barriers.size(); ++i) {
    const auto& buffer_barrier = buffer_barriers[i];
    auto& info = buffer_barrier_infos[i];
    info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info.pNext = nullptr;
    info.srcAccessMask = ConvertAccessMask(buffer_barrier.source_scope);
    info.dstAccessMask = ConvertAccessMask(buffer_barrier.target_scope);
    info.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ASSIGN_OR_RETURN(auto* device_buffer, CastBuffer(buffer_barrier.buffer));
    info.buffer = device_buffer->handle();
    info.offset = buffer_barrier.offset;
    info.size = buffer_barrier.length;
  }

  syms()->vkCmdWaitEvents(
      command_buffer_, event_handles.size(), event_handles.data(),
      ConvertPipelineStageFlags(source_stage_mask),
      ConvertPipelineStageFlags(target_stage_mask), memory_barrier_infos.size(),
      memory_barrier_infos.data(), buffer_barrier_infos.size(),
      buffer_barrier_infos.data(), 0, nullptr);
  return OkStatus();
}

Status DirectCommandBuffer::FillBuffer(Buffer* target_buffer,
                                       device_size_t target_offset,
                                       device_size_t length,
                                       const void* pattern,
                                       size_t pattern_length) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::FillBuffer");
  ASSIGN_OR_RETURN(auto* target_device_buffer, CastBuffer(target_buffer));

  // Note that fill only accepts 4-byte aligned values so we need to splat out
  // our variable-length pattern.
  target_offset += target_buffer->byte_offset();
  uint32_t dword_pattern = SplatPattern(pattern, pattern_length);
  syms()->vkCmdFillBuffer(command_buffer_, target_device_buffer->handle(),
                          target_offset, length, dword_pattern);

  return OkStatus();
}

Status DirectCommandBuffer::DiscardBuffer(Buffer* buffer) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::DiscardBuffer");
  // NOTE: we could use this to prevent queue family transitions.
  return OkStatus();
}

Status DirectCommandBuffer::UpdateBuffer(const void* source_buffer,
                                         device_size_t source_offset,
                                         Buffer* target_buffer,
                                         device_size_t target_offset,
                                         device_size_t length) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::UpdateBuffer");
  ASSIGN_OR_RETURN(auto* target_device_buffer, CastBuffer(target_buffer));

  // Vulkan only allows updates of <= 65536 because you really, really, really
  // shouldn't do large updates like this (as it wastes command buffer space and
  // may be slower than just using write-through mapped memory). The
  // recommendation in the spec for larger updates is to split the single update
  // into multiple updates over the entire desired range.
  const auto* source_buffer_ptr = static_cast<const uint8_t*>(source_buffer);
  target_offset += target_buffer->byte_offset();
  while (length > 0) {
    device_size_t chunk_length =
        std::min(static_cast<device_size_t>(65536u), length);
    syms()->vkCmdUpdateBuffer(command_buffer_, target_device_buffer->handle(),
                              target_offset, chunk_length, source_buffer_ptr);
    source_buffer_ptr += chunk_length;
    target_offset += chunk_length;
    length -= chunk_length;
  }

  return OkStatus();
}

Status DirectCommandBuffer::CopyBuffer(Buffer* source_buffer,
                                       device_size_t source_offset,
                                       Buffer* target_buffer,
                                       device_size_t target_offset,
                                       device_size_t length) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::CopyBuffer");
  ASSIGN_OR_RETURN(auto* source_device_buffer, CastBuffer(source_buffer));
  ASSIGN_OR_RETURN(auto* target_device_buffer, CastBuffer(target_buffer));

  VkBufferCopy region;
  region.srcOffset = source_buffer->byte_offset() + source_offset;
  region.dstOffset = target_buffer->byte_offset() + target_offset;
  region.size = length;
  syms()->vkCmdCopyBuffer(command_buffer_, source_device_buffer->handle(),
                          target_device_buffer->handle(), 1, &region);

  return OkStatus();
}

Status DirectCommandBuffer::PushConstants(ExecutableLayout* executable_layout,
                                          size_t offset,
                                          absl::Span<const uint32_t> values) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::PushConstants");
  ASSIGN_OR_RETURN(auto* device_executable_layout,
                   CastExecutableLayout(executable_layout));

  syms()->vkCmdPushConstants(
      command_buffer_, device_executable_layout->handle(),
      VK_SHADER_STAGE_COMPUTE_BIT, offset * sizeof(uint32_t),
      values.size() * sizeof(uint32_t), values.data());

  return OkStatus();
}

Status DirectCommandBuffer::PushDescriptorSet(
    ExecutableLayout* executable_layout, int32_t set,
    absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::PushDescriptorSet");
  ASSIGN_OR_RETURN(auto* device_executable_layout,
                   CastExecutableLayout(executable_layout));

  // Either allocate, update, and bind a descriptor set or use push descriptor
  // sets to use the command buffer pool when supported.
  return descriptor_set_arena_.BindDescriptorSet(
      command_buffer_, device_executable_layout, set, bindings);
}

Status DirectCommandBuffer::BindDescriptorSet(
    ExecutableLayout* executable_layout, int32_t set,
    DescriptorSet* descriptor_set,
    absl::Span<const device_size_t> dynamic_offsets) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::BindDescriptorSet");
  ASSIGN_OR_RETURN(auto* device_executable_layout,
                   CastExecutableLayout(executable_layout));
  ASSIGN_OR_RETURN(auto* device_descriptor_set,
                   CastDescriptorSet(descriptor_set));

  // Vulkan takes uint32_t as the size here, unlike everywhere else.
  absl::InlinedVector<uint32_t, 4> dynamic_offsets_i32(dynamic_offsets.size());
  for (int i = 0; i < dynamic_offsets.size(); ++i) {
    dynamic_offsets_i32[i] = static_cast<uint32_t>(dynamic_offsets[i]);
  }

  std::array<VkDescriptorSet, 1> descriptor_sets = {
      device_descriptor_set->handle()};
  syms()->vkCmdBindDescriptorSets(
      command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
      device_executable_layout->handle(), set, descriptor_sets.size(),
      descriptor_sets.data(), dynamic_offsets_i32.size(),
      dynamic_offsets_i32.data());

  return OkStatus();
}

Status DirectCommandBuffer::Dispatch(Executable* executable,
                                     int32_t entry_point,
                                     std::array<uint32_t, 3> workgroups) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::Dispatch");

  // Get the compiled and linked pipeline for the specified entry point and
  // bind it to the command buffer.
  ASSIGN_OR_RETURN(auto* device_executable, CastExecutable(executable));
  ASSIGN_OR_RETURN(VkPipeline pipeline,
                   device_executable->GetPipelineForEntryPoint(entry_point));
  syms()->vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline);

  syms()->vkCmdDispatch(command_buffer_, workgroups[0], workgroups[1],
                        workgroups[2]);
  return OkStatus();
}

Status DirectCommandBuffer::DispatchIndirect(Executable* executable,
                                             int32_t entry_point,
                                             Buffer* workgroups_buffer,
                                             device_size_t workgroups_offset) {
  IREE_TRACE_SCOPE0("DirectCommandBuffer::DispatchIndirect");

  // Get the compiled and linked pipeline for the specified entry point and
  // bind it to the command buffer.
  ASSIGN_OR_RETURN(auto* device_executable, CastExecutable(executable));
  ASSIGN_OR_RETURN(VkPipeline pipeline,
                   device_executable->GetPipelineForEntryPoint(entry_point));
  syms()->vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline);

  ASSIGN_OR_RETURN(auto* workgroups_device_buffer,
                   CastBuffer(workgroups_buffer));
  syms()->vkCmdDispatchIndirect(
      command_buffer_, workgroups_device_buffer->handle(), workgroups_offset);
  return OkStatus();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
