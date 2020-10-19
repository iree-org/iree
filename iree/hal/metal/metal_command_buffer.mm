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

#include "iree/base/logging.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/metal/metal_kernel_library.h"
#include "iree/hal/metal/metal_pipeline_argument_buffer.h"

namespace iree {
namespace hal {
namespace metal {

namespace {

MTLResourceUsage ConvertResourceUsage(MemoryAccessBitfield memory_access) {
  MTLResourceUsage usage = 0;
  if (AllBitsSet(memory_access, MemoryAccess::kRead)) usage |= MTLResourceUsageRead;
  if (AllBitsSet(memory_access, MemoryAccess::kWrite)) usage |= MTLResourceUsageWrite;
  return usage;
}

}  // namespace

// static
StatusOr<ref_ptr<CommandBuffer>> MetalCommandBuffer::Create(
    CommandBufferModeBitfield mode, CommandCategoryBitfield command_categories,
    id<MTLCommandBuffer> command_buffer) {
  return assign_ref(new MetalCommandBuffer(mode, command_categories, command_buffer));
}

MetalCommandBuffer::MetalCommandBuffer(CommandBufferModeBitfield mode,
                                       CommandCategoryBitfield command_categories,
                                       id<MTLCommandBuffer> command_buffer)
    : CommandBuffer(mode, command_categories), metal_handle_([command_buffer retain]) {
  metal_handle_.label = @"IREE MetalCommandBuffer";
}

MetalCommandBuffer::~MetalCommandBuffer() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::dtor");
  [metal_handle_ release];
}

StatusOr<MetalBuffer*> MetalCommandBuffer::CastBuffer(Buffer* buffer) const {
  // TODO(benvanik): assert that the buffer is from the right allocator and
  // that it is compatible with our target queue family.
  return static_cast<MetalBuffer*>(buffer->allocated_buffer());
}

id<MTLBlitCommandEncoder> MetalCommandBuffer::GetOrBeginBlitEncoder() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::GetOrBeginBlitEncoder");

  if (current_compute_encoder_) EndComputeEncoder();

  @autoreleasepool {
    if (!current_blit_encoder_) {
      current_blit_encoder_ = [[metal_handle_ blitCommandEncoder] retain];
    }
  }

  return current_blit_encoder_;
}

void MetalCommandBuffer::EndBlitEncoder() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::EndBlitEncoder");
  if (current_blit_encoder_) {
    [current_blit_encoder_ endEncoding];
    [current_blit_encoder_ release];
    current_blit_encoder_ = nil;
  }
}

id<MTLComputeCommandEncoder> MetalCommandBuffer::GetOrBeginComputeEncoder() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::GetOrBeginComputeEncoder");

  if (current_blit_encoder_) EndBlitEncoder();

  @autoreleasepool {
    if (!current_compute_encoder_) {
      current_compute_encoder_ = [[metal_handle_ computeCommandEncoder] retain];
    }
  }

  return current_compute_encoder_;
}

void MetalCommandBuffer::EndComputeEncoder() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::EndComputeEncoder");
  if (current_compute_encoder_) {
    [current_compute_encoder_ endEncoding];
    [current_compute_encoder_ release];
    current_compute_encoder_ = nil;
  }
}

Status MetalCommandBuffer::Begin() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::Begin");
  is_recording_ = true;
  return OkStatus();
}

Status MetalCommandBuffer::End() {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::End");
  EndBlitEncoder();
  EndComputeEncoder();
  is_recording_ = false;
  return OkStatus();
}

Status MetalCommandBuffer::ExecutionBarrier(ExecutionStageBitfield source_stage_mask,
                                            ExecutionStageBitfield target_stage_mask,
                                            absl::Span<const MemoryBarrier> memory_barriers,
                                            absl::Span<const BufferBarrier> buffer_barriers) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::ExecutionBarrier");

  if (AllBitsSet(source_stage_mask, ExecutionStage::kHost) ||
      AllBitsSet(target_stage_mask, ExecutionStage::kHost)) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "MetalCommandBuffer::ExecutionBarrier with host bit set";
  }

  // If there is a memory barrier specified, we have to place a catch-all barrier for all buffers.
  // Metal does not provide a more fine-grained control here. But we do have the option to specify a
  // list of buffers to synchronize if only buffer barriers are specified.
  if (!memory_barriers.empty()) {
    [GetOrBeginComputeEncoder() memoryBarrierWithScope:MTLBarrierScopeBuffers];
  } else if (!buffer_barriers.empty()) {
    std::vector<id<MTLResource>> buffers;
    buffers.reserve(buffer_barriers.size());
    for (const auto& barrier : buffer_barriers) {
      buffers.push_back(static_cast<MetalBuffer*>(barrier.buffer)->handle());
    }
    [GetOrBeginComputeEncoder() memoryBarrierWithResources:buffers.data() count:buffers.size()];
  }

  return OkStatus();
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
  IREE_ASSIGN_OR_RETURN(auto* target_device_buffer, CastBuffer(target_buffer));

  target_offset += target_buffer->byte_offset();

  // Per the spec for fillBuffer:range:value: "The alignment and length of the range must both be a
  // multiple of 4 bytes in macOS, and 1 byte in iOS and tvOS." Although iOS/tvOS is more relaxed on
  // this front, we still require 4-byte alignment for uniformity across IREE.
  if (target_offset % 4 != 0) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "MetalCommandBuffer::FillBuffer with offset that is not a multiple of 4 bytes";
  }

  // Note that fillBuffer:range:value: only accepts a single byte as the pattern but FillBuffer
  // can accept 1/2/4 bytes. If the pattern itself contains repeated bytes, we can call into
  // fillBuffer:range:value:. Otherwise we may need to find another way. Just implement the case
  // where we have a single byte to fill for now.
  if (pattern_length != 1) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "MetalCommandBuffer::FillBuffer with non-1-byte pattern";
  }
  uint8_t byte_pattern = *reinterpret_cast<const uint8_t*>(pattern);

  [GetOrBeginBlitEncoder() fillBuffer:target_device_buffer->handle()
                                range:NSMakeRange(target_offset, length)
                                value:byte_pattern];

  return OkStatus();
}

Status MetalCommandBuffer::DiscardBuffer(Buffer* buffer) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::DiscardBuffer");
  // This is a hint. Nothing to do for Metal.
  return OkStatus();
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

  IREE_ASSIGN_OR_RETURN(auto* source_device_buffer, CastBuffer(source_buffer));
  IREE_ASSIGN_OR_RETURN(auto* target_device_buffer, CastBuffer(target_buffer));

  source_offset += source_buffer->byte_offset();
  target_offset += target_buffer->byte_offset();

  // Per the spec for copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size, the source/target
  // offset must be a multiple of 4 bytes in macOS, and 1 byte in iOS and tvOS. Although iOS/tvOS
  // is more relaxed on this front, we still require 4-byte alignment for uniformity across IREE.
  if (source_offset % 4 != 0 || target_offset % 4 != 0) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "MetalCommandBuffer::CopyBuffer with offset that is not a multiple of 4 bytes";
  }

  [GetOrBeginBlitEncoder() copyFromBuffer:source_device_buffer->handle()
                             sourceOffset:source_offset
                                 toBuffer:target_device_buffer->handle()
                        destinationOffset:target_offset
                                     size:length];

  return OkStatus();
}

Status MetalCommandBuffer::PushConstants(ExecutableLayout* executable_layout, size_t offset,
                                         absl::Span<const uint32_t> values) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::PushConstants");
  return UnimplementedErrorBuilder(IREE_LOC) << "MetalCommandBuffer::PushConstants";
}

Status MetalCommandBuffer::PushDescriptorSet(ExecutableLayout* executable_layout, int32_t set,
                                             absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::PushDescriptorSet");
  if (set != 0) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "MetalCommandBuffer::PushDescriptorSet with set number > 0";
  }
  auto& push_state = pipeline_state_objects_[executable_layout].push_states[set];
  push_state.resource_bindings.assign(bindings.begin(), bindings.end());
  return OkStatus();
}

Status MetalCommandBuffer::BindDescriptorSet(ExecutableLayout* executable_layout, int32_t set,
                                             DescriptorSet* descriptor_set,
                                             absl::Span<const device_size_t> dynamic_offsets) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::BindDescriptorSet");
  if (set != 0) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "MetalCommandBuffer::BindDescriptorSet with set number > 0";
  }
  if (!dynamic_offsets.empty()) {
    return UnimplementedErrorBuilder(IREE_LOC)
           << "MetalCommandBuffer::BindDescriptorSet with dynamic offsets";
  }
  pipeline_state_objects_[executable_layout].bind_states[set].descriptor_set = descriptor_set;
  return OkStatus();
}

Status MetalCommandBuffer::Dispatch(Executable* executable, int32_t entry_point,
                                    std::array<uint32_t, 3> workgroups) {
  IREE_TRACE_SCOPE0("MetalCommandBuffer::Dispatch");
  IREE_DVLOG(2) << "MetalCommandBuffer::Dispatch";

  auto* kernel_library = static_cast<MetalKernelLibrary*>(executable);
  IREE_ASSIGN_OR_RETURN(auto metal_kernel, kernel_library->GetKernelForEntryPoint(entry_point));
  IREE_ASSIGN_OR_RETURN(auto metal_pso, kernel_library->GetPipelineStateForEntryPoint(entry_point));
  IREE_ASSIGN_OR_RETURN(auto workgroup_size,
                        kernel_library->GetThreadgroupSizeForEntryPoint(entry_point));

  id<MTLComputeCommandEncoder> compute_encoder = GetOrBeginComputeEncoder();
  [compute_encoder setComputePipelineState:metal_pso];

  // TODO(antiagainst): only update the PSO for the current executable.
  for (const auto& pso_kv : pipeline_state_objects_) {
    const auto* pipeline_layout = static_cast<MetalPipelineArgumentBufferLayout*>(pso_kv.first);
    IREE_DVLOG(3) << "Current pipeline layout: " << pipeline_layout->DebugString();

    const auto& pso = pso_kv.second;
    if (pso.push_states.size() > 1) {
      return UnimplementedErrorBuilder(IREE_LOC)
             << "MetalCommandBuffer::Dispatch with more than one push descriptor sets";
    }
    if (!pso.bind_states.empty()) {
      return UnimplementedErrorBuilder(IREE_LOC)
             << "MetalCommandBuffer::Dispatch with bound descriptor sets";
    }
    if (!pso.constant_states.empty()) {
      return UnimplementedErrorBuilder(IREE_LOC)
             << "MetalCommandBuffer::Dispatch with push constants";
    }

    IREE_DVLOG(3) << "Encoding push descriptors..";
    for (const auto& push_kv : pso.push_states) {
      int32_t set_number = push_kv.first;
      const PipelineStateObject::PushState& push_state = push_kv.second;
      IREE_DVLOG(3) << " For set #" << set_number;

      id<MTLArgumentEncoder> argument_encoder =
          [metal_kernel newArgumentEncoderWithBufferIndex:set_number];  // retained
      argument_encoder.label = @"IREE MetalCommandBuffer::Dispatch ArgumentEncoder";
      if (!argument_encoder) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Buffer index #" << set_number << " is not an argument buffer";
      }

      __block id<MTLBuffer> argument_buffer =
          [metal_handle_.device newBufferWithLength:argument_encoder.encodedLength
                                            options:MTLResourceStorageModeShared];  // retained
      argument_encoder.label = @"IREE MetalCommandBuffer::Dispatch ArgumentBuffer";
      if (!argument_buffer) {
        return InternalErrorBuilder(IREE_LOC)
               << "Failed to create argument buffer with length=" << argument_encoder.encodedLength;
      }
      [metal_handle_ addCompletedHandler:^(id<MTLCommandBuffer>) {
        [argument_buffer release];
        [argument_encoder release];
      }];

      [argument_encoder setArgumentBuffer:argument_buffer offset:0];

      for (const auto& resource_binding : push_state.resource_bindings) {
        IREE_DVLOG(3) << "  Resource @[" << resource_binding.DebugStringShort() << "]";

        if (resource_binding.length != kWholeBuffer &&
            resource_binding.length != resource_binding.buffer->allocation_size()) {
          return UnimplementedErrorBuilder(IREE_LOC)
                 << "MetalCommandBuffer::Dispatch with sub-buffer";
        }

        IREE_ASSIGN_OR_RETURN(auto buffer, CastBuffer(resource_binding.buffer));
        [argument_encoder setBuffer:buffer->handle()
                             offset:resource_binding.offset
                            atIndex:resource_binding.binding];

        const auto* set_layout = pipeline_layout->set_layouts()[set_number];
        const auto* layout_binding = set_layout->GetBindingForIndex(resource_binding.binding);
        if (!layout_binding) {
          return InvalidArgumentErrorBuilder(IREE_LOC)
                 << "Cannot find binding #" << resource_binding.binding
                 << " in argument buffer layout";
        }
        [compute_encoder useResource:buffer->handle()
                               usage:ConvertResourceUsage(layout_binding->access)];
      }

      [compute_encoder setBuffer:argument_buffer offset:0 atIndex:set_number];
    }
  }

  IREE_DVLOG(2) << "Dispatch workgroup count: (" << workgroups[0] << ", " << workgroups[1] << ", "
                << workgroups[2] << "), workgroup size: (" << workgroup_size.x() << ", "
                << workgroup_size.y() << ", " << workgroup_size.z() << ")";
  [compute_encoder dispatchThreadgroups:MTLSizeMake(workgroups[0], workgroups[1], workgroups[2])
                  threadsPerThreadgroup:MTLSizeMake(workgroup_size.x(), workgroup_size.y(),
                                                    workgroup_size.z())];

  return OkStatus();
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
