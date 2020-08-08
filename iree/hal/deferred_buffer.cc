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

#include "iree/hal/deferred_buffer.h"

#include "iree/base/status.h"

namespace iree {
namespace hal {

DeferredBuffer::DeferredBuffer(Allocator* allocator,
                               MemoryTypeBitfield memory_type,
                               MemoryAccessBitfield allowed_access,
                               BufferUsageBitfield usage,
                               device_size_t byte_length)
    : Buffer(allocator, memory_type, allowed_access, usage, 0, 0, byte_length) {
}

DeferredBuffer::~DeferredBuffer() = default;

Status DeferredBuffer::GrowByteLength(device_size_t new_byte_length) {
  if (parent_buffer_) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Attempting to set min allocation size while bound to an "
              "allocation";
  }
  if (byte_length_ != kWholeBuffer && new_byte_length < byte_length_) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Attempting to shrink a buffer to " << new_byte_length
           << " when it has a minimum size of " << byte_length_;
  }
  byte_length_ = new_byte_length;
  return OkStatus();
}

Status DeferredBuffer::BindAllocation(ref_ptr<Buffer> allocated_buffer,
                                      device_size_t byte_offset,
                                      device_size_t byte_length) {
  // We can only be bound to allocations that are compatible with our specified
  // allocator and usage.
  if (!allocator_->CanUseBuffer(allocated_buffer.get(), usage())) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Allocation is not compatible with the allocator specified for "
              "the deferred buffer";
  }

  // Calculate the range in the allocated_buffer that we are interested in.
  IREE_RETURN_IF_ERROR(
      Buffer::CalculateRange(0, allocated_buffer->byte_length(), byte_offset,
                             byte_length, &byte_offset, &byte_length));

  // Verify that we have enough bytes for what we've promised.
  if (byte_length < byte_length_) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Allocation range is too small; min_allocation_size="
           << byte_length_ << " but the range of " << byte_offset << "-"
           << (byte_offset + byte_length - 1) << " (" << byte_length
           << "b) is too small";
  }

  allocated_buffer_ = allocated_buffer.get();
  parent_buffer_ = std::move(allocated_buffer);
  byte_offset_ = byte_offset;
  return OkStatus();
}

void DeferredBuffer::ResetAllocation() {
  allocated_buffer_ = this;
  parent_buffer_.reset();
  byte_offset_ = 0;
}

StatusOr<Buffer*> DeferredBuffer::ResolveAllocation() const {
  // If you get errors here then someone allocated the buffer with
  // MemoryType::kTransient and you are trying to use it outside of the time
  // it is actually allocated (such as during CommandBuffer evaluation). If
  // you need to use the buffer in non-transient ways then allocate the buffer
  // without the MemoryType::kTransient flag.
  if (!parent_buffer_) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Attempting to use a transient buffer prior to allocation: "
           << DebugString();
  }
  return parent_buffer_.get();
}

Status DeferredBuffer::FillImpl(device_size_t byte_offset,
                                device_size_t byte_length, const void* pattern,
                                device_size_t pattern_length) {
  IREE_ASSIGN_OR_RETURN(auto* allocated_buffer, ResolveAllocation());
  return allocated_buffer->FillImpl(byte_offset, byte_length, pattern,
                                    pattern_length);
}

Status DeferredBuffer::ReadDataImpl(device_size_t source_offset, void* data,
                                    device_size_t data_length) {
  IREE_ASSIGN_OR_RETURN(auto* allocated_buffer, ResolveAllocation());
  return allocated_buffer->ReadDataImpl(source_offset, data, data_length);
}

Status DeferredBuffer::WriteDataImpl(device_size_t target_offset,
                                     const void* data,
                                     device_size_t data_length) {
  IREE_ASSIGN_OR_RETURN(auto* allocated_buffer, ResolveAllocation());
  return allocated_buffer->WriteDataImpl(target_offset, data, data_length);
}

Status DeferredBuffer::CopyDataImpl(device_size_t target_offset,
                                    Buffer* source_buffer,
                                    device_size_t source_offset,
                                    device_size_t data_length) {
  IREE_ASSIGN_OR_RETURN(auto* allocated_buffer, ResolveAllocation());
  return allocated_buffer->CopyDataImpl(target_offset, source_buffer,
                                        source_offset, data_length);
}

Status DeferredBuffer::MapMemoryImpl(MappingMode mapping_mode,
                                     MemoryAccessBitfield memory_access,
                                     device_size_t local_byte_offset,
                                     device_size_t local_byte_length,
                                     void** out_data) {
  IREE_ASSIGN_OR_RETURN(auto* allocated_buffer, ResolveAllocation());
  return allocated_buffer->MapMemoryImpl(mapping_mode, memory_access,
                                         local_byte_offset, local_byte_length,
                                         out_data);
}

Status DeferredBuffer::UnmapMemoryImpl(device_size_t local_byte_offset,
                                       device_size_t local_byte_length,
                                       void* data) {
  IREE_ASSIGN_OR_RETURN(auto* allocated_buffer, ResolveAllocation());
  return allocated_buffer->UnmapMemoryImpl(local_byte_offset, local_byte_length,
                                           data);
}

Status DeferredBuffer::InvalidateMappedMemoryImpl(
    device_size_t local_byte_offset, device_size_t local_byte_length) {
  IREE_ASSIGN_OR_RETURN(auto* allocated_buffer, ResolveAllocation());
  return allocated_buffer->InvalidateMappedMemoryImpl(local_byte_offset,
                                                      local_byte_length);
}

Status DeferredBuffer::FlushMappedMemoryImpl(device_size_t local_byte_offset,
                                             device_size_t local_byte_length) {
  IREE_ASSIGN_OR_RETURN(auto* allocated_buffer, ResolveAllocation());
  return allocated_buffer->FlushMappedMemoryImpl(local_byte_offset,
                                                 local_byte_length);
}

}  // namespace hal
}  // namespace iree
