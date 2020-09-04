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

#include "iree/hal/command_buffer_validation.h"

#include "absl/strings/str_join.h"
#include "iree/base/logging.h"
#include "iree/base/status.h"

namespace iree {
namespace hal {

namespace {

// Command buffer validation shim.
// Wraps an existing command buffer to provide in-depth validation during
// recording. This should be enabled whenever the command buffer is being driven
// by unsafe code or when early and readable diagnostics are needed.
class ValidatingCommandBuffer : public CommandBuffer {
 public:
  explicit ValidatingCommandBuffer(Allocator* allocator,
                                   ref_ptr<CommandBuffer> impl);
  ~ValidatingCommandBuffer() override;

  // Device allocator that commands encoded into the buffer share compatibility
  // with.
  Allocator* allocator() const { return allocator_; }

  CommandBuffer* impl() override { return impl_.get(); }

  bool is_recording() const override;

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
  // Returns a failure if the queue does not support the given caps.
  Status ValidateCategories(CommandCategoryBitfield required_categories) const;
  // Returns a failure if the memory type the buffer was allocated from is not
  // compatible with the given type.
  Status ValidateCompatibleMemoryType(Buffer* buffer,
                                      MemoryTypeBitfield memory_type) const;
  // Returns a failure if the buffer memory type or usage disallows the given
  // access type.
  Status ValidateAccess(Buffer* buffer,
                        MemoryAccessBitfield memory_access) const;
  // Returns a failure if the buffer was not allocated for the given usage.
  Status ValidateUsage(Buffer* buffer, BufferUsageBitfield usage) const;
  // Validates that the range provided is within the given buffer.
  Status ValidateRange(Buffer* buffer, device_size_t byte_offset,
                       device_size_t byte_length) const;

  // Validates that the currently bound descriptor sets are valid for the given
  // executable entry point.
  Status ValidateDispatchBindings(Executable* executable, int32_t entry_point);

  Allocator* const allocator_;
  ref_ptr<CommandBuffer> impl_;
};

ValidatingCommandBuffer::ValidatingCommandBuffer(Allocator* allocator,
                                                 ref_ptr<CommandBuffer> impl)
    : CommandBuffer(impl->mode(), impl->command_categories()),
      allocator_(allocator),
      impl_(std::move(impl)) {}

ValidatingCommandBuffer::~ValidatingCommandBuffer() = default;

bool ValidatingCommandBuffer::is_recording() const {
  return impl_->is_recording();
}

Status ValidatingCommandBuffer::Begin() {
  DVLOG(3) << "CommandBuffer::Begin()";
  if (impl_->is_recording()) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Command buffer is already recording";
  }
  return impl_->Begin();
}

Status ValidatingCommandBuffer::End() {
  DVLOG(3) << "CommandBuffer::End()";
  if (!impl_->is_recording()) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Command buffer is not recording";
  }
  return impl_->End();
}

Status ValidatingCommandBuffer::ValidateCategories(
    CommandCategoryBitfield required_categories) const {
  if (!AllBitsSet(command_categories(), required_categories)) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Operation requires categories "
           << CommandCategoryString(required_categories)
           << " but buffer only supports "
           << CommandCategoryString(command_categories());
  }
  return OkStatus();
}

Status ValidatingCommandBuffer::ValidateCompatibleMemoryType(
    Buffer* buffer, MemoryTypeBitfield memory_type) const {
  if ((buffer->memory_type() & memory_type) != memory_type) {
    // Missing one or more bits.
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "Buffer memory type is not compatible with the requested "
              "operation; buffer has "
           << MemoryTypeString(buffer->memory_type()) << ", operation requires "
           << MemoryTypeString(memory_type);
  }
  return OkStatus();
}

Status ValidatingCommandBuffer::ValidateAccess(
    Buffer* buffer, MemoryAccessBitfield memory_access) const {
  if ((buffer->allowed_access() & memory_access) != memory_access) {
    // Bits must match exactly.
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "The buffer does not support the requested access type; buffer "
              "allows "
           << MemoryAccessString(buffer->allowed_access())
           << ", operation requires " << MemoryAccessString(memory_access);
  }
  return OkStatus();
}

// Returns a failure if the buffer was not allocated for the given usage.
Status ValidatingCommandBuffer::ValidateUsage(Buffer* buffer,
                                              BufferUsageBitfield usage) const {
  if (!allocator()->CanUseBuffer(buffer, usage)) {
    // Buffer cannot be used on the queue for the given usage.
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "Requested usage of " << buffer->DebugString()
           << " is not supported for the buffer on this queue; "
              "buffer allows "
           << BufferUsageString(buffer->usage()) << ", queue requires "
           << BufferUsageString(usage);
  }

  if ((buffer->usage() & usage) != usage) {
    // Missing one or more bits.
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "Requested usage was not specified when the buffer was "
              "allocated; buffer allows "
           << BufferUsageString(buffer->usage()) << ", operation requires "
           << BufferUsageString(usage);
  }

  return OkStatus();
}

// Validates that the range provided is within the given buffer.
Status ValidatingCommandBuffer::ValidateRange(Buffer* buffer,
                                              device_size_t byte_offset,
                                              device_size_t byte_length) const {
  // Check if the start of the range runs off the end of the buffer.
  if (byte_offset > buffer->byte_length()) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Attempted to access an address off the end of the valid buffer "
              "range (offset="
           << byte_offset << ", length=" << byte_length
           << ", buffer byte_length=" << buffer->byte_length() << ")";
  }

  if (byte_length == 0) {
    // Fine to have a zero length.
    return OkStatus();
  }

  // Check if the end runs over the allocation.
  device_size_t end = byte_offset + byte_length;
  if (end > buffer->byte_length()) {
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Attempted to access an address outside of the valid buffer "
              "range (offset="
           << byte_offset << ", length=" << byte_length
           << ", end(inc)=" << (end - 1)
           << ", buffer byte_length=" << buffer->byte_length() << ")";
  }

  return OkStatus();
}

Status ValidatingCommandBuffer::ExecutionBarrier(
    ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  DVLOG(3) << "CommandBuffer::ExecutionBarrier(...)";

  // TODO(benvanik): additional synchronization validation.
  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kTransfer |
                                          CommandCategory::kDispatch));

  return impl_->ExecutionBarrier(source_stage_mask, target_stage_mask,
                                 memory_barriers, buffer_barriers);
}

Status ValidatingCommandBuffer::SignalEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  DVLOG(3) << "CommandBuffer::SignalEvent(...)";

  // TODO(benvanik): additional synchronization validation.
  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kDispatch));

  return impl_->SignalEvent(event, source_stage_mask);
}

Status ValidatingCommandBuffer::ResetEvent(
    Event* event, ExecutionStageBitfield source_stage_mask) {
  DVLOG(3) << "CommandBuffer::ResetEvent(...)";

  // TODO(benvanik): additional synchronization validation.
  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kDispatch));

  return impl_->ResetEvent(event, source_stage_mask);
}

Status ValidatingCommandBuffer::WaitEvents(
    absl::Span<Event*> events, ExecutionStageBitfield source_stage_mask,
    ExecutionStageBitfield target_stage_mask,
    absl::Span<const MemoryBarrier> memory_barriers,
    absl::Span<const BufferBarrier> buffer_barriers) {
  DVLOG(3) << "CommandBuffer::WaitEvents(...)";

  // TODO(benvanik): additional synchronization validation.
  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kDispatch));

  return impl_->WaitEvents(events, source_stage_mask, target_stage_mask,
                           memory_barriers, buffer_barriers);
}

Status ValidatingCommandBuffer::FillBuffer(Buffer* target_buffer,
                                           device_size_t target_offset,
                                           device_size_t length,
                                           const void* pattern,
                                           size_t pattern_length) {
  DVLOG(3) << "CommandBuffer::FillBuffer(" << target_buffer->DebugString()
           << ", " << target_offset << ", " << length << ", ??, "
           << pattern_length << ")";

  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kTransfer));
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(target_buffer, MemoryType::kDeviceVisible));
  IREE_RETURN_IF_ERROR(ValidateAccess(target_buffer, MemoryAccess::kWrite));
  IREE_RETURN_IF_ERROR(ValidateUsage(target_buffer, BufferUsage::kTransfer));
  IREE_RETURN_IF_ERROR(ValidateRange(target_buffer, target_offset, length));

  // Ensure the value length is supported.
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Fill value length is not one of the supported values "
              "(pattern_length="
           << pattern_length << ")";
  }

  // Ensure the offset and length have an alignment matching the value length.
  if ((target_offset % pattern_length) != 0 || (length % pattern_length) != 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Fill offset and/or length do not match the natural alignment of "
              "the fill value (target_offset="
           << target_offset << ", length=" << length
           << ", pattern_length=" << pattern_length << ")";
  }

  return impl_->FillBuffer(target_buffer, target_offset, length, pattern,
                           pattern_length);
}

Status ValidatingCommandBuffer::DiscardBuffer(Buffer* buffer) {
  DVLOG(3) << "CommandBuffer::DiscardBuffer(" << buffer->DebugString() << ")";

  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kTransfer));
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(buffer, MemoryType::kDeviceVisible));
  IREE_RETURN_IF_ERROR(ValidateUsage(buffer, BufferUsage::kNone));

  return impl_->DiscardBuffer(buffer);
}

Status ValidatingCommandBuffer::UpdateBuffer(const void* source_buffer,
                                             device_size_t source_offset,
                                             Buffer* target_buffer,
                                             device_size_t target_offset,
                                             device_size_t length) {
  DVLOG(3) << "CommandBuffer::UpdateBuffer(" << source_buffer << ", "
           << source_offset << ", " << target_buffer->DebugString() << ", "
           << target_offset << ", " << length << ")";

  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kTransfer));
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(target_buffer, MemoryType::kDeviceVisible));
  IREE_RETURN_IF_ERROR(ValidateAccess(target_buffer, MemoryAccess::kWrite));
  IREE_RETURN_IF_ERROR(ValidateUsage(target_buffer, BufferUsage::kTransfer));
  IREE_RETURN_IF_ERROR(ValidateRange(target_buffer, target_offset, length));

  return impl_->UpdateBuffer(source_buffer, source_offset, target_buffer,
                             target_offset, length);
}

Status ValidatingCommandBuffer::CopyBuffer(Buffer* source_buffer,
                                           device_size_t source_offset,
                                           Buffer* target_buffer,
                                           device_size_t target_offset,
                                           device_size_t length) {
  DVLOG(3) << "CommandBuffer::CopyBuffer(" << source_buffer->DebugString()
           << ", " << source_offset << ", " << target_buffer->DebugString()
           << ", " << target_offset << ", " << length << ")";

  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kTransfer));

  // At least source or destination must be device-visible to enable
  // host->device, device->host, and device->device.
  // TODO(b/117338171): host->host copies.
  if (!AnyBitSet(source_buffer->memory_type() & MemoryType::kDeviceVisible) &&
      !AnyBitSet(target_buffer->memory_type() & MemoryType::kDeviceVisible)) {
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "At least one buffer must be device-visible for a copy; "
              "source_buffer="
           << MemoryTypeString(source_buffer->memory_type())
           << ", target_buffer="
           << MemoryTypeString(target_buffer->memory_type());
  }

  IREE_RETURN_IF_ERROR(ValidateAccess(source_buffer, MemoryAccess::kRead));
  IREE_RETURN_IF_ERROR(ValidateAccess(target_buffer, MemoryAccess::kWrite));
  IREE_RETURN_IF_ERROR(ValidateUsage(source_buffer, BufferUsage::kTransfer));
  IREE_RETURN_IF_ERROR(ValidateUsage(target_buffer, BufferUsage::kTransfer));
  IREE_RETURN_IF_ERROR(ValidateRange(source_buffer, source_offset, length));
  IREE_RETURN_IF_ERROR(ValidateRange(target_buffer, target_offset, length));

  // Check for overlap - just like memcpy we don't handle that.
  if (Buffer::TestOverlap(source_buffer, source_offset, length, target_buffer,
                          target_offset,
                          length) != Buffer::Overlap::kDisjoint) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Source and target ranges overlap within the same buffer";
  }

  return impl_->CopyBuffer(source_buffer, source_offset, target_buffer,
                           target_offset, length);
}

Status ValidatingCommandBuffer::PushConstants(
    ExecutableLayout* executable_layout, size_t offset,
    absl::Span<const uint32_t> values) {
  DVLOG(3) << "CommandBuffer::PushConstants("
           << executable_layout->DebugString() << ", " << offset << ", "
           << absl::StrJoin(values, ", ") << ")";

  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kDispatch));

  // TODO(benvanik): validate offset and value count with layout.

  return impl_->PushConstants(executable_layout, offset, values);
}

Status ValidatingCommandBuffer::PushDescriptorSet(
    ExecutableLayout* executable_layout, int32_t set,
    absl::Span<const DescriptorSet::Binding> bindings) {
  DVLOG(3) << "CommandBuffer::PushDescriptorSet("
           << executable_layout->DebugString() << ", " << set << ", ["
           << absl::StrJoin(bindings, ", ", DescriptorSetBindingFormatter())
           << "])";

  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kDispatch));

  // TODO(benvanik): validate set index.
  // TODO(benvanik): validate bindings.

  return impl_->PushDescriptorSet(executable_layout, set, bindings);
}

Status ValidatingCommandBuffer::BindDescriptorSet(
    ExecutableLayout* executable_layout, int32_t set,
    DescriptorSet* descriptor_set,
    absl::Span<const device_size_t> dynamic_offsets) {
  DVLOG(3) << "CommandBuffer::BindDescriptorSet("
           << executable_layout->DebugString() << ", " << set << ", "
           << descriptor_set->DebugString() << ", ["
           << absl::StrJoin(dynamic_offsets, ", ") << "])";

  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kDispatch));

  // TODO(benvanik): validate set index.
  // TODO(benvanik): validate dynamic offsets (both count and offsets).

  return impl_->BindDescriptorSet(executable_layout, set, descriptor_set,
                                  dynamic_offsets);
}

Status ValidatingCommandBuffer::ValidateDispatchBindings(Executable* executable,
                                                         int32_t entry_point) {
  // Validate all buffers referenced have compatible memory types, access
  // rights, and usage.
  // TODO(benvanik): add validation by walking executable layout.
  // for (const auto& binding : bindings) {
  //   IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(binding.buffer,
  //                                                MemoryType::kDeviceVisible))
  //       << "input buffer: " << MemoryAccessString(binding.access) << " "
  //       << binding.buffer->DebugStringShort();
  //   IREE_RETURN_IF_ERROR(ValidateAccess(binding.buffer, binding.access));
  //   IREE_RETURN_IF_ERROR(ValidateUsage(binding.buffer,
  //   BufferUsage::kDispatch));
  // TODO(benvanik): validate it matches the executable expectations.
  // TODO(benvanik): validate buffer contains enough data for shape+size.
  // }

  // TODO(benvanik): validate no aliasing between inputs/outputs.

  return OkStatus();
}

Status ValidatingCommandBuffer::Dispatch(Executable* executable,
                                         int32_t entry_point,
                                         std::array<uint32_t, 3> workgroups) {
  DVLOG(3) << "CommandBuffer::Dispatch(" << executable->DebugString() << ", "
           << entry_point << ", [" << absl::StrJoin(workgroups, ", ") << "])";

  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kDispatch));
  IREE_RETURN_IF_ERROR(ValidateDispatchBindings(executable, entry_point));

  return impl_->Dispatch(executable, entry_point, workgroups);
}

Status ValidatingCommandBuffer::DispatchIndirect(
    Executable* executable, int32_t entry_point, Buffer* workgroups_buffer,
    device_size_t workgroups_offset) {
  DVLOG(3) << "CommandBuffer::DispatchIndirect(" << executable->DebugString()
           << ", " << entry_point << ", " << workgroups_buffer->DebugString()
           << ", " << workgroups_offset << ")";

  IREE_RETURN_IF_ERROR(ValidateCategories(CommandCategory::kDispatch));

  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(workgroups_buffer,
                                                    MemoryType::kDeviceVisible))
      << "input buffer: " << workgroups_buffer->DebugStringShort();
  IREE_RETURN_IF_ERROR(ValidateAccess(workgroups_buffer, MemoryAccess::kRead));
  IREE_RETURN_IF_ERROR(
      ValidateUsage(workgroups_buffer, BufferUsage::kDispatch));
  IREE_RETURN_IF_ERROR(ValidateRange(workgroups_buffer, workgroups_offset,
                                     sizeof(uint32_t) * 3));

  IREE_RETURN_IF_ERROR(ValidateDispatchBindings(executable, entry_point));

  return impl_->DispatchIndirect(executable, entry_point, workgroups_buffer,
                                 workgroups_offset);
}

}  // namespace

ref_ptr<CommandBuffer> WrapCommandBufferWithValidation(
    Allocator* allocator, ref_ptr<CommandBuffer> impl) {
  return make_ref<ValidatingCommandBuffer>(allocator, std::move(impl));
}

}  // namespace hal
}  // namespace iree
