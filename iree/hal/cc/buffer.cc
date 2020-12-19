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

#include "iree/hal/cc/buffer.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <sstream>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "iree/base/status.h"

namespace iree {
namespace hal {

std::string MemoryTypeString(iree_hal_memory_type_t memory_type) {
  return "TODO";
  // return FormatBitfieldValue(
  //     memory_type, {
  //                      // Combined:
  //                      {IREE_HAL_MEMORY_TYPE_HOST_LOCAL, "kHostLocal"},
  //                      {IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, "kDeviceLocal"},
  //                      // Separate:
  //                      {IREE_HAL_MEMORY_TYPE_TRANSIENT, "kTransient"},
  //                      {IREE_HAL_MEMORY_TYPE_HOST_VISIBLE, "kHostVisible"},
  //                      {IREE_HAL_MEMORY_TYPE_HOST_COHERENT, "kHostCoherent"},
  //                      {IREE_HAL_MEMORY_TYPE_HOST_CACHED, "kHostCached"},
  //                      {IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  //                      "kDeviceVisible"},
  //                  });
}

std::string MemoryAccessString(iree_hal_memory_access_t memory_access) {
  return "TODO";
  // return FormatBitfieldValue(
  //     memory_access,
  //     {
  //         // Combined:
  //         {IREE_HAL_MEMORY_ACCESS_ALL, "kAll"},
  //         {IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, "kDiscardWrite"},
  //         // Separate:
  //         {IREE_HAL_MEMORY_ACCESS_READ, "kRead"},
  //         {IREE_HAL_MEMORY_ACCESS_WRITE, "kWrite"},
  //         {IREE_HAL_MEMORY_ACCESS_DISCARD, "kDiscard"},
  //         {IREE_HAL_MEMORY_ACCESS_MAY_ALIAS, "kMayAlias"},
  //     });
}

std::string BufferUsageString(iree_hal_buffer_usage_t buffer_usage) {
  return "TODO";
  // return FormatBitfieldValue(buffer_usage,
  //                            {
  //                                // Combined:
  //                                {IREE_HAL_BUFFER_USAGE_ALL, "kAll"},
  //                                // Separate:
  //                                {IREE_HAL_BUFFER_USAGE_CONSTANT,
  //                                "kConstant"},
  //                                {IREE_HAL_BUFFER_USAGE_TRANSFER,
  //                                "kTransfer"},
  //                                {IREE_HAL_BUFFER_USAGE_MAPPING, "kMapping"},
  //                                {IREE_HAL_BUFFER_USAGE_DISPATCH,
  //                                "kDispatch"},
  //                            });
}

// Special router for buffers that just reference other buffers.
// We keep this out of the base Buffer so that it's a bit easier to track
// delegation.
class SubspanBuffer : public Buffer {
 public:
  SubspanBuffer(ref_ptr<Buffer> parent_buffer, iree_device_size_t byte_offset,
                iree_device_size_t byte_length)
      : Buffer(parent_buffer->allocator(), parent_buffer->memory_type(),
               parent_buffer->allowed_access(), parent_buffer->usage(),
               parent_buffer->allocation_size(), byte_offset, byte_length) {
    allocated_buffer_ = parent_buffer.get();
    parent_buffer_ = std::move(parent_buffer);
  }

 protected:
  Status FillImpl(iree_device_size_t byte_offset,
                  iree_device_size_t byte_length, const void* pattern,
                  iree_device_size_t pattern_length) override {
    return parent_buffer_->FillImpl(byte_offset, byte_length, pattern,
                                    pattern_length);
  }

  Status ReadDataImpl(iree_device_size_t source_offset, void* data,
                      iree_device_size_t data_length) override {
    return parent_buffer_->ReadDataImpl(source_offset, data, data_length);
  }

  Status WriteDataImpl(iree_device_size_t target_offset, const void* data,
                       iree_device_size_t data_length) override {
    return parent_buffer_->WriteDataImpl(target_offset, data, data_length);
  }

  Status CopyDataImpl(iree_device_size_t target_offset, Buffer* source_buffer,
                      iree_device_size_t source_offset,
                      iree_device_size_t data_length) override {
    return parent_buffer_->CopyDataImpl(target_offset, source_buffer,
                                        source_offset, data_length);
  }

  Status MapMemoryImpl(MappingMode mapping_mode,
                       iree_hal_memory_access_t memory_access,
                       iree_device_size_t local_byte_offset,
                       iree_device_size_t local_byte_length,
                       void** out_data) override {
    return parent_buffer_->MapMemoryImpl(mapping_mode, memory_access,
                                         local_byte_offset, local_byte_length,
                                         out_data);
  }

  Status UnmapMemoryImpl(iree_device_size_t local_byte_offset,
                         iree_device_size_t local_byte_length,
                         void* data) override {
    return parent_buffer_->UnmapMemoryImpl(local_byte_offset, local_byte_length,
                                           data);
  }

  Status InvalidateMappedMemoryImpl(
      iree_device_size_t local_byte_offset,
      iree_device_size_t local_byte_length) override {
    return parent_buffer_->InvalidateMappedMemoryImpl(local_byte_offset,
                                                      local_byte_length);
  }

  Status FlushMappedMemoryImpl(iree_device_size_t local_byte_offset,
                               iree_device_size_t local_byte_length) override {
    return parent_buffer_->FlushMappedMemoryImpl(local_byte_offset,
                                                 local_byte_length);
  }
};

// static
StatusOr<ref_ptr<Buffer>> Buffer::Subspan(const ref_ptr<Buffer>& buffer,
                                          iree_device_size_t byte_offset,
                                          iree_device_size_t byte_length) {
  IREE_RETURN_IF_ERROR(buffer->CalculateRange(byte_offset, byte_length,
                                              &byte_offset, &byte_length));
  if (byte_offset == 0 && byte_length == buffer->byte_length()) {
    // Asking for the same buffer.
    return add_ref(buffer);
  }

  // To avoid heavy nesting of subspans that just add indirection we go to the
  // parent buffer directly. If we wanted better accounting (to track where
  // buffers came from) we'd want to avoid this but I'm not sure that's worth
  // the super deep indirection that could arise.
  if (buffer->allocated_buffer() != buffer.get()) {
    IREE_CHECK(buffer->parent_buffer_);
    return Buffer::Subspan(buffer->parent_buffer_, byte_offset, byte_length);
  } else {
    return {make_ref<SubspanBuffer>(add_ref(buffer), byte_offset, byte_length)};
  }
}

// static
Buffer::Overlap Buffer::TestOverlap(Buffer* lhs_buffer,
                                    iree_device_size_t lhs_offset,
                                    iree_device_size_t lhs_length,
                                    Buffer* rhs_buffer,
                                    iree_device_size_t rhs_offset,
                                    iree_device_size_t rhs_length) {
  if (lhs_buffer->allocated_buffer() != rhs_buffer->allocated_buffer()) {
    // Not even the same buffers.
    return Overlap::kDisjoint;
  }
  // Resolve offsets into the underlying allocation.
  iree_device_size_t lhs_alloc_offset = lhs_buffer->byte_offset() + lhs_offset;
  iree_device_size_t rhs_alloc_offset = rhs_buffer->byte_offset() + rhs_offset;
  iree_device_size_t lhs_alloc_length =
      lhs_length == IREE_WHOLE_BUFFER ? lhs_buffer->byte_length() - lhs_offset
                                      : lhs_length;
  iree_device_size_t rhs_alloc_length =
      rhs_length == IREE_WHOLE_BUFFER ? rhs_buffer->byte_length() - rhs_offset
                                      : rhs_length;
  if (!lhs_alloc_length || !rhs_alloc_length) {
    return Overlap::kDisjoint;
  }
  if (lhs_alloc_offset == rhs_alloc_offset &&
      lhs_alloc_length == rhs_alloc_length) {
    return Overlap::kComplete;
  }
  return lhs_alloc_offset + lhs_alloc_length > rhs_alloc_offset &&
                 rhs_alloc_offset + rhs_alloc_length > lhs_alloc_offset
             ? Overlap::kPartial
             : Overlap::kDisjoint;
}

// static
bool Buffer::DoesOverlap(Buffer* lhs_buffer, iree_device_size_t lhs_offset,
                         iree_device_size_t lhs_length, Buffer* rhs_buffer,
                         iree_device_size_t rhs_offset,
                         iree_device_size_t rhs_length) {
  return TestOverlap(lhs_buffer, lhs_offset, lhs_length, rhs_buffer, rhs_offset,
                     rhs_length) != Overlap::kDisjoint;
}

Buffer::Buffer(Allocator* allocator, iree_hal_memory_type_t memory_type,
               iree_hal_memory_access_t allowed_access,
               iree_hal_buffer_usage_t usage,
               iree_device_size_t allocation_size,
               iree_device_size_t byte_offset, iree_device_size_t byte_length)
    : allocated_buffer_(const_cast<Buffer*>(this)),
      allocator_(allocator),
      memory_type_(memory_type),
      allowed_access_(allowed_access),
      usage_(usage),
      allocation_size_(allocation_size),
      byte_offset_(byte_offset),
      byte_length_(byte_length) {}

Buffer* Buffer::allocated_buffer() const noexcept {
  Buffer* allocated_buffer = allocated_buffer_;
  while (allocated_buffer != this &&
         allocated_buffer != allocated_buffer->allocated_buffer()) {
    allocated_buffer = allocated_buffer->allocated_buffer();
  }
  return allocated_buffer;
}

std::string Buffer::DebugString() const {
  std::ostringstream stream;
  stream << allocated_buffer()->debug_name() << "["
         << (allocation_size() == IREE_WHOLE_BUFFER
                 ? "?"
                 : std::to_string(allocation_size()))
         << "].";
  if (iree_any_bit_set(memory_type(), IREE_HAL_MEMORY_TYPE_TRANSIENT))
    stream << "Z";
  if ((memory_type() & IREE_HAL_MEMORY_TYPE_HOST_LOCAL) ==
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL) {
    stream << "h";
  } else {
    if (iree_any_bit_set(memory_type(), IREE_HAL_MEMORY_TYPE_HOST_VISIBLE))
      stream << "v";
    if (iree_any_bit_set(memory_type(), IREE_HAL_MEMORY_TYPE_HOST_COHERENT))
      stream << "x";
    if (iree_any_bit_set(memory_type(), IREE_HAL_MEMORY_TYPE_HOST_CACHED))
      stream << "c";
  }
  if (iree_all_bits_set(memory_type(), IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    stream << "D";
  } else {
    if (iree_any_bit_set(memory_type(), IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE))
      stream << "V";
  }
  stream << ".";
  if (iree_any_bit_set(usage(), IREE_HAL_BUFFER_USAGE_CONSTANT)) stream << "c";
  if (iree_any_bit_set(usage(), IREE_HAL_BUFFER_USAGE_TRANSFER)) stream << "t";
  if (iree_any_bit_set(usage(), IREE_HAL_BUFFER_USAGE_MAPPING)) stream << "m";
  if (iree_any_bit_set(usage(), IREE_HAL_BUFFER_USAGE_DISPATCH)) stream << "d";
  if (byte_offset_ || byte_length_ != allocation_size_) {
    stream << "(" << byte_offset_ << "-" << (byte_offset_ + byte_length_ - 1)
           << ")";
  }
  return stream.str();
}

std::string Buffer::DebugStringShort() const {
  // TODO(benvanik): figure out what's most useful here. Maybe a long variant?
  std::ostringstream stream;
  stream << allocated_buffer()->debug_name() << "["
         << (allocation_size() == IREE_WHOLE_BUFFER
                 ? "?"
                 : std::to_string(allocation_size()))
         << "]";
  if (byte_offset_ || byte_length_ != allocation_size_) {
    stream << "(" << byte_offset_ << "-" << (byte_offset_ + byte_length_ - 1)
           << ")";
  }
  return stream.str();
}

Status Buffer::ValidateCompatibleMemoryType(
    iree_hal_memory_type_t memory_type) const {
  if ((memory_type_ & memory_type) != memory_type) {
    // Missing one or more bits.
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "Buffer memory type is not compatible with the requested "
              "operation; buffer has "
           << MemoryTypeString(memory_type_) << ", operation requires "
           << MemoryTypeString(memory_type);
  }
  return OkStatus();
}

Status Buffer::ValidateAccess(iree_hal_memory_access_t memory_access) const {
  if (!iree_any_bit_set(memory_access, (IREE_HAL_MEMORY_ACCESS_READ |
                                        IREE_HAL_MEMORY_ACCESS_WRITE))) {
    // No actual access bits defined.
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Memory access must specify one or more of kRead or kWrite";
  } else if ((allowed_access_ & memory_access) != memory_access) {
    // Bits must match exactly.
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "The buffer does not support the requested access type; buffer "
              "allows "
           << MemoryAccessString(allowed_access_) << ", operation requires "
           << MemoryAccessString(memory_access);
  }
  return OkStatus();
}

Status Buffer::ValidateUsage(iree_hal_buffer_usage_t usage) const {
  if ((usage_ & usage) != usage) {
    // Missing one or more bits.
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "Requested usage was not specified when the buffer was "
              "allocated; buffer allows "
           << BufferUsageString(usage_) << ", operation requires "
           << BufferUsageString(usage);
  }
  return OkStatus();
}

Status Buffer::CalculateRange(iree_device_size_t base_offset,
                              iree_device_size_t max_length,
                              iree_device_size_t offset,
                              iree_device_size_t length,
                              iree_device_size_t* out_adjusted_offset,
                              iree_device_size_t* out_adjusted_length) {
  // Check if the start of the range runs off the end of the buffer.
  if (offset > max_length) {
    *out_adjusted_offset = 0;
    if (out_adjusted_length) *out_adjusted_length = 0;
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Attempted to access an address off the end of the valid buffer "
              "range (offset="
           << offset << ", length=" << length
           << ", buffer byte_length=" << max_length << ")";
  }

  // Handle length as IREE_WHOLE_BUFFER by adjusting it (if allowed).
  if (length == IREE_WHOLE_BUFFER && !out_adjusted_length) {
    *out_adjusted_offset = 0;
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "IREE_WHOLE_BUFFER may only be used with buffer ranges, not "
              "external "
              "pointer ranges";
  }

  // Calculate the real ranges adjusted for our region within the allocation.
  iree_device_size_t adjusted_offset = base_offset + offset;
  iree_device_size_t adjusted_length =
      length == IREE_WHOLE_BUFFER ? max_length - offset : length;
  if (adjusted_length == 0) {
    // Fine to have a zero length.
    *out_adjusted_offset = adjusted_offset;
    if (out_adjusted_length) *out_adjusted_length = adjusted_length;
    return OkStatus();
  }

  // Check if the end runs over the allocation.
  iree_device_size_t end = offset + adjusted_length - 1;
  if (end >= max_length) {
    *out_adjusted_offset = 0;
    if (out_adjusted_length) *out_adjusted_length = 0;
    return OutOfRangeErrorBuilder(IREE_LOC)
           << "Attempted to access an address outside of the valid buffer "
              "range (offset="
           << offset << ", adjusted_length=" << adjusted_length
           << ", end=" << end << ", buffer byte_length=" << max_length << ")";
  }

  *out_adjusted_offset = adjusted_offset;
  if (out_adjusted_length) *out_adjusted_length = adjusted_length;
  return OkStatus();
}

Status Buffer::CalculateRange(iree_device_size_t offset,
                              iree_device_size_t length,
                              iree_device_size_t* out_adjusted_offset,
                              iree_device_size_t* out_adjusted_length) const {
  return CalculateRange(byte_offset_, byte_length_, offset, length,
                        out_adjusted_offset, out_adjusted_length);
}

Status Buffer::CalculateLocalRange(iree_device_size_t max_length,
                                   iree_device_size_t offset,
                                   iree_device_size_t length,
                                   iree_device_size_t* out_adjusted_offset,
                                   iree_device_size_t* out_adjusted_length) {
  return CalculateRange(0, max_length, offset, length, out_adjusted_offset,
                        out_adjusted_length);
}

Status Buffer::Fill(iree_device_size_t byte_offset,
                    iree_device_size_t byte_length, const void* pattern,
                    iree_device_size_t pattern_length) {
  // If not host visible we'll need to issue command buffers.
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(ValidateAccess(IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(ValidateUsage(IREE_HAL_BUFFER_USAGE_MAPPING));
  IREE_RETURN_IF_ERROR(
      CalculateRange(byte_offset, byte_length, &byte_offset, &byte_length));
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Fill patterns must be 1, 2, or 4 bytes";
  }
  if ((byte_offset % pattern_length) != 0 ||
      (byte_length % pattern_length) != 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Attempting to fill a range with " << pattern_length
           << " byte values that is not "
              "aligned (offset="
           << byte_offset << ", length=" << byte_length << ")";
  }
  if (byte_length == 0) {
    return OkStatus();  // No-op.
  }
  const uint32_t kZero = 0;
  if (std::memcmp(pattern, &kZero, pattern_length) == 0) {
    // We can turn all-zero values into single-byte fills as that can be much
    // faster on devices (doing a fill8 vs fill32).
    pattern_length = 1;
  }
  return FillImpl(byte_offset, byte_length, pattern, pattern_length);
}

Status Buffer::ReadData(iree_device_size_t source_offset, void* data,
                        iree_device_size_t data_length) {
  // If not host visible we'll need to issue command buffers.
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(ValidateAccess(IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(ValidateUsage(IREE_HAL_BUFFER_USAGE_MAPPING));
  IREE_RETURN_IF_ERROR(
      CalculateRange(source_offset, data_length, &source_offset));
  if (data_length == 0) {
    return OkStatus();  // No-op.
  }
  return ReadDataImpl(source_offset, data, data_length);
}

Status Buffer::WriteData(iree_device_size_t target_offset, const void* data,
                         iree_device_size_t data_length) {
  // If not host visible we'll need to issue command buffers.
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(ValidateAccess(IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(ValidateUsage(IREE_HAL_BUFFER_USAGE_MAPPING));
  IREE_RETURN_IF_ERROR(
      CalculateRange(target_offset, data_length, &target_offset));
  if (data_length == 0) {
    return OkStatus();  // No-op.
  }
  return WriteDataImpl(target_offset, data, data_length);
}

Status Buffer::CopyData(iree_device_size_t target_offset, Buffer* source_buffer,
                        iree_device_size_t source_offset,
                        iree_device_size_t data_length) {
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(ValidateAccess(IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(ValidateUsage(IREE_HAL_BUFFER_USAGE_MAPPING));
  IREE_RETURN_IF_ERROR(source_buffer->ValidateCompatibleMemoryType(
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(
      source_buffer->ValidateAccess(IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(
      source_buffer->ValidateUsage(IREE_HAL_BUFFER_USAGE_MAPPING));

  // We need to validate both buffers.
  iree_device_size_t source_data_length = data_length;
  iree_device_size_t target_data_length = data_length;
  iree_device_size_t adjusted_source_offset;
  IREE_RETURN_IF_ERROR(source_buffer->CalculateRange(
      source_offset, source_data_length, &adjusted_source_offset,
      &source_data_length));
  IREE_RETURN_IF_ERROR(CalculateRange(target_offset, target_data_length,
                                      &target_offset, &target_data_length));
  iree_device_size_t adjusted_data_length;
  if (data_length == IREE_WHOLE_BUFFER) {
    // Whole buffer copy requested - that could mean either, so take the min.
    adjusted_data_length = std::min(source_data_length, target_data_length);
  } else {
    // Specific length requested - validate that we have matching lengths.
    IREE_CHECK_EQ(source_data_length, target_data_length);
    adjusted_data_length = source_data_length;
  }

  // Elide zero length copies.
  if (adjusted_data_length == 0) {
    return OkStatus();
  }

  // Check for overlap.
  if (this == source_buffer &&
      adjusted_source_offset <= target_offset + adjusted_data_length &&
      target_offset <= adjusted_source_offset + adjusted_data_length) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Source and target ranges overlap within the same buffer";
  }

  return CopyDataImpl(target_offset, source_buffer, source_offset,
                      adjusted_data_length);
}

Status Buffer::MapMemory(MappingMode mapping_mode,
                         iree_hal_memory_access_t memory_access,
                         iree_device_size_t* byte_offset,
                         iree_device_size_t* byte_length, void** out_data) {
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(ValidateAccess(memory_access));
  IREE_RETURN_IF_ERROR(ValidateUsage(IREE_HAL_BUFFER_USAGE_MAPPING));
  IREE_RETURN_IF_ERROR(
      CalculateRange(*byte_offset, *byte_length, byte_offset, byte_length));
  *out_data = nullptr;
  return MapMemoryImpl(mapping_mode, memory_access, *byte_offset, *byte_length,
                       out_data);
}

Status Buffer::UnmapMemory(iree_device_size_t local_byte_offset,
                           iree_device_size_t local_byte_length, void* data) {
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(ValidateUsage(IREE_HAL_BUFFER_USAGE_MAPPING));
  // NOTE: local_byte_offset/local_byte_length are already adjusted.
  return UnmapMemoryImpl(local_byte_offset, local_byte_length, data);
}

Status Buffer::InvalidateMappedMemory(iree_device_size_t local_byte_offset,
                                      iree_device_size_t local_byte_length) {
  IREE_RETURN_IF_ERROR(
      ValidateCompatibleMemoryType(IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  if (iree_any_bit_set(memory_type_, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "Buffer memory type is coherent and invalidation is not required";
  }
  IREE_RETURN_IF_ERROR(ValidateUsage(IREE_HAL_BUFFER_USAGE_MAPPING));
  // NOTE: local_byte_offset/local_byte_length are already adjusted.
  return InvalidateMappedMemoryImpl(local_byte_offset, local_byte_length);
}

Status Buffer::FlushMappedMemory(iree_device_size_t local_byte_offset,
                                 iree_device_size_t local_byte_length) {
  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED));
  IREE_RETURN_IF_ERROR(ValidateUsage(IREE_HAL_BUFFER_USAGE_MAPPING));
  // NOTE: local_byte_offset/local_byte_length are already adjusted.
  return FlushMappedMemoryImpl(local_byte_offset, local_byte_length);
}

}  // namespace hal
}  // namespace iree
