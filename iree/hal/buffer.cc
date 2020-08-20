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

#include "iree/hal/buffer.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <sstream>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "iree/base/status.h"

namespace iree {
namespace hal {

#if HAS_IREE_BUFFER_DEBUG_NAME
namespace {
// Used for diagnostic purposes only as a default buffer name.
std::atomic<int> next_buffer_id_{0};
}  // namespace
#endif  // HAS_IREE_BUFFER_DEBUG_NAME

std::string MemoryTypeString(MemoryTypeBitfield memory_type) {
  return FormatBitfieldValue(memory_type,
                             {
                                 // Combined:
                                 {MemoryType::kHostLocal, "kHostLocal"},
                                 {MemoryType::kDeviceLocal, "kDeviceLocal"},
                                 // Separate:
                                 {MemoryType::kTransient, "kTransient"},
                                 {MemoryType::kHostVisible, "kHostVisible"},
                                 {MemoryType::kHostCoherent, "kHostCoherent"},
                                 {MemoryType::kHostCached, "kHostCached"},
                                 {MemoryType::kDeviceVisible, "kDeviceVisible"},
                             });
}

std::string MemoryAccessString(MemoryAccessBitfield memory_access) {
  return FormatBitfieldValue(memory_access,
                             {
                                 // Combined:
                                 {MemoryAccess::kAll, "kAll"},
                                 {MemoryAccess::kDiscardWrite, "kDiscardWrite"},
                                 // Separate:
                                 {MemoryAccess::kRead, "kRead"},
                                 {MemoryAccess::kWrite, "kWrite"},
                                 {MemoryAccess::kDiscard, "kDiscard"},
                                 {MemoryAccess::kMayAlias, "kMayAlias"},
                             });
}

std::string BufferUsageString(BufferUsageBitfield buffer_usage) {
  return FormatBitfieldValue(buffer_usage,
                             {
                                 // Combined:
                                 {BufferUsage::kAll, "kAll"},
                                 // Separate:
                                 {BufferUsage::kConstant, "kConstant"},
                                 {BufferUsage::kTransfer, "kTransfer"},
                                 {BufferUsage::kMapping, "kMapping"},
                                 {BufferUsage::kDispatch, "kDispatch"},
                             });
}

// Special router for buffers that just reference other buffers.
// We keep this out of the base Buffer so that it's a bit easier to track
// delegation.
class SubspanBuffer : public Buffer {
 public:
  SubspanBuffer(ref_ptr<Buffer> parent_buffer, device_size_t byte_offset,
                device_size_t byte_length)
      : Buffer(parent_buffer->allocator(), parent_buffer->memory_type(),
               parent_buffer->allowed_access(), parent_buffer->usage(),
               parent_buffer->allocation_size(), byte_offset, byte_length) {
    allocated_buffer_ = parent_buffer.get();
    parent_buffer_ = std::move(parent_buffer);
  }

 protected:
  Status FillImpl(device_size_t byte_offset, device_size_t byte_length,
                  const void* pattern, device_size_t pattern_length) override {
    return parent_buffer_->FillImpl(byte_offset, byte_length, pattern,
                                    pattern_length);
  }

  Status ReadDataImpl(device_size_t source_offset, void* data,
                      device_size_t data_length) override {
    return parent_buffer_->ReadDataImpl(source_offset, data, data_length);
  }

  Status WriteDataImpl(device_size_t target_offset, const void* data,
                       device_size_t data_length) override {
    return parent_buffer_->WriteDataImpl(target_offset, data, data_length);
  }

  Status CopyDataImpl(device_size_t target_offset, Buffer* source_buffer,
                      device_size_t source_offset,
                      device_size_t data_length) override {
    return parent_buffer_->CopyDataImpl(target_offset, source_buffer,
                                        source_offset, data_length);
  }

  Status MapMemoryImpl(MappingMode mapping_mode,
                       MemoryAccessBitfield memory_access,
                       device_size_t local_byte_offset,
                       device_size_t local_byte_length,
                       void** out_data) override {
    return parent_buffer_->MapMemoryImpl(mapping_mode, memory_access,
                                         local_byte_offset, local_byte_length,
                                         out_data);
  }

  Status UnmapMemoryImpl(device_size_t local_byte_offset,
                         device_size_t local_byte_length, void* data) override {
    return parent_buffer_->UnmapMemoryImpl(local_byte_offset, local_byte_length,
                                           data);
  }

  Status InvalidateMappedMemoryImpl(device_size_t local_byte_offset,
                                    device_size_t local_byte_length) override {
    return parent_buffer_->InvalidateMappedMemoryImpl(local_byte_offset,
                                                      local_byte_length);
  }

  Status FlushMappedMemoryImpl(device_size_t local_byte_offset,
                               device_size_t local_byte_length) override {
    return parent_buffer_->FlushMappedMemoryImpl(local_byte_offset,
                                                 local_byte_length);
  }
};

// static
StatusOr<ref_ptr<Buffer>> Buffer::Subspan(const ref_ptr<Buffer>& buffer,
                                          device_size_t byte_offset,
                                          device_size_t byte_length) {
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
    CHECK(buffer->parent_buffer_);
    return Buffer::Subspan(buffer->parent_buffer_, byte_offset, byte_length);
  } else {
    return {make_ref<SubspanBuffer>(add_ref(buffer), byte_offset, byte_length)};
  }
}

// static
Buffer::Overlap Buffer::TestOverlap(
    Buffer* lhs_buffer, device_size_t lhs_offset, device_size_t lhs_length,
    Buffer* rhs_buffer, device_size_t rhs_offset, device_size_t rhs_length) {
  if (lhs_buffer->allocated_buffer() != rhs_buffer->allocated_buffer()) {
    // Not even the same buffers.
    return Overlap::kDisjoint;
  }
  // Resolve offsets into the underlying allocation.
  device_size_t lhs_alloc_offset = lhs_buffer->byte_offset() + lhs_offset;
  device_size_t rhs_alloc_offset = rhs_buffer->byte_offset() + rhs_offset;
  device_size_t lhs_alloc_length = lhs_length == kWholeBuffer
                                       ? lhs_buffer->byte_length() - lhs_offset
                                       : lhs_length;
  device_size_t rhs_alloc_length = rhs_length == kWholeBuffer
                                       ? rhs_buffer->byte_length() - rhs_offset
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
bool Buffer::DoesOverlap(Buffer* lhs_buffer, device_size_t lhs_offset,
                         device_size_t lhs_length, Buffer* rhs_buffer,
                         device_size_t rhs_offset, device_size_t rhs_length) {
  return TestOverlap(lhs_buffer, lhs_offset, lhs_length, rhs_buffer, rhs_offset,
                     rhs_length) != Overlap::kDisjoint;
}

Buffer::Buffer(Allocator* allocator, MemoryTypeBitfield memory_type,
               MemoryAccessBitfield allowed_access, BufferUsageBitfield usage,
               device_size_t allocation_size, device_size_t byte_offset,
               device_size_t byte_length)
    : allocated_buffer_(const_cast<Buffer*>(this)),
      allocator_(allocator),
      memory_type_(memory_type),
      allowed_access_(allowed_access),
      usage_(usage),
      allocation_size_(allocation_size),
      byte_offset_(byte_offset),
      byte_length_(byte_length) {
#if HAS_IREE_BUFFER_DEBUG_NAME
  // Default name for logging.
  // It'd be nice to defer this until it's required but that would require
  // synchronization or something.
  const char* debug_name_prefix = "";
  if ((memory_type_ & MemoryType::kHostLocal) == MemoryType::kHostLocal) {
    debug_name_prefix = "host_buffer_";
  } else if ((memory_type_ & MemoryType::kDeviceLocal) ==
             MemoryType::kDeviceLocal) {
    // TODO(benvanik): include allocator ID to differentiate devices.
    debug_name_prefix = "device_buffer_";
  }
  debug_name_ = absl::StrCat(debug_name_prefix, next_buffer_id_++);
#endif  // HAS_IREE_BUFFER_DEBUG_NAME
}

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
         << (allocation_size() == kWholeBuffer
                 ? "?"
                 : std::to_string(allocation_size()))
         << "].";
  if (AnyBitSet(memory_type() & MemoryType::kTransient)) stream << "Z";
  if ((memory_type() & MemoryType::kHostLocal) == MemoryType::kHostLocal) {
    stream << "h";
  } else {
    if (AnyBitSet(memory_type() & MemoryType::kHostVisible)) stream << "v";
    if (AnyBitSet(memory_type() & MemoryType::kHostCoherent)) stream << "x";
    if (AnyBitSet(memory_type() & MemoryType::kHostCached)) stream << "c";
  }
  if ((memory_type() & MemoryType::kDeviceLocal) == MemoryType::kDeviceLocal) {
    stream << "D";
  } else {
    if (AnyBitSet(memory_type() & MemoryType::kDeviceVisible)) stream << "V";
  }
  stream << ".";
  if (AnyBitSet(usage() & BufferUsage::kConstant)) stream << "c";
  if (AnyBitSet(usage() & BufferUsage::kTransfer)) stream << "t";
  if (AnyBitSet(usage() & BufferUsage::kMapping)) stream << "m";
  if (AnyBitSet(usage() & BufferUsage::kDispatch)) stream << "d";
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
         << (allocation_size() == kWholeBuffer
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
    MemoryTypeBitfield memory_type) const {
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

Status Buffer::ValidateAccess(MemoryAccessBitfield memory_access) const {
  if (!AnyBitSet(memory_access &
                 (MemoryAccess::kRead | MemoryAccess::kWrite))) {
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

Status Buffer::ValidateUsage(BufferUsageBitfield usage) const {
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

Status Buffer::CalculateRange(device_size_t base_offset,
                              device_size_t max_length, device_size_t offset,
                              device_size_t length,
                              device_size_t* out_adjusted_offset,
                              device_size_t* out_adjusted_length) {
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

  // Handle length as kWholeBuffer by adjusting it (if allowed).
  if (length == kWholeBuffer && !out_adjusted_length) {
    *out_adjusted_offset = 0;
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "kWholeBuffer may only be used with buffer ranges, not external "
              "pointer ranges";
  }

  // Calculate the real ranges adjusted for our region within the allocation.
  device_size_t adjusted_offset = base_offset + offset;
  device_size_t adjusted_length =
      length == kWholeBuffer ? max_length - offset : length;
  if (adjusted_length == 0) {
    // Fine to have a zero length.
    *out_adjusted_offset = adjusted_offset;
    if (out_adjusted_length) *out_adjusted_length = adjusted_length;
    return OkStatus();
  }

  // Check if the end runs over the allocation.
  device_size_t end = offset + adjusted_length - 1;
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

Status Buffer::CalculateRange(device_size_t offset, device_size_t length,
                              device_size_t* out_adjusted_offset,
                              device_size_t* out_adjusted_length) const {
  return CalculateRange(byte_offset_, byte_length_, offset, length,
                        out_adjusted_offset, out_adjusted_length);
}

Status Buffer::CalculateLocalRange(device_size_t max_length,
                                   device_size_t offset, device_size_t length,
                                   device_size_t* out_adjusted_offset,
                                   device_size_t* out_adjusted_length) {
  return CalculateRange(0, max_length, offset, length, out_adjusted_offset,
                        out_adjusted_length);
}

Status Buffer::Fill(device_size_t byte_offset, device_size_t byte_length,
                    const void* pattern, device_size_t pattern_length) {
  // If not host visible we'll need to issue command buffers.
  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(MemoryType::kHostVisible));
  IREE_RETURN_IF_ERROR(ValidateAccess(MemoryAccess::kWrite));
  IREE_RETURN_IF_ERROR(ValidateUsage(BufferUsage::kMapping));
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

Status Buffer::ReadData(device_size_t source_offset, void* data,
                        device_size_t data_length) {
  // If not host visible we'll need to issue command buffers.
  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(MemoryType::kHostVisible));
  IREE_RETURN_IF_ERROR(ValidateAccess(MemoryAccess::kRead));
  IREE_RETURN_IF_ERROR(ValidateUsage(BufferUsage::kMapping));
  IREE_RETURN_IF_ERROR(
      CalculateRange(source_offset, data_length, &source_offset));
  if (data_length == 0) {
    return OkStatus();  // No-op.
  }
  return ReadDataImpl(source_offset, data, data_length);
}

Status Buffer::WriteData(device_size_t target_offset, const void* data,
                         device_size_t data_length) {
  // If not host visible we'll need to issue command buffers.
  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(MemoryType::kHostVisible));
  IREE_RETURN_IF_ERROR(ValidateAccess(MemoryAccess::kWrite));
  IREE_RETURN_IF_ERROR(ValidateUsage(BufferUsage::kMapping));
  IREE_RETURN_IF_ERROR(
      CalculateRange(target_offset, data_length, &target_offset));
  if (data_length == 0) {
    return OkStatus();  // No-op.
  }
  return WriteDataImpl(target_offset, data, data_length);
}

Status Buffer::CopyData(device_size_t target_offset, Buffer* source_buffer,
                        device_size_t source_offset,
                        device_size_t data_length) {
  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(MemoryType::kHostVisible));
  IREE_RETURN_IF_ERROR(ValidateAccess(MemoryAccess::kWrite));
  IREE_RETURN_IF_ERROR(ValidateUsage(BufferUsage::kMapping));
  IREE_RETURN_IF_ERROR(
      source_buffer->ValidateCompatibleMemoryType(MemoryType::kHostVisible));
  IREE_RETURN_IF_ERROR(source_buffer->ValidateAccess(MemoryAccess::kRead));
  IREE_RETURN_IF_ERROR(source_buffer->ValidateUsage(BufferUsage::kMapping));

  // We need to validate both buffers.
  device_size_t source_data_length = data_length;
  device_size_t target_data_length = data_length;
  device_size_t adjusted_source_offset;
  IREE_RETURN_IF_ERROR(source_buffer->CalculateRange(
      source_offset, source_data_length, &adjusted_source_offset,
      &source_data_length));
  IREE_RETURN_IF_ERROR(CalculateRange(target_offset, target_data_length,
                                      &target_offset, &target_data_length));
  device_size_t adjusted_data_length;
  if (data_length == kWholeBuffer) {
    // Whole buffer copy requested - that could mean either, so take the min.
    adjusted_data_length = std::min(source_data_length, target_data_length);
  } else {
    // Specific length requested - validate that we have matching lengths.
    CHECK_EQ(source_data_length, target_data_length);
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
                         MemoryAccessBitfield memory_access,
                         device_size_t* byte_offset, device_size_t* byte_length,
                         void** out_data) {
  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(MemoryType::kHostVisible));
  IREE_RETURN_IF_ERROR(ValidateAccess(memory_access));
  IREE_RETURN_IF_ERROR(ValidateUsage(BufferUsage::kMapping));
  IREE_RETURN_IF_ERROR(
      CalculateRange(*byte_offset, *byte_length, byte_offset, byte_length));
  *out_data = nullptr;
  return MapMemoryImpl(mapping_mode, memory_access, *byte_offset, *byte_length,
                       out_data);
}

Status Buffer::UnmapMemory(device_size_t local_byte_offset,
                           device_size_t local_byte_length, void* data) {
  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(MemoryType::kHostVisible));
  IREE_RETURN_IF_ERROR(ValidateUsage(BufferUsage::kMapping));
  // NOTE: local_byte_offset/local_byte_length are already adjusted.
  return UnmapMemoryImpl(local_byte_offset, local_byte_length, data);
}

Status Buffer::InvalidateMappedMemory(device_size_t local_byte_offset,
                                      device_size_t local_byte_length) {
  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(MemoryType::kHostVisible));
  if (AnyBitSet(memory_type_ & MemoryType::kHostCoherent)) {
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "Buffer memory type is coherent and invalidation is not required";
  }
  IREE_RETURN_IF_ERROR(ValidateUsage(BufferUsage::kMapping));
  // NOTE: local_byte_offset/local_byte_length are already adjusted.
  return InvalidateMappedMemoryImpl(local_byte_offset, local_byte_length);
}

Status Buffer::FlushMappedMemory(device_size_t local_byte_offset,
                                 device_size_t local_byte_length) {
  IREE_RETURN_IF_ERROR(ValidateCompatibleMemoryType(MemoryType::kHostVisible |
                                                    MemoryType::kHostCached));
  IREE_RETURN_IF_ERROR(ValidateUsage(BufferUsage::kMapping));
  // NOTE: local_byte_offset/local_byte_length are already adjusted.
  return FlushMappedMemoryImpl(local_byte_offset, local_byte_length);
}

}  // namespace hal
}  // namespace iree
