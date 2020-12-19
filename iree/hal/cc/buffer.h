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

// Allocated memory buffer wrapper type and utilities.
//
// Buffers are the basic unit of memory used by the inference system. They may
// be allocated such that they are accessible from the host (normal C++ code
// running on the main CPU), a particular device (such as an accelerator) or
// family of devices, or from some mix of all of those.
//
// The type of memory a buffer is allocated within has implications on it's
// performance and lifetime. For example if an application attempts to use a
// host-allocated buffer (IREE_HAL_MEMORY_TYPE_HOST_LOCAL) on an accelerator
// with discrete memory the accelerator may either be unable to access the
// memory or take a non-trivial performance hit when attempting to do so
// (involving setting up kernel mappings, doing DMA transfers, etc). Likewise,
// trying to access a device-allocated buffer
// (IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) may incur similar overhead or not be
// possible at all. This may be due to restrictions in the memory visibility,
// address spaces, mixed endianness or pointer widths, and other weirdness.
//
// The memory types (defined by a bitfield of MemoryType values) that a
// particular context (host or device) may use vary from device to device and
// must be queried by the application when allocating buffers. It's strongly
// recommended that the most specific memory type be set as possible. For
// example allocating a buffer with IREE_HAL_MEMORY_TYPE_HOST_COHERENT even when
// it will never be used in a way that requires coherency may occupy address
// space reservations or memory mapping that would otherwise not be needed.
//
// As buffers may sometimes not be accessible from the host the base Buffer type
// does not allow for direct void* access and instead buffers must be either
// manipulated using utility functions (such as ReadData or WriteData) or by
// mapping them into a host-accessible address space via MapMemory. Buffer must
// be unmapped before any command may use it.
//
// Buffers may map (roughly) 1:1 with an allocation either from the host heap or
// a device. Buffer::Subspan can be used to reference subspans of buffers like
// absl::Span - though unlike absl::Span the returned Buffer holds a reference
// to the parent buffer.

#ifndef IREE_HAL_CC_BUFFER_H_
#define IREE_HAL_CC_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "iree/base/logging.h"
#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/hal/cc/resource.h"

namespace iree {
namespace hal {

class Allocator;
template <typename T>
class MappedMemory;

std::string MemoryTypeString(iree_hal_memory_type_t memory_type);
std::string MemoryAccessString(iree_hal_memory_access_t memory_access);
std::string BufferUsageString(iree_hal_buffer_usage_t buffer_usage);

// A memory buffer.
// Buffers have a specific memory_type that is used to describe the capabilities
// and behavior of the backing memory of the buffer. Buffers may be any mix of
// host-accessible, host-coherent, or device-accessible for various usages.
// Depending on these memory types the buffers may be mapped for access on the
// host as memory though certain restrictions may be imposed.
//
// See MemoryType for more information about the types and what operations they
// support.
class Buffer : public Resource {
 public:
  // Returns a reference to a subspan of the buffer.
  // If |byte_length| is IREE_WHOLE_BUFFER the remaining bytes in the buffer
  // after |byte_offset| (possibly 0) will be selected.
  //
  // The parent buffer will remain alive for the lifetime of the subspan
  // returned. If the subspan is a small portion this may cause additional
  // memory to remain allocated longer than required.
  //
  // Returns the given |buffer| if the requested span covers the entire range.
  static StatusOr<ref_ptr<Buffer>> Subspan(const ref_ptr<Buffer>& buffer,
                                           iree_device_size_t byte_offset,
                                           iree_device_size_t byte_length);

  // Overlap test results.
  enum class Overlap {
    // No overlap between the two buffers.
    kDisjoint,
    // Partial overlap between the two buffers.
    kPartial,
    // Complete overlap between the two buffers (they are the same).
    kComplete,
  };

  // Tests whether the given buffers overlap, including support for subspans.
  // IREE_WHOLE_BUFFER may be used for |lhs_length| and/or |rhs_length| to use
  // the lengths of those buffers, respectively.
  static Overlap TestOverlap(Buffer* lhs_buffer, iree_device_size_t lhs_offset,
                             iree_device_size_t lhs_length, Buffer* rhs_buffer,
                             iree_device_size_t rhs_offset,
                             iree_device_size_t rhs_length);

  // Returns true if the two buffer ranges overlap at all.
  static bool DoesOverlap(Buffer* lhs_buffer, iree_device_size_t lhs_offset,
                          iree_device_size_t lhs_length, Buffer* rhs_buffer,
                          iree_device_size_t rhs_offset,
                          iree_device_size_t rhs_length);

  // Disallow copies (as copying requires real work).
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer() override = default;

  absl::string_view debug_name() const { return ""; }
  void set_debug_name(std::string debug_name) {}

  // Memory allocator this buffer was allocated from.
  // May be nullptr if the buffer has no particular allocator and should be
  // assumed to be allocated from the host heap.
  constexpr Allocator* allocator() const {
    return allocated_buffer_ == this ? allocator_
                                     : allocated_buffer_->allocator();
  }

  // Memory type this buffer is allocated from.
  iree_hal_memory_type_t memory_type() const { return memory_type_; }

  // Memory access operations allowed on the buffer.
  iree_hal_memory_access_t allowed_access() const { return allowed_access_; }

  // Bitfield describing how the buffer is to be used.
  iree_hal_buffer_usage_t usage() const { return usage_; }

  // Returns the underlying buffer that represents the allocated memory for the
  // Buffer. In most cases this is the buffer itself but for buffer subspan
  // references it will point to the parent buffer.
  Buffer* allocated_buffer() const noexcept;

  // Size of the resource memory allocation in bytes.
  // This may be rounded up from the originally requested size or the ideal
  // size for the resource based on device restrictions.
  constexpr iree_device_size_t allocation_size() const {
    return allocated_buffer_ == this ? allocation_size_
                                     : allocated_buffer_->allocation_size();
  }

  // Range within the underlying allocation this buffer occupies.
  // For buffers that map 1:1 with an allocation this should be
  // [0, allocation_size()), however may still differ if the allocation needed
  // to be aligned.
  //
  // The offset is most often manipulated by Subspan, however it's important to
  // note that the offset may not be what was passed to Subspan as it refers to
  // the offset in the original ancestor buffer, not the buffer from which the
  // subspan was taken.
  constexpr iree_device_size_t byte_offset() const noexcept {
    return byte_offset_;
  }
  constexpr iree_device_size_t byte_length() const noexcept {
    return byte_length_;
  }

  // TODO(benvanik): add debug_name.

  // Returns a longer debug string describing the buffer and its attributes.
  std::string DebugString() const override;
  // Returns a short debug string describing the buffer.
  std::string DebugStringShort() const override;

  // Sets a range of the buffer to the given value.
  // This requires that the resource was allocated with
  // IREE_HAL_MEMORY_TYPE_HOST_VISIBLE and IREE_HAL_BUFFER_USAGE_MAPPING.
  // If |byte_length| is IREE_WHOLE_BUFFER the remaining bytes in the buffer
  // after |byte_offset| (possibly 0) will be filled.
  //
  // The |byte_offset| and |byte_length| must be aligned to the size of the fill
  // value. Multi-byte values will be written in host order for host buffers and
  // device order for device buffers.
  //
  // Only |pattern_length| values with 1, 2, or 4 bytes are supported.
  //
  // Fails if the write could not be performed; either the bounds are out of
  // range or the memory type does not support writing in this way.
  Status Fill(iree_device_size_t byte_offset, iree_device_size_t byte_length,
              const void* pattern, iree_device_size_t pattern_length);
  template <typename T>
  Status Fill8(iree_device_size_t byte_offset, iree_device_size_t byte_length,
               T value);
  template <typename T>
  Status Fill16(iree_device_size_t byte_offset, iree_device_size_t byte_length,
                T value);
  template <typename T>
  Status Fill32(iree_device_size_t byte_offset, iree_device_size_t byte_length,
                T value);
  template <typename T>
  Status Fill8(T value);
  template <typename T>
  Status Fill16(T value);
  template <typename T>
  Status Fill32(T value);

  // Reads a block of byte data from the resource at the given offset.
  // This requires that the resource was allocated with
  // IREE_HAL_MEMORY_TYPE_HOST_VISIBLE and IREE_HAL_BUFFER_USAGE_MAPPING.
  //
  // Fails if the read could not be performed; either the bounds are out of
  // range or the memory type does not support reading in this way.
  Status ReadData(iree_device_size_t source_offset, void* data,
                  iree_device_size_t data_length);

  // Writes a block of byte data into the resource at the given offset.
  // This requires that the resource was allocated with
  // IREE_HAL_MEMORY_TYPE_HOST_VISIBLE and IREE_HAL_BUFFER_USAGE_MAPPING.
  //
  // Fails if the write could not be performed; either the bounds are out of
  // range or the memory type does not support writing in this way.
  Status WriteData(iree_device_size_t target_offset, const void* data,
                   iree_device_size_t data_length);

  // Copies data from the provided source_buffer into the buffer.
  // This requires that the resource was allocated with
  // IREE_HAL_MEMORY_TYPE_HOST_VISIBLE and IREE_HAL_BUFFER_USAGE_MAPPING.
  // The source and destination may be the same buffer but the ranges must not
  // overlap (a la memcpy).
  //
  // Fails if the write could not be performed; either the bounds are out of
  // range or the memory type does not support writing in this way.
  Status CopyData(iree_device_size_t target_offset, Buffer* source_buffer,
                  iree_device_size_t source_offset,
                  iree_device_size_t data_length);
  Status CopyData(iree_device_size_t target_offset, Buffer* source_buffer) {
    return CopyData(target_offset, source_buffer, 0, IREE_WHOLE_BUFFER);
  }

  // Maps the resource memory for direct access from the host.
  // This requires that the resource was allocated with
  // IREE_HAL_MEMORY_TYPE_HOST_VISIBLE and IREE_HAL_BUFFER_USAGE_MAPPING.
  //
  // If IREE_HAL_MEMORY_TYPE_HOST_COHERENT was not specified then explicit
  // Invalidate and Flush calls must be used to control visibility of the data
  // on the device. If IREE_HAL_MEMORY_TYPE_HOST_CACHED is not set callers must
  // not attempt to read from the mapped memory as doing so may produce
  // undefined results and/or ultra slow reads.
  //
  // If the IREE_HAL_MEMORY_ACCESS_DISCARD bit is set when mapping for writes
  // the caller guarantees that they will be overwriting all data in the mapped
  // range. This is used as a hint to the device that the prior contents are no
  // longer required and can enable optimizations that save on synchronization
  // and readback. Note however that it is strictly a hint and the contents are
  // not guaranteed to be zeroed during mapping.
  //
  // This allows mapping the memory as a C++ type. Care must be taken to ensure
  // the data layout in C++ matches the expected data layout in the executables
  // that consume this data. For simple primitives like uint8_t or float this is
  // usually not a problem however struct packing may have many restrictions.
  //
  // The returned mapping should be unmapped when it is no longer required.
  // Unmapping does not implicitly flush.
  //
  // Fails if the memory could not be mapped due to mapping exhaustion, invalid
  // arguments, or unsupported memory types.
  //
  // Example:
  //  IREE_ASSIGN_OR_RETURN(auto mapping, buffer->MapForRead<MyStruct>());
  //  mapping[5].foo = 3;
  //  std::memcpy(mapping.data(), source_data, mapping.size());
  //  mapping.reset();
  template <typename T>
  StatusOr<MappedMemory<T>> MapMemory(
      iree_hal_memory_access_t memory_access,
      iree_device_size_t element_offset = 0,
      iree_device_size_t element_length = IREE_WHOLE_BUFFER);

 protected:
  template <typename T>
  friend class MappedMemory;

  // Defines the mode of a MapMemory operation.
  enum class MappingMode {
    // The call to MapMemory will always be matched with UnmapMemory.
    kScoped,
  };

  Buffer(Allocator* allocator, iree_hal_memory_type_t memory_type,
         iree_hal_memory_access_t allowed_access, iree_hal_buffer_usage_t usage,
         iree_device_size_t allocation_size, iree_device_size_t byte_offset,
         iree_device_size_t byte_length);

  // Allows subclasses to override the allowed access bits.
  // This should only be done when known safe by the allocation scheme.
  void set_allowed_access(iree_hal_memory_access_t allowed_access) {
    allowed_access_ = allowed_access;
  }

  // Sets a range of the buffer to the given value.
  // State and parameters have already been validated. For the >8bit variants
  // the offset and length have already been validated to be aligned to the
  // natural alignment of the type.
  virtual Status FillImpl(iree_device_size_t byte_offset,
                          iree_device_size_t byte_length, const void* pattern,
                          iree_device_size_t pattern_length) = 0;

  // Reads a block of byte data from the resource at the given offset.
  // State and parameters have already been validated.
  virtual Status ReadDataImpl(iree_device_size_t source_offset, void* data,
                              iree_device_size_t data_length) = 0;

  // Writes a block of byte data into the resource at the given offset.
  // State and parameters have already been validated.
  virtual Status WriteDataImpl(iree_device_size_t target_offset,
                               const void* data,
                               iree_device_size_t data_length) = 0;

  // Copies a block of byte data into the resource at the given offset.
  // State and parameters have already been validated.
  virtual Status CopyDataImpl(iree_device_size_t target_offset,
                              Buffer* source_buffer,
                              iree_device_size_t source_offset,
                              iree_device_size_t data_length) = 0;

  // Maps memory directly.
  // The output data pointer will be properly aligned to the start of the data.
  // |local_byte_offset| and |local_byte_length| are the adjusted values that
  // should map into the local space of the buffer.
  //
  // Fails if the memory could not be mapped (invalid access type, invalid
  // range, or unsupported memory type).
  // State and parameters have already been validated.
  virtual Status MapMemoryImpl(MappingMode mapping_mode,
                               iree_hal_memory_access_t memory_access,
                               iree_device_size_t local_byte_offset,
                               iree_device_size_t local_byte_length,
                               void** out_data) = 0;

  // Unmaps previously mapped memory.
  // No-op if the memory is not mapped. As this is often used in destructors
  // we can't rely on failures here propagating with anything but
  // IREE_CHECK/IREE_DCHECK. State and parameters have already been validated.
  virtual Status UnmapMemoryImpl(iree_device_size_t local_byte_offset,
                                 iree_device_size_t local_byte_length,
                                 void* data) = 0;

  // Invalidates ranges of non-coherent memory from the host caches.
  // Use this before reading from non-coherent memory.
  // This guarantees that device writes to the memory ranges provided are
  // visible on the host.
  // This is only required for memory types without kHostCoherent set.
  // State and parameters have already been validated.
  virtual Status InvalidateMappedMemoryImpl(
      iree_device_size_t local_byte_offset,
      iree_device_size_t local_byte_length) = 0;

  // Flushes ranges of non-coherent memory from the host caches.
  // Use this after writing to non-coherent memory.
  // This guarantees that host writes to the memory ranges provided are made
  // available for device access.
  // This is only required for memory types without kHostCoherent set.
  // State and parameters have already been validated.
  virtual Status FlushMappedMemoryImpl(
      iree_device_size_t local_byte_offset,
      iree_device_size_t local_byte_length) = 0;

  // Validates the given buffer range and adjusts the offset and length if the
  // provided length is IREE_WHOLE_BUFFER or the buffer is offset within its
  // allocation. This calculates the range in the given domain without adjusting
  // to any particular buffer base offsets.
  static Status CalculateLocalRange(iree_device_size_t max_length,
                                    iree_device_size_t offset,
                                    iree_device_size_t length,
                                    iree_device_size_t* out_adjusted_offset,
                                    iree_device_size_t* out_adjusted_length);

 private:
  friend class Allocator;

  // This is not great and deserves cleanup.
  friend class DeferredBuffer;
  friend class SubspanBuffer;
  friend class HeapBuffer;

  // Maps memory directly.
  // The byte offset and byte length may be adjusted for device alignment.
  // The output data pointer will be properly aligned to the start of the data.
  // Fails if the memory could not be mapped (invalid access type, invalid
  // range, or unsupported memory type).
  Status MapMemory(MappingMode mapping_mode,
                   iree_hal_memory_access_t memory_access,
                   iree_device_size_t* byte_offset,
                   iree_device_size_t* byte_length, void** out_data);

  // Unmaps previously mapped memory.
  // No-op if the memory is not mapped. As this is often used in destructors
  // we can't rely on failures here propagating with anything but
  // IREE_CHECK/IREE_DCHECK.
  Status UnmapMemory(iree_device_size_t local_byte_offset,
                     iree_device_size_t local_byte_length, void* data);

  // Invalidates ranges of non-coherent memory from the host caches.
  // Use this before reading from non-coherent memory.
  // This guarantees that device writes to the memory ranges provided are
  // visible on the host.
  // This is only required for memory types without kHostCoherent set.
  Status InvalidateMappedMemory(iree_device_size_t local_byte_offset,
                                iree_device_size_t local_byte_length);

  // Flushes ranges of non-coherent memory from the host caches.
  // Use this after writing to non-coherent memory.
  // This guarantees that host writes to the memory ranges provided are made
  // available for device access.
  // This is only required for memory types without kHostCoherent set.
  Status FlushMappedMemory(iree_device_size_t local_byte_offset,
                           iree_device_size_t local_byte_length);

  // Returns a failure if the memory type the buffer was allocated from is not
  // compatible with the given type.
  Status ValidateCompatibleMemoryType(iree_hal_memory_type_t memory_type) const;
  // Returns a failure if the buffer memory type or usage disallows the given
  // access type.
  Status ValidateAccess(iree_hal_memory_access_t memory_access) const;
  // Returns a failure if the buffer was not allocated for the given usage.
  Status ValidateUsage(iree_hal_buffer_usage_t usage) const;
  // Validates the given buffer range and optionally adjusts the offset and
  // length if the provided length is IREE_WHOLE_BUFFER or the buffer is offset
  // within its allocation.
  static Status CalculateRange(
      iree_device_size_t base_offset, iree_device_size_t max_length,
      iree_device_size_t offset, iree_device_size_t length,
      iree_device_size_t* out_adjusted_offset,
      iree_device_size_t* out_adjusted_length = nullptr);
  Status CalculateRange(
      iree_device_size_t offset, iree_device_size_t length,
      iree_device_size_t* out_adjusted_offset,
      iree_device_size_t* out_adjusted_length = nullptr) const;

  // Points to either this or parent_buffer_.get().
  Buffer* allocated_buffer_ = nullptr;

  Allocator* allocator_ = nullptr;
  iree_hal_memory_type_t memory_type_ = IREE_HAL_MEMORY_TYPE_NONE;
  iree_hal_memory_access_t allowed_access_ = IREE_HAL_MEMORY_ACCESS_NONE;
  iree_hal_buffer_usage_t usage_ = IREE_HAL_BUFFER_USAGE_NONE;

  iree_device_size_t allocation_size_ = 0;
  iree_device_size_t byte_offset_ = 0;
  iree_device_size_t byte_length_ = 0;

  // Defined when this buffer is a subspan of another buffer.
  ref_ptr<Buffer> parent_buffer_;
};

// A memory mapping RAII object.
// The mapping will stay active until it is reset and will retain the buffer.
template <typename T>
class MappedMemory {
 public:
  using unspecified_bool_type = const T* MappedMemory<T>::*;

  MappedMemory() = default;
  MappedMemory(iree_hal_memory_access_t access, ref_ptr<Buffer> buffer,
               iree_device_size_t byte_offset, iree_device_size_t byte_length,
               iree_device_size_t element_size, T* data);

  // Allow moving but disallow copying as the mapping is stateful.
  MappedMemory(MappedMemory&& rhs) noexcept;
  MappedMemory& operator=(MappedMemory&& rhs) noexcept;
  MappedMemory(const MappedMemory&) = delete;
  MappedMemory& operator=(const MappedMemory&) = delete;

  ~MappedMemory();

  // The buffer resource that this mapping references.
  const ref_ptr<Buffer>& buffer() const noexcept { return buffer_; }
  // Offset, in bytes, into the resource allocation.
  // This value is *informative only*, as it may vary from device to device.
  iree_device_size_t byte_offset() const noexcept { return byte_offset_; }
  // Length, in bytes, of the resource mapping.
  // This may be larger than the originally requested length due to alignment.
  // This value is *informative only*, as it may vary from device to device.
  iree_device_size_t byte_length() const noexcept { return byte_length_; }

  // True if the mapping is empty.
  bool empty() const noexcept { return element_size_ == 0; }
  // The size of the mapping as requested in elements.
  size_t size() const noexcept { return static_cast<size_t>(element_size_); }

  // Returns a read-only pointer to the mapped memory.
  // This will be nullptr if the mapping failed or the mapping is not readable.
  const T* data() const noexcept;
  absl::Span<const T> contents() const noexcept { return {data(), size()}; }

  // Returns a mutable pointer to the mapped memory.
  // This will be nullptr if the mapping failed or the mapping is not writable.
  // If the mapping was not made with read access it may still be possible to
  // read from this memory but behavior is undefined.
  T* mutable_data() noexcept;
  absl::Span<T> mutable_contents() noexcept { return {mutable_data(), size()}; }

  // Returns a raw pointer to the mapped data without any access checks.
  T* unsafe_data() const noexcept { return data_; }

  // Equivalent to absl::Span::subspan().
  // May return a 0-length span.
  // Fails if the buffer is not mapped or not mapped for the requested access.
  StatusOr<absl::Span<const T>> Subspan(
      iree_device_size_t element_offset = 0,
      iree_device_size_t element_length = IREE_WHOLE_BUFFER) const noexcept;
  StatusOr<absl::Span<T>> MutableSubspan(
      iree_device_size_t element_offset = 0,
      iree_device_size_t element_length = IREE_WHOLE_BUFFER) noexcept;

  // Accesses an element in the mapped memory.
  // Must be called with a valid index in [0, size()).
  const T& operator[](iree_device_size_t i) const noexcept { return data_[i]; }

  // Invalidates a range of non-coherent elements from the host caches.
  Status Invalidate(
      iree_device_size_t element_offset = 0,
      iree_device_size_t element_length = IREE_WHOLE_BUFFER) const;

  // Flushes a range of non-coherent elements from the host caches.
  Status Flush(iree_device_size_t element_offset = 0,
               iree_device_size_t element_length = IREE_WHOLE_BUFFER);

  // Unmaps the mapped memory.
  // The memory will not be implicitly flushed when unmapping.
  void reset();

 private:
  Status ValidateAccess(iree_hal_memory_access_t memory_access) const;
  Status CalculateDataRange(
      iree_device_size_t element_offset, iree_device_size_t element_length,
      iree_device_size_t* out_adjusted_element_offset,
      iree_device_size_t* out_adjusted_element_length) const;

  iree_hal_memory_access_t access_ = IREE_HAL_MEMORY_ACCESS_NONE;
  ref_ptr<Buffer> buffer_;
  iree_device_size_t byte_offset_ = 0;
  iree_device_size_t byte_length_ = 0;
  iree_device_size_t element_size_ = 0;
  T* data_ = nullptr;
};

// Inline functions and template definitions follow:

template <typename T>
Status Buffer::Fill8(iree_device_size_t byte_offset,
                     iree_device_size_t byte_length, T value) {
  auto sized_value = reinterpret_cast<uint8_t*>(&value);
  return Fill(byte_offset, byte_length, sized_value, sizeof(*sized_value));
}

template <typename T>
Status Buffer::Fill16(iree_device_size_t byte_offset,
                      iree_device_size_t byte_length, T value) {
  auto sized_value = reinterpret_cast<uint16_t*>(&value);
  return Fill(byte_offset, byte_length, sized_value, sizeof(*sized_value));
}

template <typename T>
Status Buffer::Fill32(iree_device_size_t byte_offset,
                      iree_device_size_t byte_length, T value) {
  auto sized_value = reinterpret_cast<uint32_t*>(&value);
  return Fill(byte_offset, byte_length, sized_value, sizeof(*sized_value));
}

template <typename T>
Status Buffer::Fill8(T value) {
  return Fill8(0, IREE_WHOLE_BUFFER, value);
}

template <typename T>
Status Buffer::Fill16(T value) {
  return Fill16(0, IREE_WHOLE_BUFFER, value);
}

template <typename T>
Status Buffer::Fill32(T value) {
  return Fill32(0, IREE_WHOLE_BUFFER, value);
}

template <typename T>
StatusOr<MappedMemory<T>> Buffer::MapMemory(
    iree_hal_memory_access_t memory_access, iree_device_size_t element_offset,
    iree_device_size_t element_length) {
  iree_device_size_t byte_offset = element_offset * sizeof(T);
  iree_device_size_t byte_length = element_length == IREE_WHOLE_BUFFER
                                       ? IREE_WHOLE_BUFFER
                                       : element_length * sizeof(T);
  void* data = nullptr;
  IREE_RETURN_IF_ERROR(MapMemory(MappingMode::kScoped, memory_access,
                                 &byte_offset, &byte_length, &data));
  return MappedMemory<T>{
      memory_access, add_ref(this),           byte_offset,
      byte_length,   byte_length / sizeof(T), static_cast<T*>(data)};
}

template <typename T>
MappedMemory<T>::MappedMemory(iree_hal_memory_access_t access,
                              ref_ptr<Buffer> buffer,
                              iree_device_size_t byte_offset,
                              iree_device_size_t byte_length,
                              iree_device_size_t element_size, T* data)
    : access_(access),
      buffer_(std::move(buffer)),
      byte_offset_(byte_offset),
      byte_length_(byte_length),
      element_size_(element_size),
      data_(data) {}

template <typename T>
MappedMemory<T>::MappedMemory(MappedMemory<T>&& rhs) noexcept
    : access_(rhs.access_),
      buffer_(std::move(rhs.buffer_)),
      byte_offset_(rhs.byte_offset_),
      byte_length_(rhs.byte_length_),
      element_size_(rhs.element_size_),
      data_(rhs.data_) {
  rhs.access_ = IREE_HAL_MEMORY_ACCESS_NONE;
  rhs.buffer_.reset();
  rhs.byte_offset_ = 0;
  rhs.byte_length_ = 0;
  rhs.element_size_ = 0;
  rhs.data_ = nullptr;
}

template <typename T>
MappedMemory<T>& MappedMemory<T>::operator=(MappedMemory<T>&& rhs) noexcept {
  if (this != &rhs) {
    reset();
    access_ = rhs.access_;
    buffer_ = std::move(rhs.buffer_);
    byte_offset_ = rhs.byte_offset_;
    byte_length_ = rhs.byte_length_;
    element_size_ = rhs.element_size_;
    data_ = rhs.data_;

    rhs.access_ = IREE_HAL_MEMORY_ACCESS_NONE;
    rhs.buffer_.reset();
    rhs.byte_offset_ = 0;
    rhs.byte_length_ = 0;
    rhs.element_size_ = 0;
    rhs.data_ = nullptr;
  }
  return *this;
}

template <typename T>
MappedMemory<T>::~MappedMemory() {
  // Unmap (if needed) - note that we can't fail gracefully here :(
  reset();
}

template <typename T>
const T* MappedMemory<T>::data() const noexcept {
  if (!data_ || !iree_any_bit_set(access_, IREE_HAL_MEMORY_ACCESS_READ)) {
    return nullptr;
  }
  return data_;
}

template <typename T>
T* MappedMemory<T>::mutable_data() noexcept {
  if (!data_ || !iree_any_bit_set(access_, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    return nullptr;
  }
  return data_;
}

template <typename T>
Status MappedMemory<T>::ValidateAccess(
    iree_hal_memory_access_t memory_access) const {
  if (!data_) {
    return FailedPreconditionErrorBuilder(IREE_LOC) << "Buffer is not mapped";
  } else if (!iree_any_bit_set(access_, memory_access)) {
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "Buffer is not mapped for the desired access";
  }
  return OkStatus();
}

template <typename T>
Status MappedMemory<T>::CalculateDataRange(
    iree_device_size_t element_offset, iree_device_size_t element_length,
    iree_device_size_t* out_adjusted_element_offset,
    iree_device_size_t* out_adjusted_element_length) const {
  IREE_RETURN_IF_ERROR(Buffer::CalculateLocalRange(
      element_size_ * sizeof(T), element_offset * sizeof(T),
      element_length == IREE_WHOLE_BUFFER ? IREE_WHOLE_BUFFER
                                          : element_length * sizeof(T),
      out_adjusted_element_offset, out_adjusted_element_length));
  *out_adjusted_element_offset /= sizeof(T);
  *out_adjusted_element_length /= sizeof(T);
  return OkStatus();
}

template <typename T>
inline StatusOr<absl::Span<const T>> MappedMemory<T>::Subspan(
    iree_device_size_t element_offset,
    iree_device_size_t element_length) const noexcept {
  IREE_RETURN_IF_ERROR(ValidateAccess(IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(CalculateDataRange(element_offset, element_length,
                                          &element_offset, &element_length));
  return absl::Span<const T>(data_ + element_offset, element_length);
}

template <typename T>
inline StatusOr<absl::Span<T>> MappedMemory<T>::MutableSubspan(
    iree_device_size_t element_offset,
    iree_device_size_t element_length) noexcept {
  IREE_RETURN_IF_ERROR(ValidateAccess(IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(CalculateDataRange(element_offset, element_length,
                                          &element_offset, &element_length));
  return absl::Span<T>(data_ + element_offset, element_length);
}

template <typename T>
Status MappedMemory<T>::Invalidate(iree_device_size_t element_offset,
                                   iree_device_size_t element_length) const {
  IREE_RETURN_IF_ERROR(ValidateAccess(IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(CalculateDataRange(element_offset, element_length,
                                          &element_offset, &element_length));
  if (!element_length) return OkStatus();
  return buffer_->InvalidateMappedMemory(
      byte_offset_ + element_offset * sizeof(T), element_length * sizeof(T));
}

template <typename T>
Status MappedMemory<T>::Flush(iree_device_size_t element_offset,
                              iree_device_size_t element_length) {
  IREE_RETURN_IF_ERROR(ValidateAccess(IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(CalculateDataRange(element_offset, element_length,
                                          &element_offset, &element_length));
  if (!element_length) return OkStatus();
  return buffer_->FlushMappedMemory(byte_offset_ + element_offset * sizeof(T),
                                    element_length * sizeof(T));
}

template <typename T>
void MappedMemory<T>::reset() {
  if (!buffer_) return;
  // TODO(benvanik): better handling of errors? may be fine to always warn.
  buffer_->UnmapMemory(byte_offset_, byte_length_, data_).IgnoreError();
  buffer_.reset();
  access_ = IREE_HAL_MEMORY_ACCESS_NONE;
  byte_offset_ = 0;
  byte_length_ = 0;
  element_size_ = 0;
  data_ = nullptr;
}

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CC_BUFFER_H_
