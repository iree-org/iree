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
// host-allocated buffer (MemoryType::kHostLocal) on an accelerator with
// discrete memory the accelerator may either be unable to access the memory or
// take a non-trivial performance hit when attempting to do so (involving
// setting up kernel mappings, doing DMA transfers, etc). Likewise, trying to
// access a device-allocated buffer (MemoryType::kDeviceLocal) may incur similar
// overhead or not be possible at all. This may be due to restrictions in the
// memory visibility, address spaces, mixed endianness or pointer widths,
// and other weirdness.
//
// The memory types (defined by a bitfield of MemoryType values) that a
// particular context (host or device) may use vary from device to device and
// must be queried by the application when allocating buffers. It's strongly
// recommended that the most specific memory type be set as possible. For
// example allocating a buffer with MemoryType::kHostCoherent even when it will
// never be used in a way that requires coherency may occupy address space
// reservations or memory mapping that would otherwise not be needed.
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

#ifndef IREE_HAL_BUFFER_H_
#define IREE_HAL_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "iree/base/bitfield.h"
#include "iree/base/logging.h"
#include "iree/base/status.h"
#include "iree/hal/resource.h"

// Only enable debug names in non-opt modes (unless the user forces it on).
#if !defined(NDEBUG) && !defined(HAS_IREE_BUFFER_DEBUG_NAME)
#define HAS_IREE_BUFFER_DEBUG_NAME 1
#endif  // !NDEBUG

namespace iree {

// std::size_t equivalent that is the size as used on device.
// As the device may have a larger memory address space than the host we treat
// all byte offsets as this type instead of the host-specified size_t.
using device_size_t = uint64_t;

// When used as a length value in functions causes the length to be the entire
// remaining buffer from the specified offset.
constexpr device_size_t kWholeBuffer = ~0ull;

}  // namespace iree

namespace iree {
namespace hal {

class Allocator;
template <typename T>
class MappedMemory;

// A bitfield specifying properties for a memory type.
enum class MemoryType : uint32_t {
  kNone = 0,

  // Memory is lazily allocated by the device and only exists transiently.
  // This is the optimal mode for memory used only within a single command
  // buffer. Transient buffers, even if they have kHostVisible set, should be
  // treated as device-local and opaque as they may have no memory attached to
  // them outside of the time they are being evaluated on devices.
  //
  // This flag can be treated as a hint in most cases; allocating a buffer with
  // it set _may_ return the same as if it had not be set. Certain allocation
  // routines may use the hint to more tightly control reuse or defer wiring the
  // memory.
  kTransient = 1 << 0,

  // Memory allocated with this type can be mapped for host access using
  // Buffer::MapMemory.
  kHostVisible = 1 << 1,

  // The host cache management commands MappedMemory::Flush and
  // MappedMemory::Invalidate are not needed to flush host writes
  // to the device or make device writes visible to the host, respectively.
  kHostCoherent = 1 << 2,

  // Memory allocated with this type is cached on the host. Host memory
  // accesses to uncached memory are slower than to cached memory, however
  // uncached memory is always host coherent. MappedMemory::Flush must be used
  // to ensure the device has visibility into any changes made on the host and
  // Invalidate must be used to ensure the host has visibility into any changes
  // made on the device.
  kHostCached = 1 << 3,

  // Memory is accessible as normal host allocated memory.
  kHostLocal = kHostVisible | kHostCoherent,

  // Memory allocated with this type is visible to the device for execution.
  // Being device visible does not mean the same thing as kDeviceLocal. Though
  // an allocation may be visible to the device and therefore useable for
  // execution it may require expensive mapping or implicit transfers.
  kDeviceVisible = 1 << 4,

  // Memory allocated with this type is the most efficient for device access.
  // Devices may support using memory that is not device local via
  // kDeviceVisible but doing so can incur non-trivial performance penalties.
  // Device local memory, on the other hand, is guaranteed to be fast for all
  // operations.
  kDeviceLocal = kDeviceVisible | (1 << 5),
};
IREE_BITFIELD(MemoryType);
using MemoryTypeBitfield = MemoryType;
std::string MemoryTypeString(MemoryTypeBitfield memory_type);

// A bitfield specifying how memory will be accessed in a mapped memory region.
enum class MemoryAccess : uint32_t {
  // Memory is not mapped.
  kNone = 0,

  // Memory will be read.
  // If a buffer is only mapped for reading it may still be possible to write to
  // it but the results will be undefined (as it may present coherency issues).
  kRead = 1 << 0,

  // Memory will be written.
  // If a buffer is only mapped for writing it may still be possible to read
  // from it but the results will be undefined or incredibly slow (as it may
  // be mapped by the driver as uncached).
  kWrite = 1 << 1,

  // Memory will be discarded prior to mapping.
  // The existing contents will be undefined after mapping and must be written
  // to ensure validity.
  kDiscard = 1 << 2,

  // Memory will be discarded and completely overwritten in a single operation.
  kDiscardWrite = kWrite | kDiscard,

  // A flag that can be applied to any access type to indicate that the buffer
  // storage being accessed may alias with other accesses occurring concurrently
  // within or across operations. The lack of the flag indicates that the access
  // is guaranteed not to alias (ala C's `restrict` keyword).
  kMayAlias = 1 << 3,

  // Memory may have any operation performed on it.
  kAll = kRead | kWrite | kDiscard,
};
IREE_BITFIELD(MemoryAccess);
using MemoryAccessBitfield = MemoryAccess;
std::string MemoryAccessString(MemoryAccessBitfield memory_access);

// Bitfield that defines how a buffer is intended to be used.
// Usage allows the driver to appropriately place the buffer for more
// efficient operations of the specified types.
enum class BufferUsage {
  kNone = 0,

  // The buffer, once defined, will not be mapped or updated again.
  // This should be used for uniform parameter values such as runtime
  // constants for executables. Doing so may allow drivers to inline values or
  // represent them in command buffers more efficiently (avoiding memory reads
  // or swapping, etc).
  kConstant = 1 << 0,

  // The buffer can be used as the source or target of a transfer command
  // (CopyBuffer, UpdateBuffer, etc).
  //
  // If |kMapping| is not specified drivers may safely assume that the host
  // may never need visibility of this buffer as all accesses will happen via
  // command buffers.
  kTransfer = 1 << 1,

  // The buffer can be mapped by the host application for reading and writing.
  //
  // As mapping may require placement in special address ranges or system
  // calls to enable visibility the driver can use the presence (or lack of)
  // this flag to perform allocation-type setup and avoid initial mapping
  // overhead.
  kMapping = 1 << 2,

  // The buffer can be provided as an input or output to an executable.
  // Buffers of this type may be directly used by drivers during dispatch.
  kDispatch = 1 << 3,

  // Buffer may be used for any operation.
  kAll = kTransfer | kMapping | kDispatch,
};
IREE_BITFIELD(BufferUsage);
using BufferUsageBitfield = BufferUsage;
std::string BufferUsageString(BufferUsageBitfield buffer_usage);

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
  // If |byte_length| is kWholeBuffer the remaining bytes in the buffer after
  // |byte_offset| (possibly 0) will be selected.
  //
  // The parent buffer will remain alive for the lifetime of the subspan
  // returned. If the subspan is a small portion this may cause additional
  // memory to remain allocated longer than required.
  //
  // Returns the given |buffer| if the requested span covers the entire range.
  static StatusOr<ref_ptr<Buffer>> Subspan(const ref_ptr<Buffer>& buffer,
                                           device_size_t byte_offset,
                                           device_size_t byte_length);

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
  // kWholeBuffer may be used for |lhs_length| and/or |rhs_length| to use the
  // lengths of those buffers, respectively.
  static Overlap TestOverlap(Buffer* lhs_buffer, device_size_t lhs_offset,
                             device_size_t lhs_length, Buffer* rhs_buffer,
                             device_size_t rhs_offset,
                             device_size_t rhs_length);

  // Returns true if the two buffer ranges overlap at all.
  static bool DoesOverlap(Buffer* lhs_buffer, device_size_t lhs_offset,
                          device_size_t lhs_length, Buffer* rhs_buffer,
                          device_size_t rhs_offset, device_size_t rhs_length);

  // Disallow copies (as copying requires real work).
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer() override = default;

#if HAS_IREE_BUFFER_DEBUG_NAME
  // Optionally populated name useful for logging a persistent name for the
  // buffer.
  absl::string_view debug_name() const { return debug_name_; }
  void set_debug_name(std::string debug_name) {
    debug_name_ = std::move(debug_name);
  }
#else
  absl::string_view debug_name() const { return ""; }
  void set_debug_name(std::string debug_name) {}
#endif  // HAS_IREE_BUFFER_DEBUG_NAME

  // Memory allocator this buffer was allocated from.
  // May be nullptr if the buffer has no particular allocator and should be
  // assumed to be allocated from the host heap.
  constexpr Allocator* allocator() const {
    return allocated_buffer_ == this ? allocator_
                                     : allocated_buffer_->allocator();
  }

  // Memory type this buffer is allocated from.
  MemoryTypeBitfield memory_type() const { return memory_type_; }

  // Memory access operations allowed on the buffer.
  MemoryAccessBitfield allowed_access() const { return allowed_access_; }

  // Bitfield describing how the buffer is to be used.
  BufferUsageBitfield usage() const { return usage_; }

  // Returns the underlying buffer that represents the allocated memory for the
  // Buffer. In most cases this is the buffer itself but for buffer subspan
  // references it will point to the parent buffer.
  Buffer* allocated_buffer() const noexcept;

  // Size of the resource memory allocation in bytes.
  // This may be rounded up from the originally requested size or the ideal
  // size for the resource based on device restrictions.
  constexpr device_size_t allocation_size() const {
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
  constexpr device_size_t byte_offset() const noexcept { return byte_offset_; }
  constexpr device_size_t byte_length() const noexcept { return byte_length_; }

  // TODO(benvanik): add debug_name.

  // Returns a longer debug string describing the buffer and its attributes.
  std::string DebugString() const override;
  // Returns a short debug string describing the buffer.
  std::string DebugStringShort() const override;

  // Sets a range of the buffer to the given value.
  // This requires that the resource was allocated with
  // MemoryType::kHostVisible and BufferUsage::kMapping.
  // If |byte_length| is kWholeBuffer the remaining bytes in the buffer after
  // |byte_offset| (possibly 0) will be filled.
  //
  // The |byte_offset| and |byte_length| must be aligned to the size of the fill
  // value. Multi-byte values will be written in host order for host buffers and
  // device order for device buffers.
  //
  // Only |pattern_length| values with 1, 2, or 4 bytes are supported.
  //
  // Fails if the write could not be performed; either the bounds are out of
  // range or the memory type does not support writing in this way.
  Status Fill(device_size_t byte_offset, device_size_t byte_length,
              const void* pattern, device_size_t pattern_length);
  template <typename T>
  Status Fill8(device_size_t byte_offset, device_size_t byte_length, T value);
  template <typename T>
  Status Fill16(device_size_t byte_offset, device_size_t byte_length, T value);
  template <typename T>
  Status Fill32(device_size_t byte_offset, device_size_t byte_length, T value);
  template <typename T>
  Status Fill8(T value);
  template <typename T>
  Status Fill16(T value);
  template <typename T>
  Status Fill32(T value);

  // Reads a block of byte data from the resource at the given offset.
  // This requires that the resource was allocated with
  // MemoryType::kHostVisible and BufferUsage::kMapping.
  //
  // Fails if the read could not be performed; either the bounds are out of
  // range or the memory type does not support reading in this way.
  Status ReadData(device_size_t source_offset, void* data,
                  device_size_t data_length);

  // Writes a block of byte data into the resource at the given offset.
  // This requires that the resource was allocated with
  // MemoryType::kHostVisible and BufferUsage::kMapping.
  //
  // Fails if the write could not be performed; either the bounds are out of
  // range or the memory type does not support writing in this way.
  Status WriteData(device_size_t target_offset, const void* data,
                   device_size_t data_length);

  // Copies data from the provided source_buffer into the buffer.
  // This requires that the resource was allocated with
  // MemoryType::kHostVisible and BufferUsage::kMapping.
  // The source and destination may be the same buffer but the ranges must not
  // overlap (a la memcpy).
  //
  // Fails if the write could not be performed; either the bounds are out of
  // range or the memory type does not support writing in this way.
  Status CopyData(device_size_t target_offset, Buffer* source_buffer,
                  device_size_t source_offset, device_size_t data_length);
  Status CopyData(device_size_t target_offset, Buffer* source_buffer) {
    return CopyData(target_offset, source_buffer, 0, kWholeBuffer);
  }

  // Maps the resource memory for direct access from the host.
  // This requires that the resource was allocated with
  // MemoryType::kHostVisible and BufferUsage::kMapping.
  //
  // If MemoryType::kHostCoherent was not specified then explicit
  // Invalidate and Flush calls must be used to control visibility of the data
  // on the device. If MemoryType::kHostCached is not set callers must not
  // attempt to read from the mapped memory as doing so may produce undefined
  // results and/or ultra slow reads.
  //
  // If the MemoryAccess::kDiscard bit is set when mapping for writes the caller
  // guarantees that they will be overwriting all data in the mapped range. This
  // is used as a hint to the device that the prior contents are no longer
  // required and can enable optimizations that save on synchronization and
  // readback. Note however that it is strictly a hint and the contents are not
  // guaranteed to be zeroed during mapping.
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
      MemoryAccessBitfield memory_access, device_size_t element_offset = 0,
      device_size_t element_length = kWholeBuffer);

 protected:
  template <typename T>
  friend class MappedMemory;

  // Defines the mode of a MapMemory operation.
  enum class MappingMode {
    // The call to MapMemory will always be matched with UnmapMemory.
    kScoped,
  };

  Buffer(Allocator* allocator, MemoryTypeBitfield memory_type,
         MemoryAccessBitfield allowed_access, BufferUsageBitfield usage,
         device_size_t allocation_size, device_size_t byte_offset,
         device_size_t byte_length);

  // Allows subclasses to override the allowed access bits.
  // This should only be done when known safe by the allocation scheme.
  void set_allowed_access(MemoryAccessBitfield allowed_access) {
    allowed_access_ = allowed_access;
  }

  // Sets a range of the buffer to the given value.
  // State and parameters have already been validated. For the >8bit variants
  // the offset and length have already been validated to be aligned to the
  // natural alignment of the type.
  virtual Status FillImpl(device_size_t byte_offset, device_size_t byte_length,
                          const void* pattern,
                          device_size_t pattern_length) = 0;

  // Reads a block of byte data from the resource at the given offset.
  // State and parameters have already been validated.
  virtual Status ReadDataImpl(device_size_t source_offset, void* data,
                              device_size_t data_length) = 0;

  // Writes a block of byte data into the resource at the given offset.
  // State and parameters have already been validated.
  virtual Status WriteDataImpl(device_size_t target_offset, const void* data,
                               device_size_t data_length) = 0;

  // Copies a block of byte data into the resource at the given offset.
  // State and parameters have already been validated.
  virtual Status CopyDataImpl(device_size_t target_offset,
                              Buffer* source_buffer,
                              device_size_t source_offset,
                              device_size_t data_length) = 0;

  // Maps memory directly.
  // The output data pointer will be properly aligned to the start of the data.
  // |local_byte_offset| and |local_byte_length| are the adjusted values that
  // should map into the local space of the buffer.
  //
  // Fails if the memory could not be mapped (invalid access type, invalid
  // range, or unsupported memory type).
  // State and parameters have already been validated.
  virtual Status MapMemoryImpl(MappingMode mapping_mode,
                               MemoryAccessBitfield memory_access,
                               device_size_t local_byte_offset,
                               device_size_t local_byte_length,
                               void** out_data) = 0;

  // Unmaps previously mapped memory.
  // No-op if the memory is not mapped. As this is often used in destructors
  // we can't rely on failures here propagating with anything but CHECK/DCHECK.
  // State and parameters have already been validated.
  virtual Status UnmapMemoryImpl(device_size_t local_byte_offset,
                                 device_size_t local_byte_length,
                                 void* data) = 0;

  // Invalidates ranges of non-coherent memory from the host caches.
  // Use this before reading from non-coherent memory.
  // This guarantees that device writes to the memory ranges provided are
  // visible on the host.
  // This is only required for memory types without kHostCoherent set.
  // State and parameters have already been validated.
  virtual Status InvalidateMappedMemoryImpl(
      device_size_t local_byte_offset, device_size_t local_byte_length) = 0;

  // Flushes ranges of non-coherent memory from the host caches.
  // Use this after writing to non-coherent memory.
  // This guarantees that host writes to the memory ranges provided are made
  // available for device access.
  // This is only required for memory types without kHostCoherent set.
  // State and parameters have already been validated.
  virtual Status FlushMappedMemoryImpl(device_size_t local_byte_offset,
                                       device_size_t local_byte_length) = 0;

  // Validates the given buffer range and adjusts the offset and length if the
  // provided length is kWholeBuffer or the buffer is offset within its
  // allocation. This calculates the range in the given domain without adjusting
  // to any particular buffer base offsets.
  static Status CalculateLocalRange(device_size_t max_length,
                                    device_size_t offset, device_size_t length,
                                    device_size_t* out_adjusted_offset,
                                    device_size_t* out_adjusted_length);

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
  Status MapMemory(MappingMode mapping_mode, MemoryAccessBitfield memory_access,
                   device_size_t* byte_offset, device_size_t* byte_length,
                   void** out_data);

  // Unmaps previously mapped memory.
  // No-op if the memory is not mapped. As this is often used in destructors
  // we can't rely on failures here propagating with anything but CHECK/DCHECK.
  Status UnmapMemory(device_size_t local_byte_offset,
                     device_size_t local_byte_length, void* data);

  // Invalidates ranges of non-coherent memory from the host caches.
  // Use this before reading from non-coherent memory.
  // This guarantees that device writes to the memory ranges provided are
  // visible on the host.
  // This is only required for memory types without kHostCoherent set.
  Status InvalidateMappedMemory(device_size_t local_byte_offset,
                                device_size_t local_byte_length);

  // Flushes ranges of non-coherent memory from the host caches.
  // Use this after writing to non-coherent memory.
  // This guarantees that host writes to the memory ranges provided are made
  // available for device access.
  // This is only required for memory types without kHostCoherent set.
  Status FlushMappedMemory(device_size_t local_byte_offset,
                           device_size_t local_byte_length);

  // Returns a failure if the memory type the buffer was allocated from is not
  // compatible with the given type.
  Status ValidateCompatibleMemoryType(MemoryTypeBitfield memory_type) const;
  // Returns a failure if the buffer memory type or usage disallows the given
  // access type.
  Status ValidateAccess(MemoryAccessBitfield memory_access) const;
  // Returns a failure if the buffer was not allocated for the given usage.
  Status ValidateUsage(BufferUsageBitfield usage) const;
  // Validates the given buffer range and optionally adjusts the offset and
  // length if the provided length is kWholeBuffer or the buffer is offset
  // within its allocation.
  static Status CalculateRange(device_size_t base_offset,
                               device_size_t max_length, device_size_t offset,
                               device_size_t length,
                               device_size_t* out_adjusted_offset,
                               device_size_t* out_adjusted_length = nullptr);
  Status CalculateRange(device_size_t offset, device_size_t length,
                        device_size_t* out_adjusted_offset,
                        device_size_t* out_adjusted_length = nullptr) const;

  // Points to either this or parent_buffer_.get().
  Buffer* allocated_buffer_ = nullptr;

  Allocator* allocator_ = nullptr;
  MemoryTypeBitfield memory_type_ = MemoryType::kNone;
  MemoryAccessBitfield allowed_access_ = MemoryAccess::kNone;
  BufferUsageBitfield usage_ = BufferUsage::kNone;

  device_size_t allocation_size_ = 0;
  device_size_t byte_offset_ = 0;
  device_size_t byte_length_ = 0;

#if HAS_IREE_BUFFER_DEBUG_NAME
  // Friendly name for the buffer used in DebugString. May be set by the app or
  // auto generated.
  std::string debug_name_;
#endif  // HAS_IREE_BUFFER_DEBUG_NAME

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
  MappedMemory(MemoryAccessBitfield access, ref_ptr<Buffer> buffer,
               device_size_t byte_offset, device_size_t byte_length,
               device_size_t element_size, T* data);

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
  device_size_t byte_offset() const noexcept { return byte_offset_; }
  // Length, in bytes, of the resource mapping.
  // This may be larger than the originally requested length due to alignment.
  // This value is *informative only*, as it may vary from device to device.
  device_size_t byte_length() const noexcept { return byte_length_; }

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
      device_size_t element_offset = 0,
      device_size_t element_length = kWholeBuffer) const noexcept;
  StatusOr<absl::Span<T>> MutableSubspan(
      device_size_t element_offset = 0,
      device_size_t element_length = kWholeBuffer) noexcept;

  // Accesses an element in the mapped memory.
  // Must be called with a valid index in [0, size()).
  const T& operator[](device_size_t i) const noexcept { return data_[i]; }

  // Invalidates a range of non-coherent elements from the host caches.
  Status Invalidate(device_size_t element_offset = 0,
                    device_size_t element_length = kWholeBuffer) const;

  // Flushes a range of non-coherent elements from the host caches.
  Status Flush(device_size_t element_offset = 0,
               device_size_t element_length = kWholeBuffer);

  // Unmaps the mapped memory.
  // The memory will not be implicitly flushed when unmapping.
  void reset();

 private:
  Status ValidateAccess(MemoryAccessBitfield memory_access) const;
  Status CalculateDataRange(device_size_t element_offset,
                            device_size_t element_length,
                            device_size_t* out_adjusted_element_offset,
                            device_size_t* out_adjusted_element_length) const;

  MemoryAccessBitfield access_ = MemoryAccess::kNone;
  ref_ptr<Buffer> buffer_;
  device_size_t byte_offset_ = 0;
  device_size_t byte_length_ = 0;
  device_size_t element_size_ = 0;
  T* data_ = nullptr;
};

// Inline functions and template definitions follow:

template <typename T>
Status Buffer::Fill8(device_size_t byte_offset, device_size_t byte_length,
                     T value) {
  auto sized_value = reinterpret_cast<uint8_t*>(&value);
  return Fill(byte_offset, byte_length, sized_value, sizeof(*sized_value));
}

template <typename T>
Status Buffer::Fill16(device_size_t byte_offset, device_size_t byte_length,
                      T value) {
  auto sized_value = reinterpret_cast<uint16_t*>(&value);
  return Fill(byte_offset, byte_length, sized_value, sizeof(*sized_value));
}

template <typename T>
Status Buffer::Fill32(device_size_t byte_offset, device_size_t byte_length,
                      T value) {
  auto sized_value = reinterpret_cast<uint32_t*>(&value);
  return Fill(byte_offset, byte_length, sized_value, sizeof(*sized_value));
}

template <typename T>
Status Buffer::Fill8(T value) {
  return Fill8(0, kWholeBuffer, value);
}

template <typename T>
Status Buffer::Fill16(T value) {
  return Fill16(0, kWholeBuffer, value);
}

template <typename T>
Status Buffer::Fill32(T value) {
  return Fill32(0, kWholeBuffer, value);
}

template <typename T>
StatusOr<MappedMemory<T>> Buffer::MapMemory(MemoryAccessBitfield memory_access,
                                            device_size_t element_offset,
                                            device_size_t element_length) {
  device_size_t byte_offset = element_offset * sizeof(T);
  device_size_t byte_length = element_length == kWholeBuffer
                                  ? kWholeBuffer
                                  : element_length * sizeof(T);
  void* data = nullptr;
  IREE_RETURN_IF_ERROR(MapMemory(MappingMode::kScoped, memory_access,
                                 &byte_offset, &byte_length, &data));
  return MappedMemory<T>{
      memory_access, add_ref(this),           byte_offset,
      byte_length,   byte_length / sizeof(T), static_cast<T*>(data)};
}

template <typename T>
MappedMemory<T>::MappedMemory(MemoryAccessBitfield access,
                              ref_ptr<Buffer> buffer, device_size_t byte_offset,
                              device_size_t byte_length,
                              device_size_t element_size, T* data)
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
  rhs.access_ = MemoryAccess::kNone;
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

    rhs.access_ = MemoryAccess::kNone;
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
  if (!data_ || !AnyBitSet(access_ & MemoryAccess::kRead)) {
    return nullptr;
  }
  return data_;
}

template <typename T>
T* MappedMemory<T>::mutable_data() noexcept {
  if (!data_ || !AnyBitSet(access_ & MemoryAccess::kWrite)) {
    return nullptr;
  }
  return data_;
}

template <typename T>
Status MappedMemory<T>::ValidateAccess(
    MemoryAccessBitfield memory_access) const {
  if (!data_) {
    return FailedPreconditionErrorBuilder(IREE_LOC) << "Buffer is not mapped";
  } else if (!AnyBitSet(access_ & memory_access)) {
    return PermissionDeniedErrorBuilder(IREE_LOC)
           << "Buffer is not mapped for the desired access";
  }
  return OkStatus();
}

template <typename T>
Status MappedMemory<T>::CalculateDataRange(
    device_size_t element_offset, device_size_t element_length,
    device_size_t* out_adjusted_element_offset,
    device_size_t* out_adjusted_element_length) const {
  IREE_RETURN_IF_ERROR(Buffer::CalculateLocalRange(
      element_size_ * sizeof(T), element_offset * sizeof(T),
      element_length == kWholeBuffer ? kWholeBuffer
                                     : element_length * sizeof(T),
      out_adjusted_element_offset, out_adjusted_element_length));
  *out_adjusted_element_offset /= sizeof(T);
  *out_adjusted_element_length /= sizeof(T);
  return OkStatus();
}

template <typename T>
inline StatusOr<absl::Span<const T>> MappedMemory<T>::Subspan(
    device_size_t element_offset, device_size_t element_length) const noexcept {
  IREE_RETURN_IF_ERROR(ValidateAccess(MemoryAccess::kRead));
  IREE_RETURN_IF_ERROR(CalculateDataRange(element_offset, element_length,
                                          &element_offset, &element_length));
  return absl::Span<const T>(data_ + element_offset, element_length);
}

template <typename T>
inline StatusOr<absl::Span<T>> MappedMemory<T>::MutableSubspan(
    device_size_t element_offset, device_size_t element_length) noexcept {
  IREE_RETURN_IF_ERROR(ValidateAccess(MemoryAccess::kWrite));
  IREE_RETURN_IF_ERROR(CalculateDataRange(element_offset, element_length,
                                          &element_offset, &element_length));
  return absl::Span<T>(data_ + element_offset, element_length);
}

template <typename T>
Status MappedMemory<T>::Invalidate(device_size_t element_offset,
                                   device_size_t element_length) const {
  IREE_RETURN_IF_ERROR(ValidateAccess(MemoryAccess::kRead));
  IREE_RETURN_IF_ERROR(CalculateDataRange(element_offset, element_length,
                                          &element_offset, &element_length));
  if (!element_length) return OkStatus();
  return buffer_->InvalidateMappedMemory(
      byte_offset_ + element_offset * sizeof(T), element_length * sizeof(T));
}

template <typename T>
Status MappedMemory<T>::Flush(device_size_t element_offset,
                              device_size_t element_length) {
  IREE_RETURN_IF_ERROR(ValidateAccess(MemoryAccess::kWrite));
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
  access_ = MemoryAccess::kNone;
  byte_offset_ = 0;
  byte_length_ = 0;
  element_size_ = 0;
  data_ = nullptr;
}

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_BUFFER_H_
